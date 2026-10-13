from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from trade_exp_12.config import LoadedConfig
from trade_exp_12.data.bybit import BybitClient, interval_milliseconds
from trade_exp_12.data.quality import inspect_candles
from trade_exp_12.data.storage import ParquetStorage
from trade_exp_12.features.relative import add_relative_features
from trade_exp_12.registry import Registry


class DataService:
    FEATURE_SET = "relative_v1"

    def __init__(
        self,
        config: LoadedConfig,
        registry: Registry,
        storage: ParquetStorage,
        processed_storage: ParquetStorage,
    ) -> None:
        self.config = config
        self.registry = registry
        self.storage = storage
        self.processed_storage = processed_storage
        self._frame_cache: dict[str, pd.DataFrame] = {}

    async def download(self, job_id: int, symbol: str, timeframe: str) -> None:
        try:
            self.registry.update_job(job_id, "running", 0, "Connecting to Bybit")
            latest = self.registry.latest_dataset(symbol, timeframe)
            existing = self._read_path(latest["path"]) if latest else None
            if existing is not None and not existing.empty:
                next_open = existing.iloc[-1]["open_time"] + pd.Timedelta(
                    milliseconds=interval_milliseconds(timeframe)
                )
                start_ms = int(next_open.timestamp() * 1000)
            else:
                start = datetime.now(UTC) - timedelta(
                    days=365 * self.config.settings.universe.history_years
                )
                start_ms = int(start.timestamp() * 1000)

            def progress(rows: int, earliest_ms: int) -> None:
                earliest = datetime.fromtimestamp(earliest_ms / 1000, UTC).date()
                self.registry.update_job(
                    job_id, "running", 0.5, f"Received {rows:,} candles; earliest {earliest}"
                )

            client = BybitClient(self.config.settings.bybit)
            fresh = await client.download(symbol, timeframe, start_ms, progress=progress)
            if existing is not None:
                frame = pd.concat([existing, fresh], ignore_index=True)
                frame = frame.sort_values("open_time").drop_duplicates("open_time")
            else:
                frame = fresh
            if frame.empty:
                raise ValueError("Bybit returned no closed candles")
            report = inspect_candles(frame, timeframe)
            if not report.valid:
                raise ValueError(f"Dataset quality check failed: {report}")
            record = self.storage.save(
                frame,
                self.config.settings.bybit.category,
                symbol,
                timeframe,
                report,
            )
            self.registry.add_dataset(record)
            self._frame_cache[record.path] = frame
            self.registry.update_job(
                job_id,
                "completed",
                1,
                f"Stored {record.row_count:,} candles, {record.gap_count} gaps",
            )
        except Exception as exc:
            self.registry.update_job(job_id, "failed", 1, str(exc))

    async def process(self, job_id: int, symbol: str, timeframe: str) -> None:
        try:
            self.registry.update_job(job_id, "running", 0.1, "Reading raw dataset")
            loaded = self.read_latest(symbol, timeframe)
            if loaded is None:
                raise ValueError("Raw dataset not found")
            source, frame = loaded
            self.registry.update_job(job_id, "running", 0.4, "Calculating relative features")
            processed = add_relative_features(frame)
            report = inspect_candles(frame, timeframe)
            record = self.processed_storage.save(
                processed,
                self.config.settings.bybit.category,
                symbol,
                timeframe,
                report,
            )
            self.registry.add_processed(
                record,
                source_version=source["version"],
                feature_set=self.FEATURE_SET,
            )
            ready = int(processed["feature_ready"].sum())
            self.registry.update_job(
                job_id,
                "completed",
                1,
                f"Stored {record.row_count:,} rows; {ready:,} feature-ready",
            )
        except Exception as exc:
            self.registry.update_job(job_id, "failed", 1, str(exc))

    def _read_path(self, path: str) -> pd.DataFrame:
        if path not in self._frame_cache:
            stored_path = Path(path)
            if not stored_path.is_absolute():
                stored_path = self.config.root / stored_path
            if not stored_path.exists():
                parts = Path(path).parts
                data_index = next(
                    (index for index, part in enumerate(parts) if part.lower() == "data"),
                    None,
                )
                if data_index is not None:
                    stored_path = self.config.root.joinpath(*parts[data_index:])
            self._frame_cache[path] = self.storage.read(stored_path)
        return self._frame_cache[path]

    def latest_record(self, symbol: str, timeframe: str):
        return self.registry.latest_dataset(symbol, timeframe)

    def read_latest(self, symbol: str, timeframe: str) -> tuple[object, pd.DataFrame] | None:
        record = self.latest_record(symbol, timeframe)
        if record is None:
            return None
        return record, self._read_path(record["path"])

    def read_latest_processed(
        self, symbol: str, timeframe: str
    ) -> tuple[object, pd.DataFrame] | None:
        record = self.registry.latest_processed(symbol, timeframe)
        if record is None:
            return None
        return record, self._read_path(record["path"])

    def quality(self, symbol: str, timeframe: str):
        loaded = self.read_latest(symbol, timeframe)
        if loaded is None:
            return None
        record, frame = loaded
        report = inspect_candles(frame, timeframe)
        return record, add_relative_features(frame), report
