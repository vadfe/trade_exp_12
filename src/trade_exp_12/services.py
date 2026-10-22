from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from trade_exp_12.config import LoadedConfig
from trade_exp_12.data.bybit import BybitClient, interval_milliseconds
from trade_exp_12.data.moex import MOEX_TIMEZONE, MoexClient, validate_time_metadata
from trade_exp_12.data.quality import audit_moex_history, inspect_candles
from trade_exp_12.data.storage import ParquetStorage
from trade_exp_12.features.relative import RELATIVE_FEATURE_DEFINITIONS, add_relative_features
from trade_exp_12.market_structure.causal import SMC_FEATURE_DEFINITIONS
from trade_exp_12.market_structure.smc_layer import SMCAnalysis, analyze_smc
from trade_exp_12.registry import Registry


class DataService:
    FEATURE_SET = "relative_smc_v3"

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
        self._smc_cache: dict[tuple[str, int, float], SMCAnalysis] = {}

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

    def _category(self, symbol: str) -> str:
        moex = getattr(self.config.settings, "moex", None)
        if moex is not None and symbol in moex.symbols:
            return "shares"
        return self.config.settings.bybit.category

    async def download_moex(self, job_id: int, symbol: str, timeframe: str) -> None:
        try:
            self.registry.update_job(job_id, "running", 0, "Connecting to MOEX ISS")
            latest = self.registry.latest_dataset(symbol, timeframe)
            try:
                existing = self._read_path(latest["path"]) if latest else None
            except FileNotFoundError:
                existing = None
            if existing is not None:
                try:
                    validate_time_metadata(existing, timeframe)
                except (ValueError, TypeError):
                    existing = None
                    self.registry.update_job(
                        job_id, "running", 0,
                        "Новая версия MOEX с корректным временем; старые файлы сохраняются")
            if existing is not None and not existing.empty:
                next_open = existing.iloc[-1]["open_time"] + pd.Timedelta(
                    milliseconds=interval_milliseconds(timeframe)
                )
                start_ms = int(next_open.timestamp() * 1000)
            else:
                start = datetime.now(UTC) - timedelta(
                    days=365 * self.config.settings.moex.history_years
                )
                start_ms = int(start.timestamp() * 1000)

            def progress(rows: int, earliest_ms: int) -> None:
                earliest = datetime.fromtimestamp(earliest_ms / 1000, UTC).date()
                self.registry.update_job(
                    job_id, "running", 0.5, f"Received {rows:,} candles; earliest {earliest}"
                )

            client = MoexClient(self.config.settings.moex)
            fresh = await client.download(symbol, timeframe, start_ms, progress=progress)
            if existing is not None:
                frame = pd.concat([existing, fresh], ignore_index=True)
                frame = frame.sort_values("open_time").drop_duplicates("open_time")
            else:
                frame = fresh
            if frame.empty:
                raise ValueError("MOEX returned no completed candles")
            validate_time_metadata(frame, timeframe)
            report = inspect_candles(frame, timeframe)
            if not report.valid:
                raise ValueError(f"Dataset quality check failed: {report}")
            record = self.storage.save(frame, "shares", symbol, timeframe, report)
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

    async def audit_moex(self, job_id: int, symbol: str, timeframe: str) -> None:
        try:
            self.registry.update_job(job_id, "running", 0.1, "Проверка версии исходных данных")
            passport = self.dataset_passport(symbol, timeframe)
            if not passport["verified"]:
                raise ValueError("; ".join(passport["issues"]))
            path = self._resolve_path(passport["path"])
            frame = self.storage.read(path)
            first_day = frame["open_time"].iloc[0].tz_convert("Europe/Moscow").floor("D")
            self.registry.update_job(job_id, "running", 0.3, "Сверка с дневными свечами MOEX")
            daily = await MoexClient(self.config.settings.moex).download(
                symbol, "D", int(first_day.timestamp() * 1000),
                int(frame["downloaded_at"].max().timestamp() * 1000))
            report = audit_moex_history(frame, daily)
            report.update({
                "raw_checksum": passport["checksum"], "raw_version": passport["version"],
                "symbol": symbol, "timeframe": timeframe,
                "checked_at": datetime.now(UTC).isoformat(),
                "reference_source": self.config.settings.moex.base_url,
            })
            target = path.with_suffix(".audit.json")
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
            os.replace(temporary, target)
            self.registry.update_job(
                job_id, "completed", 1,
                f"Сверка сохранена: {report['reference_days']} дат; "
                f"пропущенных дат {len(report['missing_trading_dates'])}; "
                f"расхождений OHLC {len(report['ohlc_disagreements'])}. Смотрите паспорт данных.")
        except Exception as exc:
            self.registry.update_job(job_id, "failed", 1, str(exc))

    async def process(self, job_id: int, symbol: str, timeframe: str) -> None:
        try:
            self.registry.update_job(job_id, "running", 0.1, "Reading raw dataset")
            loaded = self.read_latest(symbol, timeframe)
            if loaded is None:
                raise ValueError("Raw dataset not found")
            source, frame = loaded
            if symbol in self.config.settings.moex.symbols:
                passport = self.dataset_passport(symbol, timeframe)
                if not passport["verified"]:
                    raise ValueError("; ".join(passport["issues"]))
            self.registry.update_job(job_id, "running", 0.3, "Calculating relative features")
            processed = add_relative_features(frame)
            self.registry.update_job(job_id, "running", 0.6, "Calculating SMC structures")
            processed = analyze_smc(processed).features
            report = inspect_candles(frame, timeframe)
            record = self.processed_storage.save(
                processed,
                self._category(symbol),
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

    def ohlc_training_preview(self, symbol: str, timeframe: str, example: int = 0) -> dict:
        from trade_exp_12.model.dataset_v2 import OHLCForecastDataset

        passport = self.dataset_passport(symbol, timeframe)
        if not passport["verified"] or not passport["processed_current"]:
            raise ValueError("Нужны подтверждённые исходные данные и признаки relative_smc_v3")
        record = self.registry.latest_processed(symbol, timeframe)
        path = self._resolve_path(record["path"])
        with path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != record["checksum"]:
                raise ValueError("Файл признаков не совпадает с контрольной суммой реестра")
        dataset = OHLCForecastDataset(self.processed_storage.read(path))
        if not 0 <= example < len(dataset):
            raise ValueError(f"Номер примера должен быть от 0 до {len(dataset) - 1}")
        result = dataset.preview(example)
        result.update({
            "symbol": symbol, "timeframe": timeframe, "raw_version": passport["version"],
            "processed_version": record["version"], "feature_set": record["feature_set"],
            "processed_checksum": record["checksum"],
            "feature_definitions": {**RELATIVE_FEATURE_DEFINITIONS, **SMC_FEATURE_DEFINITIONS},
        })
        return result

    def _resolve_path(self, path: str) -> Path:
        stored_path = Path(path)
        if not stored_path.is_absolute():
            stored_path = self.config.root / stored_path
        if not stored_path.exists():
            parts = Path(path).parts
            data_index = next(
                (index for index, part in enumerate(parts) if part.lower() == "data"), None,
            )
            if data_index is not None:
                stored_path = self.config.root.joinpath(*parts[data_index:])
        return stored_path

    def _read_path(self, path: str) -> pd.DataFrame:
        if path not in self._frame_cache:
            self._frame_cache[path] = self.storage.read(self._resolve_path(path))
        return self._frame_cache[path]

    def dataset_passport(self, symbol: str, timeframe: str) -> dict:
        record = self.latest_record(symbol, timeframe)
        if record is None:
            raise ValueError("Набор данных не найден")
        result = {
            "symbol": symbol, "timeframe": timeframe, "version": record["version"],
            "path": record["path"], "checksum": record["checksum"],
            "created_at": record["created_at"], "rows": record["row_count"],
            "first_open_time": record["first_open_time"],
            "last_open_time": record["last_open_time"],
            "source": self.config.settings.moex.base_url if symbol in
            self.config.settings.moex.symbols else self.config.settings.bybit.base_url,
            "source_timezone": (
                MOEX_TIMEZONE if symbol in self.config.settings.moex.symbols else "UTC"),
            "storage_timezone": "UTC", "issues": [], "verified": False,
            "checksum_matches": False, "time_verified": False,
        }
        try:
            path = self._resolve_path(record["path"])
            with path.open("rb") as source:
                actual_checksum = hashlib.file_digest(source, "sha256").hexdigest()
            result["checksum_matches"] = actual_checksum == record["checksum"]
            if not result["checksum_matches"]:
                result["issues"].append("Контрольная сумма файла не совпадает с реестром")
            frame = self.storage.read(path)
            quality = inspect_candles(frame, timeframe)
            result["quality"] = asdict(quality)
            if not quality.valid or frame.empty:
                result["issues"].append("Проверка качества свечей не пройдена")
            if len(frame) != record["row_count"]:
                result["issues"].append("Количество строк не совпадает с реестром")
            if not frame.empty and (
                frame.iloc[0]["open_time"].isoformat() != record["first_open_time"]
                or frame.iloc[-1]["open_time"].isoformat() != record["last_open_time"]
            ):
                result["issues"].append("Границы истории не совпадают с реестром")
            if symbol in self.config.settings.moex.symbols:
                validate_time_metadata(frame, timeframe)
                result["time_verified"] = True
                result["last_available_at"] = frame["available_at"].max().isoformat()
                result["last_downloaded_at"] = frame["downloaded_at"].max().isoformat()
            else:
                result["time_verified"] = True
            result["verified"] = not result["issues"]
        except (OSError, ValueError, TypeError, KeyError) as exc:
            result["issues"].append(str(exc))
        result["history_audit"] = None
        audit_path = self._resolve_path(record["path"]).with_suffix(".audit.json")
        if audit_path.is_file():
            try:
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                if (audit.get("schema_version") == 1
                        and audit.get("raw_checksum") == record["checksum"]
                        and audit.get("raw_version") == record["version"]):
                    result["history_audit"] = audit
                else:
                    result["audit_error"] = "Сверка относится к другой версии данных"
            except (OSError, ValueError) as exc:
                result["audit_error"] = f"Ошибка чтения сверки: {exc}"
        processed = self.registry.latest_processed(symbol, timeframe)
        result["processed_current"] = bool(
            processed and processed["source_version"] == record["version"]
            and processed["feature_set"] == self.FEATURE_SET
            and self._resolve_path(processed["path"]).is_file())
        return result

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

    def smc_analysis(
        self,
        symbol: str,
        timeframe: str,
        swing_length: int = 20,
        liquidity_range: float = 0.01,
    ) -> tuple[object, SMCAnalysis] | None:
        loaded = self.read_latest_processed(symbol, timeframe)
        if loaded is None:
            return None
        record, frame = loaded
        key = (record["path"], swing_length, liquidity_range)
        if key not in self._smc_cache:
            self._smc_cache[key] = analyze_smc(frame, swing_length, liquidity_range)
        return record, self._smc_cache[key]

    def quality(self, symbol: str, timeframe: str):
        loaded = self.read_latest(symbol, timeframe)
        if loaded is None:
            return None
        record, frame = loaded
        report = inspect_candles(frame, timeframe)
        return record, add_relative_features(frame), report
