from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from trade_exp_12.data.quality import QualityReport
from trade_exp_12.registry import DatasetRecord


@dataclass(frozen=True)
class StoredDataset:
    record: DatasetRecord
    frame: pd.DataFrame


class ParquetStorage:
    def __init__(self, root: Path, project_root: Path | None = None) -> None:
        self.root = root
        self.project_root = project_root

    def directory(self, category: str, symbol: str, timeframe: str) -> Path:
        return self.root / category / symbol / timeframe

    def save(
        self,
        frame: pd.DataFrame,
        category: str,
        symbol: str,
        timeframe: str,
        report: QualityReport,
    ) -> DatasetRecord:
        if frame.empty:
            raise ValueError("Cannot store an empty dataset")
        created = datetime.now(UTC)
        version = created.strftime("%Y%m%dT%H%M%S%fZ")
        directory = self.directory(category, symbol, timeframe)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{version}.parquet"
        temporary = target.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, target)
        checksum = hashlib.sha256(target.read_bytes()).hexdigest()
        record_path = target
        if self.project_root is not None:
            record_path = target.relative_to(self.project_root)
        return DatasetRecord(
            symbol=symbol,
            timeframe=timeframe,
            version=version,
            path=record_path.as_posix(),
            first_open_time=frame.iloc[0]["open_time"].isoformat(),
            last_open_time=frame.iloc[-1]["open_time"].isoformat(),
            row_count=len(frame),
            gap_count=report.gaps,
            checksum=checksum,
            created_at=created.isoformat(),
        )

    def read(self, path: str | Path) -> pd.DataFrame:
        return pd.read_parquet(path)
