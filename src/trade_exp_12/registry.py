from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class DatasetRecord:
    symbol: str
    timeframe: str
    version: str
    path: str
    first_open_time: str
    last_open_time: str
    row_count: int
    gap_count: int
    checksum: str
    created_at: str


class Registry:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    version TEXT NOT NULL UNIQUE,
                    path TEXT NOT NULL,
                    first_open_time TEXT NOT NULL,
                    last_open_time TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    gap_count INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS datasets_pair_idx
                    ON datasets(symbol, timeframe, created_at DESC);
                CREATE TABLE IF NOT EXISTS processed_datasets (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    version TEXT NOT NULL UNIQUE,
                    source_version TEXT NOT NULL,
                    feature_set TEXT NOT NULL,
                    path TEXT NOT NULL,
                    first_open_time TEXT NOT NULL,
                    last_open_time TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    gap_count INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS processed_pair_idx
                    ON processed_datasets(symbol, timeframe, created_at DESC);
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY,
                    kind TEXT NOT NULL,
                    symbol TEXT,
                    timeframe TEXT,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    message TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def latest_datasets(self) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT d.* FROM datasets d
                JOIN (
                    SELECT symbol, timeframe, MAX(created_at) AS created_at
                    FROM datasets GROUP BY symbol, timeframe
                ) latest USING(symbol, timeframe, created_at)
                ORDER BY symbol, timeframe
                """
            ).fetchall()

    def latest_dataset(self, symbol: str, timeframe: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute(
                """SELECT * FROM datasets WHERE symbol = ? AND timeframe = ?
                ORDER BY created_at DESC LIMIT 1""",
                (symbol, timeframe),
            ).fetchone()

    def add_dataset(self, record: DatasetRecord) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO datasets
                (symbol, timeframe, version, path, first_open_time, last_open_time,
                 row_count, gap_count, checksum, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                tuple(record.__dict__.values()),
            )

    def latest_processed(self, symbol: str, timeframe: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute(
                """SELECT * FROM processed_datasets WHERE symbol = ? AND timeframe = ?
                ORDER BY created_at DESC LIMIT 1""",
                (symbol, timeframe),
            ).fetchone()

    def add_processed(
        self, record: DatasetRecord, source_version: str, feature_set: str
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO processed_datasets
                (symbol, timeframe, version, source_version, feature_set, path,
                 first_open_time, last_open_time, row_count, gap_count, checksum, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.symbol,
                    record.timeframe,
                    record.version,
                    source_version,
                    feature_set,
                    record.path,
                    record.first_open_time,
                    record.last_open_time,
                    record.row_count,
                    record.gap_count,
                    record.checksum,
                    record.created_at,
                ),
            )

    def create_job(self, kind: str, symbol: str, timeframe: str) -> int:
        now = datetime.now(UTC).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """INSERT INTO jobs(kind, symbol, timeframe, status, created_at, updated_at)
                VALUES (?, ?, ?, 'queued', ?, ?)""",
                (kind, symbol, timeframe, now, now),
            )
            return int(cursor.lastrowid)

    def update_job(self, job_id: int, status: str, progress: float, message: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """UPDATE jobs SET status = ?, progress = ?, message = ?, updated_at = ?
                WHERE id = ?""",
                (status, progress, message, datetime.now(UTC).isoformat(), job_id),
            )

    def get_job(self, job_id: int) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

    def recent_jobs(self, limit: int = 20) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
