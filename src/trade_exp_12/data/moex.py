"""MOEX ISS historical candles (public API, no auth required).

Docs: https://iss.moex.com/iss/reference/156
Endpoint: /iss/engines/stock/markets/shares/securities/{SEC}/candles.json
Intervals: 1, 10 (minutes), 60 (hour), 24 (day), 7 (week), 31 (month).
Pagination via ``start`` offset; ~500 candles per page.
"""

from __future__ import annotations

import asyncio

import httpx
import pandas as pd

from trade_exp_12.config import MoexConfig
from trade_exp_12.data.bybit import COLUMNS

ISS_INTERVALS = {"1": 1, "10": 10, "60": 60, "D": 24, "W": 7, "M": 31}

# Map our timeframe strings to ISS interval codes.
TIMEFRAME_TO_ISS = {"60": 60, "10": 10, "1": 1, "D": 24}


MOEX_TIME_SCHEMA = 2
MOEX_TIMEZONE = "Europe/Moscow"
PUBLICATION_DELAY = pd.Timedelta("15min")


def validate_time_metadata(frame: pd.DataFrame, timeframe: str) -> None:
    required = {"source_end_time", "available_at", "downloaded_at", "moex_time_schema"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError("Старая схема времени MOEX: загрузите новую версию данных")
    if not frame["moex_time_schema"].eq(MOEX_TIME_SCHEMA).all():
        raise ValueError("Смешанные или устаревшие версии времени MOEX")
    time_columns = ["open_time", "source_end_time", "available_at", "downloaded_at"]
    if any(str(getattr(frame[column].dtype, "tz", None)) != "UTC" for column in time_columns):
        raise ValueError("Метаданные времени MOEX должны быть сохранены в UTC")
    duration = pd.Timedelta("1d" if timeframe == "D" else f"{int(timeframe)}min")
    nominal_end = frame["open_time"] + duration
    expected = pd.concat(
        [nominal_end, frame["source_end_time"] + pd.Timedelta("1s")], axis=1
    ).max(axis=1) + PUBLICATION_DELAY
    if (
        frame[["open_time", "source_end_time", "available_at", "downloaded_at"]].isna().any().any()
        or (frame["source_end_time"] < frame["open_time"]).any()
        or (frame["available_at"] != expected).any()
        or (frame["downloaded_at"] < frame["available_at"]).any()
    ):
        raise ValueError("Некорректные метаданные времени или незавершённые свечи MOEX")


class MoexError(RuntimeError):
    pass


def parse_candles(rows: list[list], timeframe: str = "60") -> pd.DataFrame:
    """ISS candle columns: open, close, high, low, value, volume, begin, end."""
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.DataFrame({
        "open_time": pd.to_datetime(
            [r[6] for r in rows], format="mixed"
        ).tz_localize(MOEX_TIMEZONE).tz_convert("UTC"),
        "open": pd.to_numeric([r[0] for r in rows], errors="raise"),
        "high": pd.to_numeric([r[2] for r in rows], errors="raise"),
        "low": pd.to_numeric([r[3] for r in rows], errors="raise"),
        "close": pd.to_numeric([r[1] for r in rows], errors="raise"),
        "volume": pd.to_numeric(
            pd.Series([r[5] for r in rows]), errors="raise"),
        "turnover": pd.to_numeric(
            pd.Series([r[4] for r in rows]), errors="raise"),
    })
    frame["source_end_time"] = pd.to_datetime(
        [r[7] for r in rows], format="mixed"
    ).tz_localize(MOEX_TIMEZONE).tz_convert("UTC")
    duration = pd.Timedelta("1d" if timeframe == "D" else f"{int(timeframe)}min")
    frame["available_at"] = pd.concat([
        frame["open_time"] + duration,
        frame["source_end_time"] + pd.Timedelta("1s"),
    ], axis=1).max(axis=1) + PUBLICATION_DELAY
    frame["moex_time_schema"] = MOEX_TIME_SCHEMA
    return (
        frame.sort_values("open_time")
        .drop_duplicates("open_time")
        .reset_index(drop=True)
    )


class MoexClient:
    def __init__(self, config: MoexConfig, client: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self._external_client = client

    async def _get_page(
        self, client: httpx.AsyncClient, symbol: str, iss_interval: int,
        date_from: str, date_till: str, start: int,
    ) -> list[list]:
        error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await client.get(
                    f"/engines/stock/markets/shares/securities/{symbol}/candles.json",
                    params={
                        "from": date_from,
                        "till": date_till,
                        "interval": iss_interval,
                        "start": start,
                        "iss.only": "candles",
                    },
                )
                response.raise_for_status()
                payload = response.json()
                return payload.get("candles", {}).get("data", [])
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                error = exc
                if attempt < self.config.max_retries:
                    await asyncio.sleep(min(2**attempt, 10))
        raise MoexError(f"MOEX request failed: {error}")

    async def download(
        self,
        symbol: str,
        timeframe: str,
        start_ms: int,
        end_ms: int | None = None,
        progress: callable | None = None,
    ) -> pd.DataFrame:
        if timeframe not in TIMEFRAME_TO_ISS:
            raise ValueError(f"MOEX timeframe not supported: {timeframe}")
        iss_interval = TIMEFRAME_TO_ISS[timeframe]
        as_of = pd.Timestamp.now(tz="UTC")
        cutoff = min(as_of, pd.Timestamp(end_ms, unit="ms", tz="UTC")) if end_ms else as_of
        start_time = pd.Timestamp(start_ms, unit="ms", tz="UTC")
        date_from = start_time.tz_convert(MOEX_TIMEZONE).strftime("%Y-%m-%d")
        date_till = cutoff.tz_convert(MOEX_TIMEZONE).strftime("%Y-%m-%d")

        owns_client = self._external_client is None
        client = self._external_client or httpx.AsyncClient(
            base_url=self.config.base_url, timeout=self.config.timeout_seconds
        )
        pages: list[pd.DataFrame] = []
        start = 0
        try:
            while True:
                rows = await self._get_page(
                    client, symbol, iss_interval, date_from, date_till, start
                )
                if not rows:
                    break
                frame = parse_candles(rows, timeframe)
                pages.append(frame)
                if progress:
                    progress(
                        sum(len(p) for p in pages),
                        int(frame.iloc[0]["open_time"].timestamp() * 1000),
                    )
                if len(rows) < self.config.page_limit:
                    break
                start += len(rows)
                await asyncio.sleep(self.config.request_delay_seconds)
        finally:
            if owns_client:
                await client.aclose()

        if not pages:
            return pd.DataFrame(columns=COLUMNS)
        result = pd.concat(pages, ignore_index=True)
        result = result.sort_values("open_time").drop_duplicates("open_time")
        result = result[
            (result["open_time"] >= start_time) & (result["available_at"] <= cutoff)
        ].copy()
        result["downloaded_at"] = pd.Timestamp.now(tz="UTC")
        result = result.reset_index(drop=True)
        if not result.empty:
            validate_time_metadata(result, timeframe)
        return result
