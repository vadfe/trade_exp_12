from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import httpx
import pandas as pd

from trade_exp_12.config import BybitConfig

COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "turnover"]
INTERVAL_MINUTES = {
    "1": 1,
    "3": 3,
    "5": 5,
    "15": 15,
    "30": 30,
    "60": 60,
    "120": 120,
    "240": 240,
    "360": 360,
    "720": 720,
    "D": 1440,
    "W": 10080,
}


class BybitError(RuntimeError):
    pass


def interval_milliseconds(interval: str) -> int:
    if interval not in INTERVAL_MINUTES:
        raise ValueError(f"Fixed interval is not supported: {interval}")
    return INTERVAL_MINUTES[interval] * 60_000


def parse_klines(rows: list[list[str]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=COLUMNS)
    if frame.empty:
        return frame.astype({column: "float64" for column in COLUMNS[1:]})
    frame["open_time"] = pd.to_datetime(
        pd.to_numeric(frame["open_time"]), unit="ms", utc=True
    )
    for column in COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
    return frame.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)


class BybitClient:
    def __init__(self, config: BybitConfig, client: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self._external_client = client

    async def _request(self, client: httpx.AsyncClient, params: dict[str, object]) -> dict:
        error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await client.get("/v5/market/kline", params=params)
                response.raise_for_status()
                payload = response.json()
                if payload.get("retCode") == 0:
                    return payload
                if payload.get("retCode") != 10006:
                    code = payload.get("retCode")
                    message = payload.get("retMsg")
                    raise BybitError(f"Bybit error {code}: {message}")
                error = BybitError(payload.get("retMsg", "Bybit rate limit"))
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 429 and exc.response.status_code < 500:
                    raise BybitError(f"Bybit HTTP {exc.response.status_code}") from exc
                error = exc
            except httpx.RequestError as exc:
                error = exc
            if attempt < self.config.max_retries:
                await asyncio.sleep(min(2**attempt, 10))
        raise BybitError(f"Bybit request failed: {error}")

    async def download(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int | None = None,
        progress: callable | None = None,
    ) -> pd.DataFrame:
        cursor = end_ms or int(datetime.now(UTC).timestamp() * 1000)
        pages: list[pd.DataFrame] = []
        owns_client = self._external_client is None
        client = self._external_client or httpx.AsyncClient(
            base_url=self.config.base_url, timeout=self.config.timeout_seconds
        )
        try:
            while cursor >= start_ms:
                payload = await self._request(
                    client,
                    {
                        "category": self.config.category,
                        "symbol": symbol,
                        "interval": interval,
                        "start": start_ms,
                        "end": cursor,
                        "limit": self.config.page_limit,
                    },
                )
                frame = parse_klines(payload["result"]["list"])
                if frame.empty:
                    break
                pages.append(frame)
                earliest_ms = int(frame.iloc[0]["open_time"].timestamp() * 1000)
                if progress:
                    progress(sum(len(page) for page in pages), earliest_ms)
                if earliest_ms <= start_ms or len(frame) < self.config.page_limit:
                    break
                cursor = earliest_ms - 1
                await asyncio.sleep(self.config.request_delay_seconds)
        finally:
            if owns_client:
                await client.aclose()
        if not pages:
            return parse_klines([])
        result = pd.concat(pages, ignore_index=True)
        result = result.sort_values("open_time").drop_duplicates("open_time")
        result = result[result["open_time"] >= pd.Timestamp(start_ms, unit="ms", tz="UTC")]
        candle_ms = interval_milliseconds(interval)
        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        result = result[result["open_time"].astype("int64") // 1_000_000 + candle_ms <= now_ms]
        return result.reset_index(drop=True)
