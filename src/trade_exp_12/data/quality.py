from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trade_exp_12.data.bybit import interval_milliseconds


@dataclass(frozen=True)
class Gap:
    previous_open_time: str
    next_open_time: str
    missing_candles: int


@dataclass(frozen=True)
class QualityReport:
    rows: int
    duplicates: int
    out_of_order: int
    gaps: int
    invalid_ohlc: int
    invalid_volume: int
    gap_details: tuple[Gap, ...] = ()
    invalid_times: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not any(
            [self.duplicates, self.out_of_order, self.invalid_ohlc, self.invalid_volume]
        )


MOEX_HALT_REFERENCES = (
    ("2022-02-24 07:52", "2022-02-24 09:50", "https://www.moex.com/n41368",
     "https://www.moex.com/n41374"),
    ("2024-02-13 13:58", "2024-02-13 15:45", "https://www.moex.com/n67529",
     "https://www.moex.com/n67527"),
    ("2025-09-13 10:12", "2025-09-13 11:40", "https://www.interfax.ru/business/1047093",
     "https://www.moex.com/n93601"),
)


def audit_moex_history(hourly: pd.DataFrame, daily: pd.DataFrame) -> dict:
    if hourly.empty or daily.empty:
        raise ValueError("Нет данных для сверки с дневным рядом MOEX")
    if not inspect_candles(hourly, "60").valid or not inspect_candles(daily, "D").valid:
        raise ValueError("Невалидные свечи в проверяемом или эталонном ряду")
    local = hourly.copy()
    reference = daily.copy()
    local["date"] = local["open_time"].dt.tz_convert("Europe/Moscow").dt.strftime("%Y-%m-%d")
    reference["date"] = (
        reference["open_time"].dt.tz_convert("Europe/Moscow").dt.strftime("%Y-%m-%d"))
    if reference["date"].duplicated().any():
        raise ValueError("Дневной источник вернул несколько свечей на торговую дату")
    aggregate = local.groupby("date").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
    expected = reference.set_index("date")[["open", "high", "low", "close"]]
    expected = expected.loc[
        (expected.index >= local["date"].min()) & (expected.index <= local["date"].max())]
    if expected.empty:
        raise ValueError("Дневной источник не покрывает проверяемый период")
    common = expected.index.intersection(aggregate.index)
    common = common[common != local["date"].iloc[0]]
    missing = expected.index.difference(aggregate.index).tolist()
    mismatch = ~np.isclose(aggregate.loc[common], expected.loc[common], atol=1e-8, rtol=1e-8)
    disagreements = [
        {"date": date, "hourly": aggregate.loc[date].to_dict(),
         "daily": expected.loc[date].to_dict()}
        for date in common[mismatch.any(axis=1)]
    ]
    gaps = []
    times = local["open_time"].reset_index(drop=True)
    dates = local["date"].reset_index(drop=True)
    for i in np.flatnonzero((times.diff() > pd.Timedelta("1h")).to_numpy()):
        if dates.iloc[i] != dates.iloc[i - 1]:
            continue
        start = times.iloc[i - 1] + pd.Timedelta("1h")
        end = times.iloc[i]
        sources = []
        for pause_start, pause_end, first_url, last_url in MOEX_HALT_REFERENCES:
            pause_start = pd.Timestamp(pause_start, tz="Europe/Moscow").tz_convert("UTC")
            pause_end = pd.Timestamp(pause_end, tz="Europe/Moscow").tz_convert("UTC")
            if pause_start < end and pause_end > start:
                sources.extend([first_url, last_url])
        gaps.append({"start": start.isoformat(), "end": end.isoformat(),
                     "halt_sources": sorted(set(sources))})
    return {
        "schema_version": 1, "reference_days": len(expected), "compared_days": len(common),
        "reference_first_date": str(expected.index.min()),
        "reference_last_date": str(expected.index.max()), "missing_trading_dates": missing,
        "ohlc_disagreements": disagreements, "intraday_gaps": gaps,
        "unreferenced_intraday_gaps": sum(not gap["halt_sources"] for gap in gaps),
        "boundary_dates_not_compared": sorted(set(local["date"]) - set(common)),
        "status": (
            "needs_review" if missing or disagreements or gaps or not len(common)
            else "checks_passed"),
        "limitations": [
            "Дневные свечи подтверждают даты торгов, но не являются полным календарём сессий.",
            "Дневные и часовые OHLC могут охватывать разные сессии; "
            "расхождение не исправляется автоматически.",
            "Ссылка на приостановку подтверждает пересечение по времени, а не полноту всех сделок.",
            "Корпоративные события и дивидендные отсечки этим источником не подтверждены.",
        ],
        "reference_candles": reference.assign(
            open_time=reference["open_time"].astype(str)
        )[["date", "open_time", "open", "high", "low", "close"]].to_dict("records"),
    }


def inspect_candles(frame: pd.DataFrame, interval: str) -> QualityReport:
    if frame.empty:
        return QualityReport(0, 0, 0, 0, 0, 0)
    times = frame["open_time"]
    differences = times.diff().dt.total_seconds().mul(1000)
    expected = interval_milliseconds(interval)
    gap_indices = differences.index[differences > expected]
    gap_details = tuple(
        Gap(
            previous_open_time=times.iloc[position - 1].isoformat(),
            next_open_time=times.iloc[position].isoformat(),
            missing_candles=int(differences.iloc[position] // expected) - 1,
        )
        for position in gap_indices
    )
    bad_price = (
        ~np.isfinite(frame[["open", "high", "low", "close"]]).all(axis=1)
        | (frame[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    bad_volume = (
        ~np.isfinite(frame[["volume", "turnover"]]).all(axis=1)
        | (frame[["volume", "turnover"]] < 0).any(axis=1)
    )
    invalid_mask = (
        bad_price | bad_volume |
        (frame["low"] > frame[["open", "close"]].min(axis=1))
        | (frame["high"] < frame[["open", "close"]].max(axis=1))
        | (frame["low"] > frame["high"])
        | (frame["volume"] < 0)
        | (frame["turnover"] < 0)
    )
    invalid_ohlc = int(
        (
            bad_price | (frame["low"] > frame[["open", "close"]].min(axis=1))
            | (frame["high"] < frame[["open", "close"]].max(axis=1))
            | (frame["low"] > frame["high"])
        ).sum()
    )
    return QualityReport(
        rows=len(frame),
        duplicates=int(times.duplicated().sum()),
        out_of_order=int((differences.iloc[1:] <= 0).sum()),
        gaps=len(gap_details),
        invalid_ohlc=invalid_ohlc,
        invalid_volume=int(bad_volume.sum()),
        gap_details=gap_details,
        invalid_times=tuple(value.isoformat() for value in times[invalid_mask]),
    )
