from __future__ import annotations

import heapq

import numpy as np
import pandas as pd

SMC_FEATURE_DEFINITIONS = {
    "smc_resistance_distance_atr": "(Последний подтверждённый swing high − close) / известный ATR.",
    "smc_support_distance_atr": "(close − последний подтверждённый swing low) / известный ATR.",
    "smc_bos": "Первый пробой swing закрытием: первое или продолжающееся направление.",
    "smc_choch": "Первый пробой подтверждённого swing-уровня против направления последнего пробоя.",
    "smc_fvg_signal": "Трёхсвечный FVG известен после правой свечи; без объединения зон.",
    "smc_ob_signal": "При пробое создан OB из последней противоположной свечи от pivot до пробоя.",
    "smc_liquidity_signal": "Пара однотипных swing в пределах допуска от известного диапазона.",
    "smc_active_bullish_fvg_count": "Бычьи FVG, не затронутые low последующих закрытых свечей.",
    "smc_active_bearish_fvg_count": "Медвежьи FVG, не затронутые high последующих закрытых свечей.",
    "smc_active_bullish_ob_count": "Бычьи OB до low <= низа зоны на последующих барах.",
    "smc_active_bearish_ob_count": "Медвежьи OB до high >= верха зоны на последующих барах.",
}


def analyze_causal_smc(frame: pd.DataFrame, swing_length: int, liquidity_range: float) -> dict:
    if swing_length < 2 or not 0 < liquidity_range < 1:
        raise ValueError("Некорректные параметры причинного SMC")
    source = frame.reset_index(drop=True).copy()
    required = {"open_time", "open", "high", "low", "close", "volume", "atr", "feature_ready"}
    if not required.issubset(source.columns):
        raise ValueError(f"Нет полей для SMC: {sorted(required - set(source.columns))}")
    prices = source[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if (
        not np.isfinite(prices).all() or (prices[:, :4] <= 0).any()
        or (prices[:, 4] < 0).any()
        or (prices[:, 1] < prices[:, [0, 3]].max(axis=1)).any()
        or (prices[:, 2] > prices[:, [0, 3]].min(axis=1)).any()
    ):
        raise ValueError("Невалидные OHLCV для причинного SMC")
    n = len(source)
    open_, high, low, close, volume = prices.T
    times = source["open_time"]
    known_times = source["available_at"] if "available_at" in source else times
    output = {name: np.zeros(n, dtype=np.int32) for name in SMC_FEATURE_DEFINITIONS}
    output["smc_resistance_distance_atr"] = np.full(n, np.nan)
    output["smc_support_distance_atr"] = np.full(n, np.nan)
    source["smc_swing_high_level"] = np.nan
    source["smc_swing_low_level"] = np.nan
    swings, structure, fvgs, blocks, pools = [], [], [], [], []
    resistance = support = None
    broken = set()
    trend = 0
    prior_swings = {1: [], -1: []}
    used_liquidity_pivots = set()
    active = {kind: {1: [], -1: []} for kind in ("fvg", "ob", "liquidity")}
    events = {"fvg": fvgs, "ob": blocks, "liquidity": pools}
    cumulative_high, cumulative_low = -np.inf, np.inf

    def event(origin, known, direction):
        return {
            "direction": direction, "source_index": int(origin), "known_index": int(known),
            "end_index": n - 1, "source_time": times.iloc[origin].isoformat(),
            "known_at": known_times.iloc[known].isoformat(), "end_time": times.iloc[-1].isoformat(),
        }

    def add_zone(kind, origin, known, direction, bottom, top):
        row = event(origin, known, direction)
        row.update(bottom=float(bottom), top=float(top))
        index = len(events[kind])
        events[kind].append(row)
        threshold = top if kind == "fvg" and direction == 1 else bottom
        if kind == "ob":
            threshold = bottom if direction == 1 else top
        heapq.heappush(active[kind][direction],
                       (-float(threshold) if direction == 1 else float(threshold), index))
        return row

    for t in range(n):
        cumulative_high = max(cumulative_high, high[t])
        cumulative_low = min(cumulative_low, low[t])
        for kind in ("fvg", "ob"):
            for direction in (1, -1):
                heap = active[kind][direction]
                while heap and ((direction == 1 and low[t] <= -heap[0][0])
                                or (direction == -1 and high[t] >= heap[0][0])):
                    _, index = heapq.heappop(heap)
                    events[kind][index].update(end_index=t, end_time=times.iloc[t].isoformat())
        for direction in (1, -1):
            heap = active["liquidity"][direction]
            while heap and ((direction == 1 and high[t] > heap[0][0])
                            or (direction == -1 and low[t] < -heap[0][0])):
                _, index = heapq.heappop(heap)
                pools[index].update(end_index=t, end_time=times.iloc[t].isoformat(), swept=True)

        liquidity_signals = []
        if t >= 2 * swing_length:
            pivot = t - swing_length
            for direction, values in ((1, high), (-1, low)):
                left, right = values[pivot - swing_length:pivot], values[pivot + 1:t + 1]
                confirmed = (values[pivot] > left.max() and values[pivot] >= right.max()
                             if direction == 1 else
                             values[pivot] < left.min() and values[pivot] <= right.min())
                if not confirmed:
                    continue
                level = float(values[pivot])
                row = {
                    "direction": direction, "kind": "resistance" if direction == 1 else "support",
                    "pivot_index": pivot, "known_index": t,
                    "pivot_time": times.iloc[pivot].isoformat(),
                    "known_at": known_times.iloc[t].isoformat(), "level": level,
                }
                swings.append(row)
                if direction == 1:
                    resistance = (level, pivot)
                else:
                    support = (level, pivot)
                tolerance = liquidity_range * (cumulative_high - cumulative_low)
                match = next((old for old in reversed(prior_swings[direction])
                              if (direction, old[1]) not in used_liquidity_pivots
                              and abs(old[0] - level) <= tolerance), None)
                if match is not None:
                    pool = event(match[1], t, direction)
                    pool.update(level=(match[0] + level) / 2, swept=False)
                    pools.append(pool)
                    value = pool["level"] if direction == 1 else -pool["level"]
                    heapq.heappush(active["liquidity"][direction], (value, len(pools) - 1))
                    used_liquidity_pivots.update([(direction, match[1]), (direction, pivot)])
                    liquidity_signals.append(direction)
                prior_swings[direction].append((level, pivot))

        output["smc_liquidity_signal"][t] = (
            liquidity_signals[0] if len(set(liquidity_signals)) == 1 else 0)
        for direction, level in ((1, resistance), (-1, support)):
            if level is None or (direction, level[1]) in broken:
                continue
            if not (close[t] > level[0] if direction == 1 else close[t] < level[0]):
                continue
            kind = "CHOCH" if trend and direction != trend else "BOS"
            row = event(level[1], t, direction)
            row.update(type=kind, level=level[0])
            structure.append(row)
            output["smc_choch" if kind == "CHOCH" else "smc_bos"][t] = direction
            broken.add((direction, level[1]))
            trend = direction
            candidates = np.flatnonzero(
                close[level[1]:t] < open_[level[1]:t] if direction == 1 else
                close[level[1]:t] > open_[level[1]:t])
            if len(candidates):
                origin = level[1] + int(candidates[-1])
                block = add_zone("ob", origin, t, direction, low[origin], high[origin])
                block.update(
                    volume=float(volume[max(0, origin - 2):origin + 1].sum()),
                    strength=float(abs(close[origin] - open_[origin]) /
                                   max(high[origin] - low[origin], 1e-12) * 100))
                output["smc_ob_signal"][t] = direction

        if t >= 2:
            if close[t - 1] > open_[t - 1] and low[t] > high[t - 2]:
                add_zone("fvg", t - 1, t, 1, high[t - 2], low[t])
                output["smc_fvg_signal"][t] = 1
            elif close[t - 1] < open_[t - 1] and high[t] < low[t - 2]:
                add_zone("fvg", t - 1, t, -1, high[t], low[t - 2])
                output["smc_fvg_signal"][t] = -1
        atr = source["atr"].iloc[t]
        if resistance is not None:
            source.at[t, "smc_swing_high_level"] = resistance[0]
            if np.isfinite(atr) and atr > 0:
                output["smc_resistance_distance_atr"][t] = (resistance[0] - close[t]) / atr
        if support is not None:
            source.at[t, "smc_swing_low_level"] = support[0]
            if np.isfinite(atr) and atr > 0:
                output["smc_support_distance_atr"][t] = (close[t] - support[0]) / atr
        for kind in ("fvg", "ob"):
            for direction, label in ((1, "bullish"), (-1, "bearish")):
                output[f"smc_active_{label}_{kind}_count"][t] = len(active[kind][direction])

    for name, values in output.items():
        source[name] = values
    source["feature_ready"] = source["feature_ready"] & source[
        ["smc_resistance_distance_atr", "smc_support_distance_atr"]].notna().all(axis=1)
    return {
        "features": source, "swings": swings, "structure": structure,
        "fair_value_gaps": fvgs, "order_blocks": blocks, "liquidity": pools,
    }
