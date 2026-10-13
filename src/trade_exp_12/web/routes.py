from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from trade_exp_12.data.aggregate import aggregate_candles, aggregate_features
from trade_exp_12.reversals.zigzag import detect_atr_reversals, reversals_as_dicts

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


def context(request: Request, **values: object) -> dict[str, object]:
    return {"request": request, "config": request.app.state.config.settings, **values}


@router.get("/")
def dashboard(request: Request):
    registry = request.app.state.registry
    latest = {(row["symbol"], row["timeframe"]): row for row in registry.latest_datasets()}
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        context(
            request,
            latest=latest,
            jobs=registry.recent_jobs(),
        ),
    )


@router.get("/config")
def show_config(request: Request):
    return templates.TemplateResponse(
        request,
        "config.html",
        context(request, rendered=request.app.state.config.settings.model_dump(mode="json")),
    )


@router.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "registry": str(request.app.state.registry.path),
        "config": str(request.app.state.config.source),
    }


@router.post("/data/download")
async def start_download(
    request: Request, symbol: str = Form(...), timeframe: str = Form(...)
):
    settings = request.app.state.config.settings
    if symbol not in settings.universe.symbols or timeframe not in settings.universe.timeframes:
        raise HTTPException(status_code=400, detail="Symbol or timeframe is not configured")
    registry = request.app.state.registry
    job_id = registry.create_job("download", symbol, timeframe)
    asyncio.create_task(request.app.state.data_service.download(job_id, symbol, timeframe))
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@router.post("/data/process")
async def start_processing(
    request: Request, symbol: str = Form(...), timeframe: str = Form(...)
):
    settings = request.app.state.config.settings
    if symbol not in settings.universe.symbols or timeframe not in settings.universe.timeframes:
        raise HTTPException(status_code=400, detail="Symbol or timeframe is not configured")
    registry = request.app.state.registry
    job_id = registry.create_job("process", symbol, timeframe)
    asyncio.create_task(request.app.state.data_service.process(job_id, symbol, timeframe))
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@router.get("/jobs/{job_id}")
def job(request: Request, job_id: int):
    record = request.app.state.registry.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return templates.TemplateResponse(request, "job.html", context(request, job=record))


@router.get("/api/data/{symbol}/{timeframe}/candles")
def candle_data(
    request: Request,
    symbol: str,
    timeframe: str,
    start: datetime | None = None,
    end: datetime | None = None,
    max_points: int = 2000,
):
    loaded = request.app.state.data_service.read_latest(symbol, timeframe)
    if loaded is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    _, frame = loaded
    selected = frame
    if start is not None:
        selected = selected[selected["open_time"] >= start]
    if end is not None:
        selected = selected[selected["open_time"] <= end]
    selected = selected.reset_index(drop=True)
    chart, group_size = aggregate_candles(selected, max(100, min(max_points, 5000)))
    return {
        "source_count": len(selected),
        "point_count": len(chart),
        "group_size": group_size,
        "time": [value.isoformat() for value in chart["open_time"]],
        "open": chart["open"].tolist(),
        "high": chart["high"].tolist(),
        "low": chart["low"].tolist(),
        "close": chart["close"].tolist(),
        "volume": chart["volume"].tolist(),
    }


@router.get("/data/{symbol}/{timeframe}")
def dataset(request: Request, symbol: str, timeframe: str):
    record = request.app.state.data_service.latest_record(symbol, timeframe)
    if record is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    processed = request.app.state.registry.latest_processed(symbol, timeframe)
    return templates.TemplateResponse(
        request,
        "dataset.html",
        context(
            request,
            record=record,
            processed=processed,
            symbol=symbol,
            timeframe=timeframe,
        ),
    )


@router.get("/api/features/{symbol}/{timeframe}")
def feature_data(
    request: Request,
    symbol: str,
    timeframe: str,
    start: datetime | None = None,
    end: datetime | None = None,
    max_points: int = 2000,
):
    loaded = request.app.state.data_service.read_latest_processed(symbol, timeframe)
    if loaded is None:
        raise HTTPException(status_code=404, detail="Processed dataset not found")
    _, frame = loaded
    selected = frame
    if start is not None:
        selected = selected[selected["open_time"] >= start]
    if end is not None:
        selected = selected[selected["open_time"] <= end]
    selected = selected.copy().reset_index(drop=True)
    selected["atr_percent"] = selected["atr"] / selected["close"] * 100
    chart, group_size = aggregate_features(selected, max(100, min(max_points, 5000)))

    def values(column: str) -> list[float | None]:
        return [None if value != value else float(value) for value in chart[column]]

    return {
        "source_count": len(selected),
        "point_count": len(chart),
        "group_size": group_size,
        "time": [value.isoformat() for value in chart["open_time"]],
        "log_return": values("log_return"),
        "body_atr": values("body_atr"),
        "upper_wick_atr": values("upper_wick_atr"),
        "lower_wick_atr": values("lower_wick_atr"),
        "range_atr": values("range_atr"),
        "relative_volume": values("relative_volume"),
        "atr_percent": values("atr_percent"),
    }


@router.get("/features/{symbol}/{timeframe}")
def features(request: Request, symbol: str, timeframe: str):
    record = request.app.state.registry.latest_processed(symbol, timeframe)
    if record is None:
        raise HTTPException(status_code=404, detail="Processed dataset not found")
    return templates.TemplateResponse(
        request,
        "features.html",
        context(request, record=record, symbol=symbol, timeframe=timeframe),
    )


@router.get("/api/reversals/{symbol}/{timeframe}")
def reversal_data(
    request: Request,
    symbol: str,
    timeframe: str,
    atr_multiplier: float = 3.0,
    start: datetime | None = None,
    end: datetime | None = None,
):
    loaded = request.app.state.data_service.read_latest_processed(symbol, timeframe)
    if loaded is None:
        raise HTTPException(status_code=404, detail="Processed dataset not found")
    _, frame = loaded
    applied_multiplier = max(0.25, min(atr_multiplier, 20))
    reversals = detect_atr_reversals(frame, applied_multiplier)
    items = reversals_as_dicts(reversals)
    if start is not None:
        items = [item for item in items if datetime.fromisoformat(item["pivot_time"]) >= start]
    if end is not None:
        items = [item for item in items if datetime.fromisoformat(item["pivot_time"]) <= end]
    return {
        "count": len(items),
        "up_count": sum(item["direction"] == "up" for item in items),
        "down_count": sum(item["direction"] == "down" for item in items),
        "atr_multiplier": applied_multiplier,
        "items": items,
    }


@router.get("/reversals/{symbol}/{timeframe}")
def reversals(request: Request, symbol: str, timeframe: str):
    record = request.app.state.registry.latest_processed(symbol, timeframe)
    if record is None:
        raise HTTPException(status_code=404, detail="Processed dataset not found")
    return templates.TemplateResponse(
        request,
        "reversals.html",
        context(request, record=record, symbol=symbol, timeframe=timeframe),
    )


@router.get("/quality/{symbol}/{timeframe}")
def quality(request: Request, symbol: str, timeframe: str, limit: int = 2000):
    loaded = request.app.state.data_service.quality(symbol, timeframe)
    if loaded is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    record, frame, report = loaded
    view = frame.tail(max(50, min(limit, 5000)))
    chart = {
        "time": [value.isoformat() for value in view["open_time"]],
        "log_return": view["log_return"].where(view["log_return"].notna(), None).tolist(),
        "body_atr": view["body_atr"].where(view["body_atr"].notna(), None).tolist(),
        "range_atr": view["range_atr"].where(view["range_atr"].notna(), None).tolist(),
        "relative_volume": view["relative_volume"]
        .where(view["relative_volume"].notna(), None)
        .tolist(),
    }
    feature_rows = view[view["feature_ready"]].tail(20).iloc[::-1].to_dict("records")
    return templates.TemplateResponse(
        request,
        "quality.html",
        context(
            request,
            record=record,
            report=report,
            symbol=symbol,
            timeframe=timeframe,
            chart=chart,
            feature_rows=feature_rows,
        ),
    )
