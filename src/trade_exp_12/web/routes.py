from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import plotly
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from trade_exp_12.data.aggregate import aggregate_candles, aggregate_features
from trade_exp_12.model.experiment import list_experiments
from trade_exp_12.reversals.zigzag import detect_atr_reversals, reversals_as_dicts

logger = logging.getLogger("trade_exp_12")

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
PLOTLY_JS = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"


def context(request: Request, **values: object) -> dict[str, object]:
    return {"request": request, "config": request.app.state.config.settings, **values}


@router.get("/vendor/plotly.min.js", include_in_schema=False)
def plotly_javascript():
    return FileResponse(
        PLOTLY_JS,
        media_type="text/javascript",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


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


@router.post("/moex/download")
async def start_moex_download(
    request: Request, symbol: str = Form(...), timeframe: str = Form(...)
):
    settings = request.app.state.config.settings
    if symbol not in settings.moex.symbols or timeframe not in settings.moex.timeframes:
        raise HTTPException(status_code=400, detail="Symbol or timeframe is not configured")
    registry = request.app.state.registry
    job_id = registry.create_job("download_moex", symbol, timeframe)
    asyncio.create_task(request.app.state.data_service.download_moex(job_id, symbol, timeframe))
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@router.post("/data/process")
async def start_processing(
    request: Request, symbol: str = Form(...), timeframe: str = Form(...)
):
    settings = request.app.state.config.settings
    bybit_ok = (symbol in settings.universe.symbols
                and timeframe in settings.universe.timeframes)
    moex_ok = (symbol in settings.moex.symbols
               and timeframe in settings.moex.timeframes)
    if not (bybit_ok or moex_ok):
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
    chart, group_size = aggregate_candles(selected, max(100, min(max_points, 200_000)))
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


@router.get("/api/smc/{symbol}/{timeframe}")
def smc_data(
    request: Request,
    symbol: str,
    timeframe: str,
    swing_length: int = 20,
    liquidity_range: float = 0.01,
    start: datetime | None = None,
    end: datetime | None = None,
):
    loaded = request.app.state.data_service.smc_analysis(
        symbol,
        timeframe,
        max(2, min(swing_length, 100)),
        max(0.001, min(liquidity_range, 0.1)),
    )
    if loaded is None:
        raise HTTPException(status_code=404, detail="Processed dataset not found")
    _, analysis = loaded
    frame = analysis.features
    effective_start = start or frame.iloc[max(0, len(frame) - 500)]["open_time"].to_pydatetime()
    effective_end = end or frame.iloc[-1]["open_time"].to_pydatetime()

    def event_time(event: dict[str, object], key: str) -> datetime:
        return datetime.fromisoformat(str(event[key]))

    def points(events: list[dict[str, object]], key: str, limit: int) -> list[dict[str, object]]:
        visible = [
            event
            for event in events
            if effective_start <= event_time(event, key) <= effective_end
        ]
        return visible[-limit:]

    def zones(events: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
        visible = [
            event
            for event in events
            if event_time(event, "end_time") >= effective_start
            and event_time(event, "known_at") <= effective_end
        ]
        return visible[-limit:]

    return {
        "start": effective_start.isoformat(),
        "end": effective_end.isoformat(),
        "swing_length": swing_length,
        "liquidity_range": liquidity_range,
        "swings": points(analysis.swings, "pivot_time", 80),
        "structure": points(analysis.structure, "known_at", 80),
        "fair_value_gaps": zones(analysis.fair_value_gaps, 100),
        "order_blocks": zones(analysis.order_blocks, 50),
        "liquidity": zones(analysis.liquidity, 50),
        "totals": {
            "swings": len(analysis.swings),
            "structure": len(analysis.structure),
            "fair_value_gaps": len(analysis.fair_value_gaps),
            "order_blocks": len(analysis.order_blocks),
            "liquidity": len(analysis.liquidity),
        },
    }


@router.get("/smc/{symbol}/{timeframe}")
def smc_page(request: Request, symbol: str, timeframe: str):
    record = request.app.state.registry.latest_processed(symbol, timeframe)
    if record is None:
        raise HTTPException(status_code=404, detail="Processed dataset not found")
    return templates.TemplateResponse(
        request,
        "smc.html",
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


# --- Experiment routes ---


def _experiment_dir(request: Request) -> Path:
    return request.app.state.config.root / "data" / "experiments"


@router.get("/experiments")
def experiments_page(request: Request):
    exp_dir = _experiment_dir(request)
    experiments = list_experiments(exp_dir)
    return templates.TemplateResponse(
        request,
        "experiments.html",
        context(request, experiments=experiments),
    )


@router.post("/experiment/run")
async def start_experiment(
    request: Request,
    symbol: str = Form(...),
    timeframe: str = Form(...),
    window_size: int = Form(64),
    atr_multiplier: float = Form(3.0),
    epochs: int = Form(30),
    d_model: int = Form(64),
    n_layers: int = Form(4),
    lr: float = Form(0.001),
):
    registry = request.app.state.registry
    processed_record = registry.latest_processed(symbol, timeframe)
    if processed_record is None:
        raise HTTPException(status_code=404, detail="Processed dataset not found")

    job_id = registry.create_job("experiment", symbol, timeframe)
    exp_dir = _experiment_dir(request)

    parquet_path = processed_record["path"]
    if not Path(parquet_path).is_absolute():
        parquet_path = str(request.app.state.config.root / parquet_path)

    db_path = str(request.app.state.registry.path)

    cmd = [
        sys.executable, "-m", "trade_exp_12.model.run_training",
        "--parquet", parquet_path,
        "--symbol", symbol,
        "--timeframe", timeframe,
        "--output-root", str(exp_dir),
        "--db", db_path,
        "--job-id", str(job_id),
        "--window-size", str(max(16, min(window_size, 256))),
        "--atr-multiplier", str(max(0.5, min(atr_multiplier, 20.0))),
        "--epochs", str(max(1, min(epochs, 200))),
        "--d-model", str(max(16, min(d_model, 512))),
        "--n-layers", str(max(1, min(n_layers, 12))),
        "--lr", str(max(1e-5, min(lr, 0.1))),
    ]

    env = {**__import__("os").environ, "SMC_CREDIT": "0"}
    subprocess.Popen(cmd, env=env, cwd=str(request.app.state.config.root))
    logger.info("Spawned training subprocess for %s %sm (job %d)", symbol, timeframe, job_id)
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@router.get("/experiment/{symbol}/{timeframe}/{run_id}")
def experiment_detail(request: Request, symbol: str, timeframe: str, run_id: str):
    exp_dir = _experiment_dir(request)
    report_path = exp_dir / symbol / timeframe / run_id / "report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Experiment not found")
    with open(report_path) as f:
        report = json.load(f)

    class ResultProxy:
        pass

    result = ResultProxy()
    result.test_accuracy = report["test_accuracy"]
    result.test_per_class = report["test_per_class"]
    result.test_confusion = report["test_confusion"]
    result.hyperparameters = report["hyperparameters"]

    class SplitProxy:
        pass

    result.split = SplitProxy()
    result.split.window_size = report["split"]["window_size"]
    result.split.feature_count = report["split"]["feature_count"]
    result.split.train_size = report["split"]["train_size"]
    result.split.val_size = report["split"]["val_size"]
    result.split.test_size = report["split"]["test_size"]
    result.split.label_counts = report["split"]["label_counts"]

    return templates.TemplateResponse(
        request,
        "experiment.html",
        context(
            request,
            result=result,
            symbol=symbol,
            timeframe=timeframe,
            run_id=run_id,
            epoch_data=report["epochs"],
        ),
    )


@router.post("/predict/run")
def start_prediction(
    request: Request,
    run_id: str = Form(...),
    model_symbol: str = Form(...),
    model_timeframe: str = Form(...),
    target_symbol: str = Form(...),
    target_timeframe: str = Form(...),
):
    """Spawn prediction subprocess for any pair using a trained model."""
    exp_dir = _experiment_dir(request) / model_symbol / model_timeframe / run_id
    if not (exp_dir / "report.json").exists():
        raise HTTPException(status_code=404, detail="Experiment not found")

    registry = request.app.state.registry
    processed = registry.latest_processed(target_symbol, target_timeframe)
    if processed is None:
        raise HTTPException(
            status_code=404,
            detail=f"No processed data for {target_symbol} {target_timeframe}m",
        )
    parquet_path = processed["path"]
    if not Path(parquet_path).is_absolute():
        parquet_path = str(request.app.state.config.root / parquet_path)

    cmd = [
        sys.executable, "-m", "trade_exp_12.model.run_predict",
        "--experiment-dir", str(exp_dir),
        "--parquet", parquet_path,
        "--target-symbol", target_symbol,
        "--target-timeframe", target_timeframe,
    ]
    env = {**__import__("os").environ, "SMC_CREDIT": "0"}
    subprocess.Popen(cmd, env=env, cwd=str(request.app.state.config.root))
    logger.info(
        "Spawned prediction subprocess: model=%s/%sm, target=%s/%sm",
        model_symbol, model_timeframe, target_symbol, target_timeframe,
    )
    return RedirectResponse(
        f"/predict/{model_symbol}/{model_timeframe}/{run_id}"
        f"?target_symbol={target_symbol}&target_timeframe={target_timeframe}",
        status_code=303,
    )


@router.get("/api/predict/{model_symbol}/{model_timeframe}/{run_id}/{target_symbol}/{target_timeframe}")
def prediction_data(
    request: Request,
    model_symbol: str,
    model_timeframe: str,
    run_id: str,
    target_symbol: str,
    target_timeframe: str,
):
    """Return cached prediction result (or 202 if still computing)."""
    exp_dir = _experiment_dir(request) / model_symbol / model_timeframe / run_id
    pred_path = exp_dir / f"predictions_{target_symbol}_{target_timeframe}.json"
    if not pred_path.exists():
        return {"status": "computing"}
    with open(pred_path) as f:
        return {"status": "ready", **json.load(f)}


@router.get("/predict/{model_symbol}/{model_timeframe}/{run_id}")
def predictions_page(
    request: Request,
    model_symbol: str,
    model_timeframe: str,
    run_id: str,
    target_symbol: str | None = None,
    target_timeframe: str | None = None,
):
    exp_dir = _experiment_dir(request) / model_symbol / model_timeframe / run_id
    report_path = exp_dir / "report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Experiment not found")
    with open(report_path) as f:
        report = json.load(f)

    t_sym = target_symbol or model_symbol
    t_tf = target_timeframe or model_timeframe

    # Auto-launch prediction if not cached yet
    pred_path = exp_dir / f"predictions_{t_sym}_{t_tf}.json"
    if not pred_path.exists():
        registry = request.app.state.registry
        processed = registry.latest_processed(t_sym, t_tf)
        if processed is not None:
            parquet_path = processed["path"]
            if not Path(parquet_path).is_absolute():
                parquet_path = str(request.app.state.config.root / parquet_path)
            cmd = [
                sys.executable, "-m", "trade_exp_12.model.run_predict",
                "--experiment-dir", str(exp_dir),
                "--parquet", parquet_path,
                "--target-symbol", t_sym,
                "--target-timeframe", t_tf,
            ]
            env = {**__import__("os").environ, "SMC_CREDIT": "0"}
            subprocess.Popen(
                cmd, env=env, cwd=str(request.app.state.config.root),
            )
            logger.info(
                "Auto-launched prediction: %s/%sm on model %s/%sm/%s",
                t_sym, t_tf, model_symbol, model_timeframe, run_id,
            )

    # Check which predictions are already computed
    available = []
    if exp_dir.exists():
        for p in exp_dir.glob("predictions_*.json"):
            name = p.stem.replace("predictions_", "")
            parts = name.rsplit("_", 1)
            if len(parts) == 2:
                available.append({"symbol": parts[0], "timeframe": parts[1]})

    return templates.TemplateResponse(
        request,
        "predictions.html",
        context(
            request,
            model_symbol=model_symbol,
            model_timeframe=model_timeframe,
            run_id=run_id,
            target_symbol=t_sym,
            target_timeframe=t_tf,
            report=report,
            available_predictions=available,
        ),
    )


# --- v2: MAE Pretrain / Finetune routes ---


def _pretrain_dir(request: Request) -> Path:
    return request.app.state.config.root / "data" / "pretrain"


def _all_parquet_paths(request: Request, symbols: list[str] | None = None) -> list[str]:
    """Absolute paths of the latest processed parquet for every configured
    instrument — Bybit universe plus MOEX shares."""
    settings = request.app.state.config.settings
    registry = request.app.state.registry
    combos = [(s, tf) for s in settings.universe.symbols
              for tf in settings.universe.timeframes]
    combos += [(s, tf) for s in settings.moex.symbols
               for tf in settings.moex.timeframes]
    if symbols is not None:
        wanted = set(symbols)
        combos = [(s, tf) for s, tf in combos if s in wanted]
    paths = []
    for symbol, timeframe in combos:
        rec = registry.latest_processed(symbol, timeframe)
        if rec is not None:
            p = rec["path"]
            if not Path(p).is_absolute():
                p = str(request.app.state.config.root / p)
            paths.append(p)
    return paths


def _all_symbols(request: Request) -> tuple[list[str], list[str]]:
    """(symbols, timeframes) across Bybit universe + MOEX shares."""
    settings = request.app.state.config.settings
    symbols = list(settings.universe.symbols) + [
        s for s in settings.moex.symbols if s not in settings.universe.symbols]
    timeframes = list(settings.universe.timeframes) + [
        tf for tf in settings.moex.timeframes if tf not in settings.universe.timeframes]
    return symbols, timeframes


def _experiments_v2_dir(request: Request) -> Path:
    return request.app.state.config.root / "data" / "experiments_v2"


@router.get("/v2/experiments")
def experiments_v2_page(request: Request):
    """v2 experiments: MAE pretrain + finetune."""
    pretrain_root = _pretrain_dir(request)
    pretrain_runs = []
    if pretrain_root.exists():
        for run_dir in sorted(pretrain_root.iterdir()):
            report_path = run_dir / "report.json"
            if report_path.exists():
                with open(report_path) as f:
                    report = json.load(f)
                pairs = report.get("info", {}).get("pairs", [])
                pretrain_runs.append({
                    "run_id": run_dir.name,
                    "info": report.get("info", {}),
                    "pairs_label": ", ".join(pairs) if pairs else "?",
                    "best_val_loss": report.get("best_val_loss"),
                    "hyperparameters": report.get("hyperparameters", {}),
                    "path": str(run_dir),
                })

    exp_root = _experiments_v2_dir(request)
    finetune_runs = []
    if exp_root.exists():
        for sym_dir in sorted(exp_root.iterdir()):
            if not sym_dir.is_dir():
                continue
            for tf_dir in sorted(sym_dir.iterdir()):
                if not tf_dir.is_dir():
                    continue
                for run_dir in sorted(tf_dir.iterdir()):
                    report_path = run_dir / "report.json"
                    if report_path.exists():
                        with open(report_path) as f:
                            report = json.load(f)
                        finetune_runs.append({
                            "symbol": sym_dir.name,
                            "timeframe": tf_dir.name,
                            "run_id": run_dir.name,
                            "test_accuracy": report.get("test_accuracy"),
                            "test_per_class": report.get("test_per_class", {}),
                            "encoder_path": report.get("encoder_path", ""),
                            "hyperparameters": report.get("hyperparameters", {}),
                            "path": str(run_dir),
                        })

    return templates.TemplateResponse(
        request,
        "experiments_v2.html",
        context(
            request,
            pretrain_runs=pretrain_runs,
            finetune_runs=finetune_runs,
            all_symbols=_all_symbols(request)[0],
            all_timeframes=_all_symbols(request)[1],
        ),
    )


@router.post("/v2/pretrain/run")
async def start_pretrain(
    request: Request,
    window_size: int = Form(64),
    epochs: int = Form(50),
    mask_ratio: float = Form(0.4),
    mask_mode: str = Form("random"),
    forecast_bars: int = Form(8),
    d_model: int = Form(64),
    n_encoder_layers: int = Form(4),
    n_decoder_layers: int = Form(2),
    d_ff: int = Form(256),
    lr: float = Form(0.001),
    batch_size: int = Form(256),
    source: str = Form("all"),
    only_symbol: str = Form("ALL"),
):
    """Launch MAE pretraining; ``source`` picks all/bybit/moex, ``only_symbol``
    can restrict to a single instrument for per-instrument runs."""
    registry = request.app.state.registry
    settings = request.app.state.config.settings

    # Collect processed parquets, optionally restricted
    symbols: list[str] | None = None
    if source == "bybit":
        symbols = list(settings.universe.symbols)
    elif source == "moex":
        symbols = list(settings.moex.symbols)
    if only_symbol != "ALL":
        symbols = [only_symbol]
    parquet_paths = _all_parquet_paths(request, symbols)

    if not parquet_paths:
        raise HTTPException(status_code=404, detail="No processed datasets found")

    from datetime import UTC as _UTC
    stamp = datetime.now(_UTC).strftime("%Y%m%dT%H%M%S")
    output_dir = _pretrain_dir(request) / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    job_id = registry.create_job("pretrain_v2", only_symbol, source)
    db_path = str(registry.path)

    cmd = [
        sys.executable, "-m", "trade_exp_12.model.run_pretrain",
        "--parquets", *parquet_paths,
        "--output-dir", str(output_dir),
        "--db", db_path,
        "--job-id", str(job_id),
        "--window-size", str(max(16, min(window_size, 256))),
        "--epochs", str(max(1, min(epochs, 500))),
        "--mask-ratio", str(max(0.1, min(mask_ratio, 0.8))),
        "--d-model", str(max(16, min(d_model, 512))),
        "--n-encoder-layers", str(max(1, min(n_encoder_layers, 12))),
        "--n-decoder-layers", str(max(1, min(n_decoder_layers, 6))),
        "--d-ff", str(max(64, min(d_ff, 2048))),
        "--lr", str(max(1e-5, min(lr, 0.1))),
        "--batch-size", str(max(32, min(batch_size, 2048))),
        "--mask-mode", mask_mode if mask_mode in ("random", "tail") else "random",
        "--forecast-bars", str(max(1, min(forecast_bars, window_size // 2))),
    ]

    env = {**__import__("os").environ, "SMC_CREDIT": "0"}
    subprocess.Popen(cmd, env=env, cwd=str(request.app.state.config.root))
    logger.info("Spawned MAE pretrain subprocess (job %d), %d parquets", job_id, len(parquet_paths))
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@router.post("/v2/finetune/run")
async def start_finetune(
    request: Request,
    pretrain_run_id: str = Form(...),
    symbol: str = Form(...),
    timeframe: str = Form(...),
    epochs: int = Form(30),
    lr: float = Form(0.0003),
    freeze_encoder: bool = Form(True),
):
    """Launch finetuning of pretrained encoder on one pair."""
    pretrain_root = _pretrain_dir(request) / pretrain_run_id
    encoder_path = pretrain_root / "encoder.pt"
    if not encoder_path.exists():
        raise HTTPException(status_code=404, detail="Pretrained encoder not found")

    # Read pretrain hyperparams to match architecture
    with open(pretrain_root / "report.json") as f:
        pt_report = json.load(f)
    pt_hp = pt_report["hyperparameters"]

    registry = request.app.state.registry
    processed = registry.latest_processed(symbol, timeframe)
    if processed is None:
        raise HTTPException(status_code=404, detail="Processed dataset not found")
    parquet_path = processed["path"]
    if not Path(parquet_path).is_absolute():
        parquet_path = str(request.app.state.config.root / parquet_path)

    from datetime import UTC as _UTC
    stamp = datetime.now(_UTC).strftime("%Y%m%dT%H%M%S")
    output_dir = _experiments_v2_dir(request) / symbol / timeframe / stamp

    job_id = registry.create_job("finetune_v2", symbol, timeframe)
    db_path = str(registry.path)

    cmd = [
        sys.executable, "-m", "trade_exp_12.model.run_finetune",
        "--encoder-path", str(encoder_path),
        "--parquet", parquet_path,
        "--symbol", symbol,
        "--timeframe", timeframe,
        "--output-dir", str(output_dir),
        "--db", db_path,
        "--job-id", str(job_id),
        "--window-size", str(pt_hp["window_size"]),
        "--d-model", str(pt_hp["d_model"]),
        "--n-encoder-layers", str(pt_hp["n_encoder_layers"]),
        "--n-decoder-layers", str(pt_hp["n_decoder_layers"]),
        "--d-ff", str(pt_hp["d_ff"]),
        "--mask-ratio", str(pt_hp["mask_ratio"]),
        "--epochs", str(max(1, min(epochs, 200))),
        "--lr", str(max(1e-5, min(lr, 0.1))),
    ]
    if freeze_encoder:
        cmd.append("--freeze-encoder")
    else:
        cmd.append("--unfreeze-encoder")

    env = {**__import__("os").environ, "SMC_CREDIT": "0"}
    subprocess.Popen(cmd, env=env, cwd=str(request.app.state.config.root))
    logger.info(
        "Spawned finetune subprocess: %s/%sm encoder=%s (job %d)",
        symbol, timeframe, pretrain_run_id, job_id,
    )
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@router.get("/v2/pretrain/{run_id}")
def pretrain_detail(request: Request, run_id: str):
    """Pretrain result detail page."""
    report_path = _pretrain_dir(request) / run_id / "report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Pretrain run not found")
    with open(report_path) as f:
        report = json.load(f)
    return templates.TemplateResponse(
        request,
        "pretrain_detail.html",
        context(request, report=report, run_id=run_id),
    )


@router.get("/v2/experiment/{symbol}/{timeframe}/{run_id}")
def finetune_detail(request: Request, symbol: str, timeframe: str, run_id: str):
    """Finetune result detail page."""
    exp_dir = _experiments_v2_dir(request) / symbol / timeframe / run_id
    report_path = exp_dir / "report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Finetune run not found")
    with open(report_path) as f:
        report = json.load(f)
    return templates.TemplateResponse(
        request,
        "finetune_detail.html",
        context(
            request, report=report,
            symbol=symbol, timeframe=timeframe, run_id=run_id,
        ),
    )


@router.post("/v2/embeddings/run")
async def start_embeddings(
    request: Request,
    pretrain_run_id: str = Form(...),
    n_clusters: int = Form(8),
    max_windows: int = Form(20000),
    atr_multiplier: float = Form(3.0),
):
    """Launch embedding analysis for a pretrained encoder."""
    pretrain_root = _pretrain_dir(request) / pretrain_run_id
    encoder_path = pretrain_root / "encoder.pt"
    report_path = pretrain_root / "report.json"
    if not encoder_path.exists() or not report_path.exists():
        raise HTTPException(status_code=404, detail="Pretrained encoder not found")

    registry = request.app.state.registry

    # Collect all processed parquets
    parquet_paths = _all_parquet_paths(request)

    if not parquet_paths:
        raise HTTPException(status_code=404, detail="No processed datasets found")

    output_path = pretrain_root / "embeddings.json"
    job_id = registry.create_job("embeddings_v2", "ALL", "ALL")
    db_path = str(registry.path)

    with open(report_path) as f:
        pt_report = json.load(f)
    window_size = pt_report["hyperparameters"]["window_size"]

    cmd = [
        sys.executable, "-m", "trade_exp_12.model.run_embeddings",
        "--encoder-path", str(encoder_path),
        "--pretrain-report", str(report_path),
        "--parquets", *parquet_paths,
        "--output", str(output_path),
        "--window-size", str(window_size),
        "--atr-multiplier", str(atr_multiplier),
        "--n-clusters", str(max(2, min(n_clusters, 50))),
        "--max-windows", str(max(1000, min(max_windows, 100000))),
        "--db", db_path,
        "--job-id", str(job_id),
    ]

    env = {**__import__("os").environ, "SMC_CREDIT": "0"}
    subprocess.Popen(cmd, env=env, cwd=str(request.app.state.config.root))
    logger.info("Spawned embeddings analysis (job %d)", job_id)
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@router.get("/v2/embeddings/{run_id}")
def embeddings_page(request: Request, run_id: str):
    """Embedding visualization page."""
    emb_path = _pretrain_dir(request) / run_id / "embeddings.json"
    if not emb_path.exists():
        raise HTTPException(status_code=404, detail="Embeddings not found. Run analysis first.")
    with open(emb_path) as f:
        data = json.load(f)
    return templates.TemplateResponse(
        request,
        "embeddings.html",
        context(request, data=data, run_id=run_id),
    )


_CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


@router.post("/v2/clustermap/run")
async def start_clustermap(
    request: Request,
    pretrain_run_id: str = Form(...),
    symbol: str = Form(...),
    timeframe: str = Form(...),
    n_clusters: int = Form(8),
    fit_windows: int = Form(20000),
    atr_multiplier: float = Form(3.0),
):
    """Build a per-bar cluster map for one pair/timeframe."""
    pretrain_root = _pretrain_dir(request) / pretrain_run_id
    encoder_path = pretrain_root / "encoder.pt"
    report_path = pretrain_root / "report.json"
    if not encoder_path.exists() or not report_path.exists():
        raise HTTPException(status_code=404, detail="Pretrained encoder not found")

    registry = request.app.state.registry

    parquet_paths = _all_parquet_paths(request)

    target_rec = registry.latest_processed(symbol, timeframe)
    if target_rec is None:
        raise HTTPException(status_code=404, detail=f"No processed data for {symbol} {timeframe}m")
    target_path = target_rec["path"]
    if not Path(target_path).is_absolute():
        target_path = str(request.app.state.config.root / target_path)

    output_path = pretrain_root / f"clustermap_{symbol}_{timeframe}.json"
    job_id = registry.create_job("clustermap_v2", symbol, timeframe)
    db_path = str(registry.path)

    with open(report_path) as f:
        pt_report = json.load(f)
    window_size = pt_report["hyperparameters"]["window_size"]

    cmd = [
        sys.executable, "-m", "trade_exp_12.model.run_cluster_map",
        "--encoder-path", str(encoder_path),
        "--pretrain-report", str(report_path),
        "--parquets", *parquet_paths,
        "--target-parquet", target_path,
        "--output", str(output_path),
        "--window-size", str(window_size),
        "--atr-multiplier", str(atr_multiplier),
        "--n-clusters", str(max(2, min(n_clusters, 20))),
        "--fit-windows", str(max(1000, min(fit_windows, 100000))),
        "--db", db_path,
        "--job-id", str(job_id),
    ]

    env = {**__import__("os").environ, "SMC_CREDIT": "0"}
    subprocess.Popen(cmd, env=env, cwd=str(request.app.state.config.root))
    logger.info("Spawned cluster map (job %d) %s %s", job_id, symbol, timeframe)
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@router.get("/v2/clustermap/{run_id}/{symbol}/{timeframe}")
def clustermap_page(request: Request, run_id: str, symbol: str, timeframe: str):
    """Per-bar cluster map on the price chart."""
    cm_path = _pretrain_dir(request) / run_id / f"clustermap_{symbol}_{timeframe}.json"
    if not cm_path.exists():
        raise HTTPException(status_code=404, detail="Cluster map not found. Run it first.")
    with open(cm_path) as f:
        data = json.load(f)
    colors = {i: _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)] for i in range(data["n_clusters"])}
    return templates.TemplateResponse(
        request,
        "clustermap.html",
        context(
            request,
            run_id=run_id, symbol=symbol, timeframe=timeframe,
            n_clusters=data["n_clusters"],
            stats=data["cluster_stats"],
            colors=colors,
            bars_json=json.dumps(data["bars"]),
            colors_json=json.dumps(colors),
        ),
    )


def _pair_composition(emb_path: Path, n_clusters: int) -> list[dict] | None:
    """Cluster/pair concentration from the sampled embeddings artifact."""
    if not emb_path.exists():
        return None
    with open(emb_path) as f:
        emb = json.load(f)
    points = emb.get("points", {})
    pairs = points.get("pair")
    clusters = points.get("cluster")
    if not pairs or not clusters:
        return None
    import math
    rows = []
    for c in range(n_clusters):
        counts: dict[str, int] = {}
        for p, cl in zip(pairs, clusters, strict=False):
            if cl == c:
                counts[p] = counts.get(p, 0) + 1
        total = sum(counts.values())
        if not total:
            rows.append({"cluster": c, "total": 0})
            continue
        top_pair, top_n = max(counts.items(), key=lambda kv: kv[1])
        shares = [v / total for v in counts.values()]
        entropy = -sum(s * math.log(s) for s in shares)
        max_entropy = math.log(len(counts)) if len(counts) > 1 else 1.0
        rows.append({
            "cluster": c,
            "total": total,
            "top_pair": top_pair,
            "top_share": round(top_n / total * 100, 1),
            "n_pairs": len(counts),
            "entropy_norm": round(entropy / max_entropy, 3),
            "likely_pair_specific": top_n / total > 0.7,
            "shares": {k: round(v / total * 100, 1) for k, v in sorted(
                counts.items(), key=lambda kv: kv[1], reverse=True)},
        })
    return rows


@router.get("/v2/clusterstats/{run_id}/{symbol}/{timeframe}")
def clusterstats_page(request: Request, run_id: str, symbol: str, timeframe: str):
    """Numeric market-state analysis of embedding clusters (tables only)."""
    cm_path = _pretrain_dir(request) / run_id / f"clustermap_{symbol}_{timeframe}.json"
    if not cm_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Cluster analysis not found. Run 'Build state report' first.")
    with open(cm_path) as f:
        data = json.load(f)
    if data.get("schema_version") != 2:
        raise HTTPException(
            status_code=409,
            detail="Old artifact schema. Rebuild the state report to get numeric analysis.")
    emb_path = _pretrain_dir(request) / run_id / "embeddings.json"
    return templates.TemplateResponse(
        request,
        "clusterstats.html",
        context(
            request,
            run_id=run_id, symbol=symbol, timeframe=timeframe,
            meta=data["run_metadata"],
            n_bars=data["n_bars"],
            n_clusters=data["n_clusters"],
            stats=data["cluster_stats"],
            transitions=data["transitions"],
            boundary=data["boundary_stats"],
            pair_rows=_pair_composition(emb_path, data["n_clusters"]),
        ),
    )
