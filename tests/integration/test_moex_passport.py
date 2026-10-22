import asyncio
import socket
import sys
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import torch
import uvicorn
from fastapi.testclient import TestClient

from trade_exp_12.data.moex import MoexClient, parse_candles
from trade_exp_12.data.quality import inspect_candles
from trade_exp_12.main import create_app
from trade_exp_12.model import run_ohlc_forecast


def setup_app(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = config_dir / "project.yaml"
    config.write_text('universe:\n  symbols: []\n  timeframes: []\n'
                      'moex:\n  symbols: [SBER]\n  timeframes: ["60"]\n')
    return create_app(config)


def candles():
    frame = parse_candles([
        [100, 101, 102, 99, 10000, 100, "2026-10-01 10:00:00", "2026-10-01 10:59:59"],
        [101, 102, 103, 100, 10000, 100, "2026-10-01 11:00:00", "2026-10-01 11:59:59"],
    ])
    frame["downloaded_at"] = pd.Timestamp("2026-10-02 00:00:00Z")
    return frame


def store(app, frame):
    record = app.state.storage.save(
        frame, "shares", "SBER", "60", inspect_candles(frame, "60"))
    app.state.registry.add_dataset(record)
    return record


def test_legacy_refresh_keeps_original_and_does_not_merge_wrong_times(tmp_path, monkeypatch):
    app = setup_app(tmp_path)
    starts = []

    async def download(self, symbol, timeframe, start_ms, **kwargs):
        starts.append(start_ms)
        return candles()

    monkeypatch.setattr(MoexClient, "download", download)
    with TestClient(app) as client:
        legacy = candles().drop(columns=[
            "source_end_time", "available_at", "downloaded_at", "moex_time_schema"])
        legacy["open_time"] += pd.Timedelta(hours=3)
        old = store(app, legacy)
        passport = client.get("/api/data/SBER/60/passport").json()
        assert not passport["verified"]
        assert "Старая схема" in passport["issues"][0]
        process_job = app.state.registry.create_job("process", "SBER", "60")
        asyncio.run(app.state.data_service.process(process_job, "SBER", "60"))
        assert app.state.registry.get_job(process_job)["status"] == "failed"
        job = app.state.registry.create_job("download_moex", "SBER", "60")
        asyncio.run(app.state.data_service.download_moex(job, "SBER", "60"))
        assert app.state.registry.get_job(job)["status"] == "completed"
        current = app.state.registry.latest_dataset("SBER", "60")
        assert current["version"] != old.version
        assert (tmp_path / old.path).is_file()
        assert current["row_count"] == 2
        assert starts[0] < legacy.iloc[0]["open_time"].timestamp() * 1000
        passport = client.get("/api/data/SBER/60/passport").json()
        assert passport["verified"]
        assert passport["checksum_matches"]
        assert passport["last_open_time"].startswith("2026-10-01T08:00:00")
        assert not passport["processed_current"]
        (tmp_path / current["path"]).write_bytes(b"corrupted")
        broken = client.get("/api/data/SBER/60/passport").json()
        assert not broken["verified"]
        assert not broken["checksum_matches"]


def test_passport_real_http_and_update_action(tmp_path, monkeypatch):
    app = setup_app(tmp_path)

    async def reference_download(self, symbol, timeframe, start_ms, end_ms=None):
        assert timeframe == "D"
        daily = candles().iloc[:1].copy()
        daily["open_time"] = pd.Timestamp("2026-10-01 00:00:00Z")
        return daily

    monkeypatch.setattr(MoexClient, "download", reference_download)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.05)
        assert server.started
        store(app, candles())
        processed = app.state.processed_storage.save(
            candles(), "shares", "SBER", "60", inspect_candles(candles(), "60"))
        app.state.registry.add_processed(processed, "OLD", "fixture")
        with httpx.Client(base_url=f"http://127.0.0.1:{sock.getsockname()[1]}") as client:
            assert client.get("/").status_code == 200
            response = client.get("/data/SBER/60")
            assert response.status_code == 200
            assert "Паспорт версии данных" in response.text
            assert "Базовые проверки" in response.text
            assert 'action="/moex/download"' in response.text
            assert "Дивиденды, сплиты" in response.text
            api = client.get("/api/data/SBER/60/passport")
            assert api.status_code == 200
            assert api.json()["verified"]
            assert not api.json()["processed_current"]
            blocked = client.post("/v2/pretrain/run", data={"only_symbol": "SBER"})
            assert blocked.status_code == 409
            assert client.get("/api/data/UNKNOWN/60/passport").status_code == 404
            assert client.get("/api/data/SBER/60/candles").status_code == 200
            assert client.get("/quality/SBER/60").status_code == 200
            audit = client.post("/moex/audit", data={"symbol": "SBER", "timeframe": "60"})
            assert audit.status_code == 303
            job_id = int(audit.headers["location"].split("/")[-1])
            for _ in range(100):
                job = app.state.registry.get_job(job_id)
                if job["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.02)
            assert job["status"] == "completed", job["message"]
            data = client.get("/api/data/SBER/60/passport").json()
            assert data["history_audit"]["reference_days"] == 1
            assert data["history_audit"]["raw_checksum"] == data["checksum"]
            assert data["history_audit"]["status"] == "needs_review"
            assert "Нужен разбор" in client.get("/data/SBER/60").text
            assert client.post("/moex/audit", data={
                "symbol": "UNKNOWN", "timeframe": "60"}).status_code == 400
            assert client.get("/training/SBER/60").status_code == 409
            x = np.arange(600)
            close = 100 + np.sin(x / 8) * 8 + x * 0.02
            open_price = close + np.sin(x / 3) * 0.4
            full = pd.DataFrame({
                "open_time": pd.date_range("2025-01-01", periods=600, freq="h", tz="UTC"),
                "open": open_price, "close": close,
                "high": np.maximum(open_price, close) + 0.8,
                "low": np.minimum(open_price, close) - 0.8,
                "volume": 1000 + np.cos(x / 5) * 200, "turnover": close * 1000,
            })
            full["source_end_time"] = full["open_time"] + pd.Timedelta("59min59s")
            full["available_at"] = full["open_time"] + pd.Timedelta("75min")
            full["downloaded_at"] = pd.Timestamp.now(tz="UTC")
            full["moex_time_schema"] = 2
            store(app, full)
            process_id = app.state.registry.create_job("process", "SBER", "60")
            asyncio.run(app.state.data_service.process(process_id, "SBER", "60"))
            job = app.state.registry.get_job(process_id)
            assert job["status"] == "completed", job["message"]
            example = client.get("/api/training/SBER/60/preview")
            assert example.status_code == 200
            sample = example.json()
            assert len(sample["input_features"]) == 61
            assert len(sample["input_features"][0]) == 33
            assert len(sample["target_ohlc"]) == 3
            assert len(sample["target_ohlc"][0]) == 4
            assert len(sample["feature_definitions"]) == 33
            assert sample["feature_set"] == "relative_smc_v3"
            assert sample["context_times"][-1] < sample["target_times"][0]
            page = client.get("/training/SBER/60")
            assert page.status_code == 200
            assert "Три будущие свечи" in page.text
            assert "Обучающий пример" in client.get("/data/SBER/60").text
            assert client.get("/api/training/SBER/60/preview?example=-1").status_code == 409
            assert client.get("/ohlc").status_code == 200
            assert client.post("/ohlc/train").status_code == 400
            assert client.post("/ohlc/train", data={
                "acknowledge_development": "true"}).status_code == 409

            async def full_daily(self, symbol, timeframe, start_ms, end_ms=None):
                return full.assign(date=full.open_time.dt.tz_convert(
                    "Europe/Moscow").dt.floor("D")).groupby("date").agg(
                        open=("open", "first"), high=("high", "max"), low=("low", "min"),
                        close=("close", "last"), volume=("volume", "sum"),
                        turnover=("turnover", "sum")).reset_index().rename(
                            columns={"date": "open_time"}).assign(
                                open_time=lambda df: df.open_time.dt.tz_convert("UTC"))

            monkeypatch.setattr(MoexClient, "download", full_daily)
            audit_id = app.state.registry.create_job("audit_moex", "SBER", "60")
            asyncio.run(app.state.data_service.audit_moex(audit_id, "SBER", "60"))
            assert app.state.registry.get_job(audit_id)["status"] == "completed"
            commands = []
            monkeypatch.setattr("trade_exp_12.web.routes.subprocess.Popen",
                                lambda command, **kwargs: commands.append(command))
            response = client.post("/ohlc/train", data={
                "epochs": "1", "acknowledge_development": "true"})
            assert response.status_code == 303
            job_id = int(response.headers["location"].split("/")[-1])
            command = commands[0]
            assert command[2] == "trade_exp_12.model.run_ohlc_forecast"
            output = Path(command[command.index("--output-dir") + 1])
            torch.set_num_threads(1)
            train = run_ohlc_forecast.train_ohlc
            reports = []

            def train_small(*args, **kwargs):
                result = train(*args, **kwargs, d_model=8, layers=1)
                reports.append(result)
                return result

            with monkeypatch.context() as child:
                child.setattr(run_ohlc_forecast, "train_ohlc", train_small)
                child.setattr(sys, "argv", [command[2], *command[3:]])
                run_ohlc_forecast.main()
            report = reports[0]
            assert app.state.registry.get_job(job_id)["status"] == "completed"
            run_id = output.name
            page = client.get(f"/ohlc/{run_id}")
            assert page.status_code == 200
            assert "Все 12 будущих цен" in page.text
            assert "Автоматического перехода" in page.text
            download = client.get(f"/ohlc/{run_id}/report.json")
            assert download.status_code == 200
            assert download.json()["evaluation_windows"] == report["evaluation_windows"]
            assert client.get(f"/ohlc?job_id={job_id}").headers["location"] == f"/ohlc/{run_id}"
            assert "Открыть OHLC-отчёт" in client.get(f"/jobs/{job_id}").text
            assert client.get("/ohlc/MISSING").status_code == 404
            assert client.get(f"/ohlc/{run_id}?example=-1").status_code == 400
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
