import json
import socket
import threading
import time

import httpx
import numpy as np
import uvicorn

from trade_exp_12.main import create_app
from trade_exp_12.model.run_forecast_eval import forecast_metrics


def test_forecast_http_pages_and_actions(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = config_dir / "project.yaml"
    config.write_text('universe:\n  symbols: []\n  timeframes: []\n'
                      'moex:\n  symbols: [SBER, GAZP]\n  timeframes: ["60"]\n')
    app = create_app(config)
    root = tmp_path / "data" / "pretrain"
    current = root / "NEW"
    current.mkdir(parents=True)
    legacy = root / "OLD"
    legacy.mkdir()
    report = {
        "info": {"pairs": ["SBER/60"], "total_windows": 1000},
        "hyperparameters": {"mask_mode": "tail"}, "best_val_loss": 0.2,
        "forecast_protocol": 2,
    }
    (current / "report.json").write_text(json.dumps(report))
    (current / "encoder.pt").write_bytes(b"fixture")
    report.pop("forecast_protocol")
    (legacy / "report.json").write_text(json.dumps(report))
    (legacy / "encoder.pt").write_bytes(b"fixture")
    metrics = forecast_metrics(np.zeros((3, 2)), np.ones((3, 2)) * 0.001, np.zeros(3))
    artifact = {
        "run_metadata": {"window_size": 16, "forecast_bars": 2, "eval_windows": 3},
        "metrics": metrics, "verdict": "none",
        "example": {"times": ["2020-01-01", "2020-01-02"], "anchor_close": 100,
                    "actual_path": [100, 100.1, 100.2], "predicted_path": [100, 100, 100]},
    }
    (current / "forecast_SBER_60.json").write_text(json.dumps(artifact))
    target = tmp_path / "fixture.parquet"
    target.write_bytes(b"fixture")
    commands = []
    monkeypatch.setattr("trade_exp_12.web.routes.subprocess.Popen",
                        lambda cmd, **kwargs: commands.append(cmd))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.05)
        assert server.started
        monkeypatch.setattr(app.state.registry, "latest_processed",
                            lambda symbol, timeframe: {"path": str(target)})
        with httpx.Client(base_url=f"http://127.0.0.1:{port}") as client:
            assert client.get("/").status_code == 200
            page = client.get("/v2/experiments")
            assert page.status_code == 200
            assert "Проверить прогноз SBER/60" in page.text
            assert "Старая модель" in page.text
            assert 'value="ALL"' not in page.text
            assert "/v2/forecast/NEW/SBER/60" in page.text
            result = client.get("/v2/forecast/NEW/SBER/60")
            assert result.status_code == 200
            assert "не превзошла" in result.text
            assert "Plotly.newPlot" in result.text
            assert client.get("/v2/forecast/MISSING/SBER/60").status_code == 404
            assert client.get("/v2/forecast/NEW/BRETTUSDT/60").status_code == 404
            fields = {"pretrain_run_id": "OLD", "symbol": "SBER", "timeframe": "60"}
            assert client.post("/v2/forecast/run", data=fields).status_code == 409
            fields["pretrain_run_id"] = "NEW"
            fields["symbol"] = "GAZP"
            assert client.post("/v2/forecast/run", data=fields).status_code == 409
            fields["symbol"] = "SBER"
            response = client.post("/v2/forecast/run", data=fields)
            assert response.status_code == 303
            assert commands[0][2] == "trade_exp_12.model.run_forecast_eval"
            assert client.get(response.headers["location"]).status_code == 200
            assert client.post("/v2/pretrain/run", data={"only_symbol": "ALL"}).status_code == 400
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
