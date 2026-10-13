from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from trade_exp_12.data.quality import inspect_candles
from trade_exp_12.main import create_app


def write_config(path: Path, root: Path) -> None:
    path.write_text(
        f"""
app:
  host: 127.0.0.1
  port: 8031
bybit:
  base_url: https://api.bybit.com
  category: spot
universe:
  symbols: [ADAUSDT]
  timeframes: [\"15\"]
  history_years: 1
storage:
  raw_dir: {str(root / "raw").replace(chr(92), "/")}
  registry_path: {str(root / "registry.db").replace(chr(92), "/")}
""",
        encoding="utf-8",
    )


def test_health_and_dashboard(tmp_path) -> None:
    config = tmp_path / "project.yaml"
    write_config(config, tmp_path)

    with TestClient(create_app(config)) as client:
        health = client.get("/health")
        dashboard = client.get("/")
        plotly_javascript = client.get("/vendor/plotly.min.js")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert dashboard.status_code == 200
    assert "ADAUSDT" in dashboard.text
    assert "Загрузить" in dashboard.text
    assert plotly_javascript.status_code == 200
    assert "Plotly" in plotly_javascript.text[:1000]
    assert "max-age=31536000" in plotly_javascript.headers["cache-control"]


def test_quality_page_uses_stored_dataset(tmp_path) -> None:
    config = tmp_path / "project.yaml"
    write_config(config, tmp_path)
    app = create_app(config)
    times = pd.date_range("2026-01-01", periods=30, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open_time": times,
            "open": [100.0] * 30,
            "high": [102.0] * 30,
            "low": [99.0] * 30,
            "close": [101.0] * 30,
            "volume": [1000.0] * 30,
            "turnover": [100_000.0] * 30,
        }
    )

    with TestClient(app) as client:
        report = inspect_candles(frame, "15")
        record = app.state.storage.save(frame, "spot", "ADAUSDT", "15", report)
        assert "\\" not in record.path
        app.state.registry.add_dataset(record)
        response = client.get("/quality/ADAUSDT/15")
        dataset_response = client.get("/data/ADAUSDT/15")
        candles_response = client.get("/api/data/ADAUSDT/15/candles?max_points=10")

    assert response.status_code == 200
    assert "Контроль качества" in response.text
    assert "Relative volume" in response.text
    assert dataset_response.status_code == 200
    assert "Загрузка обзора" in dataset_response.text
    assert "/api/data/ADAUSDT/15/candles" in dataset_response.text
    assert candles_response.status_code == 200
    assert candles_response.json()["source_count"] == 30
    assert candles_response.json()["point_count"] == 30
