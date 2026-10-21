from pathlib import Path

from trade_exp_12.config import load_config


def test_project_config_loads_expected_universe() -> None:
    loaded = load_config(Path("config/project.yaml"))

    assert loaded.settings.app.port == 8031
    assert loaded.settings.bybit.category == "spot"
    assert loaded.settings.universe.symbols == []
    assert len(loaded.settings.moex.symbols) == 10
    assert loaded.settings.moex.timeframes == ["60"]
