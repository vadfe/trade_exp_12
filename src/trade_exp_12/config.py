from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class AppConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8031, ge=1, le=65535)


class BybitConfig(BaseModel):
    base_url: str = "https://api.bybit.com"
    category: str = "spot"
    page_limit: int = Field(default=1000, ge=1, le=1000)
    request_delay_seconds: float = Field(default=0.08, ge=0)
    timeout_seconds: float = Field(default=20, gt=0)
    max_retries: int = Field(default=5, ge=0)

    @field_validator("category")
    @classmethod
    def validate_category(cls, value: str) -> str:
        if value not in {"spot", "linear", "inverse"}:
            raise ValueError("Unsupported Bybit category")
        return value


class UniverseConfig(BaseModel):
    symbols: list[str]
    timeframes: list[str]
    history_years: int = Field(default=4, ge=1)

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, values: list[str]) -> list[str]:
        normalized = [value.strip().upper() for value in values]
        if not normalized or len(normalized) != len(set(normalized)):
            raise ValueError("Symbols must be non-empty and unique")
        return normalized

    @field_validator("timeframes")
    @classmethod
    def validate_timeframes(cls, values: list[str]) -> list[str]:
        supported = {"1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"}
        if not values or any(value not in supported for value in values):
            raise ValueError("Unsupported Bybit timeframe")
        return values


class StorageConfig(BaseModel):
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    registry_path: Path = Path("data/registry/trade_exp_12.db")


class ProjectConfig(BaseModel):
    app: AppConfig = AppConfig()
    bybit: BybitConfig = BybitConfig()
    universe: UniverseConfig
    storage: StorageConfig = StorageConfig()


@dataclass(frozen=True)
class LoadedConfig:
    settings: ProjectConfig
    root: Path
    source: Path

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else self.root / path


def load_config(path: Path | None = None) -> LoadedConfig:
    source = (path or Path("config/project.yaml")).resolve()
    with source.open(encoding="utf-8") as file:
        settings = ProjectConfig.model_validate(yaml.safe_load(file))
    root = source.parent.parent
    return LoadedConfig(settings=settings, root=root, source=source)
