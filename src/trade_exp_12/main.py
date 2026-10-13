from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from trade_exp_12.config import load_config
from trade_exp_12.data.storage import ParquetStorage
from trade_exp_12.registry import Registry
from trade_exp_12.services import DataService
from trade_exp_12.web.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded = load_config(Path(app.state.config_path))
    registry = Registry(loaded.resolve(loaded.settings.storage.registry_path))
    registry.initialize()
    storage = ParquetStorage(
        loaded.resolve(loaded.settings.storage.raw_dir), project_root=loaded.root
    )
    processed_storage = ParquetStorage(
        loaded.resolve(loaded.settings.storage.processed_dir), project_root=loaded.root
    )
    app.state.config = loaded
    app.state.registry = registry
    app.state.storage = storage
    app.state.processed_storage = processed_storage
    app.state.data_service = DataService(loaded, registry, storage, processed_storage)
    yield


def create_app(config_path: str | Path = "config/project.yaml") -> FastAPI:
    app = FastAPI(title="Trade Experiment 12", lifespan=lifespan)
    app.state.config_path = str(config_path)
    static_dir = Path(__file__).parent / "web" / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    app.include_router(router)
    return app


app = create_app()


def run() -> None:
    loaded = load_config()
    uvicorn.run(
        "trade_exp_12.main:app",
        host=loaded.settings.app.host,
        port=loaded.settings.app.port,
        reload=False,
    )


if __name__ == "__main__":
    run()
