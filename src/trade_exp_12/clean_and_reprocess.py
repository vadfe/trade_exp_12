"""Clean processed/pretrain/experiment data and reprocess all raw datasets.

Usage::

    python -m trade_exp_12.clean_and_reprocess

This removes stale processed parquets, pretrained encoders, and v2 experiments,
then reprocesses every raw dataset through the current feature pipeline.  Run
it whenever the feature set changes so all downstream data has one consistent
dimension.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from trade_exp_12.config import load_config
from trade_exp_12.data.storage import ParquetStorage
from trade_exp_12.registry import Registry
from trade_exp_12.services import DataService


def _rm_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


async def main() -> None:
    config = load_config(Path("config/project.yaml"))
    root = config.root

    dirs_to_clean = [
        root / "data" / "processed",
        root / "data" / "pretrain",
        root / "data" / "experiments_v2",
    ]

    for d in dirs_to_clean:
        print(f"Removing {d} ...")
        _rm_tree(d)

    registry = Registry(root / "data" / "registry" / "trade_exp_12.db")
    raw_storage = ParquetStorage(
        root / config.settings.storage.raw_dir, project_root=root
    )
    proc_storage = ParquetStorage(
        root / config.settings.storage.processed_dir, project_root=root
    )
    ds = DataService(config, registry, raw_storage, proc_storage)

    for sym in config.settings.universe.symbols:
        for tf in config.settings.universe.timeframes:
            rec = registry.latest_dataset(sym, tf)
            if rec is None:
                continue
            print(f"Reprocessing {sym} {tf}m ...")
            job_id = registry.create_job("reprocess", sym, tf)
            await ds.process(job_id, sym, tf)
            job = registry.get_job(job_id)
            print(f"  {job['status']}: {job['message']}")

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
