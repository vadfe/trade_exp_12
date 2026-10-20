"""Reprocess all raw datasets with current feature pipeline.

Usage::

    python -m trade_exp_12.reprocess
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from trade_exp_12.config import load_config
from trade_exp_12.data.storage import ParquetStorage
from trade_exp_12.registry import Registry
from trade_exp_12.services import DataService


async def main() -> None:
    config = load_config(Path("config/project.yaml"))
    root = config.root
    registry = Registry(root / "data" / "registry" / "trade_exp_12.db")
    raw_storage = ParquetStorage(
        root / config.settings.storage.raw_dir, project_root=root
    )
    proc_storage = ParquetStorage(
        root / config.settings.storage.processed_dir, project_root=root
    )
    ds = DataService(config, registry, raw_storage, proc_storage)

    combos = [(s, tf) for s in config.settings.universe.symbols
              for tf in config.settings.universe.timeframes]
    moex = getattr(config.settings, "moex", None)
    if moex is not None:
        combos += [(s, tf) for s in moex.symbols for tf in moex.timeframes]

    for sym, tf in combos:
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
