"""Remove all crypto (Bybit) data: registry rows, raw/processed parquets,
and pretrain run dirs that contain no MOEX share.

Usage::

    python -m trade_exp_12.remove_crypto
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

from trade_exp_12.config import load_config


def main() -> None:
    config = load_config(Path("config/project.yaml"))
    root = config.root
    keep = set(config.settings.moex.symbols)

    db_path = root / "data" / "registry" / "trade_exp_12.db"
    conn = sqlite3.connect(db_path)
    try:
        placeholders = ','.join('?' * len(keep))
        for table in ("datasets", "processed_datasets"):
            cur = conn.execute(
                f"DELETE FROM {table} WHERE symbol NOT IN ({placeholders})",
                sorted(keep),
            )
            print(f"{table}: removed {cur.rowcount} crypto rows")
        cur = conn.execute(
            f"DELETE FROM jobs WHERE symbol NOT IN ({placeholders}) AND symbol != 'ALL'",
            sorted(keep),
        )
        print(f"jobs: removed {cur.rowcount} crypto rows")
        conn.commit()
    finally:
        conn.close()

    removed_dirs = []
    for category_dir in (root / "data" / "raw", root / "data" / "processed"):
        if category_dir.exists():
            for sub in category_dir.iterdir():
                if sub.is_dir() and sub.name != "shares":
                    shutil.rmtree(sub)
                    removed_dirs.append(str(sub.relative_to(root)))
    for extra in ("experiments", "experiments_v2"):
        p = root / "data" / extra
        if p.exists():
            for sym_dir in p.iterdir():
                if sym_dir.is_dir() and sym_dir.name not in keep:
                    shutil.rmtree(sym_dir)
                    removed_dirs.append(str(sym_dir.relative_to(root)))

    pretrain = root / "data" / "pretrain"
    if pretrain.exists():
        for run_dir in pretrain.iterdir():
            report = run_dir / "report.json"
            if not report.exists():
                continue
            pairs = json.loads(report.read_text()).get("info", {}).get("pairs", [])
            instruments = {p.split("/")[0] for p in pairs}
            if instruments and instruments.isdisjoint(keep):
                shutil.rmtree(run_dir)
                removed_dirs.append(str(run_dir.relative_to(root)))

    for d in removed_dirs:
        print("removed", d)
    print(f"Done. {len(removed_dirs)} directories removed.")


if __name__ == "__main__":
    main()
