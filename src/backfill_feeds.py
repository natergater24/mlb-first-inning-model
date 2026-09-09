#!/usr/bin/env python3
"""
backfill_feeds.py — Download any completed MLB game feed that isn't cached yet.

The daily pipeline only fetches *today's* feeds, so after a gap the cache misses
weeks of games — which leaves lineup_position_summary, batter_game_logs and every
feed-derived table stale. This walks yrfi_outcomes.parquet (all games 2015-2026,
Final only) and fetches every feed missing from data/raw/game_feeds/.

Usage:
    caffeinate -i .../python3 src/backfill_feeds.py            # current season
    caffeinate -i .../python3 src/backfill_feeds.py --all      # every season
    caffeinate -i .../python3 src/backfill_feeds.py --since 2026-07-01
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import importlib.util
_spec = importlib.util.spec_from_file_location("m2", ROOT / "src" / "02_fetch_mlb_api.py")
_m2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m2)

FEEDS = ROOT / "data" / "raw" / "game_feeds"
FEEDS.mkdir(parents=True, exist_ok=True)
CURRENT_SEASON = 2026


def main() -> int:
    t0 = time.time()
    args = sys.argv[1:]
    y = pd.read_parquet(ROOT / "data" / "processed" / "yrfi_outcomes.parquet")
    y["game_date"] = pd.to_datetime(y["game_date"], errors="coerce")

    if "--all" in args:
        sel = y
    elif any(a.startswith("--since") for a in args):
        since = [a for a in args if a.startswith("--since")][0].split("=")[-1] \
            if "=" in "".join(args) else args[args.index("--since") + 1]
        sel = y[y["game_date"] >= pd.Timestamp(since)]
    else:
        sel = y[y["game_year"] == CURRENT_SEASON]

    pks = sorted(int(p) for p in sel["game_pk"].unique())
    missing = [p for p in pks if not (FEEDS / f"{p}.json").exists()]
    print(f"{len(pks)} games in scope · {len(missing)} feeds missing", flush=True)
    if not missing:
        print("nothing to backfill.")
        return 0

    import json
    ok = fail = 0
    for i, pk in enumerate(missing, 1):
        feed = _m2.fetch_game_feed(pk)
        if feed:
            (FEEDS / f"{pk}.json").write_text(json.dumps(feed))
            ok += 1
        else:
            fail += 1
        if i % 100 == 0:
            print(f"  {i}/{len(missing)}  ok={ok} fail={fail}  "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        time.sleep(0.15)

    print(f"\ndone: fetched {ok}, failed {fail}  ({(time.time() - t0) / 60:.1f} min)")
    print("now re-run:  src/02_fetch_mlb_api.py --build-lineup-history  ->  src/12  ->  src/14")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
