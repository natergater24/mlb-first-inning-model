"""
run_pipeline.py — YRFI / NRFI pipeline runner.

Usage:
  python run_pipeline.py --today-only      # daily: starters, rosters, weather,
                                           #        first-inning odds, predictions
  python run_pipeline.py                    # full historical rebuild + train + today
  python run_pipeline.py --rebuild-history  # alias for the full rebuild

The project predicts one thing: will a run be scored in the first inning of
today's games (YRFI), vs no run (NRFI) — compared against first-inning
sportsbook markets to find edges.
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SRC = BASE_DIR / "src"
PROC = BASE_DIR / "data" / "processed"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

PY = "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
CAFFEINATE = ["caffeinate", "-i"]

LOGFILE = LOG_DIR / "pipeline.log"


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a") as fh:
        fh.write(line + "\n")


def step(label: str, args: list[str], t_start: float, n: int, total: int) -> None:
    log(f"[{n}/{total}] {label}")
    s = time.time()
    r = subprocess.run(CAFFEINATE + [PY] + args, cwd=BASE_DIR)
    dt = time.time() - s
    if r.returncode != 0:
        log(f"    ✗ FAILED ({dt:.0f}s) — {' '.join(args)}  [exit {r.returncode}]")
        sys.exit(r.returncode)
    elapsed = time.time() - t_start
    log(f"    ✓ done in {dt:.0f}s  |  total elapsed {elapsed / 60:.1f} min")


def run_today_only() -> None:
    t0 = time.time()
    log("=" * 70)
    log(f"YRFI DAILY UPDATE — {datetime.now():%Y-%m-%d}")
    log("=" * 70)

    stale = PROC / "todays_yrfi_predictions.parquet"
    if stale.exists():
        stale.unlink()
        log(f"deleted stale {stale.name}")

    steps = [
        ("Probable pitchers",        ["src/02_fetch_mlb_api.py", "--probable-pitchers-only"]),
        ("Active rosters",           ["src/02_fetch_mlb_api.py", "--rosters-only"]),
        ("Weather forecast",         ["src/03_fetch_weather.py", "--today-only"]),
        ("First-inning odds",        ["src/04_fetch_odds.py"]),
        ("Today's YRFI predictions", ["src/14_build_todays_yrfi.py"]),
    ]
    for i, (label, args) in enumerate(steps, 1):
        step(label, args, t0, i, len(steps))

    log(f"\nDaily update complete in {(time.time() - t0) / 60:.1f} min.")


def run_full() -> None:
    t0 = time.time()
    log("=" * 70)
    log(f"YRFI FULL REBUILD — {datetime.now():%Y-%m-%d %H:%M}")
    log("=" * 70)

    steps = [
        ("Build YRFI outcomes 2015-2026",      ["src/10_build_yrfi_dataset.py"]),
        ("Rebuild lineup-position history",    ["src/02_fetch_mlb_api.py", "--build-lineup-history"]),
        ("Build pitcher NRFI profiles",        ["src/11_build_pitcher_nrfi.py"]),
        ("Build projected top-order batters",  ["src/12_build_top_order_batter.py"]),
        ("Train YRFI model",                   ["src/13_train_yrfi_model.py"]),
        ("Fetch today's probable pitchers",    ["src/02_fetch_mlb_api.py", "--probable-pitchers-only"]),
        ("Fetch today's rosters",              ["src/02_fetch_mlb_api.py", "--rosters-only"]),
        ("Rebuild weather (all stadiums)",     ["src/03_fetch_weather.py", "--rebuild"]),
        ("Fetch first-inning odds",            ["src/04_fetch_odds.py"]),
        ("Build today's YRFI predictions",     ["src/14_build_todays_yrfi.py"]),
    ]
    for i, (label, args) in enumerate(steps, 1):
        step(label, args, t0, i, len(steps))

    log(f"\nFull rebuild complete in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    if "--today-only" in sys.argv:
        run_today_only()
    else:
        run_full()
