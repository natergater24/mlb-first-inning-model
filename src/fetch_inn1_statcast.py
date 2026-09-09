#!/usr/bin/env python3
"""
fetch_inn1_statcast.py — Download inning-1-ONLY pitch-level Statcast data
for all seasons 2015-2025 (no outs filter, so every first-inning pitch is
captured, not just the leadoff PA).

Output: data/raw/statcast_inn1_{year}.csv  (one file per season, cached — a
season whose file already exists is skipped so the job is resumable).

This is the data source for the pitcher NRFI profile (src/11): first-inning
velocity, whiff rate, K rate, BB rate, pitch-type usage/whiff, first-pitch
strike rate — all of which need raw inning-1 pitches.

Intended to run overnight:
    caffeinate -i /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 \
        src/fetch_inn1_statcast.py
"""
from __future__ import annotations

import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SAVANT_CSV_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
SEASONS = list(range(2015, 2027))  # 2015..2026 inclusive (2026 = current season, partial)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://baseballsavant.mlb.com/statcast_search",
}


def build_params(season: int, date_gt: str = "", date_lt: str = "") -> dict:
    """Inning 1 only, regular season, no outs filter, all PA."""
    return {
        "all":          "true",
        "hfGT":         "R|",
        "hfSea":        f"{season}|",
        "hfInn":        "1|",            # inning 1 ONLY
        "game_date_gt": date_gt,
        "game_date_lt": date_lt,
        "player_type":  "batter",
        "min_pitches":  "0",
        "min_results":  "0",
        "group_by":     "name",
        "sort_col":     "pitches",
        "sort_order":   "desc",
        "min_abs":      "0",
        "type":         "details",
    }


def fetch_csv(params: dict, retries: int = 4, delay: float = 6.0) -> pd.DataFrame | None:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(SAVANT_CSV_URL, params=params, headers=HEADERS, timeout=180)
            resp.raise_for_status()
            txt = resp.text
            if not txt.strip() or txt.startswith("<!DOCTYPE"):
                print(f"    got HTML not CSV (attempt {attempt})", flush=True)
                time.sleep(delay * attempt)
                continue
            return pd.read_csv(StringIO(txt), low_memory=False)
        except requests.exceptions.Timeout:
            print(f"    timeout (attempt {attempt})", flush=True)
        except requests.exceptions.HTTPError as e:
            print(f"    HTTP {e.response.status_code} (attempt {attempt})", flush=True)
        except Exception as e:
            print(f"    error (attempt {attempt}): {e}", flush=True)
        time.sleep(delay * attempt)
    return None


def fetch_season(season: int) -> bool:
    out = RAW_DIR / f"statcast_inn1_{season}.csv"
    if out.exists():
        print(f"[{season}] already cached ({out.name}, "
              f"{sum(1 for _ in open(out)) - 1} rows) — skipping", flush=True)
        return True

    print(f"[{season}] downloading inning-1 pitches (month chunks)...", flush=True)
    months = pd.date_range(f"{season}-03-01", f"{season}-11-30", freq="MS")
    parts: list[pd.DataFrame] = []
    for ms in months:
        me = (ms + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        m0 = ms.strftime("%Y-%m-%d")
        print(f"  [{season}] {m0} -> {me}", flush=True)
        chunk = fetch_csv(build_params(season, m0, me))
        if chunk is not None and len(chunk):
            parts.append(chunk)
            print(f"    +{len(chunk)} rows", flush=True)
        time.sleep(3)

    if not parts:
        print(f"[{season}] NO DATA RETURNED — leaving uncached for retry", flush=True)
        return False

    df = pd.concat(parts, ignore_index=True).drop_duplicates()
    df.to_csv(out, index=False)
    print(f"[{season}] SAVED {out.name}: {len(df):,} rows, "
          f"{df['game_pk'].nunique():,} games", flush=True)
    return True


def main() -> int:
    t0 = time.time()
    print(f"=== inning-1 Statcast download :: seasons {SEASONS[0]}-{SEASONS[-1]} ===",
          flush=True)
    ok, fail = [], []
    for season in SEASONS:
        st = time.time()
        (ok if fetch_season(season) else fail).append(season)
        print(f"    [{season}] done in {time.time() - st:.0f}s | "
              f"elapsed {(time.time() - t0) / 60:.1f} min\n", flush=True)

    print("=== SUMMARY ===", flush=True)
    print(f"  succeeded: {ok}", flush=True)
    print(f"  failed:    {fail}", flush=True)
    print(f"  total time: {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
