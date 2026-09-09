#!/usr/bin/env python3
"""
10_build_yrfi_dataset.py — Ground-truth YRFI/NRFI outcomes for every
regular-season game 2015-2025.

Primary source: MLB Stats API schedule endpoint with `linescore` +
`probablePitcher` hydrate (one HTTP call per season — exact first-inning
runs straight from the official linescore).

First-inning batter identification uses whatever inning-1 Statcast is
cached (data/raw/statcast_inn1_{yr}.csv preferred, then statcast_full_{yr}.csv,
then statcast_{yr}.csv leadoff-only). Games with no Statcast coverage get
null batter slots — re-run after the overnight inning-1 download finishes to
backfill them.

Output: data/processed/yrfi_outcomes.parquet  (one row per game)

Columns:
  game_pk, game_date, game_year, home_team, away_team,
  home_starter_id, away_starter_id, home_starter_name, away_starter_name,
  venue_id, venue_name,
  first_inn_runs_home, first_inn_runs_away, total_first_inn_runs,
  yrfi (bool), nrfi (bool), home_scored (bool), away_scored (bool),
  away_b1..away_b5, home_b1..home_b5   (first-inning batter ids, top of order)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# 2015-2025 is the training/eval history; the current season (completed games
# only) is appended so src/11 can compute current-season pitcher NRFI profiles.
# src/13 restricts the training window to <=2025 itself.
SEASONS = list(range(2015, 2027))  # 2015..2026
OUT = PROC / "yrfi_outcomes.parquet"

SESS = requests.Session()
SESS.headers.update({"User-Agent": "Mozilla/5.0 (MLB-Pipeline YRFI builder)"})
SCHED_URL = "https://statsapi.mlb.com/api/v1/schedule"
LINESCORE_URL = "https://statsapi.mlb.com/api/v1/game/{pk}/linescore"


# ── team id -> abbr ─────────────────────────────────────────────────────────
def team_abbr_map() -> dict[int, str]:
    m: dict[int, str] = {}
    gm = PROC / "game_meta.parquet"
    if gm.exists():
        df = pd.read_parquet(gm)
        for _, r in df.iterrows():
            if pd.notna(r.get("home_team_id")):
                m[int(r["home_team_id"])] = str(r["home_team_abbr"])
            if pd.notna(r.get("away_team_id")):
                m[int(r["away_team_id"])] = str(r["away_team_abbr"])
    # hard fallbacks for any id not seen in game_meta
    fallback = {
        108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC", 113: "CIN",
        114: "CLE", 115: "COL", 116: "DET", 117: "HOU", 118: "KC", 119: "LAD",
        120: "WSH", 121: "NYM", 133: "OAK", 134: "PIT", 135: "SD", 136: "SEA",
        137: "SF", 138: "STL", 139: "TB", 140: "TEX", 141: "TOR", 142: "MIN",
        143: "PHI", 144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
    }
    for k, v in fallback.items():
        m.setdefault(k, v)
    return m


# ── first-inning runs from official linescore ───────────────────────────────
def fetch_season_games(season: int) -> list[dict]:
    """One call: every regular-season game with linescore + probable pitchers."""
    params = {
        "sportId": 1,
        "season": season,
        "gameType": "R",
        "hydrate": "linescore,probablePitcher",
    }
    for attempt in range(1, 5):
        try:
            d = SESS.get(SCHED_URL, params=params, timeout=90).json()
            return [g for dt in d.get("dates", []) for g in dt.get("games", [])]
        except Exception as e:
            print(f"  [{season}] schedule fetch attempt {attempt} failed: {e}", flush=True)
            time.sleep(2 * attempt)
    return []


def first_inning_from_linescore(g: dict) -> tuple[int, int] | None:
    innings = (g.get("linescore") or {}).get("innings") or []
    if not innings:
        return None
    i1 = innings[0]
    home = (i1.get("home") or {}).get("runs")
    away = (i1.get("away") or {}).get("runs")
    if home is None and away is None:
        return None
    return int(away or 0), int(home or 0)


def fetch_single_linescore(pk: int) -> tuple[int, int] | None:
    """Fallback for a game the season call didn't carry a linescore for."""
    for attempt in range(1, 4):
        try:
            d = SESS.get(LINESCORE_URL.format(pk=pk), timeout=30).json()
            innings = d.get("innings") or []
            if not innings:
                return None
            i1 = innings[0]
            return int((i1.get("away") or {}).get("runs") or 0), \
                   int((i1.get("home") or {}).get("runs") or 0)
        except Exception:
            time.sleep(0.5 * attempt)
    return None


# ── first-inning batters (top of order) from cached Statcast ────────────────
def load_inn1_statcast(season: int) -> pd.DataFrame | None:
    for name in (f"statcast_inn1_{season}.csv",
                 f"statcast_full_{season}.csv",
                 f"statcast_{season}.csv"):
        p = RAW / name
        if p.exists():
            df = pd.read_csv(p, low_memory=False,
                             usecols=lambda c: c in {
                                 "game_pk", "inning", "inning_topbot", "batter",
                                 "pitcher", "at_bat_number", "game_type"})
            if "inning" in df.columns:
                df = df[df["inning"] == 1]
            if "game_type" in df.columns:
                df = df[df["game_type"] == "R"]
            if len(df):
                print(f"  [{season}] first-inn batters from {name} "
                      f"({df['game_pk'].nunique()} games)", flush=True)
                return df
    print(f"  [{season}] no cached inning-1 Statcast — batter slots left null", flush=True)
    return None


def top5_batters(df: pd.DataFrame | None) -> dict[int, dict]:
    """game_pk -> {away_b1..5, home_b1..5} using PA order within inning 1."""
    out: dict[int, dict] = {}
    if df is None or not len(df):
        return out
    d = df.dropna(subset=["batter", "at_bat_number"]).copy()
    d["batter"] = d["batter"].astype("int64")
    d["at_bat_number"] = d["at_bat_number"].astype("int64")
    # one row per PA
    pa = (d.sort_values("at_bat_number")
            .drop_duplicates(["game_pk", "at_bat_number"]))
    for pk, grp in pa.groupby("game_pk"):
        rec: dict = {}
        for half, pref in (("Top", "away"), ("Bot", "home")):
            seq = grp[grp["inning_topbot"] == half].sort_values("at_bat_number")
            ids: list[int] = []
            for b in seq["batter"].tolist():
                if b not in ids:
                    ids.append(b)
                if len(ids) == 5:
                    break
            for i in range(5):
                rec[f"{pref}_b{i + 1}"] = ids[i] if i < len(ids) else np.nan
        out[int(pk)] = rec
    return out


# ── main ───────────────────────────────────────────────────────────────────
def main() -> int:
    t0 = time.time()
    abbr = team_abbr_map()
    rows: list[dict] = []
    single_calls = 0

    for season in SEASONS:
        st = time.time()
        games = fetch_season_games(season)
        print(f"[{season}] {len(games)} regular-season games "
              f"({time.time() - st:.1f}s)", flush=True)

        sc = load_inn1_statcast(season)
        batters = top5_batters(sc)

        n_yrfi = n_done = 0
        for g in games:
            status = (g.get("status") or {}).get("detailedState", "")
            if status not in ("Final", "Completed Early", "Game Over"):
                continue
            pk = int(g["gamePk"])
            fi = first_inning_from_linescore(g)
            if fi is None:
                fi = fetch_single_linescore(pk)
                single_calls += 1
                time.sleep(0.3)
            if fi is None:
                continue
            away_r, home_r = fi
            total = away_r + home_r

            home_t = g["teams"]["home"]["team"]
            away_t = g["teams"]["away"]["team"]
            hpp = (g["teams"]["home"].get("probablePitcher") or {})
            app = (g["teams"]["away"].get("probablePitcher") or {})
            ven = g.get("venue") or {}

            rec = {
                "game_pk": pk,
                "game_date": pd.to_datetime(g["gameDate"]).tz_convert("America/New_York").date()
                if g.get("gameDate") else pd.NaT,
                "game_year": season,
                "home_team": abbr.get(int(home_t["id"]), str(home_t.get("id"))),
                "away_team": abbr.get(int(away_t["id"]), str(away_t.get("id"))),
                "home_team_id": int(home_t["id"]),
                "away_team_id": int(away_t["id"]),
                "home_starter_id": int(hpp["id"]) if hpp.get("id") else np.nan,
                "away_starter_id": int(app["id"]) if app.get("id") else np.nan,
                "home_starter_name": hpp.get("fullName"),
                "away_starter_name": app.get("fullName"),
                "venue_id": int(ven["id"]) if ven.get("id") else np.nan,
                "venue_name": ven.get("name"),
                "first_inn_runs_home": home_r,
                "first_inn_runs_away": away_r,
                "total_first_inn_runs": total,
                "yrfi": total > 0,
                "nrfi": total == 0,
                "home_scored": home_r > 0,
                "away_scored": away_r > 0,
            }
            rec.update(batters.get(pk, {f"{p}_b{i}": np.nan
                                        for p in ("away", "home") for i in range(1, 6)}))
            rows.append(rec)
            n_done += 1
            n_yrfi += int(total > 0)

        rate = 100 * n_yrfi / n_done if n_done else 0
        print(f"[{season}] {n_done} finals, YRFI {rate:.1f}%  "
              f"| cum elapsed {(time.time() - t0) / 60:.1f} min\n", flush=True)

    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    # starter backfill from Statcast where probablePitcher was missing
    df = df.sort_values(["game_year", "game_date", "game_pk"]).reset_index(drop=True)
    df.to_parquet(OUT, index=False)

    # ── summary ────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"SAVED {OUT.relative_to(ROOT)}  ({len(df):,} games)")
    print(f"single-game linescore fallback calls: {single_calls}")
    print(f"overall YRFI rate: {100 * df['yrfi'].mean():.2f}%")
    print(f"overall NRFI rate: {100 * df['nrfi'].mean():.2f}%")
    print(f"avg total 1st-inn runs: {df['total_first_inn_runs'].mean():.3f}")

    print("\nYRFI rate by season:")
    by_s = df.groupby("game_year")["yrfi"].agg(["mean", "count"])
    for yr, r in by_s.iterrows():
        print(f"  {yr}: {100 * r['mean']:.1f}%   (n={int(r['count'])})")

    print("\nYRFI rate by month:")
    by_m = df.assign(month=df["game_date"].dt.month).groupby("month")["yrfi"].agg(["mean", "count"])
    for mo, r in by_m.iterrows():
        print(f"  {int(mo):>2}: {100 * r['mean']:.1f}%   (n={int(r['count'])})")

    print("\nHome vs away scored in 1st:")
    print(f"  home team scores: {100 * df['home_scored'].mean():.1f}%")
    print(f"  away team scores: {100 * df['away_scored'].mean():.1f}%")

    n_bat = df["away_b1"].notna().sum()
    print(f"\nfirst-inning batter slots populated: {n_bat:,}/{len(df):,} games "
          f"({100 * n_bat / len(df):.0f}%)")
    print(f"\ntotal time: {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
