#!/usr/bin/env python3
"""
yrfi_features.py — shared game-level feature builder for the YRFI model.

Used by src/13 (training, historical games) and src/14 (inference, today).

A "game row" must have at least:
  game_pk, game_date, home_team, away_team,
  home_starter_id, away_starter_id
Optionally: venue abbr, temp_f, wind_speed, wind_out_component, humidity,
            is_dome, umpire_zone_adj, and the away_/home_top5_* aggregates.

Historical caveat: pitcher profile stats and top-5 batter aggregates are the
*current* static values (no point-in-time reconstruction), so career features
carry mild look-ahead bias. Documented in yrfi_model_meta.json.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from park_factors import get_park  # noqa: E402

PRIMARY_PITCH_CODES = {"fb": 0, "sl": 1, "cb": 2, "ch": 3, None: -1, np.nan: -1}

PITCHER_FEATS = [
    "career_nrfi_pct", "seas_nrfi_pct", "l5_nrfi_pct", "first_inn_era",
    "first_inn_k_rate", "first_inn_bb_rate", "first_inn_hard_hit",
    "first_pitch_strike_rate", "avg_velo", "primary_pitch_code",
]
TOP5_FEATS = [
    "avg_obp_vs_pitcher", "avg_bvp_pa", "avg_l7_obp", "avg_l30_obp",
    "avg_seas_hr_per_pa", "avg_hard_hit", "power_score",
]
CONTEXT_FEATS = [
    "park_run_factor", "park_hr_factor", "is_dome", "temp_f", "wind_speed",
    "wind_out_component", "humidity", "umpire_zone_adj", "month",
]

# ── feature groups for the dashboard weighting sliders ──────────────────────
# Every entry maps a slider to the model feature columns it scales. Columns not
# named in any group ("month", "umpire_zone_adj") always contribute at weight 1.
FEATURE_GROUPS: dict[str, list[str]] = {
    "pitcher_stuff": [f"{s}_pitcher_{m}" for s in ("home", "away") for m in
                      ("first_inn_era", "first_inn_k_rate", "first_inn_bb_rate",
                       "first_inn_hard_hit", "first_pitch_strike_rate",
                       "avg_velo", "primary_pitch_code")],
    "pitcher_nrfi_record": [f"{s}_pitcher_{m}" for s in ("home", "away") for m in
                            ("career_nrfi_pct", "seas_nrfi_pct")],
    "pitcher_last5": [f"{s}_pitcher_l5_nrfi_pct" for s in ("home", "away")],
    "lineup_offense": [f"{s}_top5_{m}" for s in ("away", "home") for m in
                       ("avg_obp_vs_pitcher", "avg_bvp_pa", "avg_seas_hr_per_pa",
                        "avg_hard_hit", "power_score")],
    "lineup_recent_form": [f"{s}_top5_{m}" for s in ("away", "home") for m in
                           ("avg_l7_obp", "avg_l30_obp")],
    "weather": ["temp_f", "wind_speed", "wind_out_component", "humidity"],
    "ballpark": ["park_run_factor", "park_hr_factor", "is_dome"],
}
GROUP_LABELS: dict[str, str] = {
    "pitcher_stuff": "Pitcher stuff (velo · K% · BB% · contact allowed)",
    "pitcher_nrfi_record": "Pitcher NRFI track record (career + season)",
    "pitcher_last5": "Pitcher recent form (last 5 starts)",
    "lineup_offense": "Lineup power & matchup (BvP OBP · SLG · HR rate)",
    "lineup_recent_form": "Lineup recent form (L7 / L30 OBP)",
    "weather": "Weather (temperature · wind)",
    "ballpark": "Ballpark run environment",
}


def _prof_row(prof: pd.DataFrame) -> dict:
    idx = prof.set_index("pitcher_id")
    return idx.to_dict("index")


def _pitcher_features(pid, prof_map: dict, prefix: str) -> dict:
    p = prof_map.get(int(pid)) if pd.notna(pid) else None
    out = {}
    if p is None:
        for f in PITCHER_FEATS:
            out[f"{prefix}_pitcher_{f}"] = np.nan
        return out
    out[f"{prefix}_pitcher_career_nrfi_pct"] = p.get("nrfi_pct")
    out[f"{prefix}_pitcher_seas_nrfi_pct"] = p.get("seas_nrfi_pct")
    out[f"{prefix}_pitcher_l5_nrfi_pct"] = p.get("l5_nrfi_pct")
    out[f"{prefix}_pitcher_first_inn_era"] = p.get("first_inn_era")
    out[f"{prefix}_pitcher_first_inn_k_rate"] = p.get("first_inn_k_rate")
    out[f"{prefix}_pitcher_first_inn_bb_rate"] = p.get("first_inn_bb_rate")
    out[f"{prefix}_pitcher_first_inn_hard_hit"] = p.get("first_inn_hard_hit_allowed")
    out[f"{prefix}_pitcher_first_pitch_strike_rate"] = p.get("first_pitch_strike_rate")
    out[f"{prefix}_pitcher_avg_velo"] = p.get("first_inn_avg_velo")
    out[f"{prefix}_pitcher_primary_pitch_code"] = PRIMARY_PITCH_CODES.get(
        p.get("primary_pitch"), -1)
    return out


def _top5_agg(team_abbr: str, stats_by_team: dict, prefix: str) -> dict:
    rows = stats_by_team.get(team_abbr, [])
    out = {}
    def m(key):
        vals = [r[key] for r in rows if pd.notna(r.get(key))]
        return float(np.mean(vals)) if vals else np.nan
    out[f"{prefix}_top5_avg_obp_vs_pitcher"] = m("_obp_vs_pitcher")  # filled by src/14; else seas_obp
    if np.isnan(out[f"{prefix}_top5_avg_obp_vs_pitcher"]):
        out[f"{prefix}_top5_avg_obp_vs_pitcher"] = m("seas_obp")
    out[f"{prefix}_top5_avg_bvp_pa"] = m("_bvp_pa")
    out[f"{prefix}_top5_avg_l7_obp"] = m("l7_obp")
    out[f"{prefix}_top5_avg_l30_obp"] = m("l30_obp")
    out[f"{prefix}_top5_avg_seas_hr_per_pa"] = m("seas_hr_per_pa")
    out[f"{prefix}_top5_avg_hard_hit"] = m("seas_hard_hit")
    slg = m("seas_slg"); hrpa = m("seas_hr_per_pa")
    out[f"{prefix}_top5_power_score"] = (
        (slg if pd.notna(slg) else 0.38) * 0.6
        + (hrpa if pd.notna(hrpa) else 0.03) * 10 * 0.4)
    return out


def load_support():
    prof = pd.read_parquet(PROC / "pitcher_nrfi_profile.parquet")
    prof_map = _prof_row(prof)
    stats_by_team: dict[str, list] = {}
    sp = PROC / "top5_batter_stats.parquet"
    if sp.exists():
        s = pd.read_parquet(sp)
        for t, g in s.groupby("team_abbr"):
            stats_by_team[t] = g.to_dict("records")
    return prof_map, stats_by_team


def build_game_features(games: pd.DataFrame,
                        prof_map: dict | None = None,
                        stats_by_team: dict | None = None,
                        weather: pd.DataFrame | None = None) -> pd.DataFrame:
    if prof_map is None or stats_by_team is None:
        prof_map, stats_by_team = load_support()

    g = games.copy()
    g["game_date"] = pd.to_datetime(g["game_date"], errors="coerce")
    g["month"] = g["game_date"].dt.month

    if weather is None:
        wp = PROC / "game_weather.parquet"
        weather = pd.read_parquet(wp) if wp.exists() else pd.DataFrame()
    wcols = ["temp_f", "humidity", "wind_speed", "wind_out_component", "precip_prob"]
    if len(weather):
        w = weather.drop_duplicates("game_pk").set_index("game_pk")[
            [c for c in wcols if c in weather.columns]]
        g = g.merge(w, left_on="game_pk", right_index=True, how="left")
    for c in wcols:
        if c not in g.columns:
            g[c] = np.nan

    recs = []
    for _, row in g.iterrows():
        rec = {"game_pk": row["game_pk"]}
        rec.update(_pitcher_features(row.get("home_starter_id"), prof_map, "home"))
        rec.update(_pitcher_features(row.get("away_starter_id"), prof_map, "away"))
        # away team bats first
        rec.update(_top5_agg(row.get("away_team"), stats_by_team, "away"))
        rec.update(_top5_agg(row.get("home_team"), stats_by_team, "home"))

        park = get_park(row.get("home_team"))
        rec["park_run_factor"] = park["run_factor"]
        rec["park_hr_factor"] = park["hr_factor"]
        dome = bool(row["is_dome"]) if "is_dome" in row and pd.notna(row.get("is_dome")) \
            else park["is_dome"]
        rec["is_dome"] = int(dome)
        rec["temp_f"] = 72.0 if dome else row.get("temp_f")
        rec["wind_speed"] = 0.0 if dome else row.get("wind_speed")
        rec["wind_out_component"] = 0.0 if dome else row.get("wind_out_component")
        rec["humidity"] = 50.0 if dome else row.get("humidity")
        rec["umpire_zone_adj"] = row.get("umpire_zone_adj", 0.0) or 0.0
        rec["month"] = row.get("month")
        recs.append(rec)

    feat = pd.DataFrame(recs)
    return feat


def feature_columns() -> list[str]:
    cols = []
    for side in ("home", "away"):
        cols += [f"{side}_pitcher_{f}" for f in PITCHER_FEATS]
    for side in ("away", "home"):
        cols += [f"{side}_top5_{f}" for f in TOP5_FEATS]
    cols += CONTEXT_FEATS
    return cols
