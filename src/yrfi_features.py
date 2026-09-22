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

# ── small-sample correction for the NRFI-rate pitcher features ──────────────
# career_nrfi_pct / seas_nrfi_pct / l5_nrfi_pct are observed proportions built
# on wildly varying sample sizes (a rookie's 4-start season rate vs. a
# 15-year veteran's 300-start career rate) -- fed to the model raw, with no
# correction, a noisy 4-start rate moves the RandomForest exactly as hard as
# a reliable 300-start one. This is a standard credibility/empirical-Bayes
# blend: shrunk = (n*observed + k*prior) / (n+k). k is the "pseudo-start"
# weight -- how many prior-strength starts the league-average prior is worth.
# Diagnosed 2026-09-18 against a real case: ATH@CLE priced YRFI at 85%/-580
# off Mason Barnett's 25% season NRFI rate over just 4 starts, while every
# sportsbook had the game near a coin flip.
NRFI_LEAGUE_PRIOR = 70.0   # matches app.py's _WHY_EDGE_LEAGUE_NRFI constant
SHRINK_K_CAREER = 12.0     # ~12 "pseudo-starts" of prior weight
# Re-tuned 2026-09-21: k=12 for season left a 4-start rate (e.g. Mason
# Barnett's 25% seas_nrfi_pct in the ATH@CLE case that motivated this file's
# shrinkage) still moving the model almost as hard as before. A method-of-
# moments credibility estimate against established (>=20-start) pitchers'
# career nrfi_pct (between-pitcher variance vs. binomial sampling variance)
# came back far higher than 12 -- noisy (it's a small residual of two close
# numbers) but directionally consistent with under-shrinkage. Moved to 25 as
# a validated middle ground, not the raw estimate: re-checked against the
# 2024 aggregate backtest (go/no-go gate) and the ATH@CLE case directly
# before shipping, see docs/superpowers/plans/2026-09-18-pitcher-sample-size-model-fix.md.
SHRINK_K_SEASON = 25.0
SHRINK_K_L5 = 4.0          # L5's full sample is only 5 starts -- a k of 12
                           # would swamp it completely; 4 tempers without
                           # neutering it (a perfect 5/5 still moves ~4/9 of
                           # the way to the prior, not all the way).

# Added 2026-09-21: first_inn_era is the same kind of small-sample statistic
# as the NRFI-rate features above (computed from the same total_starts) but
# was never shrunk in the first pass -- it's highly correlated with
# career_nrfi_pct, so once that feature's signal was pulled toward the
# prior, the model recovered a large chunk of it from era instead (its
# logistic-surrogate coefficient jumped ~6x on retrain, confirmed against
# the ATH@CLE case: Barnett's raw 11.0 career first_inn_era on 9 starts vs a
# league (>=15-start pitchers) mean of ~4.85-5.05). Same shrink treatment,
# same n (total_starts), same k as career_nrfi_pct.
ERA_LEAGUE_PRIOR = 4.85    # start-weighted league mean across all pitchers
                           # (4.86); matches the established (>=15-start)
                           # cohort's mean (5.05) and median (4.82) closely.
SHRINK_K_ERA = 12.0

# ── post-hoc confidence blend, applied to the model's OUTPUT probability ────
# Added 2026-09-21. The input-side shrinkage above (career/seas/l5_nrfi_pct,
# first_inn_era) was confirmed -- via a diagnostic k=100 retrain that pushed
# a thin-sample rate value to be indistinguishable from the league prior --
# to have essentially no effect on the ATH@CLE motivating case (824383,
# 2026-09-18: 85.3% -> 84.2% -> 84.4% across three attempts). A RandomForest
# re-derives split-based separation from whatever residual/correlated signal
# remains and is not linearly responsive to how far a continuous input is
# shrunk toward a prior, the way a linear model would be. This blends the
# FINAL probability toward a neutral 0.5 directly, weighted by confidence =
# min(home_starts, away_starts) / (min_starts + CONFIDENCE_BLEND_K) -- a
# guaranteed, direct fix instead of hoping the model learns it indirectly.
# k=3 chosen by checking both ends against real cases: a total debut (0
# starts) always fully neutralizes regardless of k (min=0 forces weight=0);
# an established pair (e.g. 31 vs 84 career starts) drifts only ~0.6pp at
# k=3 (0.4061 -> 0.4144) vs. a much larger, unwanted ~2.3pp drift at k=10.
CONFIDENCE_BLEND_K = 3.0
NEUTRAL_P = 0.5


def blend_toward_neutral(p: float, home_starts: float | None, away_starts: float | None,
                         k: float = CONFIDENCE_BLEND_K) -> float:
    """Pull a predicted probability toward 0.5 based on how little track
    record backs it. Confidence weight = min_starts / (min_starts + k): a
    debut pitcher (0 starts) drives weight to exactly 0 (full neutral,
    regardless of the other side), two well-established starters leave the
    prediction essentially untouched."""
    h = 0.0 if home_starts is None or pd.isna(home_starts) else float(home_starts)
    a = 0.0 if away_starts is None or pd.isna(away_starts) else float(away_starts)
    m = min(h, a)
    weight = m / (m + k)
    return NEUTRAL_P + (float(p) - NEUTRAL_P) * weight


def _shrink(rate: float | None, n: float | None, prior: float, k: float) -> float:
    """Credibility-weighted blend of an observed rate toward a prior. Always
    returns a real float -- a missing/zero-sample pitcher (debut) returns
    exactly `prior`, replacing what used to be a NaN fed into training-median
    imputation with an explicit, principled default."""
    n = 0.0 if n is None or pd.isna(n) else float(n)
    rate = prior if rate is None or pd.isna(rate) else float(rate)
    return round((n * rate + k * prior) / (n + k), 4)

PITCHER_FEATS = [
    "career_nrfi_pct", "seas_nrfi_pct", "l5_nrfi_pct", "first_inn_era",
    "first_inn_k_rate", "first_inn_bb_rate", "first_inn_hard_hit",
    "first_pitch_strike_rate", "avg_velo", "primary_pitch_code",
]
# Sample-size / data-quality signals -- deliberately NOT part of any
# FEATURE_GROUPS entry (see the "Columns not named in any group... always
# contribute at weight 1" comment above) since these describe confidence in
# the data, not a team-strength factor the dashboard's sliders should scale.
PITCHER_CONFIDENCE_FEATS = ["starts_log", "seas_starts_log", "is_debut"]
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
        out[f"{prefix}_pitcher_starts_log"] = 0.0
        out[f"{prefix}_pitcher_seas_starts_log"] = 0.0
        out[f"{prefix}_pitcher_is_debut"] = 1
        return out
    out[f"{prefix}_pitcher_career_nrfi_pct"] = _shrink(
        p.get("nrfi_pct"), p.get("total_starts"), NRFI_LEAGUE_PRIOR, SHRINK_K_CAREER)
    out[f"{prefix}_pitcher_seas_nrfi_pct"] = _shrink(
        p.get("seas_nrfi_pct"), p.get("seas_starts"), NRFI_LEAGUE_PRIOR, SHRINK_K_SEASON)
    out[f"{prefix}_pitcher_l5_nrfi_pct"] = _shrink(
        p.get("l5_nrfi_pct"), p.get("l5_starts"), NRFI_LEAGUE_PRIOR, SHRINK_K_L5)
    out[f"{prefix}_pitcher_first_inn_era"] = _shrink(
        p.get("first_inn_era"), p.get("total_starts"), ERA_LEAGUE_PRIOR, SHRINK_K_ERA)
    out[f"{prefix}_pitcher_first_inn_k_rate"] = p.get("first_inn_k_rate")
    out[f"{prefix}_pitcher_first_inn_bb_rate"] = p.get("first_inn_bb_rate")
    out[f"{prefix}_pitcher_first_inn_hard_hit"] = p.get("first_inn_hard_hit_allowed")
    out[f"{prefix}_pitcher_first_pitch_strike_rate"] = p.get("first_pitch_strike_rate")
    out[f"{prefix}_pitcher_avg_velo"] = p.get("first_inn_avg_velo")
    out[f"{prefix}_pitcher_primary_pitch_code"] = PRIMARY_PITCH_CODES.get(
        p.get("primary_pitch"), -1)
    total_starts = p.get("total_starts")
    seas_starts = p.get("seas_starts")
    out[f"{prefix}_pitcher_starts_log"] = float(np.log1p(total_starts)) if pd.notna(total_starts) else 0.0
    out[f"{prefix}_pitcher_seas_starts_log"] = float(np.log1p(seas_starts)) if pd.notna(seas_starts) else 0.0
    out[f"{prefix}_pitcher_is_debut"] = 0
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
    for side in ("home", "away"):
        cols += [f"{side}_pitcher_{f}" for f in PITCHER_CONFIDENCE_FEATS]
    for side in ("away", "home"):
        cols += [f"{side}_top5_{f}" for f in TOP5_FEATS]
    cols += CONTEXT_FEATS
    return cols
