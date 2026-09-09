#!/usr/bin/env python3
"""
14_build_todays_yrfi.py — YRFI/NRFI predictions for every game on the slate.

Pulls today's probable starters, projected top-5 lineups, pitcher NRFI
profiles, BvP for each top-5 batter vs the opposing starter, weather, park
factors and (if present) the umpire assignment; assembles model features and
runs the calibrated YRFI model.

Output: data/processed/todays_yrfi_predictions.parquet  (one row per game)

If src/04_fetch_odds.py has written first-inning market odds into
data/odds/yrfi_odds_{date}.parquet they are joined here and edge / EV /
recommended-bet columns are computed; otherwise those columns are NaN and
recommended_bet is derived from the model alone (>60% threshold).
"""
from __future__ import annotations

import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
ODDS = ROOT / "data" / "odds"
MODELS = ROOT / "data" / "models"

sys.path.insert(0, str(ROOT / "src"))
from yrfi_features import build_game_features, feature_columns, FEATURE_GROUPS  # noqa: E402
from park_factors import get_park  # noqa: E402

TODAY = date.today().isoformat()


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def group_contributions(feat_row: pd.Series, feat_cols: list, surrogate: dict) -> dict:
    """Log-odds contribution of each FEATURE_GROUP for one game, from the
    logistic-regression surrogate. Columns in no group are pooled as 'other'."""
    coef, mean, std = surrogate["coef"], surrogate["mean"], surrogate["std"]
    z = {}
    for c in feat_cols:
        v = pd.to_numeric(pd.Series([feat_row.get(c)]), errors="coerce").iloc[0]
        if pd.isna(v):
            v = mean.get(c, 0.0)
        s = std.get(c, 1.0) or 1.0
        z[c] = (v - mean.get(c, 0.0)) / s
    grouped = {m for members in FEATURE_GROUPS.values() for m in members}
    out = {}
    for g, members in FEATURE_GROUPS.items():
        out[f"contrib_{g}"] = round(
            sum(coef.get(m, 0.0) * z.get(m, 0.0) for m in members), 4)
    out["contrib_other"] = round(
        sum(coef.get(c, 0.0) * z.get(c, 0.0) for c in feat_cols if c not in grouped), 4)
    return out


def team_records(season: int) -> dict[int, str]:
    """{team_id: 'W-L'} from the MLB standings endpoint (one free call)."""
    import requests
    try:
        d = requests.get(
            "https://statsapi.mlb.com/api/v1/standings",
            params={"leagueId": "103,104", "season": season,
                    "standingsTypes": "regularSeason"},
            headers={"User-Agent": "Mozilla/5.0"}, timeout=20).json()
        out = {}
        for grp in d.get("records", []):
            for tr in grp.get("teamRecords", []):
                out[int(tr["team"]["id"])] = f"{tr['wins']}-{tr['losses']}"
        return out
    except Exception as e:
        print(f"  standings fetch failed ({e}) — records left blank", flush=True)
        return {}


def team_id_abbr() -> dict[int, str]:
    m = {}
    gm = PROC / "game_meta.parquet"
    if gm.exists():
        d = pd.read_parquet(gm)
        for _, r in d.iterrows():
            m[int(r["home_team_id"])] = str(r["home_team_abbr"])
            m[int(r["away_team_id"])] = str(r["away_team_abbr"])
    fallback = {108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
                113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
                118: "KC", 119: "LAD", 120: "WSH", 121: "NYM", 133: "ATH",
                134: "PIT", 135: "SD", 136: "SEA", 137: "SF", 138: "STL",
                139: "TB", 140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
                144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL"}
    for k, v in fallback.items():
        m.setdefault(k, v)
    return m


def american_odds(p: float) -> float:
    """Implied American odds for probability p."""
    if p is None or not np.isfinite(p) or p <= 0 or p >= 1:
        return np.nan
    if p > 0.5:
        return round(-p / (1 - p) * 100)
    return round((1 - p) / p * 100)


def main() -> int:
    t0 = time.time()

    pp_path = PROC / "probable_pitchers.parquet"
    if not pp_path.exists():
        print("ERROR: probable_pitchers.parquet missing — run "
              "src/02_fetch_mlb_api.py --probable-pitchers-only", flush=True)
        return 1
    pp = pd.read_parquet(pp_path)
    pp["game_date"] = pp["game_date"].astype(str)
    slate = pp[pp["game_date"] == TODAY].copy()
    if slate.empty:
        # fall back to the max date present (useful off-day / testing)
        md = pp["game_date"].max()
        slate = pp[pp["game_date"] == md].copy()
        print(f"no games dated {TODAY}; using latest slate {md} "
              f"({len(slate)} games)", flush=True)
    else:
        print(f"slate {TODAY}: {len(slate)} games", flush=True)

    id2abbr = team_id_abbr()
    records = team_records(int(TODAY[:4]))
    prof = pd.read_parquet(PROC / "pitcher_nrfi_profile.parquet")
    prof_map = prof.set_index("pitcher_id").to_dict("index")
    top5 = pd.read_parquet(PROC / "projected_top5_by_team.parquet")
    t5s = pd.read_parquet(PROC / "top5_batter_stats.parquet")
    bvp = pd.read_parquet(PROC / "bvp_full_lifetime.parquet")
    bvp_idx = bvp.set_index(["batter", "pitcher"])
    weather = pd.read_parquet(PROC / "game_weather.parquet") \
        if (PROC / "game_weather.parquet").exists() else pd.DataFrame()
    gm = pd.read_parquet(PROC / "game_meta.parquet") \
        if (PROC / "game_meta.parquet").exists() else pd.DataFrame()
    ump_by_pk = gm.set_index("game_pk")["umpire_zone_adj"].to_dict() if len(gm) else {}

    import pickle
    bundle = pickle.load(open(MODELS / "yrfi_model.pkl", "rb"))
    model, feat_cols, medians = bundle["model"], bundle["features"], bundle["medians"]
    surrogate = bundle.get("surrogate")
    med = pd.Series(medians)

    t5_by_team = {t: g.to_dict("records") for t, g in t5s.groupby("team_abbr")}

    def top5_ids(team_abbr: str) -> list[int]:
        r = top5[top5["team_abbr"] == team_abbr]
        if r.empty:
            return []
        r = r.iloc[0]
        return [int(r[f"projected_pos_{p}_player_id"]) for p in range(1, 6)
                if pd.notna(r.get(f"projected_pos_{p}_player_id"))]

    def bvp_agg(batter_ids: list[int], pitcher_id) -> tuple[float, float, list]:
        obps, pas, detail = [], [], []
        for bid in batter_ids:
            try:
                row = bvp_idx.loc[(bid, int(pitcher_id))]
                pa = float(row["pa"])
                pas.append(pa)
                if pa >= 1:
                    obps.append(float(row["obp"]))
                detail.append({"batter": bid, "pa": pa, "obp": float(row["obp"]),
                               "avg": float(row["avg"]), "k_pct": float(row["k_pct"]),
                               "hr": float(row["hr"])})
            except (KeyError, TypeError, ValueError):
                pas.append(0.0)
                detail.append({"batter": bid, "pa": 0.0})
        return (float(np.mean(obps)) if obps else np.nan,
                float(np.mean(pas)) if pas else np.nan, detail)

    rows = []
    for _, grow in slate.iterrows():
        h_ab = id2abbr.get(int(grow["home_team_id"]), str(grow["home_team_id"]))
        a_ab = id2abbr.get(int(grow["away_team_id"]), str(grow["away_team_id"]))
        h_pid = grow.get("home_probable_pitcher_id")
        a_pid = grow.get("away_probable_pitcher_id")
        park = get_park(h_ab)

        away_ids = top5_ids(a_ab)
        home_ids = top5_ids(h_ab)
        away_obp_vp, away_bvp_pa, _ = bvp_agg(away_ids, a_pid and a_pid or h_pid) \
            if False else bvp_agg(away_ids, h_pid)  # away bats vs home pitcher
        home_obp_vp, home_bvp_pa, _ = bvp_agg(home_ids, a_pid)

        # per-game stats_by_team with BvP-vs-today's-pitcher injected
        sbt = {}
        for team, opp_obp, opp_pa in ((a_ab, away_obp_vp, away_bvp_pa),
                                      (h_ab, home_obp_vp, home_bvp_pa)):
            recs = [dict(r) for r in t5_by_team.get(team, [])]
            for r in recs:
                r["_obp_vs_pitcher"] = opp_obp
                r["_bvp_pa"] = opp_pa
            sbt[team] = recs

        gdf = pd.DataFrame([{
            "game_pk": int(grow["game_pk"]),
            "game_date": TODAY,
            "home_team": h_ab, "away_team": a_ab,
            "home_starter_id": h_pid, "away_starter_id": a_pid,
            "is_dome": park["is_dome"],
            "umpire_zone_adj": ump_by_pk.get(int(grow["game_pk"]), 0.0),
        }])
        feat = build_game_features(gdf, prof_map, sbt,
                                   weather if len(weather) else None)
        X = feat[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(med).values
        p_yrfi = float(model.predict_proba(X)[0, 1])
        p_nrfi = 1 - p_yrfi

        contribs = (group_contributions(feat.iloc[0], feat_cols, surrogate)
                    if surrogate else {})

        hp = prof_map.get(int(h_pid), {}) if pd.notna(h_pid) else {}
        ap = prof_map.get(int(a_pid), {}) if pd.notna(a_pid) else {}

        rows.append({
            **contribs,
            "base_model_logit": round(_logit(p_yrfi), 4),
            "game_pk": int(grow["game_pk"]),
            "game_date": TODAY,
            "home_team": h_ab, "away_team": a_ab,
            "home_team_id": int(grow["home_team_id"]),
            "away_team_id": int(grow["away_team_id"]),
            "home_team_record": records.get(int(grow["home_team_id"])),
            "away_team_record": records.get(int(grow["away_team_id"])),
            "game_time": pd.to_datetime(grow.get("game_date")).strftime("%Y-%m-%d")
            if pd.notna(grow.get("game_date")) else TODAY,
            "home_pitcher_id": int(h_pid) if pd.notna(h_pid) else np.nan,
            "home_pitcher_name": grow.get("home_probable_pitcher_name"),
            "home_pitcher_hand": grow.get("home_probable_pitcher_hand"),
            "away_pitcher_id": int(a_pid) if pd.notna(a_pid) else np.nan,
            "away_pitcher_name": grow.get("away_probable_pitcher_name"),
            "away_pitcher_hand": grow.get("away_probable_pitcher_hand"),
            "model_yrfi_prob": round(p_yrfi, 4),
            "model_nrfi_prob": round(p_nrfi, 4),
            "yrfi_model_odds": american_odds(p_yrfi),
            "nrfi_model_odds": american_odds(p_nrfi),
            "home_pitcher_seas_nrfi_pct": hp.get("seas_nrfi_pct"),
            "home_pitcher_l5_nrfi_pct": hp.get("l5_nrfi_pct"),
            "home_pitcher_career_nrfi_pct": hp.get("nrfi_pct"),
            "away_pitcher_seas_nrfi_pct": ap.get("seas_nrfi_pct"),
            "away_pitcher_l5_nrfi_pct": ap.get("l5_nrfi_pct"),
            "away_pitcher_career_nrfi_pct": ap.get("nrfi_pct"),
            "away_top5_avg_obp_vs_pitcher": round(away_obp_vp, 3) if pd.notna(away_obp_vp) else np.nan,
            "home_top5_avg_obp_vs_pitcher": round(home_obp_vp, 3) if pd.notna(home_obp_vp) else np.nan,
            "away_top5_avg_bvp_pa": round(away_bvp_pa, 1) if pd.notna(away_bvp_pa) else np.nan,
            "home_top5_avg_bvp_pa": round(home_bvp_pa, 1) if pd.notna(home_bvp_pa) else np.nan,
            "temp_f": feat["temp_f"].iloc[0],
            "wind_speed": feat["wind_speed"].iloc[0],
            "wind_out_component": feat["wind_out_component"].iloc[0],
            "humidity": feat["humidity"].iloc[0],
            "is_dome": bool(park["is_dome"]),
            "is_indoor": bool(park["is_dome"]),
            "park_name": park["stadium_name"],
            "park_run_factor": park["run_factor"],
            "park_hr_factor": park["hr_factor"],
            "umpire_zone_adj": ump_by_pk.get(int(grow["game_pk"]), 0.0),
        })

    df = pd.DataFrame(rows)

    # ── join first-inning odds if available ───────────────────────────────
    odds_path = ODDS / f"yrfi_odds_{TODAY}.parquet"
    have_odds = odds_path.exists()
    df["odds_checked_at"] = (
        datetime.fromtimestamp(odds_path.stat().st_mtime).isoformat(timespec="minutes")
        if have_odds else None)
    if have_odds:
        od = pd.read_parquet(odds_path)
        od = od.dropna(subset=["game_pk"]) if "game_pk" in od.columns else od
        if len(od):
            od["game_pk"] = od["game_pk"].astype(int)
            df = df.merge(od.drop(columns=["home_team", "away_team"], errors="ignore"),
                          on="game_pk", how="left", suffixes=("", "_odds"))
            print(f"joined first-inning odds ({len(od)} games)", flush=True)
        else:
            have_odds = False
    for c in ["yrfi_market_avg_implied", "nrfi_market_avg_implied",
              "best_yrfi_odds", "best_nrfi_odds", "best_yrfi_book", "best_nrfi_book"]:
        if c not in df.columns:
            df[c] = np.nan

    # ── edges / recommendation ───────────────────────────────────────────
    df["yrfi_edge_pct"] = np.where(
        df["yrfi_market_avg_implied"].notna(),
        (df["model_yrfi_prob"] - df["yrfi_market_avg_implied"]) * 100, np.nan)
    df["nrfi_edge_pct"] = np.where(
        df["nrfi_market_avg_implied"].notna(),
        (df["model_nrfi_prob"] - df["nrfi_market_avg_implied"]) * 100, np.nan)

    def recommend(r):
        if pd.notna(r["yrfi_edge_pct"]) or pd.notna(r["nrfi_edge_pct"]):
            if (r["yrfi_edge_pct"] or -99) > 4:
                return "YRFI", r.get("best_yrfi_odds")
            if (r["nrfi_edge_pct"] or -99) > 4:
                return "NRFI", r.get("best_nrfi_odds")
            return "NO EDGE", np.nan
        # model-only fallback
        if r["model_yrfi_prob"] > 0.60:
            return "YRFI MODEL ONLY", np.nan
        if r["model_nrfi_prob"] > 0.60:
            return "NRFI MODEL ONLY", np.nan
        return "NO EDGE", np.nan

    rec = df.apply(recommend, axis=1, result_type="expand")
    df["recommended_bet"] = rec[0]
    df["recommended_bet_odds"] = rec[1]

    def ev(r):
        side = str(r["recommended_bet"]).split()[0]
        if side not in ("YRFI", "NRFI"):
            return np.nan
        o = r["recommended_bet_odds"]
        p = r["model_yrfi_prob"] if side == "YRFI" else r["model_nrfi_prob"]
        if pd.isna(o):
            return np.nan
        payout = (o / 100) * 100 if o > 0 else (100 / abs(o)) * 100
        return round(p * payout - (1 - p) * 100, 2)

    df["recommended_bet_ev"] = df.apply(ev, axis=1)

    df = df.sort_values("model_nrfi_prob", ascending=False).reset_index(drop=True)
    df.to_parquet(PROC / "todays_yrfi_predictions.parquet", index=False)

    # ── summary ──────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"SAVED todays_yrfi_predictions.parquet  ({len(df)} games)")
    print(f"first-inning market odds: {'JOINED' if have_odds else 'NOT AVAILABLE (model-only)'}")
    print(f"\n{'MATCHUP':<14}{'NRFI%':>7}{'model':>8}  {'AwayP NRFI/L5':>16}  "
          f"{'HomeP NRFI/L5':>16}  REC")
    for _, r in df.iterrows():
        print(f"{r['away_team']}@{r['home_team']:<10}"
              f"{100 * r['model_nrfi_prob']:>6.1f}%{str(r['nrfi_model_odds']):>8}  "
              f"{str(r['away_pitcher_seas_nrfi_pct']):>7}/{str(r['away_pitcher_l5_nrfi_pct']):>7}  "
              f"{str(r['home_pitcher_seas_nrfi_pct']):>7}/{str(r['home_pitcher_l5_nrfi_pct']):>7}  "
              f"{r['recommended_bet']}")
    print(f"\ntotal time: {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
