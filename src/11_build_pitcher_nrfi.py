#!/usr/bin/env python3
"""
11_build_pitcher_nrfi.py — First-inning ("NRFI") profile for every pitcher.

Two data sources:
  1. data/processed/yrfi_outcomes.parquet — one start-level NRFI/YRFI label per
     pitcher per game (from the official linescore). Complete 2015-2026 coverage.
     Drives: start counts, NRFI%/YRFI% (career / season / last-5 / home / away),
     avg first-inning runs allowed, first-inning ERA.
  2. data/raw/statcast_inn1_{yr}.csv  (preferred)  ->  statcast_full_{yr}.csv
     (inning 1 slice)  ->  statcast_{yr}.csv (leadoff only).
     Drives the pitch-level rate stats: OBP/WHIP allowed, K%, BB%, hard-hit%,
     velocity, first-pitch-strike rate, pitch-type usage & whiff.

  Until the overnight statcast_inn1 download finishes, the pitch-level stats are
  based on partial coverage (statcast_full is ~40% of games); re-run afterward.

Outputs:
  data/processed/pitcher_nrfi_profile.parquet   (one row per pitcher)
  data/processed/pitcher_last5_starts.parquet   (one row per pitcher per start,
                                                 last 5 starts each)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

CURRENT_SEASON = 2026

FB = {"FF", "FA", "SI", "FT", "FC"}
SL = {"SL", "ST", "SV"}
CB = {"CU", "KC", "CS", "KN"}
CH = {"CH", "FS", "FO"}
BUCKETS = {"fb": FB, "sl": SL, "cb": CB, "ch": CH}

HIT_EVENTS = {"single", "double", "triple", "home_run"}
BB_EVENTS = {"walk", "intent_walk"}
HBP_EVENTS = {"hit_by_pitch"}
SWING_DESC = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
              "hit_into_play", "hit_into_play_score", "hit_into_play_no_out",
              "missed_bunt", "foul_bunt"}
WHIFF_DESC = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}


# ── start-level NRFI table from linescore outcomes ──────────────────────────
def build_starts() -> pd.DataFrame:
    y = pd.read_parquet(PROC / "yrfi_outcomes.parquet")
    home = y[["game_pk", "game_date", "game_year", "home_starter_id",
              "home_starter_name", "away_team", "first_inn_runs_away"]].rename(
        columns={"home_starter_id": "pitcher_id", "home_starter_name": "pitcher_name",
                 "away_team": "opponent", "first_inn_runs_away": "runs_allowed"})
    home["is_home"] = True
    away = y[["game_pk", "game_date", "game_year", "away_starter_id",
              "away_starter_name", "home_team", "first_inn_runs_home"]].rename(
        columns={"away_starter_id": "pitcher_id", "away_starter_name": "pitcher_name",
                 "home_team": "opponent", "first_inn_runs_home": "runs_allowed"})
    away["is_home"] = False
    s = pd.concat([home, away], ignore_index=True)
    s = s.dropna(subset=["pitcher_id"]).copy()
    s["pitcher_id"] = s["pitcher_id"].astype("int64")
    s["runs_allowed"] = s["runs_allowed"].fillna(0).astype(int)
    s["nrfi"] = s["runs_allowed"] == 0
    s["game_date"] = pd.to_datetime(s["game_date"], errors="coerce")
    return s.sort_values(["pitcher_id", "game_date"])


def _pct(mask_sum: float, n: float) -> float:
    return round(100 * mask_sum / n, 1) if n else np.nan


def agg_nrfi(starts: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for pid, g in starts.groupby("pitcher_id"):
        g = g.sort_values("game_date")
        name = g["pitcher_name"].dropna().iloc[-1] if g["pitcher_name"].notna().any() else None
        n = len(g)
        nrfi_n = int(g["nrfi"].sum())
        seas = g[g["game_year"] == CURRENT_SEASON]
        l5 = g.tail(5)
        home = g[g["is_home"]]
        away = g[~g["is_home"]]
        recs.append({
            "pitcher_id": pid,
            "pitcher_name": name,
            "total_starts": n,
            "nrfi_count": nrfi_n,
            "yrfi_count": n - nrfi_n,
            "nrfi_pct": _pct(nrfi_n, n),
            "yrfi_pct": _pct(n - nrfi_n, n),
            "avg_first_inn_runs_allowed": round(g["runs_allowed"].mean(), 3),
            "first_inn_era": round(g["runs_allowed"].mean() * 9, 2),
            "seas_starts": len(seas),
            "seas_nrfi_pct": _pct(seas["nrfi"].sum(), len(seas)),
            "seas_yrfi_pct": _pct((~seas["nrfi"]).sum(), len(seas)),
            "seas_first_inn_era": round(seas["runs_allowed"].mean() * 9, 2) if len(seas) else np.nan,
            "l5_starts": len(l5),
            "l5_nrfi_pct": _pct(l5["nrfi"].sum(), len(l5)),
            "l5_yrfi_pct": _pct((~l5["nrfi"]).sum(), len(l5)),
            "l5_first_inn_era": round(l5["runs_allowed"].mean() * 9, 2) if len(l5) else np.nan,
            "home_starts": len(home),
            "away_starts": len(away),
            "home_nrfi_pct": _pct(home["nrfi"].sum(), len(home)),
            "away_nrfi_pct": _pct(away["nrfi"].sum(), len(away)),
            "home_first_inn_era": round(home["runs_allowed"].mean() * 9, 2) if len(home) else np.nan,
            "away_first_inn_era": round(away["runs_allowed"].mean() * 9, 2) if len(away) else np.nan,
        })
    return pd.DataFrame(recs)


# ── statcast inning-1 pitch metrics ────────────────────────────────────────
def load_inn1_pitches() -> pd.DataFrame:
    frames = []
    src_used = []
    for yr in range(2015, CURRENT_SEASON + 1):
        for name in (f"statcast_inn1_{yr}.csv", f"statcast_full_{yr}.csv", f"statcast_{yr}.csv"):
            p = RAW / name
            if not p.exists():
                continue
            cols = ["game_pk", "game_date", "game_year", "inning", "pitcher",
                    "batter", "at_bat_number", "pitch_number", "pitch_type",
                    "release_speed", "events", "description", "type",
                    "launch_speed", "game_type", "estimated_woba_using_speedangle"]
            df = pd.read_csv(p, low_memory=False,
                             usecols=lambda c: c in set(cols))
            if "inning" in df.columns:
                df = df[df["inning"] == 1]
            if "game_type" in df.columns:
                df = df[df["game_type"] == "R"]
            if len(df):
                df["_src"] = name
                frames.append(df)
                src_used.append(f"{name} ({df['game_pk'].nunique()}g)")
            break
    if src_used:
        print("  inning-1 pitch sources:", ", ".join(src_used), flush=True)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out["release_speed"] = pd.to_numeric(out["release_speed"], errors="coerce")
    out["launch_speed"] = pd.to_numeric(out["launch_speed"], errors="coerce")
    out["pitch_number"] = pd.to_numeric(out["pitch_number"], errors="coerce")
    return out


def bucket_of(pt: str) -> str | None:
    for b, s in BUCKETS.items():
        if pt in s:
            return b
    return None


def pitch_metrics(pitches: pd.DataFrame, tag: str = "") -> pd.DataFrame:
    """Per-pitcher inning-1 rate metrics from pitch-level data."""
    if pitches.empty:
        return pd.DataFrame(columns=["pitcher_id"])
    p = pitches.dropna(subset=["pitcher"]).copy()
    p["pitcher_id"] = p["pitcher"].astype("int64")

    # PA-level terminal rows
    pa = (p[p["events"].notna() & (p["events"] != "")]
          .sort_values(["game_pk", "at_bat_number", "pitch_number"])
          .groupby(["game_pk", "at_bat_number"], as_index=False).last())
    pa_grp = pa.groupby("pitcher_id")
    n_pa = pa_grp.size().rename("_pa")
    k = pa_grp["events"].apply(lambda s: (s == "strikeout").sum()).rename("_k")
    bb = pa_grp["events"].apply(lambda s: s.isin(BB_EVENTS).sum()).rename("_bb")
    hbp = pa_grp["events"].apply(lambda s: s.isin(HBP_EVENTS).sum()).rename("_hbp")
    hits = pa_grp["events"].apply(lambda s: s.isin(HIT_EVENTS).sum()).rename("_h")

    m = pd.concat([n_pa, k, bb, hbp, hits], axis=1).reset_index()
    m[f"{tag}first_inn_k_rate" if tag else "first_inn_k_rate"] = (m["_k"] / m["_pa"]).round(3)
    m[f"{tag}first_inn_bb_rate" if tag else "first_inn_bb_rate"] = (m["_bb"] / m["_pa"]).round(3)
    m[f"{tag}first_inn_obp_allowed" if tag else "first_inn_obp_allowed"] = (
        (m["_h"] + m["_bb"] + m["_hbp"]) / m["_pa"]).round(3)
    m[f"{tag}first_inn_whip" if tag else "first_inn_whip"] = (
        (m["_h"] + m["_bb"])).astype(float)  # placeholder, normalized below

    # WHIP per start ~= (H+BB)/starts (≈1 IP/start in the 1st)
    starts_per = pa_grp["game_pk"].nunique().rename("_starts")
    m = m.merge(starts_per.reset_index(), on="pitcher_id", how="left")
    whip_col = f"{tag}first_inn_whip" if tag else "first_inn_whip"
    m[whip_col] = ((m["_h"] + m["_bb"]) / m["_starts"]).round(2)

    # hard-hit% allowed (batted balls with EV >= 95)
    bip = p[p["launch_speed"].notna()]
    hh = bip.groupby(p["pitcher"].astype("int64")).apply(
        lambda d: (d["launch_speed"] >= 95).mean()).rename("_hh")
    hh.index.name = "pitcher_id"
    m = m.merge((hh * 1).round(3).rename(
        f"{tag}first_inn_hard_hit_allowed" if tag else "first_inn_hard_hit_allowed"
    ).reset_index(), on="pitcher_id", how="left")

    # avg velo (all inning-1 pitches)
    velo = p.groupby("pitcher_id")["release_speed"].mean().round(1).rename(
        f"{tag}first_inn_avg_velo" if tag else "first_inn_avg_velo")
    m = m.merge(velo.reset_index(), on="pitcher_id", how="left")

    # first-pitch strike rate
    fp = p[p["pitch_number"] == 1]
    fps = fp.groupby("pitcher_id")["type"].apply(
        lambda s: (s != "B").mean()).round(3).rename(
        f"{tag}first_pitch_strike_rate" if tag else "first_pitch_strike_rate")
    m = m.merge(fps.reset_index(), on="pitcher_id", how="left")

    return m.drop(columns=["_pa", "_k", "_bb", "_hbp", "_h", "_starts"], errors="ignore")


def pitch_type_breakdown(pitches: pd.DataFrame) -> pd.DataFrame:
    if pitches.empty:
        return pd.DataFrame(columns=["pitcher_id"])
    p = pitches.dropna(subset=["pitcher", "pitch_type"]).copy()
    p["pitcher_id"] = p["pitcher"].astype("int64")
    p["bucket"] = p["pitch_type"].map(bucket_of)
    p = p.dropna(subset=["bucket"])
    p["_swing"] = p["description"].isin(SWING_DESC)
    p["_whiff"] = p["description"].isin(WHIFF_DESC)

    recs = []
    for pid, g in p.groupby("pitcher_id"):
        tot = len(g)
        rec = {"pitcher_id": pid}
        best_b, best_u = None, -1.0
        for b in BUCKETS:
            gb = g[g["bucket"] == b]
            usage = len(gb) / tot if tot else 0.0
            sw = gb["_swing"].sum()
            whiff = gb["_whiff"].sum() / sw if sw else np.nan
            rec[f"first_inn_{b}_usage"] = round(usage, 3)
            rec[f"first_inn_{b}_whiff"] = round(whiff, 3) if pd.notna(whiff) else np.nan
            if usage > best_u:
                best_u, best_b = usage, b
        rec["primary_pitch"] = best_b
        rec["primary_pitch_usage_pct"] = round(100 * best_u, 1)
        recs.append(rec)
    return pd.DataFrame(recs)


# ── last-5-starts detail ───────────────────────────────────────────────────
def last5_detail(starts: pd.DataFrame, pitches: pd.DataFrame) -> pd.DataFrame:
    velo_by_game = pd.DataFrame()
    if not pitches.empty:
        pp = pitches.dropna(subset=["pitcher"]).copy()
        pp["pitcher_id"] = pp["pitcher"].astype("int64")
        velo_by_game = (pp.groupby(["pitcher_id", "game_pk"])
                        .agg(velo_on_day=("release_speed", "mean"),
                             inning_1_pitches=("pitch_number", "count"))
                        .reset_index())

    rows = []
    for pid, g in starts.groupby("pitcher_id"):
        g = g.sort_values("game_date").tail(5)
        for _, r in g.iterrows():
            rows.append({
                "pitcher_id": pid,
                "pitcher_name": r["pitcher_name"],
                "start_date": r["game_date"],
                "game_pk": r["game_pk"],
                "opponent": r["opponent"],
                "is_home": r["is_home"],
                "inning_1_runs_allowed": int(r["runs_allowed"]),
                "result": "NRFI" if r["nrfi"] else "YRFI",
                "nrfi": bool(r["nrfi"]),
                "era_on_day": round(r["runs_allowed"] * 9, 2),
            })
    d = pd.DataFrame(rows)
    if not velo_by_game.empty and not d.empty:
        d = d.merge(velo_by_game, on=["pitcher_id", "game_pk"], how="left")
        d["velo_on_day"] = d["velo_on_day"].round(1)
    else:
        d["velo_on_day"] = np.nan
        d["inning_1_pitches"] = np.nan
    return d


def main() -> int:
    t0 = time.time()
    print("building start-level NRFI table from yrfi_outcomes...", flush=True)
    starts = build_starts()
    print(f"  {len(starts):,} pitcher-starts, {starts['pitcher_id'].nunique():,} pitchers",
          flush=True)

    prof = agg_nrfi(starts)

    print("loading inning-1 statcast pitches...", flush=True)
    pitches = load_inn1_pitches()
    seas_pitches = pitches[pitches["game_year"] == CURRENT_SEASON] if not pitches.empty else pitches
    l5_pks = set()
    for _, g in starts.groupby("pitcher_id"):
        l5_pks.update(g.sort_values("game_date").tail(5)["game_pk"].tolist())
    l5_pitches = pitches[pitches["game_pk"].isin(l5_pks)] if not pitches.empty else pitches

    print("computing pitch-level metrics (career / season / L5)...", flush=True)
    career_m = pitch_metrics(pitches, tag="")
    seas_m = pitch_metrics(seas_pitches, tag="seas_")
    l5_m = pitch_metrics(l5_pitches, tag="l5_")
    types = pitch_type_breakdown(pitches)

    for extra in (career_m, types):
        if not extra.empty:
            prof = prof.merge(extra, on="pitcher_id", how="left")
    # only the season/L5 subsets the spec asks for
    if not seas_m.empty:
        keep = ["pitcher_id", "seas_first_inn_obp_allowed", "seas_first_inn_k_rate",
                "seas_first_inn_bb_rate"]
        prof = prof.merge(seas_m[[c for c in keep if c in seas_m.columns]],
                          on="pitcher_id", how="left")
    if not l5_m.empty:
        keep = ["pitcher_id", "l5_first_inn_obp_allowed", "l5_first_inn_avg_velo"]
        l5_m = l5_m.rename(columns={"l5_first_inn_avg_velo": "l5_avg_velo"})
        keep = ["pitcher_id", "l5_first_inn_obp_allowed", "l5_avg_velo"]
        prof = prof.merge(l5_m[[c for c in keep if c in l5_m.columns]],
                          on="pitcher_id", how="left")

    # ensure all spec columns exist even if statcast coverage is thin
    for col in ["first_inn_obp_allowed", "first_inn_whip", "first_inn_k_rate",
                "first_inn_bb_rate", "first_inn_hard_hit_allowed", "first_inn_avg_velo",
                "first_pitch_strike_rate", "seas_first_inn_obp_allowed",
                "seas_first_inn_k_rate", "seas_first_inn_bb_rate",
                "l5_first_inn_obp_allowed", "l5_avg_velo",
                "first_inn_fb_usage", "first_inn_fb_whiff", "first_inn_sl_usage",
                "first_inn_sl_whiff", "first_inn_cb_usage", "first_inn_cb_whiff",
                "first_inn_ch_usage", "first_inn_ch_whiff", "primary_pitch",
                "primary_pitch_usage_pct"]:
        if col not in prof.columns:
            prof[col] = np.nan

    prof = prof.sort_values("total_starts", ascending=False).reset_index(drop=True)
    prof.to_parquet(PROC / "pitcher_nrfi_profile.parquet", index=False)

    l5d = last5_detail(starts, pitches)
    l5d.to_parquet(PROC / "pitcher_last5_starts.parquet", index=False)

    # ── summary ───────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"SAVED pitcher_nrfi_profile.parquet  ({len(prof):,} pitchers, "
          f"{prof.shape[1]} cols)")
    print(f"SAVED pitcher_last5_starts.parquet  ({len(l5d):,} rows)")
    qualified = prof[prof["total_starts"] >= 20]
    print(f"\npitchers with >=20 career starts: {len(qualified)}")
    print(f"  mean career NRFI%: {qualified['nrfi_pct'].mean():.1f}%")
    print(f"  statcast pitch metrics populated: "
          f"{prof['first_inn_k_rate'].notna().sum()}/{len(prof)} pitchers")
    print("\ntop 10 NRFI pitchers (>=40 starts):")
    top = prof[prof["total_starts"] >= 40].nlargest(10, "nrfi_pct")
    for _, r in top.iterrows():
        print(f"  {str(r['pitcher_name'] or r['pitcher_id'])[:24]:24} "
              f"{r['nrfi_pct']:.1f}%  ({r['total_starts']} GS, "
              f"1st-inn ERA {r['first_inn_era']:.2f}, primary {r['primary_pitch']})")
    print("\nworst 10 NRFI pitchers (>=40 starts):")
    for _, r in prof[prof["total_starts"] >= 40].nsmallest(10, "nrfi_pct").iterrows():
        print(f"  {str(r['pitcher_name'] or r['pitcher_id'])[:24]:24} "
              f"{r['nrfi_pct']:.1f}%  ({r['total_starts']} GS)")
    print(f"\ntotal time: {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
