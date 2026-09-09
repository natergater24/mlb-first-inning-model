#!/usr/bin/env python3
"""
12_build_top_order_batter.py — Projected top-5 lineup batters per team, plus
the batter stats the YRFI model needs for the top of each order.

Part A -> data/processed/projected_top5_by_team.parquet
    For every team, the 5 players most likely to occupy lineup spots 1-5,
    from lineup_position_summary.parquet (2026 season lineup-position starts).
    NOTE: the project has no career lineup-position table, so the 70/30
    season/career weighting collapses to season-only. Confidence % is
    games at that spot / team games played.

Part B -> data/processed/top5_batter_stats.parquet
    One row per projected top-5 batter with:
      - lineup-position splits (spots 1-5): games%, avg, obp, slg
      - last 7 / last 30 day form (batter_rolling.parquet — may be stale;
        refreshed only when src/01_fetch_statcast.py re-runs)
      - full-season line (from lineup_position_summary position sums +
        statcast_full_{yr} for the batted-ball quality stats)
    BvP-vs-today's-pitcher columns are filled later by src/14.

Requires data/processed/lineup_position_summary.parquet — if missing, run:
    python src/02_fetch_mlb_api.py --build-lineup-history
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

LPS_PATH = PROC / "lineup_position_summary.parquet"


def die(msg: str) -> None:
    print(f"\nERROR: {msg}", flush=True)
    sys.exit(1)


# ── player -> team / bat side ──────────────────────────────────────────────
def player_team_map() -> tuple[dict, dict, dict]:
    """(player_id -> team_abbr, player_id -> bat_side, team_abbr -> (id,name))"""
    team_of, side_of, team_meta = {}, {}, {}

    # team id/abbr/name from game_meta
    gm_path = PROC / "game_meta.parquet"
    if gm_path.exists():
        gm = pd.read_parquet(gm_path)
        for _, r in gm.iterrows():
            if pd.notna(r.get("home_team_abbr")):
                team_meta[str(r["home_team_abbr"])] = (int(r["home_team_id"]), None)
            if pd.notna(r.get("away_team_abbr")):
                team_meta[str(r["away_team_abbr"])] = (int(r["away_team_id"]), None)

    # names from probable_pitchers / active_rosters
    for p in (PROC / "probable_pitchers.parquet", PROC / "active_rosters.parquet"):
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        if "home_team_name" in d.columns:
            for _, r in d.iterrows():
                pass  # team id->name handled below via active_rosters

    # statcast_full_2026: batter -> team (most frequent), bat side
    for yr in (CURRENT_SEASON, CURRENT_SEASON - 1):
        f = RAW / f"statcast_full_{yr}.csv"
        if not f.exists():
            continue
        sc = pd.read_csv(f, low_memory=False,
                         usecols=["batter", "home_team", "away_team",
                                  "inning_topbot", "stand"])
        sc["team"] = np.where(sc["inning_topbot"] == "Bot",
                              sc["home_team"], sc["away_team"])
        for bid, g in sc.groupby("batter"):
            bid = int(bid)
            if bid not in team_of:
                team_of[bid] = g["team"].value_counts().index[0]
            if bid not in side_of and g["stand"].notna().any():
                side_of[bid] = g["stand"].value_counts().index[0]
        break

    # active_rosters fills team + name for today's teams (authoritative today)
    ar_path = PROC / "active_rosters.parquet"
    if ar_path.exists():
        ar = pd.read_parquet(ar_path)
        id2abbr = {}
        # map team_id -> abbr via game_meta
        if gm_path.exists():
            gm = pd.read_parquet(gm_path)
            for _, r in gm.iterrows():
                id2abbr[int(r["home_team_id"])] = str(r["home_team_abbr"])
                id2abbr[int(r["away_team_id"])] = str(r["away_team_abbr"])
        for _, r in ar.iterrows():
            abbr = id2abbr.get(int(r["team_id"]))
            if abbr:
                team_of[int(r["player_id"])] = abbr
                team_meta[abbr] = (int(r["team_id"]), r.get("team_name"))
                if pd.notna(r.get("bat_side")):
                    side_of.setdefault(int(r["player_id"]),
                                       str(r["bat_side"])[0].upper())

    return team_of, side_of, team_meta


# ── Part A: projected top 5 ────────────────────────────────────────────────
def project_top5(lps: pd.DataFrame, team_of: dict, team_meta: dict) -> pd.DataFrame:
    lps = lps.copy()
    lps["team"] = lps["player_id"].map(team_of)
    lps = lps.dropna(subset=["team"])

    rows = []
    for team, g in lps.groupby("team"):
        team_games = g["total_games_played"].max() or 1
        rec = {"team_abbr": team,
               "team_id": team_meta.get(team, (np.nan, None))[0],
               "team_name": team_meta.get(team, (np.nan, team))[1] or team}
        used: set[int] = set()
        for pos in range(1, 6):
            col = f"games_pos_{pos}"
            cand = g[~g["player_id"].isin(used)].copy()
            cand["_n"] = pd.to_numeric(cand.get(col), errors="coerce").fillna(0)
            cand = cand.sort_values("_n", ascending=False)
            if len(cand) and cand.iloc[0]["_n"] > 0:
                top = cand.iloc[0]
                used.add(int(top["player_id"]))
                rec[f"projected_pos_{pos}_player_id"] = int(top["player_id"])
                rec[f"projected_pos_{pos}_player_name"] = top["player_name"]
                rec[f"projected_pos_{pos}_confidence_pct"] = round(
                    100 * top["_n"] / team_games, 1)
            else:
                rec[f"projected_pos_{pos}_player_id"] = np.nan
                rec[f"projected_pos_{pos}_player_name"] = None
                rec[f"projected_pos_{pos}_confidence_pct"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("team_abbr").reset_index(drop=True)


# ── Part B: batter stats ──────────────────────────────────────────────────
def season_line_from_lps(row: pd.Series) -> dict:
    g = pa = ab = h = hr = bb = k = rbi = r = 0
    for pos in range(1, 10):
        g += _num(row.get(f"games_pos_{pos}"))
        pa += _num(row.get(f"pa_pos_{pos}"))
        ab += _num(row.get(f"ab_pos_{pos}"))
        h += _num(row.get(f"h_pos_{pos}"))
        hr += _num(row.get(f"hr_pos_{pos}"))
        bb += _num(row.get(f"bb_pos_{pos}"))
        k += _num(row.get(f"k_pos_{pos}"))
        rbi += _num(row.get(f"rbi_pos_{pos}"))
    avg = h / ab if ab else np.nan
    obp = (h + bb) / (ab + bb) if (ab + bb) else np.nan
    return {"seas_g": int(g), "seas_pa": int(pa), "seas_ab": int(ab),
            "seas_h": int(h), "seas_hr": int(hr), "seas_bb": int(bb),
            "seas_k": int(k), "seas_rbi": int(rbi),
            "seas_avg": round(avg, 3) if pd.notna(avg) else np.nan,
            "seas_obp": round(obp, 3) if pd.notna(obp) else np.nan,
            "seas_k_pct": round(k / pa, 3) if pa else np.nan,
            "seas_bb_pct": round(bb / pa, 3) if pa else np.nan}


def _num(v) -> float:
    try:
        f = float(v)
        return 0.0 if np.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0


def statcast_quality(ids: set[int]) -> pd.DataFrame:
    """2B/3B/SLG/HR + batted-ball quality from statcast_full_{yr} (partial season)."""
    f = RAW / f"statcast_full_{CURRENT_SEASON}.csv"
    if not f.exists():
        return pd.DataFrame(columns=["batter"])
    sc = pd.read_csv(f, low_memory=False,
                     usecols=["batter", "events", "launch_speed", "type",
                              "estimated_ba_using_speedangle",
                              "estimated_woba_using_speedangle",
                              "launch_speed_angle"])
    sc = sc[sc["batter"].isin(ids)]
    rows = []
    for bid, g in sc.groupby("batter"):
        ev = g["events"].dropna()
        ab_like = ev[~ev.isin(["walk", "intent_walk", "hit_by_pitch", "sac_fly",
                               "sac_bunt", "catcher_interf"])]
        singles = (ev == "single").sum()
        doubles = (ev == "double").sum()
        triples = (ev == "triple").sum()
        hr = (ev == "home_run").sum()
        hits = singles + doubles + triples + hr
        tb = singles + 2 * doubles + 3 * triples + 4 * hr
        bip = g[g["launch_speed"].notna()]
        rows.append({
            "batter": int(bid),
            "sc_2b": int(doubles), "sc_3b": int(triples),
            "sc_slg": round(tb / len(ab_like), 3) if len(ab_like) else np.nan,
            "sc_xba": round(g["estimated_ba_using_speedangle"].mean(), 3),
            "sc_woba": round(g["estimated_woba_using_speedangle"].mean(), 3),
            "sc_hard_hit": round((bip["launch_speed"] >= 95).mean(), 3) if len(bip) else np.nan,
            "sc_barrel_pct": round((bip["launch_speed_angle"] == 6).mean(), 3) if len(bip) else np.nan,
        })
    return pd.DataFrame(rows)


def build_batter_stats(top5: pd.DataFrame, lps: pd.DataFrame,
                       side_of: dict) -> pd.DataFrame:
    # collect all projected ids with their team + slot
    recs = []
    for _, tr in top5.iterrows():
        for pos in range(1, 6):
            pid = tr.get(f"projected_pos_{pos}_player_id")
            if pd.isna(pid):
                continue
            recs.append({"player_id": int(pid),
                         "player_name": tr.get(f"projected_pos_{pos}_player_name"),
                         "team_abbr": tr["team_abbr"],
                         "team_id": tr["team_id"],
                         "projected_slot": pos,
                         "slot_confidence_pct": tr.get(f"projected_pos_{pos}_confidence_pct")})
    base = pd.DataFrame(recs)
    ids = set(base["player_id"])

    lps_i = lps.set_index("player_id")
    rolling = pd.read_parquet(PROC / "batter_rolling.parquet")
    rolling["game_date"] = pd.to_datetime(rolling["game_date"], errors="coerce")
    last_roll = (rolling.sort_values("game_date").groupby("batter").tail(1)
                 .set_index("batter"))
    roll_asof = rolling["game_date"].max()
    quality = statcast_quality(ids).set_index("batter") if ids else pd.DataFrame()

    # boxscore-sourced game logs -> real L7/L30/season OBP/OPS/R/AB windows
    glp = PROC / "batter_game_logs.parquet"
    game_logs = pd.read_parquet(glp) if glp.exists() else pd.DataFrame()
    if len(game_logs):
        game_logs["game_date"] = pd.to_datetime(game_logs["game_date"], errors="coerce")
        gl_asof = game_logs["game_date"].max()
    else:
        gl_asof = pd.NaT

    def _window(pid, days):
        if not len(game_logs) or pd.isna(gl_asof):
            return {}
        cut = gl_asof - pd.Timedelta(days=days)
        g = game_logs[(game_logs["player_id"] == pid) & (game_logs["game_date"] >= cut)]
        if g.empty:
            return {}
        pa = int(g["pa"].sum()); ab = int(g["ab"].sum())
        h = int(g["hits"].sum()); bb = int(g["bb"].sum())
        tb = int(g["tb"].sum()); k = int(g["ks"].sum())
        obp = (h + bb) / (ab + bb) if (ab + bb) else np.nan
        slg = tb / ab if ab else np.nan
        return {"pa": pa, "ab": ab, "r": int(g["runs"].sum()), "h": h,
                "hr": int(g["hr"].sum()), "rbi": int(g["rbi"].sum()),
                "avg": round(h / ab, 3) if ab else np.nan,
                "obp": round(obp, 3) if pd.notna(obp) else np.nan,
                "slg": round(slg, 3) if pd.notna(slg) else np.nan,
                "ops": round(obp + slg, 3) if pd.notna(obp) and pd.notna(slg) else np.nan,
                "k_pct": round(k / pa, 3) if pa else np.nan,
                "g": len(g)}

    out = []
    for _, b in base.iterrows():
        pid = b["player_id"]
        rec = b.to_dict()
        rec["bat_side"] = side_of.get(pid)
        lr = lps_i.loc[pid] if pid in lps_i.index else None

        # lineup position splits 1-5
        if lr is not None:
            team_games = _num(lr.get("total_games_played")) or 1
            for pos in range(1, 6):
                gp = _num(lr.get(f"games_pos_{pos}"))
                rec[f"pos{pos}_games_pct"] = round(100 * gp / team_games, 1)
                rec[f"pos{pos}_avg"] = _r(lr.get(f"avg_pos_{pos}"))
                rec[f"pos{pos}_obp"] = _r(lr.get(f"obp_pos_{pos}"))
                rec[f"pos{pos}_slg"] = _r(lr.get(f"slg_pos_{pos}"))
            rec.update(season_line_from_lps(lr))
        else:
            for pos in range(1, 6):
                for s in ("games_pct", "avg", "obp", "slg"):
                    rec[f"pos{pos}_{s}"] = np.nan
            for c in ("seas_g", "seas_pa", "seas_ab", "seas_h", "seas_hr", "seas_bb",
                      "seas_k", "seas_rbi", "seas_avg", "seas_obp", "seas_k_pct",
                      "seas_bb_pct"):
                rec[c] = np.nan

        # rolling form — statcast-derived rate detail (may be stale)
        if pid in last_roll.index:
            r = last_roll.loc[pid]
            rec["l7_hard_hit"] = _r(r.get("hard_hit_7d")); rec["l7_xba"] = _r(r.get("xba_7d"))
            rec["l7_bb_pct"] = _r(r.get("bb_pct_7d"))
            rec["l30_hard_hit"] = _r(r.get("hard_hit_30d")); rec["l30_xba"] = _r(r.get("xba_30d"))
            rec["l30_bb_pct"] = _r(r.get("bb_pct_30d"))
        else:
            for c in ("l7_hard_hit", "l7_xba", "l7_bb_pct",
                      "l30_hard_hit", "l30_xba", "l30_bb_pct"):
                rec[c] = np.nan

        # L7 / L30 windows from boxscore game logs — OBP / OPS / R / AB / etc
        for win, days in (("l7", 7), ("l30", 30)):
            w = _window(pid, days)
            rec[f"{win}_g"] = w.get("g", 0)
            rec[f"{win}_pa"] = w.get("pa", np.nan)
            rec[f"{win}_ab"] = w.get("ab", np.nan)
            rec[f"{win}_r"] = w.get("r", np.nan)
            rec[f"{win}_hr"] = w.get("hr", np.nan)
            rec[f"{win}_rbi"] = w.get("rbi", np.nan)
            rec[f"{win}_avg"] = w.get("avg", np.nan)
            rec[f"{win}_obp"] = w.get("obp", np.nan)
            rec[f"{win}_slg"] = w.get("slg", np.nan)
            rec[f"{win}_ops"] = w.get("ops", np.nan)
            rec[f"{win}_k_pct"] = w.get("k_pct", np.nan)
        rec["game_logs_asof"] = gl_asof

        # statcast quality
        if not quality.empty and pid in quality.index:
            q = quality.loc[pid]
            rec["seas_2b"] = int(q["sc_2b"]); rec["seas_3b"] = int(q["sc_3b"])
            rec["seas_slg"] = _r(q["sc_slg"]); rec["seas_xba"] = _r(q["sc_xba"])
            rec["seas_woba"] = _r(q["sc_woba"]); rec["seas_hard_hit"] = _r(q["sc_hard_hit"])
            rec["seas_barrel_pct"] = _r(q["sc_barrel_pct"])
        else:
            for c in ("seas_2b", "seas_3b", "seas_slg", "seas_xba", "seas_woba",
                      "seas_hard_hit", "seas_barrel_pct"):
                rec[c] = np.nan
        rec["seas_sb"] = np.nan  # not in available data

        # season totals from the full game log (complete boxscore data): runs,
        # and an accurate AB / SLG / OPS (the statcast_full slice is partial)
        sw = _window(pid, 3650)
        if sw:
            rec["seas_r"] = sw["r"]
            rec["seas_ab"] = sw["ab"] or rec.get("seas_ab")
            if pd.notna(sw.get("slg")):
                rec["seas_slg"] = sw["slg"]
            if pd.notna(sw.get("ops")):
                rec["seas_ops"] = sw["ops"]
        else:
            rec["seas_r"] = np.nan

        if pd.isna(rec.get("seas_ops")) and pd.notna(rec.get("seas_obp")) \
                and pd.notna(rec.get("seas_slg")):
            rec["seas_ops"] = round(rec["seas_obp"] + rec["seas_slg"], 3)
        rec["seas_hr_per_pa"] = round(rec["seas_hr"] / rec["seas_pa"], 4) \
            if rec.get("seas_pa") else np.nan
        rec["rolling_asof"] = roll_asof
        out.append(rec)

    return pd.DataFrame(out)


def _r(v, nd=3):
    try:
        f = float(v)
        return round(f, nd) if not np.isnan(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def main() -> int:
    t0 = time.time()
    if not LPS_PATH.exists():
        die(f"{LPS_PATH.name} not found. Run:\n"
            f"    /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 "
            f"src/02_fetch_mlb_api.py --build-lineup-history")

    lps = pd.read_parquet(LPS_PATH)
    print(f"lineup_position_summary: {len(lps)} players", flush=True)
    team_of, side_of, team_meta = player_team_map()
    print(f"player->team resolved for {len(team_of)} players, "
          f"{len(set(team_of.values()))} teams", flush=True)

    top5 = project_top5(lps, team_of, team_meta)
    top5.to_parquet(PROC / "projected_top5_by_team.parquet", index=False)
    print(f"SAVED projected_top5_by_team.parquet  ({len(top5)} teams)", flush=True)

    stats = build_batter_stats(top5, lps, side_of)
    stats.to_parquet(PROC / "top5_batter_stats.parquet", index=False)
    print(f"SAVED top5_batter_stats.parquet  ({len(stats)} batters, "
          f"{stats.shape[1]} cols)", flush=True)

    # ── summary ───────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"teams with a full projected 1-5: "
          f"{(top5[[f'projected_pos_{p}_player_id' for p in range(1,6)]].notna().all(axis=1)).sum()}/{len(top5)}")
    if len(stats):
        print(f"batter rolling form as-of date: {stats['rolling_asof'].iloc[0]}  "
              f"(stale if far from today — refresh via src/01)")
        print(f"season line populated: {stats['seas_pa'].notna().sum()}/{len(stats)}")
        print(f"statcast quality populated: {stats['seas_slg'].notna().sum()}/{len(stats)}")
    ex = top5.iloc[0]
    print(f"\nexample — {ex['team_abbr']}:")
    for p in range(1, 6):
        print(f"  #{p} {ex.get(f'projected_pos_{p}_player_name')}  "
              f"({ex.get(f'projected_pos_{p}_confidence_pct')}% of games)")
    print(f"\ntotal time: {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
