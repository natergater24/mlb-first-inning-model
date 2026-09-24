"""
matchup_highlights.py — pure, plain-importable logic for the landing page's
"Notable batter-vs-pitcher matchups" callout (added 2026-09-24). Separated
from app.py (which calls st.set_page_config() at import time and can't be
imported outside a real `streamlit run`) so this stays unit-testable, same
reasoning as bet_tracker.py / yrfi_features.py.
"""
from __future__ import annotations

import pandas as pd

BVP_MIN_PA = 15               # matches src/01_fetch_statcast.py's own bvp_reliable threshold
BVP_NOTABLE_OBP_HIGH = 0.400  # matches app.py's _WHY_EDGE_OBP_HIGH
BVP_NOTABLE_OBP_LOW = 0.220   # matches app.py's _WHY_EDGE_OBP_LOW


def compute_notable_bvp_matchups(pred: pd.DataFrame, top5: pd.DataFrame,
                                 bvp: pd.DataFrame, n: int = 3) -> list[dict]:
    """Today's most significant individual batter-vs-pitcher matchups: a
    projected top-5 hitter with a real (>=15 PA) history against today's
    actual opposing starter, hitting well above or below league norms
    against him specifically. Ranked by PA (more history = more trustworthy),
    ties broken by how extreme the OBP is."""
    if pred.empty or top5.empty or bvp.empty:
        return []
    bvp_idx = bvp.set_index(["batter", "pitcher"])
    found = []
    for _, g in pred.iterrows():
        for side, team, opp_pid, opp_name in (
            ("away", g.get("away_team"), g.get("home_pitcher_id"), g.get("home_pitcher_name")),
            ("home", g.get("home_team"), g.get("away_pitcher_id"), g.get("away_pitcher_name")),
        ):
            if pd.isna(opp_pid) or not opp_name:
                continue
            trow = top5[top5["team_abbr"] == team]
            if trow.empty:
                continue
            tr = trow.iloc[0]
            for i in range(1, 6):
                pid = tr.get(f"projected_pos_{i}_player_id")
                pname = tr.get(f"projected_pos_{i}_player_name")
                if pd.isna(pid) or not pname:
                    continue
                try:
                    row = bvp_idx.loc[(int(pid), int(opp_pid))]
                except KeyError:
                    continue
                pa, obp = float(row["pa"]), float(row["obp"])
                if pa < BVP_MIN_PA or (BVP_NOTABLE_OBP_LOW < obp < BVP_NOTABLE_OBP_HIGH):
                    continue
                found.append({
                    "batter_name": pname, "pitcher_name": opp_name, "team": team,
                    "game_pk": int(g["game_pk"]), "pa": int(pa), "obp": obp,
                    "avg": float(row.get("avg", float("nan"))),
                    "hr": int(row.get("hr") or 0),
                    "hot": obp >= BVP_NOTABLE_OBP_HIGH,
                })
    found.sort(key=lambda m: (m["pa"], abs(m["obp"] - 0.310)), reverse=True)
    return found[:n]
