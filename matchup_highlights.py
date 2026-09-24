"""
matchup_highlights.py — pure, plain-importable logic for the "Notable
batter-vs-pitcher matchups" shown on each individual game card (added
2026-09-24, moved from a single top-of-page callout to per-game on the same
day per user feedback). Separated from app.py (which calls
st.set_page_config() at import time and can't be imported outside a real
`streamlit run`) so this stays unit-testable, same reasoning as
bet_tracker.py / yrfi_features.py.
"""
from __future__ import annotations

import pandas as pd

BVP_MIN_PA = 15               # matches src/01_fetch_statcast.py's own bvp_reliable threshold
BVP_NOTABLE_OBP_HIGH = 0.400  # matches app.py's _WHY_EDGE_OBP_HIGH
BVP_NOTABLE_OBP_LOW = 0.220   # matches app.py's _WHY_EDGE_OBP_LOW


def _game_bvp_candidates(g: pd.Series, top5: pd.DataFrame, bvp_idx: pd.DataFrame) -> list[dict]:
    """Every qualifying (>=15 PA, OBP outside league-normal range) batter-vs-
    pitcher pairing for one game, both sides: away top-5 vs the home
    starter, home top-5 vs the away starter."""
    found = []
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
    return found


def notable_bvp_for_game(game: pd.Series, top5: pd.DataFrame, bvp: pd.DataFrame,
                         n: int = 2) -> list[dict]:
    """Notable batter-vs-pitcher matchups for ONE game (both starters vs the
    opposing projected top-5), ranked by PA (more history = more
    trustworthy), ties broken by how extreme the OBP is."""
    if top5.empty or bvp.empty:
        return []
    bvp_idx = bvp.set_index(["batter", "pitcher"])
    found = _game_bvp_candidates(game, top5, bvp_idx)
    found.sort(key=lambda m: (m["pa"], abs(m["obp"] - 0.310)), reverse=True)
    return found[:n]
