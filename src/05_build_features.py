"""
05_build_features.py
--------------------
Joins all data sources into a single model-ready feature table.

For each historical first-at-bat (or today's upcoming matchups):
  - BvP career matchup stats (with reliability flag and regression)
  - Batter rolling 7/14/30-day trends
  - Pitcher first-inning home/away splits (current season + career)
  - Park factors
  - Weather features
  - Umpire zone tendency
  - Handedness / platoon advantage
  - Days rest for pitcher

Output:
  data/processed/model_features.parquet   full historical feature table
  data/processed/todays_matchups.parquet  today's matchups ready for prediction
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"
ODDS_DIR = BASE_DIR / "data" / "odds"
LOG_DIR  = BASE_DIR / "logs"

# Maps The Odds API full team names → MLB team abbreviations used in schedule/lineup data
ODDS_TO_MLB = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
    "Athletics": "OAK", "Cleveland Indians": "CLE",
}

for d in [PROC_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "features.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# League-average priors for BvP Bayesian regression
# When sample is small, regress toward league average
LEAGUE_AVG = {
    "avg":          0.243,
    "obp":          0.315,
    "k_pct":        0.223,
    "bb_pct":       0.083,
    "hard_hit_pct": 0.380,
    "barrel_pct":   0.077,
    "woba":         0.315,
}

# Regression weight: N games of league average to blend in
# Lowered to 15 now that we use full-career BvP (larger samples, more reliable)
BVP_PRIOR_PA = 15   # equivalent to 15 PA of league average


def regress_to_mean(observed_val: float, observed_n: int, prior_val: float, prior_n: int) -> float:
    """Bayesian regression toward the mean."""
    if pd.isna(observed_val) or observed_n == 0:
        return prior_val
    return (observed_val * observed_n + prior_val * prior_n) / (observed_n + prior_n)


def platoon_adjustment(batter_hand: str, pitcher_hand: str) -> float:
    """
    Return OBP adjustment for platoon advantage.
    Same-handed matchup = pitcher advantage (negative).
    Opposite-handed = batter advantage (positive).
    Values based on historical MLB platoon splits.
    """
    if pd.isna(batter_hand) or pd.isna(pitcher_hand):
        return 0.0
    b, p = str(batter_hand).upper(), str(pitcher_hand).upper()
    if b == "S":    # Switch hitter — no platoon disadvantage
        return 0.010
    if b == p:      # Same side — pitcher advantage
        return -0.018
    else:           # Opposite — batter advantage
        return 0.018


def _compute_matchup_pitch_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Join pitcher pitch-type features and batter pitch-type profile to produce
    matchup-level edge scores.
    """
    # Map pitch code → pitch category
    _PITCH_TO_CAT = {}
    for c in ("FF","SI","FC","FT"): _PITCH_TO_CAT[c] = "Fastball"
    for c in ("SL","CU","KC","ST","SV","CS"): _PITCH_TO_CAT[c] = "Breaking Ball"
    for c in ("CH","FS","FO"): _PITCH_TO_CAT[c] = "Offspeed"

    def _row_context(row):
        primary = str(row.get("pitcher_primary_pitch", "") or "").upper()
        cat = _PITCH_TO_CAT.get(primary, "Fastball")
        weakness = str(row.get("batter_biggest_weakness", "Fastball"))

        if cat == "Fastball":
            b_whiff = row.get("batter_fb_whiff_pct", np.nan)
            b_obp   = row.get("batter_fb_obp", np.nan)
        elif cat == "Breaking Ball":
            b_whiff = row.get("batter_bb_whiff_pct", np.nan)
            b_obp   = row.get("batter_bb_obp", np.nan)
        else:
            b_whiff = row.get("batter_os_whiff_pct", np.nan)
            b_obp   = row.get("batter_os_obp", np.nan)

        p_woba = row.get("pitcher_primary_pitch_woba", np.nan)

        # Edge label
        if cat == weakness:
            edge_label = "pitcher advantage"
        elif cat == "Fastball" and weakness != "Fastball":
            edge_label = "batter advantage"
        elif cat in ("Breaking Ball", "Offspeed") and weakness == "Fastball":
            edge_label = "batter advantage"
        else:
            edge_label = "neutral"

        # Numeric score: +1 = strong batter advantage, -1 = strong pitcher advantage
        # Based on (batter's OBP vs this pitch category) relative to league avg (0.315)
        score = 0.0
        if pd.notna(b_obp):
            score += (float(b_obp) - 0.315) * 2   # scale to ~[-1, +1]
        if pd.notna(b_whiff):
            score -= (float(b_whiff) - 0.25) * 2   # high whiff → pitcher edge
        score = float(np.clip(score, -1.0, 1.0))

        return pd.Series({
            "batter_vs_primary_pitch_whiff": b_whiff,
            "batter_vs_primary_pitch_obp":   b_obp,
            "matchup_pitch_edge":             edge_label,
            "matchup_pitch_edge_score":       score,
        })

    ctx = df.apply(_row_context, axis=1)
    for col in ctx.columns:
        df[col] = ctx[col]
    return df


def _compute_fab_relevant_obp(row, p_throws_col: str, is_home_expr) -> float:
    """
    Average the pitcher-hand OBP split and the venue OBP split for this matchup.
    p_throws_col : column name containing pitcher handedness (R/L)
    is_home_expr : bool or column name "side"/"inning_topbot"
    """
    # Pitcher hand
    p = str(row.get(p_throws_col, "R") or "R").upper()
    obp_hand = row.get("fab_obp_vs_rhp" if p == "R" else "fab_obp_vs_lhp", np.nan)

    # Venue
    if isinstance(is_home_expr, str) and is_home_expr == "inning_topbot":
        is_home = (str(row.get("inning_topbot", "Top")) == "Bot")
    else:
        is_home = (str(row.get("side", "away")) == "home")
    obp_venue = row.get("fab_obp_home" if is_home else "fab_obp_away", np.nan)

    vals = [v for v in [obp_hand, obp_venue] if pd.notna(v)]
    return float(sum(vals) / len(vals)) if vals else row.get("fab_obp", np.nan)


def build_historical_features() -> pd.DataFrame:
    """Build the full historical feature table."""

    # ── Load data sources ────────────────────────────────────────────────────

    log.info("Loading data sources...")

    first_ab_path = PROC_DIR / "first_at_bats.parquet"
    # Prefer full-lifetime BvP; fall back to first-inning-only if not yet built
    bvp_full_path = PROC_DIR / "bvp_full_lifetime.parquet"
    bvp_path      = bvp_full_path if bvp_full_path.exists() else PROC_DIR / "bvp_matchups.parquet"
    rolling_path  = PROC_DIR / "batter_rolling.parquet"
    splits_path   = PROC_DIR / "pitcher_splits.parquet"
    meta_path     = PROC_DIR / "game_meta.parquet"
    weather_path  = PROC_DIR / "game_weather.parquet"

    missing = [p for p in [first_ab_path, bvp_path] if not p.exists()]
    if missing:
        log.error("Missing required files: %s\nRun 01_fetch_statcast.py first.", missing)
        return pd.DataFrame()

    first_ab = pd.read_parquet(first_ab_path)
    bvp      = pd.read_parquet(bvp_path)
    log.info("BvP source: %s", bvp_path.name)
    log.info("Loaded %d first at-bats, %d BvP pairs", len(first_ab), len(bvp))

    rolling = pd.read_parquet(rolling_path) if rolling_path.exists() else pd.DataFrame()
    splits  = pd.read_parquet(splits_path)  if splits_path.exists()  else pd.DataFrame()
    meta    = pd.read_parquet(meta_path)    if meta_path.exists()    else pd.DataFrame()
    weather = pd.read_parquet(weather_path) if weather_path.exists() else pd.DataFrame()

    df = first_ab.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    # ── 1. BvP matchup features ──────────────────────────────────────────────

    log.info("Joining BvP matchup data...")
    df = df.merge(
        bvp[["batter", "pitcher", "pa", "avg", "obp", "k_pct", "bb_pct",
              "woba", "avg_xba", "avg_xwoba", "avg_exit_velo", "hard_hit_pct",
              "barrel_pct", "bvp_reliable"]].rename(columns={
            "pa":              "bvp_pa",
            "avg":             "bvp_avg",
            "obp":             "bvp_obp",
            "k_pct":           "bvp_k_pct",
            "bb_pct":          "bvp_bb_pct",
            "woba":            "bvp_woba",
            "avg_xba":         "bvp_xba",
            "avg_xwoba":       "bvp_xwoba",
            "avg_exit_velo":   "bvp_exit_velo",
            "hard_hit_pct":    "bvp_hard_hit",
            "barrel_pct":      "bvp_barrel",
        }),
        on=["batter", "pitcher"], how="left"
    )

    # Bayesian regression — blend BvP with league average when sample is small
    for stat, league_val in LEAGUE_AVG.items():
        bvp_col  = f"bvp_{stat}" if stat in ["avg","obp","k_pct","bb_pct","woba","hard_hit_pct","barrel_pct"] else None
        if bvp_col and bvp_col in df.columns:
            df[f"{bvp_col}_adj"] = df.apply(
                lambda r: regress_to_mean(
                    r.get(bvp_col, np.nan),
                    r.get("bvp_pa", 0) or 0,
                    league_val,
                    BVP_PRIOR_PA
                ),
                axis=1
            )

    # ── 2. Batter rolling trends ─────────────────────────────────────────────

    if len(rolling):
        log.info("Joining batter rolling stats...")
        rolling["game_date"] = pd.to_datetime(rolling["game_date"])
        df = df.merge(rolling, on=["batter", "game_date"], how="left")

    # ── 3. Pitcher first-inning splits ───────────────────────────────────────

    if len(splits):
        log.info("Joining pitcher splits...")
        # Current season split
        df["game_year"] = pd.to_datetime(df["game_date"]).dt.year

        # Is the pitcher at home?
        df["pitcher_is_home"] = df["inning_topbot"].apply(
            lambda x: x == "Top" if pd.notna(x) else None
        )

        pitcher_home = splits[splits["location"] == "home"].rename(columns={
            "obp_allowed": "pitch_home_obp",
            "k_pct":       "pitch_home_k",
            "bb_pct":      "pitch_home_bb",
            "avg_exit_velo":"pitch_home_exit_velo",
        })
        pitcher_away = splits[splits["location"] == "away"].rename(columns={
            "obp_allowed": "pitch_away_obp",
            "k_pct":       "pitch_away_k",
            "bb_pct":      "pitch_away_bb",
            "avg_exit_velo":"pitch_away_exit_velo",
        })

        df = df.merge(
            pitcher_home[["pitcher","game_year","pitch_home_obp","pitch_home_k","pitch_home_bb","pitch_home_exit_velo"]],
            on=["pitcher","game_year"], how="left"
        )
        df = df.merge(
            pitcher_away[["pitcher","game_year","pitch_away_obp","pitch_away_k","pitch_away_bb","pitch_away_exit_velo"]],
            on=["pitcher","game_year"], how="left"
        )

        # Select the relevant split based on home/away status
        df["pitch_split_obp"] = np.where(
            df["pitcher_is_home"],
            df["pitch_home_obp"],
            df["pitch_away_obp"]
        )
        df["pitch_split_k"] = np.where(
            df["pitcher_is_home"],
            df["pitch_home_k"],
            df["pitch_away_k"]
        )

    # ── 4. Game metadata (umpire) ─────────────────────────────────────────────

    if len(meta):
        log.info("Joining game metadata...")
        df = df.merge(
            meta[["game_pk","hp_umpire_id","hp_umpire_name",
                  "umpire_zone_adj","umpire_strike_rate"]].drop_duplicates("game_pk"),
            on="game_pk", how="left"
        )

    # ── 5. Weather ─────────────────────────────────────────────────────────────

    if len(weather):
        log.info("Joining weather data...")
        df = df.merge(
            weather[["game_pk","temp_f","humidity","wind_speed",
                      "wind_dir","precipitation","wind_out_component","temp_bucket"]],
            on="game_pk", how="left"
        )

    # ── 6. Pitcher rest days ─────────────────────────────────────────────────

    log.info("Computing pitcher rest days...")
    df_sorted = df.sort_values(["pitcher", "game_date"])
    df["prev_game_date"] = df_sorted.groupby("pitcher")["game_date"].shift(1)
    df["pitcher_rest_days"] = (df["game_date"] - df["prev_game_date"]).dt.days
    df["pitcher_rest_days"] = df["pitcher_rest_days"].clip(0, 30).fillna(5)

    # ── 7. Handedness / platoon ──────────────────────────────────────────────

    # stand = batter handedness (L/R/S), p_throws = pitcher hand (L/R)
    df["platoon_adj"] = df.apply(
        lambda r: platoon_adjustment(r.get("stand"), r.get("p_throws")), axis=1
    )
    df["same_hand_matchup"] = (
        df["stand"].str.upper() == df["p_throws"].str.upper()
    ).astype(int)

    # ── 8. Season context features ────────────────────────────────────────────

    df["month"]         = df["game_date"].dt.month
    df["day_of_season"] = df.groupby(["game_year","home_team"])["game_date"].rank(method="dense").astype(int)
    # Early season = pitchers still building arm strength
    df["early_season"]  = (df["month"] <= 4).astype(int)

    # ── 9. First-AB batter profile ────────────────────────────────────────────
    fab_path = PROC_DIR / "batter_fab_profile.parquet"
    if fab_path.exists():
        log.info("Joining first-AB batter profile...")
        fab_profile = pd.read_parquet(fab_path)
        df = df.merge(fab_profile, on="batter", how="left")
        df["fab_relevant_obp"] = df.apply(
            lambda r: _compute_fab_relevant_obp(r, "p_throws", "inning_topbot"), axis=1
        )

    # ── 10. Batter pitch-type profile ─────────────────────────────────────────
    bpt_path = PROC_DIR / "batter_pitch_type_profile.parquet"
    if bpt_path.exists():
        log.info("Joining batter pitch-type profile...")
        bpt = pd.read_parquet(bpt_path)
        df = df.merge(bpt, on="batter", how="left")

    # ── 11. Pitcher first-inning features ─────────────────────────────────────
    pitch_feat_path = PROC_DIR / "pitcher_first_inn_features.parquet"
    if pitch_feat_path.exists():
        log.info("Joining pitcher first-inning features...")
        p_feats = pd.read_parquet(pitch_feat_path)
        # Use most recent year available per pitcher as career aggregate
        p_career = p_feats.sort_values("game_year").groupby("pitcher").last().reset_index()
        df = df.merge(p_career.drop(columns=["game_year"], errors="ignore"),
                      on="pitcher", how="left")

    # ── 12. Matchup pitch context ──────────────────────────────────────────────
    has_pitcher_primary = "pitcher_primary_pitch" in df.columns
    has_batter_weakness  = "batter_biggest_weakness" in df.columns
    if has_pitcher_primary and has_batter_weakness:
        log.info("Computing matchup pitch context...")
        df = _compute_matchup_pitch_context(df)

    log.info("Feature table built: %d rows × %d columns", len(df), len(df.columns))
    return df


def _add_line_movement(df: pd.DataFrame, today: str) -> pd.DataFrame:
    """Add line movement columns to today's matchups using odds_history.parquet."""
    hist_path = ODDS_DIR / "odds_history.parquet"
    if not hist_path.exists():
        return df

    history = pd.read_parquet(hist_path)
    today_hist = history[history["game_date"] == today].copy()
    if today_hist.empty:
        return df

    relevant = today_hist[
        today_hist["market"].isin(["batter_hits", "batter_total_bases"]) &
        (today_hist["side"] == "Over")
    ].copy()
    if relevant.empty:
        return df

    relevant["ts"] = pd.to_datetime(relevant["timestamp"])

    movement_rows = []
    for player, grp in relevant.groupby("player"):
        min_ts = grp["ts"].min()
        max_ts = grp["ts"].max()

        opening_snap = grp[grp["ts"] == min_ts]
        current_snap = grp[grp["ts"] == max_ts]

        # Best odds = highest American odds for bettor (lowest implied prob)
        opening_row = opening_snap.loc[opening_snap["implied_prob"].idxmin()]
        current_row = current_snap.loc[current_snap["implied_prob"].idxmin()]

        opening_odds    = float(opening_row["american_odds"])
        opening_prob    = float(opening_row["implied_prob"])
        current_odds    = float(current_row["american_odds"])
        current_prob    = float(current_row["implied_prob"])
        movement_pct    = current_prob - opening_prob

        movement_rows.append({
            "batter_name":          player,
            "opening_odds":         opening_odds,
            "opening_implied_prob": opening_prob,
            "current_odds":         current_odds,
            "current_implied_prob": current_prob,
            "line_movement_pct":    round(movement_pct, 4),
        })

    if not movement_rows:
        return df

    mov_df = pd.DataFrame(movement_rows)
    df = df.merge(mov_df, on="batter_name", how="left")

    # Movement direction relative to model probability
    def direction(row):
        mp = row.get("line_movement_pct", np.nan)
        if pd.isna(mp):
            return None
        if abs(mp) < 0.005:
            return "neutral"
        model_p = row.get("model_prob", np.nan)
        if pd.isna(model_p):
            return "neutral"
        curr_edge = model_p - row.get("current_implied_prob", np.nan)
        open_edge = model_p - row.get("opening_implied_prob", np.nan)
        if pd.isna(curr_edge) or pd.isna(open_edge):
            return "neutral"
        return "toward model" if abs(curr_edge) < abs(open_edge) else "away from model"

    df["line_movement_direction"] = df.apply(direction, axis=1)

    # True if current edge > 4% but opening edge was ≤ 4%
    model_p_s  = df.get("model_prob",            pd.Series(np.nan, index=df.index))
    curr_imp_s = df.get("current_implied_prob",  pd.Series(np.nan, index=df.index))
    open_imp_s = df.get("opening_implied_prob",  pd.Series(np.nan, index=df.index))

    df["line_crossed_threshold"] = (
        model_p_s.notna() & curr_imp_s.notna() & open_imp_s.notna() &
        ((model_p_s - curr_imp_s) > 0.04) &
        ((model_p_s - open_imp_s) <= 0.04)
    ).fillna(False)

    log.info(
        "Line movement added: %d matchups with data, %d crossed threshold",
        mov_df["batter_name"].isin(df["batter_name"]).sum(),
        int(df["line_crossed_threshold"].sum()),
    )
    return df


def _build_probable_lineups(today: str) -> pd.DataFrame:
    """
    Fallback when batting orders aren't yet posted.
    Uses today's probable pitchers from game_meta + each team's most recent
    confirmed leadoff batter from historical lineup data.
    Returns rows in the same schema as lineup_first_batter.parquet.
    """
    meta_path   = PROC_DIR / "game_meta.parquet"
    lineup_path = PROC_DIR / "lineup_first_batter.parquet"
    if not meta_path.exists() or not lineup_path.exists():
        return pd.DataFrame()

    meta    = pd.read_parquet(meta_path)
    lineup_hist = pd.read_parquet(lineup_path)

    # Drop NaN-batter stub rows from history
    lineup_hist = lineup_hist[pd.notna(lineup_hist["batter_id"])].copy()
    lineup_hist["batter_id"] = pd.to_numeric(lineup_hist["batter_id"], errors="coerce").astype("Int64")

    # Today's games
    today_meta = meta[meta["game_date"].astype(str).str.startswith(today)].copy()
    if today_meta.empty:
        return pd.DataFrame()

    # Attach team abbrev to historical lineup rows via game_meta join
    meta_slim = meta[["game_pk", "home_team_abbr", "away_team_abbr"]].drop_duplicates("game_pk")
    lineup_with_team = lineup_hist.merge(meta_slim, on="game_pk", how="left")
    lineup_with_team["team_abbr"] = np.where(
        lineup_with_team["side"] == "home",
        lineup_with_team["home_team_abbr"],
        lineup_with_team["away_team_abbr"],
    )
    lineup_with_team = lineup_with_team[pd.notna(lineup_with_team["team_abbr"])]

    # Most recent confirmed leadoff batter per team
    recent_leadoff = (
        lineup_with_team.sort_values("game_date")
        .groupby("team_abbr")
        .last()
        .reset_index()
        [["team_abbr", "batter_id", "batter_name"]]
    )
    team_to_batter = recent_leadoff.set_index("team_abbr").to_dict("index")

    rows = []
    for _, game in today_meta.iterrows():
        for side in ["home", "away"]:
            abbr = game.get(f"{side}_team_abbr")
            opp  = "away" if side == "home" else "home"
            starter_id   = game.get(f"{opp}_starter_id")
            starter_name = game.get(f"{opp}_starter_name")

            batter_info = team_to_batter.get(abbr)
            if batter_info is None or pd.isna(batter_info.get("batter_id")):
                continue

            rows.append({
                "game_pk":            int(game["game_pk"]),
                "game_date":          today,
                "season":             int(today[:4]),
                "side":               side,
                "batter_id":          batter_info["batter_id"],
                "batter_name":        batter_info["batter_name"],
                "pitcher_id":         starter_id,
                "pitcher_name":       starter_name,
                "venue_name":         game.get("venue_name"),
                "hp_umpire_id":       game.get("hp_umpire_id"),
                "hp_umpire_name":     game.get("hp_umpire_name"),
                "umpire_zone_adj":    game.get("umpire_zone_adj", 0.0),
                "umpire_strike_rate": game.get("umpire_strike_rate", 0.455),
                "lineup_confirmed":   False,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["batter_id"]  = pd.to_numeric(df["batter_id"],  errors="coerce").astype("Int64")
    df["pitcher_id"] = pd.to_numeric(df["pitcher_id"], errors="coerce").astype("Int64")
    log.info(
        "Probable lineup fallback: %d matchup rows built from recent leadoff history", len(df)
    )
    return df


def build_todays_matchups() -> pd.DataFrame:
    """
    Build feature rows for today's games using active rosters + probable pitchers.
    One row per active batter on each team vs the opposing probable starter.
    """
    import json as _json
    today = datetime.now().strftime("%Y-%m-%d")
    log.info("Building today's matchups (%s) — roster-based approach…", today)

    # ── 1. Probable pitchers ───────────────────────────────────────────────────
    pp_path = PROC_DIR / "probable_pitchers.parquet"
    if not pp_path.exists():
        log.warning("probable_pitchers.parquet not found — run 02_fetch_mlb_api.py --probable-pitchers-only")
        return pd.DataFrame()
    pp = pd.read_parquet(pp_path)
    today_pp = pp[pp["game_date"].astype(str).str.startswith(today)].copy()
    if today_pp.empty:
        log.warning("No probable pitcher data for today (%s)", today)
        return pd.DataFrame()

    # ── 2. Active rosters ─────────────────────────────────────────────────────
    ar_path = PROC_DIR / "active_rosters.parquet"
    if not ar_path.exists():
        log.warning("active_rosters.parquet not found — run 02_fetch_mlb_api.py --rosters-only")
        return pd.DataFrame()
    rosters = pd.read_parquet(ar_path)
    rosters["player_id"] = pd.to_numeric(rosters["player_id"], errors="coerce").astype("Int64")

    # ── 3. Game meta (venue / umpire) ─────────────────────────────────────────
    meta_path = PROC_DIR / "game_meta.parquet"
    today_meta = pd.DataFrame()
    if meta_path.exists():
        meta = pd.read_parquet(meta_path)
        today_meta = meta[meta["game_date"].astype(str).str.startswith(today)].copy()

    # ── 4. Confirmed batting orders from cached game feeds ────────────────────
    confirmed_orders: dict = {}   # {game_pk: {player_id: lineup_position}}
    feeds_dir = BASE_DIR / "data" / "raw" / "game_feeds"
    if feeds_dir.exists():
        for game_pk in today_pp["game_pk"].dropna().astype(int).tolist():
            fp = feeds_dir / f"{game_pk}.json"
            if not fp.exists():
                continue
            try:
                with open(fp) as f:
                    feed = _json.load(f)
                bs = feed.get("liveData", {}).get("boxscore", {}).get("teams", {})
                order_map: dict = {}
                for side in ["home", "away"]:
                    for pos_idx, pid in enumerate(bs.get(side, {}).get("battingOrder", []), start=1):
                        order_map[int(pid)] = pos_idx
                if order_map:
                    confirmed_orders[game_pk] = order_map
            except Exception:
                pass
    log.info("Confirmed lineups for %d / %d games", len(confirmed_orders), len(today_pp))

    # ── 5. Build one row per active batter vs opposing pitcher ─────────────────
    rows = []
    for _, game in today_pp.iterrows():
        game_pk  = int(game["game_pk"])
        conf_ord = confirmed_orders.get(game_pk, {})

        meta_row     = today_meta[today_meta["game_pk"] == game_pk]
        venue_name   = meta_row["venue_name"].iloc[0]      if not meta_row.empty else None
        umpire_id    = meta_row["hp_umpire_id"].iloc[0]    if not meta_row.empty else None
        umpire_name  = meta_row["hp_umpire_name"].iloc[0]  if not meta_row.empty else None
        umpire_zone  = float(meta_row["umpire_zone_adj"].iloc[0])   if not meta_row.empty else 0.0
        umpire_str   = float(meta_row["umpire_strike_rate"].iloc[0])if not meta_row.empty else 0.455

        for side in ["home", "away"]:
            opp        = "away" if side == "home" else "home"
            team_id    = game.get(f"{side}_team_id")
            pitcher_id = game.get(f"{opp}_probable_pitcher_id")
            pitcher_nm = game.get(f"{opp}_probable_pitcher_name")
            p_hand     = str(game.get(f"{opp}_probable_pitcher_hand") or "R")

            if pd.isna(pitcher_id):
                continue

            for _, br in rosters[rosters["team_id"] == team_id].iterrows():
                b_id  = br["player_id"]
                conf_pos = conf_ord.get(int(b_id)) if pd.notna(b_id) else None
                rows.append({
                    "game_pk":                   game_pk,
                    "game_date":                 today,
                    "season":                    int(today[:4]),
                    "side":                      side,
                    "batter":                    b_id,
                    "batter_name":               br["full_name"],
                    "pitcher":                   pitcher_id,
                    "pitcher_name":              pitcher_nm,
                    "p_throws":                  p_hand,
                    "stand":                     br.get("bat_side", "R"),
                    "venue_name":                venue_name,
                    "hp_umpire_id":              umpire_id,
                    "hp_umpire_name":            umpire_name,
                    "umpire_zone_adj":           umpire_zone,
                    "umpire_strike_rate":        umpire_str,
                    "lineup_confirmed":          conf_pos is not None,
                    "confirmed_lineup_position": conf_pos,
                    "is_on_active_roster":       True,
                })

    if not rows:
        log.warning("No roster-based matchups could be built for today.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["batter"]  = pd.to_numeric(df["batter"],  errors="coerce").astype("Int64")
    df["pitcher"] = pd.to_numeric(df["pitcher"], errors="coerce").astype("Int64")
    df["game_date"] = pd.to_datetime(df["game_date"])

    log.info("Found %d roster-based matchups for today (%d games).",
             len(df), len(today_pp))

    # ── Supporting table paths (unchanged from here down) ─────────────────────
    _bvp_full  = PROC_DIR / "bvp_full_lifetime.parquet"
    bvp_path   = _bvp_full if _bvp_full.exists() else PROC_DIR / "bvp_matchups.parquet"
    rolling_path= PROC_DIR / "batter_rolling.parquet"
    splits_path = PROC_DIR / "pitcher_splits.parquet"
    weather_path= PROC_DIR / "game_weather.parquet"
    _dated_props = ODDS_DIR / f"props_{today}.parquet"
    props_path   = _dated_props if _dated_props.exists() else ODDS_DIR / "todays_props.parquet"

    if bvp_path.exists():
        bvp = pd.read_parquet(bvp_path)
        bvp["batter"]  = bvp["batter"].astype("Int64")
        bvp["pitcher"] = bvp["pitcher"].astype("Int64")
        df = df.merge(
            bvp[["batter","pitcher","pa","avg","obp","k_pct","bb_pct","woba",
                  "avg_xba","avg_exit_velo","hard_hit_pct","barrel_pct","bvp_reliable"]],
            on=["batter","pitcher"], how="left"
        )
        # Regression
        for stat, league_val in LEAGUE_AVG.items():
            col = stat if stat in df.columns else None
            if col:
                df[f"bvp_{col}_adj"] = df.apply(
                    lambda r: regress_to_mean(r.get(col, np.nan), r.get("pa", 0) or 0,
                                              league_val, BVP_PRIOR_PA), axis=1
                )

    # ── PA filter — remove matchups with no meaningful BvP history ────────────
    pre_filter = len(df)
    has_pa = df["pa"].fillna(0) >= 5 if "pa" in df.columns else pd.Series(False, index=df.index)
    if has_pa.sum() > 0:
        df = df[has_pa].copy()
        log.info("PA filter: kept %d / %d matchups with ≥ 5 career first-AB PA",
                 len(df), pre_filter)
    else:
        # Fallback: no pairs have ≥5 PA in first-AB data — keep all, mark samples
        log.warning(
            "PA filter: no matchups have ≥5 first-AB PA (max is %s). "
            "Keeping all %d matchups. Use bvp_sample_warning badge for guidance.",
            int(df["pa"].fillna(0).max()) if "pa" in df.columns else "?",
            pre_filter,
        )

    # bvp_sample_warning = True when some BvP history exists but below reliable threshold
    if "pa" in df.columns:
        pa_vals = df["pa"].fillna(0)
        df["bvp_sample_warning"] = ((pa_vals > 0) & (pa_vals < 15)).astype(bool)
    else:
        df["bvp_sample_warning"] = False

    # Most recent batter rolling stats (use today's date)
    if rolling_path.exists():
        rolling = pd.read_parquet(rolling_path)
        rolling["game_date"] = pd.to_datetime(rolling["game_date"])
        latest = rolling.sort_values("game_date").groupby("batter").last().reset_index()
        df = df.merge(latest.drop(columns=["game_date"], errors="ignore"), on="batter", how="left")

    # ── Pitcher first-inning splits (current season) ─────────────────────────
    if splits_path.exists():
        splits = pd.read_parquet(splits_path)
        df["game_year"] = pd.to_datetime(df["game_date"]).dt.year

        pitcher_home = splits[splits["location"] == "home"].rename(columns={
            "obp_allowed": "pitch_home_obp",
            "k_pct":       "pitch_home_k",
            "bb_pct":      "pitch_home_bb",
        })
        pitcher_away = splits[splits["location"] == "away"].rename(columns={
            "obp_allowed": "pitch_away_obp",
            "k_pct":       "pitch_away_k",
            "bb_pct":      "pitch_away_bb",
        })

        df = df.merge(
            pitcher_home[["pitcher","game_year","pitch_home_obp","pitch_home_k","pitch_home_bb"]],
            on=["pitcher","game_year"], how="left",
        )
        df = df.merge(
            pitcher_away[["pitcher","game_year","pitch_away_obp","pitch_away_k","pitch_away_bb"]],
            on=["pitcher","game_year"], how="left",
        )

        # Career split (latest year available) as fallback
        for loc, pfx in [("home","pitch_home"), ("away","pitch_away")]:
            for stat in ["obp","k","bb"]:
                col = f"{pfx}_{stat}"
                if col not in df.columns:
                    df[col] = np.nan

        df["pitch_split_obp"] = np.where(
            df.get("side", "?") == "home", df.get("pitch_home_obp"), df.get("pitch_away_obp")
        )
        df["pitch_split_k"] = np.where(
            df.get("side", "?") == "home", df.get("pitch_home_k"), df.get("pitch_away_k")
        )

    # ── Form trend from rolling stats ─────────────────────────────────────────
    if "obp_7d" in df.columns and "obp_30d" in df.columns:
        df["form_trend"] = df["obp_7d"] - df["obp_30d"]
    else:
        df["form_trend"] = np.nan

    def _form_label(ft):
        if pd.isna(ft): return "➡️ Neutral"
        ft = float(ft)
        if ft >  0.040: return "🔥 Hot"
        if ft >  0.010: return "📈 Trending up"
        if ft > -0.010: return "➡️ Neutral"
        if ft > -0.040: return "📉 Trending down"
        return "🧊 Cold"

    df["form_label"] = df["form_trend"].apply(_form_label)

    # ── First-AB batter profile ────────────────────────────────────────────────
    fab_profile_path = PROC_DIR / "batter_fab_profile.parquet"
    first_ab_path_2  = PROC_DIR / "first_at_bats.parquet"
    if fab_profile_path.exists():
        fab_profile = pd.read_parquet(fab_profile_path)
        fab_profile["batter"] = fab_profile["batter"].astype("Int64")
        df = df.merge(fab_profile, on="batter", how="left")

        # Build pitcher handedness lookup from first_at_bats.parquet
        if first_ab_path_2.exists():
            fab_data = pd.read_parquet(first_ab_path_2, columns=["pitcher","p_throws"])
            _pitch_hand = (
                fab_data.groupby("pitcher")["p_throws"]
                .agg(lambda x: x.mode().iloc[0] if len(x) else "R")
                .to_dict()
            )
            df["pitcher_throws"] = df["pitcher"].apply(
                lambda p: _pitch_hand.get(int(p) if pd.notna(p) else -1, "R")
            )
        else:
            df["pitcher_throws"] = "R"

        df["fab_relevant_obp"] = df.apply(
            lambda r: _compute_fab_relevant_obp(r, "pitcher_throws", "side"), axis=1
        )
        log.info("FAB profile joined: %d with fab_pa data",
                 df["fab_pa"].notna().sum() if "fab_pa" in df.columns else 0)

    # ── Batter pitch-type profile ──────────────────────────────────────────────
    bpt_path = PROC_DIR / "batter_pitch_type_profile.parquet"
    if bpt_path.exists():
        bpt = pd.read_parquet(bpt_path)
        bpt["batter"] = bpt["batter"].astype("Int64")
        df = df.merge(bpt, on="batter", how="left")
        log.info("Batter pitch-type profile joined")

    # ── Pitcher first-inning features ──────────────────────────────────────────
    pitch_feat_path = PROC_DIR / "pitcher_first_inn_features.parquet"
    if pitch_feat_path.exists():
        p_feats = pd.read_parquet(pitch_feat_path)
        p_feats["pitcher"] = p_feats["pitcher"].astype("Int64")
        p_career = p_feats.sort_values("game_year").groupby("pitcher").last().reset_index()
        df = df.merge(p_career.drop(columns=["game_year"], errors="ignore"),
                      on="pitcher", how="left")
        log.info("Pitcher first-inn features joined")

    # ── Matchup pitch context ──────────────────────────────────────────────────
    if "pitcher_primary_pitch" in df.columns and "batter_biggest_weakness" in df.columns:
        df = _compute_matchup_pitch_context(df)
        log.info("Matchup pitch context computed")

    if weather_path.exists():
        weather = pd.read_parquet(weather_path)
        today_wx = weather[weather["game_date"] == today]
        df = df.merge(today_wx, on="game_pk", how="left")

    # ── Player prop odds (batter_hits or batter_total_bases Over 0.5) ──────────
    if props_path.exists():
        props = pd.read_parquet(props_path)
        # Only use odds that match today — prevents stale cross-day contamination
        if "game_date" in props.columns:
            props = props[props["game_date"].astype(str).str.startswith(today)]
            if props.empty:
                log.info("Props file exists but contains no data for %s (quota exhausted or stale).", today)
        # Try batter_hits first, fall back to batter_total_bases as proxy
        for market in ["batter_hits", "batter_total_bases"]:
            hits_props = props[
                (props["market"] == market) &
                (props["side"] == "Over") &
                (props["line"] == 0.5)
            ][["player", "american_odds", "implied_prob"]].copy()
            if len(hits_props):
                log.info("Using '%s' Over 0.5 as reach-base proxy (%d rows)", market, len(hits_props))
                hits_props = hits_props.groupby("player").agg(
                    best_odds=("american_odds", "max"),
                    avg_implied_prob=("implied_prob", "mean"),
                ).reset_index()
                df = df.merge(
                    hits_props.rename(columns={"player": "batter_name"}),
                    on="batter_name", how="left",
                )
                break
        else:
            log.info("No batter_hits or batter_total_bases Over 0.5 props found today.")

    # ── Game-level odds fallback via ODDS_TO_MLB team name mapping ─────────────
    # When player props are absent, derive avg_implied_prob from the game total
    # line: expected total runs scales linearly with OBP (league avg = 9 runs, 0.315 OBP)
    game_odds_path = ODDS_DIR / f"game_odds_{today}.parquet"
    schedule_path  = PROC_DIR / "schedule.parquet"
    already_has_probs = ("avg_implied_prob" in df.columns and df["avg_implied_prob"].notna().any())

    if game_odds_path.exists() and schedule_path.exists():
        game_odds = pd.read_parquet(game_odds_path)
        # Map Odds API full names → abbrev for joining
        game_odds["home_abbr"] = game_odds["home_team"].map(ODDS_TO_MLB)
        game_odds["away_abbr"] = game_odds["away_team"].map(ODDS_TO_MLB)

        # Extract over/under total line per game (average across bookmakers)
        totals = game_odds[(game_odds["market"] == "totals") & (game_odds["outcome"] == "Over")]
        if len(totals):
            total_by_game = (
                totals.groupby(["home_abbr", "away_abbr"])
                .agg(total_line=("line", "mean"))
                .reset_index()
            )
            # Join schedule (game_pk → home/away abbrev)
            sched = pd.read_parquet(schedule_path)
            sched_today = sched[sched["game_date"].astype(str).str.startswith(today)][
                ["game_pk", "home_team_name", "away_team_name"]
            ].copy()
            sched_today["home_abbr"] = sched_today["home_team_name"].map(ODDS_TO_MLB)
            sched_today["away_abbr"] = sched_today["away_team_name"].map(ODDS_TO_MLB)

            sched_today = sched_today.merge(total_by_game, on=["home_abbr", "away_abbr"], how="left")
            df = df.merge(sched_today[["game_pk", "total_line"]].drop_duplicates("game_pk"),
                          on="game_pk", how="left")
            log.info("Joined game total lines for %d matchups", df["total_line"].notna().sum())

            # If no player props populated avg_implied_prob, derive from totals
            if not already_has_probs and "total_line" in df.columns:
                LEAGUE_AVG_TOTAL = 9.0
                LEAGUE_AVG_OBP   = 0.315
                df["avg_implied_prob"] = (
                    (df["total_line"] / LEAGUE_AVG_TOTAL) * LEAGUE_AVG_OBP
                ).clip(0.20, 0.44)
                log.info("Derived avg_implied_prob from game totals for %d matchups",
                         df["avg_implied_prob"].notna().sum())

    # ── Lineup position history ───────────────────────────────────────────────
    lps_path = PROC_DIR / "lineup_position_summary.parquet"
    if lps_path.exists():
        try:
            lps = pd.read_parquet(lps_path)
            lps["player_id"] = pd.to_numeric(lps["player_id"], errors="coerce").astype("Int64")
            pos_cols = (
                ["player_id", "most_common_position", "position_versatility", "total_games_played"]
                + [c for c in lps.columns
                   if c.startswith(("games_pos_", "obp_pos_", "avg_pos_", "slg_pos_",
                                    "ops_pos_", "k_pct_pos_", "bb_pct_pos_", "hr_pos_", "pa_pos_",
                                    "ab_pos_", "h_pos_", "bb_pos_", "k_pos_", "rbi_pos_"))]
            )
            pos_cols = [c for c in pos_cols if c in lps.columns]
            df = df.merge(
                lps[pos_cols].rename(columns={"player_id": "batter"}),
                on="batter", how="left",
            )
            df["most_common_lineup_position"] = pd.to_numeric(
                df.get("most_common_position"), errors="coerce"
            )
            log.info("Lineup position history joined: %d / %d batters with data",
                     df["most_common_lineup_position"].notna().sum(), len(df))
        except Exception as exc:
            log.warning("Could not join lineup position history: %s", exc)
    else:
        df["most_common_lineup_position"] = np.nan

    # Projected position: confirmed > most common historical > NaN
    df["projected_lineup_position"] = np.where(
        df["lineup_confirmed"] & df["confirmed_lineup_position"].notna(),
        df["confirmed_lineup_position"].astype("float64"),
        df["most_common_lineup_position"],
    )

    # Sort: game_pk first, then projected position (NaN batters go last)
    df = df.sort_values(
        ["game_pk", "projected_lineup_position"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)

    # ── Line movement from odds_history ─────────────────────────────────────
    df = _add_line_movement(df, today)

    out_path = PROC_DIR / "todays_matchups.parquet"
    df.to_parquet(out_path, index=False)
    log.info("Saved today's matchups: %d rows → %s", len(df), out_path)

    # Print summary
    print("\n" + "=" * 70)
    print(f"TODAY'S FIRST-BATTER MATCHUPS — {today}")
    print("=" * 70)
    display_cols = ["batter_name","pitcher_name","side","obp","k_pct",
                    "avg_exit_velo","temp_f","wind_speed","best_odds","avg_implied_prob"]
    display_cols = [c for c in display_cols if c in df.columns]
    with pd.option_context("display.max_rows", 50, "display.width", 120):
        print(df[display_cols].to_string(index=False))
    print("=" * 70)

    return df


def run():
    # Build historical feature table
    log.info("Building historical feature table...")
    hist = build_historical_features()
    if len(hist):
        out_path = PROC_DIR / "model_features.parquet"
        hist.to_parquet(out_path, index=False)
        log.info("Model features saved: %d rows × %d cols → %s",
                 len(hist), len(hist.columns), out_path)

        # Print feature summary
        log.info("\nTarget variable distribution:")
        log.info(hist["outcome"].value_counts().to_string())
        log.info("\nReach base rate: %.1f%%", hist["reached_base"].mean() * 100)

    # Build today's matchups
    todays = build_todays_matchups()

    log.info("Feature pipeline complete.")


if __name__ == "__main__":
    run()
