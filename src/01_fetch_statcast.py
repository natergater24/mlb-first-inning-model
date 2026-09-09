"""
01_fetch_statcast.py
--------------------
Pulls pitch-level Statcast data from Baseball Savant for every regular season
from 2015 (first full Statcast year) through the current season.

Filters applied at query time:
  - Regular season games only
  - Inning 1 only
  - 0 outs when batter steps in (leadoff AB of the inning / game)

Each row = one pitch. We then collapse to at-bat level (one row per PA)
with the final event (strikeout, single, walk, etc.) and all Statcast
metrics for that plate appearance.

Output:
  data/raw/statcast_{year}.csv          raw pitch-level CSV per season
  data/processed/first_at_bats.parquet  collapsed PA-level feature table
"""

import os
import json
import time
import logging
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from io import StringIO

# ── Config ──────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).resolve().parent.parent
RAW_DIR    = BASE_DIR / "data" / "raw"
PROC_DIR   = BASE_DIR / "data" / "processed"
LOG_DIR    = BASE_DIR / "logs"

for d in [RAW_DIR, PROC_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

STATE_PATH = PROC_DIR / "pipeline_state.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "statcast.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# Statcast reliable from 2015 onward
SEASONS = list(range(2015, datetime.now().year + 1))

# Baseball Savant CSV endpoint
SAVANT_CSV_URL = "https://baseballsavant.mlb.com/statcast_search/csv"

# Columns we care about at the pitch level
PITCH_COLS = [
    "pitch_type", "game_date", "release_speed", "release_pos_x", "release_pos_z",
    "player_name", "batter", "pitcher", "events", "description",
    "zone", "des", "game_type", "stand", "p_throws",
    "home_team", "away_team", "type", "hit_location", "bb_type",
    "balls", "strikes", "game_year", "pfx_x", "pfx_z",
    "plate_x", "plate_z", "on_3b", "on_2b", "on_1b",
    "outs_when_up", "inning", "inning_topbot",
    "hc_x", "hc_y", "sv_id", "vx0", "vy0", "vz0", "ax", "ay", "az",
    "sz_top", "sz_bot", "hit_distance_sc", "launch_speed", "launch_angle",
    "effective_speed", "release_spin_rate", "release_extension",
    "game_pk", "fielder_2", "fielder_3", "fielder_4", "fielder_5",
    "fielder_6", "fielder_7", "fielder_8", "fielder_9",
    "release_pos_y", "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle", "woba_value", "woba_denom",
    "babip_value", "iso_value", "launch_speed_angle",
    "at_bat_number", "pitch_number", "pitch_name",
    "home_score", "away_score", "bat_score", "fld_score",
    "post_away_score", "post_home_score", "post_bat_score", "post_fld_score",
    "if_fielding_alignment", "of_fielding_alignment",
    "spin_axis", "delta_home_win_exp", "delta_run_exp",
    "bat_speed", "swing_length",
]

# Events that end a plate appearance
TERMINAL_EVENTS = {
    "strikeout", "strikeout_double_play",
    "walk", "intent_walk",
    "hit_by_pitch",
    "single", "double", "triple", "home_run",
    "field_out", "force_out", "grounded_into_double_play",
    "double_play", "triple_play",
    "field_error", "fielders_choice", "fielders_choice_out",
    "sac_fly", "sac_bunt", "sac_fly_double_play", "sac_bunt_double_play",
    "catcher_interf",
}

# Outcome groupings for the model target variable
def classify_outcome(event: str) -> str:
    """Coarse target: reach_base | strikeout | other_out"""
    if pd.isna(event):
        return "unknown"
    e = str(event).lower()
    if e in {"walk", "intent_walk", "hit_by_pitch", "single", "double",
             "triple", "home_run", "catcher_interf", "field_error"}:
        return "reach_base"
    if e in {"strikeout", "strikeout_double_play"}:
        return "strikeout"
    if e in TERMINAL_EVENTS:
        return "other_out"
    return "unknown"


# ── Savant query helpers ─────────────────────────────────────────────────────

def build_savant_params(season: int, game_date_gt: str = "", game_date_lt: str = "") -> dict:
    """
    Build query params for Baseball Savant CSV endpoint.
    Filters: regular season, inning 1, 0 outs when up (true leadoff PA).
    """
    return {
        "all":          "true",
        "hfGT":         "R|",       # Regular season
        "hfSea":        f"{season}|",
        "hfInn":        "1|",       # Inning 1
        "hfOuts":       "0|",       # 0 outs (leadoff)
        "game_date_gt": game_date_gt,
        "game_date_lt": game_date_lt,
        "player_type":  "batter",
        "min_pitches":  "0",
        "min_results":  "0",
        "group_by":     "name",
        "sort_col":     "pitches",
        "sort_order":   "desc",
        "min_abs":      "0",
        "type":         "details",
    }


def build_savant_params_full(season: int, game_date_gt: str = "", game_date_lt: str = "") -> dict:
    """
    Build query params for the full first-5-innings dataset.
    No outs filter — captures every batter's first PA of the game
    regardless of inning or lineup position.
    Saves to statcast_full_{year}.csv (distinct from leadoff-only statcast_{year}.csv).
    """
    return {
        "all":          "true",
        "hfGT":         "R|",
        "hfSea":        f"{season}|",
        "hfInn":        "1|2|3|4|5|",   # innings 1-5, no outs filter
        "game_date_gt": game_date_gt,
        "game_date_lt": game_date_lt,
        "player_type":  "batter",
        "min_pitches":  "0",
        "min_results":  "0",
        "group_by":     "name",
        "sort_col":     "pitches",
        "sort_order":   "desc",
        "min_abs":      "0",
        "type":         "details",
    }


def fetch_savant_csv(params: dict, retries: int = 3, delay: float = 5.0) -> pd.DataFrame | None:
    """
    GET the Statcast CSV from Baseball Savant with retry logic.
    Returns a DataFrame or None on failure.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://baseballsavant.mlb.com/statcast_search",
    }
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                SAVANT_CSV_URL,
                params=params,
                headers=headers,
                timeout=120,
            )
            resp.raise_for_status()
            if not resp.text.strip() or resp.text.startswith("<!DOCTYPE"):
                log.warning("  Got HTML instead of CSV (attempt %d)", attempt)
                time.sleep(delay * attempt)
                continue
            df = pd.read_csv(StringIO(resp.text), low_memory=False)
            log.info("  Fetched %d rows", len(df))
            return df
        except requests.exceptions.Timeout:
            log.warning("  Timeout on attempt %d", attempt)
        except requests.exceptions.HTTPError as e:
            log.warning("  HTTP %s on attempt %d", e.response.status_code, attempt)
        except Exception as e:
            log.warning("  Error on attempt %d: %s", attempt, e)
        time.sleep(delay * attempt)
    return None


def fetch_season(season: int, chunk_months: bool = True) -> pd.DataFrame | None:
    """
    Fetch a full season. Optionally chunks by month to avoid Savant timeouts
    on large seasons (2017+ have 700k+ pitches in the full season).
    Returns combined DataFrame.
    """
    raw_path = RAW_DIR / f"statcast_{season}.csv"

    if raw_path.exists():
        log.info("Season %d already downloaded, loading from disk.", season)
        return pd.read_csv(raw_path, low_memory=False)

    log.info("Fetching season %d ...", season)

    if not chunk_months:
        params = build_savant_params(season)
        df = fetch_savant_csv(params)
        if df is not None and len(df):
            df.to_csv(raw_path, index=False)
        return df

    # Chunk by month to avoid Savant response size limits
    season_start = f"{season}-03-01"
    season_end   = f"{season}-11-30"
    months = pd.date_range(start=season_start, end=season_end, freq="MS")
    parts  = []

    for month_start in months:
        month_end = (month_start + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        ms = month_start.strftime("%Y-%m-%d")
        log.info("  %s → %s", ms, month_end)
        params = build_savant_params(season, game_date_gt=ms, game_date_lt=month_end)
        chunk  = fetch_savant_csv(params)
        if chunk is not None and len(chunk):
            parts.append(chunk)
        time.sleep(3)   # be polite to Savant

    if not parts:
        log.error("  No data returned for season %d", season)
        return None

    df = pd.concat(parts, ignore_index=True)
    df.drop_duplicates(inplace=True)
    log.info("  Season %d total: %d rows", season, len(df))
    df.to_csv(raw_path, index=False)
    return df


def fetch_season_full(season: int, chunk_months: bool = True) -> pd.DataFrame | None:
    """
    Fetch innings 1-5 (all PA) for a season.
    Saves to statcast_full_{season}.csv — never overwrites statcast_{season}.csv.
    Returns DataFrame or None.
    """
    raw_path = RAW_DIR / f"statcast_full_{season}.csv"

    if raw_path.exists():
        log.info("Full data season %d already cached, loading from disk.", season)
        return pd.read_csv(raw_path, low_memory=False)

    log.info("Fetching full data (inn 1-5) for season %d ...", season)

    if not chunk_months:
        params = build_savant_params_full(season)
        df = fetch_savant_csv(params)
        if df is not None and len(df):
            df.to_csv(raw_path, index=False)
        return df

    season_start = f"{season}-03-01"
    season_end   = f"{season}-11-30"
    months = pd.date_range(start=season_start, end=season_end, freq="MS")
    parts  = []

    for month_start in months:
        month_end = (month_start + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        ms = month_start.strftime("%Y-%m-%d")
        log.info("  [full] %s → %s", ms, month_end)
        params = build_savant_params_full(season, game_date_gt=ms, game_date_lt=month_end)
        chunk  = fetch_savant_csv(params)
        if chunk is not None and len(chunk):
            parts.append(chunk)
        time.sleep(3)

    if not parts:
        log.error("  No full data returned for season %d", season)
        return None

    df = pd.concat(parts, ignore_index=True)
    df.drop_duplicates(inplace=True)
    log.info("  Season %d full: %d rows", season, len(df))
    df.to_csv(raw_path, index=False)
    return df


# ── PA-level collapse ────────────────────────────────────────────────────────

def collapse_to_plate_appearances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each pitch row → one row per plate appearance.
    We keep the last pitch of each PA (which has the terminal event),
    then add aggregated pitch-sequence features.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    # Keep only inning 1, leadoff
    df = df[
        (df["inning"] == 1) &
        (df["outs_when_up"] == 0) &
        (df["game_type"] == "R")
    ].copy()

    if len(df) == 0:
        return pd.DataFrame()

    # Pitch sequence features per PA
    pa_group = ["game_pk", "at_bat_number"]

    # Per-PA aggregates
    agg = df.groupby(pa_group).agg(
        pitch_count         = ("pitch_number",    "max"),
        max_velo            = ("release_speed",   "max"),
        avg_velo            = ("release_speed",   "mean"),
        pitches_seen        = ("pitch_type",       "count"),
        first_pitch_type    = ("pitch_type",       "first"),
        first_pitch_zone    = ("zone",             "first"),
        num_strikes_seen    = ("type",             lambda x: (x == "S").sum()),
        num_balls_seen      = ("type",             lambda x: (x == "B").sum()),
        swing_miss_pct      = ("description",      lambda x: (x.str.contains("swinging_strike", na=False)).mean()),
        foul_pct            = ("description",      lambda x: (x.str.contains("foul", na=False)).mean()),
    ).reset_index()

    # Terminal pitch per PA (contains the event)
    terminal = (
        df[df["events"].notna() & df["events"].isin(TERMINAL_EVENTS)]
        .sort_values("pitch_number")
        .groupby(pa_group)
        .last()
        .reset_index()
    )

    # Merge aggregates onto terminal
    pa = terminal.merge(agg, on=pa_group, how="left")

    # Add target variable
    pa["outcome"]      = pa["events"].apply(classify_outcome)
    pa["reached_base"] = (pa["outcome"] == "reach_base").astype(int)

    # Trim to useful columns that exist in this df
    desired = [
        # identifiers
        "game_pk", "game_date", "game_year", "at_bat_number",
        "batter", "pitcher", "player_name",
        # context
        "home_team", "away_team", "inning_topbot", "stand", "p_throws",
        # outcome
        "events", "outcome", "reached_base",
        # hit quality
        "launch_speed", "launch_angle", "hit_distance_sc",
        "estimated_ba_using_speedangle", "estimated_woba_using_speedangle",
        "woba_value", "woba_denom", "babip_value",
        "launch_speed_angle",
        # pitch sequence
        "pitch_count", "max_velo", "avg_velo", "pitches_seen",
        "first_pitch_type", "first_pitch_zone",
        "num_strikes_seen", "num_balls_seen",
        "swing_miss_pct", "foul_pct",
        # alignment
        "if_fielding_alignment", "of_fielding_alignment",
        # last pitch details
        "pitch_type", "pitch_name", "release_speed", "release_spin_rate",
        "release_extension", "pfx_x", "pfx_z", "plate_x", "plate_z",
        "zone", "effective_speed", "spin_axis",
        # score state (should be 0-0 for first AB, but keep for validation)
        "bat_score", "fld_score",
        # win exp delta
        "delta_home_win_exp", "delta_run_exp",
        # bat tracking (newer seasons)
        "bat_speed", "swing_length",
    ]
    present = [c for c in desired if c in pa.columns]
    return pa[present]


# Backward-compat alias
collapse_to_leadoff_pa = collapse_to_plate_appearances


def collapse_to_first_pa_per_batter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each batter's first plate appearance of the game (any inning, any outs).
    Uses n_priorpa_thisgame_player_at_bat == 0 when available (most accurate),
    otherwise falls back to minimum at_bat_number per game_pk + batter.

    Returns one row per (game_pk, batter) with the same feature set as
    collapse_to_plate_appearances plus first_pa_inning and first_pa_outs.
    Saves to data/processed/first_at_bats_all_batters.parquet.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    # Regular season only
    df = df[df["game_type"] == "R"].copy()
    if len(df) == 0:
        return pd.DataFrame()

    # Identify each batter's first PA of the game
    if "n_priorpa_thisgame_player_at_bat" in df.columns:
        df_first = df[pd.to_numeric(df["n_priorpa_thisgame_player_at_bat"], errors="coerce").fillna(1) == 0].copy()
    else:
        min_ab = (
            df.groupby(["game_pk", "batter"])["at_bat_number"]
            .min()
            .reset_index()
            .rename(columns={"at_bat_number": "_min_ab"})
        )
        df = df.merge(min_ab, on=["game_pk", "batter"])
        df_first = df[df["at_bat_number"] == df["_min_ab"]].copy()
        df_first = df_first.drop(columns=["_min_ab"])

    if len(df_first) == 0:
        return pd.DataFrame()

    pa_group = ["game_pk", "at_bat_number"]

    agg = df_first.groupby(pa_group).agg(
        pitch_count      = ("pitch_number",  "max"),
        max_velo         = ("release_speed",  "max"),
        avg_velo         = ("release_speed",  "mean"),
        pitches_seen     = ("pitch_type",     "count"),
        first_pitch_type = ("pitch_type",     "first"),
        first_pitch_zone = ("zone",           "first"),
        num_strikes_seen = ("type",           lambda x: (x == "S").sum()),
        num_balls_seen   = ("type",           lambda x: (x == "B").sum()),
        swing_miss_pct   = ("description",    lambda x: x.str.contains("swinging_strike", na=False).mean()),
        foul_pct         = ("description",    lambda x: x.str.contains("foul", na=False).mean()),
    ).reset_index()

    terminal = (
        df_first[df_first["events"].notna() & df_first["events"].isin(TERMINAL_EVENTS)]
        .sort_values("pitch_number")
        .groupby(pa_group)
        .last()
        .reset_index()
    )

    pa = terminal.merge(agg, on=pa_group, how="left")
    pa["outcome"]      = pa["events"].apply(classify_outcome)
    pa["reached_base"] = (pa["outcome"] == "reach_base").astype(int)

    # Extra context columns
    if "inning" in pa.columns:
        pa["first_pa_inning"] = pa["inning"]
    if "outs_when_up" in pa.columns:
        pa["first_pa_outs"] = pa["outs_when_up"]

    desired = [
        "game_pk", "game_date", "game_year", "at_bat_number",
        "batter", "pitcher", "player_name",
        "home_team", "away_team", "inning_topbot", "stand", "p_throws",
        "events", "outcome", "reached_base",
        "launch_speed", "launch_angle", "hit_distance_sc",
        "estimated_ba_using_speedangle", "estimated_woba_using_speedangle",
        "woba_value", "woba_denom", "babip_value", "launch_speed_angle",
        "pitch_count", "max_velo", "avg_velo", "pitches_seen",
        "first_pitch_type", "first_pitch_zone",
        "num_strikes_seen", "num_balls_seen", "swing_miss_pct", "foul_pct",
        "if_fielding_alignment", "of_fielding_alignment",
        "pitch_type", "pitch_name", "release_speed", "release_spin_rate",
        "release_extension", "pfx_x", "pfx_z", "plate_x", "plate_z",
        "zone", "effective_speed", "spin_axis",
        "bat_score", "fld_score", "delta_home_win_exp", "delta_run_exp",
        "bat_speed", "swing_length",
        "first_pa_inning", "first_pa_outs",
    ]
    present = [c for c in desired if c in pa.columns]
    return pa[present]


def _try_build_all_batters_tables(seasons: list[int]) -> None:
    """
    If statcast_full_{year}.csv exists for all requested seasons, build
    first_at_bats_all_batters.parquet and batter_fab_profile_all.parquet.
    Silently skips if any season file is missing (download still in progress).
    """
    all_frames = []
    for season in seasons:
        full_path = RAW_DIR / f"statcast_full_{season}.csv"
        if not full_path.exists():
            log.info("Full data for %d not yet downloaded — skipping all-batters build.", season)
            return
        log.info("Loading full data for %d ...", season)
        raw = pd.read_csv(full_path, low_memory=False)
        pa  = collapse_to_first_pa_per_batter(raw)
        log.info("  Season %d: %d first-PA rows", season, len(pa))
        if len(pa):
            all_frames.append(pa)

    if not all_frames:
        log.warning("No all-batters PA data — skipping table build.")
        return

    all_pa = pd.concat(all_frames, ignore_index=True)
    all_pa.drop_duplicates(subset=["game_pk", "batter"], inplace=True)

    # Park factors
    all_pa["park_run_factor"] = all_pa["home_team"].apply(
        lambda t: get_park_factor(t).get("run_factor", 100)
    )
    all_pa["park_hr_factor"] = all_pa["home_team"].apply(
        lambda t: get_park_factor(t).get("hr_factor", 100)
    )

    ab_path = PROC_DIR / "first_at_bats_all_batters.parquet"
    all_pa.to_parquet(ab_path, index=False)
    log.info("All-batters first PA: %d rows → %s", len(all_pa), ab_path)

    fab_all = build_first_ab_batter_profile(all_pa)
    if len(fab_all):
        fab_all_path = PROC_DIR / "batter_fab_profile_all.parquet"
        fab_all.to_parquet(fab_all_path, index=False)
        log.info("All-batters FAB profile: %d batters → %s", len(fab_all), fab_all_path)


# ── BvP career matchup aggregation ──────────────────────────────────────────

def build_bvp_table(all_pa: pd.DataFrame) -> pd.DataFrame:
    """
    Build a career batter-vs-pitcher matchup table.
    For each (batter, pitcher) pair, compute:
      PA, H, BB, K, HR, AVG, OBP, SLG, wOBA, xBA, xwOBA,
      avg_exit_velo, avg_launch_angle, hard_hit_pct, barrel_pct
    """
    if len(all_pa) == 0:
        return pd.DataFrame()

    df = all_pa.copy()

    # Hard hit = exit velo >= 95 mph
    df["hard_hit"]   = (df["launch_speed"] >= 95).astype(float)
    # Barrel = launch_speed_angle == 6
    df["barrel"]     = (df["launch_speed_angle"] == 6).astype(float)
    # Hit = single/double/triple/HR
    df["is_hit"]     = df["events"].isin({"single","double","triple","home_run"}).astype(float)
    df["is_hr"]      = (df["events"] == "home_run").astype(float)
    df["is_bb"]      = df["events"].isin({"walk","intent_walk"}).astype(float)
    df["is_k"]       = df["events"].isin({"strikeout","strikeout_double_play"}).astype(float)
    # OBP events
    df["on_base"]    = df["reached_base"].astype(float)

    bvp = df.groupby(["batter", "pitcher"]).agg(
        pa               = ("events",                            "count"),
        hits             = ("is_hit",                           "sum"),
        hr               = ("is_hr",                            "sum"),
        bb               = ("is_bb",                            "sum"),
        k                = ("is_k",                             "sum"),
        on_base_events   = ("on_base",                          "sum"),
        total_woba_value = ("woba_value",                       "sum"),
        total_woba_denom = ("woba_denom",                       "sum"),
        avg_xba          = ("estimated_ba_using_speedangle",    "mean"),
        avg_xwoba        = ("estimated_woba_using_speedangle",  "mean"),
        avg_exit_velo    = ("launch_speed",                     "mean"),
        avg_launch_angle = ("launch_angle",                     "mean"),
        hard_hit_sum     = ("hard_hit",                         "sum"),
        hard_hit_denom   = ("launch_speed",                     lambda x: x.notna().sum()),
        barrel_sum       = ("barrel",                           "sum"),
        barrel_denom     = ("launch_speed_angle",               lambda x: x.notna().sum()),
        last_game_date   = ("game_date",                        "max"),
    ).reset_index()

    bvp["avg"]          = bvp["hits"]           / bvp["pa"].clip(lower=1)
    bvp["obp"]          = bvp["on_base_events"] / bvp["pa"].clip(lower=1)
    bvp["k_pct"]        = bvp["k"]              / bvp["pa"].clip(lower=1)
    bvp["bb_pct"]       = bvp["bb"]             / bvp["pa"].clip(lower=1)
    bvp["woba"]         = bvp["total_woba_value"] / bvp["total_woba_denom"].clip(lower=1)
    bvp["hard_hit_pct"] = bvp["hard_hit_sum"]   / bvp["hard_hit_denom"].clip(lower=1)
    bvp["barrel_pct"]   = bvp["barrel_sum"]     / bvp["barrel_denom"].clip(lower=1)

    # Reliability flag — BvP is noisy below ~15 PA
    bvp["bvp_reliable"] = bvp["pa"] >= 15

    return bvp


# ── Rolling batter stats ─────────────────────────────────────────────────────

def build_batter_rolling(all_pa: pd.DataFrame) -> pd.DataFrame:
    """
    For each (batter, game_date), compute rolling 7/14/30-day stats
    up to (but not including) that game.
    Vectorized via groupby + time-based rolling with closed='left' to
    exclude the current game from its own window.
    """
    if len(all_pa) == 0:
        return pd.DataFrame()

    df = all_pa.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    df["is_hit"]   = df["events"].isin({"single","double","triple","home_run"}).astype(float)
    df["is_k"]     = df["events"].isin({"strikeout","strikeout_double_play"}).astype(float)
    df["is_bb"]    = df["events"].isin({"walk","intent_walk"}).astype(float)
    df["hard_hit"] = (df["launch_speed"] >= 95).astype(float)

    metrics = [
        ("obp",      "reached_base"),
        ("k_pct",    "is_k"),
        ("bb_pct",   "is_bb"),
        ("hit_pct",  "is_hit"),
        ("hard_hit", "hard_hit"),
        ("xba",      "estimated_ba_using_speedangle"),
        ("xwoba",    "estimated_woba_using_speedangle"),
    ]
    windows = [(7, "7d"), (14, "14d"), (30, "30d")]

    # Sort by batter then date — required for time-based groupby rolling
    df = df.sort_values(["batter", "game_date"]).reset_index(drop=True)
    df_indexed = df.set_index("game_date")

    for metric_name, col in metrics:
        if col not in df_indexed.columns:
            continue
        for window_days, label in windows:
            # closed='left' → window is [t-N days, t) — excludes current game
            rolled = (
                df_indexed.groupby("batter")[col]
                .rolling(f"{window_days}D", min_periods=1, closed="left")
                .mean()
            )
            # result has MultiIndex (batter, game_date); .values aligns with df rows
            df[f"{metric_name}_{label}"] = rolled.values

    out_cols = ["batter", "game_date"]
    for metric_name, _ in metrics:
        for _, label in windows:
            col = f"{metric_name}_{label}"
            if col in df.columns:
                out_cols.append(col)

    result = df[out_cols].copy()
    rolling_cols = [c for c in out_cols if c not in ("batter", "game_date")]
    # Drop rows with no prior history across all windows
    result = result.dropna(subset=rolling_cols, how="all")
    return result


# ── Pitcher first-inning splits ──────────────────────────────────────────────

def build_pitcher_splits(all_pa: pd.DataFrame) -> pd.DataFrame:
    """
    For each pitcher, compute:
    - Season-level home vs. away splits
    - First-inning specific performance metrics
    """
    if len(all_pa) == 0:
        return pd.DataFrame()

    df = all_pa.copy()
    df["is_home_pitcher"] = (
        (df["inning_topbot"] == "Bot") == True  # pitcher is home team when batting team is away (top)
    )
    # Actually: inning_topbot='Top' means away team bats → home pitcher on mound
    df["is_home_pitcher"] = df["inning_topbot"] == "Top"

    splits = df.groupby(["pitcher", "game_year", "is_home_pitcher"]).agg(
        pa               = ("events",       "count"),
        reach_base_count = ("reached_base", "sum"),
        k_count          = ("events",       lambda x: x.isin({"strikeout","strikeout_double_play"}).sum()),
        bb_count         = ("events",       lambda x: x.isin({"walk","intent_walk"}).sum()),
        avg_exit_velo    = ("launch_speed", "mean"),
        avg_xba          = ("estimated_ba_using_speedangle", "mean"),
    ).reset_index()

    splits["obp_allowed"]  = splits["reach_base_count"] / splits["pa"].clip(lower=1)
    splits["k_pct"]        = splits["k_count"]           / splits["pa"].clip(lower=1)
    splits["bb_pct"]       = splits["bb_count"]           / splits["pa"].clip(lower=1)
    splits["location"]     = splits["is_home_pitcher"].map({True: "home", False: "away"})

    return splits.drop(columns=["is_home_pitcher"])


# ── First-at-bat batter profile ──────────────────────────────────────────────

def build_first_ab_batter_profile(all_pa: pd.DataFrame = None) -> pd.DataFrame:
    """
    Aggregate per-batter first-at-bat profile stats from first_at_bats.parquet.

    Career splits  — fab_*  columns: overall, vs RHP/LHP, home/away, last-30
    Current season — seas_fab_* columns: overall, vs RHP/LHP, home/away

    Full stat set for every split row: PA, H, AB, XBH, AVG, OBP, SLG,
    K%, BB%, Hard-Hit%, xBA, wOBA where available.

    Returns one row per batter.
    """
    if all_pa is None:
        path = PROC_DIR / "first_at_bats.parquet"
        if not path.exists():
            log.warning("first_at_bats.parquet not found — cannot build FAB profile")
            return pd.DataFrame()
        all_pa = pd.read_parquet(path)

    if len(all_pa) == 0:
        return pd.DataFrame()

    df = all_pa.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    cur_yr = int(df["game_date"].dt.year.max())

    # ── Derived indicators ────────────────────────────────────────────────────
    df["is_hit"]    = df["events"].isin({"single","double","triple","home_run"}).astype(float)
    df["is_k"]      = df["events"].isin({"strikeout","strikeout_double_play"}).astype(float)
    df["is_bb"]     = df["events"].isin({"walk","intent_walk"}).astype(float)
    df["is_hbp"]    = (df["events"] == "hit_by_pitch").astype(float)
    df["is_sf"]     = df["events"].isin({"sac_fly","sac_fly_double_play"}).astype(float)
    df["is_sh"]     = (df["events"] == "sac_bunt").astype(float)
    df["is_double"] = (df["events"] == "double").astype(float)
    df["is_triple"] = (df["events"] == "triple").astype(float)
    df["is_hr"]     = (df["events"] == "home_run").astype(float)
    df["is_ab"]     = (1 - df["is_bb"] - df["is_hbp"] - df["is_sf"] - df["is_sh"]).clip(lower=0)
    df["tb"]        = ((df["events"] == "single").astype(float) +
                       2*df["is_double"] + 3*df["is_triple"] + 4*df["is_hr"])
    df["is_xbh"]    = df["is_double"] + df["is_triple"] + df["is_hr"]
    df["hard_hit"]  = ((df["launch_speed"] >= 95) & df["launch_speed"].notna()).astype(float)
    df["ls_valid"]  = df["launch_speed"].notna().astype(float)
    df["is_home"]   = df["inning_topbot"] == "Bot"   # Bot = home team batting

    def _fab_stats(sub, pfx):
        """Aggregate all FAB stats for a subset. Returns batter + columns prefixed pfx."""
        if not len(sub):
            return pd.DataFrame(columns=["batter"])
        g = sub.groupby("batter", as_index=False).agg(
            _pa  = ("reached_base", "count"),
            _ob  = ("reached_base", "sum"),
            _h   = ("is_hit",       "sum"),
            _ab  = ("is_ab",        "sum"),
            _xbh = ("is_xbh",       "sum"),
            _tb  = ("tb",           "sum"),
            _k   = ("is_k",         "sum"),
            _bb  = ("is_bb",        "sum"),
            _hhs = ("hard_hit",     "sum"),
            _hhd = ("ls_valid",     "sum"),
            _xba = ("estimated_ba_using_speedangle", "mean"),
            _wv  = ("woba_value",   "sum"),
            _wd  = ("woba_denom",   "sum"),
        )
        g[pfx+"pa"]       = g["_pa"]
        g[pfx+"obp"]      = g["_ob"]  / g["_pa"].clip(lower=1)
        g[pfx+"avg"]      = g["_h"]   / g["_ab"].clip(lower=1)
        g[pfx+"slg"]      = g["_tb"]  / g["_ab"].clip(lower=1)
        g[pfx+"k_pct"]    = g["_k"]   / g["_pa"].clip(lower=1)
        g[pfx+"bb_pct"]   = g["_bb"]  / g["_pa"].clip(lower=1)
        g[pfx+"hard_hit"] = g["_hhs"] / g["_hhd"].clip(lower=1)
        g[pfx+"xba"]      = g["_xba"]
        g[pfx+"woba"]     = g["_wv"]  / g["_wd"].clip(lower=1)
        g[pfx+"hits"]     = g["_h"].round().astype(int)
        g[pfx+"ab"]       = g["_ab"].round().astype(int)
        g[pfx+"xbh"]      = g["_xbh"].round().astype(int)
        keep = ["batter"] + [c for c in g.columns if c.startswith(pfx)]
        return g[keep]

    def _split_rename(base_df, stat_pfx, qual_sfx):
        """
        Rename columns from '_fab_stats(sub, stat_pfx)' to the
        split-qualified form used elsewhere: fab_<stat>_<qual>.
        E.g. stat_pfx="fab_", qual_sfx="vs_rhp" →
             fab_pa → fab_pa_vs_rhp, fab_obp → fab_obp_vs_rhp ...
        """
        rename = {}
        for c in base_df.columns:
            if c == "batter":
                continue
            stat = c[len(stat_pfx):]          # strip prefix → "pa", "obp" …
            rename[c] = f"{stat_pfx}{stat}_{qual_sfx}"
        return base_df.rename(columns=rename)

    # ── 1. Career overall ─────────────────────────────────────────────────────
    ov = _fab_stats(df, "fab_")

    # ── 2. Career vs RHP ─────────────────────────────────────────────────────
    rhp = _split_rename(_fab_stats(df[df["p_throws"] == "R"], "fab_"), "fab_", "vs_rhp")

    # ── 3. Career vs LHP ─────────────────────────────────────────────────────
    lhp = _split_rename(_fab_stats(df[df["p_throws"] == "L"], "fab_"), "fab_", "vs_lhp")

    # ── 4. Career home / away ─────────────────────────────────────────────────
    home = _split_rename(_fab_stats(df[df["is_home"]],  "fab_"), "fab_", "home")
    away = _split_rename(_fab_stats(df[~df["is_home"]], "fab_"), "fab_", "away")

    # ── 5. Last 30 first ABs (most-recent 30 PAs regardless of date) ─────────
    l30_sub = (
        df.sort_values("game_date")
        .groupby("batter", group_keys=False)
        .tail(30)
    )
    l30_raw = _fab_stats(l30_sub, "fab_")
    l30 = l30_raw.rename(columns={
        c: c.replace("fab_", "fab_") if c == "batter"
           else f"fab_{c[len('fab_'):]}_last30"
        for c in l30_raw.columns
    })

    # ── 6. Current-season overall (seas_fab_*) ────────────────────────────────
    cur = df[df["game_date"].dt.year == cur_yr]
    seas_ov   = _fab_stats(cur, "seas_fab_")
    seas_rhp  = _split_rename(_fab_stats(cur[cur["p_throws"] == "R"], "seas_fab_"), "seas_fab_", "vs_rhp")
    seas_lhp  = _split_rename(_fab_stats(cur[cur["p_throws"] == "L"], "seas_fab_"), "seas_fab_", "vs_lhp")
    seas_home = _split_rename(_fab_stats(cur[cur["is_home"]],  "seas_fab_"), "seas_fab_", "home")
    seas_away = _split_rename(_fab_stats(cur[~cur["is_home"]], "seas_fab_"), "seas_fab_", "away")

    # ── Merge all splits ──────────────────────────────────────────────────────
    profile = ov.copy()
    for sdf in [rhp, lhp, home, away, l30,
                seas_ov, seas_rhp, seas_lhp, seas_home, seas_away]:
        if sdf.empty or len(sdf.columns) < 2:
            continue
        profile = profile.merge(sdf, on="batter", how="left")

    # Small-sample flag for current season (fewer than 10 first-ABs)
    if "seas_fab_pa" in profile.columns:
        profile["seas_fab_small_sample"] = profile["seas_fab_pa"].fillna(0) < 10
    else:
        profile["seas_fab_small_sample"] = True

    # ── Composite delta stats (keep for model features) ───────────────────────
    def _col(name, fallback):
        return profile[name] if name in profile.columns else profile[fallback]

    profile["fab_platoon_split"]   = (
        _col("fab_obp_vs_rhp", "fab_obp").fillna(profile["fab_obp"]) -
        _col("fab_obp_vs_lhp", "fab_obp").fillna(profile["fab_obp"])
    )
    profile["fab_home_away_split"] = (
        _col("fab_obp_home", "fab_obp").fillna(profile["fab_obp"]) -
        _col("fab_obp_away", "fab_obp").fillna(profile["fab_obp"])
    )
    # fab_obp_last30 comes from the l30 rename: fab_obp_last30
    if "fab_obp_last30" in profile.columns:
        profile["fab_trend"] = profile["fab_obp_last30"].fillna(profile["fab_obp"]) - profile["fab_obp"]
    else:
        profile["fab_trend"] = 0.0

    log.info("First-AB batter profile: %d batters, career + %d season splits built",
             len(profile), cur_yr)
    return profile


# ── Park factors ─────────────────────────────────────────────────────────────

# Static park factor table (run factor, HR factor) — 5-year averages from FanGraphs
# 100 = league average; >100 = hitter-friendly
PARK_FACTORS = {
    "COL": {"run_factor": 112, "hr_factor": 118, "lat": 39.756, "lon": -104.994},
    "CIN": {"run_factor": 108, "hr_factor": 113, "lat": 39.097, "lon": -84.507},
    "TEX": {"run_factor": 106, "hr_factor": 108, "lat": 32.751, "lon": -97.083},
    "PHI": {"run_factor": 103, "hr_factor": 105, "lat": 39.906, "lon": -75.167},
    "BOS": {"run_factor": 103, "hr_factor": 100, "lat": 42.347, "lon": -71.097},
    "CHC": {"run_factor": 102, "hr_factor": 104, "lat": 41.948, "lon": -87.656},
    "MIL": {"run_factor": 102, "hr_factor": 107, "lat": 43.028, "lon": -87.971},
    "HOU": {"run_factor": 101, "hr_factor": 100, "lat": 29.757, "lon": -95.356},
    "ARI": {"run_factor": 101, "hr_factor": 103, "lat": 33.445, "lon": -112.067},
    "BAL": {"run_factor": 101, "hr_factor": 103, "lat": 39.284, "lon": -76.622},
    "NYY": {"run_factor": 100, "hr_factor": 105, "lat": 40.829, "lon": -73.926},
    "MIN": {"run_factor": 100, "hr_factor": 101, "lat": 44.982, "lon": -93.278},
    "LAA": {"run_factor": 100, "hr_factor": 100, "lat": 33.800, "lon": -117.883},
    "ATL": {"run_factor":  99, "hr_factor":  99, "lat": 33.891, "lon": -84.468},
    "TOR": {"run_factor":  99, "hr_factor":  97, "lat": 43.641, "lon": -79.389},
    "CLE": {"run_factor":  98, "hr_factor":  96, "lat": 41.496, "lon": -81.685},
    "DET": {"run_factor":  98, "hr_factor":  96, "lat": 42.339, "lon": -83.049},
    "WSH": {"run_factor":  98, "hr_factor":  96, "lat": 38.873, "lon": -77.007},
    "TB":  {"run_factor":  97, "hr_factor":  95, "lat": 27.768, "lon": -82.653},
    "CHW": {"run_factor":  97, "hr_factor":  97, "lat": 41.830, "lon": -87.634},
    "PIT": {"run_factor":  97, "hr_factor":  96, "lat": 40.447, "lon": -80.006},
    "CWS": {"run_factor":  97, "hr_factor":  97, "lat": 41.830, "lon": -87.634},
    "MIA": {"run_factor":  96, "hr_factor":  94, "lat": 25.778, "lon": -80.220},
    "NYM": {"run_factor":  96, "hr_factor":  95, "lat": 40.757, "lon": -73.846},
    "KC":  {"run_factor":  96, "hr_factor":  95, "lat": 39.052, "lon": -94.481},
    "STL": {"run_factor":  96, "hr_factor":  95, "lat": 38.623, "lon": -90.193},
    "SDP": {"run_factor":  95, "hr_factor":  93, "lat": 32.707, "lon": -117.157},
    "SD":  {"run_factor":  95, "hr_factor":  93, "lat": 32.707, "lon": -117.157},
    "SEA": {"run_factor":  94, "hr_factor":  91, "lat": 47.591, "lon": -122.332},
    "OAK": {"run_factor":  94, "hr_factor":  92, "lat": 37.751, "lon": -122.201},
    "LAD": {"run_factor":  93, "hr_factor":  91, "lat": 34.074, "lon": -118.240},
    "SFG": {"run_factor":  92, "hr_factor":  88, "lat": 37.778, "lon": -122.389},
    "SF":  {"run_factor":  92, "hr_factor":  88, "lat": 37.778, "lon": -122.389},
}


def get_park_factor(team_code: str) -> dict:
    return PARK_FACTORS.get(team_code, {"run_factor": 100, "hr_factor": 100, "lat": None, "lon": None})


# ── Batter pitch-type hitting profile ─────────────────────────────────────────

_FB_CODES  = {"FF", "SI", "FC", "FT"}
_BB_CODES  = {"SL", "CU", "KC", "ST", "SV", "CS"}
_OS_CODES  = {"CH", "FS", "FO"}
_SWINGS_PT = {"swinging_strike", "swinging_strike_blocked", "foul_tip",
              "foul", "foul_bunt", "hit_into_play", "hit_into_play_score",
              "hit_into_play_no_out", "missed_bunt"}
_WHIFFS_PT = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
_BIP_PT    = {"single", "double", "triple", "home_run",
              "field_out", "force_out", "grounded_into_double_play",
              "double_play", "field_error", "fielders_choice", "fielders_choice_out",
              "sac_fly", "sac_bunt"}
_HIT_PT    = {"single", "double", "triple", "home_run"}
_OB_PT     = {"single", "double", "triple", "home_run",
              "walk", "intent_walk", "hit_by_pitch", "field_error", "catcher_interf"}


def _batter_pitch_stats(raw: pd.DataFrame, pitch_set: set, prefix: str) -> pd.DataFrame:
    """Compute per-batter stats for pitches in pitch_set."""
    df = raw[raw["pitch_type"].isin(pitch_set)].copy()
    if df.empty:
        return pd.DataFrame(columns=["batter",
                                     f"{prefix}_pa", f"{prefix}_obp", f"{prefix}_avg",
                                     f"{prefix}_whiff_pct", f"{prefix}_hard_hit",
                                     f"{prefix}_xba"])

    df["_is_swing"] = df["description"].isin(_SWINGS_PT)
    df["_is_whiff"] = df["description"].isin(_WHIFFS_PT)
    df["_hard_hit"] = (df["launch_speed"] >= 95) & df["launch_speed"].notna()
    df["_ls_valid"] = df["launch_speed"].notna()

    term = df[df["events"].notna()]
    term = term.copy()
    term["_is_ob"]  = term["events"].isin(_OB_PT)
    term["_is_hit"] = term["events"].isin(_HIT_PT)
    term["_is_bip"] = term["events"].isin(_BIP_PT)

    pa_stats = term.groupby("batter").agg(
        _pa    = ("events",  "count"),
        _ob    = ("_is_ob",  "sum"),
        _hit   = ("_is_hit", "sum"),
        _xba   = ("estimated_ba_using_speedangle", "mean"),
    ).reset_index()
    pa_stats[f"{prefix}_pa"]  = pa_stats["_pa"]
    pa_stats[f"{prefix}_obp"] = pa_stats["_ob"]  / pa_stats["_pa"].clip(1)
    pa_stats[f"{prefix}_avg"] = pa_stats["_hit"] / pa_stats["_pa"].clip(1)
    pa_stats[f"{prefix}_xba"] = pa_stats["_xba"]

    swing_stats = df.groupby("batter").agg(
        _sw   = ("_is_swing", "sum"),
        _wh   = ("_is_whiff", "sum"),
        _hh   = ("_hard_hit", "sum"),
        _lsv  = ("_ls_valid", "sum"),
    ).reset_index()
    swing_stats[f"{prefix}_whiff_pct"] = swing_stats["_wh"] / swing_stats["_sw"].clip(1)
    swing_stats[f"{prefix}_hard_hit"]  = swing_stats["_hh"] / swing_stats["_lsv"].clip(1)

    out_cols = ["batter", f"{prefix}_pa", f"{prefix}_obp", f"{prefix}_avg",
                f"{prefix}_xba"]
    result = pa_stats[out_cols].merge(
        swing_stats[["batter", f"{prefix}_whiff_pct", f"{prefix}_hard_hit"]],
        on="batter", how="left"
    )
    return result


def build_batter_pitch_type_profile(seasons: list = None) -> pd.DataFrame:
    """
    Aggregate batter pitch-type hitting splits from raw Statcast CSVs.
    Uses inning-1 leadoff PA data (all available seasons).
    Computes hitting performance vs Fastballs, Breaking Balls, and Offspeed.

    Returns one row per batter with columns batter_fb_*, batter_bb_*, batter_os_*.
    Saves to data/processed/batter_pitch_type_profile.parquet.
    """
    if seasons is None:
        seasons = list(range(2015, datetime.now().year + 1))

    parts = []
    for yr in seasons:
        p = RAW_DIR / f"statcast_{yr}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, low_memory=False)
        df = df[df["game_type"] == "R"]
        parts.append(df)

    if not parts:
        log.warning("No raw CSVs found for batter pitch-type profile")
        return pd.DataFrame()

    raw = pd.concat(parts, ignore_index=True)
    log.info("Batter pitch-type profile: %d pitches loaded", len(raw))

    fb  = _batter_pitch_stats(raw, _FB_CODES,  "batter_fb")
    bb  = _batter_pitch_stats(raw, _BB_CODES,  "batter_bb")
    os_ = _batter_pitch_stats(raw, _OS_CODES,  "batter_os")

    profile = fb.merge(bb, on="batter", how="outer").merge(os_, on="batter", how="outer")

    # Vulnerability composites
    profile["batter_fb_vs_bb_diff"] = (
        profile["batter_fb_obp"].fillna(0) - profile["batter_bb_obp"].fillna(0)
    )

    def _weakness(row):
        fw = row.get("batter_fb_whiff_pct", 0) or 0
        bw = row.get("batter_bb_whiff_pct", 0) or 0
        ow = row.get("batter_os_whiff_pct", 0) or 0
        if pd.isna(fw): fw = 0
        if pd.isna(bw): bw = 0
        if pd.isna(ow): ow = 0
        m = max(fw, bw, ow)
        if m == bw: return "Breaking Ball"
        if m == ow: return "Offspeed"
        return "Fastball"

    profile["batter_biggest_weakness"] = profile.apply(_weakness, axis=1)

    out = PROC_DIR / "batter_pitch_type_profile.parquet"
    profile.to_parquet(out, index=False)
    log.info("Batter pitch-type profile: %d batters → %s", len(profile), out)
    return profile


# ── Full lifetime BvP (all innings) ─────────────────────────────────────────

# Minimal columns needed to build a BvP table from full-season Statcast data
_BVP_FETCH_COLS = [
    "batter", "pitcher", "events", "game_type", "game_date",
    "launch_speed", "launch_angle", "launch_speed_angle",
    "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle", "woba_value", "woba_denom",
]


def build_full_bvp_table(seasons: list = None) -> pd.DataFrame:
    """
    Build a full lifetime batter-vs-pitcher table covering ALL innings and ALL
    game situations (not just first-inning leadoff).

    Strategy:
      1. For each season, use pybaseball.statcast() to pull all regular-season
         pitches — no inning or outs filter.
      2. Keep only terminal events (one row per plate appearance) using the
         existing TERMINAL_EVENTS set.
      3. Cache the compact terminal-event parquet per season to
         data/raw/statcast_bvp_full_{year}.parquet so re-runs are instant.
      4. Aggregate with the existing build_bvp_table() function.
      5. Save results to:
           data/processed/bvp_full_lifetime.parquet   ← primary output
           data/processed/bvp_first_inning.parquet    ← copy of original for
                                                          comparison (created once)
    """
    import shutil

    if seasons is None:
        seasons = list(range(2015, datetime.now().year + 1))

    try:
        import pybaseball as pb
        pb.cache.enable()
    except ImportError:
        log.error("pybaseball not installed — run:  pip install pybaseball")
        return pd.DataFrame()

    all_frames = []
    for season in seasons:
        cache_path = RAW_DIR / f"statcast_bvp_full_{season}.parquet"

        if cache_path.exists():
            log.info("  [%d] loading from cache → %s", season, cache_path.name)
            pa_df = pd.read_parquet(cache_path)
        else:
            log.info("  [%d] fetching via pybaseball (all innings, parallel)…", season)
            start = f"{season}-03-01"
            end   = f"{season}-11-30"
            try:
                raw = pb.statcast(start_dt=start, end_dt=end,
                                  verbose=True, parallel=True)
            except Exception as exc:
                log.error("  [%d] pybaseball fetch failed: %s", season, exc)
                continue

            if raw is None or len(raw) == 0:
                log.warning("  [%d] no data returned, skipping", season)
                continue

            # Keep only regular-season terminal events; slim to needed columns
            avail  = [c for c in _BVP_FETCH_COLS if c in raw.columns]
            pa_df  = raw[
                (raw["game_type"] == "R") &
                raw["events"].notna() &
                raw["events"].isin(TERMINAL_EVENTS)
            ][avail].copy()

            pa_df["game_date"] = pd.to_datetime(pa_df["game_date"], errors="coerce")
            pa_df.to_parquet(cache_path, index=False)
            log.info("  [%d] cached %d terminal PAs → %s",
                     season, len(pa_df), cache_path.name)

        if len(pa_df):
            all_frames.append(pa_df)

    if not all_frames:
        log.error("No data collected — cannot build full lifetime BvP table.")
        return pd.DataFrame()

    all_pa = pd.concat(all_frames, ignore_index=True)
    log.info("Total full-career terminal PAs: %d across %d seasons",
             len(all_pa), len(all_frames))

    # build_bvp_table() requires a reached_base column
    all_pa["outcome"]      = all_pa["events"].apply(classify_outcome)
    all_pa["reached_base"] = (all_pa["outcome"] == "reach_base").astype(int)

    bvp = build_bvp_table(all_pa)

    # ── Save outputs ───────────────────────────────────────────────────────────
    bvp_full_path = PROC_DIR / "bvp_full_lifetime.parquet"
    bvp.to_parquet(bvp_full_path, index=False)
    log.info("Saved full lifetime BvP: %d pairs → %s", len(bvp), bvp_full_path)

    # Preserve first-inning-only version for comparison (copy once, don't overwrite)
    bvp_orig_path = PROC_DIR / "bvp_matchups.parquet"
    bvp_fi_path   = PROC_DIR / "bvp_first_inning.parquet"
    if bvp_orig_path.exists() and not bvp_fi_path.exists():
        shutil.copy2(bvp_orig_path, bvp_fi_path)
        log.info("Preserved first-inning BvP → %s", bvp_fi_path)

    # Print summary
    log.info("=" * 60)
    log.info("Full lifetime BvP summary:")
    log.info("  Total pairs  : %d", len(bvp))
    log.info("  Mean PA      : %.1f", bvp["pa"].mean())
    log.info("  Median PA    : %.1f", bvp["pa"].median())
    log.info("  Max PA       : %d",   bvp["pa"].max())
    log.info("  Pairs ≥ 5 PA : %d",  (bvp["pa"] >= 5).sum())
    log.info("  Pairs ≥ 15 PA: %d",  (bvp["pa"] >= 15).sum())
    log.info("=" * 60)

    return bvp


# ── Pipeline state ────────────────────────────────────────────────────────────

def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def _save_state(seasons: list, all_pa, bvp, splits, rolling) -> None:
    """Record processed seasons and fingerprint each CSV so freshness can be checked later."""
    now = datetime.now().isoformat(timespec="seconds")
    fingerprints = {}
    for season in seasons:
        p = RAW_DIR / f"statcast_{season}.csv"
        if p.exists():
            s = p.stat()
            fingerprints[str(season)] = {"size": s.st_size, "mtime": s.st_mtime}
    state = {
        "processed_seasons": sorted(seasons),
        "last_built": {k: now for k in
                       ["first_at_bats", "bvp_matchups", "pitcher_splits", "batter_rolling"]},
        "row_counts": {
            "first_at_bats":  len(all_pa),
            "bvp_matchups":   len(bvp),
            "pitcher_splits": len(splits),
            "batter_rolling": len(rolling),
        },
        "statcast_fingerprints": fingerprints,
    }
    STATE_PATH.write_text(json.dumps(state, indent=2))
    log.info("Pipeline state saved → %s", STATE_PATH)


def _historical_is_fresh(seasons: list) -> bool:
    """
    Returns True when all four derived parquets exist, the same seasons were
    last processed, and no statcast CSV has grown since the last build.
    A larger CSV means new pitch data was appended for an ongoing season.
    """
    state = _load_state()
    if not state:
        return False

    required = ["first_at_bats.parquet", "bvp_matchups.parquet",
                "pitcher_splits.parquet", "batter_rolling.parquet"]
    if not all((PROC_DIR / f).exists() for f in required):
        return False

    if set(state.get("processed_seasons", [])) != set(seasons):
        return False

    recorded = state.get("statcast_fingerprints", {})
    for season in seasons:
        csv_path = RAW_DIR / f"statcast_{season}.csv"
        if not csv_path.exists():
            return False
        key = str(season)
        if key not in recorded:
            return False
        if csv_path.stat().st_size != recorded[key]["size"]:
            return False  # file grew — new pitch data available

    return True


# ── Master runner ─────────────────────────────────────────────────────────────

def run(seasons: list[int] = None, rebuild_historical: bool = False):
    if seasons is None:
        seasons = SEASONS

    # Phase 1 — Ensure leadoff CSVs on disk (statcast_{year}.csv)
    raw_by_season = {}
    for season in seasons:
        log.info("=" * 60)
        log.info("Processing season %d", season)
        raw = fetch_season(season, chunk_months=True)
        if raw is not None and len(raw):
            raw_by_season[season] = raw
        else:
            log.warning("Skipping season %d — no data returned", season)

    # Phase 1b — Download full innings 1-5 CSVs (statcast_full_{year}.csv)
    # This runs even when leadoff tables are fresh so the overnight job makes progress.
    log.info("Checking full-data CSV cache (innings 1-5)...")
    for season in seasons:
        full_path = RAW_DIR / f"statcast_full_{season}.csv"
        if not full_path.exists():
            log.info("Downloading full data for season %d ...", season)
            fetch_season_full(season, chunk_months=True)
        else:
            log.info("  Full data for %d already cached.", season)

    # Phase 2 — Skip the heavy leadoff rebuild if nothing has changed on disk
    if not rebuild_historical and _historical_is_fresh(seasons):
        state = _load_state()
        log.info("Historical leadoff tables are current — skipping rebuild.")
        log.info("  Last built : %s",
                 state.get("last_built", {}).get("first_at_bats", "unknown"))
        for k, v in state.get("row_counts", {}).items():
            log.info("    %-20s %s rows", k + ":", f"{v:,}")
        # Still try to build all-batters tables if full CSVs were just downloaded
        _try_build_all_batters_tables(seasons)
        return

    if rebuild_historical:
        log.info("--rebuild-historical: forcing full rebuild of all derived tables.")

    # Phase 3 — Collapse pitch rows → plate appearances
    all_frames = []
    for season in sorted(raw_by_season):
        pa = collapse_to_plate_appearances(raw_by_season[season])
        log.info("  Season %d: collapsed to %d plate appearances", season, len(pa))
        if len(pa):
            all_frames.append(pa)

    if not all_frames:
        log.error("No data collected. Check your network connection.")
        return

    all_pa = pd.concat(all_frames, ignore_index=True)
    all_pa.drop_duplicates(subset=["game_pk", "at_bat_number"], inplace=True)

    # Add park factors
    all_pa["park_run_factor"] = all_pa["home_team"].apply(
        lambda t: get_park_factor(t).get("run_factor", 100)
    )
    all_pa["park_hr_factor"] = all_pa["home_team"].apply(
        lambda t: get_park_factor(t).get("hr_factor", 100)
    )

    log.info("Total first-inning leadoff PAs: %d", len(all_pa))

    # Save processed first at-bats
    out_path = PROC_DIR / "first_at_bats.parquet"
    all_pa.to_parquet(out_path, index=False)
    log.info("Saved: %s", out_path)

    # Build and save BvP table
    log.info("Building batter-vs-pitcher table...")
    bvp = build_bvp_table(all_pa)
    bvp_path = PROC_DIR / "bvp_matchups.parquet"
    bvp.to_parquet(bvp_path, index=False)
    log.info("Saved BvP table: %d matchup pairs → %s", len(bvp), bvp_path)

    # Build and save pitcher splits
    log.info("Building pitcher splits...")
    splits = build_pitcher_splits(all_pa)
    splits_path = PROC_DIR / "pitcher_splits.parquet"
    splits.to_parquet(splits_path, index=False)
    log.info("Saved pitcher splits → %s", splits_path)

    # Build rolling batter stats
    log.info("Building rolling batter trends...")
    rolling = build_batter_rolling(all_pa)
    rolling_path = PROC_DIR / "batter_rolling.parquet"
    if len(rolling):
        rolling.to_parquet(rolling_path, index=False)
        log.info("Saved batter rolling stats → %s", rolling_path)

    # Build batter pitch-type hitting profile
    log.info("Building batter pitch-type profile...")
    build_batter_pitch_type_profile(seasons)

    # Build first-AB batter profile (overall + handedness + home/away + recent form)
    log.info("Building first-AB batter profile...")
    fab_profile = build_first_ab_batter_profile(all_pa)
    if len(fab_profile):
        fab_path = PROC_DIR / "batter_fab_profile.parquet"
        fab_profile.to_parquet(fab_path, index=False)
        log.info("Saved FAB profile: %d batters → %s", len(fab_profile), fab_path)
        # Also save as explicit leadoff version
        fab_lead_path = PROC_DIR / "batter_fab_profile_leadoff.parquet"
        fab_profile.to_parquet(fab_lead_path, index=False)
        log.info("Saved leadoff FAB profile → %s", fab_lead_path)

    # Save explicit leadoff PA alias
    leadoff_path = PROC_DIR / "first_at_bats_leadoff.parquet"
    all_pa.to_parquet(leadoff_path, index=False)
    log.info("Saved leadoff alias → %s", leadoff_path)

    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info("  first_at_bats:  %s rows", len(all_pa))
    log.info("  bvp_matchups:   %s rows", len(bvp))
    log.info("  pitcher_splits: %s rows", len(splits))
    if len(rolling):
        log.info("  batter_rolling: %s rows", len(rolling))

    # Persist state so the next run can detect whether a rebuild is needed
    _save_state(seasons, all_pa, bvp, splits, rolling)

    # Build all-batters tables if full CSVs are now available
    _try_build_all_batters_tables(seasons)


if __name__ == "__main__":
    import sys
    rebuild = "--rebuild-historical" in sys.argv
    if len(sys.argv) > 1:
        seasons = [int(y) for y in sys.argv[1:] if y.lstrip("-").isdigit()]
    else:
        seasons = SEASONS
    run(seasons, rebuild_historical=rebuild)
