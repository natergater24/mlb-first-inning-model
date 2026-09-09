"""
02_fetch_mlb_api.py
-------------------
Pulls from the free MLB Stats API (statsapi.mlb.com):
  - Full regular-season schedule (2015–present)
  - Probable pitchers for each game
  - Starting lineups (via game feed)
  - Home plate umpire assignments
  - Player metadata (handedness, position, DOB)

No API key required. Be respectful — we add 0.5s delay between requests.

Output:
  data/raw/mlb_schedule_{year}.json     raw schedule JSON
  data/processed/game_meta.parquet      one row per game with pitchers, umpire
  data/processed/player_meta.parquet    player handedness, DOB, position
  data/processed/lineup_first_batter.parquet  confirmed leadoff batters per game
"""

import os
import json
import time
import logging
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, date

BASE_DIR  = Path(__file__).resolve().parent.parent
RAW_DIR   = BASE_DIR / "data" / "raw"
PROC_DIR  = BASE_DIR / "data" / "processed"
LOG_DIR   = BASE_DIR / "logs"

for d in [RAW_DIR, PROC_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "mlb_api.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

MLB_API      = "https://statsapi.mlb.com/api/v1"
MLB_API_FEED = "https://statsapi.mlb.com/api/v1.1"
HEADERS   = {"User-Agent": "mlb-first-ab-pipeline/1.0"}
SEASONS   = list(range(2015, datetime.now().year + 1))


def mlb_get(path: str, params: dict = None) -> dict | None:
    """GET from MLB Stats API with basic error handling."""
    url = f"{MLB_API}{path}"
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning("MLB API error %s: %s", path, e)
        return None


# ── Schedule ─────────────────────────────────────────────────────────────────

def fetch_schedule(season: int) -> list[dict]:
    """Return list of game dicts for a full regular season."""
    cache_path = RAW_DIR / f"mlb_schedule_{season}.json"
    # Never cache the current season — game statuses change daily and stale
    # statuses cause today's games to be excluded from the feed-fetch filter.
    current_year = datetime.now().year
    if cache_path.exists() and season != current_year:
        log.info("Schedule %d loaded from cache.", season)
        with open(cache_path) as f:
            return json.load(f)

    log.info("Fetching schedule for %d ...", season)
    data = mlb_get("/schedule", {
        "sportId":   1,
        "season":    season,
        "gameType":  "R",
        "startDate": f"{season}-03-01",
        "endDate":   f"{season}-11-30",
        "fields":    (
            "dates,date,games,gamePk,gameDate,status,statusCode,"
            "teams,home,away,team,id,name,"
            "venue,id,name,"
            "weather,condition,temp,wind"
        ),
    })
    if not data:
        return []

    games = []
    for date_obj in data.get("dates", []):
        for game in date_obj.get("games", []):
            games.append(game)

    with open(cache_path, "w") as f:
        json.dump(games, f)
    log.info("  %d games found for %d", len(games), season)
    return games


# ── Game feed (lineups + umpires) ─────────────────────────────────────────────

def fetch_game_feed(game_pk: int) -> dict | None:
    """
    Pull the live game feed for one game.
    Contains confirmed lineups, umpires, boxscore.
    Only useful after game has started/ended.
    Uses v1.1 endpoint which is required for the feed/live path.
    """
    url = f"{MLB_API_FEED}/game/{game_pk}/feed/live"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning("MLB API error /game/%s/feed/live: %s", game_pk, e)
        return None


def extract_game_meta(game_pk: int, feed: dict) -> dict:
    """Extract key fields from a game feed."""
    meta = {"game_pk": game_pk}
    try:
        gd = feed.get("gameData", {})
        ld = feed.get("liveData", {})

        # Teams
        teams = gd.get("teams", {})
        meta["home_team_id"]   = teams.get("home", {}).get("id")
        meta["home_team_abbr"] = teams.get("home", {}).get("abbreviation")
        meta["away_team_id"]   = teams.get("away", {}).get("id")
        meta["away_team_abbr"] = teams.get("away", {}).get("abbreviation")

        # Venue
        venue = gd.get("venue", {})
        meta["venue_id"]   = venue.get("id")
        meta["venue_name"] = venue.get("name")

        # Weather (from MLB feed when available)
        wx = gd.get("weather", {})
        meta["weather_condition"] = wx.get("condition")
        meta["weather_temp"]      = wx.get("temp")
        meta["weather_wind"]      = wx.get("wind")

        # Probable pitchers from game data
        pp = gd.get("probablePitchers", {})
        meta["home_starter_id"]   = pp.get("home", {}).get("id")
        meta["home_starter_name"] = pp.get("home", {}).get("fullName")
        meta["away_starter_id"]   = pp.get("away", {}).get("id")
        meta["away_starter_name"] = pp.get("away", {}).get("fullName")

        # Umpires from officials
        officials = gd.get("officials", [])
        for official in officials:
            if official.get("officialType") == "Home Plate":
                meta["hp_umpire_id"]   = official.get("official", {}).get("id")
                meta["hp_umpire_name"] = official.get("official", {}).get("fullName")
                break

        # Lineups from boxscore
        boxscore = ld.get("boxscore", {})
        for side in ["home", "away"]:
            team_box = boxscore.get("teams", {}).get(side, {})
            batting_order = team_box.get("battingOrder", [])
            players = team_box.get("players", {})
            if batting_order:
                leadoff_id = batting_order[0]
                player_key = f"ID{leadoff_id}"
                player_info = players.get(player_key, {})
                meta[f"{side}_leadoff_id"]   = leadoff_id
                meta[f"{side}_leadoff_name"] = player_info.get("person", {}).get("fullName")

    except Exception as e:
        log.warning("  Error extracting meta for game %d: %s", game_pk, e)

    return meta


# ── Player metadata ───────────────────────────────────────────────────────────

def fetch_player_meta(player_id: int) -> dict | None:
    """Get player handedness, position, DOB."""
    data = mlb_get(f"/people/{player_id}", {"hydrate": "stats(group=[hitting,pitching],type=career)"})
    if not data or not data.get("people"):
        return None
    p = data["people"][0]
    return {
        "player_id":    p.get("id"),
        "full_name":    p.get("fullName"),
        "bat_side":     p.get("batSide", {}).get("code"),    # L / R / S
        "pitch_hand":   p.get("pitchHand", {}).get("code"),  # L / R
        "position":     p.get("primaryPosition", {}).get("abbreviation"),
        "birth_date":   p.get("birthDate"),
        "height":       p.get("height"),
        "weight":       p.get("weight"),
        "mlb_debut":    p.get("mlbDebutDate"),
    }


# ── Umpire historical K/BB rates ──────────────────────────────────────────────

# Umpire tendency lookup — sourced from UmpScorecards.com data
# strike_rate: fraction of called pitches that are called strikes (league avg ~0.455)
# Positive = larger zone (more K-friendly), negative = smaller zone
UMPIRE_TENDENCIES = {
    # name: {strike_rate, k_rate_adj, bb_rate_adj}
    # These are approximations — update from umpscorecards.com for current season
    "Angel Hernandez":   {"strike_rate": 0.440, "zone_adj": -0.015},
    "CB Bucknor":        {"strike_rate": 0.438, "zone_adj": -0.017},
    "Adrian Johnson":    {"strike_rate": 0.448, "zone_adj": -0.007},
    "Tom Hallion":       {"strike_rate": 0.448, "zone_adj": -0.007},
    "Bill Miller":       {"strike_rate": 0.449, "zone_adj": -0.006},
    "Mark Carlson":      {"strike_rate": 0.452, "zone_adj": -0.003},
    "Ted Barrett":       {"strike_rate": 0.453, "zone_adj": -0.002},
    "Laz Diaz":          {"strike_rate": 0.454, "zone_adj": -0.001},
    "Dan Iassogna":      {"strike_rate": 0.455, "zone_adj":  0.000},
    "Paul Nauert":       {"strike_rate": 0.457, "zone_adj":  0.002},
    "Sam Holbrook":      {"strike_rate": 0.458, "zone_adj":  0.003},
    "Eric Cooper":       {"strike_rate": 0.459, "zone_adj":  0.004},
    "Lance Barksdale":   {"strike_rate": 0.460, "zone_adj":  0.005},
    "Ron Kulpa":         {"strike_rate": 0.461, "zone_adj":  0.006},
    "Jim Reynolds":      {"strike_rate": 0.462, "zone_adj":  0.007},
    "Doug Eddings":      {"strike_rate": 0.463, "zone_adj":  0.008},
    "Joe West":          {"strike_rate": 0.463, "zone_adj":  0.008},
    "Jerry Meals":       {"strike_rate": 0.464, "zone_adj":  0.009},
    "Jordan Baker":      {"strike_rate": 0.465, "zone_adj":  0.010},
    "Chris Guccione":    {"strike_rate": 0.466, "zone_adj":  0.011},
    "Marvin Hudson":     {"strike_rate": 0.467, "zone_adj":  0.012},
    "Pat Hoberg":        {"strike_rate": 0.469, "zone_adj":  0.014},
    "Nic Lentz":         {"strike_rate": 0.470, "zone_adj":  0.015},
    "Ben May":           {"strike_rate": 0.470, "zone_adj":  0.015},
    "John Tumpane":      {"strike_rate": 0.471, "zone_adj":  0.016},
}

def get_umpire_tendency(name: str) -> dict:
    return UMPIRE_TENDENCIES.get(name, {"strike_rate": 0.455, "zone_adj": 0.0})


# ── Main runner ───────────────────────────────────────────────────────────────

def run(seasons: list[int] = None, fetch_feeds: bool = True):
    """
    fetch_feeds=True will hit the API for every game — slow but comprehensive.
    Set to False to only get schedule data (much faster).
    """
    if seasons is None:
        seasons = SEASONS

    all_schedule_rows = []
    all_game_meta     = []
    player_ids_seen   = set()

    for season in seasons:
        games = fetch_schedule(season)
        for game in games:
            game_pk    = game.get("gamePk")
            game_date  = game.get("gameDate", "")[:10]
            home_info  = game.get("teams", {}).get("home", {})
            away_info  = game.get("teams", {}).get("away", {})
            venue_info = game.get("venue", {})

            row = {
                "game_pk":        game_pk,
                "game_date":      game_date,
                "season":         season,
                "home_team_id":   home_info.get("team", {}).get("id"),
                "home_team_name": home_info.get("team", {}).get("name"),
                "away_team_id":   away_info.get("team", {}).get("id"),
                "away_team_name": away_info.get("team", {}).get("name"),
                "venue_id":       venue_info.get("id"),
                "venue_name":     venue_info.get("name"),
                "status":         game.get("status", {}).get("statusCode"),
            }
            all_schedule_rows.append(row)

    schedule_df = pd.DataFrame(all_schedule_rows)
    schedule_path = PROC_DIR / "schedule.parquet"
    schedule_df.to_parquet(schedule_path, index=False)
    log.info("Schedule saved: %d games → %s", len(schedule_df), schedule_path)

    if not fetch_feeds:
        log.info("Skipping game feeds (fetch_feeds=False)")
        return

    # Fetch game feeds for completed games to get lineups + umpires.
    # For today's games also include Pre-game (P) and In-Progress (I/IR) since
    # batting orders are submitted and available in the feed before Final status.
    today_str = date.today().isoformat()
    final_statuses = {"F", "FR", "O"}
    # Include "S" (Scheduled) so today's games are processed even before
    # they reach pre-game status in the API.
    todays_live_statuses = {"S", "P", "PW", "I", "IR", "MA", "NF"}
    final_games = schedule_df[
        schedule_df["status"].isin(final_statuses) |
        (schedule_df["game_date"].astype(str).str.startswith(today_str) &
         schedule_df["status"].isin(todays_live_statuses))
    ].copy()
    log.info("Fetching game feeds for %d games (completed + today's live)...",
             len(final_games))

    feeds_cache = RAW_DIR / "game_feeds"
    feeds_cache.mkdir(exist_ok=True)

    for _, row in final_games.iterrows():
        game_pk = int(row["game_pk"])
        cache_file = feeds_cache / f"{game_pk}.json"
        is_today = str(row["game_date"]).startswith(today_str)

        # Never use a cached feed for today's games — lineups are submitted
        # progressively and a cached pre-lineup feed would stay stale.
        if cache_file.exists() and not is_today:
            with open(cache_file) as f:
                feed = json.load(f)
        else:
            if is_today and cache_file.exists():
                cache_file.unlink()
            feed = fetch_game_feed(game_pk)
            if feed and not is_today:
                with open(cache_file, "w") as f:
                    json.dump(feed, f)
            time.sleep(0.4)  # polite rate limiting

        if feed:
            meta = extract_game_meta(game_pk, feed)
            meta["game_date"] = row["game_date"]
            meta["season"]    = row["season"]
            all_game_meta.append(meta)

    if all_game_meta:
        meta_df = pd.DataFrame(all_game_meta)

        # Ensure umpire columns exist (v1.1 feed may not include officials data)
        for col, default in [("hp_umpire_id", None), ("hp_umpire_name", None)]:
            if col not in meta_df.columns:
                meta_df[col] = None

        # Add umpire tendency (falls back to league average when name is unknown)
        meta_df["umpire_zone_adj"] = meta_df["hp_umpire_name"].apply(
            lambda n: get_umpire_tendency(str(n)).get("zone_adj", 0.0) if pd.notna(n) else 0.0
        )
        meta_df["umpire_strike_rate"] = meta_df["hp_umpire_name"].apply(
            lambda n: get_umpire_tendency(str(n)).get("strike_rate", 0.455) if pd.notna(n) else 0.455
        )

        meta_path = PROC_DIR / "game_meta.parquet"
        meta_df.to_parquet(meta_path, index=False)
        log.info("Game meta saved: %d games → %s", len(meta_df), meta_path)

        # Leadoff batter table
        leadoff_rows = []
        for _, row in meta_df.iterrows():
            for side in ["home", "away"]:
                batter_id   = row.get(f"{side}_leadoff_id")
                batter_name = row.get(f"{side}_leadoff_name")
                pitcher_id  = row.get("away_starter_id" if side == "home" else "home_starter_id")
                pitcher_name= row.get("away_starter_name" if side == "home" else "home_starter_name")
                if pd.notna(batter_id) and batter_id:
                    leadoff_rows.append({
                        "game_pk":       row["game_pk"],
                        "game_date":     row["game_date"],
                        "season":        row["season"],
                        "side":          side,
                        "batter_id":     batter_id,
                        "batter_name":   batter_name,
                        "pitcher_id":    pitcher_id,
                        "pitcher_name":  pitcher_name,
                        "venue_name":    row.get("venue_name"),
                        "hp_umpire_id":  row.get("hp_umpire_id"),
                        "hp_umpire_name":row.get("hp_umpire_name"),
                        "umpire_zone_adj":   row.get("umpire_zone_adj", 0.0),
                        "umpire_strike_rate":row.get("umpire_strike_rate", 0.455),
                    })
        if leadoff_rows:
            lb_df = pd.DataFrame(leadoff_rows)
            lb_path = PROC_DIR / "lineup_first_batter.parquet"
            lb_df.to_parquet(lb_path, index=False)
            log.info("Leadoff batter table: %d rows → %s", len(lb_df), lb_path)

    log.info("MLB API pipeline complete.")


# ── Active roster fetch ───────────────────────────────────────────────────────

# Positions that bat (keep). Exclude SP, RP, P.
_BATTER_POSITIONS = {
    "C","1B","2B","3B","SS","LF","CF","RF","DH","OF","IF","UT","PH","PR",
    "TWP",  # two-way player
}

def fetch_active_roster(team_id: int, season: int = None) -> list[dict]:
    """Return active-roster batters for a team from the MLB Stats API."""
    if season is None:
        season = datetime.now().year
    url = f"{MLB_API}/teams/{team_id}/roster"
    params = {"rosterType": "active", "season": season,
              "hydrate": "person(batSide,pitchHand)"}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning("Roster fetch error team %s: %s", team_id, e)
        return []

    players = []
    for entry in data.get("roster", []):
        person   = entry.get("person", {})
        position = entry.get("position", {})
        pos_code = position.get("abbreviation", "")
        pos_type = position.get("type", "")

        # Skip pure pitchers
        if pos_type == "Pitcher" and pos_code not in ("TWP",):
            continue
        if pos_code in ("SP", "RP", "P") and pos_code not in _BATTER_POSITIONS:
            continue

        players.append({
            "player_id":      person.get("id"),
            "full_name":      person.get("fullName"),
            "position":       pos_code,
            "bat_side":       person.get("batSide", {}).get("code", "R"),
            "jersey_number":  entry.get("jerseyNumber"),
            "status":         entry.get("status", {}).get("description", "Active"),
        })
    return players


def fetch_all_rosters_for_today(today: str = None) -> pd.DataFrame:
    """
    Fetch active rosters for every team playing today.
    Saves to data/processed/active_rosters.parquet.
    """
    if today is None:
        today = date.today().isoformat()

    out_path = PROC_DIR / "active_rosters.parquet"
    # Cache: skip if already fetched today
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        if "as_of_date" in existing.columns and existing["as_of_date"].astype(str).str.startswith(today).any():
            log.info("Active rosters already fetched today — using cache.")
            return existing

    # Determine today's teams. game_meta is only as fresh as the last full
    # src/02 run(), so the daily flow can leave it weeks stale — fall back to
    # probable_pitchers.parquet (refreshed every daily run) and then
    # schedule.parquet before giving up.
    def _today_team_pairs() -> list[tuple[int, str]]:
        sources = [
            (PROC_DIR / "game_meta.parquet",       "home_team_abbr", "away_team_abbr"),
            (PROC_DIR / "probable_pitchers.parquet", "home_team_name", "away_team_name"),
            (PROC_DIR / "schedule.parquet",          "home_team_name", "away_team_name"),
        ]
        for path, home_lbl, away_lbl in sources:
            if not path.exists():
                continue
            d = pd.read_parquet(path)
            if "game_date" not in d.columns:
                continue
            d = d[d["game_date"].astype(str).str.startswith(today)]
            if d.empty or "home_team_id" not in d.columns:
                continue
            pairs = (
                list(zip(d["home_team_id"].dropna().astype(int),
                         d.get(home_lbl, d["home_team_id"]).astype(str).tolist())) +
                list(zip(d["away_team_id"].dropna().astype(int),
                         d.get(away_lbl, d["away_team_id"]).astype(str).tolist()))
            )
            if pairs:
                log.info("Today's teams resolved from %s", path.name)
                return pairs
        return []

    team_pairs = _today_team_pairs()
    if not team_pairs:
        log.warning("No games found for %s in game_meta / probable_pitchers / schedule", today)
        return pd.DataFrame()
    seen = set()
    teams = []
    for tid, tname in team_pairs:
        if tid not in seen:
            seen.add(tid)
            teams.append((tid, tname))

    season = int(today[:4])
    all_rows = []
    for team_id, team_name in teams:
        log.info("  Fetching roster: team %s (%s)…", team_id, team_name)
        players = fetch_active_roster(team_id, season)
        for p in players:
            p["team_id"]   = team_id
            p["team_name"] = team_name
            p["as_of_date"] = today
        all_rows.extend(players)
        time.sleep(0.3)

    if not all_rows:
        log.warning("No roster data fetched.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.to_parquet(out_path, index=False)
    log.info("Active rosters saved: %d players across %d teams → %s",
             len(df), len(teams), out_path)
    return df


# ── Probable pitchers fetch ───────────────────────────────────────────────────

def fetch_probable_pitchers(today: str = None) -> pd.DataFrame:
    """
    Fetch probable pitchers for today from the schedule hydrated endpoint.
    Cache-aware: re-fetches if file is older than 2 hours, always on --force-refresh.
    Saves to data/processed/probable_pitchers.parquet.
    """
    if today is None:
        today = date.today().isoformat()

    out_path = PROC_DIR / "probable_pitchers.parquet"
    # Re-use cache if <2 hours old and has today's data
    if out_path.exists():
        import os as _os
        age_hours = (time.time() - _os.path.getmtime(out_path)) / 3600
        if age_hours < 2:
            existing = pd.read_parquet(out_path)
            if "game_date" in existing.columns and existing["game_date"].astype(str).str.startswith(today).any():
                log.info("Probable pitchers cached (%.0f min ago) — using cache.", age_hours * 60)
                return existing

    url = f"{MLB_API}/schedule"
    params = {
        "sportId": 1,
        "date":    today,
        "hydrate": "probablePitcher,team",
        "gameType": "R",
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning("Probable pitchers fetch error: %s", e)
        return pd.DataFrame()

    rows = []
    for date_obj in data.get("dates", []):
        for game in date_obj.get("games", []):
            game_pk   = game.get("gamePk")
            game_date = game.get("gameDate", "")[:10]
            teams     = game.get("teams", {})

            def _pitcher_info(side_data):
                pp = side_data.get("probablePitcher", {})
                if not pp:
                    return None, None, "R"
                pid  = pp.get("id")
                name = pp.get("fullName")
                # Try to get handedness from pitchHand
                hand = pp.get("pitchHand", {}).get("code", "R") if isinstance(pp.get("pitchHand"), dict) else "R"
                return pid, name, hand

            home = teams.get("home", {})
            away = teams.get("away", {})

            h_pid, h_name, h_hand = _pitcher_info(home)
            a_pid, a_name, a_hand = _pitcher_info(away)

            rows.append({
                "game_pk":                   game_pk,
                "game_date":                 game_date,
                "home_team_id":              home.get("team", {}).get("id"),
                "home_team_name":            home.get("team", {}).get("name"),
                "away_team_id":              away.get("team", {}).get("id"),
                "away_team_name":            away.get("team", {}).get("name"),
                "home_probable_pitcher_id":  h_pid,
                "home_probable_pitcher_name":h_name,
                "home_probable_pitcher_hand":h_hand,
                "away_probable_pitcher_id":  a_pid,
                "away_probable_pitcher_name":a_name,
                "away_probable_pitcher_hand":a_hand,
            })

    if not rows:
        log.warning("No probable pitcher data returned for %s", today)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    log.info("Probable pitchers saved: %d games → %s", len(df), out_path)
    return df


# ── Today-only game feed refresh (lineup confirmation check) ──────────────────

def fetch_todays_game_feeds(today: str = None) -> dict:
    """
    Re-fetch only today's game feeds to check if batting orders have been posted.
    Returns dict of {game_pk: feed} for games where lineups are confirmed.
    """
    if today is None:
        today = date.today().isoformat()

    sched_path = PROC_DIR / "schedule.parquet"
    if not sched_path.exists():
        log.warning("schedule.parquet not found")
        return {}

    sched = pd.read_parquet(sched_path)
    today_games = sched[sched["game_date"].astype(str).str.startswith(today)]

    feeds_cache = RAW_DIR / "game_feeds"
    feeds_cache.mkdir(exist_ok=True)

    confirmed_lineups = {}
    log.info("Fetching %d today's game feeds for lineup check…", len(today_games))

    for _, row in today_games.iterrows():
        game_pk = int(row["game_pk"])
        cache_file = feeds_cache / f"{game_pk}.json"
        # Always delete stale today's feed
        if cache_file.exists():
            cache_file.unlink()
        feed = fetch_game_feed(game_pk)
        time.sleep(0.3)
        if not feed:
            continue
        # Check if batting order is confirmed
        bs = feed.get("liveData", {}).get("boxscore", {}).get("teams", {})
        home_bo = bs.get("home", {}).get("battingOrder", [])
        away_bo = bs.get("away", {}).get("battingOrder", [])
        if home_bo or away_bo:
            confirmed_lineups[game_pk] = feed
            # Cache the feed with confirmed lineup
            with open(cache_file, "w") as f:
                json.dump(feed, f)

    log.info("Lineup confirmed for %d / %d games", len(confirmed_lineups), len(today_games))

    # Update game_meta with confirmed leadoff batters
    if confirmed_lineups:
        meta_path = PROC_DIR / "game_meta.parquet"
        if meta_path.exists():
            meta_df = pd.read_parquet(meta_path)
            for game_pk, feed in confirmed_lineups.items():
                extracted = extract_game_meta(game_pk, feed)
                for side in ["home", "away"]:
                    lid_col  = f"{side}_leadoff_id"
                    lnm_col  = f"{side}_leadoff_name"
                    if extracted.get(lid_col):
                        mask = meta_df["game_pk"] == game_pk
                        meta_df.loc[mask, lid_col] = extracted[lid_col]
                        meta_df.loc[mask, lnm_col] = extracted.get(lnm_col)
            meta_df.to_parquet(meta_path, index=False)
            log.info("game_meta.parquet updated with confirmed lineup data")

        # Also update lineup_first_batter.parquet
        lineup_path = PROC_DIR / "lineup_first_batter.parquet"
        existing_lb = pd.read_parquet(lineup_path) if lineup_path.exists() else pd.DataFrame()
        # Remove stale today's rows (in case previous NaN stubs snuck in)
        if not existing_lb.empty:
            existing_lb = existing_lb[~existing_lb["game_date"].astype(str).str.startswith(today)]

        new_lb_rows = []
        for game_pk, feed in confirmed_lineups.items():
            extracted = extract_game_meta(game_pk, feed)
            meta_row_candidates = []
            if not existing_lb.empty:
                pass  # game_meta has full info; we rebuild from extracted
            # Get season and date from schedule
            sched_row = today_games[today_games["game_pk"] == game_pk]
            gdate = today
            gseason = int(today[:4])

            for side in ["home", "away"]:
                batter_id   = extracted.get(f"{side}_leadoff_id")
                batter_name = extracted.get(f"{side}_leadoff_name")
                pitcher_id  = extracted.get("away_starter_id" if side == "home" else "home_starter_id")
                pitcher_name= extracted.get("away_starter_name" if side == "home" else "home_starter_name")
                if pd.notna(batter_id) and batter_id:
                    new_lb_rows.append({
                        "game_pk":            game_pk,
                        "game_date":          gdate,
                        "season":             gseason,
                        "side":               side,
                        "batter_id":          batter_id,
                        "batter_name":        batter_name,
                        "pitcher_id":         pitcher_id,
                        "pitcher_name":       pitcher_name,
                        "venue_name":         extracted.get("venue_name"),
                        "hp_umpire_id":       extracted.get("hp_umpire_id"),
                        "hp_umpire_name":     extracted.get("hp_umpire_name"),
                        "umpire_zone_adj":    extracted.get("umpire_zone_adj", 0.0),
                        "umpire_strike_rate": extracted.get("umpire_strike_rate", 0.455),
                    })

        if new_lb_rows:
            new_df = pd.DataFrame(new_lb_rows)
            combined = pd.concat([existing_lb, new_df], ignore_index=True)
            combined.to_parquet(lineup_path, index=False)
            log.info("lineup_first_batter.parquet updated: +%d confirmed rows", len(new_lb_rows))

    return confirmed_lineups


# ── Lineup position history ───────────────────────────────────────────────────

def build_lineup_position_stats(season: int = None) -> pd.DataFrame:
    """
    Read cached game feeds for the given season and build per-player lineup position stats.
    Saves:
      data/processed/lineup_position_history.parquet  (one row per player per position)
      data/processed/lineup_position_summary.parquet  (one row per player, wide format)
    """
    if season is None:
        season = datetime.now().year

    feeds_dir = RAW_DIR / "game_feeds"
    if not feeds_dir.exists():
        log.warning("No game_feeds directory found.")
        return pd.DataFrame()

    feed_files = list(feeds_dir.glob("*.json"))
    log.info("Building lineup position history from %d cached game feeds (season=%d)…",
             len(feed_files), season)

    season_str = str(season)

    # Step 1: Extract batting orders — current season only
    position_rows = []
    for i, fp in enumerate(feed_files):
        if i % 500 == 0:
            log.info("  Processing feed %d / %d…", i, len(feed_files))
        try:
            with open(fp) as f:
                feed = json.load(f)
        except Exception:
            continue

        gd = feed.get("gameData", {})
        ld = feed.get("liveData", {})
        game_date = gd.get("datetime", {}).get("officialDate", "")
        game_pk   = gd.get("game", {}).get("pk")
        if not game_pk or not game_date:
            try:
                game_pk = int(fp.stem)
            except Exception:
                continue

        # Filter to the requested season
        if not str(game_date).startswith(season_str):
            continue

        bs = ld.get("boxscore", {}).get("teams", {})
        for side in ["home", "away"]:
            team_box = bs.get(side, {})
            players  = team_box.get("players", {})
            # Use each player's individual battingOrder field (N*100 encoding:
            # 100=spot1, 200=spot2, ..., 801=substitute at spot8) so that both
            # starters AND substitutes who actually batted are captured.
            for key, info in players.items():
                bo = info.get("battingOrder")
                if bo is None:
                    continue
                pos_idx = int(bo) // 100
                if pos_idx < 1 or pos_idx > 9:
                    continue
                bat_stats = info.get("stats", {}).get("batting", {})
                pa = bat_stats.get("plateAppearances", 0)
                if not pa:
                    continue
                player_id = info.get("person", {}).get("id")
                name      = info.get("person", {}).get("fullName", "")
                if not player_id:
                    continue
                position_rows.append({
                    "game_pk":         game_pk,
                    "game_date":       game_date,
                    "side":            side,
                    "lineup_position": pos_idx,
                    "player_id":       int(player_id),
                    "player_name":     name,
                    "pa":              pa,
                    "ab":              bat_stats.get("atBats", 0),
                    "hits":            bat_stats.get("hits", 0),
                    "hr":              bat_stats.get("homeRuns", 0),
                    "bb":              bat_stats.get("baseOnBalls", 0),
                    "ks":              bat_stats.get("strikeOuts", 0),
                    "rbi":             bat_stats.get("rbi", 0),
                    "runs":            bat_stats.get("runs", 0),
                    "doubles":         bat_stats.get("doubles", 0),
                    "triples":         bat_stats.get("triples", 0),
                })

    if not position_rows:
        log.warning("No batting order data found in cached feeds.")
        return pd.DataFrame()

    pos_df = pd.DataFrame(position_rows)
    pos_df["game_date"] = pd.to_datetime(pos_df["game_date"], errors="coerce")
    log.info("Extracted %d player-game-position rows", len(pos_df))

    # Per-batter game logs (one row per player per game) — powers the dashboard's
    # L7 / L30 batter windows (OBP, OPS, R, AB) in the "Top of Order" sections.
    gl = (pos_df.groupby(["player_id", "player_name", "game_pk", "game_date"], as_index=False)
          [["pa", "ab", "hits", "hr", "bb", "ks", "rbi", "runs", "doubles", "triples"]].sum())
    gl["tb"] = gl["hits"] + gl["doubles"] + 2 * gl["triples"] + 3 * gl["hr"]
    gl["reach"] = gl["hits"] + gl["bb"]
    gl_path = PROC_DIR / "batter_game_logs.parquet"
    gl.sort_values(["player_id", "game_date"]).to_parquet(gl_path, index=False)
    log.info("Batter game logs saved: %d rows → %s", len(gl), gl_path)

    # Step 2/3: Stats come directly from the boxscore — no Statcast join needed
    log.info("Using boxscore stats from game feeds — no Statcast join needed.")
    merged = pos_df.copy()
    # Compute total bases from component hits
    merged["tb"] = (merged["hits"] - merged["doubles"] - merged["triples"] - merged["hr"]
                    + merged["doubles"] * 2
                    + merged["triples"] * 3
                    + merged["hr"] * 4)
    # reach_cnt = hits + walks (BB)
    merged["reach_cnt"] = merged["hits"] + merged["bb"]

    # Step 4: Aggregate by player + lineup position
    agg = merged.groupby(["player_id", "player_name", "lineup_position"]).agg(
        games_started_at_position = ("game_pk", "nunique"),
        pa_at_position            = ("pa",    "sum"),
        hits_at_position          = ("hits",  "sum"),
        ab_at_position            = ("ab",    "sum"),
        walks_at_position         = ("bb",    "sum"),
        ks_at_position            = ("ks",    "sum"),
        hrs_at_position           = ("hr",    "sum"),
        rbi_at_position           = ("rbi",   "sum"),
        runs_at_position          = ("runs",  "sum"),
        tb_at_position            = ("tb",    "sum"),
        reach_at_position         = ("reach_cnt", "sum"),
    ).reset_index()

    agg["avg_at_position"] = (agg["hits_at_position"] / agg["ab_at_position"].replace(0, np.nan)).round(3)
    agg["obp_at_position"] = (agg["reach_at_position"] / agg["pa_at_position"].replace(0, np.nan)).round(3)
    agg["slg_at_position"] = (agg["tb_at_position"]    / agg["ab_at_position"].replace(0, np.nan)).round(3)
    agg["ops_at_position"] = (agg["obp_at_position"].fillna(0) + agg["slg_at_position"].fillna(0)).round(3)
    agg["k_pct_at_position"]  = (agg["ks_at_position"]    / agg["pa_at_position"].replace(0, np.nan)).round(3)
    agg["bb_pct_at_position"] = (agg["walks_at_position"] / agg["pa_at_position"].replace(0, np.nan)).round(3)

    # Total games and most common position
    total_games = merged.groupby("player_id")["game_pk"].nunique().reset_index()
    total_games.columns = ["player_id", "total_games_played"]

    most_common = (
        agg.sort_values("games_started_at_position", ascending=False)
        .groupby("player_id").first()
        .reset_index()[["player_id", "lineup_position"]]
        .rename(columns={"lineup_position": "most_common_position"})
    )

    versatility = agg.groupby("player_id")["lineup_position"].nunique().reset_index()
    versatility.columns = ["player_id", "position_versatility"]

    agg = agg.merge(total_games,   on="player_id", how="left")
    agg = agg.merge(most_common,   on="player_id", how="left")
    agg = agg.merge(versatility,   on="player_id", how="left")

    hist_path = PROC_DIR / "lineup_position_history.parquet"
    agg.to_parquet(hist_path, index=False)
    log.info("Lineup position history saved: %d rows → %s", len(agg), hist_path)

    # Step 5: Build wide summary (one row per player, stats at each position as columns)
    summary_rows = {}
    for _, row in agg.iterrows():
        pid  = int(row["player_id"])
        pos  = int(row["lineup_position"])
        if pid not in summary_rows:
            summary_rows[pid] = {
                "player_id":            pid,
                "player_name":          row["player_name"],
                "total_games_played":   safe_int(row.get("total_games_played", 0)),
                "most_common_position": safe_int(row.get("most_common_position", 0)),
                "position_versatility": safe_int(row.get("position_versatility", 0)),
            }
        p = summary_rows[pid]
        p[f"games_pos_{pos}"] = safe_int(row["games_started_at_position"])
        p[f"pa_pos_{pos}"]    = safe_int(row["pa_at_position"])
        p[f"ab_pos_{pos}"]    = safe_int(row["ab_at_position"])
        p[f"h_pos_{pos}"]     = safe_int(row["hits_at_position"])
        p[f"hr_pos_{pos}"]    = safe_int(row["hrs_at_position"])
        p[f"bb_pos_{pos}"]    = safe_int(row["walks_at_position"])
        p[f"k_pos_{pos}"]     = safe_int(row["ks_at_position"])
        p[f"rbi_pos_{pos}"]   = safe_int(row["rbi_at_position"])
        p[f"avg_pos_{pos}"]   = row["avg_at_position"]
        p[f"obp_pos_{pos}"]   = row["obp_at_position"]
        p[f"slg_pos_{pos}"]   = row["slg_at_position"]
        p[f"ops_pos_{pos}"]   = row["ops_at_position"]
        p[f"k_pct_pos_{pos}"] = row["k_pct_at_position"]
        p[f"bb_pct_pos_{pos}"]= row["bb_pct_at_position"]

    summary_df = pd.DataFrame(list(summary_rows.values()))
    sum_path = PROC_DIR / "lineup_position_summary.parquet"
    summary_df.to_parquet(sum_path, index=False)
    log.info("Lineup position summary saved: %d players → %s", len(summary_df), sum_path)

    return agg


def safe_int(v):
    try:
        f = float(v)
        return 0 if pd.isna(f) else int(f)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]

    if "--probable-pitchers-only" in args:
        fetch_probable_pitchers()
        sys.exit(0)

    if "--rosters-only" in args:
        fetch_all_rosters_for_today()
        sys.exit(0)

    if "--todays-feeds-only" in args:
        fetch_todays_game_feeds()
        sys.exit(0)

    if "--build-lineup-history" in args:
        build_lineup_position_stats()
        sys.exit(0)

    # Legacy usage: python 02_fetch_mlb_api.py [--no-feeds] [year year ...]
    fetch_feeds = "--no-feeds" not in args
    seasons = [int(a) for a in args if a.isdigit()] or None
    run(seasons=seasons, fetch_feeds=fetch_feeds)
