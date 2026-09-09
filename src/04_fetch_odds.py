"""
04_fetch_odds.py
----------------
Fetches MLB betting odds from The Odds API (free tier: 500 req/month).
https://the-odds-api.com

Markets fetched:
  - batter_first_home_run    (if available)
  - batter_hits              (hits in game — proxy for first AB reach)
  - h2h (moneyline)          (for context / implied run environment)
  - totals                   (game over/under — proxy for run environment)

The "first batter to reach base" market is not universally available
on all books. We capture:
  1. Any first-inning props if available
  2. batter_hits as a proxy (strong correlation with first-AB reach rate)
  3. moneyline + total for market context

Free tier gives 500 requests/month. We call once per day (~30 games max).

Output:
  data/odds/odds_{YYYY-MM-DD}.json    raw API response
  data/odds/todays_props.parquet      cleaned props table
"""

import os
import json
import logging
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

BASE_DIR  = Path(__file__).resolve().parent.parent
ODDS_DIR  = BASE_DIR / "data" / "odds"
LOG_DIR   = BASE_DIR / "logs"

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

for d in [ODDS_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / "env.txt")
API_KEY  = os.getenv("ODDS_API_KEY", "")
SGO_KEY  = os.getenv("SGO_API_KEY", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "odds.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Markets to request from The Odds API (check remaining quota before adding more).
# NOTE: FanDuel does NOT expose batter props via The Odds API — only pitcher_strikeouts.
# FanDuel batter props (incl. Plate Appearances / 1st PA) come from SportsGameOdds instead.
MARKETS_GAME    = "h2h,totals"
MARKETS_PLAYER  = "batter_hits,batter_walks,batter_singles,batter_total_bases,pitcher_strikeouts"

# First-inning (YRFI / NRFI) market keys to try, in priority order. The Odds API
# does not always carry these on the free/standard MLB feed — the fetcher tries
# each and uses the first that returns data, else falls back to model-only.
MARKETS_FIRST_INNING = [
    "totals_1st_1_innings",
    "totals_first_inning",
    "team_totals_1st_1_innings",
    "team_totals_first_inning",
    "h2h_1st_1_innings",
    "h2h_first_inning",
    "alternate_totals_1st_1_innings",
]
# First-inning (YRFI/NRFI) is sourced from SportsGameOdds — it carries the
# 0.5-run first-inning O/U for DraftKings, FanDuel, BetMGM AND Caesars, whereas
# The Odds API only exposes it for FanDuel + BetMGM (DK / Caesars withhold that
# derivative market from that feed). The Odds API path is kept as a fallback.
# NOTE (checked 2026-09-09): neither SportsGameOdds nor The Odds API carries
# bet365 or Fanatics Sportsbook for MLB — they cannot be added without a
# dedicated third data source.
FI_BOOK_KEYS = {"draftkings": "DK", "fanduel": "FD", "betmgm": "MGM", "caesars": "CZR"}
SGO_FI_OVER_KEY = "points-all-1i-ou-over"    # YRFI
SGO_FI_UNDER_KEY = "points-all-1i-ou-under"  # NRFI

# Sportsbooks to include (North American + major)
BOOKMAKERS = "draftkings,fanduel,betmgm,caesars,pointsbet,williamhill_us,betonlineag,bovada"

# ── SportsGameOdds (SGO) constants ────────────────────────────────────────────
SGO_BASE = "https://api.sportsgameodds.com/v2"

# Map SGO statID → our internal market key
SGO_STAT_TO_MARKET = {
    "batting_hits":         "batter_hits",
    "batting_basesOnBalls": "batter_walks",
    "batting_singles":      "batter_singles",
    "batting_totalBases":   "batter_total_bases",
    "batting_homeRuns":     "batter_home_runs",
    "batting_RBI":          "batter_rbis",
    "batting_doubles":      "batter_doubles",
    "batting_triples":      "batter_triples",
    "pitching_strikeouts":  "pitcher_strikeouts",
    "pitching_outs":        "pitcher_outs",
    "pitching_earnedRuns":  "pitcher_earned_runs",
}
SGO_TARGET_STATS    = set(SGO_STAT_TO_MARKET.keys())
SGO_TARGET_BET_TYPE = "ou"   # over/under only (yn yes/no has no line value)

def american_to_prob(american_odds: int) -> float:
    """Convert American odds to implied probability (with vig)."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)

def remove_vig(prob_yes: float, prob_no: float) -> tuple[float, float]:
    """
    Remove the vig from a two-sided market.
    Returns (true_prob_yes, true_prob_no) that sum to 1.0.
    """
    total = prob_yes + prob_no
    return prob_yes / total, prob_no / total

def get_remaining_requests() -> int:
    """Check how many free-tier requests remain."""
    if not API_KEY:
        return 0
    try:
        r = requests.get(f"{ODDS_API_BASE}/sports", params={"apiKey": API_KEY}, timeout=10)
        remaining = int(r.headers.get("x-requests-remaining", 0))
        used      = int(r.headers.get("x-requests-used", 0))
        log.info("Odds API: %d requests used, %d remaining", used, remaining)
        return remaining
    except Exception as e:
        log.warning("Could not check remaining requests: %s", e)
        return -1


def fetch_mlb_game_odds(date_str: str = None) -> list[dict]:
    """
    Fetch game-level odds (moneyline + totals) for MLB games.
    date_str format: YYYY-MM-DD (defaults to today)
    """
    if not API_KEY:
        log.error("ODDS_API_KEY not set in .env — cannot fetch odds.")
        return []

    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    params = {
        "apiKey":     API_KEY,
        "regions":    "us",
        "markets":    MARKETS_GAME,
        "oddsFormat": "american",
        "dateFormat": "iso",
        "bookmakers": BOOKMAKERS,
        "commenceTimeFrom": f"{date_str}T00:00:00Z",
        "commenceTimeTo":   f"{date_str}T23:59:59Z",
    }

    try:
        r = requests.get(f"{ODDS_API_BASE}/sports/baseball_mlb/odds", params=params, timeout=20)
        r.raise_for_status()
        log.info("Requests remaining: %s", r.headers.get("x-requests-remaining", "?"))
        return r.json()
    except Exception as e:
        log.error("Failed to fetch game odds: %s", e)
        return []


def fetch_mlb_player_props(event_id: str) -> list[dict] | None:
    """
    Fetch player props for a specific game event ID.
    Returns None when the API signals quota exhaustion (401) so the caller
    can break out of the loop immediately instead of burning more requests.
    Returns [] on other transient errors.
    """
    if not API_KEY:
        return []
    params = {
        "apiKey":     API_KEY,
        "regions":    "us",
        "markets":    MARKETS_PLAYER,
        "oddsFormat": "american",
        "bookmakers": BOOKMAKERS,
    }
    try:
        r = requests.get(
            f"{ODDS_API_BASE}/sports/baseball_mlb/events/{event_id}/odds",
            params=params, timeout=20
        )
        if r.status_code == 401:
            log.warning("  Quota exhausted (401) for event %s — stopping props loop.", event_id)
            return None  # sentinel: caller must break
        r.raise_for_status()
        log.info("  Props fetched for event %s (remaining: %s)",
                 event_id, r.headers.get("x-requests-remaining", "?"))
        return r.json().get("bookmakers", [])
    except Exception as e:
        log.warning("  Props fetch error for %s: %s", event_id, e)
        return []


def parse_game_odds(games_data: list[dict]) -> pd.DataFrame:
    """
    Flatten the API response into a clean DataFrame.
    One row per (game, bookmaker, market, outcome).
    """
    rows = []
    for game in games_data:
        event_id   = game.get("id")
        home_team  = game.get("home_team")
        away_team  = game.get("away_team")
        commence   = game.get("commence_time", "")[:10]

        for bm in game.get("bookmakers", []):
            book = bm.get("key")
            for market in bm.get("markets", []):
                mkey = market.get("key")
                outcomes = market.get("outcomes", [])

                if mkey == "h2h":
                    # Moneyline — find implied probs
                    p = {o["name"]: o["price"] for o in outcomes}
                    if home_team in p and away_team in p:
                        p_home = american_to_prob(p[home_team])
                        p_away = american_to_prob(p[away_team])
                        tp_home, tp_away = remove_vig(p_home, p_away)
                        rows.append({
                            "event_id": event_id, "game_date": commence,
                            "home_team": home_team, "away_team": away_team,
                            "bookmaker": book, "market": mkey,
                            "outcome": "home_win",
                            "american_odds": p[home_team],
                            "implied_prob": p_home,
                            "true_prob": tp_home,
                        })
                        rows.append({
                            "event_id": event_id, "game_date": commence,
                            "home_team": home_team, "away_team": away_team,
                            "bookmaker": book, "market": mkey,
                            "outcome": "away_win",
                            "american_odds": p[away_team],
                            "implied_prob": p_away,
                            "true_prob": tp_away,
                        })

                elif mkey == "totals":
                    for o in outcomes:
                        price = o["price"]
                        point = o.get("point")
                        rows.append({
                            "event_id": event_id, "game_date": commence,
                            "home_team": home_team, "away_team": away_team,
                            "bookmaker": book, "market": mkey,
                            "outcome": o["name"],   # "Over" or "Under"
                            "line": point,
                            "american_odds": price,
                            "implied_prob": american_to_prob(price),
                        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def parse_player_props(event_id: str, home_team: str, away_team: str,
                       game_date: str, bookmakers_data: list[dict]) -> pd.DataFrame:
    """Flatten player props for one game."""
    rows = []
    for bm in bookmakers_data:
        book = bm.get("key")
        for market in bm.get("markets", []):
            mkey     = market.get("key")
            for o in market.get("outcomes", []):
                player = o.get("description", o.get("name", ""))
                side   = o.get("name")    # "Over" or "Under"
                line   = o.get("point")
                price  = o.get("price")
                rows.append({
                    "event_id":    event_id,
                    "game_date":   game_date,
                    "home_team":   home_team,
                    "away_team":   away_team,
                    "bookmaker":   book,
                    "market":      mkey,
                    "player":      player,
                    "side":        side,
                    "line":        line,
                    "american_odds": price,
                    "implied_prob": american_to_prob(price) if price else None,
                })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _sgo_player_id_to_name(player_id: str) -> str:
    """Convert 'JUAN_SOTO_1_MLB' → 'Juan Soto'."""
    parts = player_id.split("_")
    # Last two parts are the disambiguation number and sport league (e.g. '1', 'MLB')
    # Everything before those is the player name
    name_parts = parts[:-2]
    return " ".join(p.capitalize() for p in name_parts)


def fetch_sgo_events(date_str: str) -> list[dict]:
    """Page through SportsGameOdds MLB events for date_str, returning all with odds embedded."""
    if not SGO_KEY:
        log.warning("SGO_API_KEY not set in env.txt — skipping SportsGameOdds fetch.")
        return []

    events: list[dict] = []
    cursor = None
    date_from = f"{date_str}T00:00:00.000Z"
    date_to   = f"{date_str}T23:59:59.000Z"

    while True:
        params: dict = {
            "apiKey":      SGO_KEY,
            "leagueID":    "MLB",
            "oddsAvailable": "true",
            "includeOdds": "true",
            "limit":       20,
            "startsAfter": date_from,
            "startsBefore": date_to,
        }
        if cursor:
            params["cursor"] = cursor

        try:
            r = requests.get(f"{SGO_BASE}/events/", params=params, timeout=30)
            r.raise_for_status()
            body   = r.json()
            batch  = body.get("data", [])
            events.extend(batch)
            log.info("SGO batch: %d events (total so far: %d)", len(batch), len(events))
            cursor = body.get("nextCursor")
            if not cursor or len(batch) < 20:
                break
        except Exception as e:
            log.warning("SGO fetch error: %s", e)
            break

    log.info("SGO: fetched %d MLB events for %s", len(events), date_str)
    return events


def parse_sgo_props(events: list[dict], date_str: str) -> pd.DataFrame:
    """
    Flatten SportsGameOdds event odds into the same props DataFrame format
    as parse_player_props() so both sources can be concatenated.
    Only includes over/under player props for target stats (game period, available=True).
    Preserves FanDuel deeplinks in a separate 'deeplink' column.
    """
    rows = []
    for event in events:
        home_team = event["teams"]["home"]["names"]["long"]
        away_team = event["teams"]["away"]["names"]["long"]

        for odd_key, odd in event.get("odds", {}).items():
            stat_id   = odd.get("statID", "")
            bet_type  = odd.get("betTypeID", "")
            side_id   = odd.get("sideID", "")
            player_id = odd.get("playerID")
            period_id = odd.get("periodID", "game")

            if stat_id not in SGO_TARGET_STATS:
                continue
            if period_id != "game":
                continue
            if not player_id:
                continue
            if bet_type != SGO_TARGET_BET_TYPE:
                continue

            market      = SGO_STAT_TO_MARKET[stat_id]
            player_name = _sgo_player_id_to_name(player_id)
            side        = "Over" if side_id == "over" else "Under"

            for book, bm in odd.get("byBookmaker", {}).items():
                if not bm.get("available"):
                    continue
                odds_raw = bm.get("odds")
                if odds_raw is None:
                    continue
                try:
                    odds_int = int(odds_raw)
                except (ValueError, TypeError):
                    continue

                line_raw = bm.get("overUnder")
                try:
                    line = float(line_raw) if line_raw is not None else None
                except (ValueError, TypeError):
                    line = None

                rows.append({
                    "event_id":      event.get("eventID"),
                    "game_date":     date_str,
                    "home_team":     home_team,
                    "away_team":     away_team,
                    "bookmaker":     book,
                    "market":        market,
                    "player":        player_name,
                    "side":          side,
                    "line":          line,
                    "american_odds": odds_int,
                    "implied_prob":  american_to_prob(odds_int),
                    "deeplink":      bm.get("deeplink"),
                    "source":        "sgo",
                })

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not df.empty:
        log.info("SGO props parsed: %d rows across %d bookmakers",
                 len(df), df["bookmaker"].nunique())
    return df


def run_sgo(date_str: str = None, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch player props from SportsGameOdds and save to
      data/odds/sgo_props_{date_str}.parquet
    Returns the parsed DataFrame (empty if fetch fails).
    Also merges into props_{date_str}.parquet if it already exists.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    sgo_path   = ODDS_DIR / f"sgo_props_{date_str}.parquet"
    props_path = ODDS_DIR / f"props_{date_str}.parquet"
    today_path = ODDS_DIR / "todays_props.parquet"

    if not force_refresh and sgo_path.exists():
        log.info("SGO props cache hit for %s — loading from disk.", date_str)
        return pd.read_parquet(sgo_path)

    events = fetch_sgo_events(date_str)
    if not events:
        log.warning("SGO: no events returned for %s.", date_str)
        return pd.DataFrame()

    sgo_df = parse_sgo_props(events, date_str)
    if sgo_df.empty:
        log.warning("SGO: props DataFrame is empty after parsing.")
        return pd.DataFrame()

    sgo_df.to_parquet(sgo_path, index=False)
    log.info("SGO props saved: %d rows → %s", len(sgo_df), sgo_path)

    # Merge into the combined props file (add 'source' col to existing if absent)
    if props_path.exists():
        existing = pd.read_parquet(props_path)
        if "source" not in existing.columns:
            existing["source"] = "odds_api"
        if "deeplink" not in existing.columns:
            existing["deeplink"] = None
        # Drop any stale SGO rows so we don't double-count on re-run
        existing = existing[existing.get("source", "odds_api") != "sgo"]
        combined = pd.concat([existing, sgo_df], ignore_index=True)
    else:
        combined = sgo_df

    combined.to_parquet(props_path, index=False)
    combined.to_parquet(today_path, index=False)
    log.info("Combined props saved: %d rows → %s", len(combined), props_path)

    return sgo_df


def append_to_odds_history(props_df: pd.DataFrame, snapshot_ts: str) -> None:
    """Append a timestamped snapshot of player props to odds_history.parquet."""
    if props_df.empty:
        return

    hist_path = ODDS_DIR / "odds_history.parquet"

    snap = props_df.copy()
    # Keep only player prop rows (skip game-level odds that have no player)
    snap = snap[snap["player"].notna() & (snap["player"].astype(str).str.strip() != "")]
    if snap.empty:
        return

    snap["timestamp"] = snapshot_ts
    snap["game_date"] = snap["game_date"].astype(str)

    # Standardize team names to MLB abbreviations
    snap["home_team"] = snap["home_team"].map(ODDS_TO_MLB).fillna(snap["home_team"])
    snap["away_team"] = snap["away_team"].map(ODDS_TO_MLB).fillna(snap["away_team"])

    keep_cols = ["game_date", "timestamp", "home_team", "away_team",
                 "player", "market", "side", "line", "american_odds", "implied_prob", "bookmaker"]
    snap = snap[[c for c in keep_cols if c in snap.columns]]

    if hist_path.exists():
        existing = pd.read_parquet(hist_path)
        combined = pd.concat([existing, snap], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["timestamp", "player", "bookmaker", "market", "side"],
            keep="first",
        )
    else:
        combined = snap

    combined.to_parquet(hist_path, index=False)
    log.info("Odds history updated: %d total rows → %s", len(combined), hist_path)


def run(date_str: str = None, fetch_props: bool = True, force_refresh: bool = False):
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    snapshot_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    raw_path   = ODDS_DIR / f"odds_{date_str}.json"
    game_path  = ODDS_DIR / f"game_odds_{date_str}.parquet"
    props_path = ODDS_DIR / f"props_{date_str}.parquet"
    today_path = ODDS_DIR / "todays_props.parquet"

    # ── Full cache hit: all three outputs already on disk ─────────────────────
    if not force_refresh and raw_path.exists() and game_path.exists() and props_path.exists():
        print("Odds already fetched for today, loading from cache.")
        log.info("Odds cache hit for %s — skipping all API calls.", date_str)
        pd.read_parquet(props_path).to_parquet(today_path, index=False)
        games_data = json.loads(raw_path.read_text())
        log.info("\n%s", "=" * 60)
        log.info("Today's games (%s) [cached]:", date_str)
        for game in games_data:
            log.info("  %s @ %s", game["away_team"], game["home_team"])
        log.info("=" * 60)
        return

    log.info("Fetching odds for %s ...", date_str)

    # ── Partial cache hit: raw JSON saved but parquets missing ────────────────
    if not force_refresh and raw_path.exists():
        log.info("Raw odds JSON found — re-parsing without API call.")
        games_data = json.loads(raw_path.read_text())
        remaining  = None   # only fetched if props are needed
    else:
        # ── Full fetch from API ───────────────────────────────────────────────
        if not API_KEY:
            log.error(
                "Set ODDS_API_KEY in env.txt.\n"
                "Get a free key at: https://the-odds-api.com\n"
                "Free tier: 500 requests/month."
            )
            return

        remaining = get_remaining_requests()
        if remaining == 0:
            log.error("No API requests remaining this month.")
            return

        games_data = fetch_mlb_game_odds(date_str)
        if not games_data:
            log.warning("No games found for %s.", date_str)
            return

        with open(raw_path, "w") as f:
            json.dump(games_data, f, indent=2)
        log.info("Saved raw odds: %s (%d games)", raw_path, len(games_data))

    # ── Game-level odds ───────────────────────────────────────────────────────
    if not force_refresh and game_path.exists():
        game_df = pd.read_parquet(game_path)
        log.info("Game odds loaded from cache: %d rows", len(game_df))
    else:
        game_df = parse_game_odds(games_data)
        if len(game_df):
            game_df.to_parquet(game_path, index=False)
            log.info("Saved game odds: %d rows → %s", len(game_df), game_path)

    # ── Player props ──────────────────────────────────────────────────────────
    all_props = []

    if not force_refresh and props_path.exists():
        log.info("Player props loaded from cache.")
        all_props = [pd.read_parquet(props_path)]
    elif fetch_props:
        if remaining is None:
            remaining = get_remaining_requests()
        if remaining > len(games_data) + 2:
            log.info("Fetching player props for %d games...", len(games_data))
            for game in games_data:
                event_id  = game["id"]
                home_team = game["home_team"]
                away_team = game["away_team"]
                commence  = game.get("commence_time", "")[:10]
                props_data = fetch_mlb_player_props(event_id)
                if props_data is None:
                    # 401 = quota exhausted — save whatever we have and stop
                    log.warning("Stopping props fetch after %d/%d games — quota exhausted.",
                                len(all_props), len(games_data))
                    break
                if props_data:
                    props_df = parse_player_props(event_id, home_team, away_team, commence, props_data)
                    if len(props_df):
                        all_props.append(props_df)
        else:
            log.warning("Low request budget (%d remaining), skipping player props.", remaining)
    else:
        log.info("Skipping player props (fetch_props=False)")

    # ── Save props + today alias ──────────────────────────────────────────────
    if all_props:
        props_combined = pd.concat(all_props, ignore_index=True)
        props_combined.to_parquet(props_path, index=False)
        props_combined.to_parquet(today_path, index=False)
        log.info("Saved player props: %d rows → %s", len(props_combined), props_path)
        append_to_odds_history(props_combined, snapshot_ts)

    # ── SportsGameOdds props (primary FanDuel batter prop source) ────────────
    # Run regardless of The Odds API quota — SGO has its own key and free tier.
    log.info("Fetching player props from SportsGameOdds...")
    run_sgo(date_str=date_str, force_refresh=force_refresh)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n%s", "=" * 60)
    log.info("Today's games (%s):", date_str)
    for game in games_data:
        log.info("  %s @ %s", game["away_team"], game["home_team"])
    log.info("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
#  YRFI / NRFI  —  first-inning over/under 0.5 runs
# ═══════════════════════════════════════════════════════════════════════════
def _devig_pair(p_yes: float, p_no: float) -> tuple[float, float]:
    """Remove the vig from a two-way market (normalise implied probs to sum 1)."""
    if p_yes is None or p_no is None:
        return (np.nan, np.nan)
    s = p_yes + p_no
    if s <= 0:
        return (np.nan, np.nan)
    return (p_yes / s, p_no / s)


def _fi_outcome_side(name: str) -> str | None:
    """Classify a first-inning market outcome name as YRFI or NRFI."""
    n = (name or "").strip().lower()
    if n in ("over", "yes") or n.startswith("over "):
        return "YRFI"
    if n in ("under", "no") or n.startswith("under "):
        return "NRFI"
    return None


def fetch_first_inning_odds(date_str: str) -> pd.DataFrame:
    """
    Try each first-inning market key against the events-odds endpoint.
    Returns a long DataFrame: game_pk, home_team, away_team, book, side, american, implied.
    Empty DataFrame if no first-inning market is available on this API tier.
    """
    if not API_KEY:
        log.warning("ODDS_API_KEY not set — cannot fetch first-inning odds.")
        return pd.DataFrame()

    # event list first (1 request)
    try:
        ev = requests.get(
            f"{ODDS_API_BASE}/sports/baseball_mlb/events",
            params={"apiKey": API_KEY,
                    "commenceTimeFrom": f"{date_str}T00:00:00Z",
                    "commenceTimeTo":   f"{date_str}T23:59:59Z"},
            timeout=20).json()
    except Exception as e:
        log.warning("first-inning event list failed: %s", e)
        return pd.DataFrame()
    if not isinstance(ev, list) or not ev:
        log.warning("no MLB events listed for %s", date_str)
        return pd.DataFrame()

    # probe market keys on the first event
    probe_id = ev[0]["id"]
    working_market = None
    for mk in MARKETS_FIRST_INNING:
        try:
            r = requests.get(
                f"{ODDS_API_BASE}/sports/baseball_mlb/events/{probe_id}/odds",
                params={"apiKey": API_KEY, "regions": "us", "markets": mk,
                        "oddsFormat": "american", "bookmakers": BOOKMAKERS},
                timeout=20)
            if r.status_code == 422:   # invalid market for this sport/tier
                continue
            r.raise_for_status()
            if r.json().get("bookmakers"):
                working_market = mk
                log.info("first-inning market available: %s", mk)
                break
        except Exception:
            continue

    if working_market is None:
        log.warning("YRFI/NRFI market not available on current Odds API tier — "
                    "dashboard will show model predictions only. Consider "
                    "upgrading to access first inning props.")
        return pd.DataFrame()

    rows = []
    for e in ev:
        try:
            data = requests.get(
                f"{ODDS_API_BASE}/sports/baseball_mlb/events/{e['id']}/odds",
                params={"apiKey": API_KEY, "regions": "us", "markets": working_market,
                        "oddsFormat": "american", "bookmakers": BOOKMAKERS},
                timeout=20).json()
        except Exception:
            continue
        home = ODDS_TO_MLB.get(data.get("home_team"), data.get("home_team"))
        away = ODDS_TO_MLB.get(data.get("away_team"), data.get("away_team"))
        for bm in data.get("bookmakers", []):
            book = FI_BOOK_KEYS.get(bm.get("key"))
            if not book:
                continue
            for m in bm.get("markets", []):
                for o in m.get("outcomes", []):
                    # skip team-total legs that aren't the combined 0.5 line
                    pt = o.get("point")
                    if pt is not None and abs(float(pt) - 0.5) > 0.01:
                        continue
                    side = _fi_outcome_side(o.get("name"))
                    if side is None:
                        continue
                    price = o.get("price")
                    if price is None:
                        continue
                    rows.append({
                        "home_team": home, "away_team": away, "book": book,
                        "side": side, "american": int(price),
                        "implied": american_to_prob(int(price)),
                    })
    return pd.DataFrame(rows)


def _sgo_american(v):
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return None


def fetch_sgo_first_inning(date_str: str) -> tuple[pd.DataFrame, dict]:
    """
    First-inning YRFI/NRFI (0.5 line) from SportsGameOdds.
    Returns (long DataFrame [home_team, away_team, book, side, american, implied],
             fair-odds map {(home_abbr, away_abbr): {'YRFI': implied, 'NRFI': implied}}).
    """
    if not SGO_KEY:
        log.warning("SGO_API_KEY not set — cannot fetch SGO first-inning odds.")
        return pd.DataFrame(), {}

    from datetime import datetime as _dt, timedelta as _td
    d0 = _dt.strptime(date_str, "%Y-%m-%d")
    date_from = d0.strftime("%Y-%m-%dT00:00:00.000Z")
    date_to = (d0 + _td(days=1)).strftime("%Y-%m-%dT08:00:00.000Z")  # cover late/west-coast games

    events, cursor = [], None
    while True:
        params = {"apiKey": SGO_KEY, "leagueID": "MLB", "oddsAvailable": "true",
                  "includeOdds": "true", "limit": 25,
                  "startsAfter": date_from, "startsBefore": date_to}
        if cursor:
            params["cursor"] = cursor
        try:
            body = requests.get(f"{SGO_BASE}/events/", params=params, timeout=30).json()
        except Exception as e:
            log.warning("SGO first-inning fetch error: %s", e)
            break
        batch = body.get("data", [])
        events.extend(batch)
        cursor = body.get("nextCursor")
        if not cursor or len(batch) < 25:
            break

    rows, fair = [], {}
    for ev in events:
        try:
            home = ODDS_TO_MLB.get(ev["teams"]["home"]["names"]["long"],
                                   ev["teams"]["home"]["names"]["long"])
            away = ODDS_TO_MLB.get(ev["teams"]["away"]["names"]["long"],
                                   ev["teams"]["away"]["names"]["long"])
        except (KeyError, TypeError):
            continue
        odds = ev.get("odds", {})
        for key, sidelbl in ((SGO_FI_OVER_KEY, "YRFI"), (SGO_FI_UNDER_KEY, "NRFI")):
            o = odds.get(key)
            if not o:
                continue
            fo = _sgo_american(o.get("fairOdds"))
            if fo is not None:
                fair.setdefault((home, away), {})[sidelbl] = american_to_prob(fo)
            for bk_key, bk_val in (o.get("byBookmaker") or {}).items():
                short = FI_BOOK_KEYS.get(bk_key)
                if not short:
                    continue
                if bk_val.get("available") is False:
                    continue
                # only the 0.5 combined-runs line
                ou = bk_val.get("overUnder") or o.get("bookOverUnder")
                try:
                    if ou is not None and abs(float(ou) - 0.5) > 0.01:
                        continue
                except (TypeError, ValueError):
                    pass
                am = _sgo_american(bk_val.get("odds"))
                if am is None:
                    continue
                dl = bk_val.get("deeplink")
                rows.append({"home_team": home, "away_team": away, "book": short,
                             "side": sidelbl, "american": am,
                             "implied": american_to_prob(am),
                             "deeplink": dl if isinstance(dl, str) and dl else None})
    df = pd.DataFrame(rows)
    if len(df):
        # Reject deeplinks whose *selection* token repeats across games — some
        # books return a generic TEMPLATE selection id (FanDuel: selectionId
        # 7017905 for every game's Over, 7017906 for every Under) that the
        # sportsbook then rejects: "Selection not added / no longer available".
        # The market id in the URL is still unique per game, so we must look at
        # the selection param specifically.
        import re as _re
        def _sel_token(url: str) -> str:
            if not isinstance(url, str):
                return ""
            m = _re.search(r"(?:selectionIds?|selection)=([^&]+)", url)
            if m:
                return m.group(1)
            m = _re.search(r"options=[^&]*?-([^&\-]+)$", url)  # BetMGM: last segment
            return m.group(1) if m else url
        have = df[df["deeplink"].notna()].copy()
        have["_tok"] = have["deeplink"].map(_sel_token)
        for (bk, side), grp in have.groupby(["book", "side"]):
            # a real per-game selection token should be ~unique across the slate
            if len(grp) >= 3 and grp["_tok"].nunique() <= 2:
                df.loc[(df["book"] == bk) & (df["side"] == side), "deeplink"] = None
                log.warning("SGO %s %s deeplinks use a template selection id "
                            "(%d distinct across %d games) — dropped, dashboard "
                            "will link to the book's lobby instead",
                            bk, side, grp["_tok"].nunique(), len(grp))
        n_dl = df["deeplink"].notna().sum()
        log.info("SGO first-inning: %d book lines across %d games (%s) · %d instant deeplinks",
                 len(df), df.groupby(["home_team", "away_team"]).ngroups,
                 ", ".join(sorted(df["book"].unique())), n_dl)
    return df, fair


def build_yrfi_odds_table(date_str: str) -> pd.DataFrame:
    """
    Wide per-game YRFI/NRFI odds table joined to today's game_pk.
    Saved to data/odds/yrfi_odds_{date}.parquet. Returns the DataFrame
    (empty if no market data).

    Primary source: SportsGameOdds (DK / FD / MGM / CZR + a de-vigged fairOdds).
    Fallback: The Odds API totals_1st_1_innings (FD / MGM only).
    """
    long, fair = fetch_sgo_first_inning(date_str)
    source = "SportsGameOdds"
    if long.empty:
        log.info("SGO first-inning empty — falling back to The Odds API.")
        long = fetch_first_inning_odds(date_str)
        fair = {}
        source = "TheOddsAPI"
    out_path = ODDS_DIR / f"yrfi_odds_{date_str}.parquet"
    if long.empty:
        # still write an empty file so downstream knows we tried
        pd.DataFrame(columns=["game_pk"]).to_parquet(out_path, index=False)
        return pd.DataFrame()

    # map to game_pk via probable_pitchers
    pp_path = BASE_DIR / "data" / "processed" / "probable_pitchers.parquet"
    gm_path = BASE_DIR / "data" / "processed" / "game_meta.parquet"
    id2abbr = {}
    if gm_path.exists():
        gm = pd.read_parquet(gm_path)
        for _, r in gm.iterrows():
            id2abbr[int(r["home_team_id"])] = str(r["home_team_abbr"])
            id2abbr[int(r["away_team_id"])] = str(r["away_team_abbr"])
    _ALIAS = {"AZ": "ARI", "ATH": "OAK", "CHW": "CWS", "KCR": "KC", "SDP": "SD",
              "SFG": "SF", "TBR": "TB", "WSN": "WSH", "ANA": "LAA", "FLA": "MIA"}
    def _norm(x):
        x = str(x).upper()
        return _ALIAS.get(x, x)

    key2pk = {}
    if pp_path.exists():
        pp = pd.read_parquet(pp_path)
        pp = pp[pp["game_date"].astype(str) == date_str]
        for _, r in pp.iterrows():
            h = id2abbr.get(int(r["home_team_id"]))
            a = id2abbr.get(int(r["away_team_id"]))
            if h and a:
                key2pk[(_norm(h), _norm(a))] = int(r["game_pk"])

    recs = []
    for (home, away), g in long.groupby(["home_team", "away_team"]):
        pk = key2pk.get((_norm(home), _norm(away)))
        rec = {"game_pk": int(pk) if pk is not None else np.nan,
               "home_team": home, "away_team": away}
        best_y, best_n, best_yb, best_nb = None, None, None, None
        yes_imp, no_imp = [], []
        for book_short in FI_BOOK_KEYS.values():
            gy = g[(g["book"] == book_short) & (g["side"] == "YRFI")]
            gn = g[(g["book"] == book_short) & (g["side"] == "NRFI")]
            oy = int(gy["american"].iloc[0]) if len(gy) else np.nan
            on = int(gn["american"].iloc[0]) if len(gn) else np.nan
            rec[f"yrfi_odds_{book_short}"] = oy
            rec[f"nrfi_odds_{book_short}"] = on
            rec[f"yrfi_link_{book_short}"] = (
                gy["deeplink"].iloc[0] if len(gy) and "deeplink" in gy.columns
                and isinstance(gy["deeplink"].iloc[0], str) else None)
            rec[f"nrfi_link_{book_short}"] = (
                gn["deeplink"].iloc[0] if len(gn) and "deeplink" in gn.columns
                and isinstance(gn["deeplink"].iloc[0], str) else None)
            if len(gy) and len(gn):
                iy, in_ = _devig_pair(american_to_prob(oy), american_to_prob(on))
                rec[f"yrfi_implied_{book_short}"] = round(iy, 4) if pd.notna(iy) else np.nan
                rec[f"nrfi_implied_{book_short}"] = round(in_, 4) if pd.notna(in_) else np.nan
                if pd.notna(iy):
                    yes_imp.append(iy); no_imp.append(in_)
            if pd.notna(oy) and (best_y is None or oy > best_y):
                best_y, best_yb = oy, book_short
            if pd.notna(on) and (best_n is None or on > best_n):
                best_n, best_nb = on, book_short
        rec["best_yrfi_odds"] = best_y
        rec["best_nrfi_odds"] = best_n
        rec["best_yrfi_book"] = best_yb
        rec["best_nrfi_book"] = best_nb
        rec["best_yrfi_link"] = rec.get(f"yrfi_link_{best_yb}") if best_yb else None
        rec["best_nrfi_link"] = rec.get(f"nrfi_link_{best_nb}") if best_nb else None
        # prefer SGO's de-vigged fairOdds; else the cross-book de-vig average
        fpair = fair.get((_norm(home), _norm(away))) or fair.get((home, away)) or {}
        rec["yrfi_market_avg_implied"] = round(fpair["YRFI"], 4) if fpair.get("YRFI") is not None \
            else (round(float(np.mean(yes_imp)), 4) if yes_imp else np.nan)
        rec["nrfi_market_avg_implied"] = round(fpair["NRFI"], 4) if fpair.get("NRFI") is not None \
            else (round(float(np.mean(no_imp)), 4) if no_imp else np.nan)
        rec["odds_source"] = source
        recs.append(rec)

    df = pd.DataFrame(recs)
    df.to_parquet(out_path, index=False)
    log.info("Saved first-inning odds: %d games from %s → %s", len(df), source, out_path)
    return df


if __name__ == "__main__":
    import sys
    date_arg      = next((a for a in sys.argv[1:] if not a.startswith("-")), None)
    skip_props    = "--no-props"      in sys.argv
    force_refresh = "--force-refresh" in sys.argv
    sgo_only      = "--sgo-only"      in sys.argv
    yrfi_only     = "--yrfi-only"     in sys.argv

    if yrfi_only:
        build_yrfi_odds_table(date_arg or datetime.now().strftime("%Y-%m-%d"))
    elif sgo_only:
        run_sgo(date_str=date_arg, force_refresh=force_refresh)
    else:
        run(date_str=date_arg, fetch_props=not skip_props, force_refresh=force_refresh)
        build_yrfi_odds_table(date_arg or datetime.now().strftime("%Y-%m-%d"))
