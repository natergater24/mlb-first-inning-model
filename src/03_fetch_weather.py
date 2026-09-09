"""
03_fetch_weather.py
-------------------
Fetches historical + forecast weather for each game using Open-Meteo (free, no key).

Stadium resolution is by MLB team id (schedule.parquet.home_team_id) -> canonical
abbreviation -> coordinates from src/park_factors.py. The old version matched the
team abbreviation as a *substring* of the full team name, which silently:
  - missed 14 stadiums (Rate Field/CWS, Yankee Stadium/NYY, Wrigley/CHC, Citi/NYM,
    Kauffman/KC, Petco/SD, Oracle/SF, Busch/STL, Tropicana/TB, Dodger/LAD,
    Nationals Park/WSH, Angel Stadium/LAA, Sutter Health Park/OAK, and the old
    Florida Marlins name), and
  - MIS-assigned every Seattle Mariners home game to Arizona ("ARI" is inside
    "mARIners") — i.e. Phoenix desert weather on Mariners games.

Weather is fetched per stadium-season in a single ranged API call (≈330 calls for
full history instead of ~25k per-game calls).

Output: data/processed/game_weather.parquet — one row per game:
  game_pk, game_date, home_abbr, temp_f, humidity, wind_speed, wind_dir,
  precipitation, precip_prob, wind_out_component, temp_bucket
"""

import math
import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

import requests
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"
LOG_DIR = BASE_DIR / "logs"
for d in (PROC_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR / "src"))
from park_factors import PARK_FACTORS, get_park  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "weather.log"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

OPEN_METEO_HISTORICAL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

# ── canonical MLB team id -> abbreviation (matches park_factors keys) ────────
TEAM_ID_TO_ABBR = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC", 113: "CIN",
    114: "CLE", 115: "COL", 116: "DET", 117: "HOU", 118: "KC", 119: "LAD",
    120: "WSH", 121: "NYM", 133: "OAK", 134: "PIT", 135: "SD", 136: "SEA",
    137: "SF", 138: "STL", 139: "TB", 140: "TEX", 141: "TOR", 142: "MIN",
    143: "PHI", 144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}
# fallback name -> abbr for rows without a usable team id
NAME_TO_ABBR = {
    "angels": "LAA", "diamondbacks": "ARI", "orioles": "BAL", "red sox": "BOS",
    "cubs": "CHC", "reds": "CIN", "guardians": "CLE", "indians": "CLE",
    "rockies": "COL", "tigers": "DET", "astros": "HOU", "royals": "KC",
    "dodgers": "LAD", "nationals": "WSH", "mets": "NYM", "athletics": "OAK",
    "pirates": "PIT", "padres": "SD", "mariners": "SEA", "giants": "SF",
    "cardinals": "STL", "rays": "TB", "rangers": "TEX", "blue jays": "TOR",
    "twins": "MIN", "phillies": "PHI", "braves": "ATL", "white sox": "CWS",
    "marlins": "MIA", "yankees": "NYY", "brewers": "MIL",
}

# Compass bearing (degrees) from home plate toward center field — used to decide
# whether wind is blowing out (hitter-friendly, +) or in (pitcher-friendly, -).
CF_DIRECTION = {
    "ARI": 345, "ATL": 22, "BAL": 35, "BOS": 355, "CHC": 55, "CWS": 3, "CIN": 0,
    "CLE": 22, "COL": 292, "DET": 345, "HOU": 20, "KC": 0, "LAA": 0, "LAD": 345,
    "MIA": 359, "MIL": 310, "MIN": 0, "NYM": 355, "NYY": 10, "OAK": 350,
    "PHI": 335, "PIT": 0, "SD": 330, "SEA": 0, "SF": 10, "STL": 0, "TB": 25,
    "TEX": 28, "TOR": 10, "WSH": 15,
}
FIRST_PITCH_HOUR = 19  # local time; schedule.parquet has no game time


def resolve_abbr(team_id, team_name) -> str | None:
    try:
        tid = int(team_id)
        if tid in TEAM_ID_TO_ABBR:
            return TEAM_ID_TO_ABBR[tid]
    except (TypeError, ValueError):
        pass
    n = str(team_name or "").lower()
    for key, abbr in NAME_TO_ABBR.items():
        if key in n:
            return abbr
    return None


def wind_component_toward_cf(wind_dir_deg: float, cf_dir_deg: float) -> float:
    """cos of the angle between wind direction and the home-plate→CF bearing.
    +1 = blowing straight out, -1 = straight in."""
    return math.cos(math.radians(wind_dir_deg - cf_dir_deg))


def _nearest_hour_index(times: list[str], day: str, hour: int) -> int | None:
    target = f"{day}T{hour:02d}:00"
    for i, h in enumerate(times):
        if h >= target:
            return i
    return None


def fetch_range(api: str, lat: float, lon: float, start: str, end: str,
                forecast: bool) -> dict:
    hourly = ("temperature_2m,relative_humidity_2m,wind_speed_10m,"
              "wind_direction_10m,precipitation")
    if forecast:
        hourly += ",precipitation_probability"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": hourly,
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
        "timezone": "auto",
    }
    if forecast:
        params["start_date"] = start
        params["end_date"] = end
    else:
        params["start_date"] = start
        params["end_date"] = end
    for attempt in range(1, 4):
        try:
            r = requests.get(api, params=params, timeout=30)
            r.raise_for_status()
            return r.json().get("hourly", {})
        except Exception as e:
            log.warning("  weather fetch attempt %d failed (%s): %s", attempt,
                        api.split("//")[1][:20], e)
            time.sleep(2 * attempt)
    return {}


def temp_bucket(t) -> str | None:
    if t is None:
        return None
    return "cold" if t < 50 else "cool" if t < 65 else "normal" if t < 80 else "warm"


def _game_universe() -> pd.DataFrame:
    """All games needing weather: yrfi_outcomes (full 2015-2026 history) unioned
    with the current schedule.parquet (today / upcoming games not yet final)."""
    frames = []
    yo = PROC_DIR / "yrfi_outcomes.parquet"
    if yo.exists():
        y = pd.read_parquet(yo)[["game_pk", "game_date", "home_team_id", "home_team"]]
        y = y.rename(columns={"home_team": "home_team_name"})
        frames.append(y)
    sp = PROC_DIR / "schedule.parquet"
    if sp.exists():
        s = pd.read_parquet(sp)[["game_pk", "game_date", "home_team_id", "home_team_name"]]
        frames.append(s)
    if not frames:
        return pd.DataFrame()
    allg = pd.concat(frames, ignore_index=True)
    allg["game_pk"] = allg["game_pk"].astype("int64")
    return allg.drop_duplicates(subset=["game_pk"], keep="last")


def run(today_only: bool = False, rebuild: bool = False):
    schedule = _game_universe()
    if schedule.empty:
        log.error("No game source found (need yrfi_outcomes.parquet or schedule.parquet).")
        return
    schedule["_abbr"] = [
        resolve_abbr(r.get("home_team_id"), r.get("home_team_name"))
        for _, r in schedule.iterrows()
    ]
    unresolved = schedule[schedule["_abbr"].isna()]
    if len(unresolved):
        log.warning("%d games with unresolvable home team — skipped: %s",
                    len(unresolved),
                    unresolved["home_team_name"].dropna().unique()[:5])
    schedule = schedule.dropna(subset=["_abbr"]).copy()
    schedule["game_date"] = schedule["game_date"].astype(str).str[:10]
    schedule["_year"] = schedule["game_date"].str[:4].astype(int)

    today = datetime.now().date()
    weather_path = PROC_DIR / "game_weather.parquet"
    existing = pd.read_parquet(weather_path) if weather_path.exists() else pd.DataFrame()

    if rebuild:
        existing = pd.DataFrame()
        log.info("REBUILD: refetching weather for every scheduled game.")
        targets = schedule
    else:
        done = set(existing["game_pk"].tolist()) if len(existing) else set()
        # always refetch games from today onward (forecast changes; today's games
        # were previously written from the lagging archive API or skipped)
        future_cut = (today - timedelta(days=1)).isoformat()
        stale = schedule[schedule["game_date"] >= future_cut]["game_pk"].tolist()
        if today_only:
            targets = schedule[schedule["game_date"] >= future_cut]
        else:
            targets = schedule[
                (~schedule["game_pk"].isin(done)) | schedule["game_pk"].isin(stale)
            ]
        if len(existing):
            existing = existing[~existing["game_pk"].isin(targets["game_pk"])]

    if targets.empty:
        log.info("No games need weather. (%d already cached)", len(existing))
        return

    log.info("Fetching weather for %d games across %d stadium-seasons...",
             len(targets), targets.groupby(["_abbr", "_year"]).ngroups)

    rows = []
    for (abbr, year), grp in targets.groupby(["_abbr", "_year"]):
        park = get_park(abbr)
        lat, lon = park.get("lat"), park.get("lon")
        if lat is None:
            log.warning("  no coordinates for %s — skipping %d games", abbr, len(grp))
            continue
        cf_deg = CF_DIRECTION.get(abbr, 0)
        days = sorted(grp["game_date"].unique())
        start, end = days[0], days[-1]

        # split the span at the archive/forecast boundary (~5-day archive lag)
        boundary = (today - timedelta(days=5)).isoformat()
        segments = []
        hist_days = [d for d in days if d < boundary]
        fc_days = [d for d in days if d >= boundary]
        if hist_days:
            segments.append((OPEN_METEO_HISTORICAL, hist_days[0], hist_days[-1], False))
        if fc_days:
            fc_end = min(fc_days[-1], (today + timedelta(days=15)).isoformat())
            segments.append((OPEN_METEO_FORECAST, min(fc_days[0], today.isoformat()),
                             fc_end, True))

        hourly_by_seg = []
        for api, s, e, is_fc in segments:
            h = fetch_range(api, lat, lon, s, e, is_fc)
            if h.get("time"):
                hourly_by_seg.append(h)
            time.sleep(0.3)

        for _, g in grp.iterrows():
            day = g["game_date"]
            wx = {}
            for h in hourly_by_seg:
                times = h.get("time", [])
                idx = _nearest_hour_index(times, day, FIRST_PITCH_HOUR)
                if idx is None:
                    continue
                def gv(key):
                    arr = h.get(key, [])
                    return arr[idx] if idx < len(arr) else None
                wx = {
                    "temp_f": gv("temperature_2m"),
                    "humidity": gv("relative_humidity_2m"),
                    "wind_speed": gv("wind_speed_10m"),
                    "wind_dir": gv("wind_direction_10m"),
                    "precipitation": gv("precipitation"),
                    "precip_prob": gv("precipitation_probability"),
                }
                break
            if wx.get("wind_speed") is not None and wx.get("wind_dir") is not None:
                wx["wind_out_component"] = wx["wind_speed"] * wind_component_toward_cf(
                    wx["wind_dir"], cf_deg)
            else:
                wx["wind_out_component"] = 0.0
            wx["temp_bucket"] = temp_bucket(wx.get("temp_f"))
            rows.append({"game_pk": int(g["game_pk"]), "game_date": day,
                         "home_abbr": abbr, **wx})

        log.info("  %s %d: %d games", abbr, year, len(grp))

    new_df = pd.DataFrame(rows)
    if len(new_df):
        combined = pd.concat([existing, new_df], ignore_index=True) if len(existing) else new_df
        combined.drop_duplicates(subset=["game_pk"], keep="last", inplace=True)
        combined.to_parquet(weather_path, index=False)
        log.info("Saved weather: %d total games (%d stadiums) → %s",
                 len(combined), combined["home_abbr"].nunique(), weather_path)
    else:
        log.info("No new weather rows produced.")


if __name__ == "__main__":
    run(today_only="--today-only" in sys.argv, rebuild="--rebuild" in sys.argv)
