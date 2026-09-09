#!/usr/bin/env python3
"""
park_factors.py — Static MLB park factors + stadium metadata.

Values are published multi-year (3-yr) park factors on the 100 = league-average
scale (>100 hitter-friendly, <100 pitcher-friendly), rounded from FanGraphs /
Statcast park-factor tables. Used as model features for YRFI/NRFI:
first-inning run scoring tracks strongly with overall park run environment.

`is_dome` == True means the venue is played indoors as the norm — a fixed roof,
or a retractable roof that is closed for the large majority of games (hot / wet
climates). For those parks the dashboard shows a 🏠 icon instead of weather.

Keyed by MLB team abbreviation (matches game_meta.parquet *_team_abbr).

Fields: run_factor, hr_factor, is_dome, stadium_name, lat, lon
"""
from __future__ import annotations

PARK_FACTORS: dict[str, dict] = {
    "ARI": {"run_factor": 103, "hr_factor": 105, "is_dome": True,
            "stadium_name": "Chase Field", "lat": 33.4453, "lon": -112.0667},
    "ATL": {"run_factor": 101, "hr_factor": 103, "is_dome": False,
            "stadium_name": "Truist Park", "lat": 33.8907, "lon": -84.4677},
    "BAL": {"run_factor": 101, "hr_factor": 101, "is_dome": False,
            "stadium_name": "Oriole Park at Camden Yards", "lat": 39.2839, "lon": -76.6218},
    "BOS": {"run_factor": 108, "hr_factor": 100, "is_dome": False,
            "stadium_name": "Fenway Park", "lat": 42.3467, "lon": -71.0972},
    "CHC": {"run_factor": 101, "hr_factor": 102, "is_dome": False,
            "stadium_name": "Wrigley Field", "lat": 41.9484, "lon": -87.6553},
    "CWS": {"run_factor": 101, "hr_factor": 104, "is_dome": False,
            "stadium_name": "Rate Field", "lat": 41.8299, "lon": -87.6338},
    "CIN": {"run_factor": 104, "hr_factor": 114, "is_dome": False,
            "stadium_name": "Great American Ball Park", "lat": 39.0975, "lon": -84.5069},
    "CLE": {"run_factor": 99, "hr_factor": 98, "is_dome": False,
            "stadium_name": "Progressive Field", "lat": 41.4962, "lon": -81.6852},
    "COL": {"run_factor": 112, "hr_factor": 110, "is_dome": False,
            "stadium_name": "Coors Field", "lat": 39.7559, "lon": -104.9942},
    "DET": {"run_factor": 98, "hr_factor": 95, "is_dome": False,
            "stadium_name": "Comerica Park", "lat": 42.3390, "lon": -83.0485},
    "HOU": {"run_factor": 100, "hr_factor": 101, "is_dome": True,
            "stadium_name": "Daikin Park", "lat": 29.7570, "lon": -95.3555},
    "KC":  {"run_factor": 100, "hr_factor": 95, "is_dome": False,
            "stadium_name": "Kauffman Stadium", "lat": 39.0517, "lon": -94.4803},
    "LAA": {"run_factor": 98, "hr_factor": 100, "is_dome": False,
            "stadium_name": "Angel Stadium", "lat": 33.8003, "lon": -117.8827},
    "LAD": {"run_factor": 99, "hr_factor": 102, "is_dome": False,
            "stadium_name": "Dodger Stadium", "lat": 34.0739, "lon": -118.2400},
    "MIA": {"run_factor": 96, "hr_factor": 94, "is_dome": True,
            "stadium_name": "loanDepot park", "lat": 25.7781, "lon": -80.2197},
    "MIL": {"run_factor": 101, "hr_factor": 103, "is_dome": True,
            "stadium_name": "American Family Field", "lat": 43.0280, "lon": -87.9712},
    "MIN": {"run_factor": 100, "hr_factor": 99, "is_dome": False,
            "stadium_name": "Target Field", "lat": 44.9817, "lon": -93.2776},
    "NYM": {"run_factor": 96, "hr_factor": 97, "is_dome": False,
            "stadium_name": "Citi Field", "lat": 40.7571, "lon": -73.8458},
    "NYY": {"run_factor": 101, "hr_factor": 108, "is_dome": False,
            "stadium_name": "Yankee Stadium", "lat": 40.8296, "lon": -73.9262},
    "OAK": {"run_factor": 97, "hr_factor": 92, "is_dome": False,
            "stadium_name": "Sutter Health Park", "lat": 38.5800, "lon": -121.5133},
    "PHI": {"run_factor": 102, "hr_factor": 107, "is_dome": False,
            "stadium_name": "Citizens Bank Park", "lat": 39.9061, "lon": -75.1665},
    "PIT": {"run_factor": 98, "hr_factor": 92, "is_dome": False,
            "stadium_name": "PNC Park", "lat": 40.4469, "lon": -80.0057},
    "SD":  {"run_factor": 96, "hr_factor": 96, "is_dome": False,
            "stadium_name": "Petco Park", "lat": 32.7073, "lon": -117.1566},
    "SF":  {"run_factor": 95, "hr_factor": 90, "is_dome": False,
            "stadium_name": "Oracle Park", "lat": 37.7786, "lon": -122.3893},
    "SEA": {"run_factor": 96, "hr_factor": 97, "is_dome": True,
            "stadium_name": "T-Mobile Park", "lat": 47.5914, "lon": -122.3325},
    "STL": {"run_factor": 99, "hr_factor": 96, "is_dome": False,
            "stadium_name": "Busch Stadium", "lat": 38.6226, "lon": -90.1928},
    "TB":  {"run_factor": 97, "hr_factor": 96, "is_dome": True,
            "stadium_name": "Tropicana Field", "lat": 27.7683, "lon": -82.6534},
    "TEX": {"run_factor": 101, "hr_factor": 102, "is_dome": True,
            "stadium_name": "Globe Life Field", "lat": 32.7473, "lon": -97.0847},
    "TOR": {"run_factor": 100, "hr_factor": 103, "is_dome": True,
            "stadium_name": "Rogers Centre", "lat": 43.6414, "lon": -79.3894},
    "WSH": {"run_factor": 101, "hr_factor": 101, "is_dome": False,
            "stadium_name": "Nationals Park", "lat": 38.8730, "lon": -77.0074},
}

# common alternate abbreviations seen in odds feeds / older data
ALIASES = {
    "CHW": "CWS", "KCR": "KC", "SDP": "SD", "SFG": "SF", "TBR": "TB",
    "WSN": "WSH", "ANA": "LAA", "FLA": "MIA",
    "AZ": "ARI", "ATH": "OAK", "OAK": "OAK",  # MLB now abbreviates Athletics "ATH"
}

LEAGUE_AVG_RUN_FACTOR = 100.0
LEAGUE_AVG_HR_FACTOR = 100.0


def get_park(team_abbr: str | None) -> dict:
    """Return the park-factor record for a team abbr, resolving aliases.
    Unknown team -> neutral league-average record (no dome)."""
    if not team_abbr:
        return {"run_factor": 100, "hr_factor": 100, "is_dome": False,
                "stadium_name": "Unknown", "lat": None, "lon": None}
    key = str(team_abbr).upper()
    key = ALIASES.get(key, key)
    return PARK_FACTORS.get(key, {
        "run_factor": 100, "hr_factor": 100, "is_dome": False,
        "stadium_name": f"Unknown ({team_abbr})", "lat": None, "lon": None,
    })


def park_run_factor(team_abbr: str | None) -> float:
    return float(get_park(team_abbr)["run_factor"])


def park_hr_factor(team_abbr: str | None) -> float:
    return float(get_park(team_abbr)["hr_factor"])


def is_dome(team_abbr: str | None) -> bool:
    return bool(get_park(team_abbr)["is_dome"])


if __name__ == "__main__":
    print(f"PARK_FACTORS: {len(PARK_FACTORS)} teams\n")
    for i, (k, v) in enumerate(PARK_FACTORS.items()):
        if i < 5:
            print(f"  {k}: {v}")
    n_dome = sum(1 for v in PARK_FACTORS.values() if v["is_dome"])
    print(f"\ndome/indoor parks: {n_dome} "
          f"({', '.join(k for k, v in PARK_FACTORS.items() if v['is_dome'])})")
