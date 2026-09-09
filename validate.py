"""
validate.py
-----------
Quick sanity-check script. Run this before the full pipeline to confirm:
  1. Dependencies are installed
  2. Baseball Savant is reachable
  3. MLB Stats API is reachable
  4. The Odds API key is valid (if configured)
  5. Open-Meteo weather API is reachable
  6. Output directories are writable

Usage: python validate.py
"""

import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

PASS = "✓"
FAIL = "✗"
WARN = "⚠"

def check(label: str, fn) -> bool:
    try:
        result = fn()
        if result is True or result is None:
            print(f"  {PASS} {label}")
            return True
        elif result is False:
            print(f"  {FAIL} {label}")
            return False
        else:
            print(f"  {PASS} {label}: {result}")
            return True
    except Exception as e:
        print(f"  {FAIL} {label}: {e}")
        return False

print("\n" + "=" * 60)
print("MLB PIPELINE VALIDATION")
print("=" * 60)

all_ok = True

# ── Dependencies ─────────────────────────────────────────────────────────────
print("\n[1] Python Dependencies")

def check_import(name):
    def fn():
        __import__(name)
        return True
    return fn

for pkg in ["pandas","numpy","requests","dotenv","pyarrow","bs4","lxml"]:
    ok = check(pkg, check_import(pkg))
    all_ok = all_ok and ok

try:
    import pybaseball
    check("pybaseball", lambda: True)
except ImportError:
    print(f"  {WARN} pybaseball: Not installed — will use direct Savant CSV endpoint instead")
    print("       Install with: pip install pybaseball")

# ── Directories ────────────────────────────────────────────────────────────
print("\n[2] Directory Structure")

for d in ["data/raw","data/processed","data/odds","logs"]:
    path = BASE_DIR / d
    def mk(p=path):
        p.mkdir(parents=True, exist_ok=True)
        (p / ".gitkeep").touch()
        return True
    check(f"data/{d}", mk)

# ── Network ────────────────────────────────────────────────────────────────
print("\n[3] Network Connectivity")

import requests as req

def check_url(url, expected_status=200, name=None):
    def fn():
        r = req.get(url, timeout=10, headers={"User-Agent": "mlb-pipeline-validate/1.0"})
        if r.status_code == expected_status:
            return f"HTTP {r.status_code}"
        return f"HTTP {r.status_code} (expected {expected_status})"
    return fn

check("Baseball Savant",
      check_url("https://baseballsavant.mlb.com/statcast_search", 200))

check("MLB Stats API",
      check_url("https://statsapi.mlb.com/api/v1/sports/1", 200))

check("Open-Meteo (weather)",
      check_url("https://api.open-meteo.com/v1/forecast?latitude=40.7&longitude=-74.0&hourly=temperature_2m&forecast_days=1", 200))

# ── Odds API ────────────────────────────────────────────────────────────────
print("\n[4] The Odds API")

from dotenv import load_dotenv
import os
load_dotenv(BASE_DIR / "env.txt")
api_key = os.getenv("ODDS_API_KEY", "")

if not api_key or api_key == "your_key_here":
    print(f"  {WARN} ODDS_API_KEY not set in .env")
    print("       Get a free key at: https://the-odds-api.com")
    print("       Free tier: 500 requests/month")
else:
    def check_odds_api():
        r = req.get(
            "https://api.the-odds-api.com/v4/sports",
            params={"apiKey": api_key},
            timeout=10
        )
        if r.status_code == 200:
            remaining = r.headers.get("x-requests-remaining", "?")
            return f"Valid key. {remaining} requests remaining."
        elif r.status_code == 401:
            raise Exception("Invalid API key")
        else:
            raise Exception(f"HTTP {r.status_code}")
    check("Odds API key", check_odds_api)

# ── Alert config ────────────────────────────────────────────────────────────
print("\n[5] Alert Configuration (optional)")

email_from = os.getenv("ALERT_EMAIL_FROM", "")
twilio_sid = os.getenv("TWILIO_ACCOUNT_SID", "")

if email_from and email_from != "your@gmail.com":
    print(f"  {PASS} Email alerts configured ({email_from})")
else:
    print(f"  {WARN} Email alerts not configured (optional — set in .env)")

if twilio_sid and twilio_sid != "your_sid_here":
    print(f"  {PASS} Twilio SMS configured")
else:
    print(f"  {WARN} SMS alerts not configured (optional — set in .env)")

# ── Quick Savant test ────────────────────────────────────────────────────────
print("\n[6] Baseball Savant CSV Endpoint Test")

def test_savant():
    url = "https://baseballsavant.mlb.com/statcast_search/csv"
    params = {
        "all": "true", "hfGT": "R|", "hfSea": "2024|", "hfInn": "1|",
        "hfOuts": "0|", "player_type": "batter", "min_pitches": "0",
        "min_results": "0", "group_by": "name", "sort_col": "pitches",
        "sort_order": "desc", "min_abs": "0", "type": "details",
        "game_date_gt": "2024-04-01", "game_date_lt": "2024-04-02",
    }
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    r = req.get(url, params=params, headers=headers, timeout=30)
    if r.status_code == 200 and "pitch_type" in r.text[:500]:
        import io
        import pandas as pd
        df = pd.read_csv(io.StringIO(r.text), low_memory=False)
        return f"{len(df)} pitches returned for 2024-04-01"
    elif r.status_code == 200:
        return f"HTTP 200 but unexpected content: {r.text[:100]}"
    else:
        raise Exception(f"HTTP {r.status_code}")

check("Savant CSV endpoint", test_savant)

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("NEXT STEPS:")
print("=" * 60)
print("""
1. Set your Odds API key in .env:
   ODDS_API_KEY=your_key_here

2. Run the full historical pull (first time only, ~30-90 min):
   python run_pipeline.py

   Or start with a single recent season to test:
   python src/01_fetch_statcast.py 2024

3. Set up daily cron job:
   bash setup_cron.sh

4. After data is collected, train the model:
   python src/train_model.py  (coming in next phase)
""")
