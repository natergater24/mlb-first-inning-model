# MLB First At-Bat Pipeline — User Manual

## Critical: Always Use Full Python Path

```
PYTHON  = /Library/Frameworks/Python.framework/Versions/3.14/bin/python3
STREAMLIT = /Library/Frameworks/Python.framework/Versions/3.14/bin/streamlit
```

Never use `python3` or `streamlit` alone — packages are installed in system Python only. The venv does not have packages.

---

## Starting the App

### Option A — Double-click desktop shortcut (recommended)
Double-click `~/Desktop/Launch_MLB_App.command`

Runs the full morning pipeline, then opens Chrome to `http://localhost:8501`.

### Option B — Start dashboard only (pipeline already ran today)

```bash
cd ~/Desktop/Nathan/Claude/MLB_Pipeline
nohup /Library/Frameworks/Python.framework/Versions/3.14/bin/streamlit run app.py --server.headless true > logs/streamlit.log 2>&1 &
```

Then open: **http://localhost:8501**

### Option C — Run pipeline first, then start dashboard

```bash
cd ~/Desktop/Nathan/Claude/MLB_Pipeline

# Step 1: Run today's pipeline (probable pitchers, rosters, feeds, weather, odds, features)
caffeinate /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 run_pipeline.py --today-only

# Step 2: Score today's matchups
caffeinate /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 src/07_train_model.py --infer-only

# Step 3: Start the dashboard
nohup /Library/Frameworks/Python.framework/Versions/3.14/bin/streamlit run app.py --server.headless true > logs/streamlit.log 2>&1 &
```

### Stopping the app
Double-click `~/Desktop/Stop_MLB_App.command`

Or manually:
```bash
pkill -f streamlit
```

---

## Running Individual Pipeline Steps

All commands must be run from the project root:
```bash
cd ~/Desktop/Nathan/Claude/MLB_Pipeline
```

| Step | Command | Notes |
|---|---|---|
| Fetch probable pitchers | `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 src/02_fetch_mlb_api.py --probable-pitchers-only` | Refreshes if >2 hrs old |
| Fetch active rosters | `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 src/02_fetch_mlb_api.py --rosters-only` | Once per day |
| Fetch today's game feeds | `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 src/02_fetch_mlb_api.py --todays-feeds-only` | Re-run after lineups post (~6 PM ET) |
| Fetch weather | `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 src/03_fetch_weather.py` | |
| Fetch odds | `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 src/04_fetch_odds.py` | Costs API quota — see below |
| Build features & matchups | `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 src/05_build_features.py` | |
| Score today's matchups | `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 src/07_train_model.py --infer-only` | |
| Retrain models | `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 src/07_train_model.py` | Not needed daily |
| Rebuild lineup history | `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3 src/02_fetch_mlb_api.py --build-lineup-history` | Run weekly or after >2 week gap |

---

## Automatic Daily Schedule

Two LaunchAgent jobs run automatically:

| Time | What runs |
|---|---|
| 10:30 AM | Full daily pipeline + inference (`run_pipeline.py --today-only` then `07_train_model.py --infer-only`) |
| 1:00 PM | Odds refresh only — **requires paid Odds API tier**. Comment out in `scripts/pipeline_afternoon.sh` if on free tier. |

---

## Dashboard URL and Pages

**URL:** http://localhost:8501

| Page | What it shows |
|---|---|
| Best Bets | Today's matchup cards, sorted by edge. Expand a card to see BvP stats, lineup position history, model vs market, batter/pitcher splits, Savant charts |
| Matchup Detail | Deep dive on one selected matchup |
| Historical Performance | Model accuracy by season, calibration curve, feature importance |
| BvP Explorer | Search any batter-pitcher pair for career stats |

---

## Lineup Position History

The "2026 Stats by Lineup Position" table inside each expanded matchup card shows all at-bats in each batting order spot for the current season, sourced directly from MLB boxscore data.

To refresh this table with the most recent games (run weekly or after a long gap):
```bash
cd ~/Desktop/Nathan/Claude/MLB_Pipeline
caffeinate /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 src/02_fetch_mlb_api.py --build-lineup-history
```

Then rebuild matchups:
```bash
caffeinate /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 src/05_build_features.py
caffeinate /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 src/07_train_model.py --infer-only
```

---

## Odds API Quota (Free Tier)

**500 requests/month.** Each props fetch costs ~3 requests per game (10 games = 30 requests).

- Do not run the pipeline with odds more than once per day
- Do not use `--force-refresh` multiple times in one day when quota is below 30
- When quota hits 0, odds will show as unavailable until the monthly reset
- The dashboard clearly labels when odds data is stale or missing

---

## Environment Variables

Stored in `env.txt` (not `.env` — macOS hides dot-prefixed files).

Key variable: `ODDS_API_KEY` — get a free key at https://the-odds-api.com

---

## Checking Logs

```bash
cd ~/Desktop/Nathan/Claude/MLB_Pipeline

# Streamlit dashboard logs
tail -50 logs/streamlit.log

# Pipeline run logs
tail -50 logs/pipeline.log

# LaunchAgent (scheduled job) logs
tail -50 logs/launchagent.log
```
