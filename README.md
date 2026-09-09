# MLB YRFI/NRFI Prediction Pipeline

A machine learning pipeline that predicts whether a run will be scored in the first inning of MLB games (YRFI = Yes Run First Inning, NRFI = No Run First Inning).

## What This Does
- Pulls MLB schedule, probable pitchers, and active rosters from the free MLB Stats API
- Downloads pitch-level Statcast data from Baseball Savant going back to 2015
- Builds pitcher first inning profiles including NRFI%, velocity, K rate, and pitch type breakdown
- Identifies projected top 5 batters per team and computes their BvP stats vs opposing pitcher
- Trains a RandomForest model to predict YRFI probability for each game
- Compares model implied odds to sportsbook lines to identify edges
- Displays predictions in a Streamlit dashboard

## Setup

### Requirements
- Python 3.14
- See requirements.txt for all dependencies

### Installation
1. Clone this repository
2. Install dependencies: pip install -r requirements.txt
3. Copy env.txt.example to env.txt and add your API keys (SGO_API_KEY, ODDS_API_KEY)
4. Run the historical data build + train: python run_pipeline.py
5. Launch the dashboard (runs the daily update first): bash launch.sh

For the daily update only: python run_pipeline.py --today-only

### Daily Use
Double-click Launch_MLB_App.command on your Desktop to run the daily update and open the dashboard.

## Data Sources
- MLB Stats API — schedules, rosters, lineups, first-inning linescores (ground truth), free
- Baseball Savant (Statcast) — inning-1 pitch-level data, free
- Open-Meteo — weather (historical + forecast), free
- SportsGameOdds — first-inning YRFI/NRFI market odds + betslip deeplinks, free tier (primary)
- The Odds API — first-inning odds fallback, free tier (500 req/month)

## Deploying to Streamlit Community Cloud
The hosted app reads its data from the repo (it has no local `data/` build). `.gitignore` keeps
`data/` excluded **except** the files the app needs at runtime, which are committed:
`data/models/yrfi_model.pkl` and `data/processed/{todays_yrfi_predictions, pitcher_nrfi_profile,
top5_batter_stats, projected_top5_by_team, probable_pitchers}.parquet`.

`launch.sh` (Step 6.5) commits and pushes the two daily files
(`todays_yrfi_predictions.parquet`, `probable_pitchers.parquet`) every morning after inference,
so Streamlit Cloud redeploys with the current slate within a few minutes. The other four change
only on a full rebuild — commit them by hand after `run_pipeline.py`.

`.devcontainer/` gives Codespaces / VS Code a Python 3.11 container that installs
`requirements.txt` and runs the app.

## Project Structure
- src/ — all pipeline scripts numbered 01-14
- app.py — Streamlit dashboard
- run_pipeline.py — master pipeline runner
- data/ — mostly untracked (rebuild locally); a runtime subset is committed for Streamlit Cloud

## Secrets
Real API keys live only in `env.txt`, which is git-ignored (as is any `*env*.txt`). Never commit
a file with real keys — commit `env.txt.example` only. If a key is ever committed, rotate it.

## Model Architecture
Calibrated RandomForest classifier: train 2015-2022, validate 2023, test 2024; production fit
2015-2024 with isotonic calibration on 2025. A logistic-regression surrogate is trained alongside
it to power the dashboard's ⚙️ feature-group weighting sliders (it never replaces the RF output).
Key features: home/away starter first-inning NRFI% (career/season/L5) and 1st-inn ERA, pitcher
stuff (velo, K%, BB%, hard-hit% allowed, first-pitch-strike%), last-5-starts trend, projected
top-5 batter BvP/OBP/HR-per-PA, park factors, weather, month, umpire tendency.
YRFI is close to a coin flip (~50.5% base rate); test AUC ~0.55, well calibrated (ECE ~0.04).

See USER_MANUAL.md for full operating instructions and claude-context.txt for the design briefing.
