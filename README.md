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
3. Copy env.txt.example to env.txt and add your API keys
4. Run the historical data build: python run_pipeline.py --full-rebuild
5. Launch the dashboard: bash launch.sh

### Daily Use
Double-click Launch_MLB_App.command on your Desktop to run the daily update and open the dashboard.

## Data Sources
- Baseball Savant (Statcast) — pitch-level data, free
- MLB Stats API — schedules, rosters, lineups, free
- Open-Meteo — weather forecasts, free
- The Odds API — YRFI/NRFI market odds, free tier available

## Project Structure
- src/ — all pipeline scripts numbered 01-14
- app.py — Streamlit dashboard
- run_pipeline.py — master pipeline runner
- data/ — data directory (not tracked in git, rebuild locally)

## Model Architecture
RandomForest classifier trained on 2015-2022 data, validated on 2023, tested on 2024.
Key features: pitcher first inning NRFI%, last 5 starts trend, top of order BvP stats, park factors, weather, umpire tendency.
