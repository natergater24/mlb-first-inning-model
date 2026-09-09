#!/bin/bash
# Afternoon job (1:00pm) — refresh first-inning odds + rebuild today's predictions
# once lineups / lines have firmed up. SportsGameOdds (primary) is free; The Odds
# API fallback costs quota, so keep this to once/day. (05_build_features.py and
# 07_train_model.py were archived in the 2026-09-09 pivot.)
PROJECT="/Users/nathantessler/Desktop/Nathan/Claude/MLB_Pipeline"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
LOG="$PROJECT/logs/launchagent.log"
mkdir -p "$PROJECT/logs"
cd "$PROJECT"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Afternoon odds refresh (1:00pm) starting ===" >> "$LOG"
caffeinate -i "$PYTHON" src/02_fetch_mlb_api.py --probable-pitchers-only >> "$LOG" 2>&1
caffeinate -i "$PYTHON" src/04_fetch_odds.py --yrfi-only >> "$LOG" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Odds fetched" >> "$LOG"
caffeinate -i "$PYTHON" src/14_build_todays_yrfi.py >> "$LOG" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Predictions rebuilt" >> "$LOG"
