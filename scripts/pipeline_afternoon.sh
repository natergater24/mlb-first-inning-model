#!/bin/bash
# 1pm job requires paid Odds API tier — comment out block below if on free tier
PROJECT="/Users/nathantessler/Desktop/Nathan/Claude/MLB_Pipeline"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
LOG="$PROJECT/logs/launchagent.log"
mkdir -p "$PROJECT/logs"
cd "$PROJECT"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Afternoon odds refresh (1:00pm) starting ===" >> "$LOG"
"$PYTHON" src/04_fetch_odds.py --force-refresh >> "$LOG" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Odds fetched" >> "$LOG"
"$PYTHON" src/05_build_features.py >> "$LOG" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Features built" >> "$LOG"
"$PYTHON" src/07_train_model.py --infer-only >> "$LOG" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Scoring complete" >> "$LOG"
