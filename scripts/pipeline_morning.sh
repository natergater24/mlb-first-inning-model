#!/bin/bash
# Morning job (10:30am) — YRFI/NRFI daily update.
# Runs the 5-step daily flow: probable pitchers, rosters, weather, first-inning
# odds, today's predictions. (07_train_model.py was archived in the 2026-09-09
# pivot — inference now happens inside src/14_build_todays_yrfi.py.)
PROJECT="/Users/nathantessler/Desktop/Nathan/Claude/MLB_Pipeline"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
LOG="$PROJECT/logs/launchagent.log"
mkdir -p "$PROJECT/logs"
cd "$PROJECT"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Morning YRFI pipeline (10:30am) starting ===" >> "$LOG"
caffeinate -i "$PYTHON" run_pipeline.py --today-only >> "$LOG" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Daily update complete" >> "$LOG"
