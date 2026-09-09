#!/bin/bash
PROJECT="/Users/nathantessler/Desktop/Nathan/Claude/MLB_Pipeline"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
LOG="$PROJECT/logs/launchagent.log"
mkdir -p "$PROJECT/logs"
cd "$PROJECT"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Morning pipeline (10:30am) starting ===" >> "$LOG"
caffeinate -i "$PYTHON" run_pipeline.py --today-only >> "$LOG" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pipeline complete" >> "$LOG"
caffeinate -i "$PYTHON" src/07_train_model.py --infer-only >> "$LOG" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Scoring complete" >> "$LOG"
