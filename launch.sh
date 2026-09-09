#!/bin/bash
# launch.sh — YRFI / NRFI pipeline launcher
# Runs the daily update, then starts the dashboard and opens Chrome.

set -e

PROJECT="/Users/nathantessler/Desktop/Nathan/Claude/MLB_Pipeline"
PYTHON="/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
STREAMLIT="/Library/Frameworks/Python.framework/Versions/3.14/bin/streamlit"
LOG="$PROJECT/logs/streamlit.log"
PIPELINE_LOG="$PROJECT/logs/pipeline.log"
TODAY="$(date +'%Y-%m-%d')"

cd "$PROJECT"
mkdir -p "$PROJECT/logs"

log() {
    echo "$@"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" >> "$PIPELINE_LOG"
}

log ""
log "=================================================="
log " YRFI/NRFI Pipeline — start $TODAY $(date +'%H:%M:%S')"
log "=================================================="

# ── Step 1: clear stale daily files ─────────────────────────────────────────
echo ""
echo "[$(date +'%H:%M:%S')] Step 1 — clearing stale daily files..."
rm -f "$PROJECT/data/processed/todays_yrfi_predictions.parquet" && echo "  cleared todays_yrfi_predictions.parquet"
rm -f "$PROJECT/data/processed/todays_matchups.parquet"          && echo "  cleared todays_matchups.parquet (legacy)"
rm -f "$PROJECT/data/odds/todays_props.parquet"                  && echo "  cleared todays_props.parquet"
rm -f "$PROJECT/data/odds/odds_${TODAY}.json"                    && echo "  cleared odds_${TODAY}.json"
rm -f "$PROJECT/data/odds/yrfi_odds_${TODAY}.parquet"            && echo "  cleared yrfi_odds_${TODAY}.parquet"
rm -f "$PROJECT/data/raw/mlb_schedule_2026.json"                 && echo "  cleared mlb_schedule_2026.json"

# ── Step 2: probable pitchers ──────────────────────────────────────────────
echo ""
echo "[$(date +'%H:%M:%S')] Step 2 — probable pitchers..."
caffeinate -i "$PYTHON" src/02_fetch_mlb_api.py --probable-pitchers-only

# ── Step 3: active rosters ─────────────────────────────────────────────────
echo ""
echo "[$(date +'%H:%M:%S')] Step 3 — active rosters..."
caffeinate -i "$PYTHON" src/02_fetch_mlb_api.py --rosters-only || true

# ── Step 4: weather ───────────────────────────────────────────────────────
echo ""
echo "[$(date +'%H:%M:%S')] Step 4 — weather forecast (today + upcoming)..."
caffeinate -i "$PYTHON" src/03_fetch_weather.py --today-only || true

# ── Step 5: first-inning odds ─────────────────────────────────────────────
echo ""
echo "[$(date +'%H:%M:%S')] Step 5 — first-inning (YRFI/NRFI) odds..."
caffeinate -i "$PYTHON" src/04_fetch_odds.py || true

# ── Step 6: today's YRFI predictions ──────────────────────────────────────
echo ""
echo "[$(date +'%H:%M:%S')] Step 6 — building today's YRFI predictions..."
caffeinate -i "$PYTHON" src/14_build_todays_yrfi.py

# ── Step 6.5: publish fresh predictions to GitHub (for Streamlit Cloud) ────
echo ""
echo "[$(date +'%H:%M:%S')] Step 6.5 — pushing fresh predictions to GitHub..."
# Never let a git failure (offline, auth, nothing-to-commit) abort the launch.
{
    git add data/processed/todays_yrfi_predictions.parquet \
            data/processed/probable_pitchers.parquet \
            data/bet_log.csv
    if git diff --cached --quiet; then
        echo "  no prediction / bet-log changes to commit"
    else
        git commit -m "Daily predictions update $(date +%Y-%m-%d)" \
            && git push \
            && log "pushed daily predictions to GitHub" \
            || log "WARNING: git push of daily predictions failed — dashboard still starting"
    fi
} || log "WARNING: git publish step errored — continuing"

# ── Step 7: (re)start Streamlit ───────────────────────────────────────────
echo ""
echo "[$(date +'%H:%M:%S')] Step 7 — starting dashboard..."
pkill -f "streamlit run app.py" 2>/dev/null || true
sleep 1

nohup caffeinate -i "$STREAMLIT" run app.py \
    --server.headless true \
    --server.port 8501 \
    >> "$LOG" 2>&1 &
STREAMLIT_PID=$!
echo "  Streamlit PID: $STREAMLIT_PID"

echo "  waiting 5 seconds..."
sleep 5

if lsof -i :8501 -sTCP:LISTEN >/dev/null 2>&1; then
    log "SUCCESS: Streamlit running on port 8501"
    open -a "Google Chrome" http://localhost:8501
else
    echo "  port not open — waiting 5 more seconds..."
    sleep 5
    if lsof -i :8501 -sTCP:LISTEN >/dev/null 2>&1; then
        log "SUCCESS: Streamlit running on port 8501"
        open -a "Google Chrome" http://localhost:8501
    else
        log "ERROR: Streamlit failed to start — see logs/streamlit.log"
        tail -20 "$LOG"
        exit 1
    fi
fi

echo ""
echo "=================================================="
echo " Launch complete — $(date +'%H:%M:%S')"
echo " Dashboard: http://localhost:8501"
echo "=================================================="
echo ""
