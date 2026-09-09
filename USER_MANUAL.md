# MLB YRFI / NRFI Pipeline — User Manual

The project answers one question: **will a run be scored in the first inning of
today's games?**

- **YRFI** = Yes Run First Inning (a run scores in the 1st, either team)
- **NRFI** = No Run First Inning

A calibrated RandomForest predicts P(YRFI) per game; the dashboard compares it to
first-inning sportsbook markets (0.5-run O/U) to surface edges.

> The pre-2026-09-09 "first at-bat reach base" model, its blended sub-models, and
> the old 4-page dashboard were archived (`data/archive/`, `src/archive/`), not
> deleted. Ignore any older doc that references `05_build_features.py`,
> `07/08/09_*_model.py`, "Best Bets", or `batter_hits` props.

---

## Critical: Always Use the Full Python Path

```
PYTHON    = /Library/Frameworks/Python.framework/Versions/3.14/bin/python3
STREAMLIT = /Library/Frameworks/Python.framework/Versions/3.14/bin/streamlit
```

Never use bare `python3` / `streamlit` — packages are installed in system Python
only, not in any venv. Start Streamlit with `--server.headless true`.
Use `caffeinate -i` on every long-running command.

---

## Starting the App

### Option A — one command (recommended)
```bash
bash ~/Desktop/Nathan/Claude/MLB_Pipeline/launch.sh
```
Clears stale daily files, runs the daily sequence (probable pitchers, rosters,
weather `--today-only`, first-inning odds, predictions), restarts headless
Streamlit on :8501, opens Chrome.

### Option B — dashboard only (pipeline already ran today)
```bash
cd ~/Desktop/Nathan/Claude/MLB_Pipeline
nohup caffeinate -i /Library/Frameworks/Python.framework/Versions/3.14/bin/streamlit \
  run app.py --server.headless true --server.port 8501 > logs/streamlit.log 2>&1 &
```
Open **http://localhost:8501**

### Option C — run the daily pipeline, then start the dashboard
```bash
cd ~/Desktop/Nathan/Claude/MLB_Pipeline
caffeinate -i /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 run_pipeline.py --today-only
# then Option B
```

### Stopping
```bash
pkill -f streamlit          # or: lsof -ti :8501 | xargs kill
```

---

## Pipeline Runners

| Command | What it does | Time |
|---|---|---|
| `python3 run_pipeline.py --today-only` | Daily: probable pitchers → rosters → weather (`--today-only`) → first-inning odds → `14_build_todays_yrfi.py` | ~0.2 min |
| `python3 run_pipeline.py` | Full rebuild: `10` → `02 --build-lineup-history` → `11` → `12` → `13` (train) → today's data → `14`. Weather rebuilt for all stadiums. | a few min (longer if Statcast/feeds need downloading) |
| `python3 run_pipeline.py --rebuild-history` | Alias for the full rebuild | — |

**Before a full rebuild**, refresh the inputs the daily flow never touches:
```bash
caffeinate -i python3 src/backfill_feeds.py            # missing completed game feeds
caffeinate -i python3 src/fetch_inn1_statcast.py       # 2026 inning-1 pitch data (resumable)
```

---

## Individual Steps

Run from the project root (`cd ~/Desktop/Nathan/Claude/MLB_Pipeline`), always
with the full `python3` path.

| Step | Command | Notes |
|---|---|---|
| Probable pitchers | `src/02_fetch_mlb_api.py --probable-pitchers-only` | refreshes if >2 h old |
| Active rosters | `src/02_fetch_mlb_api.py --rosters-only` | once/day; resolves today's teams from probable_pitchers/schedule (no longer needs game_meta) |
| Today's game feeds | `src/02_fetch_mlb_api.py --todays-feeds-only` | re-run after lineups post (~2–3 h before first pitch) |
| Rebuild lineup-position history | `src/02_fetch_mlb_api.py --build-lineup-history` | weekly, or after a >2-week gap; also writes `batter_game_logs.parquet` |
| Backfill completed feeds | `src/backfill_feeds.py` (`--all` / `--since YYYY-MM-DD`) | run before `--build-lineup-history` after a gap |
| Weather | `src/03_fetch_weather.py --today-only` (daily) / `--rebuild` (all games) | Open-Meteo, free |
| First-inning odds | `src/04_fetch_odds.py --yrfi-only` | SGO primary (free), The Odds API fallback (quota) |
| Inning-1 Statcast | `src/fetch_inn1_statcast.py` | refreshes `statcast_inn1_2026.csv`; then re-run `src/11` |
| Build YRFI outcomes | `src/10_build_yrfi_dataset.py` | 27k+ games 2015-2026 from MLB linescores |
| Pitcher NRFI profiles | `src/11_build_pitcher_nrfi.py` | needs inning-1 Statcast |
| Projected top-5 batters | `src/12_build_top_order_batter.py` | needs `lineup_position_summary.parquet` |
| Train model | `src/13_train_yrfi_model.py` | RF + isotonic calibration + logistic surrogate for the ⚙️ sliders |
| Today's predictions | `src/14_build_todays_yrfi.py` | scores today's games, joins odds, computes edges/EV/recommendation |

After changing any data or model file: re-run `src/14_build_todays_yrfi.py` and
restart Streamlit to verify end to end.

---

## Automatic Daily Schedule (LaunchAgents)

Plists in `~/Library/LaunchAgents/`, shell in `scripts/`:

| Time | Job | Runs |
|---|---|---|
| 10:30 AM | `com.mlb.pipeline.morning` → `scripts/pipeline_morning.sh` | `run_pipeline.py --today-only` |
| 11:00 AM | `com.mlb.pipeline` (inline) | `run_pipeline.py --today-only` |
| 1:00 PM | `com.mlb.pipeline.afternoon` → `scripts/pipeline_afternoon.sh` | probable pitchers + `04 --yrfi-only` + `14` |

The 10:30 and 11:00 jobs overlap — disable one if you don't want the duplicate:
```bash
launchctl unload ~/Library/LaunchAgents/com.mlb.pipeline.morning.plist
```
Neither restarts Streamlit; use `launch.sh` or Option B for that.

---

## Odds

- **Primary: SportsGameOdds** (`SGO_API_KEY`) — free tier, carries the 0.5-run
  first-inning O/U for DraftKings, FanDuel, BetMGM, Caesars plus a de-vigged
  `fairOdds` consensus. Also returns per-selection betslip deeplinks.
- **Fallback: The Odds API** (`ODDS_API_KEY`) — 500 requests/month free; only
  FanDuel + BetMGM expose the first-inning total here. Don't run odds more than
  once/day; don't `--force-refresh` repeatedly when quota is low.
- **Instant "add to betslip" (⚡)** works for **Caesars & BetMGM** only.
  DraftKings has no deeplink; FanDuel's is a template id that FD rejects — both
  fall back to the book's MLB lobby (↗). Deeplinks are US/NJ-scoped.
- bet365 / Fanatics are on neither feed for MLB (see `claude-context.txt`).
- If neither source returns first-inning data, `04` writes an empty file and the
  dashboard falls back to model-only.

---

## Dashboard (http://localhost:8501)

Single page, two views.

**Landing (game list)**
- ⚙️ **Model weighting** expander at the top: one slider per feature group
  (pitcher NRFI record, pitcher stuff, pitcher last-5, lineup power/matchup,
  lineup recent form, ballpark, weather). 100% = the model exactly; sliders scale
  each group's log-odds contribution. "Effective weighting right now" bar +
  "Reset to model" button. Every card/edge/EV/recommendation recomputes live.
- Header with colour-coded "updated N min ago" + "Refresh slate" button.
- Metric row, filters (All / NRFI Edge / YRFI Edge / No Edge), sort, confirmed-only toggle.
- One card per game: teams + records + probable pitchers + NRFI% (left);
  both model odds with the better-value side marked ◄ (middle); 2×4 book-lines
  table DK/FD/MGM/CZR with ⚡/↗ links (right); recommended-bet badge + edge% +
  EV/$100 + "View details →".

**Detail view (7 sections):** model summary + book table · weather/park (with the
HR-factor explainer) · away pitcher profile · home pitcher profile · away team
projected top-5 vs pitcher · home team projected top-5 · historical YRFI context.

Sidebar shows model test AUC / Brier / ECE / accuracy and the caveats.

---

## Known Data Caveats

- Pitcher NRFI profiles & top-5 batter aggregates are **current static values**,
  not point-in-time → mild look-ahead bias in training; test AUC is a small
  over-estimate.
- `batter_rolling.parquet` (Statcast, `src/01`) can be stale → refresh `src/01`
  then `src/12`.
- `lineup_position_summary` / `batter_game_logs` are only as current as the
  cached game feeds → run `src/backfill_feeds.py` then `02 --build-lineup-history`.
- `game_meta.parquet` is only rebuilt by a full `src/02 run()`; the daily flow
  leaves it stale. The roster step works around this; run a full `src/02` (or
  backfill) to bring it current.
- `umpire_zone_adj` is a placeholder (0.0) for almost all games.
- Isotonic calibration + limited feature spread → today's probabilities cluster
  onto a handful of distinct values.

---

## Logs

```bash
cd ~/Desktop/Nathan/Claude/MLB_Pipeline
tail -50 logs/streamlit.log       # dashboard
tail -50 logs/pipeline.log        # pipeline runs
tail -50 logs/launchagent.log     # scheduled jobs
```

---

## Environment Variables

`env.txt` at the project root (not `.env` — macOS hides dot-files). See
`env.txt.example`.

| Var | For |
|---|---|
| `SGO_API_KEY` | SportsGameOdds — primary first-inning odds + betslip deeplinks |
| `ODDS_API_KEY` | The Odds API — fallback first-inning odds (https://the-odds-api.com) |

---

## Project Structure

```
src/               numbered pipeline scripts (01–05 legacy/support, 10–14 YRFI)
  yrfi_features.py    shared feature builder (src/13 + src/14)
  park_factors.py     30-team park factors + stadium coords
  fetch_inn1_statcast.py   inning-1-only Statcast downloader (resumable)
  backfill_feeds.py   downloads missing completed game feeds
  archive/            pre-pivot model scripts (06–09)
app.py             Streamlit dashboard
run_pipeline.py    daily / full pipeline runner
launch.sh          clear stale files → daily sequence → restart Streamlit → Chrome
scripts/           LaunchAgent shell wrappers
data/              not tracked in git — rebuild locally
  processed/         parquet feature/output tables
  raw/              cached Statcast CSVs, MLB schedules, game feeds
  odds/             yrfi_odds_{date}.parquet
  models/           yrfi_model.pkl, feature_importance.csv, model_meta.json
  archive/          pre-pivot processed files + models
claude-context.txt   full project briefing (read first)
claude-prompts.txt   dated task log
```
