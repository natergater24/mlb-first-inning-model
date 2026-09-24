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
- Tracks the bets you place (side, book, odds, units), auto-grades them from the MLB Stats API,
  and shows a running W-L / units / dollars record at the top of the page

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
top5_batter_stats, projected_top5_by_team, probable_pitchers, bvp_full_lifetime}.parquet`.

`launch.sh` (Step 6.5) commits and pushes the two daily files
(`todays_yrfi_predictions.parquet`, `probable_pitchers.parquet`) every morning after inference,
so Streamlit Cloud redeploys with the current slate within a few minutes. The other five change
only on a full rebuild — commit them by hand after `run_pipeline.py`.

`src/14_build_todays_yrfi.py` falls back to an empty frame for any support file that's genuinely
optional on Cloud (`game_weather.parquet`, `game_meta.parquet`, and now `bvp_full_lifetime.parquet`
too) rather than crashing — but the batter-vs-pitcher feature is real model input, so the hosted
predictions only match local quality once that file is actually committed.

**Odds keys on Cloud (`ODDS_API_KEY` / `SGO_API_KEY`):** add them in the app's *Settings → Secrets*
as flat (non-nested) `KEY = "value"` lines, matching the names in `env.txt.example`. Setting them
there is necessary but not sufficient — see the "Secrets" section below for why `app.py` also has
to explicitly pull `st.secrets` into `os.environ` before those keys reach the pipeline scripts.

`.devcontainer/` gives Codespaces / VS Code a Python 3.11 container that installs
`requirements.txt` and runs the app.

The in-app **"Refresh slate"** button (`run_refresh()` in `app.py`) runs the daily pipeline
scripts via `subprocess` using `sys.executable` (whatever Python is running the app) and only
prepends `caffeinate` when that binary exists, so it works unmodified on both a local macOS run
and the hosted Cloud container — it previously hardcoded a macOS-only Python path and
unconditionally shelled out to `caffeinate`, which crashed the hosted app with a
`FileNotFoundError` the first time anyone clicked it there (fixed 2026-09-14).

**Bet logging stays local-only.** The hosted app has no git push credentials, so a bet logged
there only lives in that session's ephemeral filesystem and is lost on the next reboot/redeploy
(this happened for real once — see `claude-prompts.txt`, 2026-09-16). `tracker_bar()` now shows a
warning + a direct "Open local app" link at the top of every hosted-app page (`_is_hosted()`
reuses the same caffeinate-presence check as `run_refresh()`) so this can't be missed. A Google
Sheets–backed storage layer would make hosted-side logging durable and is noted as a possible
future enhancement in `claude-context.txt`, not built.

## Project Structure
- src/ — all pipeline scripts numbered 01-14
- app.py — Streamlit dashboard
- run_pipeline.py — master pipeline runner
- data/ — mostly untracked (rebuild locally); a runtime subset is committed for Streamlit Cloud

## Secrets
Real API keys live only in `env.txt`, which is git-ignored (as is any `*env*.txt`). Never commit
a file with real keys — commit `env.txt.example` only. If a key is ever committed, rotate it.

**On Streamlit Community Cloud**, `env.txt` doesn't exist (it's never in the repo), so keys go in
the app's *Settings → Secrets* instead. That alone isn't enough, though: Streamlit only mirrors
`secrets.toml` values into `os.environ` the first time something in the app actually accesses
`st.secrets` — and the pipeline scripts (`src/04_fetch_odds.py` etc.) read keys via `os.getenv`
in a plain `subprocess.run()` child process, which only inherits what's already in `os.environ`
at the time `run_refresh()` spawns it. `app.py` calls `_load_cloud_secrets_into_env()` at import
time specifically to force that mirroring early (walking one level of `[section]` nesting too),
so Cloud secrets actually reach those subprocesses. Fixed 2026-09-14 — before this, keys
correctly set in Cloud's Secrets panel still silently never reached the odds fetch, producing
empty book lines with no error (identical symptom to the keys being unset at all).

## Model Architecture
Calibrated RandomForest classifier: train 2015-2022, validate 2023, test 2024; production fit
2015-2024 with isotonic calibration on 2025. A logistic-regression surrogate is trained alongside
it to power the dashboard's ⚙️ feature-group weighting sliders (it never replaces the RF output).
Key features: home/away starter first-inning NRFI% (career/season/L5) and 1st-inn ERA, pitcher
stuff (velo, K%, BB%, hard-hit% allowed, first-pitch-strike%), last-5-starts trend, projected
top-5 batter BvP/OBP/HR-per-PA, park factors, weather, month, umpire tendency.
YRFI is close to a coin flip (~50.5% base rate); test AUC ~0.55, well calibrated (ECE ~0.04).

**Pitcher small-sample correction (2026-09-21):** diagnosed against a real case (ATH@CLE,
2026-09-18) where the model priced YRFI at 85%/-580 off a pitcher's 25% NRFI rate over just 4
starts, while every sportsbook had the game near a coin flip. Two layers, in `src/yrfi_features.py`:
(1) the three NRFI-rate features (career/season/L5) and `first_inn_era` are Bayesian-shrunk toward
a league prior, weighted by how many starts back them (`_shrink()`); (2) confirmed via a live
spot-check that shrinking the *input* alone barely moved this RandomForest's output (85%→84%,
since a tree ensemble re-derives separation from whatever correlated signal remains, unlike a
linear model), so the model's *output* probability is also blended toward a neutral 0.5, weighted
by `min(home_starts, away_starts)` (`blend_toward_neutral()`) — a debut pitcher (0 starts) forces
an exact coin flip; an established pair drifts under 1pp. `model_yrfi_prob_raw` and
`confidence_blend_weight` are kept alongside the blended `model_yrfi_prob` for transparency.

**Lineup recent-form small samples (2026-09-24, display-only — see caveat):** the same
overconfidence problem exists on the batter side — early in a season, "last 30 days" (L7/L30 OBP)
might be 12 real plate appearances, fed to the model with the same weight as a genuine 100+ PA
window later in the year. A training-time fix (shrinking L7/L30 toward league-average OBP, mirroring
the pitcher fix) was tried and **reverted the same day**: `top5_batter_stats.parquet` is a single
frozen snapshot reused for *every* historical training row regardless of that game's real date, so
a PA-based shrink never sees real early-season variation during training — it just adds noise, and
measurably dropped test AUC below the regression budget (confirmed via retrain). Fixing this
properly needs point-in-time L7/L30 reconstruction (already on the outstanding list below).
Shipped instead: `app.py`'s `_thin_lineup_form_reason()` — an inference-only caution flag (same
UI pattern as the pitcher low-confidence warning) using **today's real, non-frozen** L7/L30 PA
counts, which do vary meaningfully day to day. Doesn't change the model's probability, just warns
when the lineup-recent-form signal driving a lean is backed by too few current-season PAs.

**Notable batter-vs-pitcher matchups (2026-09-24):** the landing page now surfaces the day's most
significant individual BvP matchups (a projected top-5 hitter with >=15 career PA against today's
actual starter, hitting well above/below league norms specifically against him) — pure display,
computed in `matchup_highlights.py` from data the app already loads, ranked by PA.

See USER_MANUAL.md for full operating instructions and claude-context.txt for the design briefing.
