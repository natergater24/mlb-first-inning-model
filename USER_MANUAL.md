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
  fall back to the book's MLB lobby (↗). Deeplinks are US/NJ-scoped, except
  BetMGM's, which is rewritten to the NC subdomain (`sports.nc.betmgm.com`)
  before display (added 2026-09-18 — SGO's API has no region param, so this is
  a string-replace at capture time; Caesars is still NJ-scoped).
- bet365 / Fanatics are on neither feed for MLB (see `claude-context.txt`).
- If neither source returns first-inning data, `04` writes an empty file and the
  dashboard falls back to model-only.

---

## Dashboard (http://localhost:8501)

Single page, three views.

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
  EV/$100 + "View details →"; a **"Why this edge"** 1-2 sentence caption (added
  2026-09-18, same logic as the detail view below) naming what's driving the
  lean; then a collapsed **➕ Log a Bet** at the bottom.

**Detail view (7 sections + Track a Bet):** model summary + book table — including
a **"Why this edge"** 1-2 sentence caption right under the probability bar (added
2026-09-18) naming what's actually driving the lean; it only names a specific
pitcher or team-vs-pitcher matchup when the number is genuinely significant
(enough starts/PA and a real outlier value), otherwise it just names the
dominant factor group (e.g. "the pitcher NRFI track record") — then **Track a
Bet**, then weather/park (with the HR-factor explainer) · away pitcher
profile · home pitcher profile · away team projected top-5 vs pitcher · home team
projected top-5 · historical YRFI context.

**All Bets view:** reached via "View all N bets →" under the **🧾 Bet log**
table (see Bet Tracker below) — the complete bet history as the same table,
unpaginated. A **← Back** button returns to the previous view.

Sidebar shows model test AUC / Brier / ECE / accuracy and the caveats.

---

## Bet Tracker

**Top of every page (combined record, prominent):**
`W-L | +Xu | +$X,XXX` · open bets · ROI. Next to it a **`$ / unit (new bets)`**
box (default **$25**) and a **Grade bets** button. Two expanders:
- **📚 Record by sportsbook** — W-L, net units and net $ per book.
- **🧾 Bet log** — a true table (`st.dataframe`), one row per bet: **Date**
  (MM/DD/YYYY, the game's date — not when you placed the bet), Game, Side,
  Book, Odds, Units, **Stake ($)**, **Payout ($)**, Status, **1st-Inn Runs**
  (always a whole number). Sorted by game date descending, newest first. Shows
  the **10 most recent bets**; a **"View all N bets →"** button underneath
  drills through to a dedicated **All Bets** page with the complete history (a
  **← Back** button returns to wherever you were). Ungraded bets whose game
  hasn't started still get an inline **Edit or delete** expander below the
  table (side / book / odds / units), on both the summary view and the full
  history page.
  - **Payout** = what you'd get back *including your original stake* — not
    just profit. Won: stake + winnings. Lost: `$0`. Open: the *potential*
    payout if it wins, at the odds you logged. (`Stake ($) = units × that
    bet's own $/unit`, frozen at log time — same value tracker_stats() already
    used for net $, just surfaced per-row now.)

**Units, not dollars.** You stake in **units**. Each bet stores the `$ / unit`
value that was set *when you logged it*, so changing `$ / unit` later only
affects new bets — the historical record keeps every past bet at its original
size. `$ / unit` defaults to the most recent bet's size.

**Logging a bet:** open a game -> **Track a Bet**. Side defaults to the
recommended bet, book to DraftKings, odds pre-fill from that book's line for that
side (editable), **units staked** default to 1 (the caption shows the dollar
equivalent). **Log this bet**.

**Or log directly from the landing page** — every game card has a collapsed
**➕ Log a Bet** section at the bottom (no need to open the game). It shows the
matchup, pitchers and the model's NRFI read for context, then a compact row:
bet type, book, **odds (American, type it in — e.g. `+140` or `-140`)**, and
**stake in units** (default 1, same units convention as the detail-view form).
A line under the row updates live as you type: `{units}u @ ${$/unit} = ${total}`
plus a **potential win** figure computed from the odds you entered. **Log Bet**
saves it (uses the same `$ / unit` box at the top of the page) and a toast
confirms it.

**Grading is automatic** — `bet_tracker.py` reads the 1st-inning runs from the
MLB Stats API linescore (`statsapi.mlb.com/api/v1/game/{pk}/linescore`, free,
keyless — the same source `src/10` uses). NRFI wins on 0 first-inning runs, YRFI
on >= 1. It runs once per browser session on load and on the **Grade bets**
button. First-inning bets never push. `result_units` = `units x odds payout` on a
win, `-units` on a loss. Net dollars = sum of each bet's `result_units x its own
unit_size`.

**Storage:** `data/bet_log.csv` in the repo, read/written on the local
filesystem. Every `save_bets()` call — add/edit/delete/grade — now
**auto-commits and pushes immediately** (`_auto_commit_bet_log()` in
`bet_tracker.py`, added 2026-09-16), on top of `launch.sh` Step 6.5's morning
push. It's best-effort: failure (offline, git missing, nothing changed) is
caught silently, exactly like Step 6.5, and a git failure never blocks the
actual CSV write. **Log and edit bets from the local app only** — the hosted
site has no push credentials, so `_auto_commit_bet_log()` there always fails
silently and a bet logged there only lives in that session's ephemeral
filesystem, gone on the next reboot/redeploy. (This bit real: someone logged
several bets directly on the hosted app and lost all of them on a reboot —
see the 2026-09-16 entry in `claude-prompts.txt`.) `bet_tracker.load_bets()` /
`save_bets()` isolate storage, so a GitHub-API write-back or a real database
could replace the CSV without touching `app.py`'s call sites.

**Decision (2026-09-17): stick with the local-only workflow.** Two options were
considered for durable hosted-side logging — (1) GitHub Contents API
write-back on every bet (small change, but each hosted-side bet would trigger
a real commit and a full Streamlit Cloud redeploy of the app) and (2) an
external store like Google Sheets or a small hosted DB (no redeploy side
effect, but needs a new Google Cloud project + service account + credential,
more setup). Neither was built — logging stays **local-app-only**. Google
Sheets is noted as a **possible future enhancement** if hosted-side logging
is ever needed (see claude-context.txt).

**Hosted-app reminder banner (added 2026-09-17):** `tracker_bar()` in `app.py`
now shows a warning + a "🖥️ Open local app (localhost:8501)" link button at
the very top of every hosted-app page, so this can't quietly happen again.
`_is_hosted()` reuses the same `shutil.which("caffeinate") is None` signal
`run_refresh()` already uses to tell the Mac apart from Streamlit Cloud's
Linux container — the banner is suppressed entirely on the local app (it
only ever shows on the hosted deployment).

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
| `OPENWEATHER_API_KEY` | legacy — `src/03` now uses Open-Meteo (keyless); unused |
| `TWILIO_*`, `ALERT_EMAIL_*` | legacy alerting (`src/archive/06_alert.py`); unused by the YRFI flow |

**Secrets never go in git.** `.gitignore` excludes `env.txt` and any `*env*.txt`
(e.g. an "env copy.txt" backup) — only `env.txt.example` (placeholders) is
committed. If a real key ever lands in a commit, rotate it: it stays in git
history even after the file is removed.

**On Streamlit Community Cloud, `env.txt` doesn't exist** — add `SGO_API_KEY` /
`ODDS_API_KEY` as flat (not nested under a `[section]`) entries in the app's
*Settings → Secrets* instead. Fixed 2026-09-14: this alone used to still leave
book lines blank on the hosted app, even with the keys correctly set — Streamlit
only copies `secrets.toml` into `os.environ` the first time `st.secrets` is
accessed, and nothing did that anywhere in this app (the odds/pipeline scripts
run as subprocesses reading `os.getenv`, not `st.secrets`, directly). `app.py`
now forces that mirroring at startup (`_load_cloud_secrets_into_env()`), so
Cloud secrets actually reach the pipeline subprocesses `run_refresh()` spawns.

---

## Deploying to Streamlit Community Cloud

The hosted app has no local `data/` build, so the files it reads at runtime are
committed to the repo (everything else under `data/` stays git-ignored):

| File | Refreshed |
|---|---|
| `data/models/yrfi_model.pkl` (~23 MB) | full rebuild only — `git add` by hand |
| `data/processed/pitcher_nrfi_profile.parquet` | full rebuild only |
| `data/processed/top5_batter_stats.parquet` | full rebuild only |
| `data/processed/projected_top5_by_team.parquet` | full rebuild only |
| `data/processed/bvp_full_lifetime.parquet` (~14 MB) | full rebuild only — `src/01` |
| `data/processed/todays_yrfi_predictions.parquet` | **daily — auto** (launch.sh Step 6.5) |
| `data/processed/probable_pitchers.parquet` | **daily — auto** (launch.sh Step 6.5) |
| `data/bet_log.csv` | **on every bet + daily — auto** (launch.sh Step 6.5) |

`launch.sh` Step 6.5 runs `git add` → `git commit -m "Daily predictions update
<date>"` → `git push` for the two daily files after inference; Streamlit Cloud
redeploys within a few minutes. The step is failure-tolerant — offline / auth /
nothing-to-commit logs a warning and the dashboard still starts.

After a full `run_pipeline.py`, commit the other five files manually:
```bash
git add data/models/yrfi_model.pkl \
        data/processed/pitcher_nrfi_profile.parquet \
        data/processed/top5_batter_stats.parquet \
        data/processed/projected_top5_by_team.parquet \
        data/processed/bvp_full_lifetime.parquet
git commit -m "Rebuild: refresh model + profiles" && git push
```

**If you ever see a `FileNotFoundError` on the hosted app after clicking Refresh
slate:** it means a support file `src/14_build_todays_yrfi.py` reads isn't in
this whitelist. `game_weather.parquet` and `game_meta.parquet` already degrade
gracefully (empty frame) when missing; `bvp_full_lifetime.parquet` does too as
of 2026-09-14, but the model still needs the *real* file committed for BvP
features to actually feed hosted predictions — a graceful fallback avoids a
crash, it doesn't restore data quality.

**If book lines show empty ("—") on the hosted app after a successful-looking
Refresh slate (no error shown):** this happened for real on 2026-09-17.
`src/04_fetch_odds.py` resolved each game's `game_pk` via `game_meta.parquet`
for the odds table's team-abbreviation lookup — but `game_meta.parquet` isn't
in the Cloud whitelist above, so on Cloud it doesn't exist, every lookup came
back empty, every row's `game_pk` was `NaN`, and `src/14` then dropped all of
them (`dropna(subset=["game_pk"])`) before predictions were built. The odds
fetch itself succeeds and logs "Saved first-inning odds: N games" — the
failure is invisible downstream. Fixed by giving `04_fetch_odds.py` its own
hardcoded team_id → abbreviation table (no file dependency), so this no
longer depends on `game_meta.parquet` existing at all. Diagnosing this
required first fixing `run_refresh()` to print each step's stdout/stderr
unconditionally (previously `subprocess.run(capture_output=True)` silently
swallowed it whenever a step "succeeded," even into the hosted app's own
"Manage app" logs) — if odds ever go empty again, check those logs for what
each pipeline step actually printed before assuming it's a keys/quota issue.

`.devcontainer/devcontainer.json` provides a Python 3.11 container for Codespaces
/ VS Code that installs `requirements.txt` and serves the app on port 8501.

---

## Project Structure

```
src/               numbered pipeline scripts (01–05 legacy/support, 10–14 YRFI)
  yrfi_features.py    shared feature builder (src/13 + src/14)
  park_factors.py     30-team park factors + stadium coords
  fetch_inn1_statcast.py   inning-1-only Statcast downloader (resumable)
  backfill_feeds.py   downloads missing completed game feeds
  archive/            pre-pivot model scripts (06–09)
app.py             Streamlit dashboard (predictions + bet tracker)
bet_tracker.py     bet log I/O, MLB-API grading, bankroll stats
run_pipeline.py    daily / full pipeline runner
launch.sh          clear stale files → daily sequence → push predictions → restart Streamlit → Chrome
scripts/           LaunchAgent shell wrappers
.devcontainer/     Codespaces / VS Code Python container
data/              mostly untracked (rebuild locally); a 6-file runtime subset is committed
  processed/         parquet feature/output tables
  raw/              cached Statcast CSVs, MLB schedules, game feeds
  odds/             yrfi_odds_{date}.parquet
  models/           yrfi_model.pkl (tracked), feature_importance.csv, model_meta.json
  archive/          pre-pivot processed files + models
  bet_log.csv       bet tracker log (tracked)
claude-context.txt   full project briefing (read first)
claude-prompts.txt   dated task log
```
