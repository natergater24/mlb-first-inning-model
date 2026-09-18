# Pitcher Small-Sample Model Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the YRFI/NRFI model from producing overconfident predictions (e.g. ATH@CLE's -580 YRFI, 85% probability, while every book had it near a coin flip) when the pick rests on a pitcher with almost no MLB track record.

**Architecture:** Options 3+4+5 from the 2026-09-18 diagnosis, applied together as one feature-engineering change in the shared `yrfi_features.py` builder (used identically by training and inference), followed by a retrain, then a two-legged validation (aggregate historical backtest + targeted manual spot-check) before the new model artifact replaces the live one.

**Tech Stack:** pandas / numpy / scikit-learn (RandomForestClassifier + isotonic CalibratedClassifierCV), existing repo conventions (venv at `venv/`, pandas parquet files, git-tracked model artifact).

**Spec:** This plan *is* the spec — written directly from the live diagnosis in this session (ATH@CLE game_pk 824383, 2026-09-18 slate) and direct inspection of `src/yrfi_features.py`, `src/11_build_pitcher_nrfi.py`, `src/13_train_yrfi_model.py`, `src/14_build_todays_yrfi.py`, and `data/models/yrfi_model_meta.json`. No separate spec doc exists.

## Global Constraints

- Do NOT touch `data/processed/pitcher_nrfi_profile.parquet`'s existing columns or the raw `seas_nrfi_pct` / `l5_nrfi_pct` / `career (nrfi_pct)` values shown to the user in the UI (game cards, "why this edge", the low-confidence flag) — those must keep reading the *raw* rate. Shrinkage applies only inside the model's own feature matrix in `yrfi_features.py`.
- Do NOT overwrite `data/models/yrfi_model.pkl` (or its `meta.json`/`feature_importance.csv` siblings) with an unvalidated retrain. It's git-tracked; treat "committed" as the go/no-go gate, not "ran once."
- The current model has real, live money behind it (6-3, +1.8u per the dashboard) — this is a strict non-regression exercise, not a green-field retrain. If validation is ambiguous, the default action is **do not ship**, not "ship and monitor."
- Every new model-input feature this plan adds must be added to `feature_columns()` in `src/yrfi_features.py` — that function is the single source of truth `src/13` (training) and `src/14` (inference) both read from.

---

## Known limitation (read before Task 4)

`yrfi_features.py`'s own docstring already documents: *"pitcher profile stats ... are the current static values (no point-in-time reconstruction), so career features carry mild look-ahead bias."* This means a pitcher who had 3 career starts as of a 2024 test-set game may show 150 career starts in *today's* profile snapshot when the 2024 backtest re-runs — so a naive "slice the 2024 test set by current total_starts < 10" filter will miss most of the actual thin-sample historical cases (they've since accumulated a normal-looking start count). The aggregate 2024 backtest in Task 3 is still required (it's the standard non-regression check), but it cannot cleanly prove the shrinkage fix helps thin-sample games specifically — that's what Task 4's live spot-check is for. Building true point-in-time reconstruction is a separate, larger project already tracked in `claude-context.txt`'s OUTSTANDING section — out of scope here.

---

### Task 1: Bayesian shrinkage on the three NRFI-rate features

**Files:**
- Modify: [src/yrfi_features.py](MLB_Pipeline/src/yrfi_features.py) (`_pitcher_features()`, lines ~82-101)
- Test: `tests/test_yrfi_features.py` (new file — **confirmed 2026-09-18: no `tests/` directory and no `pytest` in `venv/` exist in this repo at all.** This is the first test file the project will have.)
- Modify: [requirements.txt](MLB_Pipeline/requirements.txt) (add `pytest>=8.0.0` under a new `# Testing` comment, near the other tooling deps)

**Interfaces:**
- Produces: `_shrink(rate: float | None, n: float | None, prior: float, k: float) -> float` — a pure function, always returns a float (never NaN/None), used by `_pitcher_features()` for `career_nrfi_pct`, `seas_nrfi_pct`, `l5_nrfi_pct`.

- [ ] **Step 0: Install pytest (one-time, first test file in this repo)**

```bash
cd /Users/nathantessler/Desktop/Nathan/Claude/MLB_Pipeline
./venv/bin/pip install pytest
echo "" >> requirements.txt
echo "# Testing" >> requirements.txt
echo "pytest>=8.0.0" >> requirements.txt
mkdir -p tests
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_yrfi_features.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yrfi_features import _shrink

def test_shrink_pulls_thin_sample_toward_prior():
    # Mason Barnett's real 2026-09-18 case: 25.0% seas_nrfi_pct over 4 starts
    result = _shrink(rate=25.0, n=4, prior=70.0, k=12)
    assert result == round((4 * 25.0 + 12 * 70.0) / (4 + 12), 4)
    assert 55 < result < 62  # pulled well above the raw 25%, not all the way to 70

def test_shrink_barely_moves_large_sample():
    # An established pitcher: 68% over 30 starts should stay close to 68
    result = _shrink(rate=68.0, n=30, prior=70.0, k=12)
    assert 67.0 < result < 69.0

def test_shrink_returns_exact_prior_for_debut_pitcher():
    # No profile row at all -> n is None/0 -> shrunk value IS the league prior
    assert _shrink(rate=None, n=None, prior=70.0, k=12) == 70.0
    assert _shrink(rate=None, n=0, prior=70.0, k=12) == 70.0

def test_shrink_never_returns_nan():
    import math
    result = _shrink(rate=float("nan"), n=float("nan"), prior=70.0, k=12)
    assert not math.isnan(result)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/nathantessler/Desktop/Nathan/Claude/MLB_Pipeline
./venv/bin/python -m pytest tests/test_yrfi_features.py -v
```

Expected: FAIL with `ImportError: cannot import name '_shrink'`.

- [ ] **Step 3: Implement `_shrink()` and wire it into `_pitcher_features()`**

Add near the top of `src/yrfi_features.py`, right after `PRIMARY_PITCH_CODES`:

```python
# ── small-sample correction for the NRFI-rate pitcher features ──────────────
# career_nrfi_pct / seas_nrfi_pct / l5_nrfi_pct are observed proportions built
# on wildly varying sample sizes (a rookie's 4-start season rate vs. a
# 15-year veteran's 300-start career rate) -- fed to the model raw, with no
# correction, a noisy 4-start rate moves the RandomForest exactly as hard as
# a reliable 300-start one. This is a standard credibility/empirical-Bayes
# blend: shrunk = (n*observed + k*prior) / (n+k). k is the "pseudo-start"
# weight -- how many prior-strength starts the league-average prior is worth.
# Diagnosed 2026-09-18 against a real case: ATH@CLE priced YRFI at 85%/-580
# off Mason Barnett's 25% season NRFI rate over just 4 starts, while every
# sportsbook had the game near a coin flip.
NRFI_LEAGUE_PRIOR = 70.0   # matches app.py's _WHY_EDGE_LEAGUE_NRFI constant
SHRINK_K_CAREER = 12.0     # ~12 "pseudo-starts" of prior weight
SHRINK_K_SEASON = 12.0
SHRINK_K_L5 = 4.0          # L5's full sample is only 5 starts -- a k of 12
                           # would swamp it completely; 4 tempers without
                           # neutering it (a perfect 5/5 still moves ~4/9 of
                           # the way to the prior, not all the way).


def _shrink(rate: float | None, n: float | None, prior: float, k: float) -> float:
    """Credibility-weighted blend of an observed rate toward a prior. Always
    returns a real float -- a missing/zero-sample pitcher (debut) returns
    exactly `prior`, replacing what used to be a NaN fed into training-median
    imputation with an explicit, principled default."""
    n = 0.0 if n is None or pd.isna(n) else float(n)
    rate = prior if rate is None or pd.isna(rate) else float(rate)
    return round((n * rate + k * prior) / (n + k), 4)
```

Then in `_pitcher_features()`, replace the three raw assignments:

```python
    out[f"{prefix}_pitcher_career_nrfi_pct"] = p.get("nrfi_pct")
    out[f"{prefix}_pitcher_seas_nrfi_pct"] = p.get("seas_nrfi_pct")
    out[f"{prefix}_pitcher_l5_nrfi_pct"] = p.get("l5_nrfi_pct")
```

with:

```python
    out[f"{prefix}_pitcher_career_nrfi_pct"] = _shrink(
        p.get("nrfi_pct"), p.get("total_starts"), NRFI_LEAGUE_PRIOR, SHRINK_K_CAREER)
    out[f"{prefix}_pitcher_seas_nrfi_pct"] = _shrink(
        p.get("seas_nrfi_pct"), p.get("seas_starts"), NRFI_LEAGUE_PRIOR, SHRINK_K_SEASON)
    out[f"{prefix}_pitcher_l5_nrfi_pct"] = _shrink(
        p.get("l5_nrfi_pct"), p.get("l5_starts"), NRFI_LEAGUE_PRIOR, SHRINK_K_L5)
```

Also update the `p is None` branch (debut pitcher, no profile row) a few lines above — it currently sets every `PITCHER_FEATS` entry to `np.nan`. Leave that branch as-is for now; `_shrink` is only called from the `p is not None` path. A fully-missing pitcher's `career_nrfi_pct`/`seas_nrfi_pct`/`l5_nrfi_pct` will still be `np.nan` after this task alone — Task 2's `is_debut` flag plus this task's Step 4 verification both confirm that's fine, because `_shrink` returning `prior` for the *found-but-zero-starts* case is what actually matters (a pitcher who has 0 recorded starts in `pitcher_nrfi_profile.parquet` who is nonetheless a row in it, vs. a pitcher not in the parquet at all — check which case Daniel Espino actually is before assuming; see Step 4).

- [ ] **Step 4: Run tests to verify they pass, then check the real ATH@CLE case**

```bash
cd /Users/nathantessler/Desktop/Nathan/Claude/MLB_Pipeline
./venv/bin/python -m pytest tests/test_yrfi_features.py -v
```

Expected: PASS (4/4).

Then confirm against live data (Daniel Espino has ZERO rows in `pitcher_nrfi_profile.parquet` at all per this session's earlier investigation — confirm `_pitcher_features()`'s `p is None` branch is what actually fires for him, meaning his `career_nrfi_pct`/`seas_nrfi_pct`/`l5_nrfi_pct` stay `np.nan` from this task alone and get the training-median fallback same as before; `_shrink` only changes Mason Barnett's side in this specific game):

```bash
cd /Users/nathantessler/Desktop/Nathan/Claude/MLB_Pipeline
./venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
from yrfi_features import load_support, _pitcher_features
prof_map, _ = load_support()
print('Espino (686930 is Barnett; find Espino id from todays_yrfi_predictions.parquet if needed) in profile:', 682982 in prof_map)
print('Barnett features:', _pitcher_features(686930, prof_map, 'away'))
"
```

Expected: Barnett's `away_pitcher_seas_nrfi_pct` should now read ~58.75 (shrunk from 25.0), `away_pitcher_career_nrfi_pct` ~59.03 (shrunk from 44.4) — matches the hand-computed values in the Step 1 test.

- [ ] **Step 5: Commit**

```bash
cd /Users/nathantessler/Desktop/Nathan/Claude/MLB_Pipeline
git add src/yrfi_features.py tests/test_yrfi_features.py requirements.txt
git commit -m "$(cat <<'EOF'
Add Bayesian shrinkage to pitcher NRFI-rate model features

career_nrfi_pct/seas_nrfi_pct/l5_nrfi_pct fed the RandomForest raw
observed rates with no sample-size correction, so a 4-start pitcher's
25% rate moved the model exactly as hard as a 300-start pitcher's. Now
blends each rate toward a 70% league-average prior, weighted by how
many starts back it (k=12 for career/season, k=4 for L5's inherently
tiny 5-start ceiling). Display values (what the UI shows) are
untouched -- this only changes what the model itself is trained/scored
on, via the shared yrfi_features.py builder.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Add starts-count and debut-flag confidence features

**Files:**
- Modify: [src/yrfi_features.py](MLB_Pipeline/src/yrfi_features.py) (`_pitcher_features()`, `feature_columns()`)
- Test: `tests/test_yrfi_features.py` (extend)

**Interfaces:**
- Consumes: `_pitcher_features()` from Task 1 (same function, extended).
- Produces: three new per-side columns — `{prefix}_pitcher_starts_log`, `{prefix}_pitcher_seas_starts_log`, `{prefix}_pitcher_is_debut` — appended to `feature_columns()`'s output **outside** `FEATURE_GROUPS` (same treatment as `month`/`umpire_zone_adj`: always weight 1, not adjustable by the dashboard's weighting sliders — these are data-quality signals, not a "team strength" factor a user should be able to dial up/down).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_yrfi_features.py
from yrfi_features import _pitcher_features, feature_columns

def test_confidence_features_for_known_pitcher():
    prof_map = {686930: {"total_starts": 9, "seas_starts": 4, "nrfi_pct": 44.4,
                          "seas_nrfi_pct": 25.0, "l5_starts": 5, "l5_nrfi_pct": 40.0}}
    out = _pitcher_features(686930, prof_map, "away")
    assert out["away_pitcher_is_debut"] == 0
    import math
    assert math.isclose(out["away_pitcher_starts_log"], math.log1p(9))
    assert math.isclose(out["away_pitcher_seas_starts_log"], math.log1p(4))

def test_confidence_features_for_debut_pitcher():
    out = _pitcher_features(999999, {}, "home")
    assert out["home_pitcher_is_debut"] == 1
    assert out["home_pitcher_starts_log"] == 0.0
    assert out["home_pitcher_seas_starts_log"] == 0.0

def test_confidence_features_not_in_any_weight_group():
    from yrfi_features import FEATURE_GROUPS
    grouped = {c for cols in FEATURE_GROUPS.values() for c in cols}
    for side in ("home", "away"):
        assert f"{side}_pitcher_is_debut" not in grouped
        assert f"{side}_pitcher_starts_log" not in grouped
        assert f"{side}_pitcher_seas_starts_log" not in grouped

def test_confidence_features_present_in_feature_columns():
    cols = feature_columns()
    for side in ("home", "away"):
        assert f"{side}_pitcher_starts_log" in cols
        assert f"{side}_pitcher_seas_starts_log" in cols
        assert f"{side}_pitcher_is_debut" in cols
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/nathantessler/Desktop/Nathan/Claude/MLB_Pipeline
./venv/bin/python -m pytest tests/test_yrfi_features.py -v
```

Expected: the 4 new tests FAIL (`KeyError` on the new column names).

- [ ] **Step 3: Implement**

Add near `PITCHER_FEATS` in `src/yrfi_features.py`:

```python
# Sample-size / data-quality signals -- deliberately NOT part of any
# FEATURE_GROUPS entry (see the "Columns not named in any group... always
# contribute at weight 1" comment above) since these describe confidence in
# the data, not a team-strength factor the dashboard's sliders should scale.
PITCHER_CONFIDENCE_FEATS = ["starts_log", "seas_starts_log", "is_debut"]
```

In `_pitcher_features()`, in the `p is None` branch (debut pitcher, no profile row at all), add after the existing `for f in PITCHER_FEATS: out[...] = np.nan` loop:

```python
        out[f"{prefix}_pitcher_starts_log"] = 0.0
        out[f"{prefix}_pitcher_seas_starts_log"] = 0.0
        out[f"{prefix}_pitcher_is_debut"] = 1
        return out
```

And in the main (`p is not None`) branch, after the existing `out[f"{prefix}_pitcher_primary_pitch_code"] = ...` line, add:

```python
    total_starts = p.get("total_starts")
    seas_starts = p.get("seas_starts")
    out[f"{prefix}_pitcher_starts_log"] = float(np.log1p(total_starts)) if pd.notna(total_starts) else 0.0
    out[f"{prefix}_pitcher_seas_starts_log"] = float(np.log1p(seas_starts)) if pd.notna(seas_starts) else 0.0
    out[f"{prefix}_pitcher_is_debut"] = 0
```

In `feature_columns()`, add the new columns the same way `CONTEXT_FEATS` is appended (ungrouped):

```python
def feature_columns() -> list[str]:
    cols = []
    for side in ("home", "away"):
        cols += [f"{side}_pitcher_{f}" for f in PITCHER_FEATS]
    for side in ("home", "away"):
        cols += [f"{side}_pitcher_{f}" for f in PITCHER_CONFIDENCE_FEATS]
    for side in ("away", "home"):
        cols += [f"{side}_top5_{f}" for f in TOP5_FEATS]
    cols += CONTEXT_FEATS
    return cols
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/nathantessler/Desktop/Nathan/Claude/MLB_Pipeline
./venv/bin/python -m pytest tests/test_yrfi_features.py -v
```

Expected: PASS (8/8 total).

- [ ] **Step 5: Commit**

```bash
cd /Users/nathantessler/Desktop/Nathan/Claude/MLB_Pipeline
git add src/yrfi_features.py tests/test_yrfi_features.py
git commit -m "$(cat <<'EOF'
Add pitcher starts-count and debut-flag as model features

Lets the RandomForest learn its own weighting of sample-size
confidence directly, rather than relying only on Task 1's shrinkage
of the rate features. log1p-transformed starts counts + a binary
is_debut flag for pitchers with no profile row at all. Added outside
FEATURE_GROUPS (always weight 1, same as month/umpire_zone_adj) --
these are data-quality signals, not something the dashboard's
per-group weighting sliders should scale.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Retrain and compare against the currently-deployed model

**Files:**
- Run (no code changes expected): [src/13_train_yrfi_model.py](MLB_Pipeline/src/13_train_yrfi_model.py)
- Produces (overwrites in place, git-tracked): `data/models/yrfi_model.pkl`, `data/models/yrfi_model_meta.json`, `data/models/yrfi_feature_importance.csv`

**Interfaces:**
- Consumes: `feature_columns()` from Tasks 1+2 (the retrain automatically picks up the new/changed features — no wiring needed, `src/13` calls `feature_columns()` directly).

**Baseline to beat** (current committed `yrfi_model_meta.json`, test set = 2024, n=2434):
- AUC 0.5548 · log_loss 0.7024 · Brier 0.2479 · ECE 0.0422 · accuracy 0.5464

- [ ] **Step 1: Confirm a clean starting point**

```bash
cd /Users/nathantessler/Desktop/Nathan/Claude/MLB_Pipeline
git status --short
```

Expected: clean except Tasks 1-2's already-committed changes. If anything else is dirty, stop and check what it is before proceeding (don't retrain over unrelated uncommitted work).

- [ ] **Step 2: Run the retrain, timed**

```bash
cd /Users/nathantessler/Desktop/Nathan/Claude/MLB_Pipeline
time ./venv/bin/python src/13_train_yrfi_model.py 2>&1 | tee /tmp/retrain_log.txt
```

Record the wall-clock time in the session notes (unknown ahead of time — 27,450 games × 600 trees is the only sizing data available; if it runs long, that's useful to know for next time, not a reason to interrupt it).

- [ ] **Step 3: Compare new metrics against baseline — go/no-go gate**

```bash
cd /Users/nathantessler/Desktop/Nathan/Claude/MLB_Pipeline
./venv/bin/python -c "
import json
m = json.load(open('data/models/yrfi_model_meta.json'))
test = next(x for x in m['metrics'] if x['set'] == 'test_2024')
print(test)
"
```

**Go/no-go criteria** (all must hold, compared to the baseline numbers above):
- `auc >= 0.5448` (no more than 0.01 regression)
- `brier <= 0.2499` (no more than 0.002 regression)
- `ece <= 0.0522` (no more than 0.01 regression)

- [ ] **Step 4a: If criteria FAIL — revert and report**

```bash
cd /Users/nathantessler/Desktop/Nathan/Claude/MLB_Pipeline
git checkout -- data/models/
```

Report the actual numbers to the user (which metric(s) regressed, by how much) before doing anything else — do not silently retune `k`/`prior` and retry without checking in; that's a real modeling decision, not a mechanical fix.

- [ ] **Step 4b: If criteria PASS — do not commit yet, proceed to Task 4**

Leave the new `data/models/*` files as uncommitted working-tree changes (visible in `git status`) — Task 4's manual spot-check is still a gate before this ships.

---

### Task 4: Targeted validation (historical subset + live spot-check)

This task is manual analysis, not code the user will ship — per the "Known limitation" section above, the historical subset check is necessarily approximate.

**Files:**
- Scratch script only, not committed: `/tmp/validate_thin_sample.py` or similar in the scratchpad directory.

- [ ] **Step 1: Historical subset check (approximate — read the caveat above)**

Using the already-loaded 2024 test predictions from Task 3's retrain run, slice to games where either starter's **current** `total_starts < 15` (acknowledging this under-counts true 2024-era thin-sample cases due to look-ahead bias) and compare Brier score on that subset for the old vs. new model. This is a supporting signal, not a proof — note it as such when reporting results.

- [ ] **Step 2: Live spot-check against real upcoming games**

With the new (uncommitted) `yrfi_model.pkl` in place locally, restart Streamlit (`kill $(lsof -ti :8501); cd /Users/nathantessler/Desktop/Nathan/Claude/MLB_Pipeline && nohup caffeinate -i /Library/Frameworks/Python.framework/Versions/3.14/bin/streamlit run app.py --server.headless true --server.port 8501 >> logs/streamlit.log 2>&1 &`) and re-run `src/14_build_todays_yrfi.py` against the current slate. Specifically check:
- Does ATH@CLE (or whatever thin-sample game is live that day) move meaningfully toward the market's ~coin-flip pricing, rather than staying at an extreme like -580?
- Do well-supported games (established pitchers on both sides) stay roughly where they were — confirming the fix targets thin samples specifically rather than flattening every prediction toward 50/50?

- [ ] **Step 3: Decision**

If both legs look reasonable: proceed to Task 5. If either is ambiguous or concerning: revert (`git checkout -- data/models/`) and report specifics to the user rather than shipping on a judgment call alone — this is the point in the plan where a human call is appropriate given the "early success, don't regress it" constraint.

---

### Task 5: Ship

**Files:**
- Commit: `data/models/yrfi_model.pkl`, `data/models/yrfi_model_meta.json`, `data/models/yrfi_feature_importance.csv`
- Modify: [claude-context.txt](MLB_Pipeline/claude-context.txt), [claude-prompts.txt](MLB_Pipeline/claude-prompts.txt), [USER_MANUAL.md](MLB_Pipeline/USER_MANUAL.md) — per this project's established documentation convention (see prior entries in both files from this same session for the expected level of detail: what changed, why, what was verified, what wasn't).

- [ ] **Step 1: Commit the new model artifacts**

```bash
cd /Users/nathantessler/Desktop/Nathan/Claude/MLB_Pipeline
git add data/models/yrfi_model.pkl data/models/yrfi_model_meta.json data/models/yrfi_feature_importance.csv
git commit -m "$(cat <<'EOF'
Retrain YRFI model with pitcher small-sample corrections

Retrained after adding Bayesian shrinkage on pitcher NRFI-rate
features and starts-count/debut confidence features (see prior two
commits). Test-2024 metrics: [fill in actual numbers from Task 3].
Validated against the currently-deployed model on the same held-out
set plus a live spot-check on real thin-sample games -- see
claude-context.txt for the full writeup.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 2: Update documentation**

Follow this session's established pattern in `claude-context.txt` (a dated section explaining root cause, fix, and verification results) and `claude-prompts.txt` (a dated session-log entry). Reference the 2026-09-18 diagnosis session and this plan file.

- [ ] **Step 3: Push and verify live**

```bash
cd /Users/nathantessler/Desktop/Nathan/Claude/MLB_Pipeline
git push origin main
```

Note: unlike the odds/deeplink fix from earlier this session, a model artifact change does NOT need the hosted app's "Refresh slate" button clicked — `yrfi_model.pkl` is loaded directly from the repo on redeploy, not regenerated by the daily pipeline run. A plain redeploy should be sufficient; confirm by checking the hosted app's sidebar "Test AUC (2024)" figure updates to match the new `yrfi_model_meta.json` value.

---

## Self-Review Notes

- **Spec coverage:** Option 3 (shrinkage) → Task 1. Option 4 (starts-count feature) → Task 2. Option 5 (debut flag) → Task 2 (folded in — same file, same `p is None` branch, splitting it into its own task would just add ceremony). Retrain/validate/ship → Tasks 3-5.
- **Placeholder scan:** Task 5 Step 1's commit message has one intentional `[fill in actual numbers from Task 3]` — this is correct, not a plan defect, since those numbers don't exist until Task 3 runs; the executor fills them in from real output, not a guess.
- **Type consistency:** `_shrink()`'s signature is used identically in Task 1's implementation and its Task-1 tests. `PITCHER_CONFIDENCE_FEATS` is defined once in Task 2 and referenced only there and in `feature_columns()`. No name drift found.
