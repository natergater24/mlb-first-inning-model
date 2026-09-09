"""
07_train_model.py
-----------------
Trains a RandomForest binary classifier to predict first-at-bat reach-base
probability. Uses scikit-learn only — no XGBoost or LightGBM required.

A LogisticRegression baseline is also trained for comparison.

Time split (no data leakage):
  Train  : 2015–2022  (~65k PA across 8 full seasons)
  Val    : 2023       (isotonic calibration)
  Test   : 2024       (held-out evaluation, never seen during training)

NaN handling:
  SimpleImputer(median) inside each Pipeline so training medians are applied
  consistently to val, test, and today's matchups.

Calibration:
  CalibratedClassifierCV(method='isotonic', cv='prefit') fitted on val 2023.

Outputs:
  data/models/rf_model.pkl            calibrated RandomForest pipeline
  data/models/model_meta.json         metrics + config snapshot
  data/models/feature_importance.csv  Gini-importance-ranked feature table
  data/processed/todays_matchups.parquet  (model_prob column added in-place)

Usage:
  python src/07_train_model.py            # train + evaluate + infer today
  python src/07_train_model.py --no-infer # skip today's matchup inference
"""

import json
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss, accuracy_score,
    precision_score, recall_score, f1_score,
)

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR  = Path(__file__).resolve().parent.parent
PROC_DIR  = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "data" / "models"
LOG_DIR   = BASE_DIR / "logs"

for d in [MODEL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "train.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Time splits ────────────────────────────────────────────────────────────────

TRAIN_YEARS = list(range(2015, 2023))   # 2015–2022 inclusive
VAL_YEAR    = 2023
TEST_YEAR   = 2024
MODEL_PATH  = MODEL_DIR / "rf_model.pkl"

# ── Feature manifest (priority order per spec) ─────────────────────────────────

FEATURE_GROUPS = {
    "bvp": [
        "bvp_obp_adj", "bvp_k_pct_adj", "bvp_bb_pct_adj", "bvp_woba_adj",
        "bvp_hard_hit", "bvp_exit_velo", "bvp_pa", "bvp_reliable",
    ],
    "rolling": [
        "obp_7d",      "obp_14d",      "obp_30d",
        "k_pct_7d",    "k_pct_14d",    "k_pct_30d",
        "bb_pct_7d",   "bb_pct_14d",   "bb_pct_30d",
        "hit_pct_7d",  "hit_pct_14d",  "hit_pct_30d",
        "hard_hit_7d", "hard_hit_14d", "hard_hit_30d",
        "xba_7d",      "xba_14d",      "xba_30d",
        "xwoba_7d",    "xwoba_14d",    "xwoba_30d",
        "form_trend",
    ],
    "pitcher_splits": [
        "pitch_split_obp", "pitch_split_k",
    ],
    "park": [
        "park_run_factor", "park_hr_factor",
    ],
    "weather": [
        "temp_f", "wind_speed", "wind_out_component", "humidity",
    ],
    "umpire": [
        "umpire_zone_adj",
    ],
    "platoon": [
        "platoon_adj", "same_hand_matchup",
    ],
    "rest": [
        "pitcher_rest_days",
    ],
    "context": [
        "month", "early_season",
    ],
    "fab": [
        "fab_pa", "fab_obp", "fab_k_pct", "fab_bb_pct",
        "fab_hard_hit", "fab_xba", "fab_woba",
        "fab_relevant_obp",
        "fab_obp_vs_rhp", "fab_pa_vs_rhp", "fab_k_pct_vs_rhp",
        "fab_obp_vs_lhp", "fab_pa_vs_lhp", "fab_k_pct_vs_lhp",
        "fab_obp_home", "fab_pa_home",
        "fab_obp_away", "fab_pa_away",
        "fab_platoon_split", "fab_home_away_split",
        "fab_obp_last30", "fab_k_pct_last30", "fab_trend",
    ],
}

ALL_FEATURES = [f for grp in FEATURE_GROUPS.values() for f in grp]

# ── Helpers ────────────────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the feature matrix. Columns absent from df are added as NaN
    (forward-compatible when today's matchups lack historical columns).
    Bool columns cast to float32 for sklearn compatibility.
    """
    X = pd.DataFrame(index=df.index)
    for feat in ALL_FEATURES:
        if feat in df.columns:
            col = df[feat]
            if col.dtype == bool:
                col = col.astype(np.float32)
            X[feat] = col.astype(np.float32)
        else:
            X[feat] = np.nan
    return X


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                               n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece


def print_calibration_table(y_true: np.ndarray, y_prob: np.ndarray,
                            label: str = "") -> None:
    df = pd.DataFrame({"prob": y_prob, "actual": y_true})
    df["bucket"] = pd.qcut(df["prob"], q=10, duplicates="drop")
    tbl = (
        df.groupby("bucket", observed=True)
        .agg(n=("actual", "count"), pred=("prob", "mean"), actual=("actual", "mean"))
        .reset_index()
    )
    tbl["gap"] = tbl["pred"] - tbl["actual"]
    print(f"\n  Calibration table {label}")
    print(f"  {'Bucket':<22} {'N':>6}  {'Pred':>7}  {'Actual':>7}  {'Gap':>7}")
    print("  " + "─" * 54)
    for _, r in tbl.iterrows():
        print(f"  {str(r['bucket']):<22} {int(r['n']):>6}  "
              f"{r['pred']:>7.3f}  {r['actual']:>7.3f}  {r['gap']:>+7.3f}")


def evaluate(model, X: pd.DataFrame, y: np.ndarray, label: str = "") -> dict:
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    m = {
        "auc":       roc_auc_score(y, probs),
        "logloss":   log_loss(y, probs),
        "brier":     brier_score_loss(y, probs),
        "accuracy":  accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall":    recall_score(y, preds, zero_division=0),
        "f1":        f1_score(y, preds, zero_division=0),
        "ece":       expected_calibration_error(y, probs),
        "probs":     probs,
    }
    log.info(
        "  %-14s  AUC %s  |  LogLoss %s  |  Brier %s  |  ECE %s  |  Acc %s",
        label,
        f"{m['auc']:.4f}", f"{m['logloss']:.4f}",
        f"{m['brier']:.4f}", f"{m['ece']:.4f}", f"{m['accuracy']:.4f}",
    )
    return m


# ── Calibration wrapper ────────────────────────────────────────────────────────

class CalibratedPipeline:
    """
    Thin wrapper that applies IsotonicRegression on top of a fitted sklearn
    Pipeline. Replaces CalibratedClassifierCV(cv='prefit') which was removed
    in sklearn 1.9. Fully picklable via joblib.
    """
    def __init__(self, pipeline: Pipeline, calibrator: IsotonicRegression):
        self.pipeline   = pipeline
        self.calibrator = calibrator

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw  = self.pipeline.predict_proba(X)[:, 1]
        cal  = self.calibrator.predict(raw)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ── Main training pipeline ─────────────────────────────────────────────────────

def run(run_inference_step: bool = True) -> None:

    # ── 1. Load & filter ───────────────────────────────────────────────────────
    log.info("Loading model_features.parquet ...")
    df = pd.read_parquet(PROC_DIR / "model_features.parquet")
    log.info("  Loaded %d rows × %d cols", len(df), len(df.columns))

    model_years = TRAIN_YEARS + [VAL_YEAR, TEST_YEAR]
    df = df[df["game_year"].isin(model_years)].copy()
    log.info("  Filtered to seasons %d–%d: %d rows",
             min(model_years), max(model_years), len(df))

    missing_feats = [f for f in ALL_FEATURES if f not in df.columns]
    if missing_feats:
        log.warning("Features absent from dataset (will be NaN): %s", missing_feats)

    # ── 2. Time-based splits (adaptive to available years) ────────────────────
    available_years = sorted(df["game_year"].dropna().unique().astype(int).tolist())
    log.info("  Available years in dataset: %s", available_years)

    if len(available_years) >= 3:
        # Standard year-based split
        n = len(available_years)
        # Use first 70% for train, next 15% for val, last 15% for test
        train_yrs = available_years[:max(1, int(n * 0.70))]
        val_yr    = available_years[max(1, int(n * 0.70))]
        test_yr   = available_years[-1]
        train_df  = df[df["game_year"].isin(train_yrs)]
        val_df    = df[df["game_year"] == val_yr]
        test_df   = df[df["game_year"] == test_yr]
        split_desc = f"Train {min(train_yrs)}–{max(train_yrs)} | Val {val_yr} | Test {test_yr}"
    else:
        # Fallback: date-based split within available data
        df = df.sort_values("game_date")
        n  = len(df)
        i_train = int(n * 0.70)
        i_val   = int(n * 0.85)
        train_df = df.iloc[:i_train]
        val_df   = df.iloc[i_train:i_val]
        test_df  = df.iloc[i_val:]
        val_yr   = available_years[-1]
        test_yr  = available_years[-1]
        split_desc = (f"Date-based split (only {available_years} available) — "
                      "run 01_fetch_statcast.py to restore full 2015–2024 history")
        log.warning("Limited historical data. %s", split_desc)

    log.info("  Split: %s", split_desc)

    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        if len(split):
            log.info("  %s  %6d rows  reach-base %.1f%%",
                     name, len(split), split["reached_base"].mean() * 100)

    X_train = prepare_features(train_df);  y_train = train_df["reached_base"].values
    X_val   = prepare_features(val_df);    y_val   = val_df["reached_base"].values
    X_test  = prepare_features(test_df);   y_test  = test_df["reached_base"].values

    # ── 3. Build pipelines ─────────────────────────────────────────────────────
    # Imputer inside each pipeline so training medians are locked in and applied
    # consistently at inference time. class_weight='balanced' handles ~2:1 imbalance.

    rf_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("rf",      RandomForestClassifier(
            n_estimators   = 300,
            max_depth      = 20,
            min_samples_leaf = 15,
            max_features   = "sqrt",
            class_weight   = "balanced",
            n_jobs         = -1,
            random_state   = 42,
        )),
    ])

    lr_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler",  StandardScaler()),
        ("lr",      LogisticRegression(
            C            = 0.1,
            max_iter     = 1000,
            class_weight = "balanced",
            n_jobs       = -1,
            random_state = 42,
        )),
    ])

    # ── 4. Train ───────────────────────────────────────────────────────────────
    log.info("Training RandomForest (300 trees, max_depth=20) ...")
    rf_pipe.fit(X_train, y_train)

    log.info("Training LogisticRegression baseline ...")
    lr_pipe.fit(X_train, y_train)

    # ── 5. Calibrate RF on val set ─────────────────────────────────────────────
    log.info("Calibrating RF probabilities (isotonic on val %d) ...", VAL_YEAR)
    raw_val_probs = rf_pipe.predict_proba(X_val)[:, 1]
    isotonic      = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(raw_val_probs, y_val)
    calibrated_rf = CalibratedPipeline(rf_pipe, isotonic)

    # ── 6. Evaluate ────────────────────────────────────────────────────────────
    SEP = "─" * 68
    print(f"\n{SEP}")
    print("EVALUATION  (raw RF  |  calibrated RF  |  logistic baseline)")
    print(SEP)

    print("\n  Raw RandomForest:")
    raw_val  = evaluate(rf_pipe,       X_val,  y_val,  f"val  {VAL_YEAR}")
    raw_test = evaluate(rf_pipe,       X_test, y_test, f"test {TEST_YEAR}")

    print("\n  Calibrated RandomForest:")
    cal_val  = evaluate(calibrated_rf, X_val,  y_val,  f"val  {VAL_YEAR}")
    cal_test = evaluate(calibrated_rf, X_test, y_test, f"test {TEST_YEAR}")

    print("\n  Logistic Regression (baseline):")
    lr_val   = evaluate(lr_pipe,       X_val,  y_val,  f"val  {VAL_YEAR}")
    lr_test  = evaluate(lr_pipe,       X_test, y_test, f"test {TEST_YEAR}")

    print_calibration_table(y_test, cal_test["probs"],
                            label=f"— calibrated RF, test {TEST_YEAR}")

    print(f"\n  ECE improvement (calibrated RF vs raw RF):")
    print(f"    Val  {VAL_YEAR}: {raw_val['ece']:.4f} → {cal_val['ece']:.4f}"
          f"  ({(raw_val['ece']-cal_val['ece'])*100:+.2f}pp)")
    print(f"    Test {TEST_YEAR}: {raw_test['ece']:.4f} → {cal_test['ece']:.4f}"
          f"  ({(raw_test['ece']-cal_test['ece'])*100:+.2f}pp)")

    # ── 7. Feature importance ──────────────────────────────────────────────────
    rf_model   = rf_pipe.named_steps["rf"]
    feat_group = {f: g for g, feats in FEATURE_GROUPS.items() for f in feats}

    imp_df = (
        pd.DataFrame({"feature": ALL_FEATURES,
                      "importance": rf_model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    imp_df["group"]   = imp_df["feature"].map(feat_group)
    imp_df["imp_pct"] = imp_df["importance"] / imp_df["importance"].sum() * 100
    imp_df.to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    print(f"\n{SEP}")
    print("FEATURE IMPORTANCE  (Gini, top 20)")
    print(SEP)
    print(f"  {'Feature':<26}  {'Group':<18}  {'Imp%':>6}")
    print(f"  {'─'*26}  {'─'*18}  {'─'*6}")
    for _, row in imp_df.head(20).iterrows():
        print(f"  {row['feature']:<26}  {str(row['group']):<18}  {row['imp_pct']:>5.2f}%")

    grp_totals = imp_df.groupby("group")["imp_pct"].sum().sort_values(ascending=False)
    print(f"\n  Group totals:")
    for grp, pct in grp_totals.items():
        bar = "█" * int(pct / 2)
        print(f"    {str(grp):<18}  {pct:>5.1f}%  {bar}")

    # ── 8. Save model + metadata ───────────────────────────────────────────────
    meta = {
        "trained_at":      datetime.now().isoformat(timespec="seconds"),
        "model_type":      "RandomForestClassifier + isotonic calibration",
        "train_years":     [int(y) for y in train_df["game_year"].dropna().unique()],
        "val_year":        int(val_yr),
        "test_year":       int(test_yr),
        "features":        ALL_FEATURES,
        "n_features":      len(ALL_FEATURES),
        "rf_params": {
            "n_estimators": 300, "max_depth": 20,
            "min_samples_leaf": 15, "max_features": "sqrt",
        },
        "metrics": {
            f"val_{val_yr}":    {k: float(v) for k, v in cal_val.items()  if k != "probs"},
            f"test_{test_yr}":  {k: float(v) for k, v in cal_test.items() if k != "probs"},
            f"lr_test_{test_yr}": {k: float(v) for k, v in lr_test.items() if k != "probs"},
        },
    }

    joblib.dump({"model": calibrated_rf, "meta": meta}, MODEL_PATH)
    log.info("Model saved → %s", MODEL_PATH)

    with open(MODEL_DIR / "model_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    log.info("Metadata saved → %s", MODEL_DIR / "model_meta.json")

    if run_inference_step:
        run_inference(calibrated_rf)


# ── Inference ──────────────────────────────────────────────────────────────────

def _load_specialist_model(path: Path):
    """Load a pitcher or batter specialist model if it exists."""
    if not path.exists():
        return None, None
    bundle = joblib.load(path)
    return bundle.get("model"), bundle.get("features", [])


def run_inference(model=None) -> pd.DataFrame:
    """
    Score today's matchups with three models, blend probabilities,
    and write all component scores to todays_matchups.parquet.
    """
    today = datetime.now().strftime("%Y-%m-%d")

    # ── Load models ─────────────────────────────────────────────────────────
    if model is None:
        if not MODEL_PATH.exists():
            log.error("No saved model at %s — run training first.", MODEL_PATH)
            return pd.DataFrame()
        bundle = joblib.load(MODEL_PATH)
        model  = bundle["model"]
        log.info("Loaded combined model trained at %s",
                 bundle.get("meta", {}).get("trained_at", "?"))

    pitcher_model, pitcher_feats = _load_specialist_model(MODEL_DIR / "pitcher_model.pkl")
    batter_model,  batter_feats  = _load_specialist_model(MODEL_DIR / "batter_model.pkl")

    if pitcher_model:
        log.info("Loaded pitcher model (%d features)", len(pitcher_feats or []))
    if batter_model:
        log.info("Loaded batter model (%d features)", len(batter_feats or []))

    # ── Load matchups ────────────────────────────────────────────────────────
    matchup_path = PROC_DIR / "todays_matchups.parquet"
    if not matchup_path.exists():
        log.warning("todays_matchups.parquet not found.")
        return pd.DataFrame()

    matchups = pd.read_parquet(matchup_path)
    if len(matchups) == 0:
        log.warning("todays_matchups.parquet is empty.")
        return pd.DataFrame()

    # Refuse to re-score a stale file — this prevents yesterday's matchups from
    # being rescored and re-saved when today's lineup fetch hasn't run yet.
    date_col = next((c for c in ("game_date_x", "game_date") if c in matchups.columns), None)
    if date_col:
        file_dates = matchups[date_col].astype(str).str[:10].unique()
        if not any(d == today for d in file_dates):
            log.warning(
                "todays_matchups.parquet contains dates %s — not today (%s). "
                "Skipping inference to avoid overwriting with stale data.",
                sorted(file_dates), today,
            )
            return pd.DataFrame()

    log.info("Scoring %d today's matchups (3-model blend)…", len(matchups))

    # ── Combined model ────────────────────────────────────────────────────────
    X_today = prepare_features(matchups)
    combined_prob = model.predict_proba(X_today)[:, 1].clip(0.05, 0.95)
    matchups["combined_model_prob"] = combined_prob

    # ── Pitcher model ─────────────────────────────────────────────────────────
    if pitcher_model and pitcher_feats:
        X_p = pd.DataFrame(index=matchups.index)
        for f in pitcher_feats:
            X_p[f] = matchups[f].astype(float) if f in matchups.columns else np.nan
        matchups["pitcher_model_prob"] = pitcher_model.predict_proba(X_p)[:, 1].clip(0.05, 0.95)
    else:
        matchups["pitcher_model_prob"] = combined_prob

    # ── Batter model ──────────────────────────────────────────────────────────
    if batter_model and batter_feats:
        # Ensure all trained features present; fill missing with NaN (imputer handles it)
        X_b = pd.DataFrame(index=matchups.index)
        for f in batter_feats:
            X_b[f] = matchups[f].astype(float) if f in matchups.columns else np.nan
        matchups["batter_model_prob"] = batter_model.predict_proba(X_b)[:, 1].clip(0.05, 0.95)
    else:
        matchups["batter_model_prob"] = combined_prob

    # ── Blended final probability ─────────────────────────────────────────────
    matchups["final_model_prob"] = (
        0.35 * matchups["pitcher_model_prob"] +
        0.45 * matchups["batter_model_prob"]  +
        0.20 * matchups["combined_model_prob"]
    ).clip(0.05, 0.95)

    # Use final blended probability as the primary model_prob
    matchups["model_prob"] = matchups["final_model_prob"]

    # ── Model agreement ────────────────────────────────────────────────────────
    def _agreement(row):
        probs = [row.get("pitcher_model_prob", np.nan),
                 row.get("batter_model_prob",  np.nan),
                 row.get("combined_model_prob", np.nan)]
        probs = [p for p in probs if pd.notna(p)]
        if len(probs) < 2:
            return "strong", "All models agree"
        spread = max(probs) - min(probs)
        if spread <= 0.05:
            return "strong", "✅ All models agree"
        if spread <= 0.10:
            return "moderate", "📊 Models mostly agree"
        return "split", "⚠️ Models split — use caution"

    agree = matchups.apply(_agreement, axis=1)
    matchups["model_agreement"]       = agree.apply(lambda x: x[0])
    matchups["model_agreement_label"] = agree.apply(lambda x: x[1])

    matchups.to_parquet(matchup_path, index=False)
    log.info("model_prob (blended) written → %s", matchup_path)

    # ── Display table ──────────────────────────────────────────────────────────
    show = matchups.copy()
    for id_col, name_col in [("batter", "batter_name"), ("pitcher", "pitcher_name")]:
        if name_col not in show.columns and id_col in show.columns:
            show[name_col] = show[id_col].astype(str)

    col_map = {
        "batter_name":     "Batter",
        "pitcher_name":    "Pitcher",
        "side":            "Side",
        "model_prob":      "Model%",
        "avg_implied_prob":"Market%",
        "best_odds":       "BestOdds",
    }
    present = [c for c in col_map if c in show.columns]
    disp    = show[present].rename(columns=col_map).copy()

    if "Model%"  in disp.columns: disp["Model%"]  = (disp["Model%"]  * 100).round(1)
    if "Market%" in disp.columns: disp["Market%"] = (disp["Market%"] * 100).round(1)

    if "Model%" in disp.columns and "Market%" in disp.columns:
        disp["Edge%"] = (disp["Model%"] - disp["Market%"]).round(1)
        disp = disp.sort_values("Edge%", ascending=False)

    SEP = "═" * 72
    print(f"\n{SEP}")
    print(f"  TODAY'S FIRST-AT-BAT PREDICTIONS  —  {today}")
    print(SEP)

    if len(disp) == 0:
        print("  No matchups scored.")
    else:
        with pd.option_context("display.max_rows", 60, "display.width", 100):
            for line in disp.to_string(index=False).splitlines():
                print("  " + line)

        if "Edge%" in disp.columns:
            pos = disp[disp["Edge%"] > 0]
            print(f"\n  {len(pos)} matchup(s) where model > market implied prob")
            if len(pos):
                top = disp.iloc[0]
                print(f"  Best edge: {top.get('Batter','?')}"
                      f"  +{top['Edge%']:.1f}pp"
                      f"  (model {top['Model%']:.1f}%  vs  market {top['Market%']:.1f}%)")

        # Supporting stats for top 5 by model probability
        stat_cols = ["batter_name", "model_prob", "bvp_obp_adj",
                     "obp_30d", "pitch_split_obp", "temp_f", "wind_speed"]
        stat_cols = [c for c in stat_cols if c in matchups.columns]
        top5 = matchups.nlargest(5, "model_prob")[stat_cols]
        if len(top5):
            print(f"\n  Supporting stats — top 5 by model probability:")
            with pd.option_context("display.float_format", "{:.3f}".format,
                                   "display.width", 100):
                for line in top5.to_string(index=False).splitlines():
                    print("  " + line)

    print(SEP + "\n")
    return matchups


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    infer_only = "--infer-only" in sys.argv   # skip retraining, just re-score today
    no_infer   = "--no-infer"   in sys.argv   # full train but skip today's inference

    if infer_only:
        # Fast daily path: load saved model and re-score todays_matchups.parquet only
        run_inference()
    else:
        run(run_inference_step=not no_infer)
