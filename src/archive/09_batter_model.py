"""
09_batter_model.py
------------------
Trains a batter-focused RandomForest model predicting first-at-bat reach base
using career stats, pitch-type hitting splits, recent rolling form, and context.

Output:
  data/models/batter_model.pkl
  data/models/batter_feature_importance.csv
"""

import json
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, accuracy_score

BASE_DIR  = Path(__file__).resolve().parent.parent
PROC_DIR  = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "data" / "models"
LOG_DIR   = BASE_DIR / "logs"

for d in [MODEL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "batter_model.log"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

BATTER_MODEL_FEATURES = [
    # Career first-AB stats
    "fab_obp", "fab_k_pct", "fab_bb_pct", "fab_hard_hit", "fab_xba", "fab_woba",
    "fab_obp_vs_rhp", "fab_obp_vs_lhp", "fab_obp_home", "fab_obp_away",
    "fab_platoon_split", "fab_home_away_split", "fab_trend",
    # Pitch-type hitting splits
    "batter_fb_obp", "batter_fb_whiff_pct", "batter_fb_hard_hit", "batter_fb_xba",
    "batter_bb_obp", "batter_bb_whiff_pct", "batter_bb_hard_hit", "batter_bb_xba",
    "batter_os_obp", "batter_os_whiff_pct",
    "batter_fb_vs_bb_diff",
    # Recent rolling form
    "obp_7d", "obp_14d", "obp_30d",
    "k_pct_7d", "bb_pct_7d",
    "hard_hit_7d", "xba_7d",
    "form_trend",
    # Context
    "platoon_adj", "park_run_factor",
    "temp_f", "wind_out_component",
    "umpire_zone_adj",
    # FAB context-aware
    "fab_relevant_obp",
]


class CalibratedPipeline:
    def __init__(self, pipeline, calibrator):
        self.pipeline = pipeline; self.calibrator = calibrator
    def predict_proba(self, X):
        raw = self.pipeline.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        return np.column_stack([1.0 - cal, cal])
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def run():
    log.info("=" * 60)
    log.info("BATTER CAREER MODEL")
    log.info("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────
    model_feat_path = PROC_DIR / "model_features.parquet"
    if not model_feat_path.exists():
        log.error("model_features.parquet not found — run 05_build_features.py first"); return

    df = pd.read_parquet(model_feat_path)
    log.info("Loaded model_features: %d rows × %d cols", len(df), len(df.columns))

    # Pitch-type profile
    ptp_path = PROC_DIR / "batter_pitch_type_profile.parquet"
    if ptp_path.exists():
        ptp = pd.read_parquet(ptp_path)
        df = df.merge(ptp, on="batter", how="left")
        log.info("Joined batter pitch-type profile: %d cols", len(df.columns))

    df["game_date"] = pd.to_datetime(df["game_date"])
    available_years = sorted(df["game_year"].dropna().unique().astype(int).tolist())
    n = len(available_years)

    if n >= 3:
        train_yrs = available_years[:max(1, int(n * 0.70))]
        val_yr    = available_years[max(1, int(n * 0.70))]
        test_yr   = available_years[-1]
        train_df  = df[df["game_year"].isin(train_yrs)]
        val_df    = df[df["game_year"] == val_yr]
        test_df   = df[df["game_year"] == test_yr]
    else:
        df = df.sort_values("game_date")
        n_all = len(df)
        train_df = df.iloc[:int(n_all * 0.70)]
        val_df   = df.iloc[int(n_all * 0.70):int(n_all * 0.85)]
        test_df  = df.iloc[int(n_all * 0.85):]
        val_yr = test_yr = available_years[-1]

    log.info("Split: train %d | val %d | test %d (years %s)",
             len(train_df), len(val_df), len(test_df), available_years)

    use_feats = [f for f in BATTER_MODEL_FEATURES if f in df.columns]
    log.info("Using %d batter features", len(use_feats))

    X_tr = train_df[use_feats].astype(float)
    y_tr = train_df["reached_base"].values
    X_va = val_df[use_feats].astype(float)
    y_va = val_df["reached_base"].values
    X_te = test_df[use_feats].astype(float)
    y_te = test_df["reached_base"].values

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("rf",  RandomForestClassifier(n_estimators=300, max_depth=15,
                                        min_samples_leaf=15, max_features="sqrt",
                                        class_weight="balanced", n_jobs=-1, random_state=42)),
    ])
    pipe.fit(X_tr, y_tr)

    raw_val = pipe.predict_proba(X_va)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_val, y_va)
    model = CalibratedPipeline(pipe, iso)

    for (X, y, label) in [(X_va, y_va, f"val {val_yr}"), (X_te, y_te, f"test {test_yr}")]:
        p = model.predict_proba(X)[:, 1]
        log.info("  %-12s  AUC %.4f  Acc %.4f",
                 label, roc_auc_score(y, p), accuracy_score(y, (p >= 0.5).astype(int)))

    rf_est = pipe.named_steps["rf"]
    imp_df = pd.DataFrame({"feature": use_feats, "importance": rf_est.feature_importances_})
    imp_df["imp_pct"] = imp_df["importance"] / imp_df["importance"].sum() * 100
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    imp_df.to_csv(MODEL_DIR / "batter_feature_importance.csv", index=False)

    log.info("\nBatter feature importance (top 15):")
    for _, r in imp_df.head(15).iterrows():
        log.info("  %-35s  %.2f%%", r["feature"], r["imp_pct"])

    joblib.dump({"model": model, "features": use_feats,
                 "val_yr": val_yr, "test_yr": test_yr}, MODEL_DIR / "batter_model.pkl")
    log.info("Batter model saved → %s", MODEL_DIR / "batter_model.pkl")


if __name__ == "__main__":
    run()
