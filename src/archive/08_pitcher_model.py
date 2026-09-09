"""
08_pitcher_model.py
-------------------
Trains a pitcher-focused RandomForest model predicting first-at-bat reach base
using only pitcher first-inning pitch-type and performance features.

Data source: data/raw/statcast_20XX.csv (inning-1 leadoff pitch-level data)
Output:
  data/processed/pitcher_first_inn_features.parquet  per-pitcher career features
  data/models/pitcher_model.pkl                       calibrated RF model
  data/models/pitcher_feature_importance.csv          Gini importance

Usage:
  python src/08_pitcher_model.py               # train + save
  python src/08_pitcher_model.py --features-only  # only build feature table
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
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score

BASE_DIR  = Path(__file__).resolve().parent.parent
RAW_DIR   = BASE_DIR / "data" / "raw"
PROC_DIR  = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "data" / "models"
LOG_DIR   = BASE_DIR / "logs"

for d in [MODEL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "pitcher_model.log"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

TRAIN_YEARS = list(range(2015, 2023))
VAL_YEAR    = 2023
TEST_YEAR   = 2024

# Pitch type groupings
PITCH_CODES = {
    "ff": "FF", "si": "SI", "sl": "SL", "cu": "CU",
    "ch": "CH", "kc": "KC", "fc": "FC", "st": "ST",
}
FASTBALL_CODES    = {"FF", "SI", "FC", "FT"}
BREAKINGBALL_CODES = {"SL", "CU", "KC", "ST", "SV", "CS"}
OFFSPEED_CODES    = {"CH", "FS", "FO"}

SWINGS = {"swinging_strike", "swinging_strike_blocked", "foul_tip",
          "foul", "foul_bunt", "hit_into_play", "hit_into_play_score",
          "hit_into_play_no_out", "missed_bunt"}
WHIFFS = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
STRIKES_CALLED = {"swinging_strike", "swinging_strike_blocked", "foul_tip",
                  "called_strike", "foul", "foul_bunt", "automatic_strike"}
HIT_EVENTS = {"single", "double", "triple", "home_run"}
BIP_EVENTS = {"single", "double", "triple", "home_run",
              "field_out", "force_out", "grounded_into_double_play",
              "double_play", "triple_play", "field_error",
              "fielders_choice", "fielders_choice_out", "sac_fly", "sac_bunt"}
REACH_EVENTS = {"single", "double", "triple", "home_run",
                "walk", "intent_walk", "hit_by_pitch",
                "field_error", "catcher_interf"}
K_EVENTS  = {"strikeout", "strikeout_double_play"}
BB_EVENTS = {"walk", "intent_walk"}
HR_EVENTS = {"home_run"}
HBP_EVENTS = {"hit_by_pitch"}


def load_raw_pitches(seasons=None) -> pd.DataFrame:
    """Load and concatenate all raw statcast CSVs."""
    if seasons is None:
        seasons = list(range(2015, datetime.now().year + 1))
    parts = []
    for yr in seasons:
        p = RAW_DIR / f"statcast_{yr}.csv"
        if not p.exists():
            log.warning("  Missing: %s", p)
            continue
        df = pd.read_csv(p, low_memory=False)
        df = df[df["game_type"] == "R"].copy()
        if "game_year" not in df.columns:
            df["game_year"] = yr
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    log.info("Loaded %d pitches across %d seasons", len(combined), len(parts))
    return combined


def build_pitcher_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate pitch-level data into per-(pitcher, game_year) feature vectors.
    Returns one row per (pitcher, game_year).
    """
    if raw.empty:
        return pd.DataFrame()

    raw = raw.copy()
    raw["is_swing"]       = raw["description"].isin(SWINGS)
    raw["is_whiff"]       = raw["description"].isin(WHIFFS)
    raw["is_fp_strike"]   = (raw["pitch_number"] == 1) & raw["description"].isin(STRIKES_CALLED)
    raw["is_first_pitch"] = raw["pitch_number"] == 1
    raw["is_fb_pitch"]    = raw["pitch_type"].isin(FASTBALL_CODES)
    raw["hard_hit"]       = (raw["launch_speed"] >= 95) & raw["launch_speed"].notna()
    raw["is_bip"]         = raw["events"].isin(BIP_EVENTS)
    raw["is_hit"]         = raw["events"].isin(HIT_EVENTS)
    raw["is_ob"]          = raw["events"].isin(REACH_EVENTS)
    raw["is_k"]           = raw["events"].isin(K_EVENTS)
    raw["is_bb"]          = raw["events"].isin(BB_EVENTS)
    raw["is_hr"]          = raw["events"].isin(HR_EVENTS)
    raw["is_hbp"]         = raw["events"].isin(HBP_EVENTS)
    raw["is_pa"]          = raw["events"].notna()

    grp = ["pitcher", "game_year"]

    # ── General PA-level stats ────────────────────────────────────────────────
    term = raw[raw["is_pa"]]
    pa_stats = term.groupby(grp).agg(
        _pa          = ("is_pa",    "sum"),
        _ob          = ("is_ob",    "sum"),
        _k           = ("is_k",     "sum"),
        _bb          = ("is_bb",    "sum"),
        _hr          = ("is_hr",    "sum"),
        _hbp         = ("is_hbp",   "sum"),
        _woba_v      = ("woba_value", "sum"),
        _woba_d      = ("woba_denom", "sum"),
    ).reset_index()

    pa_stats["first_inn_k_rate"]       = pa_stats["_k"]  / pa_stats["_pa"].clip(1)
    pa_stats["first_inn_bb_rate"]      = pa_stats["_bb"] / pa_stats["_pa"].clip(1)
    pa_stats["first_inn_obp_allowed"]  = pa_stats["_ob"] / pa_stats["_pa"].clip(1)
    pa_stats["first_inn_woba_allowed"] = pa_stats["_woba_v"] / pa_stats["_woba_d"].clip(1)

    bip = term[term["is_bip"]]
    hh_stats = bip.groupby(grp).agg(
        _hh_sum  = ("hard_hit",    "sum"),
        _hh_denom= ("launch_speed", lambda x: x.notna().sum()),
    ).reset_index()
    hh_stats["first_inn_hard_hit_allowed"] = hh_stats["_hh_sum"] / hh_stats["_hh_denom"].clip(1)

    # FIP: (13*HR + 3*(BB+HBP) - 2*K) / PA   (simplified, IP proxy)
    pa_stats["first_inn_fip"] = (
        (13 * pa_stats["_hr"] + 3 * (pa_stats["_bb"] + pa_stats["_hbp"]) - 2 * pa_stats["_k"])
        / pa_stats["_pa"].clip(1)
    ) + 3.20   # + FIP constant

    # ── Velocity (fastballs only) ─────────────────────────────────────────────
    fb = raw[raw["is_fb_pitch"] & raw["release_speed"].notna()]
    velo = fb.groupby(grp)["release_speed"].mean().rename("first_inn_avg_velo").reset_index()

    career_velo = raw[raw["is_fb_pitch"] & raw["release_speed"].notna()].groupby(
        "pitcher"
    )["release_speed"].mean().rename("_career_velo")

    # ── First-pitch strike rate ───────────────────────────────────────────────
    fp = raw[raw["is_first_pitch"]]
    fps = fp.groupby(grp).agg(
        _fp      = ("is_first_pitch",  "sum"),
        _fp_str  = ("is_fp_strike",    "sum"),
    ).reset_index()
    fps["first_pitch_strike_rate"] = fps["_fp_str"] / fps["_fp"].clip(1)

    # ── Per-pitch-type features ───────────────────────────────────────────────
    pitch_dfs = []
    total_pitches = raw.groupby(grp).size().rename("_total")

    for abbr, code in PITCH_CODES.items():
        pt   = raw[raw["pitch_type"] == code]
        ptdf = pd.DataFrame(index=total_pitches.index).reset_index()

        usage = (pt.groupby(grp).size() / total_pitches).rename(f"first_inn_{abbr}_usage_pct")

        sw    = pt[pt["is_swing"]]
        wh    = pt[pt["is_whiff"]]
        whiff = (wh.groupby(grp).size() / sw.groupby(grp).size().clip(1)).rename(f"first_inn_{abbr}_whiff_pct")

        pt_term = pt[pt["is_pa"]]
        woba_v  = pt_term.groupby(grp)["woba_value"].sum()
        woba_d  = pt_term.groupby(grp)["woba_denom"].sum()
        woba    = (woba_v / woba_d.clip(1)).rename(f"first_inn_{abbr}_woba_against")

        bip_pt  = pt_term[pt_term["is_bip"]]
        hit_pt  = pt_term[pt_term["is_hit"]]
        ba      = (hit_pt.groupby(grp).size() / bip_pt.groupby(grp).size().clip(1))\
                  .rename(f"first_inn_{abbr}_ba_against")

        pitch_dfs.append(pd.concat([usage, whiff, woba, ba], axis=1))

    all_pitch = pd.concat(pitch_dfs, axis=1).reset_index()

    # ── Merge everything ──────────────────────────────────────────────────────
    keep_pa = ["pitcher","game_year",
               "first_inn_k_rate","first_inn_bb_rate","first_inn_obp_allowed",
               "first_inn_woba_allowed","first_inn_fip"]
    feats = pa_stats[keep_pa].copy()
    for df_part in [hh_stats[["pitcher","game_year","first_inn_hard_hit_allowed"]],
                    velo,
                    fps[["pitcher","game_year","first_pitch_strike_rate"]],
                    all_pitch]:
        feats = feats.merge(df_part, on=["pitcher","game_year"], how="left")

    # Compute velo_vs_career delta
    feats = feats.merge(career_velo.reset_index(), on="pitcher", how="left")
    feats["first_inn_velo_vs_career"] = feats["first_inn_avg_velo"] - feats["_career_velo"]
    feats = feats.drop(columns=["_career_velo"], errors="ignore")

    # Primary pitch (highest usage in inning 1)
    usage_cols = [f"first_inn_{a}_usage_pct" for a in PITCH_CODES]
    usage_cols = [c for c in usage_cols if c in feats.columns]
    if usage_cols:
        def _primary_pitch(row):
            vals = {c.replace("first_inn_","").replace("_usage_pct","").upper(): row.get(c, 0)
                    for c in usage_cols}
            best = max(vals, key=lambda k: vals[k] if pd.notna(vals[k]) else 0)
            return best
        feats["pitcher_primary_pitch_code"] = feats.apply(_primary_pitch, axis=1)
        feats["pitcher_primary_pitch"] = feats["pitcher_primary_pitch_code"]

        def _primary_woba(row):
            code = str(row.get("pitcher_primary_pitch_code","")).lower()
            return row.get(f"first_inn_{code}_woba_against", np.nan)
        feats["pitcher_primary_pitch_woba"] = feats.apply(_primary_woba, axis=1)

    log.info("Pitcher features: %d rows × %d cols", len(feats), len(feats.columns))
    return feats


PITCHER_MODEL_FEATURES = [
    "first_inn_ff_usage_pct", "first_inn_si_usage_pct", "first_inn_sl_usage_pct",
    "first_inn_cu_usage_pct", "first_inn_ch_usage_pct", "first_inn_kc_usage_pct",
    "first_inn_fc_usage_pct", "first_inn_st_usage_pct",
    "first_inn_ff_whiff_pct", "first_inn_sl_whiff_pct", "first_inn_cu_whiff_pct",
    "first_inn_ch_whiff_pct", "first_inn_st_whiff_pct",
    "first_inn_ff_woba_against", "first_inn_sl_woba_against", "first_inn_cu_woba_against",
    "first_inn_ch_woba_against",
    "first_inn_avg_velo", "first_inn_velo_vs_career",
    "first_inn_k_rate", "first_inn_bb_rate",
    "first_inn_obp_allowed", "first_inn_hard_hit_allowed",
    "first_inn_fip", "first_pitch_strike_rate",
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


def run(features_only: bool = False):
    log.info("=" * 60)
    log.info("PITCHER FIRST-INN MODEL")
    log.info("=" * 60)

    # ── Build pitcher features ─────────────────────────────────────────────
    seasons = list(range(2015, datetime.now().year + 1))
    log.info("Loading raw pitches for seasons %s…", seasons)
    raw = load_raw_pitches(seasons)
    if raw.empty:
        log.error("No raw pitch data found."); return

    feats = build_pitcher_features(raw)
    out = PROC_DIR / "pitcher_first_inn_features.parquet"
    feats.to_parquet(out, index=False)
    log.info("Saved pitcher features → %s", out)

    if features_only:
        return feats

    # ── Prepare training data ──────────────────────────────────────────────
    fab_path = PROC_DIR / "first_at_bats.parquet"
    if not fab_path.exists():
        log.error("first_at_bats.parquet not found"); return
    first_ab = pd.read_parquet(fab_path)

    # Career aggregate features per pitcher (collapse across all years)
    career_feats = feats.groupby("pitcher").agg({
        c: "mean" for c in PITCHER_MODEL_FEATURES if c in feats.columns
    }).reset_index()

    df = first_ab.merge(career_feats, on="pitcher", how="left")
    df["game_date"] = pd.to_datetime(df["game_date"])
    log.info("Training rows: %d", len(df))

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

    use_feats = [f for f in PITCHER_MODEL_FEATURES if f in df.columns]
    X_tr = df.reindex(train_df.index)[use_feats].astype(float)
    y_tr = train_df["reached_base"].values
    X_va = df.reindex(val_df.index)[use_feats].astype(float)
    y_va = val_df["reached_base"].values
    X_te = df.reindex(test_df.index)[use_feats].astype(float)
    y_te = test_df["reached_base"].values

    log.info("Train %d | Val %d | Test %d | Features %d",
             len(X_tr), len(X_va), len(X_te), len(use_feats))

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("rf",  RandomForestClassifier(n_estimators=300, max_depth=12,
                                        min_samples_leaf=15, max_features="sqrt",
                                        class_weight="balanced", n_jobs=-1, random_state=42)),
    ])
    pipe.fit(X_tr, y_tr)

    raw_val = pipe.predict_proba(X_va)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_val, y_va)
    model = CalibratedPipeline(pipe, iso)

    def eval_model(X, y, label):
        p = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, p)
        acc = accuracy_score(y, (p >= 0.5).astype(int))
        log.info("  %-10s  AUC %.4f  Acc %.4f", label, auc, acc)
        return auc

    eval_model(X_va, y_va, f"val {val_yr}")
    eval_model(X_te, y_te, f"test {test_yr}")

    rf_est = pipe.named_steps["rf"]
    imp_df = pd.DataFrame({"feature": use_feats, "importance": rf_est.feature_importances_})
    imp_df["imp_pct"] = imp_df["importance"] / imp_df["importance"].sum() * 100
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    imp_df.to_csv(MODEL_DIR / "pitcher_feature_importance.csv", index=False)

    log.info("\nPitcher feature importance (top 10):")
    for _, r in imp_df.head(10).iterrows():
        log.info("  %-40s  %.2f%%", r["feature"], r["imp_pct"])

    joblib.dump({"model": model, "features": use_feats,
                 "val_yr": val_yr, "test_yr": test_yr}, MODEL_DIR / "pitcher_model.pkl")
    log.info("Pitcher model saved → %s", MODEL_DIR / "pitcher_model.pkl")


if __name__ == "__main__":
    import sys
    run(features_only="--features-only" in sys.argv)
