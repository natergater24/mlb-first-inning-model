#!/usr/bin/env python3
"""
13_train_yrfi_model.py — Train a calibrated RandomForest to predict the
probability that a run scores in the first inning (YRFI).

Data:  data/processed/yrfi_outcomes.parquet  (target + game context)
Feats: src/yrfi_features.build_game_features()  (pitcher NRFI profile, top-5
       batter aggregates, park factors, weather, month, umpire)

Split (time-based):  train 2015-2022 · validate 2023 · test 2024
       (2025-2026 held out of eval; folded into the final fit)
Model:  RandomForestClassifier(class_weight="balanced")
Calibration:  CalibratedClassifierCV(method="isotonic", cv="prefit") on 2023
Eval:  AUC · log loss · Brier · ECE · accuracy · predicted vs actual YRFI rate

Outputs:
  data/models/yrfi_model.pkl            (dict: model, features, imputer medians)
  data/models/yrfi_feature_importance.csv
  data/models/yrfi_model_meta.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                             roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "data" / "models"
MODELS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))
from yrfi_features import (build_game_features, feature_columns, load_support,  # noqa: E402
                           FEATURE_GROUPS, GROUP_LABELS)

TRAIN_YEARS = range(2015, 2023)   # 2015-2022
VAL_YEAR = 2023
TEST_YEAR = 2024
FINAL_EXTRA = (2025,)             # added to the final production fit


def ece(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(p, edges[1:-1])
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            e += m.mean() * abs(y_true[m].mean() - p[m].mean())
    return float(e)


def main() -> int:
    t0 = time.time()
    y = pd.read_parquet(PROC / "yrfi_outcomes.parquet")
    y = y.dropna(subset=["home_starter_id", "away_starter_id"]).copy()
    print(f"games with both starters: {len(y):,}", flush=True)

    # weather join needs venue is_dome — attach park dome per home_team
    prof_map, stats_by_team = load_support()
    print("building features...", flush=True)
    feats = build_game_features(y, prof_map, stats_by_team)
    df = y[["game_pk", "game_year", "yrfi"]].merge(feats, on="game_pk", how="inner")

    cols = feature_columns()
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    tr = df[df["game_year"].isin(TRAIN_YEARS)]
    va = df[df["game_year"] == VAL_YEAR]
    te = df[df["game_year"] == TEST_YEAR]
    print(f"train {len(tr):,} ({min(TRAIN_YEARS)}-{max(TRAIN_YEARS)}) | "
          f"val {len(va):,} ({VAL_YEAR}) | test {len(te):,} ({TEST_YEAR})", flush=True)

    medians = tr[cols].median(numeric_only=True)

    def X(frame):
        return frame[cols].fillna(medians).values

    ytr = tr["yrfi"].astype(int).values
    yva = va["yrfi"].astype(int).values
    yte = te["yrfi"].astype(int).values

    pos = ytr.mean()
    print(f"train YRFI rate: {pos:.3f}  (scale_pos_weight ~ {(1 - pos) / pos:.2f})", flush=True)

    rf = RandomForestClassifier(
        n_estimators=600, max_depth=12, min_samples_leaf=25,
        max_features="sqrt", class_weight="balanced",
        n_jobs=-1, random_state=42,
    )
    rf.fit(X(tr), ytr)

    cal = CalibratedClassifierCV(FrozenEstimator(rf), method="isotonic")
    cal.fit(X(va), yva)

    def evalset(name, frame, ytrue):
        p = cal.predict_proba(X(frame))[:, 1]
        row = {
            "set": name, "n": int(len(ytrue)),
            "auc": round(roc_auc_score(ytrue, p), 4),
            "log_loss": round(log_loss(ytrue, p), 4),
            "brier": round(brier_score_loss(ytrue, p), 4),
            "ece": round(ece(ytrue, p), 4),
            "accuracy": round(accuracy_score(ytrue, (p >= 0.5).astype(int)), 4),
            "pred_yrfi_rate": round(float(p.mean()), 4),
            "actual_yrfi_rate": round(float(ytrue.mean()), 4),
        }
        return row, p

    print("\n=== EVALUATION ===", flush=True)
    results = []
    for nm, fr, yt in (("val_2023", va, yva), ("test_2024", te, yte)):
        r, _ = evalset(nm, fr, yt)
        results.append(r)
        print(f"  {nm}: AUC {r['auc']} | logloss {r['log_loss']} | Brier {r['brier']} "
              f"| ECE {r['ece']} | acc {r['accuracy']} | "
              f"pred {r['pred_yrfi_rate']} vs actual {r['actual_yrfi_rate']}", flush=True)

    # ── final production fit: train + val + test + 2025 ────────────────────
    final_years = list(TRAIN_YEARS) + [VAL_YEAR, TEST_YEAR] + list(FINAL_EXTRA)
    fin = df[df["game_year"].isin(final_years)]
    calib_hold = df[df["game_year"] == 2025]  # calibrate on most recent full season
    fit_part = fin[~fin["game_year"].isin([2025])]
    rf_final = RandomForestClassifier(
        n_estimators=600, max_depth=12, min_samples_leaf=25,
        max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=42)
    rf_final.fit(fit_part[cols].fillna(medians).values, fit_part["yrfi"].astype(int).values)
    cal_final = CalibratedClassifierCV(FrozenEstimator(rf_final), method="isotonic")
    cal_final.fit(calib_hold[cols].fillna(medians).values,
                  calib_hold["yrfi"].astype(int).values)

    fi = (pd.DataFrame({"feature": cols, "importance": rf_final.feature_importances_})
          .sort_values("importance", ascending=False).reset_index(drop=True))
    fi.to_csv(MODELS / "yrfi_feature_importance.csv", index=False)

    # ── interpretable additive surrogate (for the dashboard weighting sliders) ──
    # A logistic regression on standardised features gives an additive log-odds
    # decomposition: contribution of feature f to game g = coef_f * z_f(g).
    # src/14 sums these by FEATURE_GROUPS; the dashboard sliders then scale each
    # group's contribution up or down around the model's own prediction.
    from sklearn.linear_model import LogisticRegression
    Xfit = fit_part[cols].fillna(medians).fillna(0.0)
    z_mean = Xfit.mean()
    z_std = Xfit.std().replace(0, 1.0)
    Xz = ((Xfit - z_mean) / z_std).fillna(0.0)
    lr = LogisticRegression(max_iter=2000, C=0.5)
    lr.fit(Xz.values, fit_part["yrfi"].astype(int).values)
    _te_z = ((te[cols].fillna(medians).fillna(0.0) - z_mean) / z_std).fillna(0.0)
    lr_auc = roc_auc_score(te["yrfi"].astype(int).values,
                           lr.predict_proba(_te_z.values)[:, 1])
    surrogate = {
        "coef": dict(zip(cols, [float(c) for c in lr.coef_[0]])),
        "intercept": float(lr.intercept_[0]),
        "mean": z_mean.to_dict(),
        "std": z_std.to_dict(),
        "groups": FEATURE_GROUPS,
        "group_labels": GROUP_LABELS,
        "test_auc": round(float(lr_auc), 4),
    }
    # average absolute per-group contribution over the training set (a picture of
    # "what the model currently leans on") — saved for the dashboard.
    Zall = Xz
    group_influence = {}
    for gname, members in FEATURE_GROUPS.items():
        mem = [m for m in members if m in cols]
        contrib = Zall[mem].values @ np.array([surrogate["coef"][m] for m in mem])
        group_influence[gname] = round(float(np.mean(np.abs(contrib))), 4)
    surrogate["group_influence"] = group_influence

    import pickle
    with open(MODELS / "yrfi_model.pkl", "wb") as fh:
        pickle.dump({
            "model": cal_final,
            "features": cols,
            "medians": medians.to_dict(),
            "surrogate": surrogate,
            "trained": time.strftime("%Y-%m-%d %H:%M"),
        }, fh)

    print(f"\nsurrogate LR test AUC {surrogate['test_auc']} | group influence "
          f"(avg |log-odds|): " + ", ".join(
              f"{g}={v}" for g, v in sorted(group_influence.items(),
                                            key=lambda kv: -kv[1])), flush=True)

    meta = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target": "yrfi (a run scores in the 1st inning, either team)",
        "n_games_total": int(len(df)),
        "split": {"train": f"{min(TRAIN_YEARS)}-{max(TRAIN_YEARS)}",
                  "val": VAL_YEAR, "test": TEST_YEAR,
                  "final_fit": f"{min(final_years)}-{max(final_years)} (calib on 2025)"},
        "model": "RandomForestClassifier(n=600, depth=12, leaf=25, class_weight=balanced)"
                 " + isotonic CalibratedClassifierCV",
        "train_yrfi_rate": round(float(pos), 4),
        "metrics": results,
        "top_features": fi.head(15).to_dict("records"),
        "weighting": {
            "surrogate_test_auc": surrogate["test_auc"],
            "group_labels": GROUP_LABELS,
            "group_influence": group_influence,   # avg |log-odds| the model leans on
        },
        "caveats": [
            "Pitcher NRFI profile & top-5 batter aggregates are current static "
            "values, not point-in-time — career features carry mild look-ahead "
            "bias; test AUC is a modest over-estimate.",
            "top5 batter aggregates for historical games use 2026 projected "
            "lineups (team-level approximation).",
            "umpire_zone_adj is a placeholder (0.0) for almost all games.",
        ],
    }
    (MODELS / "yrfi_model_meta.json").write_text(json.dumps(meta, indent=2))

    print("\n=== FEATURE IMPORTANCE (all) ===", flush=True)
    for _, r in fi.iterrows():
        print(f"  {r['feature']:<40} {r['importance']:.4f}", flush=True)

    print(f"\nSAVED yrfi_model.pkl · yrfi_feature_importance.csv · yrfi_model_meta.json")
    print(f"total time: {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
