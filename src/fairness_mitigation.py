"""
HealthGuard AI - Fairness Mitigation
====================================

The fairness audit (src/fairness.py) found the largest disparity by AGE: under a
single global decision threshold, true-positive rates differ markedly across age
bands. Auditing is not enough - here we *apply* a mitigation and quantify the
trade-off.

Method: equal-opportunity post-processing (a special case of Hardt et al.,
2016). Instead of one threshold for everyone, we choose a per-group threshold so
that each group achieves approximately the same true-positive rate (recall).
This is the most defensible mitigation for a screening tool whose goal is to not
*miss* high-risk patients in any group.

We report the disparity before vs after, and the cost (change in overall
precision / alert volume) - because fairness is never free.

Outputs:
    reports/fairness_mitigation.csv
    reports/figures/fairness_mitigation_tpr.png

Run:
    python src/fairness_mitigation.py
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

import config as C

warnings.filterwarnings("ignore")

SENSITIVE = "age_group"  # the attribute with the largest measured disparity


def load_scored():
    # Use the uncalibrated best model + its tuned threshold, matching the
    # fairness audit (src/fairness.py) so the mitigation is coherent with the
    # disparity it is correcting.
    model = joblib.load(C.MODELS_DIR / "best_model.pkl")
    df = pd.read_csv(C.DATA_PROCESSED)
    X = df[C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES].copy()
    for col in C.CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)
    y = df[C.TARGET].astype(int)
    idx = np.arange(len(df))
    _, te = train_test_split(idx, test_size=C.TEST_SIZE, stratify=y,
                             random_state=C.RANDOM_STATE)
    proba = model.predict_proba(X.iloc[te])[:, 1]
    groups = df.iloc[te][SENSITIVE].reset_index(drop=True)
    return y.iloc[te].values, proba, groups


def tpr_at(y, p, thr):
    pred = (p >= thr).astype(int)
    pos = y == 1
    return float(pred[pos].mean()) if pos.any() else np.nan


def group_threshold_for_tpr(y_g, p_g, target_tpr):
    """Smallest threshold achieving at least target TPR within the group."""
    grid = np.linspace(0.01, 0.99, 197)
    best_thr, best_gap = 0.5, np.inf
    for t in grid:
        tpr = tpr_at(y_g, p_g, t)
        gap = abs(tpr - target_tpr)
        if gap < best_gap:
            best_gap, best_thr = gap, t
    return best_thr


def metrics_by_group(y, pred, groups):
    rows = []
    for g in sorted(groups.unique()):
        m = (groups == g).values
        rows.append({
            "group": g, "n": int(m.sum()),
            "tpr": recall_score(y[m], pred[m], zero_division=0),
            "selection_rate": float(pred[m].mean()),
        })
    return pd.DataFrame(rows)


def run():
    print("Fairness mitigation: equal-opportunity post-processing by age...")
    y, proba, groups = load_scored()

    # Global threshold from the trained model metadata.
    try:
        global_thr = json.load(open(C.MODELS_DIR / "best_model_meta.json"))[
            "decision_threshold"]
    except Exception:
        global_thr = 0.5

    # --- Baseline: single global threshold ---
    pred_base = (proba >= global_thr).astype(int)
    base = metrics_by_group(y, pred_base, groups)
    target_tpr = recall_score(y, pred_base, zero_division=0)  # overall TPR target

    # --- Mitigation: per-group thresholds targeting the overall TPR ---
    thresholds = {}
    pred_mit = np.zeros_like(pred_base)
    for g in groups.unique():
        m = (groups == g).values
        t = group_threshold_for_tpr(y[m], proba[m], target_tpr)
        thresholds[g] = round(float(t), 3)
        pred_mit[m] = (proba[m] >= t).astype(int)
    mit = metrics_by_group(y, pred_mit, groups)

    # --- Compare ---
    disparity_before = base["tpr"].max() - base["tpr"].min()
    disparity_after = mit["tpr"].max() - mit["tpr"].min()
    summary = base.merge(mit, on="group", suffixes=("_before", "_after"))
    summary["threshold"] = summary["group"].map(thresholds)
    summary.to_csv(C.REPORTS_DIR / "fairness_mitigation.csv", index=False)

    print(f"\n  Per-group thresholds: {thresholds}")
    print("\n  TPR by age group, before vs after:")
    print(summary[["group", "n_before", "tpr_before", "tpr_after",
                   "selection_rate_before", "selection_rate_after"]]
          .rename(columns={"n_before": "n"})
          .round(3).to_string(index=False))

    overall = {
        "tpr_disparity_before": round(float(disparity_before), 4),
        "tpr_disparity_after": round(float(disparity_after), 4),
        "disparity_reduction_pct": round(
            100 * (1 - disparity_after / disparity_before), 1)
        if disparity_before > 0 else 0.0,
        "overall_recall_before": round(recall_score(y, pred_base, zero_division=0), 4),
        "overall_recall_after": round(recall_score(y, pred_mit, zero_division=0), 4),
        "overall_precision_before": round(precision_score(y, pred_base, zero_division=0), 4),
        "overall_precision_after": round(precision_score(y, pred_mit, zero_division=0), 4),
        "alert_rate_before": round(float(pred_base.mean()), 4),
        "alert_rate_after": round(float(pred_mit.mean()), 4),
    }
    print("\n  Trade-off summary:")
    for k, v in overall.items():
        print(f"    {k:28s}: {v}")
    print(f"\n  TPR disparity reduced {disparity_before:.3f} -> {disparity_after:.3f} "
          f"({overall['disparity_reduction_pct']:.0f}% reduction).")

    # Plot before/after TPR by group.
    x = np.arange(len(summary))
    plt.figure(figsize=(9, 5))
    plt.bar(x - 0.2, summary["tpr_before"], 0.4, label="Before (global threshold)",
            color="#fc8d59")
    plt.bar(x + 0.2, summary["tpr_after"], 0.4, label="After (equal opportunity)",
            color="#2c7fb8")
    plt.axhline(target_tpr, color="black", ls="--", alpha=0.6,
                label=f"target TPR={target_tpr:.2f}")
    plt.xticks(x, summary["group"]); plt.ylabel("True positive rate (recall)")
    plt.title("Fairness Mitigation: TPR by Age Group, Before vs After")
    plt.legend(); plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "fairness_mitigation_tpr.png", dpi=130)
    plt.close()
    print(f"  Figure -> {C.FIGURES_DIR}/fairness_mitigation_tpr.png")
    return summary, overall


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    run()
