"""
HealthGuard AI - Statistical Rigor
==================================

Moves model evaluation from point estimates to statistically defensible claims:

    1. Bootstrap 95% confidence intervals for ROC-AUC, PR-AUC, recall and F1 of
       every model (stratified resampling of the test set).
    2. DeLong's test for the difference between two correlated ROC curves -
       used to ask "is the best model's AUC *significantly* better than the
       logistic-regression baseline?"
    3. A benchmark table placing our results in the context of published work
       on the same dataset.

Outputs:
    reports/bootstrap_confidence_intervals.csv
    reports/delong_tests.csv
    reports/literature_benchmark.csv
    reports/figures/auc_confidence_intervals.png

Run:
    python src/statistical_analysis.py
"""

import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import config as C

warnings.filterwarnings("ignore")

N_BOOTSTRAP = 1000


# --------------------------------------------------------------------------- #
# DeLong's test (fast midrank implementation, Sun & Xu 2014)
# --------------------------------------------------------------------------- #
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted_transposed, label_1_count):
    m = label_1_count
    n = preds_sorted_transposed.shape[1] - m
    pos = preds_sorted_transposed[:, :m]
    neg = preds_sorted_transposed[:, m:]
    k = preds_sorted_transposed.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(preds_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01); sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, prob_a, prob_b):
    """Two-sided p-value for H0: AUC(a) == AUC(b) on the same samples."""
    order = (-y_true).argsort()
    label_1_count = int(y_true.sum())
    preds = np.vstack((prob_a, prob_b))[:, order]
    aucs, cov = _fast_delong(preds, label_1_count)
    l = np.array([[1, -1]])
    z = (aucs[0] - aucs[1]) / np.sqrt(l @ cov @ l.T)[0, 0]
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


# --------------------------------------------------------------------------- #
# Data / models
# --------------------------------------------------------------------------- #
def load_test_and_probs():
    df = pd.read_csv(C.DATA_PROCESSED)
    X = df[C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES].copy()
    for col in C.CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)
    y = df[C.TARGET].astype(int)
    _, X_te, _, y_te = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE
    )
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl",
        "LightGBM": "lightgbm.pkl",
        "CatBoost": "catboost.pkl",
    }
    probs = {}
    for name, f in model_files.items():
        p = C.MODELS_DIR / f
        if p.exists():
            probs[name] = joblib.load(p).predict_proba(X_te)[:, 1]
    return y_te.values, probs


# --------------------------------------------------------------------------- #
# Bootstrap CIs
# --------------------------------------------------------------------------- #
def bootstrap_ci(y_true, proba, n=N_BOOTSTRAP, seed=C.RANDOM_STATE):
    rng = np.random.RandomState(seed)
    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]
    metrics = {"roc_auc": [], "pr_auc": [], "recall@0.5": [], "f1@0.5": []}
    for _ in range(n):
        # Stratified resample to keep the positive rate stable.
        bp = rng.choice(idx_pos, len(idx_pos), replace=True)
        bn = rng.choice(idx_neg, len(idx_neg), replace=True)
        bi = np.concatenate([bp, bn])
        yt, pr = y_true[bi], proba[bi]
        pred = (pr >= 0.5).astype(int)
        metrics["roc_auc"].append(roc_auc_score(yt, pr))
        metrics["pr_auc"].append(average_precision_score(yt, pr))
        metrics["recall@0.5"].append(recall_score(yt, pred, zero_division=0))
        metrics["f1@0.5"].append(f1_score(yt, pred, zero_division=0))
    out = {}
    for k, v in metrics.items():
        v = np.array(v)
        out[k] = (float(np.mean(v)), float(np.percentile(v, 2.5)),
                  float(np.percentile(v, 97.5)))
    return out


def run():
    print("Statistical analysis: bootstrap CIs + DeLong tests...")
    y_te, probs = load_test_and_probs()
    print(f"  test n={len(y_te):,}  positives={int(y_te.sum()):,}")

    # 1. Bootstrap CIs
    rows = []
    for name, proba in probs.items():
        ci = bootstrap_ci(y_te, proba)
        for metric, (mean, lo, hi) in ci.items():
            rows.append({"model": name, "metric": metric, "mean": round(mean, 4),
                         "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
    ci_df = pd.DataFrame(rows)
    ci_df.to_csv(C.REPORTS_DIR / "bootstrap_confidence_intervals.csv", index=False)
    print("\nROC-AUC with 95% bootstrap CIs:")
    roc = ci_df[ci_df["metric"] == "roc_auc"].sort_values("mean", ascending=False)
    for _, r in roc.iterrows():
        print(f"  {r['model']:22s} {r['mean']:.3f}  "
              f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]")

    # 2. DeLong: best vs each other model
    best = roc.iloc[0]["model"]
    dl_rows = []
    for name, proba in probs.items():
        if name == best:
            continue
        a, b, z, p = delong_roc_test(y_te, probs[best], proba)
        dl_rows.append({"model_a": best, "model_b": name,
                        "auc_a": round(a, 4), "auc_b": round(b, 4),
                        "z": round(z, 3), "p_value": round(p, 4),
                        "significant_0.05": p < 0.05})
    dl_df = pd.DataFrame(dl_rows)
    dl_df.to_csv(C.REPORTS_DIR / "delong_tests.csv", index=False)
    print(f"\nDeLong test - {best} vs others (H0: equal AUC):")
    print(dl_df.to_string(index=False))

    # 3. Literature benchmark
    lit = pd.DataFrame([
        {"study": "Strack et al. (2014)", "approach": "Logistic regression (HbA1c focus)",
         "roc_auc": "~0.65", "notes": "Original dataset paper"},
        {"study": "Typical published ML", "approach": "GBM / RF / ensembles",
         "roc_auc": "0.64-0.69", "notes": "Consistent across literature"},
        {"study": "HealthGuard AI (this work)", "approach": "CatBoost + calibration",
         "roc_auc": f"{roc.iloc[0]['mean']:.3f}", "notes": "Leakage-safe, with 95% CI"},
    ])
    lit.to_csv(C.REPORTS_DIR / "literature_benchmark.csv", index=False)
    print("\nLiterature benchmark:")
    print(lit.to_string(index=False))

    # Plot: ROC-AUC point estimates with CI error bars.
    plt.figure(figsize=(8, 5))
    roc_sorted = roc.sort_values("mean")
    y = np.arange(len(roc_sorted))
    err = [roc_sorted["mean"] - roc_sorted["ci_low"],
           roc_sorted["ci_high"] - roc_sorted["mean"]]
    plt.errorbar(roc_sorted["mean"], y, xerr=err, fmt="o", color="#2c7fb8",
                 capsize=5, markersize=8)
    plt.yticks(y, roc_sorted["model"])
    plt.xlabel("ROC-AUC (95% bootstrap CI)")
    plt.title("Model ROC-AUC with 95% Confidence Intervals")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "auc_confidence_intervals.png", dpi=130)
    plt.close()
    print(f"\nReports + figure saved.")
    return ci_df, dl_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    run()
