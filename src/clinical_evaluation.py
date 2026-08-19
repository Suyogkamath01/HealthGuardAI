"""
HealthGuard AI - Clinical Utility Evaluation
============================================

Discrimination (AUC) and calibration are necessary but not sufficient to show a
model is *clinically useful*. This module adds the two analyses that clinical
prediction papers rely on:

    1. Decision Curve Analysis (Vickers & Elkin, 2006) - plots NET BENEFIT
       across the range of threshold probabilities a clinician might use, versus
       the "treat all" and "treat none" default strategies. A model is useful
       only where its curve sits above both defaults.

    2. Subgroup calibration - calibration is checked WITHIN age and race groups,
       because a model can be well calibrated overall yet miscalibrated for a
       subgroup (a fairness-relevant failure mode).

Outputs:
    reports/figures/decision_curve.png
    reports/figures/subgroup_calibration.png
    reports/clinical_utility.csv

Run:
    python src/clinical_evaluation.py
"""

import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

import config as C

warnings.filterwarnings("ignore")


def load_scored():
    """Load test set with calibrated probabilities + sensitive attributes."""
    model_path = (C.MODELS_DIR / "calibrated_model.pkl")
    if not model_path.exists():
        model_path = C.MODELS_DIR / "best_model.pkl"
    model = joblib.load(model_path)

    df = pd.read_csv(C.DATA_PROCESSED)
    feat = C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES
    X = df[feat].copy()
    for col in C.CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)
    y = df[C.TARGET].astype(int)

    idx = np.arange(len(df))
    _, te = train_test_split(idx, test_size=C.TEST_SIZE, stratify=y,
                             random_state=C.RANDOM_STATE)
    X_te, y_te = X.iloc[te], y.iloc[te].values
    proba = model.predict_proba(X_te)[:, 1]
    sens = df.iloc[te][["age_group", "race"]].reset_index(drop=True)
    return y_te, proba, sens


# --------------------------------------------------------------------------- #
# Decision Curve Analysis
# --------------------------------------------------------------------------- #
def net_benefit(y_true, proba, thresholds):
    """
    Net benefit = TP/n - FP/n * (pt / (1 - pt)), for each threshold probability
    pt. Compared against treat-all and treat-none.
    """
    n = len(y_true)
    prevalence = y_true.mean()
    nb_model, nb_all = [], []
    for pt in thresholds:
        pred = (proba >= pt).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        w = pt / (1 - pt) if pt < 1 else np.inf
        nb_model.append(tp / n - (fp / n) * w)
        # Treat-all: everyone flagged positive.
        nb_all.append(prevalence - (1 - prevalence) * w)
    return np.array(nb_model), np.array(nb_all)


def plot_decision_curve(y_true, proba):
    thresholds = np.linspace(0.01, 0.50, 50)
    nb_model, nb_all = net_benefit(y_true, proba, thresholds)
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, nb_model, label="HealthGuard model", color="#2c7fb8", lw=2)
    plt.plot(thresholds, nb_all, label="Treat all", color="#999999", ls="--")
    plt.axhline(0, color="black", lw=1, label="Treat none")
    plt.ylim(min(-0.02, nb_model.min()), max(nb_model.max(), 0.05) * 1.2)
    plt.xlabel("Threshold probability (pt)")
    plt.ylabel("Net benefit")
    plt.title("Decision Curve Analysis - 30-day Readmission")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "decision_curve.png", dpi=130)
    plt.close()
    # Range where model beats both defaults.
    better = thresholds[(nb_model > nb_all) & (nb_model > 0)]
    rng = (float(better.min()), float(better.max())) if len(better) else (None, None)
    return rng


# --------------------------------------------------------------------------- #
# Subgroup calibration
# --------------------------------------------------------------------------- #
def plot_subgroup_calibration(y_true, proba, sens):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rows = []
    for ax, attr in zip(axes, ["age_group", "race"]):
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        groups = sens[attr].value_counts()
        for g in groups[groups >= 200].index:  # only groups with enough data
            m = (sens[attr] == g).values
            if y_true[m].sum() < 10:
                continue
            try:
                frac, mean_pred = calibration_curve(
                    y_true[m], proba[m], n_bins=5, strategy="quantile")
                ax.plot(mean_pred, frac, "o-", label=f"{g} (n={m.sum()})")
                rows.append({"attribute": attr, "group": g, "n": int(m.sum()),
                             "brier": round(brier_score_loss(y_true[m], proba[m]), 4),
                             "observed_rate": round(float(y_true[m].mean()), 4),
                             "mean_pred": round(float(proba[m].mean()), 4)})
            except Exception:
                continue
        ax.set_title(f"Calibration within {attr}")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "subgroup_calibration.png", dpi=130)
    plt.close(fig)
    return pd.DataFrame(rows)


def run():
    print("Clinical utility evaluation...")
    y_te, proba, sens = load_scored()

    rng = plot_decision_curve(y_te, proba)
    if rng[0] is not None:
        print(f"  Decision curve: model gives positive net benefit above "
              f"both defaults for threshold probabilities ~{rng[0]:.2f}-{rng[1]:.2f}.")
    else:
        print("  Decision curve generated.")

    sub = plot_subgroup_calibration(y_te, proba, sens)
    sub.to_csv(C.REPORTS_DIR / "clinical_utility.csv", index=False)
    print("\nSubgroup calibration (Brier score; lower is better):")
    print(sub.to_string(index=False))
    print(f"\nFigures -> {C.FIGURES_DIR}")
    return sub


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    run()
