"""
HealthGuard AI - Risk Scoring Engine
====================================

Turns the model's raw probability into a clinician-friendly 0-100 risk score and
a tier (LOW / MODERATE / HIGH / VERY HIGH).

Calibration matters here: an imbalance-weighted classifier outputs distorted
probabilities, so we wrap the best model in isotonic calibration (fit on a
held-out split) before scoring. The 0-100 score is the calibrated probability
rescaled, so "score 87" means a genuinely high estimated probability.

Outputs:
    models/calibrated_model.pkl     calibrated pipeline
    reports/figures/calibration_curve.png
    reports/figures/risk_tier_distribution.png

Run:
    python src/risk_scoring.py
"""

import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split

import config as C

warnings.filterwarnings("ignore")


def score_to_tier(score: float) -> str:
    """Map a 0-100 risk score to a tier label."""
    p = score / 100.0
    for label, lo, hi in C.RISK_TIERS:
        if lo <= p < hi:
            return label
    return C.RISK_TIERS[-1][0]


def probability_to_score(proba) -> np.ndarray:
    """Calibrated probability -> 0-100 integer-ish score."""
    return np.round(np.asarray(proba) * 100, 1)


def build_calibrated_model():
    """Refit the best pipeline with isotonic calibration on a held-out split."""
    base = joblib.load(C.MODELS_DIR / "best_model.pkl")
    df = pd.read_csv(C.DATA_PROCESSED)
    X = df[C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES].copy()
    for col in C.CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)
    y = df[C.TARGET].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE
    )
    # Clone-fit the same pipeline, then calibrate with cross-val isotonic.
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
    calibrated.fit(X_tr, y_tr)
    joblib.dump(calibrated, C.MODELS_DIR / "calibrated_model.pkl")
    return calibrated, X_te, y_te


def run():
    print("Building calibrated risk-scoring model...")
    model, X_te, y_te = build_calibrated_model()
    proba = model.predict_proba(X_te)[:, 1]
    scores = probability_to_score(proba)
    tiers = pd.Series([score_to_tier(s) for s in scores])

    # Calibration curve (reliability diagram).
    frac_pos, mean_pred = calibration_curve(y_te, proba, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot(mean_pred, frac_pos, "o-", label="Calibrated model")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency")
    plt.title("Calibration Curve - 30-day Readmission")
    plt.legend(); plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "calibration_curve.png", dpi=130)
    plt.close()

    # Tier distribution + observed readmission rate per tier (does risk rank?).
    order = [t[0] for t in C.RISK_TIERS]
    summary = (
        pd.DataFrame({"tier": tiers.values, "y": y_te.values})
        .groupby("tier")["y"].agg(["count", "mean"])
        .reindex(order)
    )
    print("\nObserved readmission rate by risk tier (should increase):")
    print(summary.rename(columns={"count": "patients", "mean": "readmit_rate"})
          .round(3).to_string())

    plt.figure(figsize=(7, 5))
    colors = ["#2ca25f", "#fec44f", "#fc8d59", "#d73027"]
    plt.bar(summary.index, summary["mean"], color=colors[: len(summary)])
    for i, (cnt, rate) in enumerate(zip(summary["count"], summary["mean"])):
        if not np.isnan(rate):
            plt.text(i, rate, f"{rate:.1%}\n(n={int(cnt):,})", ha="center", va="bottom")
    plt.ylabel("Observed 30-day readmission rate")
    plt.title("Readmission Rate by Predicted Risk Tier")
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "risk_tier_distribution.png", dpi=130)
    plt.close()

    print(f"\nCalibrated model -> models/calibrated_model.pkl")
    print(f"Figures -> {C.FIGURES_DIR}")
    return summary


def score_patients(model, X: pd.DataFrame) -> pd.DataFrame:
    """Convenience: return scores + tiers for a batch of patients."""
    proba = model.predict_proba(X)[:, 1]
    scores = probability_to_score(proba)
    return pd.DataFrame(
        {"risk_score": scores, "risk_tier": [score_to_tier(s) for s in scores]}
    )


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    run()
