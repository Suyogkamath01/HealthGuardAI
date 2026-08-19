"""
HealthGuard AI - Original Investigation
=======================================

Research question
-----------------
*How much does patient-level data leakage inflate 30-day readmission model
performance on the Diabetes-130 dataset?*

Many published analyses of this dataset split encounters at random. But ~30% of
encounters are repeat visits by the same patient, so a random split places the
SAME patient in both train and test - letting the model memorise
patient-specific signal and reporting optimistic performance. HealthGuard AI
avoids this by keeping one encounter per patient; this module *quantifies the
size of the bias* that the naive approach introduces.

Design (controlled experiment - only the split changes)
-------------------------------------------------------
- Same data (all encounters after removing death/hospice), same features, same
  model (XGBoost), same number of folds.
- Condition A - NAIVE: StratifiedKFold (a patient may appear in train & test).
- Condition B - LEAKAGE-SAFE: StratifiedGroupKFold grouped by `patient_nbr`
  (a patient appears in only one fold).
- The gap (A - B) is the optimism attributable purely to patient leakage.

Outputs:
    reports/research_leakage.csv
    reports/figures/leakage_optimism.png

Run:
    python src/research_leakage.py
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import config as C
import preprocessing as P
from modeling import build_preprocessor

warnings.filterwarnings("ignore")

N_SPLITS = 5
SCORING = ["roc_auc", "average_precision"]


def load_all_encounters():
    """Clean like preprocessing, but KEEP all encounters (no dedup) + patient id."""
    df = pd.read_csv(C.DATA_RAW).replace(C.MISSING_SENTINEL, np.nan)
    df = df[~df["discharge_disposition_id"].isin(C.EXPIRED_HOSPICE_DISPOSITIONS)]
    df["gender"] = df["gender"].replace("Unknown/Invalid", np.nan)
    df["gender"] = df["gender"].fillna(df["gender"].mode().iloc[0])
    for col in C.NOT_MEASURED_COLUMNS:
        df[col] = df[col].fillna("Not measured")
    for col in ["race", "medical_specialty"]:
        df[col] = df[col].fillna("Unknown")
    # Engineer features; patient_nbr is preserved (only IDs dropped in the
    # production loader, which we deliberately bypass here).
    df = P.engineer(df.reset_index(drop=True))
    return df


def model():
    pos_weight_placeholder = 10.0  # set per-fit below; cross_validate clones est.
    clf = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_weight_placeholder,
        eval_metric="aucpr", tree_method="hist", n_jobs=-1,
        random_state=C.RANDOM_STATE,
    )
    return Pipeline([("pre", build_preprocessor()), ("clf", clf)])


def run():
    print("Original investigation: quantifying patient-leakage optimism...")
    df = load_all_encounters()
    X = df[C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES].copy()
    for col in C.CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)
    y = df[C.TARGET].astype(int)
    groups = df["patient_nbr"].values

    n_patients = df["patient_nbr"].nunique()
    print(f"  encounters={len(df):,}  unique patients={n_patients:,}  "
          f"repeat-visit rows={len(df) - n_patients:,}  "
          f"positive rate={y.mean():.3%}")

    pipe = model()
    naive_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                               random_state=C.RANDOM_STATE)
    grouped_cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                      random_state=C.RANDOM_STATE)

    print("\n  [A] Naive random split (patient may leak across folds)...")
    a = cross_validate(pipe, X, y, scoring=SCORING, cv=naive_cv, n_jobs=-1)
    print("  [B] Patient-grouped split (leakage-safe)...")
    b = cross_validate(pipe, X, y, scoring=SCORING, cv=grouped_cv,
                       groups=groups, n_jobs=-1)

    rows = []
    for metric in SCORING:
        a_mean, a_std = a[f"test_{metric}"].mean(), a[f"test_{metric}"].std()
        b_mean, b_std = b[f"test_{metric}"].mean(), b[f"test_{metric}"].std()
        rows.append({
            "metric": metric,
            "naive_mean": round(a_mean, 4), "naive_std": round(a_std, 4),
            "grouped_mean": round(b_mean, 4), "grouped_std": round(b_std, 4),
            "optimism_abs": round(a_mean - b_mean, 4),
            "optimism_pct": round(100 * (a_mean - b_mean) / b_mean, 2),
        })
    res = pd.DataFrame(rows)
    res.to_csv(C.REPORTS_DIR / "research_leakage.csv", index=False)

    print("\n=== Patient-leakage optimism (naive vs leakage-safe) ===")
    print(res.to_string(index=False))
    for _, r in res.iterrows():
        print(f"  -> Naive splitting inflates {r['metric']} by "
              f"{r['optimism_abs']:+.4f} ({r['optimism_pct']:+.1f}%).")

    # Plot.
    fig, axes = plt.subplots(1, len(SCORING), figsize=(12, 5))
    for ax, metric in zip(np.atleast_1d(axes), SCORING):
        row = res[res["metric"] == metric].iloc[0]
        ax.bar(["Naive\n(leaky)", "Grouped\n(safe)"],
               [row["naive_mean"], row["grouped_mean"]],
               yerr=[row["naive_std"], row["grouped_std"]],
               color=["#d73027", "#2c7fb8"], capsize=6)
        ax.set_title(f"{metric}: optimism = {row['optimism_abs']:+.3f} "
                     f"({row['optimism_pct']:+.1f}%)")
        ax.set_ylabel(metric)
    fig.suptitle("Patient-level leakage inflates apparent performance",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "leakage_optimism.png", dpi=130)
    plt.close(fig)
    print(f"\nReport -> reports/research_leakage.csv  |  Figure -> {C.FIGURES_DIR}")
    return res


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    run()
