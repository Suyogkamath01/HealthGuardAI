"""
HealthGuard AI - Fairness & Bias Analysis
=========================================

Audits the model for disparate behaviour across sensitive groups (race, gender,
age band) using three standard group-fairness criteria:

    - Demographic Parity  : P(predict positive | group) equal across groups.
    - Equal Opportunity   : true-positive rate equal across groups.
    - Equalized Odds       : TPR *and* FPR equal across groups.

For each criterion we report the per-group rate and the disparity (max - min,
plus the ratio min/max - the "4/5ths rule" reference is 0.8).

Outputs:
    reports/fairness_report.csv
    reports/figures/fairness_<attribute>.png

Run:
    python src/fairness.py
"""

import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config as C

warnings.filterwarnings("ignore")


def group_metrics(y_true, y_pred, group_series) -> pd.DataFrame:
    """Per-group selection rate, TPR and FPR."""
    rows = []
    for g, mask in group_series.groupby(group_series).groups.items():
        idx = group_series.index.isin(mask)
        yt, yp = y_true[idx], y_pred[idx]
        pos = yt == 1
        neg = yt == 0
        rows.append(
            {
                "group": g,
                "n": int(idx.sum()),
                "selection_rate": float(yp.mean()),                       # P(ŷ=1)
                "tpr": float(yp[pos].mean()) if pos.any() else np.nan,     # recall
                "fpr": float(yp[neg].mean()) if neg.any() else np.nan,
                "actual_rate": float(yt.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def disparity(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return {"difference": np.nan, "ratio": np.nan}
    return {
        "difference": float(s.max() - s.min()),
        "ratio": float(s.min() / s.max()) if s.max() > 0 else np.nan,
    }


def plot_attribute(attr, gm: pd.DataFrame):
    metrics = ["selection_rate", "tpr", "fpr"]
    titles = ["Demographic Parity\n(selection rate)",
              "Equal Opportunity\n(true positive rate)",
              "Equalized Odds\n(false positive rate)"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, m, t in zip(axes, metrics, titles):
        ax.bar(gm["group"].astype(str), gm[m], color="#3182bd")
        ax.set_title(t)
        ax.set_ylabel(m)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle(f"Fairness across {attr}", fontsize=14)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / f"fairness_{attr}.png", dpi=130)
    plt.close(fig)


def run():
    print("Running fairness analysis on the best model...")
    model = joblib.load(C.MODELS_DIR / "best_model.pkl")
    meta_thr = 0.5
    try:
        import json
        meta_thr = json.load(open(C.MODELS_DIR / "best_model_meta.json"))[
            "decision_threshold"
        ]
    except Exception:
        pass

    df = pd.read_csv(C.DATA_PROCESSED)
    feat = C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES
    X = df[feat].copy()
    for col in C.CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)
    y = df[C.TARGET].astype(int)

    # Reproduce the modelling split, carrying sensitive attributes for the test set.
    idx = np.arange(len(df))
    _, te_idx = train_test_split(
        idx, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE
    )
    X_te, y_te = X.iloc[te_idx], y.iloc[te_idx].reset_index(drop=True)
    sens = df.iloc[te_idx][C.SENSITIVE_FEATURES].reset_index(drop=True)

    proba = model.predict_proba(X_te)[:, 1]
    y_pred = (proba >= meta_thr).astype(int)
    y_te = y_te.values

    all_rows = []
    summary_rows = []
    for attr in C.SENSITIVE_FEATURES:
        gm = group_metrics(pd.Series(y_te), pd.Series(y_pred),
                           sens[attr].reset_index(drop=True))
        gm.insert(0, "attribute", attr)
        all_rows.append(gm)
        plot_attribute(attr, gm)

        dp = disparity(gm["selection_rate"])
        eo = disparity(gm["tpr"])
        fpr_d = disparity(gm["fpr"])
        summary_rows.append(
            {
                "attribute": attr,
                "demographic_parity_diff": round(dp["difference"], 4),
                "demographic_parity_ratio": round(dp["ratio"], 4),
                "equal_opportunity_diff": round(eo["difference"], 4),
                "equalized_odds_fpr_diff": round(fpr_d["difference"], 4),
                "passes_4_5ths_rule": bool(dp["ratio"] >= 0.8)
                if not np.isnan(dp["ratio"]) else None,
            }
        )
        print(f"\n--- {attr} ---")
        print(gm.round(3).to_string(index=False))

    detail = pd.concat(all_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    detail.to_csv(C.REPORTS_DIR / "fairness_detail.csv", index=False)
    summary.to_csv(C.REPORTS_DIR / "fairness_report.csv", index=False)

    print("\n=== Fairness summary (disparities; lower is fairer) ===")
    print(summary.to_string(index=False))
    print(f"\nReports -> {C.REPORTS_DIR}/fairness_report.csv")
    print(f"Figures -> {C.FIGURES_DIR}")
    return summary


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    run()
