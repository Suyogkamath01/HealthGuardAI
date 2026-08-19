"""
HealthGuard AI - Explainable AI (SHAP + LIME)
=============================================

Explains the best model both globally (which features drive readmission risk
across the population) and locally (why a specific patient is high risk).

- SHAP TreeExplainer for the global summary + bar importance (fast, exact for
  tree ensembles). Falls back to a model-agnostic explainer otherwise.
- A reusable `explain_patient()` (SHAP) and `explain_patient_lime()` (LIME) for a
  single encounter, plus `shap_vs_lime()` which cross-checks the two independent
  methods on the same patient - consumed by the dashboard and recommendations.

Outputs:
    reports/figures/shap_summary.png
    reports/figures/shap_importance_bar.png
    reports/figures/shap_vs_lime.png
    reports/shap_global_importance.csv
    reports/example_patient_{shap,lime}.csv

Run:
    python src/explainability.py
"""

import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import config as C

warnings.filterwarnings("ignore")

SAMPLE_SIZE = 2000  # rows used to estimate global SHAP values (speed)


def load_artifacts():
    pipe = joblib.load(C.MODELS_DIR / "best_model.pkl")
    df = pd.read_csv(C.DATA_PROCESSED)
    X = df[C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES].copy()
    for col in C.CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)
    return pipe, X, df[C.TARGET]


def transformed_frame(pipe, X):
    """Apply the fitted preprocessor and return a named DataFrame."""
    pre = pipe.named_steps["pre"]
    Xt = pre.transform(X)
    names = pre.get_feature_names_out()
    # Tidy names: strip the transformer prefix ("num__" / "cat__").
    names = [n.split("__", 1)[-1] for n in names]
    return pd.DataFrame(Xt, columns=names, index=X.index)


def get_explainer(clf, background):
    """TreeExplainer when possible, else a model-agnostic sampler."""
    try:
        return shap.TreeExplainer(clf), "tree"
    except Exception:
        return shap.Explainer(clf.predict_proba, background), "agnostic"


def _positive_class_shap(shap_values):
    """Normalise SHAP output to a 2-D array for the positive class."""
    vals = shap_values.values if hasattr(shap_values, "values") else shap_values
    if isinstance(vals, list):           # older API: [class0, class1]
        return np.asarray(vals[1])
    vals = np.asarray(vals)
    if vals.ndim == 3:                   # (n, features, classes)
        return vals[:, :, 1]
    return vals


def run():
    print("Computing SHAP explanations for the best model...")
    pipe, X, y = load_artifacts()
    clf = pipe.named_steps["clf"]

    rng = np.random.RandomState(C.RANDOM_STATE)
    idx = rng.choice(len(X), size=min(SAMPLE_SIZE, len(X)), replace=False)
    Xt = transformed_frame(pipe, X.iloc[idx])

    explainer, kind = get_explainer(clf, Xt)
    print(f"  using {kind} explainer on {len(Xt):,} rows, {Xt.shape[1]} features")
    shap_values = explainer(Xt) if kind == "tree" else explainer(Xt)
    sv = _positive_class_shap(shap_values)

    # Global importance = mean |SHAP|.
    importance = (
        pd.Series(np.abs(sv).mean(axis=0), index=Xt.columns)
        .sort_values(ascending=False)
    )
    importance.to_csv(C.REPORTS_DIR / "shap_global_importance.csv",
                      header=["mean_abs_shap"])
    print("\nTop 15 drivers of 30-day readmission risk:")
    print(importance.head(15).round(4).to_string())

    # Bar plot.
    top = importance.head(20)[::-1]
    plt.figure(figsize=(8, 8))
    plt.barh(top.index, top.values, color="#2c7fb8")
    plt.xlabel("Mean |SHAP value|")
    plt.title("Global Feature Importance (SHAP) - Top 20")
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "shap_importance_bar.png", dpi=130)
    plt.close()

    # Beeswarm summary.
    try:
        plt.figure()
        shap.summary_plot(sv, Xt, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(C.FIGURES_DIR / "shap_summary.png", dpi=130,
                    bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  (summary plot skipped: {e})")

    # Cross-method check: SHAP vs LIME on the highest-risk sampled patient.
    try:
        proba_sample = pipe.predict_proba(X.iloc[idx])[:, 1]
        hi = X.iloc[idx].iloc[[int(np.argmax(proba_sample))]]
        shap_vs_lime(pipe, hi, background=X.iloc[idx])
        print(f"  SHAP-vs-LIME comparison saved for highest-risk sampled patient "
              f"(p={proba_sample.max():.2f}).")
    except Exception as e:
        print(f"  (SHAP-vs-LIME comparison skipped: {e})")

    print(f"\nFigures -> {C.FIGURES_DIR}")
    return importance


# --------------------------------------------------------------------------- #
# Per-patient explanation (used by dashboard + recommendations)
# --------------------------------------------------------------------------- #
def explain_patient(pipe, patient_row: pd.DataFrame, top_n: int = 8):
    """
    Return a DataFrame of the top_n features pushing this patient's risk up or
    down, with signed SHAP contributions. `patient_row` is a 1-row DataFrame
    with the raw feature columns.
    """
    clf = pipe.named_steps["clf"]
    Xt = transformed_frame(pipe, patient_row)
    explainer, kind = get_explainer(clf, Xt)
    sv = _positive_class_shap(explainer(Xt))
    contrib = pd.Series(sv[0], index=Xt.columns)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top_n)
    return pd.DataFrame(
        {
            "feature": top.index,
            "shap_value": top.values,
            "direction": np.where(top.values >= 0, "increases risk", "decreases risk"),
        }
    )


def explain_patient_lime(pipe, patient_row: pd.DataFrame,
                         background: pd.DataFrame, top_n: int = 8):
    """
    LIME local explanation for one patient, in the transformed feature space.

    LIME fits a sparse linear surrogate around the instance, giving an
    independent second opinion to SHAP. `background` is a sample of raw rows used
    to characterise the feature distribution.
    """
    from lime.lime_tabular import LimeTabularExplainer

    clf = pipe.named_steps["clf"]
    bg = transformed_frame(pipe, background)
    xt = transformed_frame(pipe, patient_row)
    explainer = LimeTabularExplainer(
        bg.values, feature_names=list(bg.columns),
        class_names=["not readmitted", "readmitted"], mode="classification",
        discretize_continuous=True, random_state=C.RANDOM_STATE,
    )
    exp = explainer.explain_instance(
        xt.values[0], clf.predict_proba, num_features=top_n)
    pairs = exp.as_list()  # [(description, signed weight), ...]
    return pd.DataFrame(pairs, columns=["feature", "lime_weight"])


def shap_vs_lime(pipe, patient_row: pd.DataFrame, background: pd.DataFrame,
                 top_n: int = 8):
    """Side-by-side SHAP vs LIME local explanation for one patient."""
    shap_df = explain_patient(pipe, patient_row, top_n=top_n)
    lime_df = explain_patient_lime(pipe, patient_row, background, top_n=top_n)
    shap_df.to_csv(C.REPORTS_DIR / "example_patient_shap.csv", index=False)
    lime_df.to_csv(C.REPORTS_DIR / "example_patient_lime.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    s = shap_df.iloc[::-1]
    axes[0].barh(s["feature"], s["shap_value"],
                 color=np.where(s["shap_value"] >= 0, "#d73027", "#2c7fb8"))
    axes[0].set_title("SHAP (signed contribution)")
    axes[0].axvline(0, color="k", lw=0.8)

    l = lime_df.iloc[::-1]
    axes[1].barh(l["feature"], l["lime_weight"],
                 color=np.where(l["lime_weight"] >= 0, "#d73027", "#2c7fb8"))
    axes[1].set_title("LIME (surrogate weight)")
    axes[1].axvline(0, color="k", lw=0.8)

    fig.suptitle("Per-patient explanation: SHAP vs LIME "
                 "(red = increases risk, blue = decreases)", fontsize=13)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "shap_vs_lime.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return shap_df, lime_df


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    run()
