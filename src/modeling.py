"""
HealthGuard AI - Model Training & Evaluation
============================================

Trains five classifiers to predict 30-day readmission, all with explicit
class-imbalance handling, and evaluates them with metrics appropriate for an
imbalanced problem (PR-AUC, ROC-AUC, recall, F1) rather than accuracy.

Models:
    Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost.

Outputs:
    models/<model>.pkl              fitted sklearn pipelines
    models/best_model.pkl           best pipeline (by PR-AUC)
    models/best_model_meta.json     name, threshold, feature lists
    reports/model_comparison.csv    metric table
    reports/figures/*.png           ROC, PR, confusion-matrix plots

Run:
    python src/modeling.py
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import config as C

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_split():
    df = pd.read_csv(C.DATA_PROCESSED)
    X = df[C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES].copy()
    y = df[C.TARGET].astype(int)
    # Categoricals as strings so OneHotEncoder treats numeric IDs as categories.
    for col in C.CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)
    return train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE
    )


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), C.NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                C.CATEGORICAL_FEATURES,
            ),
        ]
    )


# --------------------------------------------------------------------------- #
# Models (all imbalance-aware)
# --------------------------------------------------------------------------- #
def get_models(pos_weight: float) -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0, n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=18, min_samples_leaf=20,
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=C.RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_weight,
            eval_metric="aucpr", tree_method="hist", n_jobs=-1,
            random_state=C.RANDOM_STATE,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500, num_leaves=48, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
            n_jobs=-1, random_state=C.RANDOM_STATE, verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            auto_class_weights="Balanced", random_seed=C.RANDOM_STATE,
            verbose=False, allow_writing_files=False,
        ),
    }


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def best_f1_threshold(y_true, proba):
    """Threshold on predicted probability that maximises F1."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = np.divide(
        2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0
    )
    # precision_recall_curve returns one extra point with no threshold.
    best = int(np.nanargmax(f1[:-1]))
    return float(thr[best]), float(f1[best])


def evaluate(name, pipe, X_te, y_te):
    proba = pipe.predict_proba(X_te)[:, 1]
    thr, f1_at_thr = best_f1_threshold(y_te, proba)
    pred_def = (proba >= 0.5).astype(int)
    pred_thr = (proba >= thr).astype(int)
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_te, proba),
        "pr_auc": average_precision_score(y_te, proba),
        "brier": brier_score_loss(y_te, proba),
        "recall@0.5": recall_score(y_te, pred_def, zero_division=0),
        "precision@0.5": precision_score(y_te, pred_def, zero_division=0),
        "f1@0.5": f1_score(y_te, pred_def, zero_division=0),
        "best_threshold": thr,
        "recall@best": recall_score(y_te, pred_thr, zero_division=0),
        "precision@best": precision_score(y_te, pred_thr, zero_division=0),
        "f1@best": f1_at_thr,
        "_proba": proba,
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_curves(results, y_te):
    # ROC
    plt.figure(figsize=(7, 6))
    for r in results:
        fpr, tpr, _ = roc_curve(y_te, r["_proba"])
        plt.plot(fpr, tpr, label=f"{r['model']} (AUC={r['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - 30-day Readmission"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(C.FIGURES_DIR / "roc_curves.png", dpi=130)
    plt.close()

    # PR
    base = float(np.mean(y_te))
    plt.figure(figsize=(7, 6))
    for r in results:
        prec, rec, _ = precision_recall_curve(y_te, r["_proba"])
        plt.plot(rec, prec, label=f"{r['model']} (AP={r['pr_auc']:.3f})")
    plt.axhline(base, color="k", ls="--", alpha=0.4, label=f"baseline={base:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curves - 30-day Readmission")
    plt.legend(loc="upper right")
    plt.tight_layout(); plt.savefig(C.FIGURES_DIR / "pr_curves.png", dpi=130)
    plt.close()


def plot_confusion(name, proba, thr, y_te):
    pred = (proba >= thr).astype(int)
    cm = confusion_matrix(y_te, pred)
    plt.figure(figsize=(5, 4.5))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["Not readmit", "Readmit <30d"])
    plt.yticks([0, 1], ["Not readmit", "Readmit <30d"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name} (thr={thr:.2f})")
    plt.colorbar(fraction=0.046)
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "confusion_matrix_best.png", dpi=130)
    plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run():
    print("Loading processed data...")
    X_tr, X_te, y_tr, y_te = load_split()
    pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())
    print(f"  train={len(X_tr):,}  test={len(X_te):,}  pos_weight={pos_weight:.2f}")

    pre = build_preprocessor()
    models = get_models(pos_weight)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)

    results = []
    for name, clf in models.items():
        print(f"\nTraining {name} ...")
        pipe = Pipeline([("pre", pre), ("clf", clf)])

        # 5-fold CV PR-AUC on training data (robustness check).
        cv_ap = cross_val_score(
            pipe, X_tr, y_tr, scoring="average_precision", cv=cv, n_jobs=-1
        )
        pipe.fit(X_tr, y_tr)
        res = evaluate(name, pipe, X_te, y_te)
        res["cv_pr_auc_mean"] = float(cv_ap.mean())
        res["cv_pr_auc_std"] = float(cv_ap.std())
        results.append(res)
        joblib.dump(pipe, C.MODELS_DIR / f"{name.lower().replace(' ', '_')}.pkl")
        print(
            f"  PR-AUC={res['pr_auc']:.3f}  ROC-AUC={res['roc_auc']:.3f}  "
            f"recall@best={res['recall@best']:.3f}  F1@best={res['f1@best']:.3f}  "
            f"CV PR-AUC={res['cv_pr_auc_mean']:.3f}±{res['cv_pr_auc_std']:.3f}"
        )

    # Comparison table (drop the heavy _proba column).
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "_proba"}
                          for r in results])
    table = table.sort_values("pr_auc", ascending=False).reset_index(drop=True)
    table.to_csv(C.REPORTS_DIR / "model_comparison.csv", index=False)
    print("\n=== Model comparison (sorted by PR-AUC) ===")
    print(table.round(4).to_string(index=False))

    # Plots + best model.
    plot_curves(results, y_te)
    best_name = table.iloc[0]["model"]
    best = next(r for r in results if r["model"] == best_name)
    plot_confusion(best_name, best["_proba"], best["best_threshold"], y_te)

    best_pipe = joblib.load(
        C.MODELS_DIR / f"{best_name.lower().replace(' ', '_')}.pkl"
    )
    joblib.dump(best_pipe, C.MODELS_DIR / "best_model.pkl")
    meta = {
        "best_model": best_name,
        "decision_threshold": best["best_threshold"],
        "pr_auc": best["pr_auc"],
        "roc_auc": best["roc_auc"],
        "numeric_features": C.NUMERIC_FEATURES,
        "categorical_features": C.CATEGORICAL_FEATURES,
        "target": C.TARGET,
        "train_readmission_rate": float(y_tr.mean()),
    }
    with open(C.MODELS_DIR / "best_model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nBest model: {best_name}  ->  models/best_model.pkl")
    print(f"Figures -> {C.FIGURES_DIR}")
    return table


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    run()
