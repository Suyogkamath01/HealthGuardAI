"""
HealthGuard AI - Hyperparameter Optimization (Optuna)
=====================================================

Replaces hand-set hyperparameters with a documented Bayesian search (Optuna
TPE sampler), optimising 3-fold cross-validated PR-AUC on the training set -
the metric that matters under class imbalance.

We tune XGBoost and LightGBM (fast, strong on tabular data), select the better,
refit on the full training set, and report whether tuning yields a meaningful,
*honest* improvement over the defaults.

Outputs:
    models/tuned_model.pkl
    reports/optuna_best_params.json
    reports/optuna_study_summary.csv
    reports/figures/optuna_optimization_history.png

Run:
    python src/tuning.py            # uses N_TRIALS trials per model
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import config as C

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 30


def load_train():
    df = pd.read_csv(C.DATA_PROCESSED)
    X = df[C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES].copy()
    for col in C.CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)
    y = df[C.TARGET].astype(int)
    X_tr, _, y_tr, _ = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE
    )
    return X_tr, y_tr


def preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), C.NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         C.CATEGORICAL_FEATURES),
    ])


def make_objective(kind, X, y, pos_weight, cv):
    def objective(trial):
        if kind == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 700, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            }
            clf = XGBClassifier(
                **params, scale_pos_weight=pos_weight, eval_metric="aucpr",
                tree_method="hist", n_jobs=-1, random_state=C.RANDOM_STATE)
        else:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
                "num_leaves": trial.suggest_int("num_leaves", 16, 128),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            }
            clf = LGBMClassifier(
                **params, class_weight="balanced", n_jobs=-1,
                random_state=C.RANDOM_STATE, verbose=-1)
        pipe = Pipeline([("pre", preprocessor()), ("clf", clf)])
        score = cross_val_score(pipe, X, y, scoring="average_precision",
                                cv=cv, n_jobs=-1)
        return score.mean()
    return objective


def build_model(kind, params, pos_weight):
    if kind == "xgboost":
        clf = XGBClassifier(**params, scale_pos_weight=pos_weight,
                            eval_metric="aucpr", tree_method="hist",
                            n_jobs=-1, random_state=C.RANDOM_STATE)
    else:
        clf = LGBMClassifier(**params, class_weight="balanced", n_jobs=-1,
                            random_state=C.RANDOM_STATE, verbose=-1)
    return Pipeline([("pre", preprocessor()), ("clf", clf)])


def run(n_trials=N_TRIALS):
    print(f"Optuna hyperparameter search ({n_trials} trials/model)...")
    X, y = load_train()
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=C.RANDOM_STATE)

    results = {}
    histories = {}
    for kind in ["xgboost", "lightgbm"]:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=C.RANDOM_STATE))
        study.optimize(make_objective(kind, X, y, pos_weight, cv),
                       n_trials=n_trials, show_progress_bar=False)
        results[kind] = {"best_value": study.best_value, "best_params": study.best_params}
        histories[kind] = [t.value for t in study.trials if t.value is not None]
        print(f"  {kind:9s} best CV PR-AUC = {study.best_value:.4f}")

    best_kind = max(results, key=lambda k: results[k]["best_value"])
    best_params = results[best_kind]["best_params"]
    print(f"\nBest tuned model: {best_kind}  (CV PR-AUC={results[best_kind]['best_value']:.4f})")

    # Refit on full training set and persist.
    tuned = build_model(best_kind, best_params, pos_weight)
    tuned.fit(X, y)
    joblib.dump(tuned, C.MODELS_DIR / "tuned_model.pkl")

    with open(C.REPORTS_DIR / "optuna_best_params.json", "w") as f:
        json.dump({"best_kind": best_kind, **results}, f, indent=2)
    pd.DataFrame([
        {"model": k, "best_cv_pr_auc": round(v["best_value"], 4)}
        for k, v in results.items()
    ]).to_csv(C.REPORTS_DIR / "optuna_study_summary.csv", index=False)

    # Optimization-history plot.
    plt.figure(figsize=(8, 5))
    for kind, vals in histories.items():
        running = np.maximum.accumulate(vals)
        plt.plot(range(1, len(vals) + 1), running, marker="o", label=f"{kind} (best-so-far)")
    plt.xlabel("Trial"); plt.ylabel("CV PR-AUC")
    plt.title("Optuna Optimization History")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "optuna_optimization_history.png", dpi=130)
    plt.close()
    print("Saved tuned model + study artefacts.")
    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    run()
