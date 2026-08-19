# 🏥 HealthGuard AI

### Explainable Hospital Readmission Risk Prediction & Clinical Decision Support

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://healthguardai-1.streamlit.app/)
[![CI](https://github.com/Suyogkamath01/HealthGuardAI/actions/workflows/ci.yml/badge.svg)](https://github.com/Suyogkamath01/HealthGuardAI/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/)

**▶️ Try the live app: https://healthguardai-1.streamlit.app/**

HealthGuard AI predicts whether a diabetic patient will be **readmitted to
hospital within 30 days of discharge**, explains *why*, audits the model for
**fairness**, and turns each prediction into an **actionable clinical care
plan** - delivered through an interactive dashboard.

Built on the **Diabetes 130-US Hospitals** dataset (101,766 encounters, 130
hospitals, 1999-2008).

---

## Why this project

30-day readmissions are expensive, are used as a hospital quality metric (e.g.
CMS HRRP), and reflect patient outcomes. This project goes beyond a Kaggle-style
notebook to resemble a healthcare-analytics *product*: leakage-safe data
engineering, honest evaluation of a genuinely hard imbalanced problem,
explainability, a fairness audit, and a decision-support UI.

---

## Headline results

| | |
|---|---|
| Patients (leakage-safe, first encounter each) | **69,990** |
| 30-day readmission rate (positive class) | **9.0%** |
| Best model | **CatBoost** |
| ROC-AUC | **0.656** |
| PR-AUC | **0.181** (≈ 2× the 0.09 random baseline) |
| Risk-tier separation (observed readmit rate) | LOW 6.5% → MODERATE 13.6% → HIGH 35.6% → VERY HIGH 42.9% |

> These figures are consistent with the published literature on this dataset
> (ROC-AUC typically 0.64-0.69). Readmission is hard to predict; the value here
> is **risk stratification + explanation + fairness**, not a magic accuracy
> number.

---

## Architecture

```
data/raw/diabetic_data.csv
        │
        ▼
 src/data_quality.py      → reports/data_quality_report.csv
 src/preprocessing.py     → data/processed/clean_encounters.csv
 src/modeling.py          → models/*.pkl, reports/model_comparison.csv, ROC/PR plots
 src/explainability.py    → SHAP global + per-patient (reports/figures)
 src/risk_scoring.py      → models/calibrated_model.pkl, calibration curve, tiers
 src/fairness.py          → reports/fairness_report.csv, fairness plots
 src/recommendations.py   → rule-based clinical care plans
        │  (advanced analyses)
 src/statistical_analysis.py → bootstrap 95% CIs, DeLong tests, lit. benchmark
 src/clinical_evaluation.py  → decision curve analysis, subgroup calibration
 src/fairness_mitigation.py  → equal-opportunity post-processing + trade-off
 src/tuning.py               → Optuna hyperparameter search (optional)
        │
        ▼
 dashboard/app.py (Streamlit) - 6 pages
```

### Rigour & responsible-AI extras

- **Statistical:** bootstrap 95% confidence intervals on every metric +
  **DeLong's test** - which shows the five models are *not* significantly
  different (the data, not the algorithm, is the ceiling).
- **Clinical:** **Decision Curve Analysis** (net benefit) and subgroup
  calibration - the methods clinical-prediction papers actually use.
- **Fairness:** not just audited but **mitigated** - equal-opportunity
  post-processing cuts the age-group TPR disparity ~96% with a quantified
  precision trade-off.
- **Docs:** a **Model Card** and **Datasheet** ([reports/MODEL_CARD.md](reports/MODEL_CARD.md),
  [reports/DATASHEET.md](reports/DATASHEET.md)).
- **Engineering:** `pytest` suite, `Makefile`, GitHub Actions CI, an executed
  **EDA notebook** ([notebooks/](notebooks/)).
- **Original investigation** ([src/research_leakage.py](src/research_leakage.py)):
  a controlled experiment quantifying patient-level leakage optimism on this
  dataset - and the honest, slightly counter-intuitive finding that it's
  **negligible (+0.1% ROC-AUC)**, showing the conclusions are robust.

---

## Quickstart

```bash
# 1. Create the environment
python -m venv venv && source venv/bin/activate

# 2a. To REPRODUCE the full pipeline (training, SHAP, tuning, etc.):
pip install -r requirements-dev.txt
python src/run_pipeline.py

# 2b. To just RUN the dashboard (lighter install - what the live demo uses):
pip install -r requirements.txt
streamlit run dashboard/app.py
```

> **Two requirements files, on purpose:** `requirements.txt` is the lean
> *runtime* set the deployed dashboard needs (pinned for model compatibility);
> `requirements-dev.txt` is the full set to retrain/reproduce everything.

Each stage is also runnable on its own, e.g. `python src/modeling.py`.

---

## Dashboard pages

1. **Overview** - project summary + model leaderboard
2. **Data Analytics** - cohort EDA (readmission by age/race/diagnosis/utilisation)
3. **Risk Prediction** - enter a patient → 0-100 risk score, tier, and care plan
4. **Explainability** - global SHAP drivers + per-patient attribution
5. **Fairness Analysis** - demographic parity, equal opportunity, equalized odds
6. **Forecasting** - population readmission burden + intervention what-if

---

## Methodology highlights

- **Leakage prevention** - only the *first* encounter per patient is kept, and
  encounters ending in death/hospice are removed (they cannot be readmitted),
  following Strack et al. (2014).
- **Domain feature engineering** - ~900 ICD-9 codes collapsed into 9 clinical
  groups; service-utilisation, prior-visit and medication-change features.
- **Imbalance-aware modelling** - class weights / `scale_pos_weight`; evaluated
  with PR-AUC, recall and F1 at a tuned threshold, not accuracy.
- **Calibration** - isotonic calibration so the 0-100 risk score is meaningful.
- **Explainability** - SHAP (TreeExplainer) globally and per patient.
- **Fairness** - three group-fairness criteria across race, gender and age,
  with the 4/5ths rule as a reference.

See [reports/REPORT.md](reports/REPORT.md) for the full write-up.

---

## Repository layout

```
HealthGuardAI/
├── data/{raw,processed}/
├── src/            config, data_quality, preprocessing, modeling,
│                   explainability, risk_scoring, fairness, recommendations,
│                   run_pipeline
├── models/         trained pipelines + calibrated model + metadata
├── reports/        metrics, fairness report, feature dict, figures/, REPORT.md
├── dashboard/      Streamlit app
└── requirements.txt
```

---

## Deployment

Deployable to **Streamlit Community Cloud** (point it at `dashboard/app.py`) or
any container host. The trained models are committed under `models/`, so the
dashboard runs without retraining. See `.streamlit/config.toml`.

---

*Dataset: Strack et al., “Impact of HbA1c Measurement on Hospital Readmission
Rates,” BioMed Research International, 2014 - via the UCI ML Repository.*
