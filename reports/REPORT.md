# HealthGuard AI
## Explainable Hospital Readmission Risk Prediction & Clinical Decision Support

*A research-style technical report*

---

### Abstract

Unplanned hospital readmission within 30 days of discharge is a costly,
clinically important and policy-relevant outcome - it is used by payers (e.g.
the US CMS Hospital Readmissions Reduction Program) as a quality metric and is
associated with avoidable patient harm. This project develops **HealthGuard
AI**, an end-to-end, explainable machine-learning system that predicts 30-day
readmission risk for diabetic inpatients, attributes each prediction to its
clinical drivers, audits the model for demographic fairness, and converts risk
into an actionable care plan delivered through an interactive dashboard.

Using the Diabetes 130-US Hospitals dataset (101,766 encounters), we apply a
leakage-safe preprocessing methodology, engineer clinically meaningful features
(including a 9-group ICD-9 diagnosis taxonomy), and benchmark five
imbalance-aware classifiers. The best model (CatBoost) achieves **ROC-AUC 0.656**
and **PR-AUC 0.181** - roughly twice the 0.090 random baseline - with risk tiers
that stratify observed readmission from 6.5% (LOW) to 42.9% (VERY HIGH).
SHAP analysis identifies discharge disposition, prior inpatient utilisation,
length of stay and age as the dominant drivers. A fairness audit reveals
disparities across age and race that exceed differences in base rates,
motivating the mitigation discussion. Finally, an original controlled experiment
shows that **patient-level leakage inflates performance only marginally on this
dataset (+0.1% ROC-AUC)** - a slightly counter-intuitive result that reassures
the literature's random-split analyses and demonstrates the pipeline's
robustness. The contribution is not a headline accuracy number - readmission is
intrinsically hard to predict - but a **trustworthy, transparent,
fairness-aware decision-support pipeline**, evaluated with research-grade rigour.

---

## 1. Introduction

### 1.1 Clinical and economic motivation

Hospital readmissions are a major driver of healthcare cost and a recognised
indicator of care quality. A substantial fraction are considered preventable
through better discharge planning, medication reconciliation and timely
follow-up. Diabetic patients are a particularly relevant population: diabetes is
a chronic condition with frequent comorbidities and complex medication
regimens, and HbA1c management has been linked to readmission outcomes.

### 1.2 Problem statement

Given the information available at the time of a patient's discharge, **predict
whether that patient will be readmitted within 30 days**, and do so in a way
that a clinician can understand, trust and act on. Formally, for encounter *i*
with features *xᵢ*, we model

> *P(readmittedᵢ < 30 days | xᵢ)*

and binarise the original three-class label (`NO`, `>30`, `<30`) into

> *y = 1 if `readmitted == "<30"`, else 0.*

### 1.3 Objectives

1. Build a **leakage-safe, reproducible** data pipeline.
2. Benchmark multiple models with **metrics appropriate to class imbalance**.
3. Provide **global and per-patient explanations**.
4. Translate probability into a **calibrated risk score and tier**.
5. **Audit fairness** across race, gender and age.
6. Generate **transparent clinical recommendations**.
7. Deliver everything through an **interactive dashboard**.

---

## 2. Literature Review (brief)

The dataset originates with **Strack et al. (2014)**, *“Impact of HbA1c
Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database
Patient Records”* (BioMed Research International). Their analysis established two
methodological points we adopt: (i) restricting to a single (first) encounter
per patient to avoid statistical dependence, and (ii) removing encounters ending
in death or hospice transfer, since readmission is undefined for them.

Subsequent machine-learning studies on this dataset consistently report
**ROC-AUC in the 0.64-0.69 range**, confirming that 30-day readmission is a
genuinely hard prediction problem with a weak but real signal. This frames our
evaluation: we emphasise **PR-AUC, recall and calibration** over accuracy, and
we treat **explainability and fairness** as first-class deliverables rather than
afterthoughts. The explainability approach follows **Lundberg & Lee (2017)**
(SHAP), and the fairness criteria follow the standard group-fairness literature
(demographic parity, equal opportunity - Hardt et al., 2016 - and equalized
odds).

---

## 3. Dataset

| Property | Value |
|---|---|
| Source | UCI ML Repository / Strack et al. (2014) |
| Hospitals | 130 (US) |
| Period | 1999-2008 |
| Raw encounters | 101,766 |
| Raw features | 50 |
| Original target | `readmitted` ∈ {`NO`, `>30`, `<30`} |

### 3.1 Target distribution (raw)

| Category | Count | Share |
|---|---:|---:|
| NO | 54,864 | 53.9% |
| >30 | 35,545 | 34.9% |
| <30 (positive) | 11,357 | 11.2% |

The positive class is the minority at ~11% before cleaning, making this an
**imbalanced classification** problem.

---

## 4. Data Quality Audit

`src/data_quality.py` normalises the `"?"` sentinel to missing and classifies
every column by a recommended handling strategy. Key findings:

| Column | Missing % | Decision | Reason |
|---|---:|---|---|
| `weight` | 96.9% | **Drop** | Too sparse to impute reliably |
| `max_glu_serum` | 94.8% | **Keep as category** | Missing = *test not performed* (clinically meaningful) |
| `A1Cresult` | 83.3% | **Keep as category** | Missing = *test not performed* |
| `medical_specialty` | 49.1% | Impute `"Unknown"` | Informative when present |
| `payer_code` | 39.6% | **Drop** | Administrative/billing - no clinical signal |
| `race` | 2.2% | Impute `"Unknown"` | Low missingness |
| `diag_1/2/3` | ≤1.4% | Impute / group | Diagnosis codes |

Two further audit results drove design decisions:

- **15 zero-variance drug columns** (a single value covers ≥99% of rows, e.g.
  `examide`, `citoglipton`) carry no predictive signal and were dropped.
- **30,248 encounters are repeat visits by the same patient** (71,518 unique
  patients across 101,766 rows) - a direct **leakage risk** addressed in §5.

> *Insight:* naïvely dropping columns by missingness alone would have discarded
> `A1Cresult` and `max_glu_serum`, whose *absence* is itself predictive. The
> audit distinguishes “data lost” from “test not ordered”.

---

## 5. Methodology - Preprocessing

`src/preprocessing.py`, following Strack et al. (2014):

1. **Normalise** `"?"` → `NaN`.
2. **Remove death/hospice discharges** (`discharge_disposition_id` ∈
   {11,13,14,19,20,21}) → 99,343 rows.
3. **Keep first encounter per patient** → **69,990 rows** (one per patient).
4. **Drop** high-missing (`weight`), zero-variance drugs, administrative
   (`payer_code`) and identifier columns.
5. **Fix** `gender == "Unknown/Invalid"` (3 rows).
6. **Impute** meaningful missingness with explicit categories
   (`"Not measured"`, `"Unknown"`).

After this pipeline the **30-day readmission rate is 9.0%** (the positive
class), which is the baseline any model must beat.

---

## 6. Feature Engineering

| Engineered feature | Description |
|---|---|
| `readmitted_30d` | Binary target (1 if `<30`) |
| `diag_1/2/3_group` | ~900 ICD-9 codes → 9 clinical groups (Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms, Other) |
| `age_ordinal` / `age_group` | Age band → ordinal midpoint + readable band |
| `total_prior_visits` | outpatient + emergency + inpatient |
| `service_utilization` | prior visits + number of diagnoses |
| `n_meds_changed` | count of drugs changed (Up/Down) |
| `n_active_meds` | count of actively prescribed drugs |

The **ICD-9 grouping** is the most important step: with 700-900 distinct codes
per diagnosis column, one-hot encoding is infeasible and noisy; collapsing into
clinically coherent groups makes the diagnosis signal both usable and
interpretable.

Final modelling matrix: **13 numeric + 22 categorical features**.

---

## 7. Modelling

### 7.1 Setup

- **Split:** stratified 80/20 train/test (`random_state=42`).
- **Preprocessing:** `StandardScaler` (numeric) + `OneHotEncoder(handle_unknown="ignore")` (categorical) in a `ColumnTransformer`, wrapped in a `Pipeline` with each estimator.
- **Imbalance handling:** `class_weight="balanced"` (LogReg, RF, LightGBM),
  `scale_pos_weight ≈ 10.1` (XGBoost), `auto_class_weights="Balanced"` (CatBoost).
- **Validation:** 5-fold stratified cross-validation (PR-AUC) on the training
  set, plus held-out test metrics.
- **Threshold:** decision threshold tuned to maximise F1 (the default 0.5 is
  inappropriate under imbalance + class weighting).

### 7.2 Results

Test-set performance, sorted by PR-AUC:

| Model | ROC-AUC | PR-AUC | Recall@best | Precision@best | F1@best | CV PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| **CatBoost** | **0.656** | **0.181** | 0.304 | 0.213 | 0.251 | 0.177 ± 0.009 |
| XGBoost | 0.652 | 0.179 | 0.366 | 0.177 | 0.238 | 0.174 ± 0.010 |
| Random Forest | 0.656 | 0.177 | 0.384 | 0.174 | 0.240 | 0.168 ± 0.008 |
| LightGBM | 0.645 | 0.177 | 0.326 | 0.183 | 0.235 | 0.170 ± 0.008 |
| Logistic Regression | 0.651 | 0.169 | 0.325 | 0.189 | 0.239 | 0.167 ± 0.010 |

*Figures: `reports/figures/roc_curves.png`, `pr_curves.png`,
`confusion_matrix_best.png`.*

### 7.3 Interpretation

- **PR-AUC ≈ 0.18 vs a 0.09 baseline** = roughly a 2× lift in the metric that
  matters under imbalance.
- The gradient-boosted models (CatBoost, XGBoost, LightGBM) edge out the
  baselines, but the **gap between all five models is small** - consistent with
  the literature view that the ceiling on this dataset is modest. This is an
  honest, defensible result rather than an over-fit headline.
- Tree ensembles and a well-regularised logistic regression land in the same
  region, which is itself evidence that the signal - not the algorithm - is the
  limiting factor.

---

## 8. Explainability

`src/explainability.py` uses **SHAP TreeExplainer** on a 2,000-row sample.

**Top global drivers of 30-day readmission risk** (mean |SHAP|):

1. `discharge_disposition_id = 1` (discharged to home)
2. `number_inpatient` (prior inpatient visits)
3. `time_in_hospital` (length of stay)
4. `age_ordinal`
5. `num_lab_procedures`
6. `num_medications`
7. `diag_1_group = Circulatory`
8. `total_prior_visits`
9. `n_active_meds`
10. `number_diagnoses`

*Figures: `reports/figures/shap_importance_bar.png`, `shap_summary.png`.*

These are **clinically coherent**: prior utilisation, complexity (length of
stay, diagnoses, medications) and age are exactly the factors clinicians
associate with readmission risk, which builds trust in the model. The system
also exposes `explain_patient()`, returning the signed top contributors for an
individual.

**SHAP vs LIME cross-check.** Per-patient explanations are produced with two
*independent* methods - SHAP (game-theoretic) and LIME (a local linear
surrogate, `explain_patient_lime()`) - and compared side by side
(`reports/figures/shap_vs_lime.png`). For the example high-risk patient
(p ≈ 0.91), **both methods independently rank the discharge disposition as the
dominant driver**, which increases confidence in that explanation; they diverge
on secondary features, illustrating that local explanations are
method-sensitive and should be cross-validated rather than trusted blindly. This
agreement-where-it-matters, divergence-elsewhere pattern is exactly why we
report both rather than a single method.

---

## 9. Risk Scoring & Calibration

Imbalance-weighted classifiers produce distorted probabilities, so
`src/risk_scoring.py` wraps the best model in **isotonic calibration** and maps
the calibrated probability to a **0-100 score** and four tiers.

**Observed 30-day readmission rate by predicted tier (held-out):**

| Tier | Patients | Observed readmit rate |
|---|---:|---:|
| LOW | 9,961 | 6.5% |
| MODERATE | 3,776 | 13.6% |
| HIGH | 247 | 35.6% |
| VERY HIGH | 14 | 42.9% |

*Figures: `reports/figures/calibration_curve.png`, `risk_tier_distribution.png`.*

The **monotonic increase** from 6.5% to 42.9% shows the score is a genuine risk
stratifier: a clinician can act on “HIGH” knowing ~1 in 3 such patients is
readmitted, versus ~1 in 15 for “LOW”.

---

## 10. Fairness & Bias Analysis

`src/fairness.py` evaluates three group-fairness criteria across **race, gender
and age band** (disparity = max - min across groups; lower is fairer):

| Attribute | Demographic-parity diff | Parity ratio (4/5ths) | Equal-opportunity diff | Equalized-odds (FPR) diff | Passes 4/5ths? |
|---|---:|---:|---:|---:|:--:|
| **Gender** | 0.015 | 0.888 | 0.004 | 0.017 | ✅ Yes |
| **Race** | 0.060 | 0.557 | 0.227 | 0.066 | ❌ No |
| **Age** | 0.166 | 0.217 | 0.192 | 0.160 | ❌ No |

*Figures: `reports/figures/fairness_gender.png`, `fairness_race.png`,
`fairness_age_group.png`.*

### Interpretation & mitigation

- **Gender** behaviour is fair on all three criteria.
- **Age** shows the largest disparity: the 80+ group is flagged positive at
  21.2% vs 4.6% for the 0-30 group. This *partly* reflects a real difference in
  base readmission rate (10.2% vs 5.8%), but the **selection-rate gap (0.166)
  exceeds the base-rate gap**, so the model amplifies the disparity.
- **Race** fails demographic parity, though equalized-odds FPR difference is
  smaller (0.066), suggesting the issue is partly driven by differing base rates
  and sample sizes (e.g. small Asian/Other groups).

**Recommended mitigations** (discussed, not all applied): group-specific
decision thresholds (equalized-odds post-processing à la Hardt et al.),
reweighing during training, or reporting risk *within* age strata so that
clinical resources are allocated by relative rather than absolute risk. Crucially,
because age and comorbidity are *legitimate clinical risk factors*, fairness here
is about **avoiding unjustified amplification**, not removing all disparity.

---

## 10A. Statistical Rigour

Point estimates are not enough to make defensible claims. `src/statistical_analysis.py`
adds:

- **Bootstrap 95% confidence intervals** (1,000 stratified resamples). The best
  model's ROC-AUC is **0.655 [0.638, 0.671]**; all five models' CIs overlap
  heavily.
- **DeLong's test** for correlated ROC curves. Comparing the top model against
  the others, **most pairwise differences are not statistically significant**
  (e.g. CatBoost vs Random Forest p ≈ 0.997; vs XGBoost p ≈ 0.37). Only the
  LightGBM gap reaches p < 0.05.
- **Literature benchmark** (`reports/literature_benchmark.csv`) placing our
  ROC-AUC of 0.655 squarely in the published 0.64-0.69 range.

> *Insight:* the choice of algorithm is **not** the limiting factor on this
> dataset - the signal is. This is exactly the kind of conclusion that point
> estimates would hide and that a confidence-interval + significance-test view
> makes explicit.

*Figure: `reports/figures/auc_confidence_intervals.png`.*

## 10B. Hyperparameter Optimisation

`src/tuning.py` runs an **Optuna** (TPE) Bayesian search over XGBoost and
LightGBM, optimising 3-fold CV PR-AUC. The tuned XGBoost reaches **CV PR-AUC
0.1772**, essentially matching the hand-set defaults (~0.174) and CatBoost
(0.177). Tuning **does not materially move performance**, reinforcing §10A:
returns are capped by the data, not the configuration. (Reported honestly rather
than cherry-picking a lucky seed.)

*Figure: `reports/figures/optuna_optimization_history.png`.*

## 10C. Clinical Utility

Discrimination and calibration still don't tell a clinician whether *acting* on
the model helps. `src/clinical_evaluation.py` adds:

- **Decision Curve Analysis** (Vickers & Elkin, 2006): the model yields
  **positive net benefit over both “treat all” and “treat none”** across
  threshold probabilities ≈ **0.01-0.45** - the clinically plausible range for a
  screening/triage tool.
- **Subgroup calibration:** calibration holds *within* age and race groups
  (predicted ≈ observed; e.g. 80+: predicted 0.108 vs observed 0.102). So the
  earlier fairness gap is about the **operating threshold**, not miscalibration -
  an important distinction.

*Figures: `reports/figures/decision_curve.png`, `subgroup_calibration.png`.*

## 10D. Fairness Mitigation (applied, not just audited)

`src/fairness_mitigation.py` applies **equal-opportunity post-processing**
(per-group thresholds chosen so each age band reaches the same true-positive
rate). Result:

| | Before (global threshold) | After (equal opportunity) |
|---|---:|---:|
| Age TPR disparity | 0.192 | **0.008** |
| Overall recall | 0.304 | 0.305 |
| Overall precision | 0.213 | 0.201 |
| Alert rate | 12.8% | 13.6% |

The TPR disparity across age groups is reduced by **~96%** at the cost of a
small precision drop and a slightly higher alert volume - a concrete, quantified
**fairness-accuracy trade-off** rather than a hand-wave.

*Figure: `reports/figures/fairness_mitigation_tpr.png`.*

## 11. Clinical Recommendation Engine

`src/recommendations.py` is a **transparent, rule-based** layer (deliberately
not a second black-box model) so every recommendation is auditable. It combines:

- a **tier baseline bundle** (e.g. follow-up within 24-48h for VERY HIGH), and
- **feature-gated rules** tied to the SHAP drivers (prior inpatient visits →
  root-cause review; polypharmacy → pharmacist review; elevated A1C → glycaemic
  optimisation; high comorbidity → multidisciplinary coordination).

Recommendations are de-duplicated and ranked Urgent → Recommended → Routine.
Example (VERY HIGH, score 88): rapid follow-up, dedicated case manager, home
monitoring, medication reconciliation, prior-admission review, polypharmacy
review, glycaemic optimisation.

---

## 12. System & Dashboard

A six-page **Streamlit** application (`dashboard/app.py`):

1. **Overview** - summary + model leaderboard
2. **Data Analytics** - interactive cohort EDA (Plotly)
3. **Risk Prediction** - patient form → gauge score, tier, care plan
4. **Explainability** - global SHAP + per-patient attribution
5. **Fairness Analysis** - the audit above, interactively
6. **Forecasting** - population readmission burden + intervention what-if
   (estimates avoidable readmissions and cost from targeting high-risk tiers)

Trained models are persisted under `models/`, so the app runs without
retraining and is deployable to Streamlit Community Cloud.

Two **responsible-AI artefacts** accompany the model: a **Model Card**
([MODEL_CARD.md](MODEL_CARD.md), Mitchell et al., 2019) and a **Datasheet**
([DATASHEET.md](DATASHEET.md), Gebru et al., 2021). The codebase ships with a
**pytest** suite, a **Makefile** and a **CI workflow**, so every result is
reproducible via `python src/run_pipeline.py`.

---

## 12A. Original Investigation - Does Patient Leakage Actually Inflate Performance?

Most published analyses of this dataset split encounters at random. Because ~30%
of encounters are repeat visits by the same patient, a random split lets the
*same patient* appear in both train and test - a textbook leakage risk.
HealthGuard AI avoids this by keeping one encounter per patient. But that raised
a research question worth answering rather than assuming:

> **How large is the optimism that patient-level leakage actually introduces on
> this dataset?**

**Controlled experiment** (`src/research_leakage.py`): identical data (all 99,343
non-death/hospice encounters), identical features, identical model (XGBoost),
identical fold count - *only the splitting strategy changes*:

- **A (naive):** `StratifiedKFold` - a patient may span train and test.
- **B (leakage-safe):** `StratifiedGroupKFold` grouped by `patient_nbr`.

| Metric | Naive (leaky) | Grouped (safe) | Optimism |
|---|---:|---:|---:|
| ROC-AUC | 0.6728 ± 0.005 | 0.6720 ± 0.004 | **+0.0007 (+0.1%)** |
| PR-AUC | 0.2302 ± 0.004 | 0.2277 ± 0.013 | **+0.0025 (+1.1%)** |

**Finding (honest and slightly counter-intuitive):** patient leakage inflates
performance only **marginally** here - about +0.1% ROC-AUC and +1.1% PR-AUC.
This contradicts the usual assumption that leakage badly inflates results.

**Interpretation:** readmission risk in this cohort is driven by
*encounter-level clinical complexity* (length of stay, prior utilisation,
diagnoses) rather than by *memorisable patient identity*, so seeing a patient
before barely helps predict their next outcome. Two consequences: (i) the large
body of published random-split results on this dataset is **not** badly
optimistic - a useful reassurance for the literature; and (ii) the leakage-safe
design is retained as a **principled default**, but we can now *demonstrate* the
project's conclusions are robust to the choice rather than merely asserting it.

*Figure: `reports/figures/leakage_optimism.png`.* (Absolute scores differ
slightly from §7 because this experiment uses all encounters and an XGBoost
configuration, isolating the split effect.)

## 13. Limitations

- **Temporal scope:** data is 1999-2008; practice patterns have since changed.
- **Modest discriminative ceiling:** ROC-AUC ~0.66 means many readmissions are
  driven by post-discharge factors absent from the dataset (social
  determinants, outpatient adherence, home environment).
- **First-encounter restriction** trades realism for statistical cleanliness;
  a production system would model repeat encounters with patient-grouped CV.
- **Forecasting page** uses an illustrative linear intervention-effect model,
  not a causal estimate.
- **Fairness** is assessed at the group level only; individual fairness and
  intersectional subgroups are future work.

---

## 14. Conclusion

HealthGuard AI demonstrates a complete, defensible healthcare-analytics pipeline
on a hard, imbalanced clinical prediction task. Rather than chasing an inflated
accuracy figure, it delivers what a clinical decision-support tool actually
requires: **honest evaluation, calibrated and well-separated risk tiers,
transparent global and per-patient explanations, an explicit fairness audit with
a mitigation plan, and actionable recommendations** - all surfaced through an
interactive dashboard. The result is closer to a healthcare product prototype
than a standalone notebook, and the methodology (leakage prevention, imbalance
-aware metrics, calibration, fairness) generalises to other clinical risk
problems.

---

### References

1. Strack, B. et al. (2014). *Impact of HbA1c Measurement on Hospital
   Readmission Rates.* BioMed Research International.
2. Lundberg, S. & Lee, S. (2017). *A Unified Approach to Interpreting Model
   Predictions (SHAP).* NeurIPS.
3. Hardt, M., Price, E. & Srebro, N. (2016). *Equality of Opportunity in
   Supervised Learning.* NeurIPS.
4. Vickers, A.J. & Elkin, E.B. (2006). *Decision Curve Analysis: A Novel Method
   for Evaluating Prediction Models.* Medical Decision Making.
5. DeLong, E.R., DeLong, D.M. & Clarke-Pearson, D.L. (1988). *Comparing the
   Areas under Two or More Correlated ROC Curves.* Biometrics.
6. Mitchell, M. et al. (2019). *Model Cards for Model Reporting.* FAT*.
7. Gebru, T. et al. (2021). *Datasheets for Datasets.* Communications of the ACM.
8. Akiba, T. et al. (2019). *Optuna: A Next-generation Hyperparameter
   Optimization Framework.* KDD.
9. UCI Machine Learning Repository - *Diabetes 130-US hospitals for years
   1999-2008 Data Set.*

---

*Generated as part of the HealthGuard AI project. All figures in
`reports/figures/`; all metrics reproducible via `python src/run_pipeline.py`.*
