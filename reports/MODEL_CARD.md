# Model Card - HealthGuard AI 30-day Readmission Risk Model

*Following the Model Cards framework of Mitchell et al. (2019).*

---

## Model details

- **Developed by:** HealthGuard AI project (academic / portfolio context).
- **Model type:** Gradient-boosted decision tree classifier (CatBoost; best of
  five benchmarked models), wrapped in an `sklearn` pipeline with
  `StandardScaler` + `OneHotEncoder` preprocessing, and isotonic probability
  calibration for the risk-scoring variant.
- **Version / date:** 1.0, 2026.
- **Task:** Binary classification - probability a diabetic inpatient is
  readmitted within **30 days** of discharge.
- **Input:** 13 numeric + 22 categorical features available at discharge
  (demographics, utilisation history, diagnoses grouped from ICD-9, medications,
  lab-test indicators). No free text, no images.
- **Output:** Calibrated probability → 0-100 risk score → tier
  (LOW / MODERATE / HIGH / VERY HIGH).
- **License / repo:** See project `LICENSE`. Reproduce via
  `python src/run_pipeline.py`.

## Intended use

- **Primary intended use:** *Decision support* - to help care teams prioritise
  transitional-care resources (follow-up, medication reconciliation, case
  management) toward higher-risk diabetic patients at discharge. Educational /
  research demonstration.
- **Intended users:** Care-coordination teams, quality-improvement analysts,
  and (in this context) admissions reviewers assessing methodology.
- **Out of scope:** This model is **not** a medical device and must **not** be
  used for autonomous clinical decisions, denial of care, insurance/payment
  decisions, or for non-diabetic or non-US-hospital populations without
  revalidation.

## Factors

- **Relevant groups evaluated:** age band, race, gender (see Fairness audit).
- **Instrumentation / environment:** Trained on retrospective EHR-derived data
  from 130 US hospitals, 1999-2008; practice patterns have since evolved.

## Metrics

- **Discrimination:** ROC-AUC **0.656** (95% bootstrap CI ≈ [0.638, 0.671]);
  PR-AUC **0.181** vs a 0.090 baseline.
- **Operating point:** threshold tuned for F1 → recall ≈ 0.30, precision ≈ 0.21.
- **Calibration:** isotonic-calibrated; Brier ≈ 0.08; calibration holds within
  age and race subgroups (see `reports/clinical_utility.csv`).
- **Clinical utility:** positive net benefit over treat-all / treat-none across
  threshold probabilities ≈ 0.01-0.45 (Decision Curve Analysis).
- **Risk stratification:** observed readmission 6.5% (LOW) → 42.9% (VERY HIGH).
- **Statistical note:** DeLong tests show the five models are **not**
  significantly different in AUC - the signal, not the algorithm, is limiting.

## Training data

- Diabetes 130-US Hospitals dataset (Strack et al., 2014; UCI ML Repository).
- **101,766** raw encounters → **69,990** after removing death/hospice
  discharges and keeping the first encounter per patient (leakage control).
- Positive (30-day readmission) prevalence after cleaning: **9.0%**.
- Imbalance handled via class weights / `scale_pos_weight`.

## Quantitative analyses (fairness)

| Attribute | Demographic-parity ratio | Passes 4/5ths? |
|---|---:|:--:|
| Gender | 0.89 | ✅ |
| Race | 0.56 | ❌ |
| Age | 0.22 | ❌ |

- Largest disparity is by **age**; partly reflects genuinely higher base
  readmission rates in older patients, but the model amplifies it under a single
  threshold.
- **Mitigation applied:** equal-opportunity post-processing (per-group
  thresholds) reduces the age TPR disparity by **~96%** (0.192 → 0.008) at the
  cost of a small precision drop (0.213 → 0.201). See
  `reports/fairness_mitigation.csv`.

## Ethical considerations

- **Risk of feedback loops:** allocating resources by predicted risk can change
  future outcomes; monitor for drift and re-audit fairness periodically.
- **Legitimate vs illegitimate disparity:** age and comorbidity are valid
  clinical risk factors; the goal is to avoid *unjustified amplification*, not
  to erase real risk differences.
- **Human oversight required:** outputs are advisory; a clinician remains
  responsible for care decisions.
- **Data limitations:** historical (1999-2008) US-only diabetic cohort; missing
  social-determinant and post-discharge variables cap achievable performance.

## Caveats and recommendations

- Revalidate before any real-world use, on contemporary, local data.
- Prefer reporting risk **within** age strata operationally.
- Pair with the rule-based recommendation engine so actions are transparent and
  auditable.
