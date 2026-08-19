# Datasheet - Diabetes 130-US Hospitals Dataset

*Following the Datasheets for Datasets framework of Gebru et al. (2021), as used
in HealthGuard AI.*

---

## Motivation

- **Why was the dataset created?** To study the relationship between HbA1c
  measurement and hospital readmission in diabetic inpatients (Strack et al.,
  2014). It has since become a standard benchmark for readmission prediction.
- **Who created / funded it?** Compiled from the Health Facts database (Cerner
  Corporation) by the authors of Strack et al.; distributed via the UCI Machine
  Learning Repository.

## Composition

- **What does each instance represent?** One **hospital encounter** for a
  patient with a diabetes diagnosis.
- **How many instances?** 101,766 encounters; 50 attributes.
- **Unit of analysis caveat:** patients can appear multiple times (71,518 unique
  patients). HealthGuard AI keeps only the **first encounter per patient** to
  avoid statistical dependence / leakage.
- **What features?** Demographics (race, gender, age band), admission/discharge
  codes, time in hospital, counts of procedures / labs / medications / prior
  visits, up to three ICD-9 diagnoses, 23 medication indicators, HbA1c and
  glucose-serum test results, and the readmission label.
- **Label:** `readmitted` ∈ {`NO`, `>30`, `<30`}; HealthGuard binarises to
  `<30` = positive.
- **Missing data?** Yes - encoded as `"?"`. Notably `weight` (~97%),
  `medical_specialty` (~49%), `payer_code` (~40%); `max_glu_serum` /
  `A1Cresult` are usually absent because the **test was not ordered** (a
  meaningful, not missing-at-random, signal).
- **Is it a sample or a census?** A sample of US inpatient diabetic encounters
  1999-2008, restricted to stays of 1-14 days with labs and medications
  administered.

## Collection process

- **How was it acquired?** Extracted from a clinical EHR data warehouse; derived
  from routine care documentation rather than purpose-collected research data.
- **Timeframe:** 1999-2008.
- **Ethics / consent:** De-identified per HIPAA prior to release; the public UCI
  version contains no direct identifiers.

## Preprocessing / cleaning / labelling (as applied here)

- `"?"` → missing; death/hospice discharges removed; first encounter per patient
  retained; high-missing / zero-variance / administrative columns dropped;
  ICD-9 → 9 clinical groups; utilisation & medication features engineered.
- See `src/data_quality.py` and `src/preprocessing.py` for the exact, auditable
  steps; the cleaned table is `data/processed/clean_encounters.csv`.

## Uses

- **Suitable for:** benchmarking readmission-risk models; methodology
  demonstrations (imbalance handling, calibration, explainability, fairness).
- **Not suitable for:** drawing contemporary clinical conclusions (data is
  15-25 years old), non-diabetic or non-US populations, or any deployment
  without revalidation.
- **Known biases / risks:** demographic composition skews Caucasian; small
  Asian/Hispanic/Other subgroups limit subgroup statistical power; historical
  coding practices (ICD-9) differ from current (ICD-10).

## Distribution & maintenance

- **Available from:** UCI Machine Learning Repository (“Diabetes 130-US
  hospitals for years 1999-2008”).
- **License:** released for research use via UCI; cite Strack et al. (2014).
- **Maintenance:** static dataset; not updated.

## Citation

Strack, B., DeShazo, J.P., Gennings, C., Olmo, J.L., Ventura, S., Cios, K.J.,
Clore, J.N. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates:
Analysis of 70,000 Clinical Database Patient Records.* BioMed Research
International.
