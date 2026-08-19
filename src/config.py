"""
HealthGuard AI - Central Configuration
======================================

Single source of truth for paths, column groups, modelling constants and the
domain mappings (ICD-9 diagnosis grouping, discharge dispositions) used across
the pipeline. Importing from here keeps preprocessing, modelling, the dashboard
and the report perfectly consistent.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "diabetic_data.csv"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
DATA_PROCESSED = DATA_PROCESSED_DIR / "clean_encounters.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Target
# --------------------------------------------------------------------------- #
RAW_TARGET = "readmitted"          # values: 'NO', '>30', '<30'
TARGET = "readmitted_30d"          # engineered binary target: 1 if '<30'
POSITIVE_LABEL = "<30"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# --------------------------------------------------------------------------- #
# Data quality / cleaning constants
# --------------------------------------------------------------------------- #
MISSING_SENTINEL = "?"

# Identifiers - excluded from features (leakage / no signal).
ID_COLUMNS = ["encounter_id", "patient_nbr"]

# >90% missing or zero-variance - dropped (see data_quality.py report).
DROP_HIGH_MISSING = ["weight"]
DROP_ZERO_VARIANCE = [
    "nateglinide", "chlorpropamide", "acetohexamide", "tolbutamide",
    "acarbose", "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]
# Administrative billing code - no clinical signal, high missingness.
DROP_ADMIN = ["payer_code"]

# Lab columns where NaN means "test not performed" (clinically meaningful).
NOT_MEASURED_COLUMNS = ["max_glu_serum", "A1Cresult"]

# Discharge dispositions meaning the patient died or went to hospice - these
# patients cannot be readmitted, so the encounters are removed (Strack et al.).
EXPIRED_HOSPICE_DISPOSITIONS = [11, 13, 14, 19, 20, 21]

# --------------------------------------------------------------------------- #
# Feature groups (after cleaning + engineering)
# --------------------------------------------------------------------------- #
NUMERIC_FEATURES = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
    # engineered
    "age_ordinal", "n_meds_changed", "n_active_meds", "total_prior_visits",
    "service_utilization",
]

CATEGORICAL_FEATURES = [
    "race", "gender", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "medical_specialty",
    "max_glu_serum", "A1Cresult",
    "metformin", "repaglinide", "glimepiride", "glipizide", "glyburide",
    "pioglitazone", "rosiglitazone", "insulin",
    "change", "diabetesMed",
    # engineered diagnosis groups
    "diag_1_group", "diag_2_group", "diag_3_group",
]

# Sensitive attributes for fairness analysis.
SENSITIVE_FEATURES = ["race", "gender", "age_group"]

# Drugs tracked for the "medication change" engineered feature.
MEDICATION_COLUMNS = [
    "metformin", "repaglinide", "glimepiride", "glipizide", "glyburide",
    "pioglitazone", "rosiglitazone", "insulin",
]

# --------------------------------------------------------------------------- #
# Risk scoring thresholds (probability -> tier)
# --------------------------------------------------------------------------- #
RISK_TIERS = [
    ("LOW", 0.00, 0.10),
    ("MODERATE", 0.10, 0.25),
    ("HIGH", 0.25, 0.50),
    ("VERY HIGH", 0.50, 1.01),
]


# --------------------------------------------------------------------------- #
# ICD-9 diagnosis -> disease category mapping
# --------------------------------------------------------------------------- #
def map_icd9_to_group(code) -> str:
    """
    Map an ICD-9 diagnosis code to one of nine clinical categories, following
    the grouping used in Strack et al. (2014), BioMed Research International.

    The diag_* columns contain ~700-900 distinct ICD-9 codes - far too many to
    one-hot encode. Collapsing them into clinically coherent groups is the key
    feature-engineering step that makes the diagnosis signal usable.
    """
    if code is None:
        return "Missing"
    code = str(code).strip()
    if code in ("", "nan", "None", "?"):
        return "Missing"

    # 'V' and 'E' codes are supplementary / external-cause classifications.
    if code.startswith(("V", "E")):
        return "Other"

    # Diabetes is the 250.xx family.
    if code.startswith("250"):
        return "Diabetes"

    try:
        num = float(code)
    except ValueError:
        return "Other"

    if 390 <= num <= 459 or num == 785:
        return "Circulatory"
    if 460 <= num <= 519 or num == 786:
        return "Respiratory"
    if 520 <= num <= 579 or num == 787:
        return "Digestive"
    if 800 <= num <= 999:
        return "Injury"
    if 710 <= num <= 739:
        return "Musculoskeletal"
    if 580 <= num <= 629 or num == 788:
        return "Genitourinary"
    if 140 <= num <= 239:
        return "Neoplasms"
    return "Other"


# Human-readable labels for admission type / discharge / source IDs are kept in
# the IDS_MAPPING.csv shipped with the dataset; for modelling we treat the IDs
# as categorical codes, which is sufficient and avoids brittle hand-mapping.
