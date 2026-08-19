"""
HealthGuard AI - Preprocessing & Feature Engineering
====================================================

Turns the raw 101,766-row encounter table into a clean, leakage-free, modelling
-ready dataset and writes it to data/processed/.

Pipeline (methodology follows Strack et al., 2014):
    1. Normalise the "?" sentinel to NaN.
    2. Remove encounters ending in death / hospice (cannot be readmitted).
    3. Keep only the FIRST encounter per patient (prevents patient-level leakage).
    4. Drop high-missing, zero-variance and administrative columns.
    5. Fix invalid categories (gender == 'Unknown/Invalid').
    6. Impute meaningful missingness with explicit categories.
    7. Engineer features:
         - binary 30-day readmission target
         - ICD-9 diagnoses -> 9 clinical groups
         - ordinal age + age band
         - service utilisation, prior-visit and medication-change counts
    8. Save the clean table + a human-readable feature dictionary.

Run:
    python src/preprocessing.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import config as C


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(C.DATA_RAW)
    df = df.replace(C.MISSING_SENTINEL, np.nan)
    n0 = len(df)

    # 2. Remove death / hospice discharges (outcome is undefined for them).
    df = df[~df["discharge_disposition_id"].isin(C.EXPIRED_HOSPICE_DISPOSITIONS)]
    n1 = len(df)

    # 3. First encounter per patient -> one row per patient, no leakage.
    df = df.sort_values("encounter_id").drop_duplicates(
        subset="patient_nbr", keep="first"
    )
    n2 = len(df)

    # 4. Drop columns flagged by the data-quality audit.
    drop_cols = (
        C.ID_COLUMNS
        + C.DROP_HIGH_MISSING
        + C.DROP_ZERO_VARIANCE
        + C.DROP_ADMIN
    )
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 5. Fix invalid gender values (only 3 rows) -> treat as missing then mode.
    df["gender"] = df["gender"].replace("Unknown/Invalid", np.nan)
    df["gender"] = df["gender"].fillna(df["gender"].mode().iloc[0])

    # 6. Explicit categories for meaningful missingness.
    for col in C.NOT_MEASURED_COLUMNS:          # lab not performed
        df[col] = df[col].fillna("Not measured")
    for col in ["race", "medical_specialty"]:
        df[col] = df[col].fillna("Unknown")

    print(
        f"  rows: {n0:,} -> remove expired/hospice -> {n1:,} "
        f"-> first encounter -> {n2:,}"
    )
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
AGE_ORDINAL = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
    "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95,
}
AGE_BAND = {
    "[0-10)": "0-30", "[10-20)": "0-30", "[20-30)": "0-30",
    "[30-40)": "30-60", "[40-50)": "30-60", "[50-60)": "30-60",
    "[60-70)": "60-80", "[70-80)": "60-80",
    "[80-90)": "80+", "[90-100)": "80+",
}


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    # Binary target: readmitted within 30 days.
    df[C.TARGET] = (df[C.RAW_TARGET] == C.POSITIVE_LABEL).astype(int)
    df = df.drop(columns=[C.RAW_TARGET])

    # Diagnoses -> clinical groups.
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[f"{col}_group"] = df[col].apply(C.map_icd9_to_group)
    df = df.drop(columns=["diag_1", "diag_2", "diag_3"])

    # Age -> ordinal midpoint + readable band.
    df["age_ordinal"] = df["age"].map(AGE_ORDINAL)
    df["age_group"] = df["age"].map(AGE_BAND)
    df = df.drop(columns=["age"])

    # Service utilisation = total touchpoints with the health system.
    df["total_prior_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )
    df["service_utilization"] = df["total_prior_visits"] + df["number_diagnoses"]

    # Medication dynamics: how many drugs were changed (Up/Down) and how many
    # are actively prescribed (Up/Down/Steady).
    changed = pd.DataFrame(index=df.index)
    active = pd.DataFrame(index=df.index)
    for med in C.MEDICATION_COLUMNS:
        changed[med] = df[med].isin(["Up", "Down"]).astype(int)
        active[med] = (df[med] != "No").astype(int)
    df["n_meds_changed"] = changed.sum(axis=1)
    df["n_active_meds"] = active.sum(axis=1)

    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Feature dictionary (for the report / dashboard)
# --------------------------------------------------------------------------- #
def write_feature_dictionary(df: pd.DataFrame) -> None:
    rows = []
    for col in df.columns:
        rows.append(
            {
                "feature": col,
                "dtype": str(df[col].dtype),
                "n_unique": int(df[col].nunique()),
                "example": df[col].dropna().iloc[0] if df[col].notna().any() else "",
            }
        )
    out = C.REPORTS_DIR / "feature_dictionary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  feature dictionary -> {out}")


def run() -> pd.DataFrame:
    print("Preprocessing HealthGuard AI dataset...")
    df = load_and_clean()
    df = engineer(df)

    # Sanity: every declared feature exists.
    expected = set(C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES + [C.TARGET])
    missing = expected - set(df.columns)
    assert not missing, f"Engineered frame missing columns: {missing}"

    df.to_csv(C.DATA_PROCESSED, index=False)
    write_feature_dictionary(df)

    rate = df[C.TARGET].mean()
    print(f"  final shape: {df.shape}")
    print(f"  30-day readmission rate: {rate:.3%}  (positive class)")
    print(f"  saved -> {C.DATA_PROCESSED}")
    return df


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    run()
