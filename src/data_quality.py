"""
HealthGuard AI - Data Quality Audit
===================================

First preprocessing stage. This module inspects the raw diabetes dataset and
produces a column-by-column data quality report that drives later cleaning,
feature engineering and modelling decisions.

What it does:
    1. Loads the raw data and normalises the "?" sentinel to NaN.
    2. Measures missingness (count + percentage) per column.
    3. Recommends a handling strategy per column:
         - DROP_HIGH_MISSING : too sparse to be useful (e.g. weight ~97%)
         - DROP_ZERO_VARIANCE: constant column, no predictive signal
         - DROP_ID           : identifier / leakage risk
         - CATEGORY_NOT_MEASURED: NaN is meaningful ("test not performed")
         - IMPUTE_CATEGORY   : fill with an explicit "Unknown" level
         - OK                : usable as-is
    4. Flags duplicate patient encounters (a leakage concern for modelling).
    5. Flags invalid category values (e.g. gender == "Unknown/Invalid").
    6. Saves the report to reports/ for the write-up and downstream steps.

Run:
    python src/data_quality.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RAW_PATH = Path("data/raw/diabetic_data.csv")
REPORTS_DIR = Path("reports")

# A missing-percentage above this is too sparse to reliably impute -> drop.
HIGH_MISSING_THRESHOLD = 90.0

# A single category covering more than this share carries almost no signal.
ZERO_VARIANCE_THRESHOLD = 0.99

# The "?" sentinel used throughout this dataset for unknown values.
MISSING_SENTINEL = "?"

# Identifier columns: unique per row or leak patient identity across encounters.
ID_COLUMNS = ["encounter_id", "patient_nbr"]

# Lab columns where a missing value means "the test was not performed" rather
# than "data lost". These are clinically informative and must be kept as a
# distinct category, NOT dropped despite their high missing rate.
NOT_MEASURED_COLUMNS = ["max_glu_serum", "A1Cresult"]

# Known invalid / placeholder category values to flag (column -> bad values).
INVALID_VALUES = {"gender": ["Unknown/Invalid"]}


# --------------------------------------------------------------------------- #
# Core steps
# --------------------------------------------------------------------------- #
def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV and normalise the '?' sentinel to NaN."""
    df = pd.read_csv(path)
    df = df.replace(MISSING_SENTINEL, np.nan)
    return df


def missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column missing count and percentage, sorted descending."""
    report = pd.DataFrame(
        {
            "missing_count": df.isnull().sum(),
            "missing_pct": df.isnull().mean().mul(100).round(2),
        }
    )
    return report.sort_values("missing_pct", ascending=False)


def dominant_share(series: pd.Series) -> float:
    """Share of the single most common value (NaN included)."""
    return series.value_counts(normalize=True, dropna=False).iloc[0]


def recommend_strategy(df: pd.DataFrame, miss: pd.DataFrame) -> pd.DataFrame:
    """Attach a handling recommendation to each column."""
    rows = []
    for col in df.columns:
        pct = float(miss.loc[col, "missing_pct"])
        top_share = dominant_share(df[col])

        if col in ID_COLUMNS:
            strategy = "DROP_ID"
            note = "Identifier / leakage risk - exclude from features."
        elif col in NOT_MEASURED_COLUMNS:
            strategy = "CATEGORY_NOT_MEASURED"
            note = "NaN = test not performed; fill with 'Not measured'."
        elif top_share >= ZERO_VARIANCE_THRESHOLD:
            strategy = "DROP_ZERO_VARIANCE"
            note = f"One value covers {top_share:.2%} of rows - no signal."
        elif pct >= HIGH_MISSING_THRESHOLD:
            strategy = "DROP_HIGH_MISSING"
            note = f"{pct:.1f}% missing - too sparse to impute."
        elif pct > 0:
            strategy = "IMPUTE_CATEGORY"
            note = "Fill missing with an explicit 'Unknown' category."
        else:
            strategy = "OK"
            note = "Complete - usable as-is."

        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "n_unique": int(df[col].nunique(dropna=True)),
                "missing_pct": round(pct, 2),
                "dominant_share": round(top_share, 4),
                "strategy": strategy,
                "note": note,
            }
        )
    return pd.DataFrame(rows).set_index("column")


def find_invalid_values(df: pd.DataFrame) -> dict:
    """Count occurrences of known invalid category values."""
    found = {}
    for col, bad_values in INVALID_VALUES.items():
        if col not in df.columns:
            continue
        for val in bad_values:
            count = int((df[col] == val).sum())
            if count:
                found[f"{col} == '{val}'"] = count
    return found


def duplicate_summary(df: pd.DataFrame) -> dict:
    """Summarise duplicate encounters/patients (a modelling leakage concern)."""
    summary = {}
    if "encounter_id" in df.columns:
        summary["duplicate_encounter_id"] = int(df["encounter_id"].duplicated().sum())
    if "patient_nbr" in df.columns:
        dup_rows = int(df["patient_nbr"].duplicated().sum())
        summary["patients_with_repeat_encounters"] = dup_rows
        summary["unique_patients"] = int(df["patient_nbr"].nunique())
        summary["total_encounters"] = len(df)
    return summary


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run() -> pd.DataFrame:
    """Run the full audit, print a report, and save it to reports/."""
    df = load_raw()
    miss = missingness(df)
    report = recommend_strategy(df, miss)

    _section("DATASET SHAPE")
    print(f"{df.shape[0]:,} rows  x  {df.shape[1]} columns")

    _section("COLUMNS WITH MISSING VALUES")
    has_missing = miss[miss["missing_count"] > 0]
    print(has_missing.to_string() if len(has_missing) else "None")

    _section("RECOMMENDED HANDLING (grouped by strategy)")
    for strategy, group in report.groupby("strategy"):
        cols = ", ".join(group.index)
        print(f"\n[{strategy}]  ({len(group)} columns)")
        print(f"  {cols}")
        print(f"  -> {group['note'].iloc[0]}")

    _section("DUPLICATE / LEAKAGE CHECK")
    for key, value in duplicate_summary(df).items():
        print(f"  {key:32s}: {value:,}")
    print(
        "\n  Note: ~30k encounters are repeat visits by the same patient.\n"
        "  Consider keeping only the first encounter per patient_nbr, or use\n"
        "  patient-grouped splits, to avoid train/test leakage."
    )

    _section("INVALID CATEGORY VALUES")
    invalid = find_invalid_values(df)
    if invalid:
        for key, count in invalid.items():
            print(f"  {key:32s}: {count:,}")
    else:
        print("  None found.")

    _section("COLUMNS TO DROP")
    to_drop = report[report["strategy"].str.startswith("DROP")]
    print(f"  {len(to_drop)} columns flagged for dropping:")
    print(f"  {', '.join(to_drop.index)}")

    # Persist for the write-up and downstream steps.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "data_quality_report.csv"
    report.to_csv(out_path)
    print(f"\nSaved full report -> {out_path}")

    return report


if __name__ == "__main__":
    run()
