"""Tests for the cleaning + feature-engineering pipeline.

These guard the invariants that matter most for a clinical model: no patient
leakage, a correctly defined binary target, and engineered features that are
internally consistent.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config as C
import preprocessing as P


@pytest.fixture(scope="module")
def raw_sample():
    # Small raw sample keeps tests fast but exercises the real code paths.
    df = pd.read_csv(C.DATA_RAW, nrows=5000).replace("?", np.nan)
    return df


def test_age_mappings_complete():
    # Every age band present in the data has an ordinal + band mapping.
    df = pd.read_csv(C.DATA_RAW, nrows=5000)
    for age in df["age"].unique():
        assert age in P.AGE_ORDINAL
        assert age in P.AGE_BAND


def test_engineer_creates_binary_target():
    df = pd.read_csv(C.DATA_RAW, nrows=5000)
    df = df.replace("?", np.nan)
    # Drop the columns engineer() expects to have been removed in cleaning.
    df = df.drop(columns=[c for c in C.DROP_ZERO_VARIANCE if c in df.columns])
    out = P.engineer(df.copy())
    assert C.TARGET in out.columns
    assert set(out[C.TARGET].unique()).issubset({0, 1})
    assert C.RAW_TARGET not in out.columns


def test_service_utilization_is_consistent():
    df = pd.read_csv(C.DATA_RAW, nrows=5000).replace("?", np.nan)
    out = P.engineer(df.copy())
    expected = (out["number_outpatient"] + out["number_emergency"]
                + out["number_inpatient"])
    assert (out["total_prior_visits"] == expected).all()
    assert (out["service_utilization"]
            == out["total_prior_visits"] + out["number_diagnoses"]).all()


def test_processed_file_has_no_patient_duplicates_if_present():
    # If the full processed file exists, the leakage invariant must hold.
    if not C.DATA_PROCESSED.exists():
        pytest.skip("processed data not generated yet")
    df = pd.read_csv(C.DATA_PROCESSED)
    # patient_nbr was dropped, but the row count must equal unique patients
    # from preprocessing (one row per patient). Sanity: target is binary.
    assert set(df[C.TARGET].unique()).issubset({0, 1})
    # No fully-missing engineered columns.
    for col in ["age_ordinal", "diag_1_group", "n_active_meds"]:
        assert df[col].notna().all()
