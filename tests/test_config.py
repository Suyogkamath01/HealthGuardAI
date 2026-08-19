"""Tests for configuration and domain mappings."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config as C


def test_icd9_diabetes_group():
    assert C.map_icd9_to_group("250") == "Diabetes"
    assert C.map_icd9_to_group("250.83") == "Diabetes"


def test_icd9_circulatory_group():
    assert C.map_icd9_to_group("410") == "Circulatory"   # MI
    assert C.map_icd9_to_group("428") == "Circulatory"   # heart failure
    assert C.map_icd9_to_group("785") == "Circulatory"


def test_icd9_respiratory_and_digestive():
    assert C.map_icd9_to_group("486") == "Respiratory"   # pneumonia
    assert C.map_icd9_to_group("550") == "Digestive"


def test_icd9_v_and_e_codes_are_other():
    assert C.map_icd9_to_group("V45") == "Other"
    assert C.map_icd9_to_group("E885") == "Other"


def test_icd9_missing_handling():
    assert C.map_icd9_to_group(None) == "Missing"
    assert C.map_icd9_to_group("?") == "Missing"
    assert C.map_icd9_to_group("nan") == "Missing"


def test_risk_tiers_are_contiguous_and_cover_unit_interval():
    tiers = C.RISK_TIERS
    assert tiers[0][1] == 0.0
    assert tiers[-1][2] >= 1.0
    # No gaps between consecutive tiers.
    for (_, _, hi), (_, lo, _) in zip(tiers, tiers[1:]):
        assert abs(hi - lo) < 1e-9


def test_feature_lists_are_disjoint():
    num = set(C.NUMERIC_FEATURES)
    cat = set(C.CATEGORICAL_FEATURES)
    assert num.isdisjoint(cat)
    assert C.TARGET not in num and C.TARGET not in cat
