"""Tests for the risk-scoring tiers and the clinical recommendation rules."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config as C
import risk_scoring as RS
from recommendations import recommend


def test_score_to_tier_boundaries():
    assert RS.score_to_tier(5) == "LOW"
    assert RS.score_to_tier(15) == "MODERATE"
    assert RS.score_to_tier(30) == "HIGH"
    assert RS.score_to_tier(80) == "VERY HIGH"


def test_score_to_tier_monotonic():
    tiers = [RS.score_to_tier(s) for s in range(0, 101, 5)]
    order = {t[0]: i for i, t in enumerate(C.RISK_TIERS)}
    ranks = [order[t] for t in tiers]
    assert ranks == sorted(ranks)  # never decreases as score increases


def test_probability_to_score_range():
    s = RS.probability_to_score(np.array([0.0, 0.5, 1.0]))
    assert s.min() >= 0 and s.max() <= 100


def test_recommendations_nonempty_and_ranked():
    patient = {"number_inpatient": 3, "num_medications": 18,
               "time_in_hospital": 9, "A1Cresult": ">8", "number_diagnoses": 9}
    recs = recommend(patient, risk_tier="VERY HIGH", risk_score=88)
    assert len(recs) > 0
    # Priorities must be sorted Urgent -> Recommended -> Routine.
    rank = {"Urgent": 0, "Recommended": 1, "Routine": 2}
    ranks = [rank[r.priority] for r in recs]
    assert ranks == sorted(ranks)


def test_high_inpatient_triggers_specific_rule():
    patient = {"number_inpatient": 4}
    actions = [r.action for r in recommend(patient, "HIGH", 40)]
    assert any("prior inpatient" in a.lower() for a in actions)


def test_recommendations_are_deduplicated():
    patient = {"number_inpatient": 2, "num_medications": 20}
    recs = recommend(patient, "VERY HIGH", 90)
    actions = [r.action for r in recs]
    assert len(actions) == len(set(actions))
