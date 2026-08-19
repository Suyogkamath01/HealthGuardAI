"""
HealthGuard AI - Clinical Recommendation Engine
===============================================

Translates a patient's risk tier and concrete risk factors into a prioritised,
actionable care plan. This is a transparent, rule-based layer (not a second
model) so every recommendation is traceable to a clinical reason - appropriate
for a decision-support tool that clinicians must be able to trust and audit.

The rules encode well-established readmission-reduction interventions
(transitional care, medication reconciliation, follow-up scheduling, etc.) and
are gated on features the model and SHAP analysis flag as important:
prior inpatient visits, emergency utilisation, number of medications, length of
stay, diabetes management and discharge disposition.

Usage:
    from recommendations import recommend
    plan = recommend(patient_dict, risk_tier="HIGH", risk_score=82)
"""

from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# Recommendation model
# --------------------------------------------------------------------------- #
@dataclass
class Recommendation:
    action: str
    rationale: str
    priority: str  # "Routine" | "Recommended" | "Urgent"


# Baseline bundle by tier - every patient gets tier-appropriate follow-up.
TIER_BASELINE = {
    "LOW": [
        ("Standard discharge instructions", "Low predicted risk.", "Routine"),
        ("Primary-care follow-up within 14 days",
         "Maintains continuity of care.", "Routine"),
    ],
    "MODERATE": [
        ("Follow-up appointment within 7 days",
         "Moderate risk benefits from earlier review.", "Recommended"),
        ("Telephone check-in within 72 hours",
         "Early detection of post-discharge problems.", "Recommended"),
    ],
    "HIGH": [
        ("Follow-up appointment within 48-72 hours",
         "High risk warrants rapid clinical review.", "Urgent"),
        ("Enrol in transitional care / care-coordination program",
         "Structured transitional care reduces readmissions.", "Urgent"),
        ("Medication reconciliation before discharge",
         "Prevents post-discharge medication errors.", "Recommended"),
    ],
    "VERY HIGH": [
        ("Follow-up appointment within 24-48 hours",
         "Very high risk requires immediate post-discharge oversight.", "Urgent"),
        ("Assign dedicated case manager / care coordinator",
         "Intensive coordination for the highest-risk patients.", "Urgent"),
        ("Consider home health visit or remote monitoring",
         "Bridges the gap between hospital and home.", "Urgent"),
        ("Medication reconciliation + pharmacist review",
         "Complex regimens drive avoidable readmissions.", "Urgent"),
    ],
}


def _g(patient: dict, key, default=0):
    v = patient.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def recommend(
    patient: dict, risk_tier: str, risk_score: Optional[float] = None
) -> list:
    """
    Build a prioritised list of Recommendation objects for one patient.

    `patient` is a dict of the engineered/raw features (e.g. number_inpatient,
    number_emergency, num_medications, time_in_hospital, diabetesMed, A1Cresult,
    discharge_disposition_id, n_active_meds).
    """
    recs = [
        Recommendation(a, r, p) for a, r, p in TIER_BASELINE.get(risk_tier, [])
    ]

    # --- Feature-driven, patient-specific rules -------------------------------
    if _g(patient, "number_inpatient") >= 1:
        recs.append(Recommendation(
            "Review prior inpatient admissions and root causes",
            f"{int(_g(patient,'number_inpatient'))} prior inpatient stay(s) - "
            "the strongest readmission predictor.", "Urgent"))

    if _g(patient, "number_emergency") >= 1:
        recs.append(Recommendation(
            "Connect to outpatient services to reduce ED reliance",
            f"{int(_g(patient,'number_emergency'))} prior emergency visit(s) "
            "indicate gaps in outpatient access.", "Recommended"))

    if _g(patient, "num_medications") >= 15 or _g(patient, "n_active_meds") >= 6:
        recs.append(Recommendation(
            "Pharmacist-led polypharmacy review",
            "High medication count raises adverse-event and adherence risk.",
            "Recommended"))

    if _g(patient, "time_in_hospital") >= 7:
        recs.append(Recommendation(
            "Assess functional status and discharge readiness",
            f"Long stay ({int(_g(patient,'time_in_hospital'))} days) signals "
            "clinical complexity.", "Recommended"))

    # Diabetes-specific management (this is a diabetic cohort).
    a1c = str(patient.get("A1Cresult", "")).strip()
    if a1c in (">7", ">8"):
        recs.append(Recommendation(
            "Optimise glycaemic control and diabetes education",
            f"Elevated A1C ({a1c}) indicates poorly controlled diabetes.",
            "Recommended"))
    if str(patient.get("diabetesMed", "")).strip() == "No":
        recs.append(Recommendation(
            "Reassess need for diabetes pharmacotherapy",
            "No diabetes medication recorded in a diabetic admission.",
            "Routine"))

    if _g(patient, "number_diagnoses") >= 9:
        recs.append(Recommendation(
            "Coordinate multidisciplinary care for comorbidities",
            "High diagnosis count reflects significant comorbidity burden.",
            "Recommended"))

    # De-duplicate while preserving order, then sort by priority.
    seen = set()
    unique = []
    for r in recs:
        if r.action not in seen:
            seen.add(r.action)
            unique.append(r)
    rank = {"Urgent": 0, "Recommended": 1, "Routine": 2}
    unique.sort(key=lambda r: rank.get(r.priority, 3))
    return unique


def format_plan(patient: dict, risk_tier: str, risk_score: float) -> str:
    """Human-readable care plan (used by CLI / report examples)."""
    recs = recommend(patient, risk_tier, risk_score)
    lines = [
        f"Risk Score: {risk_score:.0f} / 100",
        f"Risk Tier : {risk_tier}",
        "",
        "Recommended Care Plan:",
    ]
    for i, r in enumerate(recs, 1):
        lines.append(f"  {i}. [{r.priority}] {r.action}")
        lines.append(f"       reason: {r.rationale}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Demo with a synthetic high-risk patient.
    demo = {
        "number_inpatient": 3, "number_emergency": 2, "num_medications": 18,
        "n_active_meds": 7, "time_in_hospital": 9, "number_diagnoses": 9,
        "A1Cresult": ">8", "diabetesMed": "Yes",
    }
    print(format_plan(demo, risk_tier="VERY HIGH", risk_score=88))
