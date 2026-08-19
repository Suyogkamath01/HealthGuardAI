"""
HealthGuard AI - One-command pipeline runner
=============================================

Runs the full offline pipeline in order:
    data_quality -> preprocessing -> modeling -> explainability
    -> risk_scoring -> fairness

After this completes, launch the dashboard with:
    streamlit run dashboard/app.py

Run:
    python src/run_pipeline.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_quality
import preprocessing
import modeling
import explainability
import risk_scoring
import fairness
import statistical_analysis
import clinical_evaluation
import fairness_mitigation

STEPS = [
    ("Data quality audit", data_quality.run),
    ("Preprocessing & feature engineering", preprocessing.run),
    ("Model training & evaluation", modeling.run),
    ("Explainability (SHAP)", explainability.run),
    ("Risk scoring & calibration", risk_scoring.run),
    ("Fairness analysis", fairness.run),
    ("Statistical rigour (CIs + DeLong)", statistical_analysis.run),
    ("Clinical utility (decision curve + calibration)", clinical_evaluation.run),
    ("Fairness mitigation (equal opportunity)", fairness_mitigation.run),
]

# Note: hyperparameter optimisation (src/tuning.py) is intentionally excluded
# from the default pipeline because it is slow; run it on demand with
# `python src/tuning.py` or `make tune`.


def main():
    t0 = time.time()
    for i, (name, fn) in enumerate(STEPS, 1):
        print("\n" + "#" * 72)
        print(f"# STEP {i}/{len(STEPS)}: {name}")
        print("#" * 72)
        fn()
    print(f"\nPipeline complete in {time.time() - t0:.0f}s.")
    print("Launch the dashboard:  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
