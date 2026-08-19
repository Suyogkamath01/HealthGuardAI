# HealthGuard AI — developer workflow
# Usage: make <target>   (uses the project venv if present)

PY := $(shell [ -x venv/bin/python ] && echo venv/bin/python || echo python)

.PHONY: help install audit preprocess train explain score fairness \
        stats tune clinical mitigate pipeline analysis test dashboard clean

help:
	@echo "HealthGuard AI targets:"
	@echo "  install     Install dependencies into the active environment"
	@echo "  pipeline    Core pipeline: audit -> preprocess -> train -> explain -> score -> fairness"
	@echo "  analysis    Advanced analyses: stats + clinical utility + fairness mitigation"
	@echo "  tune        Optuna hyperparameter optimization"
	@echo "  test        Run the pytest suite"
	@echo "  dashboard   Launch the Streamlit dashboard"
	@echo "  clean       Remove generated artefacts (models, processed data, figures)"

install:
	$(PY) -m pip install -r requirements.txt

# ---- individual stages -----------------------------------------------------
audit:       ; $(PY) src/data_quality.py
preprocess:  ; $(PY) src/preprocessing.py
train:       ; $(PY) src/modeling.py
explain:     ; $(PY) src/explainability.py
score:       ; $(PY) src/risk_scoring.py
fairness:    ; $(PY) src/fairness.py
stats:       ; $(PY) src/statistical_analysis.py
tune:        ; $(PY) src/tuning.py
clinical:    ; $(PY) src/clinical_evaluation.py
mitigate:    ; $(PY) src/fairness_mitigation.py
research:    ; $(PY) src/research_leakage.py

# ---- composite -------------------------------------------------------------
pipeline:
	$(PY) src/run_pipeline.py

analysis: stats clinical mitigate
	@echo "Advanced analyses complete."

test:
	$(PY) -m pytest tests/ -q

dashboard:
	$(PY) -m streamlit run dashboard/app.py

clean:
	rm -rf models/*.pkl models/*.json data/processed/*.csv reports/figures/*.png
	@echo "Cleaned generated artefacts (raw data and source preserved)."
