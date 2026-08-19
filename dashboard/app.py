"""
HealthGuard AI - Clinical Decision Support Dashboard
====================================================

Interactive Streamlit application with six pages:
    1. Overview            project summary + headline metrics
    2. Data Analytics      exploratory analysis of the patient cohort
    3. Risk Prediction     score an individual patient + care plan
    4. Explainability      global SHAP drivers + per-patient explanation
    5. Fairness Analysis   bias audit across race / gender / age
    6. Forecasting         population readmission burden + intervention what-if

Run:
    streamlit run dashboard/app.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make src importable.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config as C  # noqa: E402
from recommendations import recommend  # noqa: E402
from risk_scoring import score_to_tier, probability_to_score  # noqa: E402

st.set_page_config(
    page_title="HealthGuard AI", page_icon="🏥", layout="wide",
    initial_sidebar_state="expanded",
)

TIER_COLORS = {"LOW": "#2ca25f", "MODERATE": "#fec44f",
               "HIGH": "#fc8d59", "VERY HIGH": "#d73027"}


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(C.DATA_PROCESSED)


@st.cache_resource(show_spinner=False)
def load_model():
    path = C.MODELS_DIR / "calibrated_model.pkl"
    if not path.exists():
        path = C.MODELS_DIR / "best_model.pkl"
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_json(name):
    p = C.MODELS_DIR / name
    return json.load(open(p)) if p.exists() else {}


@st.cache_data(show_spinner=False)
def load_report(name):
    p = C.REPORTS_DIR / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def fig_path(name):
    p = C.FIGURES_DIR / name
    return str(p) if p.exists() else None


df = load_data()
meta = load_json("best_model_meta.json")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def cohort_defaults():
    """Default feature values (mode/median) for filling a prediction row."""
    defaults = {}
    for col in C.NUMERIC_FEATURES:
        defaults[col] = float(df[col].median())
    for col in C.CATEGORICAL_FEATURES:
        defaults[col] = str(df[col].mode().iloc[0])
    return defaults


def build_patient_row(user_inputs: dict) -> pd.DataFrame:
    """Merge user inputs with cohort defaults into a full single-row frame."""
    row = cohort_defaults()
    row.update(user_inputs)
    # Re-derive engineered features from inputs for consistency.
    row["total_prior_visits"] = (
        row["number_outpatient"] + row["number_emergency"] + row["number_inpatient"]
    )
    row["service_utilization"] = row["total_prior_visits"] + row["number_diagnoses"]
    X = pd.DataFrame([row])[C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES]
    for col in C.CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)
    return X


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("🏥 HealthGuard AI")
st.sidebar.caption("Explainable 30-day Readmission Risk")
PAGE = st.sidebar.radio(
    "Navigate",
    ["Overview", "Data Analytics", "Risk Prediction",
     "Explainability", "Fairness Analysis", "Forecasting"],
)
st.sidebar.markdown("---")
if meta:
    st.sidebar.metric("Best model", meta.get("best_model", "-"))
    st.sidebar.metric("ROC-AUC", f"{meta.get('roc_auc', 0):.3f}")
    st.sidebar.metric("PR-AUC", f"{meta.get('pr_auc', 0):.3f}")


# --------------------------------------------------------------------------- #
# Page: Overview
# --------------------------------------------------------------------------- #
if PAGE == "Overview":
    st.title("HealthGuard AI")
    st.subheader("Explainable Hospital Readmission Risk & Clinical Decision Support")
    st.markdown(
        "Predicts whether a diabetic patient will be **readmitted within 30 days "
        "of discharge**, explains *why*, audits the model for **fairness**, and "
        "generates **clinical recommendations** - built on 130 US hospitals "
        "(1999-2008)."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients (encounters)", f"{len(df):,}")
    c2.metric("30-day readmission rate", f"{df[C.TARGET].mean():.1%}")
    c3.metric("Features used", f"{len(C.NUMERIC_FEATURES)+len(C.CATEGORICAL_FEATURES)}")
    c4.metric("Models benchmarked", "5")

    st.markdown("### Model performance")
    comp = load_report("model_comparison.csv")
    if not comp.empty:
        show = comp[["model", "roc_auc", "pr_auc", "recall@best",
                     "precision@best", "f1@best"]].round(3)
        st.dataframe(show, width="stretch", hide_index=True)
        fig = px.bar(comp, x="model", y="pr_auc", color="model",
                     title="PR-AUC by model (higher = better on imbalanced data)")
        fig.add_hline(y=df[C.TARGET].mean(), line_dash="dash",
                      annotation_text="random baseline")
        st.plotly_chart(fig, width="stretch")

    st.markdown("### How it works")
    st.markdown(
        "1. **Data engineering** - leakage-safe cleaning (first encounter per "
        "patient, expired/hospice removed), ICD-9 → 9 clinical groups, "
        "utilisation & medication features.\n"
        "2. **Modelling** - 5 imbalance-aware classifiers, evaluated with "
        "PR-AUC / recall / F1, not accuracy.\n"
        "3. **Explainability** - SHAP global + per-patient attributions.\n"
        "4. **Risk scoring** - isotonic-calibrated 0-100 score & tiers.\n"
        "5. **Fairness** - demographic parity, equal opportunity, equalized odds.\n"
        "6. **Recommendations** - transparent, rule-based care plans."
    )


# --------------------------------------------------------------------------- #
# Page: Data Analytics
# --------------------------------------------------------------------------- #
elif PAGE == "Data Analytics":
    st.title("Cohort Analytics")
    rate = df[C.TARGET].mean()

    c1, c2 = st.columns(2)
    with c1:
        counts = df[C.TARGET].map({0: "Not readmitted <30d", 1: "Readmitted <30d"})
        vc = counts.value_counts().reset_index()
        vc.columns = ["label", "count"]
        fig = px.pie(vc, names="label", values="count",
                     title="Readmission class balance", hole=0.45)
        st.plotly_chart(fig, width="stretch")
    with c2:
        by_age = df.groupby("age_group")[C.TARGET].mean().reset_index()
        fig = px.bar(by_age, x="age_group", y=C.TARGET,
                     title="Readmission rate by age band",
                     labels={C.TARGET: "readmission rate"})
        fig.add_hline(y=rate, line_dash="dash", annotation_text="overall")
        st.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        by_race = df.groupby("race")[C.TARGET].agg(["mean", "count"]).reset_index()
        fig = px.bar(by_race, x="race", y="mean", title="Readmission rate by race",
                     labels={"mean": "readmission rate"})
        fig.add_hline(y=rate, line_dash="dash")
        st.plotly_chart(fig, width="stretch")
    with c4:
        diag = df["diag_1_group"].value_counts().reset_index()
        diag.columns = ["primary_diagnosis_group", "count"]
        fig = px.bar(diag, x="primary_diagnosis_group", y="count",
                     title="Primary diagnosis groups")
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Readmission rate vs. prior inpatient visits")
    tmp = df.copy()
    tmp["inpatient_bin"] = np.where(tmp["number_inpatient"] >= 3, "3+",
                                    tmp["number_inpatient"].astype(int).astype(str))
    g = tmp.groupby("inpatient_bin")[C.TARGET].mean().reset_index()
    fig = px.line(g, x="inpatient_bin", y=C.TARGET, markers=True,
                  labels={C.TARGET: "readmission rate",
                          "inpatient_bin": "prior inpatient visits"})
    st.plotly_chart(fig, width="stretch")

    with st.expander("Browse the processed data"):
        st.dataframe(df.head(200), width="stretch")


# --------------------------------------------------------------------------- #
# Page: Risk Prediction
# --------------------------------------------------------------------------- #
elif PAGE == "Risk Prediction":
    st.title("Patient Risk Prediction")
    st.caption("Enter a patient's discharge profile to estimate 30-day "
               "readmission risk and generate a care plan.")
    model = load_model()

    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age_ord = st.select_slider(
                "Age band (midpoint)",
                options=[5, 15, 25, 35, 45, 55, 65, 75, 85, 95], value=65)
            gender = st.selectbox("Gender", sorted(df["gender"].unique()))
            time_in_hospital = st.slider("Length of stay (days)", 1, 14, 4)
            num_medications = st.slider("Number of medications", 1, 60, 13)
        with c2:
            number_inpatient = st.slider("Prior inpatient visits", 0, 10, 0)
            number_emergency = st.slider("Prior emergency visits", 0, 10, 0)
            number_outpatient = st.slider("Prior outpatient visits", 0, 20, 0)
            number_diagnoses = st.slider("Number of diagnoses", 1, 16, 7)
        with c3:
            num_lab_procedures = st.slider("Lab procedures", 0, 120, 43)
            num_procedures = st.slider("Procedures", 0, 6, 1)
            a1c = st.selectbox("A1C result", ["Not measured", "Norm", ">7", ">8"])
            diabetes_med = st.selectbox("On diabetes medication?", ["Yes", "No"])
            diag1 = st.selectbox("Primary diagnosis group",
                                 sorted(df["diag_1_group"].unique()))
        submitted = st.form_submit_button("Assess risk", width="stretch")

    if submitted:
        inputs = {
            "age_ordinal": age_ord, "gender": gender,
            "time_in_hospital": time_in_hospital,
            "num_medications": num_medications,
            "number_inpatient": number_inpatient,
            "number_emergency": number_emergency,
            "number_outpatient": number_outpatient,
            "number_diagnoses": number_diagnoses,
            "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures,
            "A1Cresult": a1c, "diabetesMed": diabetes_med,
            "diag_1_group": diag1,
        }
        X = build_patient_row(inputs)
        proba = float(model.predict_proba(X)[:, 1][0])
        score = float(probability_to_score(proba))
        tier = score_to_tier(score)

        c1, c2 = st.columns([1, 1])
        with c1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=score,
                title={"text": f"Risk Score - {tier}"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": TIER_COLORS[tier]},
                       "steps": [
                           {"range": [0, 10], "color": "#e5f5e0"},
                           {"range": [10, 25], "color": "#fff7bc"},
                           {"range": [25, 50], "color": "#fee8c8"},
                           {"range": [50, 100], "color": "#fde0dd"}]}))
            gauge.update_layout(height=320)
            st.plotly_chart(gauge, width="stretch")
        with c2:
            st.metric("Estimated probability of 30-day readmission",
                      f"{proba:.1%}")
            st.markdown(f"### Risk tier: "
                        f"<span style='color:{TIER_COLORS[tier]}'>{tier}</span>",
                        unsafe_allow_html=True)
            st.caption("Score = isotonic-calibrated probability × 100.")

        st.markdown("### Recommended care plan")
        plan = recommend(inputs, risk_tier=tier, risk_score=score)
        for i, r in enumerate(plan, 1):
            badge = {"Urgent": "🔴", "Recommended": "🟠", "Routine": "🟢"}[r.priority]
            st.markdown(f"{badge} **{i}. {r.action}** - _{r.rationale}_")


# --------------------------------------------------------------------------- #
# Page: Explainability
# --------------------------------------------------------------------------- #
elif PAGE == "Explainability":
    st.title("Explainable AI")
    st.markdown("**Global drivers** of 30-day readmission risk (mean |SHAP|).")
    imp = load_report("shap_global_importance.csv")
    if not imp.empty:
        imp.columns = ["feature", "mean_abs_shap"]
        top = imp.head(20).sort_values("mean_abs_shap")
        fig = px.bar(top, x="mean_abs_shap", y="feature", orientation="h",
                     title="Top 20 features (SHAP)")
        fig.update_layout(height=600)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Run `python src/explainability.py` to generate SHAP values.")

    p = fig_path("shap_summary.png")
    if p:
        st.markdown("### SHAP summary (beeswarm)")
        st.image(p, width="stretch")

    # SHAP vs LIME cross-method check on an example patient.
    p2 = fig_path("shap_vs_lime.png")
    if p2:
        st.markdown("### Per-patient explanation: SHAP vs LIME")
        st.markdown(
            "We explain a single high-risk patient with **two independent "
            "methods** - SHAP (game-theoretic attributions) and LIME (a local "
            "linear surrogate). Where they **agree**, we can be more confident "
            "in the explanation; where they **diverge**, the explanation is "
            "method-sensitive and should be treated with caution."
        )
        st.image(p2, width="stretch")
        shap_ex = load_report("example_patient_shap.csv")
        lime_ex = load_report("example_patient_lime.csv")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.caption("SHAP top features")
            if not shap_ex.empty:
                st.dataframe(shap_ex, width="stretch", hide_index=True)
        with cc2:
            st.caption("LIME top features")
            if not lime_ex.empty:
                st.dataframe(lime_ex, width="stretch", hide_index=True)

    st.markdown(
        "Per-patient attributions are reusable via "
        "`explainability.explain_patient()` (SHAP) and "
        "`explainability.explain_patient_lime()` (LIME)."
    )


# --------------------------------------------------------------------------- #
# Page: Fairness Analysis
# --------------------------------------------------------------------------- #
elif PAGE == "Fairness Analysis":
    st.title("Fairness & Bias Audit")
    st.markdown(
        "Group-fairness criteria. The **4/5ths rule** flags a selection-rate "
        "ratio below 0.8 as potential adverse impact."
    )
    summary = load_report("fairness_report.csv")
    detail = load_report("fairness_detail.csv")
    if summary.empty:
        st.info("Run `python src/fairness.py` to generate the audit.")
    else:
        st.dataframe(summary, width="stretch", hide_index=True)
        for attr in summary["attribute"]:
            sub = detail[detail["attribute"] == attr]
            st.markdown(f"### {attr}")
            fig = px.bar(sub, x="group", y=["selection_rate", "tpr", "fpr"],
                         barmode="group",
                         title=f"Selection rate / TPR / FPR by {attr}")
            st.plotly_chart(fig, width="stretch")
        st.warning(
            "Disparities by **age** and **race** partly reflect real differences "
            "in base readmission rates, but exceed them - mitigation "
            "(threshold adjustment, reweighing, or post-processing) is discussed "
            "in the report."
        )


# --------------------------------------------------------------------------- #
# Page: Forecasting
# --------------------------------------------------------------------------- #
elif PAGE == "Forecasting":
    st.title("Population Readmission Burden & Intervention What-if")
    st.caption("Project the readmission burden across a cohort and simulate the "
               "impact of targeting high-risk patients with interventions.")
    model = load_model()

    n_cohort = st.slider("Cohort size (patients discharged / month)",
                         100, 5000, 1000, step=100)
    cost = st.number_input("Average cost per readmission ($)",
                           value=15000, step=1000)
    effectiveness = st.slider(
        "Intervention effectiveness (relative risk reduction on targeted patients)",
        0.0, 0.5, 0.20, step=0.05)
    target_tier = st.multiselect(
        "Target tiers for intervention", list(TIER_COLORS.keys()),
        default=["HIGH", "VERY HIGH"])

    # Score the whole processed cohort once, then scale to n_cohort.
    @st.cache_data(show_spinner=True)
    def scored_cohort():
        X = df[C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES].copy()
        for col in C.CATEGORICAL_FEATURES:
            X[col] = X[col].astype(str)
        proba = load_model().predict_proba(X)[:, 1]
        out = pd.DataFrame({"proba": proba})
        out["tier"] = [score_to_tier(s) for s in probability_to_score(proba)]
        return out

    sc = scored_cohort()
    scale = n_cohort / len(sc)

    tier_counts = (sc["tier"].value_counts() * scale).round().astype(int)
    expected_readmits = sc["proba"].sum() * scale

    targeted = sc[sc["tier"].isin(target_tier)]
    prevented = targeted["proba"].sum() * scale * effectiveness
    net_readmits = expected_readmits - prevented

    c1, c2, c3 = st.columns(3)
    c1.metric("Expected readmissions / month", f"{expected_readmits:,.0f}")
    c2.metric("Prevented by intervention", f"{prevented:,.0f}",
              delta=f"-{prevented/expected_readmits:.0%}")
    c3.metric("Avoided cost / month", f"${prevented*cost:,.0f}")

    order = list(TIER_COLORS.keys())
    td = tier_counts.reindex(order).fillna(0).reset_index()
    td.columns = ["tier", "patients"]
    fig = px.bar(td, x="tier", y="patients", color="tier",
                 color_discrete_map=TIER_COLORS,
                 title="Projected patients per risk tier")
    st.plotly_chart(fig, width="stretch")

    comp = pd.DataFrame({
        "scenario": ["No intervention", "With intervention"],
        "readmissions": [expected_readmits, net_readmits]})
    fig2 = px.bar(comp, x="scenario", y="readmissions", color="scenario",
                  title="Readmissions: baseline vs. targeted intervention")
    st.plotly_chart(fig2, width="stretch")
    st.caption("Illustrative model: assumes the stated relative risk reduction "
               "applies to summed predicted risk in the targeted tiers.")
