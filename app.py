"""Streamlit frontend: loan approval prediction with LIME local explanations.

Run with:  streamlit run app.py
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from credit_xai.data import (
    CATEGORIES,
    FEATURE_LABELS,
    FEATURES,
    MODEL_PATH,
    NUMERIC_FEATURES,
    encode,
    loan_percent_income,
)
from credit_xai.explain import APPROVED, build_explainer, explain

st.set_page_config(page_title="Explainable Loan Approval", page_icon="🏦", layout="wide")

INTENT_LABELS = {"DEBTCONSOLIDATION": "Debt consolidation", "HOMEIMPROVEMENT": "Home improvement"}


@st.cache_resource
def load_artifacts():
    bundle = joblib.load(MODEL_PATH)
    return bundle, build_explainer(bundle["background"])


if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python -m credit_xai.train` first.")
    st.stop()

bundle, explainer = load_artifacts()
model, threshold, metrics = bundle["model"], bundle["threshold"], bundle["metrics"]

st.title("Explainable Credit Risk Assessment")
st.caption("XGBoost loan approval prediction with LIME local explanations")

with st.sidebar:
    st.header("Model performance")
    st.caption(f"Held-out test set, {metrics['test_rows']:,} applications")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    c2.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
    c1.metric("Precision", f"{metrics['precision']:.1%}")
    c2.metric("Recall", f"{metrics['recall']:.1%}")

    st.header("LIME settings")
    num_features = st.slider("Features in explanation", 3, len(FEATURES), 10)
    num_samples = st.select_slider(
        "Perturbation samples", options=[1000, 2500, 5000, 10000], value=5000
    )

with st.form("application_form"):
    applicant_col, loan_col, credit_col = st.columns(3)
    with applicant_col:
        st.subheader("Applicant")
        applicant = {
            "person_age": st.number_input("Age", 18, 100, 26),
            "person_gender": st.selectbox(
                "Gender", CATEGORIES["person_gender"], index=1, format_func=str.title
            ),
            "person_education": st.selectbox(
                "Education", CATEGORIES["person_education"], index=2
            ),
            "person_income": st.number_input(
                "Annual income", 8_000, 10_000_000, 67_000, step=1_000
            ),
            "person_emp_exp": st.number_input("Employment experience (years)", 0, 80, 4),
            "person_home_ownership": st.selectbox(
                "Home ownership", CATEGORIES["person_home_ownership"], format_func=str.title
            ),
        }
    with loan_col:
        st.subheader("Loan")
        applicant |= {
            "loan_amnt": st.number_input("Loan amount", 500, 35_000, 8_000, step=500),
            "loan_intent": st.selectbox(
                "Loan intent",
                CATEGORIES["loan_intent"],
                format_func=lambda v: INTENT_LABELS.get(v, v.title()),
            ),
            "loan_int_rate": st.number_input("Interest rate (%)", 5.0, 20.0, 11.0, step=0.25),
        }
    with credit_col:
        st.subheader("Credit history")
        applicant |= {
            "cb_person_cred_hist_length": st.number_input(
                "Credit history length (years)", 2, 30, 4
            ),
            "credit_score": st.number_input("Credit score", 300, 850, 640),
            "previous_loan_defaults_on_file": st.radio(
                "Previous loan default on file", CATEGORIES["previous_loan_defaults_on_file"],
                horizontal=True,
            ),
        }
    submitted = st.form_submit_button("Assess application", type="primary")

if submitted:
    applicant["loan_percent_income"] = loan_percent_income(
        applicant["loan_amnt"], applicant["person_income"]
    )
    st.session_state.applicant = applicant

if "applicant" not in st.session_state:
    st.info("Fill in the application and press **Assess application**.")
    st.stop()

applicant = st.session_state.applicant
row = encode(pd.DataFrame([applicant]))
p_approved = float(model.predict_proba(row)[0, APPROVED])

st.divider()
decision_col, proba_col, ratio_col = st.columns([2, 1, 1])
if p_approved >= threshold:
    decision_col.success("## ✅ APPROVED")
else:
    decision_col.error("## ❌ REJECTED")
proba_col.metric("Approval probability", f"{p_approved:.1%}")
ratio_col.metric("Loan / income ratio", f"{applicant['loan_percent_income']:.2f}")

background = bundle["background"]
out_of_range = [
    FEATURE_LABELS[f]
    for i, f in enumerate(FEATURES)
    if f in NUMERIC_FEATURES and not background[:, i].min() <= applicant[f] <= background[:, i].max()
]
if out_of_range:
    st.warning(
        "Outside the range seen in training: " + ", ".join(out_of_range)
        + ". The prediction and explanation are less reliable here."
    )

st.subheader("Why this decision? Local explanation (LIME)")
with st.spinner("Generating LIME explanation..."):
    exp = explain(explainer, model, row.to_numpy()[0], num_features, num_samples)
st.caption(
    "LIME fits a simple linear model around this one applicant. Positive weights push "
    "toward **Approved**, negative weights toward **Rejected**. "
    f"Local surrogate fit R² = {exp.score:.2f}."
)

chart_tab, html_tab, table_tab = st.tabs(["LIME chart", "LIME interactive view", "Weights table"])
with chart_tab:
    fig = exp.as_pyplot_figure(label=APPROVED)
    fig.set_size_inches(9, 0.45 * num_features + 1.5)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
with html_tab:
    # LIME's own HTML/JS widget; a white body keeps it readable in dark theme.
    html = exp.as_html(labels=(APPROVED,)).replace("<body>", '<body style="background:#fff">', 1)
    st.iframe(html, height=max(300, 32 * num_features + 80))
with table_tab:
    weights = pd.DataFrame(exp.as_list(label=APPROVED), columns=["Condition", "Weight"])
    weights["Pushes toward"] = weights["Weight"].map(lambda w: "Approved" if w > 0 else "Rejected")
    st.dataframe(weights, hide_index=True)
