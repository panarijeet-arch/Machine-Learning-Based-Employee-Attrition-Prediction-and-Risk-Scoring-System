"""
app.py -- Palo Alto Networks | Employee Attrition Risk Intelligence
=====================================================================
Home page. Trains the model (cached) on startup and shows a project
overview plus headline KPIs.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st

from theme import inject_global_css, hero, COLORS
from utils import get_clean_data, get_data_quality_report, get_scored_employees, get_trained_models

st.set_page_config(
    page_title="Palo Alto Networks | Attrition Risk Intelligence",
    page_icon="\U0001F6E1\uFE0F",
    layout="wide",
)
inject_global_css()

hero(
    "Employee Attrition Risk Intelligence",
    "Predictive workforce analytics for Palo Alto Networks HR — built on real employee records, "
    "an interpretable model, and transparent per-employee explanations.",
)

dq = get_data_quality_report()
df = get_clean_data()
train_out = get_trained_models()
scored = get_scored_employees()

st.subheader("Dataset & Model Snapshot")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Employees", f"{dq['n_rows']:,}")
c2.metric("Historical Attrition Rate", f"{dq['attrition_rate_pct']}%")
c3.metric("Best Model", train_out["best_model_name"])
c4.metric("Model ROC-AUC", f"{train_out['results'][train_out['best_model_name']]['test_metrics']['roc_auc']:.3f}")
high_risk_now = int((scored["RiskCategory"] == "High Risk").sum())
c5.metric("Currently High-Risk", f"{high_risk_now}")

st.divider()

st.subheader("Use the sidebar to explore:")
st.markdown(
    """
- **Attrition Risk Dashboard** — overall risk distribution, high-risk counts, model performance
- **Employee Risk Profile** — look up any employee's individual risk score and top contributing factors
- **Department Risk View** — aggregated risk by department and job role
- **Explainability Panel** — global feature importance and what-if scenario exploration
"""
)

st.divider()

st.info(
    f"**Data quality note:** {dq['n_rows']:,} employee records audited — "
    f"{dq['missing_values_total']} missing values, {dq['duplicate_rows']} duplicate rows found. "
    f"`PerformanceRating` was found to take only 2 distinct values ({dq['low_variance_columns'][0]['unique_values']}) "
    "across the entire dataset, limiting its usefulness as a predictive signal — this is disclosed "
    "here rather than silently included as if it carried more information than it does."
)

st.caption(
    "Model note: trained fresh at app startup on whatever scikit-learn version is installed "
    "(cached for the session), rather than loaded from a pre-saved file — this avoids version-"
    "mismatch issues between training and deployment environments."
)
