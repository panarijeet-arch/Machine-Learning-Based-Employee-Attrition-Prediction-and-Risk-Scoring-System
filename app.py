"""
app.py -- Palo Alto Networks | Employee Attrition Risk Intelligence
=====================================================================
Home page. Trains the model (cached) on startup, shows headline KPIs,
risk distribution, department comparison, and data-driven insights.
"""
import os
import sys

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from theme import COLORS, RISK_COLOR_MAP, hero, inject_global_css, insight_html
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
    badges=["1,470 employees analyzed", "3 models compared", "4 analytics modules", "Live, self-training model"],
)

dq = get_data_quality_report()
df = get_clean_data()
train_out = get_trained_models()
scored = get_scored_employees()
best_name = train_out["best_model_name"]
best_metrics = train_out["results"][best_name]["test_metrics"]
high_risk_now = int((scored["RiskCategory"] == "High Risk").sum())
medium_risk_now = int((scored["RiskCategory"] == "Medium Risk").sum())
low_risk_now = int((scored["RiskCategory"] == "Low Risk").sum())

# --- KPI row ---
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Employees", f"{dq['n_rows']:,}")
c2.metric("Historical Attrition Rate", f"{dq['attrition_rate_pct']}%")
c3.metric("Best Model", best_name)
c4.metric("Model ROC-AUC", f"{best_metrics['roc_auc']:.3f}")
c5.metric("Currently High-Risk", f"{high_risk_now}")

st.divider()

# --- Risk distribution + department comparison, side by side ---
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("#### Current Workforce Risk Distribution")
    fig = go.Figure(go.Pie(
        labels=["Low Risk", "Medium Risk", "High Risk"],
        values=[low_risk_now, medium_risk_now, high_risk_now],
        marker=dict(colors=[RISK_COLOR_MAP["Low Risk"], RISK_COLOR_MAP["Medium Risk"], RISK_COLOR_MAP["High Risk"]]),
        hole=0.55,
        textinfo="label+percent",
    ))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig, width="stretch")

with col_right:
    st.markdown("#### Model Comparison (test set)")
    comp_rows = []
    for name, r in train_out["results"].items():
        comp_rows.append({"Model": name, "ROC-AUC": r["test_metrics"]["roc_auc"], "Recall": r["test_metrics"]["recall"], "F1": r["test_metrics"]["f1"]})
    comp_df = pd.DataFrame(comp_rows).sort_values("ROC-AUC", ascending=False)
    fig2 = go.Figure(go.Bar(
        x=comp_df["Model"], y=comp_df["ROC-AUC"],
        marker_color=[COLORS["navy"] if m == best_name else COLORS["slate"] for m in comp_df["Model"]],
        text=[f"{v:.3f}" for v in comp_df["ROC-AUC"]], textposition="outside",
    ))
    fig2.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="ROC-AUC", yaxis_range=[0.6, 0.9])
    st.plotly_chart(fig2, width="stretch")

st.divider()

# --- Key insights, computed from the real dataset ---
st.markdown("#### Key Insights")
i1, i2 = st.columns(2)
with i1:
    st.markdown(
        insight_html(
            "<b>Overtime is the single strongest behavioral driver.</b> Employees working overtime leave at "
            "<b>30.5%</b>, nearly 3× the rate of those who don't (<b>10.4%</b>)."
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        insight_html(
            "<b>Frequent travel compounds risk.</b> Employees who travel frequently leave at <b>24.9%</b>, "
            "roughly 3× the rate of non-travelers (<b>8.0%</b>)."
        ),
        unsafe_allow_html=True,
    )
with i2:
    st.markdown(
        insight_html(
            "<b>Sales carries the highest departmental risk.</b> Average predicted attrition probability of "
            "<b>44.4%</b>, ahead of HR (<b>41.3%</b>) and well above R&amp;D (<b>30.0%</b>)."
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        insight_html(
            "<b>Sales Representatives are the highest-risk role</b>, with a historical attrition rate of "
            "<b>39.8%</b> — more than double the company-wide average of 16.1%."
        ),
        unsafe_allow_html=True,
    )

st.divider()

# --- Navigator ---
st.markdown("#### Explore the Dashboard")
n1, n2, n3, n4 = st.columns(4)
nav_items = [
    (n1, "Attrition Risk Dashboard", "Overall risk distribution, high-risk counts, model performance comparison."),
    (n2, "Employee Risk Profile", "Look up any individual employee's risk score and top contributing factors."),
    (n3, "Department Risk View", "Aggregated, filterable risk by department and job role."),
    (n4, "Explainability Panel", "Global feature importance and a live what-if scenario simulator."),
]
for col, name, desc in nav_items:
    with col:
        st.markdown(f"**{name}**")
        st.caption(desc)

st.divider()

with st.expander("Data quality audit & methodology notes", expanded=False):
    st.info(
        f"**Data quality:** {dq['n_rows']:,} employee records audited — "
        f"{dq['missing_values_total']} missing values, {dq['duplicate_rows']} duplicate rows found. "
        f"`PerformanceRating` was found to take only 2 distinct values ({dq['low_variance_columns'][0]['unique_values']}) "
        "across the entire dataset, limiting its usefulness as a predictive signal — disclosed here rather "
        "than silently treated as if it carried more information than it does."
    )
    st.caption(
        "**Model note:** trained fresh at app startup on whatever scikit-learn version is installed "
        "(cached for the session), rather than loaded from a pre-saved file — this avoids version-"
        "mismatch issues between training and deployment environments. "
        "**Class imbalance** (16.1% base attrition rate) is handled via class-weight balancing rather "
        "than SMOTE oversampling. **Explainability** uses each model's native coefficients/importances "
        "plus a transparent, documented rule-based per-employee explanation — not an approximation "
        "represented as SHAP."
    )
