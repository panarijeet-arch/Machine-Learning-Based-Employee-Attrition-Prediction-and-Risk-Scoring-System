import os
import sys

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import plotly.graph_objects as go
import streamlit as st

from explainability import reason_codes
from theme import COLORS, RISK_COLOR_MAP, hero, inject_global_css, risk_badge_html
from utils import get_scored_employees, get_trained_models

st.set_page_config(page_title="Employee Risk Profile", page_icon="\U0001F464", layout="wide")
inject_global_css()
hero("Employee Risk Profile", "Look up any individual employee's attrition risk and the factors driving it.")

scored = get_scored_employees()
train_out = get_trained_models()
pipeline = train_out["best_pipeline"]

employee_id = st.selectbox("Select Employee ID", scored["EmployeeID"].tolist())
emp_row = scored[scored["EmployeeID"] == employee_id].iloc[0]

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    st.metric("Attrition Probability", f"{emp_row['AttritionProbability']*100:.1f}%")
with c2:
    st.markdown("**Risk Category**")
    st.markdown(risk_badge_html(emp_row["RiskCategory"]), unsafe_allow_html=True)
with c3:
    st.markdown("**Role**")
    st.write(f"{emp_row['JobRole']} — {emp_row['Department']}")

st.divider()

col_left, col_right = st.columns([1, 1])
with col_left:
    st.markdown("#### Employee Details")
    detail_fields = [
        "Age", "Gender", "MaritalStatus", "MonthlyIncome", "JobLevel",
        "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
        "OverTime", "BusinessTravel", "WorkLifeBalance", "JobSatisfaction",
        "EnvironmentSatisfaction", "NumCompaniesWorked",
    ]
    for f in detail_fields:
        st.write(f"**{f}:** {emp_row[f]}")

with col_right:
    st.markdown("#### Top Contributing Factors")
    from feature_engineering import FEATURE_COLUMNS
    emp_X = scored[scored["EmployeeID"] == employee_id][FEATURE_COLUMNS]
    ref_X = train_out["X_train"]
    rc = reason_codes(pipeline, emp_X, ref_X, top_n=6)

    fig = go.Figure(go.Bar(
        x=rc["contribution_score"], y=rc["feature"], orientation="h",
        marker_color=[COLORS["high_risk"] if v > 0 else COLORS["low_risk"] for v in rc["contribution_score"]],
    ))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Contribution to risk (relative)", yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Positive bars push this employee's risk above the average employee; negative bars pull it "
        "below average. This is a transparent, rule-based explanation (deviation from the training "
        "population mean, weighted by global feature importance) — not a SHAP value, though it serves "
        "the same purpose of surfacing individual reason codes for HR review."
    )
