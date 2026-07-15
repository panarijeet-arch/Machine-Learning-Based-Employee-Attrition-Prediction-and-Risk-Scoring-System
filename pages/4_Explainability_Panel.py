import os
import sys

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from explainability import global_feature_importance, risk_category
from feature_engineering import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, engineer_features
from theme import COLORS, hero, inject_global_css, risk_badge_html
from utils import get_scored_employees, get_trained_models

st.set_page_config(page_title="Explainability Panel", page_icon="\U0001F50D", layout="wide")
inject_global_css()
hero("Explainability Panel", "Global feature importance, plus a live what-if scenario simulator.")

train_out = get_trained_models()
pipeline = train_out["best_pipeline"]
best_name = train_out["best_model_name"]
scored = get_scored_employees()

st.markdown(f"#### Global Feature Importance — {best_name}")
imp = global_feature_importance(pipeline, top_n=15)
fig = go.Figure(go.Bar(
    x=imp["importance_pct"], y=imp["feature"], orientation="h",
    marker_color=COLORS["navy"],
))
fig.update_layout(
    height=460, margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="Relative importance (%)", yaxis=dict(autorange="reversed"),
)
st.plotly_chart(fig, width="stretch")
st.caption(
    f"Importance is native to the {best_name} model — coefficients (absolute value) for Logistic "
    "Regression, or impurity-based importances for the tree ensembles. No approximation."
)

st.divider()
st.markdown("#### What-If Scenario Simulator")
st.write("Start from a real employee record, then adjust the sliders to see how predicted risk changes.")

base_id = st.selectbox("Base employee to start from", scored["EmployeeID"].tolist())
base_row = scored[scored["EmployeeID"] == base_id].iloc[0].copy()

c1, c2, c3 = st.columns(3)
with c1:
    overtime = st.selectbox("OverTime", ["Yes", "No"], index=0 if base_row["OverTime"] == "Yes" else 1)
    monthly_income = st.slider("Monthly Income ($)", 1000, 20000, int(base_row["MonthlyIncome"]), step=100)
    years_at_company = st.slider("Years At Company", 0, 40, int(base_row["YearsAtCompany"]))
with c2:
    work_life_balance = st.slider("Work-Life Balance (1=Poor, 4=Excellent)", 1, 4, int(base_row["WorkLifeBalance"]))
    job_satisfaction = st.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, int(base_row["JobSatisfaction"]))
    env_satisfaction = st.slider("Environment Satisfaction (1=Low, 4=High)", 1, 4, int(base_row["EnvironmentSatisfaction"]))
with c3:
    business_travel = st.selectbox(
        "Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
        index=["Non-Travel", "Travel_Rarely", "Travel_Frequently"].index(base_row["BusinessTravel"]),
    )
    years_since_promotion = st.slider("Years Since Last Promotion", 0, 15, int(base_row["YearsSinceLastPromotion"]))
    num_companies = st.slider("Num Companies Worked", 0, 9, int(base_row["NumCompaniesWorked"]))

scenario = base_row.copy()
scenario["OverTime"] = overtime
scenario["MonthlyIncome"] = monthly_income
scenario["YearsAtCompany"] = years_at_company
scenario["WorkLifeBalance"] = work_life_balance
scenario["JobSatisfaction"] = job_satisfaction
scenario["EnvironmentSatisfaction"] = env_satisfaction
scenario["BusinessTravel"] = business_travel
scenario["YearsSinceLastPromotion"] = years_since_promotion
scenario["NumCompaniesWorked"] = num_companies

scenario_df = pd.DataFrame([scenario])
scenario_engineered = engineer_features(scenario_df)
scenario_X = scenario_engineered[FEATURE_COLUMNS]

new_proba = pipeline.predict_proba(scenario_X)[0, 1]
new_category = risk_category(new_proba)
original_proba = base_row["AttritionProbability"]

st.divider()
r1, r2, r3 = st.columns(3)
r1.metric("Original Risk", f"{original_proba*100:.1f}%", help="Predicted probability with this employee's actual recorded values.")
r2.metric("Scenario Risk", f"{new_proba*100:.1f}%", delta=f"{(new_proba-original_proba)*100:+.1f} pts")
with r3:
    st.markdown("**Scenario Risk Category**")
    st.markdown(risk_badge_html(new_category), unsafe_allow_html=True)

st.caption(
    "This simulator re-runs the actual trained model on the modified feature values — it is a live "
    "prediction, not a lookup table."
)
