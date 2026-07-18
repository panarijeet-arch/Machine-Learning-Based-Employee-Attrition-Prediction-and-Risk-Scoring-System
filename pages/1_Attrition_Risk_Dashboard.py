import os
import sys

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from theme import COLORS, PLOTLY_COLORWAY, RISK_COLOR_MAP, hero, inject_global_css, rgba
from utils import get_scored_employees, get_trained_models

st.set_page_config(page_title="Attrition Risk Dashboard", page_icon="\U0001F4CA", layout="wide")
inject_global_css()
hero("Attrition Risk Dashboard", "Overall risk distribution and model performance across the current workforce.")

scored = get_scored_employees()
train_out = get_trained_models()
best_name = train_out["best_model_name"]
metrics = train_out["results"][best_name]["test_metrics"]
cv = train_out["results"][best_name]["cv_roc_auc"]

# --- KPIs ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Employees", f"{len(scored):,}")
c2.metric("High Risk", int((scored["RiskCategory"] == "High Risk").sum()))
c3.metric("Medium Risk", int((scored["RiskCategory"] == "Medium Risk").sum()))
c4.metric("Low Risk", int((scored["RiskCategory"] == "Low Risk").sum()))

st.divider()

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("#### Risk Category Distribution")
    counts = scored["RiskCategory"].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"]).fillna(0)
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=[RISK_COLOR_MAP[c] for c in counts.index],
        text=counts.values, textposition="outside",
    ))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Employees")
    st.plotly_chart(fig, width="stretch")

with col_right:
    st.markdown("#### Attrition Probability Distribution")
    fig2 = go.Figure(go.Histogram(
        x=scored["AttritionProbability"], nbinsx=30,
        marker_color=COLORS["navy"],
    ))
    fig2.add_vline(x=0.30, line_dash="dash", line_color=COLORS["medium_risk"])
    fig2.add_vline(x=0.60, line_dash="dash", line_color=COLORS["high_risk"])
    fig2.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="Predicted Attrition Probability", yaxis_title="Employees")
    st.plotly_chart(fig2, width="stretch")

st.divider()
st.markdown("#### Model Performance Comparison")
comp_rows = []
for name, r in train_out["results"].items():
    row = {"Model": name, **r["test_metrics"], "CV ROC-AUC (mean)": r["cv_roc_auc"]["mean"]}
    comp_rows.append(row)
comp_df = pd.DataFrame(comp_rows).sort_values("roc_auc", ascending=False)
st.dataframe(comp_df, width="stretch", hide_index=True)
st.caption(
    f"**{best_name}** was selected as the production model (highest test ROC-AUC = {metrics['roc_auc']}, "
    f"5-fold cross-validated ROC-AUC = {cv['mean']} ± {cv['std']})."
)

st.divider()
st.markdown("#### High-Risk Employees (Top 15 by probability)")
top15 = scored.sort_values("AttritionProbability", ascending=False).head(15)[
    ["EmployeeID", "Department", "JobRole", "AttritionProbability", "RiskCategory", "OverTime", "MonthlyIncome"]
]
st.dataframe(top15, width="stretch", hide_index=True)
