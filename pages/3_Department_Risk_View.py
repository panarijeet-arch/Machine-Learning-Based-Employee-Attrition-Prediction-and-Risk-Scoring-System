import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from theme import COLORS, PLOTLY_COLORWAY, hero, inject_global_css
from utils import department_role_summary, get_scored_employees

st.set_page_config(page_title="Department Risk View", page_icon="\U0001F3E2", layout="wide")
inject_global_css()
hero("Department-Level Risk View", "Aggregated attrition risk by department and job role.")

scored = get_scored_employees()

departments = sorted(scored["Department"].unique())
selected_depts = st.multiselect("Filter by Department", departments, default=departments)

filtered = scored[scored["Department"].isin(selected_depts)]
summary = department_role_summary(filtered)

st.markdown("#### Average Predicted Risk by Department")
dept_avg = filtered.groupby("Department")["AttritionProbability"].mean().sort_values(ascending=False) * 100
fig = go.Figure(go.Bar(
    x=dept_avg.index, y=dept_avg.values, marker_color=COLORS["navy"],
    text=[f"{v:.1f}%" for v in dept_avg.values], textposition="outside",
))
fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Avg. Predicted Attrition Risk (%)")
st.plotly_chart(fig, width="stretch")

st.divider()
st.markdown("#### Department × Job Role Breakdown")
st.dataframe(
    summary.rename(columns={
        "AvgRiskProbability": "Avg Risk %",
        "HighRiskCount": "High-Risk Count",
        "ActualAttritionRate": "Historical Attrition %",
    }),
    width="stretch", hide_index=True,
)

st.divider()
st.markdown("#### High-Risk Concentration (bubble = headcount)")
fig2 = go.Figure(go.Scatter(
    x=summary["JobRole"], y=summary["AvgRiskProbability"],
    mode="markers",
    marker=dict(
        size=summary["Headcount"] * 3, sizemode="area",
        color=summary["AvgRiskProbability"], colorscale=[[0, COLORS["low_risk"]], [0.5, COLORS["medium_risk"]], [1, COLORS["high_risk"]]],
        showscale=True,
    ),
    text=summary["Department"],
))
fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Avg Risk %", xaxis_tickangle=-30)
st.plotly_chart(fig2, width="stretch")
