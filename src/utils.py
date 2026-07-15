"""
utils.py
========
Shared, cached data and model loading for every page of the app, plus
department/role-level risk aggregation.

Design note: the model is trained fresh at app startup (st.cache_resource)
rather than loaded from a pickled file. This was a deliberate choice
based on a prior project's deployment issues -- shipping a pre-trained
pickle creates a hard dependency on the exact scikit-learn version used
to create it, and a version mismatch on the hosting platform produces
silent warnings at best and broken predictions at worst. Training on
every cold start costs a few seconds once (cached for the rest of the
session) and completely removes that failure mode: whatever scikit-learn
version Streamlit Cloud installs is the same one used to train.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from preprocessing import clean_data, load_raw, run_data_quality_report
from model_training import train_all_models
from feature_engineering import engineer_features
from explainability import risk_category


@st.cache_data(show_spinner=False)
def get_clean_data() -> pd.DataFrame:
    return clean_data(load_raw())


@st.cache_data(show_spinner=False)
def get_data_quality_report() -> dict:
    return run_data_quality_report(load_raw())


@st.cache_resource(show_spinner="Training models on startup (one-time, ~10 seconds)...")
def get_trained_models():
    df = get_clean_data()
    return train_all_models(df)


@st.cache_data(show_spinner=False)
def get_scored_employees() -> pd.DataFrame:
    """Full employee table with predicted attrition probability and risk
    category attached, using the best-performing trained model."""
    train_out = get_trained_models()
    pipeline = train_out["best_pipeline"]

    df = get_clean_data()
    engineered = engineer_features(df)
    from feature_engineering import FEATURE_COLUMNS
    X = engineered[FEATURE_COLUMNS]

    proba = pipeline.predict_proba(X)[:, 1]
    engineered["AttritionProbability"] = proba.round(4)
    engineered["RiskCategory"] = engineered["AttritionProbability"].apply(risk_category)
    engineered["EmployeeID"] = engineered.index + 1
    return engineered


def department_role_summary(scored_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        scored_df.groupby(["Department", "JobRole"])
        .agg(
            Headcount=("EmployeeID", "count"),
            AvgRiskProbability=("AttritionProbability", "mean"),
            HighRiskCount=("RiskCategory", lambda s: (s == "High Risk").sum()),
            ActualAttritionRate=("Attrition", "mean"),
        )
        .reset_index()
    )
    summary["AvgRiskProbability"] = (summary["AvgRiskProbability"] * 100).round(1)
    summary["ActualAttritionRate"] = (summary["ActualAttritionRate"] * 100).round(1)
    return summary.sort_values("AvgRiskProbability", ascending=False)
