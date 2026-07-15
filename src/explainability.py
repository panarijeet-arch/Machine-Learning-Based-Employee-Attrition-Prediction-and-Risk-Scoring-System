"""
explainability.py
==================
Model explainability without SHAP. SHAP is listed as optional in the
project spec; it is deliberately left out here in favor of two
lightweight, fully-transparent alternatives that carry zero extra
dependency risk (the project's dependency list is kept to
pandas/numpy/scikit-learn/streamlit/plotly only, all of which are
directly testable):

1. Global feature importance -- native to each model type
   (coefficients for Logistic Regression, impurity-based importances
   for the tree ensembles). No approximation involved.
2. Per-employee reason codes -- a transparent, rule-based explanation:
   for each feature, compare the employee's (scaled) value against the
   training population mean, weight that deviation by the feature's
   global importance, and surface the top contributors. This is
   explicitly documented as a simplified stand-in for SHAP, not
   represented as SHAP itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from feature_engineering import ALL_NUMERIC_COLUMNS, CATEGORICAL_COLUMNS


def get_feature_names(pipeline) -> list[str]:
    preprocess = pipeline.named_steps["preprocess"]
    cat_encoder = preprocess.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_COLUMNS))
    return list(ALL_NUMERIC_COLUMNS) + cat_names


def global_feature_importance(pipeline, top_n: int = 15) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    names = get_feature_names(pipeline)

    if hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        raise ValueError("Model does not expose an importance/coefficient attribute.")

    df = pd.DataFrame({"feature": names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    df["importance_pct"] = (df["importance"] / df["importance"].sum() * 100).round(1)
    return df


def risk_category(probability: float) -> str:
    if probability < 0.30:
        return "Low Risk"
    elif probability < 0.60:
        return "Medium Risk"
    else:
        return "High Risk"


def reason_codes(pipeline, employee_row: pd.DataFrame, X_reference: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Transparent, rule-based local explanation (documented stand-in for
    SHAP -- see module docstring). Compares the employee's transformed
    feature values against the training-set mean, scaled by global
    importance, and returns the features pushing risk up the most."""
    preprocess = pipeline.named_steps["preprocess"]
    names = get_feature_names(pipeline)

    X_emp_transformed = preprocess.transform(employee_row)
    X_ref_transformed = preprocess.transform(X_reference)
    if hasattr(X_emp_transformed, "toarray"):
        X_emp_transformed = X_emp_transformed.toarray()
        X_ref_transformed = X_ref_transformed.toarray()

    ref_mean = X_ref_transformed.mean(axis=0)
    deviation = X_emp_transformed[0] - ref_mean

    imp_df = global_feature_importance(pipeline, top_n=len(names))
    imp_map = dict(zip(imp_df["feature"], imp_df["importance"]))
    weights = np.array([imp_map.get(n, 0.0) for n in names])

    signed_contribution = deviation * weights
    result = pd.DataFrame({
        "feature": names,
        "deviation_from_average": deviation.round(3),
        "contribution_score": signed_contribution.round(4),
    })
    result = result.sort_values("contribution_score", ascending=False).head(top_n).reset_index(drop=True)
    return result
