"""
model_training.py
==================
Trains and evaluates the three models called for in the project spec:
Logistic Regression (interpretable baseline), Random Forest, and
Gradient Boosting. Uses scikit-learn's own GradientBoostingClassifier
rather than the external XGBoost package -- this keeps every model in
one well-tested library with no extra compiled dependency, which
matters for deployment reliability (a lesson learned from a prior
project's deployment troubleshooting).

Class imbalance (base attrition rate ~16%) is handled via class-weight
balancing rather than SMOTE oversampling -- the project spec explicitly
allows either approach, and class weighting avoids adding another
dependency (imbalanced-learn) whose interaction with a fixed cross-
validation split is easy to get subtly wrong (oversampling before the
train/test split, causing leakage, is a common mistake this avoids by
construction).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from feature_engineering import (
    ALL_NUMERIC_COLUMNS, CATEGORICAL_COLUMNS, get_feature_target,
)

RANDOM_STATE = 42


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ALL_NUMERIC_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )


def build_candidate_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate(pipeline: Pipeline, X_test, y_test, sample_weight=None) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
    }


def cross_validate_roc_auc(pipeline: Pipeline, X, y, n_splits=5) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        pipe_clone = Pipeline(pipeline.steps)
        model_step = pipe_clone.steps[-1][1]
        if hasattr(model_step, "class_weight") and model_step.class_weight == "balanced":
            pipe_clone.fit(X_tr, y_tr)
        else:
            sw = compute_sample_weight("balanced", y_tr)
            pipe_clone.fit(X_tr, y_tr, model__sample_weight=sw)
        proba = pipe_clone.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, proba))
    return {
        "mean": round(float(np.mean(scores)), 4),
        "std": round(float(np.std(scores)), 4),
        "folds": [round(float(s), 4) for s in scores],
    }


def train_all_models(df: pd.DataFrame) -> dict:
    """Trains all three candidate models on the given dataframe and
    returns fitted pipelines plus evaluation metrics. Designed to be
    called fresh at app startup (wrapped in st.cache_resource by the
    Streamlit app) rather than unpickled from a saved file -- this
    sidesteps scikit-learn version-compatibility issues between the
    training environment and the deployment environment entirely, since
    the model is always trained with whatever sklearn version is
    actually installed at runtime."""
    X, y = get_feature_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    results = {}
    for name, model in build_candidate_models().items():
        pipeline = Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", model),
        ])

        if hasattr(model, "class_weight"):
            pipeline.fit(X_train, y_train)
        else:
            sw = compute_sample_weight("balanced", y_train)
            pipeline.fit(X_train, y_train, model__sample_weight=sw)

        metrics = evaluate(pipeline, X_test, y_test)
        cv = cross_validate_roc_auc(pipeline, X_train, y_train)

        results[name] = {
            "pipeline": pipeline,
            "test_metrics": metrics,
            "cv_roc_auc": cv,
        }

    best_name = max(results, key=lambda n: results[n]["test_metrics"]["roc_auc"])

    return {
        "results": results,
        "best_model_name": best_name,
        "best_pipeline": results[best_name]["pipeline"],
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
    }
