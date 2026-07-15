"""
feature_engineering.py
=======================
Builds the model-ready feature set from the raw Palo Alto Networks HR
export. All engineered features are derived strictly from verified,
non-null source columns -- no external or simulated data (unlike the
prior project, this dataset has no data-quality issues to work around,
confirmed in preprocessing.py's audit).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL_COLUMNS = [
    "BusinessTravel", "Department", "EducationField", "Gender",
    "JobRole", "MaritalStatus", "OverTime",
]

ENGINEERED_COLUMNS = [
    "IncomeToExperienceRatio", "PromotionDelayFlag",
    "EngagementScore", "WorkloadStressFlag",
]

BASE_NUMERIC_COLUMNS = [
    "Age", "DailyRate", "DistanceFromHome", "Education",
    "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement", "JobLevel",
    "JobSatisfaction", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked",
    "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction",
    "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear",
    "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager",
]

ALL_NUMERIC_COLUMNS = BASE_NUMERIC_COLUMNS + ENGINEERED_COLUMNS
FEATURE_COLUMNS = ALL_NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
TARGET_COLUMN = "Attrition"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the four engineered features called for in the project spec.
    Each is built only from verified source columns, with a documented
    formula so it can be audited (no black-box derivation)."""
    out = df.copy()

    # Income-to-experience ratio: monthly income per year of total
    # working experience. +1 avoids divide-by-zero for new-to-workforce
    # employees (TotalWorkingYears == 0).
    out["IncomeToExperienceRatio"] = (
        out["MonthlyIncome"] / (out["TotalWorkingYears"] + 1)
    ).round(2)

    # Promotion delay flag: employee has gone 4+ years without a
    # promotion while still at the company (a common attrition driver).
    out["PromotionDelayFlag"] = (
        (out["YearsSinceLastPromotion"] >= 4) & (out["YearsAtCompany"] > 0)
    ).astype(int)

    # Engagement composite score: mean of the four 1-4 satisfaction /
    # involvement survey fields already in the source data. Higher =
    # more engaged.
    out["EngagementScore"] = out[
        ["EnvironmentSatisfaction", "JobSatisfaction",
         "JobInvolvement", "RelationshipSatisfaction"]
    ].mean(axis=1).round(2)

    # Workload stress flag: frequent travel + regular overtime + below-
    # average work-life balance rating, occurring together.
    out["WorkloadStressFlag"] = (
        (out["BusinessTravel"] == "Travel_Frequently")
        & (out["OverTime"] == "Yes")
        & (out["WorkLifeBalance"] <= 2)
    ).astype(int)

    return out


def get_feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    engineered = engineer_features(df)
    X = engineered[FEATURE_COLUMNS].copy()
    y = engineered[TARGET_COLUMN].astype(int)
    return X, y
