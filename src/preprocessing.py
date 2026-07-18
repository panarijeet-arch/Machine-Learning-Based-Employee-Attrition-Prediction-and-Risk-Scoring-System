"""
preprocessing.py
=================
Loads the raw dataset and runs a structured data-quality audit, in the
same disciplined style used for prior projects: check first, model
second, and disclose findings rather than silently proceeding.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Palo_Alto_Networks.csv")


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def run_data_quality_report(df: pd.DataFrame) -> dict:
    report = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "missing_values_total": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "attrition_positive": int(df["Attrition"].sum()),
        "attrition_negative": int((df["Attrition"] == 0).sum()),
        "attrition_rate_pct": round(float(df["Attrition"].mean()) * 100, 2),
        "low_variance_columns": [],
    }
    # Flag any numeric column with only 1-2 distinct values -- limited
    # predictive value, worth disclosing rather than silently including.
    for col in df.select_dtypes(include=[np.number]).columns:
        n_unique = df[col].nunique()
        if n_unique <= 2 and col != "Attrition":
            report["low_variance_columns"].append(
                {"column": col, "unique_values": sorted(df[col].unique().tolist())}
            )
    return report


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """No missing values or duplicates exist in this dataset (confirmed by
    the audit), so this is a pass-through kept for pipeline symmetry with
    datasets that do need cleaning, and as a single place to add
    validation if the source file changes."""
    df = df.drop_duplicates().reset_index(drop=True)
    return df


if __name__ == "__main__":
    raw = load_raw()
    report = run_data_quality_report(raw)
    out_path = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(out_path, "data_quality_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
