import pandas as pd
import numpy as np
from scipy import stats

def report_missing(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum()

def report_duplicates(df: pd.DataFrame) -> int:
    return df.duplicated().sum()

def detect_outliers_zscore(df: pd.DataFrame, threshold=3.0) -> pd.DataFrame:
    """
    Mark outliers where any feature’s Z-score exceeds threshold.
    """
    zscores = np.abs(stats.zscore(df.select_dtypes(include=[np.number]), nan_policy='omit'))
    mask = (zscores > threshold).any(axis=1)
    return df[mask]

def quality_report(df: pd.DataFrame):
    print("=== Missing Values ===")
    print(report_missing(df))
    print("\n=== Duplicate Rows ===")
    print(report_duplicates(df))
    print("\n=== Outliers (Z-score) ===")
    outliers = detect_outliers_zscore(df)
    print(f"Found {len(outliers)} outlier rows.")
    return {
        "missing": report_missing(df),
        "duplicates": report_duplicates(df),
        "outliers": outliers
    }
