"""
Feature engineering for heat-risk prediction.

Transforms raw LST + NDVI readings into a feature matrix
suitable for scikit-learn regression models.

Feature set (per ward per date):
    ndvi_mean, ndvi_min, ndvi_max, ndvi_std  (from MODIS)
    ndvi_range = ndvi_max - ndvi_min          (derived — vegetation heterogeneity)
    lst_mean                                   (target variable)

Usage:
    from app.ml.features import build_feature_set
    X, y, feature_names = build_feature_set(joined_gdf)
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that must exist in the input GeoDataFrame
REQUIRED_COLUMNS = [
    "lst_mean",
    "ndvi_mean",
    "ndvi_min",
    "ndvi_max",
    "ndvi_std",
]


def _validate_input(df: pd.DataFrame) -> None:
    """Raise if any required column is missing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_feature_set(
    joined_gdf: pd.DataFrame,
    drop_na: bool = True,
    target_col: str = "lst_mean",
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Build feature matrix (X) and target vector (y) from a joined GeoDataFrame.

    Feature engineering steps:
        1. Select NDVI columns as base features
        2. Add derived feature: ndvi_range (max - min)
        3. Separate target (LST mean)

    Returns:
        X  — Feature DataFrame (wards × features)
        y  — Target Series (LST values)
        feature_names — List of feature column names used
    """
    _validate_input(joined_gdf)
    df = joined_gdf.copy()

    # Base features: NDVI statistics
    feature_cols = ["ndvi_mean", "ndvi_min", "ndvi_max", "ndvi_std"]

    # Derived feature: range of NDVI within the ward
    df["ndvi_range"] = df["ndvi_max"] - df["ndvi_min"]
    feature_cols.append("ndvi_range")

    # Drop rows with missing values
    if drop_na:
        before = len(df)
        df = df.dropna(subset=feature_cols + [target_col])
        after = len(df)
        if after < before:
            logger.info(f"Dropped {before - after} rows with missing values")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    logger.info(
        f"Feature set: {X.shape[0]} samples × {X.shape[1]} features → "
        f"target range [{y.min():.2f}, {y.max():.2f}]"
    )

    return X, y, feature_cols


def get_feature_importance_df(model, feature_names: list[str]) -> pd.DataFrame:
    """
    Extract feature importances from a trained tree-based model.

    Returns a DataFrame sorted by importance (descending).
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = model.coef_
        # Take absolute value for linear models
        importances = [abs(c) for c in importances]
    else:
        raise AttributeError("Model does not expose feature_importances_ or coef_")

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    importance_df["importance_pct"] = (
        importance_df["importance"] / importance_df["importance"].sum() * 100
    )

    return importance_df
