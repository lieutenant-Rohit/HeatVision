"""
ML model: train, predict, evaluate, and persist.

Uses scikit-learn's RandomForestRegressor to predict Land Surface
Temperature from land-cover features (NDVI statistics).

Workflow:
    1. Train on historical LST + NDVI data
    2. Evaluate on held-out test set
    3. Predict heat risk for all wards
    4. Save model + metadata for API serving

Risk categorisation (based on predicted LST):
    Low:    below mean - 0.5 * std
    Medium: within mean ± 0.5 * std
    High:   above mean + 0.5 * std
"""

import json
import logging
import os
import pickle
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from app.ml.features import build_feature_set, get_feature_importance_df

logger = logging.getLogger(__name__)

# Default path for saved model artifacts
MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelMetadata:
    """Metadata stored alongside the trained model."""
    model_version: str = ""
    trained_at: str = ""
    n_samples: int = 0
    n_features: int = 0
    feature_names: list[str] = field(default_factory=list)
    test_r2: float = 0.0
    test_mae: float = 0.0
    test_rmse: float = 0.0
    lst_mean: float = 0.0  # training target mean (for risk thresholds)
    lst_std: float = 0.0   # training target std  (for risk thresholds)


# ── Training ────────────────────────────────────────────────────────────


def train_model(
    joined_gdf: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    **kwargs,
) -> tuple[RandomForestRegressor, dict, ModelMetadata]:
    """
    Train a RandomForestRegressor to predict LST from land-cover features.

    Args:
        joined_gdf: Joined GeoDataFrame from app.analysis.joiner
        test_size: Fraction of data to hold out for evaluation
        random_state: Seed for reproducibility
        **kwargs: Passed to RandomForestRegressor (e.g. n_estimators, max_depth)

    Returns:
        (trained_model, eval_metrics, metadata)
    """
    # ── Prepare features ──
    X, y, feature_names = build_feature_set(joined_gdf)

    if len(X) < 4:
        raise ValueError(
            f"Not enough samples ({len(X)}). Need at least 4 to train."
        )

    # ── Train/test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # ── Train model ──
    model = RandomForestRegressor(
        n_estimators=kwargs.get("n_estimators", 100),
        max_depth=kwargs.get("max_depth", 10),
        min_samples_leaf=kwargs.get("min_samples_leaf", 2),
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ── Evaluate ──
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    eval_metrics = {
        "r2": round(r2, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    logger.info(
        f"Model trained — R²={r2:.4f}, MAE={mae:.2f}°C, "
        f"RMSE={rmse:.2f}°C  ({len(X_train)} train / {len(X_test)} test)"
    )

    # ── Build metadata ──
    metadata = ModelMetadata(
        model_version=f"rf_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        trained_at=datetime.now().isoformat(),
        n_samples=len(X),
        n_features=len(feature_names),
        feature_names=feature_names,
        test_r2=round(r2, 4),
        test_mae=round(mae, 4),
        test_rmse=round(rmse, 4),
        lst_mean=float(y.mean()),
        lst_std=float(y.std()),
    )

    return model, eval_metrics, metadata


# ── Prediction ──────────────────────────────────────────────────────────


def predict_risk(
    model: RandomForestRegressor,
    joined_gdf: pd.DataFrame,
    metadata: ModelMetadata,
) -> pd.DataFrame:
    """
    Predict LST and assign risk categories for all wards.

    Risk thresholds (based on training distribution):
        Low:    predicted LST <  mean - 0.5 * std
        Medium: mean ± 0.5 * std
        High:   predicted LST >  mean + 0.5 * std

    Returns a DataFrame with columns:
        ward_id, ward_name, ward_code, lst_actual (if available),
        lst_predicted, risk_score (0–1 normalised), risk_category
    """
    X, _, _ = build_feature_set(joined_gdf)

    # Predict
    y_pred = model.predict(X)

    # Build result DataFrame
    result = joined_gdf[["ward_id", "ward_name", "ward_code"]].copy()
    result["lst_predicted"] = y_pred

    if "lst_mean" in joined_gdf.columns:
        result["lst_actual"] = joined_gdf["lst_mean"].values

    # Risk score: normalise predicted LST to 0–1 across all wards
    lst_min, lst_max = y_pred.min(), y_pred.max()
    result["risk_score"] = (y_pred - lst_min) / (lst_max - lst_min + 1e-9)

    # Risk categories based on training distribution
    mu = metadata.lst_mean
    sigma = metadata.lst_std
    low_thresh = mu - 0.5 * sigma
    high_thresh = mu + 0.5 * sigma

    def _categorise(predicted_lst: float) -> str:
        if predicted_lst < low_thresh:
            return "low"
        elif predicted_lst > high_thresh:
            return "high"
        return "medium"

    result["risk_category"] = result["lst_predicted"].apply(_categorise)

    logger.info(
        f"Predicted risk for {len(result)} wards — "
        f"low: {(result['risk_category'] == 'low').sum()}, "
        f"medium: {(result['risk_category'] == 'medium').sum()}, "
        f"high: {(result['risk_category'] == 'high').sum()}"
    )

    return result


# ── Persistence ─────────────────────────────────────────────────────────


def save_model(
    model: RandomForestRegressor,
    metadata: ModelMetadata,
    path: Optional[str] = None,
) -> str:
    """
    Save trained model and metadata to disk.

    Creates:
        {path}/model.pkl      — the scikit-learn model
        {path}/metadata.json  — model metadata + evaluation results
        {path}/feature_names.json — feature column names

    Returns the path to the saved model directory.
    """
    if path is None:
        path = str(MODEL_DIR / metadata.model_version)

    save_dir = Path(path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = save_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Save metadata
    meta_path = save_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(asdict(metadata), f, indent=2)

    # Save feature names
    feat_path = save_dir / "feature_names.json"
    with open(feat_path, "w") as f:
        json.dump(metadata.feature_names, f, indent=2)

    logger.info(f"Model saved to {save_dir}/")
    return str(save_dir)


def load_model(path: str) -> tuple[RandomForestRegressor, ModelMetadata]:
    """
    Load a trained model and its metadata from disk.

    Args:
        path: Directory containing model.pkl + metadata.json

    Returns:
        (model, metadata)
    """
    model_dir = Path(path)

    model_path = model_dir / "model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    meta_path = model_dir / "metadata.json"
    with open(meta_path, "r") as f:
        meta_dict = json.load(f)
    metadata = ModelMetadata(**meta_dict)

    logger.info(f"Model loaded from {model_dir}/ (version: {metadata.model_version})")
    return model, metadata


def get_latest_model_path() -> Optional[str]:
    """
    Find the most recently saved model directory.

    Returns None if no models exist yet.
    """
    if not MODEL_DIR.exists():
        return None
    versions = sorted(MODEL_DIR.iterdir())
    if not versions:
        return None
    return str(versions[-1])


# ── Store predictions in DB ────────────────────────────────────────────


async def store_predictions(
    db,
    predictions_df: pd.DataFrame,
    model_version: str,
    prediction_date: Optional[date] = None,
) -> list:
    """
    Write prediction results to the 'predictions' table.

    Args:
        db: Async DB session
        predictions_df: DataFrame with columns ward_id, risk_score, risk_category, ...
        model_version: Version string to tag these predictions
        prediction_date: Date for the prediction (defaults to today)

    Returns:
        List of created Prediction ORM objects
    """
    from app.models.prediction import Prediction

    if prediction_date is None:
        prediction_date = date.today()

    predictions = []
    for _, row in predictions_df.iterrows():
        pred = Prediction(
            ward_id=int(row["ward_id"]),
            prediction_date=prediction_date,
            risk_score=float(row["risk_score"]),
            risk_category=str(row["risk_category"]),
            model_version=model_version,
            features_used=None,  # store as JSON string if desired
        )
        db.add(pred)
        predictions.append(pred)

    await db.flush()
    logger.info(f"Stored {len(predictions)} predictions in database")
    return predictions
