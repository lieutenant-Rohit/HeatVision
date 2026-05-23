"""
CLI script to train, evaluate, and save the heat-risk model.

Modes:
    --dry-run   Use sample GeoJSON + GeoTIFFs (no DB required)
    (default)   Load joined data from PostgreSQL

Examples:
    # Dry-run: train on sample data, print results
    python scripts/train_model.py --dry-run

    # Train with live DB data
    python scripts/train_model.py

    # List saved models
    python scripts/train_model.py --list

    # Predict using a previously saved model
    python scripts/train_model.py --predict <model_path> --dry-run
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from app.database import async_session_factory
from app.ml.features import get_feature_importance_df
from app.ml.model import (
    ModelMetadata,
    get_latest_model_path,
    load_model,
    predict_risk,
    save_model,
    train_model,
)
from app.raster.ingestion import _compute_zonal_stats, load_wards_from_geojson

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("train")


# ── Dry-run helpers ─────────────────────────────────────────────────────


def _build_joined_from_sample() -> pd.DataFrame:
    """Build a joined GeoDataFrame from sample files (no DB)."""
    base = Path("data/sample")
    geojson_path = base / "wards.geojson"
    lst_path = base / "lst_20240515.tif"
    ndvi_path = base / "ndvi_20240515.tif"

    if not geojson_path.exists() or not lst_path.exists() or not ndvi_path.exists():
        from app.raster.sample_data import generate_all
        generate_all()

    # Load wards
    wards_gdf = load_wards_from_geojson(str(geojson_path))
    wards_gdf["ward_id"] = range(1, len(wards_gdf) + 1)
    wards_gdf["ward_name"] = wards_gdf.get("name", "Ward")
    wards_gdf["ward_code"] = wards_gdf.get("code", "W000")

    # Compute raster stats
    lst_stats = _compute_zonal_stats(str(lst_path), wards_gdf)
    ndvi_stats = _compute_zonal_stats(str(ndvi_path), wards_gdf)

    lst_df = pd.DataFrame(lst_stats)
    ndvi_df = pd.DataFrame(ndvi_stats)
    reading_date = "2024-05-15"
    lst_df["reading_date"] = reading_date
    ndvi_df["reading_date"] = reading_date

    # Merge
    joined = wards_gdf.merge(
        lst_df[["ward_id", "mean", "min", "max", "std"]], on="ward_id", how="left"
    ).rename(columns={"mean": "lst_mean", "min": "lst_min", "max": "lst_max", "std": "lst_std"})

    ndvi_renamed = ndvi_df[["ward_id", "mean", "min", "max", "std"]].rename(
        columns={"mean": "ndvi_mean", "min": "ndvi_min", "max": "ndvi_max", "std": "ndvi_std"}
    )
    joined = joined.merge(ndvi_renamed, on="ward_id", how="left")

    return joined


async def _load_joined_from_db():
    """Load joined data from PostgreSQL."""
    from app.analysis.joiner import load_and_join

    async with async_session_factory() as db:
        joined = await load_and_join(db)
        return joined


# ── Main ────────────────────────────────────────────────────────────────


def dry_run(args):
    """Train + evaluate using sample data."""
    print("═══ Loading sample data ═══")
    joined = _build_joined_from_sample()
    print(f"Loaded {len(joined)} wards × {len(joined.columns)} columns")

    print("\n═══ Training RandomForest model ═══")
    model, metrics, metadata = train_model(
        joined,
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )

    print(f"  R²  = {metrics['r2']}")
    print(f"  MAE = {metrics['mae']} °C")
    print(f"  RMSE = {metrics['rmse']} °C")
    print(f"  Train samples: {metrics['n_train']}")
    print(f"  Test samples:  {metrics['n_test']}")

    print("\n═══ Feature importance ═══")
    imp_df = get_feature_importance_df(model, metadata.feature_names)
    print(imp_df.to_string(index=False))

    print("\n═══ Predicting risk for all wards ═══")
    predictions = predict_risk(model, joined, metadata)
    print(predictions[["ward_name", "lst_actual", "lst_predicted", "risk_score", "risk_category"]].to_string(index=False))

    # Save model
    save_path = save_model(model, metadata)
    print(f"\n═══ Model saved to {save_path} ═══")


async def live_train(args):
    """Train using live DB data."""
    print("═══ Loading joined data from PostgreSQL ═══")
    joined = await _load_joined_from_db()

    if joined.empty:
        print("No data in database. Run sample ingestion first:")
        print("  python scripts/ingest_raster.py --sample")
        return

    print(f"Loaded {len(joined)} rows")

    print("\n═══ Training RandomForest model ═══")
    model, metrics, metadata = train_model(
        joined,
        n_estimators=args.estimators or 100,
        max_depth=args.max_depth or 10,
        random_state=42,
    )

    print(f"  R²   = {metrics['r2']}")
    print(f"  MAE  = {metrics['mae']} °C")
    print(f"  RMSE = {metrics['rmse']} °C")

    print("\n═══ Feature importance ═══")
    imp_df = get_feature_importance_df(model, metadata.feature_names)
    print(imp_df.to_string(index=False))

    # Predict and store
    predictions = predict_risk(model, joined, metadata)
    print(predictions[["ward_name", "lst_actual", "lst_predicted", "risk_category"]].head(10).to_string(index=False))

    save_path = save_model(model, metadata)
    print(f"\n═══ Model saved to {save_path} ═══")

    if not args.no_store:
        print("\n═══ Storing predictions in database ═══")
        from app.ml.model import store_predictions
        async with async_session_factory() as db:
            await store_predictions(db, predictions, metadata.model_version)
            await db.commit()
        print("Done")


def list_models():
    """List all saved models."""
    from app.ml.model import MODEL_DIR

    if not MODEL_DIR.exists():
        print("No models found.")
        return

    versions = sorted(MODEL_DIR.iterdir())
    for v in versions:
        meta_path = v / "metadata.json"
        if meta_path.exists():
            import json
            meta = json.loads(meta_path.read_text())
            print(f"  {meta['model_version']:40s}  R²={meta['test_r2']:.4f}  MAE={meta['test_mae']:.2f}°C  n={meta['n_samples']}")
        else:
            print(f"  {v.name}  (no metadata)")


def predict_with_saved(args):
    """Load a saved model and predict risk."""
    model_path = args.predict
    print(f"═══ Loading model from {model_path} ═══")
    model, metadata = load_model(model_path)

    print(f"  Version: {metadata.model_version}")
    print(f"  Trained: {metadata.trained_at}")
    print(f"  R²:      {metadata.test_r2}")

    # Load data
    if args.dry_run:
        joined = _build_joined_from_sample()
    else:
        joined = asyncio.run(_load_joined_from_db())

    print("\n═══ Predicting ═══")
    predictions = predict_risk(model, joined, metadata)
    print(predictions[["ward_name", "lst_actual", "lst_predicted", "risk_score", "risk_category"]].to_string(index=False))


async def main():
    parser = argparse.ArgumentParser(description="Train / evaluate / run the heat-risk ML model")
    parser.add_argument("--dry-run", action="store_true", help="Use sample files, no DB")
    parser.add_argument("--list", action="store_true", help="List saved models")
    parser.add_argument("--predict", type=str, help="Path to saved model directory for prediction")
    parser.add_argument("--no-store", action="store_true", help="Skip storing predictions in DB")
    parser.add_argument("--estimators", type=int, help="Number of trees (default: 100)")
    parser.add_argument("--max-depth", type=int, help="Max tree depth (default: 10)")
    args = parser.parse_args()

    if args.list:
        list_models()
    elif args.predict:
        predict_with_saved(args)
    elif args.dry_run:
        dry_run(args)
    else:
        await live_train(args)


if __name__ == "__main__":
    asyncio.run(main())
