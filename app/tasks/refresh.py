"""
Celery background tasks for the nightly data refresh pipeline.

Task chain:
    ingest_raster_task  →  refresh_view_task  →  train_model_task
    (download & process)   (rebuild mat-view)     (retrain & store predictions)

Each task runs asynchronously in a Celery worker process.
Database operations use asyncpg via asyncio.run() wrappers.
"""

import asyncio
import logging
import sys
from datetime import date, datetime
from pathlib import Path

# Ensure project root is on the path for Celery workers
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from celery import shared_task
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────────


def _run_async(coro):
    """Run an async coroutine inside a synchronous Celery task."""
    return asyncio.run(coro)


# ── Task 1: Ingest raster data ─────────────────────────────────────────


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def ingest_raster_task(self, raster_path: str | None = None, date_str: str | None = None):
    """
    Download (or generate) MODIS raster data and ingest into the database.

    Steps:
        1. If no raster_path, generate sample data (dev mode)
        2. Compute per-ward LST + NDVI stats
        3. Write results to lst_readings and ndvi_values tables
    """
    from app.database import async_session_factory

    async def _ingest():
        date_str_actual = date_str or datetime.now().strftime("%Y%m%d")
        reading_date = date(
            int(date_str_actual[:4]),
            int(date_str_actual[4:6]),
            int(date_str_actual[6:8]),
        )

        # If no raster provided, generate sample data for dev/testing
        if raster_path is None:
            from app.raster.sample_data import generate_all

            logger.info("No raster path provided — generating sample data")
            generate_all(date_str_actual)
            base = Path("data/sample")
            lst_path = str(base / f"lst_{date_str_actual}.tif")
            ndvi_path = str(base / f"ndvi_{date_str_actual}.tif")
        else:
            lst_path = raster_path
            ndvi_path = raster_path.replace("lst_", "ndvi_")

        from app.raster.ingestion import load_wards_from_db, process_lst_raster, process_ndvi_raster

        async with async_session_factory() as db:
            wards_gdf = await load_wards_from_db(db)

            lst_count = 0
            ndvi_count = 0

            lst_path_obj = Path(lst_path)
            if lst_path_obj.exists():
                readings = await process_lst_raster(str(lst_path_obj), db, wards_gdf, reading_date)
                lst_count = len(readings)
                logger.info(f"Ingested {lst_count} LST readings")

            ndvi_path_obj = Path(ndvi_path)
            if ndvi_path_obj.exists():
                values = await process_ndvi_raster(str(ndvi_path_obj), db, wards_gdf, reading_date)
                ndvi_count = len(values)
                logger.info(f"Ingested {ndvi_count} NDVI values")

            await db.commit()

        return {
            "status": "success",
            "lst_readings": lst_count,
            "ndvi_values": ndvi_count,
            "date": date_str_actual,
            "raster_path": str(lst_path_obj) if raster_path is None else lst_path,
        }

    try:
        return _run_async(_ingest())
    except Exception as exc:
        logger.error(f"Ingest task failed: {exc}")
        raise self.retry(exc=exc)


# ── Task 2: Refresh materialized view ──────────────────────────────────


@shared_task(bind=True, max_retries=2)
def refresh_view_task(self):
    """Rebuild the heat_ward_view materialized view with latest data."""
    from app.database import async_session_factory

    async def _refresh():
        from app.analysis.joiner import create_heat_ward_view

        async with async_session_factory() as db:
            await create_heat_ward_view(db)
            logger.info("Materialized view refreshed")

        return {"status": "success", "view": "heat_ward_view"}

    try:
        return _run_async(_refresh())
    except Exception as exc:
        logger.error(f"View refresh failed: {exc}")
        raise self.retry(exc=exc)


# ── Task 3: Train ML model & store predictions ─────────────────────────


@shared_task(bind=True, max_retries=2, default_retry_delay=120)
def train_model_task(self, model_version: str | None = None):
    """
    Load joined data, train a new RandomForest model,
    save it to disk, and store predictions in the database.
    """
    from app.database import async_session_factory
    from app.ml.model import (
        predict_risk,
        save_model,
        store_predictions,
        train_model,
    )

    async def _train():
        from app.analysis.joiner import load_and_join

        async with async_session_factory() as db:
            # 1. Load joined data
            logger.info("Loading joined data from database")
            joined = await load_and_join(db)
            if joined.empty or "lst_mean" not in joined.columns:
                raise ValueError("No joined data available for training")

            logger.info(f"Loaded {len(joined)} rows for training")

        # 2. Train model (pure sklearn — no DB needed)
        model, metrics, metadata = train_model(
            joined,
            n_estimators=100,
            max_depth=10,
            random_state=42,
        )

        logger.info(f"Model trained — R²={metrics['r2']}, MAE={metrics['mae']}°C")

        if model_version:
            metadata.model_version = model_version

        # 3. Save to disk
        save_path = save_model(model, metadata)
        logger.info(f"Model saved to {save_path}")

        # 4. Predict & store in DB
        predictions = predict_risk(model, joined, metadata)

        async with async_session_factory() as db:
            stored = await store_predictions(
                db, predictions, metadata.model_version,
            )
            await db.commit()

        logger.info(f"Stored {len(stored)} predictions in database")

        return {
            "status": "success",
            "model_version": metadata.model_version,
            "r2": metrics["r2"],
            "mae": metrics["mae"],
            "predictions_stored": len(stored),
            "save_path": save_path,
        }

    try:
        return _run_async(_train())
    except Exception as exc:
        logger.error(f"Train task failed: {exc}")
        raise self.retry(exc=exc)


# ── Orchestrator: full refresh pipeline ────────────────────────────────


@shared_task(bind=True, max_retries=1)
def refresh_pipeline(self, auto: bool = False):
    """
    Orchestrate the full nightly refresh pipeline.

    Chain:
        1. ingest_raster_task   — process new satellite data
        2. train_model_task     — retrain ML model with updated data
        3. refresh_view_task    — rebuild materialized view

    The `auto` flag is set by Celery Beat for the scheduled run.
    """
    logger.info(f"Starting refresh pipeline (auto={'yes' if auto else 'no'})")

    # Step 1: Ingest new raster data
    logger.info("── Step 1/3: Ingesting raster data ──")
    ingest_result = ingest_raster_task()
    logger.info(f"Ingest complete: {ingest_result}")

    # Step 2: Retrain ML model
    logger.info("── Step 2/3: Training ML model ──")
    train_result = train_model_task()
    logger.info(f"Training complete: {train_result}")

    # Step 3: Refresh materialised view
    logger.info("── Step 3/3: Refreshing materialised view ──")
    view_result = refresh_view_task()
    logger.info(f"View refresh complete: {view_result}")

    summary = {
        "status": "success",
        "pipeline_run_at": datetime.now().isoformat(),
        "auto": auto,
        "steps": {
            "ingest": ingest_result,
            "train": train_result,
            "refresh_view": view_result,
        },
    }
    logger.info(f"Pipeline complete: {summary}")
    return summary
