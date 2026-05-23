"""
Seed city data into the database.

Generates mock ward boundaries, LST rasters, NDVI rasters,
ingests them, trains an ML model, and runs equity analysis —
all for a user-specified city.
"""

import asyncio
import json
import logging
from datetime import date
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session_factory
from app.models.ward import Ward

logger = logging.getLogger(__name__)


def _city_slug(city: str) -> str:
    return city.lower().replace(" ", "_")


async def seed_city(city: str, date_str: str = "20240515"):
    """
    Full seed pipeline for a single city:

        1. Generate sample GeoTIFFs + GeoJSON
        2. Insert ward boundaries into DB
        3. Ingest LST + NDVI rasters
        4. Train ML model → store predictions
        5. Run equity analysis → store metrics
    """
    from app.raster.sample_data import generate_all as generate_sample_data
    from app.raster.ingestion import (
        _compute_zonal_stats,
        load_wards_from_geojson,
        load_wards_from_db,
        process_lst_raster,
        process_ndvi_raster,
    )

    print(f"\n═══ Seeding {city} ═══")

    # 1. Generate sample rasters + GeoJSON
    geojson_path, lst_path, ndvi_path = generate_sample_data(city, date_str)

    # 2. Insert ward boundaries into DB
    async with async_session_factory() as db:
        # Check if already seeded
        result = await db.execute(select(Ward).where(Ward.city == city))
        existing = result.scalars().all()
        if existing:
            print(f"  ✓ {city} already seeded ({len(existing)} wards) — skipping ward insert")
        else:
            with open(geojson_path) as f:
                geojson = json.load(f)
            for feat in geojson["features"]:
                props = feat["properties"]
                ward = Ward(
                    name=props["name"],
                    code=props["code"],
                    city=props.get("city", city),
                    geometry=json.dumps(feat["geometry"]),
                )
                db.add(ward)
            await db.commit()
            print(f"  ✓ Inserted {len(geojson['features'])} wards for {city}")

        # 3. Ingest LST + NDVI rasters
        wards_gdf = await load_wards_from_db(db, city)

        reading_date = date(
            int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
        )

        lst_readings = await process_lst_raster(str(lst_path), db, wards_gdf, reading_date)
        ndvi_values = await process_ndvi_raster(str(ndvi_path), db, wards_gdf, reading_date)
        print(f"  ✓ Ingested {len(lst_readings)} LST + {len(ndvi_values)} NDVI records")

        await db.commit()

    # 4. Train ML model
    from app.ml.model import predict_risk, save_model, store_predictions, train_model
    from app.analysis.joiner import load_and_join

    async with async_session_factory() as db:
        joined = await load_and_join(db, city=city)
        if joined.empty or "lst_mean" not in joined.columns:
            print("  ⚠ No joined data for training — skipping ML")
        else:
            model, metrics, metadata = train_model(
                joined, n_estimators=100, max_depth=10, random_state=42
            )
            save_path = save_model(model, metadata)
            print(f"  ✓ Model trained — R²={metrics['r2']}, saved to {save_path}")

            predictions = predict_risk(model, joined, metadata)
            stored = await store_predictions(db, predictions, metadata.model_version)
            await db.commit()
            print(f"  ✓ Stored {len(stored)} predictions")

    # 5. Equity analysis
    from app.analysis.equity import (
        generate_mock_equity_data,
        run_equity_analysis,
        store_equity_metrics,
    )

    async with async_session_factory() as db:
        from app.analysis.joiner import load_and_join
        joined = await load_and_join(db, city=city)

        ward_names = joined["ward_name"].unique().tolist() if not joined.empty else []
        equity_df = generate_mock_equity_data(ward_names) if ward_names else None

        results = run_equity_analysis(joined, equity_df)
        stored = await store_equity_metrics(db, results["data"])
        await db.commit()
        print(f"  ✓ Stored {len(stored)} equity metrics")
        if results["hotspots"]:
            for h in results["hotspots"]:
                print(f"  ⚠ Hotspot: {h['ward_name']}")

    print(f"═══ {city} seed complete ✅ ═══\n")


async def seed_all_cities():
    """Seed all available cities."""
    from app.raster.sample_data import AVAILABLE_CITIES

    for city in AVAILABLE_CITIES:
        await seed_city(city)


if __name__ == "__main__":
    asyncio.run(seed_all_cities())
