"""
Raster ingestion pipeline.

Reads a GeoTIFF (LST or NDVI), overlays it with ward boundary
polygons, and computes per-ward statistics using NumPy.

Flow:
    1. Open raster with rasterio
    2. Load ward boundaries (from DB or GeoJSON)
    3. For each ward → mask raster → compute stats (mean, min, max, std)
    4. Write results to the database

Usage:
    from app.raster.ingestion import process_lst_raster
    await process_lst_raster("data/sample/lst_20240515.tif", db_session)
"""

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import shape
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.lst import LSTReading
from app.models.ndvi import NDVIValue
from app.models.ward import Ward

logger = logging.getLogger(__name__)


def _extract_date_from_path(path: Path) -> date:
    """
    Guess the reading date from the filename.

    Expects filenames like: lst_20240515.tif or ndvi_20240515.tif
    Falls back to today if parsing fails.
    """
    stem = path.stem  # e.g. "lst_20240515"
    parts = stem.split("_")
    for part in parts:
        if part.isdigit() and len(part) == 8:
            year = int(part[:4])
            month = int(part[4:6])
            day = int(part[6:8])
            return date(year, month, day)
    return date.today()


async def load_wards_from_db(db: AsyncSession, city: str | None = None) -> gpd.GeoDataFrame:
    """
    Load ward boundaries from PostgreSQL as a GeoDataFrame.

    Args:
        db: Database session
        city: Filter by city name (None = load all)

    Converts GeoJSON text geometry to shapely objects.
    """
    import json
    from shapely.geometry import shape

    stmt = select(Ward)
    if city:
        stmt = stmt.where(Ward.city == city)
    result = await db.execute(stmt)
    wards = result.scalars().all()

    rows = []
    geoms = []
    for ward in wards:
        geom_dict = json.loads(ward.geometry)
        geom = shape(geom_dict)
        rows.append({
            "id": ward.id,
            "name": ward.name,
            "code": ward.code,
        })
        geoms.append(geom)

    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
    logger.info(f"Loaded {len(gdf)} wards from database")
    return gdf


def load_wards_from_geojson(path: str) -> gpd.GeoDataFrame:
    """Load ward boundaries from a GeoJSON file (used when DB is not available)."""
    gdf = gpd.read_file(path)
    logger.info(f"Loaded {len(gdf)} wards from {path}")
    return gdf


def _compute_zonal_stats(
    raster_path: str,
    wards_gdf: gpd.GeoDataFrame,
    band: int = 1,
) -> list[dict]:
    """
    Core raster-on-wards computation.

    For each ward polygon:
      1. Mask the raster to the polygon boundary
      2. Extract valid (non-no-data) pixel values
      3. Return mean, min, max, std

    Returns a list of dicts, one per ward.
    """
    results = []

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        nodata = src.nodata

        # Ensure wards are in the same CRS as the raster
        if wards_gdf.crs != raster_crs:
            wards_gdf = wards_gdf.to_crs(raster_crs)

        for _, ward in wards_gdf.iterrows():
            geom = ward["geometry"]

            try:
                # Mask raster to this ward's polygon
                # `mask` returns (masked_array, transform)
                out_image, _ = mask(src, [geom], crop=True, nodata=nodata)
                pixel_values = out_image[band - 1]  # 2D array for this band

                # Flatten and remove no-data / NaN values
                valid = pixel_values[
                    (pixel_values != nodata) & (~np.isnan(pixel_values))
                ]

                if valid.size == 0:
                    logger.warning(f"Ward '{ward.get('name', ward.get('code', '?'))}' — no valid pixels")
                    continue

                results.append({
                    "ward_id": ward.get("ward_id") or ward.get("id"),
                    "ward_name": ward.get("name"),
                    "ward_code": ward.get("code"),
                    "mean": float(np.mean(valid)),
                    "min": float(np.min(valid)),
                    "max": float(np.max(valid)),
                    "std": float(np.std(valid)),
                    "pixel_count": int(valid.size),
                })

            except Exception as exc:
                logger.error(f"Failed to process ward '{ward.get('name', '?')}': {exc}")
                continue

    logger.info(f"Computed stats for {len(results)} / {len(wards_gdf)} wards")
    return results


async def process_lst_raster(
    raster_path: str,
    db: AsyncSession,
    wards_gdf: Optional[gpd.GeoDataFrame] = None,
    reading_date: Optional[date] = None,
) -> list[LSTReading]:
    """
    Full pipeline: load wards → compute zonal stats → write LST to DB.

    Args:
        raster_path: Path to LST GeoTIFF.
        db: Async DB session.
        wards_gdf: Pre-loaded ward boundaries (auto-loads from DB if None).
        reading_date: Override date (auto-detected from filename if None).

    Returns:
        List of created LSTReading ORM objects.
    """
    if wards_gdf is None:
        wards_gdf = await load_wards_from_db(db)

    if reading_date is None:
        reading_date = _extract_date_from_path(Path(raster_path))

    logger.info(f"Processing LST raster: {raster_path} (date: {reading_date})")

    stats = _compute_zonal_stats(raster_path, wards_gdf)

    readings = []
    for s in stats:
        if s["ward_id"] is None:
            continue  # skip wards not yet in the database
        reading = LSTReading(
            ward_id=s["ward_id"],
            reading_date=reading_date,
            lst_mean=s["mean"],
            lst_min=s["min"],
            lst_max=s["max"],
            lst_std=s["std"],
            source="MODIS",
        )
        db.add(reading)
        readings.append(reading)

    await db.flush()
    logger.info(f"Saved {len(readings)} LST readings")
    return readings


async def process_ndvi_raster(
    raster_path: str,
    db: AsyncSession,
    wards_gdf: Optional[gpd.GeoDataFrame] = None,
    reading_date: Optional[date] = None,
) -> list[NDVIValue]:
    """
    Same as process_lst_raster but writes NDVI values.
    """
    if wards_gdf is None:
        wards_gdf = await load_wards_from_db(db)

    if reading_date is None:
        reading_date = _extract_date_from_path(Path(raster_path))

    logger.info(f"Processing NDVI raster: {raster_path} (date: {reading_date})")

    stats = _compute_zonal_stats(raster_path, wards_gdf)

    values = []
    for s in stats:
        if s["ward_id"] is None:
            continue
        val = NDVIValue(
            ward_id=s["ward_id"],
            reading_date=reading_date,
            ndvi_mean=s["mean"],
            ndvi_min=s["min"],
            ndvi_max=s["max"],
            ndvi_std=s["std"],
            source="MODIS",
        )
        db.add(val)
        values.append(val)

    await db.flush()
    logger.info(f"Saved {len(values)} NDVI values")
    return values
