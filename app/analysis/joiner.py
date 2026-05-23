"""
GeoPandas spatial joins — ward boundaries × LST readings × NDVI values.

Three layers of functionality:
    1. load_and_join()       — Load all three datasets & merge into one GeoDataFrame
    2. Materialized view     — Persistent PostGIS view for fast API queries
    3. analyze_correlation() — Pearson r between LST and NDVI
"""

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import json

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Base, async_session_factory
from app.models.lst import LSTReading
from app.models.ndvi import NDVIValue
from app.models.ward import Ward

logger = logging.getLogger(__name__)

# ── 1. Load + join ──────────────────────────────────────────────────────


async def load_wards_gdf(db: AsyncSession, city: str | None = None) -> gpd.GeoDataFrame:
    """Load ward boundaries as a GeoDataFrame from PostgreSQL."""
    from shapely.geometry import shape
    from sqlalchemy import select

    stmt = select(Ward)
    if city:
        stmt = stmt.where(Ward.city == city)
    result = await db.execute(stmt)
    wards = result.scalars().all()

    rows = []
    for w in wards:
        geom = shape(json.loads(w.geometry))
        rows.append({
            "ward_id": w.id,
            "ward_name": w.name,
            "ward_code": w.code,
            "area_sqkm": w.area_sqkm,
            "city": w.city,
            "geometry": geom,
        })

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326", geometry="geometry")
    logger.info(f"Loaded {len(gdf)} wards{ ' for ' + city if city else ''}")
    return gdf


def load_wards_from_file(geojson_path: str) -> gpd.GeoDataFrame:
    """Load ward boundaries from a GeoJSON file (dry-run mode)."""
    gdf = gpd.read_file(geojson_path)
    # Assign synthetic IDs so joins still work
    gdf["ward_id"] = range(1, len(gdf) + 1)
    gdf["ward_name"] = gdf.get("name", gdf.index.map(lambda i: f"Ward_{i+1}"))
    gdf["ward_code"] = gdf.get("code", gdf.index.map(lambda i: f"W{i+1:03d}"))
    logger.info(f"Loaded {len(gdf)} wards from {geojson_path}")
    return gdf


async def load_lst_df(db: AsyncSession, target_date: Optional[date] = None) -> pd.DataFrame:
    """Load LST readings as a plain DataFrame."""
    from sqlalchemy import select

    stmt = select(LSTReading)
    if target_date:
        stmt = stmt.where(LSTReading.reading_date == target_date)

    result = await db.execute(stmt)
    rows = result.scalars().all()

    data = [
        {
            "ward_id": r.ward_id,
            "reading_date": r.reading_date,
            "lst_mean": r.lst_mean,
            "lst_min": r.lst_min,
            "lst_max": r.lst_max,
            "lst_std": r.lst_std,
        }
        for r in rows
    ]
    df = pd.DataFrame(data)
    logger.info(f"Loaded {len(df)} LST readings{' for ' + str(target_date) if target_date else ''}")
    return df


async def load_ndvi_df(db: AsyncSession, target_date: Optional[date] = None) -> pd.DataFrame:
    """Load NDVI values as a plain DataFrame."""
    from sqlalchemy import select

    stmt = select(NDVIValue)
    if target_date:
        stmt = stmt.where(NDVIValue.reading_date == target_date)

    result = await db.execute(stmt)
    rows = result.scalars().all()

    data = [
        {
            "ward_id": r.ward_id,
            "reading_date": r.reading_date,
            "ndvi_mean": r.ndvi_mean,
            "ndvi_min": r.ndvi_min,
            "ndvi_max": r.ndvi_max,
            "ndvi_std": r.ndvi_std,
        }
        for r in rows
    ]
    df = pd.DataFrame(data)
    logger.info(f"Loaded {len(df)} NDVI values{' for ' + str(target_date) if target_date else ''}")
    return df


async def load_and_join(
    db: AsyncSession,
    target_date: Optional[date] = None,
    city: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Load wards + LST + NDVI and merge into a single GeoDataFrame.

    Join chain:
        wards (GeoDataFrame) ──left──► LST (DataFrame) ──left──► NDVI (DataFrame)
                              on ward_id                    on ward_id + date

    Args:
        db: Database session
        target_date: Filter by specific date
        city: Filter by city name

    Returns a GeoDataFrame with geometry + both sets of measurements.
    """
    # 1. Load all three
    wards_gdf = await load_wards_gdf(db, city)
    lst_df = await load_lst_df(db, target_date)
    ndvi_df = await load_ndvi_df(db, target_date)

    # 2. Merge LST onto wards (attribute join on ward_id)
    joined = wards_gdf.merge(lst_df, on="ward_id", how="left")

    # 3. Merge NDVI onto result (on both ward_id and reading_date)
    if not ndvi_df.empty and not lst_df.empty:
        joined = joined.merge(
            ndvi_df,
            on=["ward_id", "reading_date"],
            how="left",
            suffixes=("", "_ndvi"),
        )
    elif not ndvi_df.empty:
        joined = joined.merge(ndvi_df, on="ward_id", how="left", suffixes=("", "_ndvi"))

    logger.info(f"Joined dataset: {len(joined)} rows × {len(joined.columns)} columns")
    return joined


# ── 2. Materialized view ───────────────────────────────────────────────


MAT_VIEW_NAME = "heat_ward_view"

CREATE_MAT_VIEW_SQL = f"""
CREATE MATERIALIZED VIEW IF NOT EXISTS {MAT_VIEW_NAME} AS
SELECT
    w.id              AS ward_id,
    w.name            AS ward_name,
    w.code            AS ward_code,
    w.geometry        AS geometry,
    l.reading_date,
    l.lst_mean,   l.lst_min,   l.lst_max,   l.lst_std,
    n.ndvi_mean,  n.ndvi_min,  n.ndvi_max,  n.ndvi_std,
    -- A simple derived index: higher LST + lower NDVI = more vulnerable
    CASE
        WHEN n.ndvi_mean IS NOT NULL AND n.ndvi_mean > 0
        THEN l.lst_mean / (n.ndvi_mean + 0.01)
        ELSE NULL
    END AS heat_ndvi_ratio
FROM wards w
LEFT JOIN lst_readings  l ON w.id = l.ward_id
LEFT JOIN ndvi_values   n ON w.id = n.ward_id AND l.reading_date = n.reading_date
WITH DATA;
"""

REFRESH_MAT_VIEW_SQL = f"REFRESH MATERIALIZED VIEW {MAT_VIEW_NAME};"

DROP_MAT_VIEW_SQL = f"DROP MATERIALIZED VIEW IF EXISTS {MAT_VIEW_NAME};"


async def create_heat_ward_view(db: AsyncSession):
    """Create (or recreate) the materialized view."""
    await db.execute(text(DROP_MAT_VIEW_SQL))
    await db.execute(text(CREATE_MAT_VIEW_SQL))
    await db.commit()
    logger.info(f"Materialized view '{MAT_VIEW_NAME}' created/refreshed")


async def refresh_heat_ward_view(db: AsyncSession):
    """Refresh the materialized view with latest data."""
    await db.execute(text(REFRESH_MAT_VIEW_SQL))
    await db.commit()
    logger.info(f"Materialized view '{MAT_VIEW_NAME}' refreshed")


async def query_joined_view(db: AsyncSession) -> gpd.GeoDataFrame:
    """
    Query the materialized view and return results as a GeoDataFrame.

    This is what the FastAPI endpoint will call to serve GeoJSON.
    """
    from shapely.geometry import shape

    result = await db.execute(text(f"SELECT * FROM {MAT_VIEW_NAME}"))
    rows = result.fetchall()
    column_names = result.keys()

    records = []
    for row in rows:
        row_dict = dict(zip(column_names, row))
        # Geometry is stored as GeoJSON text → shapely
        geom_json = row_dict.pop("geometry")
        if isinstance(geom_json, str):
            row_dict["geometry"] = shape(json.loads(geom_json))
        else:
            row_dict["geometry"] = shape(geom_json)
        records.append(row_dict)

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326", geometry="geometry")
    logger.info(f"Queried {len(gdf)} rows from '{MAT_VIEW_NAME}'")
    return gdf


# ── 3. Correlation analysis ────────────────────────────────────────────


def analyze_correlation(joined_gdf: gpd.GeoDataFrame) -> dict:
    """
    Compute Pearson correlation between LST and NDVI.

    Expected columns: lst_mean, ndvi_mean

    Returns:
        {
            "pearson_r": float,
            "p_value": float,
            "interpretation": str,
            "n_wards": int,
        }
    """
    df = joined_gdf.dropna(subset=["lst_mean", "ndvi_mean"]).copy()

    if len(df) < 3:
        return {
            "pearson_r": None,
            "p_value": None,
            "interpretation": "Not enough data points (< 3)",
            "n_wards": len(df),
        }

    r, p = pearsonr(df["lst_mean"], df["ndvi_mean"])

    if abs(r) > 0.7:
        interpretation = "Strong correlation"
    elif abs(r) > 0.4:
        interpretation = "Moderate correlation"
    else:
        interpretation = "Weak or no correlation"

    if r < 0:
        interpretation += " (negative — higher vegetation → cooler temps, as expected)"

    return {
        "pearson_r": round(r, 4),
        "p_value": round(p, 6),
        "interpretation": interpretation,
        "n_wards": len(df),
    }


def rank_wards_by_heat_vulnerability(joined_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Rank wards by heat vulnerability.

    Uses a simple composite score:
        vulnerability = normalized(lst_mean) - normalized(ndvi_mean)

    Higher score = more vulnerable (hotter + less green cover).
    """
    df = joined_gdf.dropna(subset=["lst_mean", "ndvi_mean"]).copy()

    if df.empty:
        return pd.DataFrame()

    # Min-max normalise each metric to 0-1
    df["lst_norm"] = (df["lst_mean"] - df["lst_mean"].min()) / (
        df["lst_mean"].max() - df["lst_mean"].min() + 1e-9
    )
    df["ndvi_norm"] = (df["ndvi_mean"] - df["ndvi_mean"].min()) / (
        df["ndvi_mean"].max() - df["ndvi_mean"].min() + 1e-9
    )

    df["vulnerability_score"] = df["lst_norm"] - df["ndvi_norm"]

    ranking = (
        df[["ward_name", "ward_code", "lst_mean", "ndvi_mean", "vulnerability_score"]]
        .sort_values("vulnerability_score", ascending=False)
        .reset_index(drop=True)
    )
    ranking["rank"] = range(1, len(ranking) + 1)

    logger.info(f"Ranked {len(ranking)} wards by heat vulnerability")
    return ranking
