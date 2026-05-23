"""
FastAPI route handlers for the Urban Heat Island Mapper API.

Each endpoint returns GeoJSON FeatureCollections consumable
by Leaflet.js on the frontend.

Endpoints:
    GET  /api/wards         — All ward boundaries
    GET  /api/heat-map      — Wards × LST × NDVI (joined)
    GET  /api/predictions   — ML risk predictions
    GET  /api/equity        — Equity metrics
    GET  /api/wards/{id}    — Single ward details
    GET  /api/stats         — Dashboard summary statistics
"""

import asyncio
import json
import logging
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from shapely.geometry import shape
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from pydantic import BaseModel

from app.database import get_db
from app.models.equity import EquityMetric
from app.models.lst import LSTReading
from app.models.ndvi import NDVIValue
from app.models.prediction import Prediction
from app.models.ward import Ward

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Heat Mapper API"])


# ── Helper: build GeoJSON from any SQL query ───────────────────────────


async def _query_geojson(
    db: AsyncSession,
    sql: str,
    params: dict | None = None,
) -> dict:
    """
    Execute a SQL query and return results as a GeoJSON FeatureCollection.

    The SQL must select a column aliased as 'geometry' containing a
    GeoJSON string (e.g. from the wards.geometry column).
    All other columns become feature properties.
    """
    result = await db.execute(text(sql), params or {})
    rows = result.fetchall()
    col_names = list(result.keys())

    features = []
    for row in rows:
        row_dict = dict(zip(col_names, row))
        geojson_str = row_dict.pop("geometry", None)
        if geojson_str is None:
            continue

        # geometry is stored as a GeoJSON string — parse it
        geom_dict = json.loads(geojson_str) if isinstance(geojson_str, str) else geojson_str

        features.append({
            "type": "Feature",
            "geometry": geom_dict,
            "properties": {
                k: (
                    str(v) if isinstance(v, (date,))
                    else float(v) if isinstance(v, (float,))
                    else int(v) if isinstance(v, (int,))
                    else v
                )
                for k, v in row_dict.items()
                if v is not None
            },
        })

    return {
        "type": "FeatureCollection",
        "features": features,
    }


# ── Endpoints ───────────────────────────────────────────────────────────


@router.get("/cities", summary="List available cities")
async def get_cities(db: AsyncSession = Depends(get_db)):
    """Return all cities that have data in the database."""
    from app.raster.sample_data import AVAILABLE_CITIES

    result = await db.execute(
        select(Ward.city).distinct().order_by(Ward.city)
    )
    db_cities = [row[0] for row in result.fetchall()]

    return {
        "cities": db_cities or AVAILABLE_CITIES,
        "default": db_cities[0] if db_cities else (AVAILABLE_CITIES[0] if AVAILABLE_CITIES else "Bangalore"),
    }


@router.get("/wards", summary="All ward boundaries")
async def get_wards(
    city: str | None = Query(None, description="Filter by city name"),
    db: AsyncSession = Depends(get_db),
):
    """Return ward polygons as GeoJSON (optionally filtered by city)."""
    city_filter = ""
    if city:
        city_filter = f" WHERE w.city = '{city}'"

    sql = f"""
        SELECT
            w.id          AS ward_id,
            w.name        AS ward_name,
            w.code        AS ward_code,
            w.area_sqkm,
            w.population,
            w.city,
            w.geometry
        FROM wards w
        {city_filter}
        ORDER BY w.name
    """
    return await _query_geojson(db, sql)


@router.get("/heat-map", summary="LST + NDVI joined with ward boundaries")
async def get_heat_map(
    city: str | None = Query(None, description="Filter by city name"),
    date_from: str | None = Query(None, description="Filter: start date (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="Filter: end date (YYYY-MM-DD)"),
    db: AsyncSession = Depends(get_db),
):
    """
    Return wards joined with their latest LST and NDVI readings as GeoJSON.

    Properties include: lst_mean, ndvi_mean, heat_ndvi_ratio
    Supports ?city=Bangalore&city=Mumbai filtering.
    """
    filters = ""
    if city:
        filters += f" AND w.city = '{city}'"
    if date_from:
        filters += f" AND l.reading_date >= '{date_from}'"
    if date_to:
        filters += f" AND l.reading_date <= '{date_to}'"

    where_clause = "WHERE 1=1" + filters

    sql = f"""
        SELECT
            w.id          AS ward_id,
            w.name        AS ward_name,
            w.code        AS ward_code,
            w.city,
            l.reading_date::text,
            l.lst_mean,   l.lst_min,   l.lst_max,   l.lst_std,
            n.ndvi_mean,  n.ndvi_min,  n.ndvi_max,  n.ndvi_std,
            CASE
                WHEN n.ndvi_mean IS NOT NULL AND n.ndvi_mean > 0
                THEN l.lst_mean / (n.ndvi_mean + 0.01)
                ELSE NULL
            END AS heat_ndvi_ratio,
            w.geometry
        FROM wards w
        LEFT JOIN LATERAL (
            SELECT * FROM lst_readings l2
            WHERE l2.ward_id = w.id
            ORDER BY l2.reading_date DESC
            LIMIT 1
        ) l ON TRUE
        LEFT JOIN LATERAL (
            SELECT * FROM ndvi_values n2
            WHERE n2.ward_id = w.id
            ORDER BY n2.reading_date DESC
            LIMIT 1
        ) n ON TRUE
        {where_clause}
        ORDER BY w.name
    """
    return await _query_geojson(db, sql)


@router.get("/predictions", summary="ML risk predictions")
async def get_predictions(
    city: str | None = Query(None, description="Filter by city name"),
    category: str | None = Query(None, description="Filter: low / medium / high"),
    db: AsyncSession = Depends(get_db),
):
    """
    Return ML model predictions as GeoJSON.

    Properties include: risk_score (0-1), risk_category, lst_predicted.
    """
    filters = ""
    if city:
        filters += f" AND w.city = '{city}'"
    if category:
        filters += f" AND p.risk_category = '{category}'"

    sql = f"""
        SELECT
            w.id          AS ward_id,
            w.name        AS ward_name,
            w.code        AS ward_code,
            w.city,
            p.risk_score,
            p.risk_category,
            p.model_version,
            p.prediction_date::text,
            w.geometry
        FROM wards w
        JOIN predictions p ON w.id = p.ward_id
        WHERE p.id IN (
            SELECT MAX(id) FROM predictions GROUP BY ward_id
        )
        {filters}
        ORDER BY p.risk_score DESC
    """
    return await _query_geojson(db, sql)


@router.get("/equity", summary="Equity & environmental justice metrics")
async def get_equity(
    city: str | None = Query(None, description="Filter by city name"),
    db: AsyncSession = Depends(get_db),
):
    """
    Return equity metrics as GeoJSON.

    Properties include: median_income, green_cover_pct,
    heat_vulnerability_index, population_density.
    """
    city_filter = ""
    if city:
        city_filter = f" WHERE w.city = '{city}'"

    sql = f"""
        SELECT
            w.id          AS ward_id,
            w.name        AS ward_name,
            w.code        AS ward_code,
            w.city,
            e.median_income,
            e.green_cover_pct,
            e.heat_vulnerability_index,
            e.population_density,
            e.metric_date::text,
            w.geometry
        FROM wards w
        JOIN equity_metrics e ON w.id = e.ward_id
        {city_filter}
        ORDER BY e.heat_vulnerability_index DESC NULLS LAST
    """
    return await _query_geojson(db, sql)


@router.get("/wards/{ward_id}", summary="Single ward details")
async def get_ward_detail(
    ward_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Return a single ward with its LST, NDVI, and prediction data."""
    sql = """
        SELECT
            w.id             AS ward_id,
            w.name           AS ward_name,
            w.code           AS ward_code,
            w.area_sqkm,
            w.population,
            l.lst_mean,      l.lst_min,      l.lst_max,
            n.ndvi_mean,     n.ndvi_min,     n.ndvi_max,
            p.risk_score,    p.risk_category,
            e.median_income, e.green_cover_pct,
            e.heat_vulnerability_index,
            w.geometry
        FROM wards w
        LEFT JOIN LATERAL (SELECT * FROM lst_readings WHERE ward_id = :wid ORDER BY reading_date DESC LIMIT 1) l ON TRUE
        LEFT JOIN LATERAL (SELECT * FROM ndvi_values WHERE ward_id = :wid ORDER BY reading_date DESC LIMIT 1) n ON TRUE
        LEFT JOIN LATERAL (SELECT * FROM predictions WHERE ward_id = :wid ORDER BY created_at DESC LIMIT 1) p ON TRUE
        LEFT JOIN LATERAL (SELECT * FROM equity_metrics WHERE ward_id = :wid ORDER BY metric_date DESC LIMIT 1) e ON TRUE
        WHERE w.id = :wid
    """
    result = await _query_geojson(db, sql, {"wid": ward_id})
    if not result["features"]:
        raise HTTPException(status_code=404, detail=f"Ward {ward_id} not found")
    return result


@router.get("/stats", summary="Dashboard summary statistics")
async def get_stats(
    city: str | None = Query(None, description="Filter by city name"),
    db: AsyncSession = Depends(get_db),
):
    """Return aggregate statistics for the dashboard header."""
    city_join = ""
    city_filter = ""
    if city:
        city_join = " JOIN wards w ON l.ward_id = w.id"
        city_filter = f" WHERE w.city = '{city}'"

    # Count wards
    ward_count = (await db.execute(
        select(func.count(Ward.id)).where(Ward.city == city) if city else select(func.count(Ward.id))
    )).scalar()

    # Average LST from latest readings
    lst_stats = await db.execute(text(f"""
        SELECT
            ROUND(AVG(l.lst_mean)::numeric, 2) AS avg_lst,
            ROUND(MIN(l.lst_min)::numeric, 2)  AS min_lst,
            ROUND(MAX(l.lst_max)::numeric, 2)  AS max_lst
        FROM (
            SELECT DISTINCT ON (ward_id) ward_id, lst_mean, lst_min, lst_max
            FROM lst_readings
            ORDER BY ward_id, reading_date DESC
        ) l
        {city_join}
        {city_filter}
    """))
    lst_row = lst_stats.one()

    # Average NDVI from latest readings
    ndvi_avg = await db.execute(text(f"""
        SELECT ROUND(AVG(n.ndvi_mean)::numeric, 3)
        FROM (
            SELECT DISTINCT ON (ward_id) ward_id, ndvi_mean
            FROM ndvi_values
            ORDER BY ward_id, reading_date DESC
        ) n
        {city_join.replace('l', 'n')}
        {city_filter}
    """))
    avg_ndvi = ndvi_avg.scalar()

    # Risk counts
    risk_counts = await db.execute(text(f"""
        SELECT
            COALESCE(SUM(CASE WHEN risk_category = 'high'   THEN 1 ELSE 0 END), 0) AS high,
            COALESCE(SUM(CASE WHEN risk_category = 'medium' THEN 1 ELSE 0 END), 0) AS medium,
            COALESCE(SUM(CASE WHEN risk_category = 'low'    THEN 1 ELSE 0 END), 0) AS low
        FROM (
            SELECT DISTINCT ON (ward_id) ward_id, risk_category
            FROM predictions
            ORDER BY ward_id, created_at DESC
        ) p
        JOIN wards w ON p.ward_id = w.id
        {city_filter}
    """))
    risk_row = risk_counts.one()

    # Last updated
    last_upd = await db.execute(text("""
        SELECT MAX(created_at)::text FROM (
            SELECT created_at FROM lst_readings
            UNION ALL
            SELECT created_at FROM predictions
        ) t
    """))
    last_updated = last_upd.scalar()

    return {
        "total_wards": ward_count or 0,
        "avg_lst": lst_row.avg_lst,
        "min_lst": lst_row.min_lst,
        "max_lst": lst_row.max_lst,
        "avg_ndvi": avg_ndvi,
        "high_risk_wards": risk_row.high or 0,
        "medium_risk_wards": risk_row.medium or 0,
        "low_risk_wards": risk_row.low or 0,
        "last_updated": last_updated,
    }


# ── Celery pipeline trigger ────────────────────────────────────────────


@router.post("/refresh", summary="Trigger the data refresh pipeline")
async def trigger_refresh():
    """
    Start the full refresh pipeline as a background Celery task.

    Chain: ingest raster → retrain ML model → refresh materialized view.

    Returns immediately with a task ID. Poll GET /api/tasks/{task_id}
    for status.

    Requires Redis + Celery worker:
        celery -A app.celery_app worker --beat --loglevel=info
    """
    try:
        from app.tasks.refresh import refresh_pipeline

        task = refresh_pipeline.delay(auto=False)
        logger.info(f"Refresh pipeline triggered — task_id={task.id}")

        return {
            "status": "submitted",
            "task_id": task.id,
            "message": "Refresh pipeline started. Check /api/tasks/{task_id} for progress.",
        }
    except ImportError:
        return {
            "status": "error",
            "message": "Celery not available. Install celery or start the worker.",
        }
    except Exception as exc:
        error_msg = str(exc)
        if "Connection refused" in error_msg or "Error 61" in error_msg:
            return {
                "status": "error",
                "message": "Redis is not running. Start it with: redis-server",
            }
        return {"status": "error", "message": f"Failed to submit task: {exc}"}


@router.get("/tasks/{task_id}", summary="Check Celery task status")
async def get_task_status(task_id: str):
    """Return the status and result of a Celery task."""
    try:
        from celery.result import AsyncResult
        from app.celery_app import celery_app

        result = AsyncResult(task_id, app=celery_app)

        response = {
            "task_id": task_id,
            "status": result.status,
            "ready": result.ready(),
        }

        if result.ready():
            if result.successful():
                response["result"] = result.get()
            else:
                response["error"] = str(result.result)

        return response
    except ImportError:
        return {"status": "error", "message": "Celery not available"}
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Redis/Celery backend unavailable: {exc}",
            "hint": "Start Redis: redis-server",
        }


class AnalyzeAreaRequest(BaseModel):
    city: str | None = None
    west: float
    south: float
    east: float
    north: float
    grid_size: int = 16


@router.post("/analyze-area", summary="Analyse a user-drawn rectangle")
async def analyze_area(body: AnalyzeAreaRequest):
    """
    Generate heat data for any rectangle on Earth.
    If a known city is provided, uses its climate profile for synthetic data.
    Otherwise fetches real temperature data from Open-Meteo API.
    Returns GeoJSON consumable by the Leaflet frontend.
    """
    if body.city:
        from app.analysis.area_analysis import analyze_bounds
        try:
            return analyze_bounds(
                city=body.city,
                west=body.west,
                south=body.south,
                east=body.east,
                north=body.north,
                grid_size=body.grid_size,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # No city — use global mode with real Open-Meteo data
    from app.analysis.global_analysis import analyze_global_bounds
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, analyze_global_bounds,
            body.west, body.south, body.east, body.north,
            body.grid_size,
        )
        return result
    except Exception as e:
        logger.exception("Global area analysis failed")
        raise HTTPException(status_code=500, detail=f"Global analysis failed: {e}")
