"""
Analyse a user-drawn rectangle on the map — generate raster,
grid cells, zonal stats, predictions, and equity on-the-fly.
"""

import logging
from datetime import datetime

import numpy as np
from shapely.geometry import Polygon

from app.raster.sample_data import _get_city_config

logger = logging.getLogger(__name__)

GRID_SIZE = 16


def _generate_raster_for_bounds(
    bounds: tuple[float, float, float, float],
    config: dict,
    date_str: str,
    raster_type: str,
) -> np.ndarray:
    """Generate an LST or NDVI raster array for arbitrary bounds using a city's climate profile."""
    west, south, east, north = bounds
    cx, cy = config["centre"]
    res = 0.0005
    width = max(int((east - west) / res), 10)
    height = max(int((north - south) / res), 10)

    x = np.linspace(west, east, width)
    y = np.linspace(south, north, height)
    xx, yy = np.meshgrid(x, y)

    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(east - west, north - south, 0.01)

    if raster_type == "lst":
        data = config["base_temp"] - dist * config["temp_drop"]
        data += np.random.RandomState(config["seed"]).normal(0, 1.5, data.shape)

        ix1, iy1, ix2, iy2 = config["industrial_corner"]
        industrial_mask = (xx >= ix1) & (xx <= ix2) & (yy >= iy1) & (yy <= iy2)
        data[industrial_mask] += config["urban_extra"]

        gx1, gy1, gx2, gy2 = config["green_corner"]
        green_mask = (xx >= gx1) & (xx <= gx2) & (yy >= gy1) & (yy <= gy2)
        data[green_mask] -= config["green_boost"]

        data = np.clip(data, 15.0, 50.0).astype(np.float32)
    else:
        data = config["ndvi_urban"] + dist * (config["ndvi_rural"] - config["ndvi_urban"])
        data = np.clip(data, -0.1, 0.95)

        gx1, gy1, gx2, gy2 = config["green_corner"]
        green_mask = (xx >= gx1) & (xx <= gx2) & (yy >= gy1) & (yy <= gy2)
        data[green_mask] += 0.25

        ix1, iy1, ix2, iy2 = config["industrial_corner"]
        industrial_mask = (xx >= ix1) & (xx <= ix2) & (yy >= iy1) & (yy <= iy2)
        data[industrial_mask] -= 0.15

        data += np.random.RandomState(config["ndvi_seed"]).normal(0, 0.05, data.shape)
        data = np.clip(data, -0.1, 0.95).astype(np.float32)

    return data, (west, south, east, north, width, height)


def _make_grid_for_bounds(bounds: tuple, grid_size: int) -> list[tuple[str, str, Polygon]]:
    """Create grid cells covering the given bounds."""
    west, south, east, north = bounds
    cell_w = (east - west) / grid_size
    cell_h = (north - south) / grid_size

    cells = []
    for r in range(grid_size):
        for c in range(grid_size):
            x0 = west + c * cell_w
            y0 = south + r * cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            name = f"Cell {r:02d}x{c:02d}"
            code = f"U{r:03d}_{c:03d}"
            poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            cells.append((name, code, poly))
    return cells


def _zonal_stats_from_blocks(array: np.ndarray, grid_rows: int, grid_cols: int) -> list[dict]:
    """Compute zonal stats by dividing the raster into equal grid blocks."""
    h, w = array.shape
    row_h = h // grid_rows
    col_w = w // grid_cols
    results = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            y0 = r * row_h
            y1 = y0 + row_h if r < grid_rows - 1 else h
            x0 = c * col_w
            x1 = x0 + col_w if c < grid_cols - 1 else w
            block = array[y0:y1, x0:x1]
            results.append({
                "mean": float(np.mean(block)),
                "min": float(np.min(block)),
                "max": float(np.max(block)),
                "std": float(np.std(block)),
            })
    return results


def _predict_risk_simple(lst_mean: float | None, ndvi_mean: float | None) -> dict:
    """Simple risk score based on LST and NDVI."""
    if lst_mean is None:
        return {"risk_score": None, "risk_category": "unknown"}
    score = (lst_mean - 20) / 30
    if ndvi_mean is not None:
        score = score * (1 - ndvi_mean * 0.3)
    score = max(0, min(1, score))
    if score > 0.66:
        cat = "high"
    elif score > 0.33:
        cat = "medium"
    else:
        cat = "low"
    return {"risk_score": round(score, 4), "risk_category": cat}


def _detect_hotspots(features: list[dict], top_n: int = 5) -> list[dict]:
    """Find the hottest cells."""
    temps = [(i, f["properties"]["lst_mean"]) for i, f in enumerate(features) if f["properties"].get("lst_mean") is not None]
    temps.sort(key=lambda x: -x[1])
    hot_indices = [i for i, _ in temps[:top_n]]
    return [features[i] for i in hot_indices]


def _generate_insight(features: list[dict], avg_temp: float, high_count: int, city: str) -> str:
    total = len(features)
    if not total:
        return "No data available for this area."
    temps = [f["properties"]["lst_mean"] for f in features if f["properties"].get("lst_mean") is not None]
    ndvis = [f["properties"]["ndvi_mean"] for f in features if f["properties"].get("ndvi_mean") is not None]
    max_t = max(temps) if temps else 0
    min_t = min(temps) if temps else 0
    parts = [f"Analysis of {total} cells in {city}."]
    if temps:
        parts.append(f"Temperature ranges from {min_t:.1f}°C to {max_t:.1f}°C, averaging {avg_temp:.1f}°C.")
    if high_count > 0:
        pct = high_count / total * 100
        parts.append(f"{high_count} cell{'s' if high_count != 1 else ''} ({pct:.0f}%) are high risk.")
        if pct > 50:
            parts.append("Significant UHI effect — priority intervention zone.")
    if ndvis:
        avg_ndvi = sum(ndvis) / len(ndvis)
        if avg_ndvi < 0.2:
            parts.append("Very low vegetation — tree planting recommended.")
        elif avg_ndvi < 0.4:
            parts.append("Moderate vegetation — more green infrastructure would help.")
        else:
            parts.append("Good vegetation cover.")
    return " ".join(parts)


def analyze_bounds(
    city: str,
    west: float,
    south: float,
    east: float,
    north: float,
    grid_size: int = GRID_SIZE,
) -> dict:
    """
    Full analysis for a user-drawn rectangle.
    Returns a GeoJSON FeatureCollection.
    """
    config = _get_city_config(city)
    bounds = (west, south, east, north)
    rows = cols = grid_size

    # 1. Generate grid cells
    cells = _make_grid_for_bounds(bounds, grid_size)

    # 2. Generate rasters
    lst_array, lst_info = _generate_raster_for_bounds(bounds, config, "", "lst")
    ndvi_array, _ = _generate_raster_for_bounds(bounds, config, "", "ndvi")

    # 3. Zonal stats
    lst_stats = _zonal_stats_from_blocks(lst_array, rows, cols)
    ndvi_stats = _zonal_stats_from_blocks(ndvi_array, rows, cols)

    # 4. Build features
    features = []
    for (name, code, poly), lst_s, ndvi_s in zip(cells, lst_stats, ndvi_stats):
        lst_mean = lst_s["mean"]
        ndvi_mean = ndvi_s["mean"]
        risk = _predict_risk_simple(lst_mean, ndvi_mean)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [list(poly.exterior.coords)]},
            "properties": {
                "ward_name": name,
                "ward_code": code,
                "city": city,
                "lst_mean": lst_mean,
                "lst_min": lst_s["min"],
                "lst_max": lst_s["max"],
                "ndvi_mean": ndvi_mean,
                "risk_score": risk["risk_score"],
                "risk_category": risk["risk_category"],
                "is_hotspot": False,
            },
        })

    # 5. Detect hotspots
    hotspots = _detect_hotspots(features, top_n=max(5, grid_size))
    hotspot_ids = {f["properties"]["ward_name"] for f in hotspots}
    for f in features:
        if f["properties"]["ward_name"] in hotspot_ids:
            f["properties"]["is_hotspot"] = True

    # 6. Insight
    avg_temp = float(np.mean([f["properties"]["lst_mean"] for f in features]))
    high_count = sum(1 for f in features if f["properties"]["risk_category"] == "high")
    insight = _generate_insight(features, avg_temp, high_count, city)

    return {
        "type": "FeatureCollection",
        "features": features,
        "insight": insight,
        "stats": {
            "avg_temp": round(avg_temp, 1),
            "high_count": high_count,
            "total_cells": len(features),
            "source": f"synthetic ({city})",
        },
    }
