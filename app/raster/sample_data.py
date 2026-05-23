"""
Generate mock MODIS-like GeoTIFF files for multiple Indian cities.

Each city has its own:
    - Geographic coordinates (longitude, latitude)
    - Mock ward boundaries (6 wards with different shapes)
    - Climate pattern (hot centre, cooler edges, coastal vs inland variation)

Supported cities:
    Bangalore, Mumbai, Delhi, Chennai, Kolkata
"""

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import Polygon

OUTPUT_DIR = Path("data/sample")
RESOLUTION_DEG = 0.0005  # ~50 metres per pixel

# ── City configurations ────────────────────────────────────────────────
# Each city has: bounds (west, south, east, north), centre, climate profile

GRID_ROWS = 20
GRID_COLS = 20

CITY_CONFIGS = {
    "Bangalore": {
        "bounds": (77.45, 12.90, 77.65, 13.00),
        "centre": (77.55, 12.95),
        "base_temp": 38.0,
        "temp_drop": 15.0,
        "urban_extra": 4.0,
        "green_boost": 5.0,
        "ndvi_urban": 0.20,
        "ndvi_rural": 0.60,
        "industrial_corner": (77.58, 12.94, 77.64, 12.98),
        "green_corner": (77.50, 12.90, 77.56, 12.94),
        "seed": 42,
        "ndvi_seed": 99,
    },
    "Mumbai": {
        "bounds": (72.75, 18.90, 72.95, 19.05),
        "centre": (72.85, 18.98),
        "base_temp": 35.0,
        "temp_drop": 12.0,
        "urban_extra": 3.0,
        "green_boost": 4.0,
        "ndvi_urban": 0.15,
        "ndvi_rural": 0.50,
        "industrial_corner": (72.88, 18.94, 72.94, 18.98),
        "green_corner": (72.78, 18.90, 72.84, 18.94),
        "seed": 100,
        "ndvi_seed": 200,
    },
    "Delhi": {
        "bounds": (77.10, 28.50, 77.30, 28.65),
        "centre": (77.20, 28.58),
        "base_temp": 42.0,
        "temp_drop": 14.0,
        "urban_extra": 5.0,
        "green_boost": 6.0,
        "ndvi_urban": 0.10,
        "ndvi_rural": 0.45,
        "industrial_corner": (77.25, 28.54, 77.30, 28.58),
        "green_corner": (77.12, 28.50, 77.18, 28.54),
        "seed": 300,
        "ndvi_seed": 400,
    },
    "Chennai": {
        "bounds": (80.18, 12.98, 80.35, 13.12),
        "centre": (80.27, 13.05),
        "base_temp": 36.0,
        "temp_drop": 10.0,
        "urban_extra": 3.0,
        "green_boost": 4.0,
        "ndvi_urban": 0.18,
        "ndvi_rural": 0.55,
        "industrial_corner": (80.30, 13.02, 80.35, 13.06),
        "green_corner": (80.20, 12.98, 80.26, 13.02),
        "seed": 500,
        "ndvi_seed": 600,
    },
    "Kolkata": {
        "bounds": (88.30, 22.48, 88.48, 22.62),
        "centre": (88.39, 22.55),
        "base_temp": 37.0,
        "temp_drop": 11.0,
        "urban_extra": 3.5,
        "green_boost": 4.5,
        "ndvi_urban": 0.15,
        "ndvi_rural": 0.52,
        "industrial_corner": (88.43, 22.52, 88.48, 22.56),
        "green_corner": (88.33, 22.48, 88.39, 22.52),
        "seed": 700,
        "ndvi_seed": 800,
    },
}

AVAILABLE_CITIES = list(CITY_CONFIGS.keys())


def _get_city_config(city: str) -> dict:
    """Get config for a city (case-insensitive)."""
    for name, config in CITY_CONFIGS.items():
        if name.lower() == city.lower():
            return config
    raise ValueError(f"Unknown city: {city}. Available: {AVAILABLE_CITIES}")


def _make_grid_cells(city: str):
    """
    Create a regular grid of cells covering the full city bounding box.
    Returns list of (name, code, Polygon).
    """
    config = _get_city_config(city)
    west, south, east, north = config["bounds"]
    slug = city.lower().replace(" ", "_")[:4]
    cell_w = (east - west) / GRID_COLS
    cell_h = (north - south) / GRID_ROWS

    cells = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x0 = west + col * cell_w
            y0 = south + row * cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            name = f"Cell R{row:02d} C{col:02d}"
            code = f"{slug}_{row:03d}_{col:03d}"
            poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            cells.append((name, code, poly))
    return cells


def generate_wards_geojson(city: str):
    """Write grid cells as GeoJSON for a given city."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = _make_grid_cells(city)

    features = []
    for name, code, polygon in cells:
        features.append({
            "type": "Feature",
            "properties": {"name": name, "code": code, "city": city},
            "geometry": {
                "type": "Polygon",
                "coordinates": [list(polygon.exterior.coords)],
            },
        })

    geojson = {"type": "FeatureCollection", "features": features}
    path = OUTPUT_DIR / f"wards_{city.lower()}.geojson"
    with open(path, "w") as f:
        json.dump(geojson, f, indent=2)
    print(f"  ✓ Wrote {path}  ({len(cells)} cells)")
    return path


def generate_lst_raster(city: str, date_str: str):
    """Generate a sample LST GeoTIFF for a specific city."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = _get_city_config(city)
    west, south, east, north = config["bounds"]
    cx, cy = config["centre"]
    base_temp = config["base_temp"]
    temp_drop = config["temp_drop"]

    width = int((east - west) / RESOLUTION_DEG)
    height = int((north - south) / RESOLUTION_DEG)

    x = np.linspace(west, east, width)
    y = np.linspace(south, north, height)
    xx, yy = np.meshgrid(x, y)

    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (max(east - west, north - south))

    lst = base_temp - dist * temp_drop
    lst += np.random.RandomState(config["seed"]).normal(0, 1.5, lst.shape)

    # Industrial zone
    ix1, iy1, ix2, iy2 = config["industrial_corner"]
    industrial_mask = (xx > ix1) & (xx < ix2) & (yy > iy1) & (yy < iy2)
    lst[industrial_mask] += config["urban_extra"]

    # Green belt
    gx1, gy1, gx2, gy2 = config["green_corner"]
    green_mask = (xx > gx1) & (xx < gx2) & (yy > gy1) & (yy < gy2)
    lst[green_mask] -= config["green_boost"]

    lst = np.clip(lst, 15.0, 50.0).astype(np.float32)

    transform = from_bounds(west, south, east, north, width, height)
    path = OUTPUT_DIR / f"lst_{city.lower()}_{date_str}.tif"

    with rasterio.open(
            path, "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs=CRS.from_epsg(4326),
            transform=transform,
            nodata=-9999,
    ) as dst:
        dst.write(lst, 1)

    print(f"  ✓ Wrote {path}  ({height}×{width} pixels)")
    return path


def generate_ndvi_raster(city: str, date_str: str):
    """Generate a sample NDVI GeoTIFF for a specific city."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = _get_city_config(city)
    west, south, east, north = config["bounds"]
    cx, cy = config["centre"]

    width = int((east - west) / RESOLUTION_DEG)
    height = int((north - south) / RESOLUTION_DEG)

    x = np.linspace(west, east, width)
    y = np.linspace(south, north, height)
    xx, yy = np.meshgrid(x, y)

    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (max(east - west, north - south))

    ndvi = config["ndvi_urban"] + dist * (config["ndvi_rural"] - config["ndvi_urban"])
    ndvi = np.clip(ndvi, -0.1, 0.95)

    # Green belt
    gx1, gy1, gx2, gy2 = config["green_corner"]
    green_mask = (xx > gx1) & (xx < gx2) & (yy > gy1) & (yy < gy2)
    ndvi[green_mask] += 0.25

    # Industrial zone
    ix1, iy1, ix2, iy2 = config["industrial_corner"]
    industrial_mask = (xx > ix1) & (xx < ix2) & (yy > iy1) & (yy < iy2)
    ndvi[industrial_mask] -= 0.15

    ndvi += np.random.RandomState(config["ndvi_seed"]).normal(0, 0.05, ndvi.shape)
    ndvi = np.clip(ndvi, -0.1, 0.95).astype(np.float32)

    transform = from_bounds(west, south, east, north, width, height)
    path = OUTPUT_DIR / f"ndvi_{city.lower()}_{date_str}.tif"

    with rasterio.open(
            path, "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=np.float32,
            crs=CRS.from_epsg(4326),
            transform=transform,
            nodata=-9999,
    ) as dst:
        dst.write(ndvi, 1)

    print(f"  ✓ Wrote {path}  ({height}×{width} pixels)")
    return path


def generate_all(city: str = "Bangalore", date_str: str = "20240515"):
    """Generate all sample data files for a given city."""
    print(f"Generating sample data for {city}...")
    geojson = generate_wards_geojson(city)
    lst = generate_lst_raster(city, date_str)
    ndvi = generate_ndvi_raster(city, date_str)
    return geojson, lst, ndvi


if __name__ == "__main__":
    # Generate for all cities
    for city in AVAILABLE_CITIES:
        generate_all(city)
