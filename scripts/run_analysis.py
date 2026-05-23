"""
CLI script to exercise the spatial join + analysis pipeline.

Two modes:
    --dry-run   Use sample GeoJSON + GeoTIFF files (no DB required)
    (default)   Use live PostgreSQL database

Examples:
    # Dry-run with sample data
    python scripts/run_analysis.py --dry-run

    # Live mode (requires running PostgreSQL)
    python scripts/run_analysis.py

    # Also create/refresh the materialized view
    python scripts/run_analysis.py --create-view
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from app.analysis.joiner import load_wards_from_file
from app.database import async_session_factory
from app.raster.ingestion import _compute_zonal_stats

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("analysis")


def dry_run_join():
    """
    Full dry-run without a database:

    1. Load ward boundaries from the sample GeoJSON
    2. Compute LST + NDVI stats directly from GeoTIFFs via rasterio
    3. Build the joined GeoDataFrame using pandas merges
    4. Run correlation + vulnerability ranking
    """
    base = Path("data/sample")
    geojson_path = base / "wards.geojson"
    lst_path = base / "lst_20240515.tif"
    ndvi_path = base / "ndvi_20240515.tif"

    if not geojson_path.exists() or not lst_path.exists() or not ndvi_path.exists():
        print("Sample data not found. Generating now...")
        from app.raster.sample_data import generate_all
        generate_all()

    # ── 1. Load wards ──
    print("\n═══ Loading ward boundaries ═══")
    wards_gdf = load_wards_from_file(str(geojson_path))
    print(wards_gdf[["ward_name", "ward_code"]].to_string(index=False))

    # ── 2. Compute raster stats ──
    print("\n═══ Computing per-ward LST stats ═══")
    lst_stats = _compute_zonal_stats(str(lst_path), wards_gdf)
    lst_df = pd.DataFrame(lst_stats)
    lst_df["reading_date"] = "2024-05-15"
    print(lst_df[["ward_name", "mean", "min", "max"]].to_string(index=False))

    print("\n═══ Computing per-ward NDVI stats ═══")
    ndvi_stats = _compute_zonal_stats(str(ndvi_path), wards_gdf)
    ndvi_df = pd.DataFrame(ndvi_stats)
    ndvi_df["reading_date"] = "2024-05-15"
    print(ndvi_df[["ward_name", "mean", "min", "max"]].to_string(index=False))

    # ── 3. Spatial join ──
    print("\n═══ Joining wards × LST × NDVI (GeoPandas merge) ═══")
    # Merge LST
    joined = wards_gdf.merge(
        lst_df[["ward_id", "reading_date", "mean", "min", "max", "std"]],
        on="ward_id", how="left"
    )
    joined = joined.rename(columns={
        "mean": "lst_mean", "min": "lst_min", "max": "lst_max", "std": "lst_std"
    })

    # Merge NDVI
    ndvi_rename = ndvi_df[["ward_id", "reading_date", "mean", "min", "max", "std"]].rename(
        columns={"mean": "ndvi_mean", "min": "ndvi_min", "max": "ndvi_max", "std": "ndvi_std"}
    )
    joined = joined.merge(ndvi_rename, on=["ward_id", "reading_date"], how="left")

    print(f"Joined dataset: {len(joined)} rows × {len(joined.columns)} columns")
    print(f"Columns: {list(joined.columns)}")

    # Show the joined table
    display_cols = ["ward_name", "lst_mean", "lst_min", "lst_max", "ndvi_mean", "ndvi_min", "ndvi_max"]
    print(joined[display_cols].to_string(index=False))

    # ── 4. Correlation analysis ──
    from app.analysis.joiner import analyze_correlation, rank_wards_by_heat_vulnerability

    print("\n═══ LST–NDVI Correlation ═══")
    corr = analyze_correlation(joined)
    for k, v in corr.items():
        print(f"  {k}: {v}")

    print("\n═══ Heat Vulnerability Ranking (most vulnerable first) ═══")
    ranking = rank_wards_by_heat_vulnerability(joined)
    if not ranking.empty:
        print(ranking[["rank", "ward_name", "lst_mean", "ndvi_mean", "vulnerability_score"]].to_string(index=False))
    else:
        print("  (insufficient data)")


async def live_join(args):
    """Run the join pipeline against PostgreSQL."""
    from app.analysis.joiner import (
        analyze_correlation,
        create_heat_ward_view,
        load_and_join,
        query_joined_view,
        rank_wards_by_heat_vulnerability,
    )

    async with async_session_factory() as db:
        # Create materialized view if requested
        if args.create_view:
            await create_heat_ward_view(db)

        print("\n═══ Loading & joining from PostgreSQL ═══")
        joined = await load_and_join(db)

        if joined.empty:
            print("No data found. Run `python scripts/ingest_raster.py --sample` first.")
            return

        print(f"Joined {len(joined)} rows")

        # Show a preview
        cols = [c for c in ["ward_name", "lst_mean", "ndvi_mean"] if c in joined.columns]
        if cols:
            print(joined[cols].head(10).to_string(index=False))

        print("\n═══ LST–NDVI Correlation ═══")
        corr = analyze_correlation(joined)
        for k, v in corr.items():
            print(f"  {k}: {v}")

        print("\n═══ Vulnerability Ranking ═══")
        ranking = rank_wards_by_heat_vulnerability(joined)
        if not ranking.empty:
            print(ranking[["rank", "ward_name", "lst_mean", "ndvi_mean", "vulnerability_score"]].to_string(index=False))

        # Query from the materialized view if it exists
        if args.create_view:
            print("\n═══ Materialized view preview ═══")
            view_gdf = await query_joined_view(db)
            print(f"  Queried {len(view_gdf)} rows from '{'heat_ward_view'}'")


async def main():
    parser = argparse.ArgumentParser(description="Run spatial join + analysis")
    parser.add_argument("--dry-run", action="store_true", help="Use sample files, no DB")
    parser.add_argument("--create-view", action="store_true", help="Create/refresh materialized view")
    args = parser.parse_args()

    if args.dry_run:
        dry_run_join()
    else:
        await live_join(args)


if __name__ == "__main__":
    asyncio.run(main())
