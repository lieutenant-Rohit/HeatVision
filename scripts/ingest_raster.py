"""
CLI script to run the raster ingestion pipeline.

Two modes:
  1. Generate sample data + ingest (for testing without real MODIS data)
  2. Ingest an existing GeoTIFF against the database

Examples:
    # Generate sample data and ingest into DB
    python scripts/ingest_raster.py --sample

    # Ingest a specific LST raster
    python scripts/ingest_raster.py --lst data/sample/lst_20240515.tif

    # Ingest both LST and NDVI
    python scripts/ingest_raster.py \\
        --lst data/sample/lst_20240515.tif \\
        --ndvi data/sample/ndvi_20240515.tif

    # Skip DB — just print stats from a GeoJSON file
    python scripts/ingest_raster.py --lst data/sample/lst_20240515.tif \\
        --geojson data/sample/wards.geojson --dry-run
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path so we can import app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import async_session_factory
from app.raster.ingestion import (
    _compute_zonal_stats,
    load_wards_from_geojson,
    process_lst_raster,
    process_ndvi_raster,
)
from app.raster.sample_data import generate_all


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest")


async def _ingest_with_db(args: argparse.Namespace):
    """Run ingestion with a live PostgreSQL connection."""
    async with async_session_factory() as db:
        if args.lst:
            readings = await process_lst_raster(args.lst, db)
            logger.info(f"Ingested {len(readings)} LST readings")

        if args.ndvi:
            values = await process_ndvi_raster(args.ndvi, db)
            logger.info(f"Ingested {len(values)} NDVI values")

        await db.commit()
        logger.info("Transaction committed successfully")


def _dry_run(args: argparse.Namespace):
    """Print per-ward stats without a database."""
    wards = load_wards_from_geojson(args.geojson)

    for raster_path, label in [(args.lst, "LST"), (args.ndvi, "NDVI")]:
        if not raster_path:
            continue
        print(f"\n── {label} stats from {raster_path} ──")
        stats = _compute_zonal_stats(raster_path, wards)
        for s in stats:
            print(
                f"  {s['ward_name']:20s}  "
                f"mean={s['mean']:7.2f}  "
                f"min={s['min']:7.2f}  "
                f"max={s['max']:7.2f}  "
                f"std={s['std']:.2f}  "
                f"(pixels={s['pixel_count']})"
            )


async def main():
    parser = argparse.ArgumentParser(description="Ingest MODIS raster data into the database")
    parser.add_argument("--sample", action="store_true", help="Generate sample rasters + wards first")
    parser.add_argument("--lst", type=str, help="Path to LST GeoTIFF")
    parser.add_argument("--ndvi", type=str, help="Path to NDVI GeoTIFF")
    parser.add_argument("--geojson", type=str, default="data/sample/wards.geojson",
                        help="Ward boundaries GeoJSON (default: data/sample/wards.geojson)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing to DB (requires --geojson)")
    args = parser.parse_args()

    if args.sample:
        print("── Generating sample data ──")
        generate_all()

    if args.dry_run:
        if not args.lst and not args.ndvi:
            parser.error("--dry-run requires --lst and/or --ndvi")
        if not Path(args.geojson).exists():
            parser.error(f"GeoJSON file not found: {args.geojson}")
        _dry_run(args)
        return

    if args.lst or args.ndvi:
        print("── Running ingestion with database ──")
        await _ingest_with_db(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
