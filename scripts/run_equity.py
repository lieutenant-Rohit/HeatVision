"""
CLI script for equity & environmental justice analysis.

Modes:
    --dry-run   Use sample data (no DB required)
    (default)   Load from PostgreSQL and store results

Examples:
    python scripts/run_equity.py --dry-run
    python scripts/run_equity.py
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from app.database import async_session_factory
from app.raster.ingestion import _compute_zonal_stats, load_wards_from_geojson

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("equity")


def _build_joined_from_sample() -> pd.DataFrame:
    """Build a joined DataFrame from sample GeoTIFFs (same as train_model dry-run)."""
    base = Path("data/sample")
    geojson_path = base / "wards.geojson"
    lst_path = base / "lst_20240515.tif"
    ndvi_path = base / "ndvi_20240515.tif"

    if not all(p.exists() for p in [geojson_path, lst_path, ndvi_path]):
        from app.raster.sample_data import generate_all
        generate_all()

    wards_gdf = load_wards_from_geojson(str(geojson_path))
    wards_gdf["ward_id"] = range(1, len(wards_gdf) + 1)
    wards_gdf["ward_name"] = wards_gdf.get("name", "Ward")
    wards_gdf["ward_code"] = wards_gdf.get("code", "W000")

    lst_stats = _compute_zonal_stats(str(lst_path), wards_gdf)
    ndvi_stats = _compute_zonal_stats(str(ndvi_path), wards_gdf)

    lst_df = pd.DataFrame(lst_stats)
    ndvi_df = pd.DataFrame(ndvi_stats)
    lst_df["reading_date"] = "2024-05-15"
    ndvi_df["reading_date"] = "2024-05-15"

    joined = wards_gdf.merge(lst_df[["ward_id", "mean", "min", "max", "std"]], on="ward_id", how="left") \
        .rename(columns={"mean": "lst_mean", "min": "lst_min", "max": "lst_max", "std": "lst_std"})

    ndvi_renamed = ndvi_df[["ward_id", "mean", "min", "max", "std"]] \
        .rename(columns={"mean": "ndvi_mean", "min": "ndvi_min", "max": "ndvi_max", "std": "ndvi_std"})
    joined = joined.merge(ndvi_renamed, on="ward_id", how="left")

    return joined


def dry_run():
    """Run equity analysis using sample data."""
    print("═══ Loading sample heat data ═══")
    joined = _build_joined_from_sample()
    print(f"Loaded {len(joined)} wards")

    print("\n═══ Running equity analysis ═══")
    from app.analysis.equity import (
        generate_mock_equity_data,
        run_equity_analysis,
    )

    # Generate mock socio-economic data
    ward_names = joined["ward_name"].unique().tolist()
    equity_df = generate_mock_equity_data(ward_names)

    print("\nMock equity data:")
    print(equity_df.to_string(index=False))

    # Run analysis
    results = run_equity_analysis(joined, equity_df)

    # ── Correlations ──
    print("\n═══ Correlation Analysis ═══")
    for label, corr in results["correlations"].items():
        print(f"  {label}:")
        print(f"    r = {corr['r']},  p = {corr['p']},  n = {corr['n']}")
        print(f"    → {corr['interpretation']}")
        if "policy_note" in corr:
            print(f"    💡 {corr['policy_note']}")

    # ── HVI Ranking ──
    print("\n═══ Heat Vulnerability Ranking (most vulnerable first) ═══")
    ranking = results["ranking"]
    if not ranking.empty:
        print(ranking.to_string(index=False))
    else:
        print("  (HVI could not be computed — missing data)")

    # ── Hotspots ──
    print("\n═══ Environmental Justice Hotspots ═══")
    hotspots = results["hotspots"]
    if hotspots:
        for h in hotspots:
            print(f"  ⚠ {h['ward_name']} — LST: {h['lst_mean']:.1f}°C, "
                  f"Income: ${h['median_income']:,.0f}, "
                  f"HVI: {h['heat_vulnerability_index']:.3f}")
        print(f"\n  → {len(hotspots)} ward(s) need priority intervention")
    else:
        print("  No environmental justice hotspots identified")

    # ── Summary ──
    print("\n═══ Equity Summary ═══")
    corr_list = list(results["correlations"].values())
    if corr_list:
        avg_abs_r = sum(abs(c["r"]) for c in corr_list if c.get("r") is not None) / max(len(corr_list), 1)
        print(f"  Average |r| across all correlations: {avg_abs_r:.3f}")

    if "heat_vulnerability_index" in joined.columns or any("heat_vulnerability_index" in str(c) for c in joined.columns):
        # Already merged
        pass
    # Get HVI from result data
    data = results["data"]
    if "heat_vulnerability_index" in data.columns:
        avg_hvi = data["heat_vulnerability_index"].mean()
        print(f"  Mean Heat Vulnerability Index: {avg_hvi:.3f} (0=least, 1=most vulnerable)")


async def live_run():
    """Run equity analysis using PostgreSQL data and store results."""
    from app.analysis.equity import (
        run_equity_analysis,
        store_equity_metrics,
    )
    from app.analysis.joiner import load_and_join

    async with async_session_factory() as db:
        print("═══ Loading joined data from PostgreSQL ═══")
        joined = await load_and_join(db)
        if joined.empty:
            print("No data in DB. Run ingest first.")
            return

        print(f"Loaded {len(joined)} wards")

        print("\n═══ Running equity analysis ═══")
        results = run_equity_analysis(joined)

        # Print summary
        print("\nCorrelations:")
        for label, corr in results["correlations"].items():
            print(f"  {label}: r={corr['r']}, p={corr['p']}")

        print("\nTop 5 most vulnerable wards:")
        ranking = results["ranking"]
        if not ranking.empty:
            print(ranking.head(5).to_string(index=False))

        print(f"\nHotspots: {len(results['hotspots'])}")

        # Store in database
        print("\n═══ Storing equity metrics in DB ═══")
        stored = await store_equity_metrics(db, results["data"])
        await db.commit()
        print(f"Stored {len(stored)} equity metric records")


def main():
    parser = argparse.ArgumentParser(description="Run equity & environmental justice analysis")
    parser.add_argument("--dry-run", action="store_true", help="Use sample files, no DB")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
    else:
        asyncio.run(live_run())


if __name__ == "__main__":
    main()
