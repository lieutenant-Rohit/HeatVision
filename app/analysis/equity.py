"""
Equity & environmental justice analysis.

Correlates heat-risk zones with socio-economic data (income, green cover,
population density) to identify wards that are disproportionately affected
by urban heat and may need priority intervention.

Analysis outputs:
    1. Correlation matrix — LST vs income, LST vs green cover, etc.
    2. Heat Vulnerability Index (HVI) — composite score per ward
    3. Environmental justice hotspots — high heat + low income wards
    4. Rankings — most vulnerable wards for policy targeting
"""

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.equity import EquityMetric

logger = logging.getLogger(__name__)


# ── 1. Mock data generator (for dev / demo) ────────────────────────────


def generate_mock_equity_data(ward_names: list[str]) -> pd.DataFrame:
    """
    Create realistic mock socio-economic data for each ward.

    In a production system, this would come from census data,
    OpenStreetMap, or government APIs.

    Pattern:
        - City Centre: high income, low green cover, high density
        - Green Belt: low income, high green cover, low density
        - Industrial Zone: low income, low green cover, moderate density
        - Suburbs: moderate income, high green cover, low density
    """
    np.random.seed(2024)

    profiles = {
        "City Centre":     {"income": 85000, "green": 0.12, "density": 12000},
        "Industrial Zone": {"income": 35000, "green": 0.08, "density": 8000},
        "Green Belt":      {"income": 28000, "green": 0.55, "density": 2500},
        "Suburban North":  {"income": 65000, "green": 0.45, "density": 4000},
        "East District":   {"income": 48000, "green": 0.25, "density": 6000},
        "South Ward":      {"income": 52000, "green": 0.35, "density": 3500},
    }

    rows = []
    for name in ward_names:
        profile = profiles.get(name)
        if profile:
            # Add ±10% noise for realism
            noise = np.random.normal(1, 0.05)
            income = int(profile["income"] * noise)
            green = round(min(profile["green"] * noise, 0.95), 3)
            density = int(profile["density"] * noise)
        else:
            income = int(np.random.normal(50000, 15000))
            green = round(np.random.uniform(0.1, 0.6), 3)
            density = int(np.random.normal(5000, 2000))

        rows.append({
            "ward_name": name,
            "median_income": max(income, 15000),
            "green_cover_pct": max(min(green, 0.95), 0.02),
            "population_density": max(density, 500),
        })

    return pd.DataFrame(rows)


# ── 2. Join equity data with heat data ─────────────────────────────────


def load_and_join_equity(
    joined_gdf: pd.DataFrame,
    equity_df: Optional[pd.DataFrame] = None,
    ward_name_col: str = "ward_name",
) -> pd.DataFrame:
    """
    Merge equity (income/green/density) data onto the joined LST + NDVI dataset.

    Args:
        joined_gdf: The joined GeoDataFrame from joiner.load_and_join()
        equity_df: Equity data (auto-generates mock data if None)
        ward_name_col: Column name to join on

    Returns:
        DataFrame with heat data + equity data + predictions
    """
    df = joined_gdf.copy()

    if equity_df is None:
        ward_names = df[ward_name_col].unique().tolist()
        equity_df = generate_mock_equity_data(ward_names)

    # Merge on ward name
    result = df.merge(equity_df, on=ward_name_col, how="left")

    # Also try to merge predictions if available
    if "risk_score" not in result.columns and "lst_predicted" in result.columns:
        # Already has predictions — nothing extra needed
        pass

    logger.info(
        f"Equity join complete: {len(result)} rows, "
        f"{len([c for c in result.columns if 'income' in c or 'green' in c or 'density' in c])} equity columns"
    )
    return result


# ── 3. Correlation analysis ────────────────────────────────────────────


def compute_correlations(df: pd.DataFrame) -> dict:
    """
    Compute Pearson correlations between heat metrics and equity metrics.

    Returns a dict of:
        {
            "lst_vs_income": {"r": ..., "p": ..., "interpretation": ...},
            "lst_vs_green_cover": {...},
            "risk_vs_income": {...},
            "risk_vs_green_cover": {...},
            "ndvi_vs_income": {...},
        }
    """
    pairs = [
        ("lst_mean", "median_income", "LST vs Income"),
        ("lst_mean", "green_cover_pct", "LST vs Green Cover"),
        ("risk_score", "median_income", "Risk Score vs Income"),
        ("risk_score", "green_cover_pct", "Risk Score vs Green Cover"),
        ("ndvi_mean", "median_income", "NDVI vs Income"),
    ]

    results = {}
    for col_x, col_y, label in pairs:
        if col_x not in df.columns or col_y not in df.columns:
            continue

        clean = df[[col_x, col_y]].dropna()
        if len(clean) < 3:
            results[label] = {"r": None, "p": None, "n": len(clean), "interpretation": "Insufficient data"}
            continue

        r, p = pearsonr(clean[col_x], clean[col_y])

        if abs(r) > 0.7:
            interp = "Strong"
        elif abs(r) > 0.4:
            interp = "Moderate"
        else:
            interp = "Weak"

        direction = "positive" if r > 0 else "negative"

        results[label] = {
            "r": round(r, 4),
            "p": round(p, 6),
            "n": len(clean),
            "direction": direction,
            "interpretation": f"{interp} {direction} correlation",
        }

        # Add policy-relevant interpretation
        if "Income" in label and r > 0:
            results[label]["policy_note"] = "Higher-income areas are hotter (unexpected — may indicate dense urban core)"
        elif "Income" in label and r < 0:
            results[label]["policy_note"] = "Lower-income areas are hotter — potential environmental justice concern"
        if "Green Cover" in label and r < 0:
            results[label]["policy_note"] = "More green cover = cooler temps, confirming UHI mitigation value of vegetation"

    return results


# ── 4. Heat Vulnerability Index (HVI) ──────────────────────────────────


def compute_heat_vulnerability_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a composite Heat Vulnerability Index per ward.

    HVI formula (each component normalised 0–1):
        HVI = 0.4 * LST_norm + 0.3 * (1 - green_cover_norm) + 0.3 * (1 - income_norm)
            = 0.4 * heat        + 0.3 * lack of greenery   + 0.3 * low income

    Higher HVI = more vulnerable (hotter + less green + lower income).
    Range: 0 (least vulnerable) to 1 (most vulnerable).
    """
    df = df.copy()

    required = ["lst_mean", "green_cover_pct", "median_income"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(f"Cannot compute HVI — missing columns: {missing}")
        df["heat_vulnerability_index"] = None
        return df

    clean = df.dropna(subset=required)
    if clean.empty:
        df["heat_vulnerability_index"] = None
        return df

    # Min-max normalise each component to [0, 1]
    for col in ["lst_mean", "green_cover_pct", "median_income"]:
        col_min = clean[col].min()
        col_max = clean[col].max()
        if col_max > col_min:
            clean[f"{col}_norm"] = (clean[col] - col_min) / (col_max - col_min)
        else:
            clean[f"{col}_norm"] = 0.5

    # HVI = weighted composite
    clean["heat_vulnerability_index"] = (
        0.4 * clean["lst_mean_norm"]
        + 0.3 * (1 - clean["green_cover_pct_norm"])
        + 0.3 * (1 - clean["median_income_norm"])
    )

    # Merge back
    df = df.merge(
        clean[["ward_name", "heat_vulnerability_index"]],
        on="ward_name", how="left", suffixes=("", "_new")
    )
    hvi_col = "heat_vulnerability_index_new" if "heat_vulnerability_index_new" in df.columns else "heat_vulnerability_index"
    if hvi_col.endswith("_new"):
        df["heat_vulnerability_index"] = df[hvi_col]
        df.drop(columns=[hvi_col], inplace=True)

    logger.info(
        f"HVI computed — range [{df['heat_vulnerability_index'].min():.3f}, "
        f"{df['heat_vulnerability_index'].max():.3f}]"
    )
    return df


# ── 5. Environmental justice hotspots ─────────────────────────────────


def identify_hotspots(
    df: pd.DataFrame,
    heat_threshold: float = 0.75,    # top 25% hottest
    income_threshold: float = 0.25,  # bottom 25% income
) -> pd.DataFrame:
    """
    Identify environmental justice hotspot wards.

    A ward is a hotspot if:
        - LST is in the top `heat_threshold` percentile (hottest)
        - AND median income is in the bottom `income_threshold` percentile (poorest)

    These wards suffer from double burden: extreme heat + low adaptive capacity.
    """
    df = df.copy()

    if "lst_mean" not in df.columns or "median_income" not in df.columns:
        logger.warning("Cannot identify hotspots — missing lst_mean or median_income")
        df["is_hotspot"] = False
        return df

    clean = df.dropna(subset=["lst_mean", "median_income"])
    if clean.empty:
        df["is_hotspot"] = False
        return df

    heat_cutoff = clean["lst_mean"].quantile(heat_threshold)
    income_cutoff = clean["median_income"].quantile(income_threshold)

    clean["is_hotspot"] = (clean["lst_mean"] >= heat_cutoff) & (clean["median_income"] <= income_cutoff)

    # Merge back
    df = df.merge(
        clean[["ward_name", "is_hotspot"]],
        on="ward_name", how="left", suffixes=("", "_new")
    )

    hotspot_count = clean["is_hotspot"].sum()
    if hotspot_count:
        hotspot_names = clean[clean["is_hotspot"]]["ward_name"].tolist()
        logger.info(
            f"Identified {hotspot_count} environmental justice hotspot(s): {hotspot_names}"
        )
    else:
        logger.info("No environmental justice hotspots identified")

    return df


# ── 6. Run full analysis ───────────────────────────────────────────────


def run_equity_analysis(
    joined_gdf: pd.DataFrame,
    equity_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Run the full equity analysis pipeline.

    Steps:
        1. Join equity data with heat data
        2. Compute correlations
        3. Compute HVI
        4. Identify hotspots
        5. Rank wards by vulnerability

    Returns a dict with all results.
    """
    # Step 1: Join
    df = load_and_join_equity(joined_gdf, equity_df)
    logger.info(f"Equity dataset: {len(df)} wards")

    # Step 2: Correlations
    correlations = compute_correlations(df)
    logger.info(f"Computed {len(correlations)} correlations")

    # Step 3: HVI
    df = compute_heat_vulnerability_index(df)

    # Step 4: Hotspots
    df = identify_hotspots(df)

    # Step 5: Vulnerability ranking
    ranking = _rank_by_vulnerability(df)

    return {
        "data": df,
        "correlations": correlations,
        "ranking": ranking,
        "hotspots": df[df["is_hotspot"] == True].to_dict("records") if "is_hotspot" in df.columns else [],
    }


def _rank_by_vulnerability(df: pd.DataFrame) -> pd.DataFrame:
    """Rank wards by HVI descending (most vulnerable first)."""
    if "heat_vulnerability_index" not in df.columns:
        return pd.DataFrame()

    ranking = df.dropna(subset=["heat_vulnerability_index"]).copy()
    ranking = ranking.sort_values("heat_vulnerability_index", ascending=False)

    cols = [c for c in
            ["ward_name", "lst_mean", "median_income", "green_cover_pct",
             "heat_vulnerability_index", "risk_category", "is_hotspot"]
            if c in ranking.columns]

    ranking = ranking[cols].reset_index(drop=True)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    return ranking


# ── 7. Store in database ──────────────────────────────────────────────


async def store_equity_metrics(
    db: AsyncSession,
    df: pd.DataFrame,
    metric_date: Optional[date] = None,
) -> list[EquityMetric]:
    """
    Write equity analysis results to the equity_metrics table.

    Args:
        db: Async DB session
        df: DataFrame with equity + HVI data (must include ward_id, ward_name)
        metric_date: Date for the metric record

    Returns:
        List of created EquityMetric ORM objects
    """
    if metric_date is None:
        metric_date = date.today()

    metrics = []
    for _, row in df.iterrows():
        if "ward_id" not in row or pd.isna(row.get("ward_id")):
            continue

        metric = EquityMetric(
            ward_id=int(row["ward_id"]),
            metric_date=metric_date,
            median_income=float(row["median_income"]) if pd.notna(row.get("median_income")) else None,
            green_cover_pct=float(row["green_cover_pct"]) if pd.notna(row.get("green_cover_pct")) else None,
            heat_vulnerability_index=float(row["heat_vulnerability_index"])
            if pd.notna(row.get("heat_vulnerability_index")) else None,
            population_density=float(row["population_density"]) if pd.notna(row.get("population_density")) else None,
        )
        db.add(metric)
        metrics.append(metric)

    await db.flush()
    logger.info(f"Stored {len(metrics)} equity metrics in database")
    return metrics
