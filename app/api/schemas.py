"""
Pydantic schemas for API request/response validation.

These ensure the API returns consistent, well-typed responses.
"""

from datetime import date
from typing import Any, Optional

from pydantic import BaseModel, Field


class WardOut(BaseModel):
    """Single ward in the API response."""
    id: int
    name: str
    code: str
    area_sqkm: Optional[float] = None
    population: Optional[int] = None


class HeatMapPoint(BaseModel):
    """A ward with its LST + NDVI measurements (returned as GeoJSON properties)."""
    ward_id: int
    ward_name: str
    ward_code: str
    reading_date: Optional[str] = None
    lst_mean: Optional[float] = None
    lst_min: Optional[float] = None
    lst_max: Optional[float] = None
    ndvi_mean: Optional[float] = None
    ndvi_min: Optional[float] = None
    ndvi_max: Optional[float] = None
    heat_ndvi_ratio: Optional[float] = None


class PredictionOut(BaseModel):
    """ML prediction result for a ward."""
    ward_id: int
    ward_name: str
    ward_code: str
    lst_predicted: Optional[float] = None
    risk_score: Optional[float] = Field(None, ge=0, le=1)
    risk_category: Optional[str] = None


class EquityOut(BaseModel):
    """Equity metrics for a ward."""
    ward_id: int
    ward_name: str
    ward_code: str
    median_income: Optional[float] = None
    green_cover_pct: Optional[float] = None
    heat_vulnerability_index: Optional[float] = None
    population_density: Optional[float] = None


class StatsOut(BaseModel):
    """Summary statistics for the dashboard."""
    total_wards: int
    avg_lst: Optional[float] = None
    min_lst: Optional[float] = None
    max_lst: Optional[float] = None
    avg_ndvi: Optional[float] = None
    high_risk_wards: int = 0
    medium_risk_wards: int = 0
    low_risk_wards: int = 0
    last_updated: Optional[str] = None


class GeoJSONFeature(BaseModel):
    """A single GeoJSON Feature."""
    type: str = "Feature"
    geometry: dict[str, Any]
    properties: dict[str, Any]


class GeoJSONFeatureCollection(BaseModel):
    """A GeoJSON FeatureCollection."""
    type: str = "FeatureCollection"
    features: list[GeoJSONFeature]
