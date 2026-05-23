import datetime

from sqlalchemy import Date, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class NDVIValue(Base):
    """
    Normalized Difference Vegetation Index for a ward.

    NDVI indicates vegetation health/density (range -1 to +1).
    Higher values = more green cover → cooling effect.
    """

    __tablename__ = "ndvi_values"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ward_id: Mapped[int] = mapped_column(Integer, ForeignKey("wards.id", ondelete="CASCADE"), nullable=False, index=True)

    reading_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    ndvi_mean: Mapped[float] = mapped_column(Float, nullable=False)
    ndvi_min: Mapped[float] = mapped_column(Float, nullable=True)
    ndvi_max: Mapped[float] = mapped_column(Float, nullable=True)
    ndvi_std: Mapped[float] = mapped_column(Float, nullable=True)

    source: Mapped[str] = mapped_column(String(50), nullable=False, default="MODIS")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<NDVIValue(ward_id={self.ward_id}, date={self.reading_date}, mean={self.ndvi_mean:.3f})>"
