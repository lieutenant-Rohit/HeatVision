import datetime

from sqlalchemy import Date, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class LSTReading(Base):
    """
    Land Surface Temperature reading for a ward.

    Each row stores the per-ward summary statistics computed from
    NASA MODIS raster data (e.g. MOD11A2 product).
    """

    __tablename__ = "lst_readings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ward_id: Mapped[int] = mapped_column(Integer, ForeignKey("wards.id", ondelete="CASCADE"), nullable=False, index=True)

    reading_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    lst_mean: Mapped[float] = mapped_column(Float, nullable=False, comment="Mean LST in Celsius")
    lst_min: Mapped[float] = mapped_column(Float, nullable=True)
    lst_max: Mapped[float] = mapped_column(Float, nullable=True)
    lst_std: Mapped[float] = mapped_column(Float, nullable=True)

    source: Mapped[str] = mapped_column(String(50), nullable=False, default="MODIS")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<LSTReading(ward_id={self.ward_id}, date={self.reading_date}, mean={self.lst_mean:.2f})>"
