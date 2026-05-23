import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Ward(Base):
    """
    City ward / administrative boundary.

    Stores ward polygons as GeoJSON text — no PostGIS required.
    """

    __tablename__ = "wards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    code: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    city: Mapped[str] = mapped_column(String(100), nullable=False, default="Bangalore", index=True)

    # Geometry stored as GeoJSON string (e.g. {"type":"MultiPolygon","coordinates":[[[...]]]})
    geometry: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        # Unique constraint: code is unique within a city
        UniqueConstraint("code", "city", name="uq_ward_code_city"),
    )

    area_sqkm: Mapped[float | None] = mapped_column(Float, nullable=True)
    population: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<Ward(id={self.id}, name='{self.name}', code='{self.code}')>"
