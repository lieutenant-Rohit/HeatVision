import datetime

from sqlalchemy import Date, DateTime, Float, ForeignKey, Integer, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class EquityMetric(Base):
    """
    Equity analysis metrics per ward.

    Correlates heat vulnerability with socio-economic factors
    (income, green cover) to identify environmental justice hotspots.
    """

    __tablename__ = "equity_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ward_id: Mapped[int] = mapped_column(Integer, ForeignKey("wards.id", ondelete="CASCADE"), nullable=False, index=True)

    metric_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)

    median_income: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Median household income (USD)"
    )
    green_cover_pct: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Percentage of ward area covered by vegetation"
    )
    heat_vulnerability_index: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Composite index (higher = more vulnerable)"
    )
    population_density: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="People per sq km"
    )

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<EquityMetric(ward_id={self.ward_id}, date={self.metric_date}, hvi={self.heat_vulnerability_index:.3f})>"
