import datetime

from sqlalchemy import Date, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Prediction(Base):
    """
    ML model prediction for a ward.

    Stores the heat-risk score produced by the scikit-learn model
    so the API can serve it as GeoJSON without re-running inference.
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ward_id: Mapped[int] = mapped_column(Integer, ForeignKey("wards.id", ondelete="CASCADE"), nullable=False, index=True)

    prediction_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False, comment="0 = low risk, 1 = high risk")
    risk_category: Mapped[str] = mapped_column(String(20), nullable=False, comment="low / medium / high")

    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    features_used: Mapped[dict | None] = mapped_column(
        # JSON column — stores which features were used and their values
        type_=String,
        nullable=True,
        comment="JSON dict of feature_name -> value",
    )

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<Prediction(ward_id={self.ward_id}, score={self.risk_score:.3f}, cat='{self.risk_category}')>"
