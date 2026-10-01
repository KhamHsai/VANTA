"""Database model for persisted waste predictions."""

from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class PredictionHistory(Base):
    """Useful prediction metadata without retaining uploaded image data."""

    __tablename__ = "prediction_history"
    __table_args__ = (
        CheckConstraint(
            "predicted_label IN ('general', 'metal', 'organic', 'paper', 'plastic')",
            name="ck_prediction_history_label",
        ),
        CheckConstraint(
            "source_type IN ('camera', 'upload')",
            name="ck_prediction_history_source_type",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    predicted_label: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    is_uncertain: Mapped[bool] = mapped_column(Boolean, nullable=False)
    score_general: Mapped[float] = mapped_column(Float, nullable=False)
    score_metal: Mapped[float] = mapped_column(Float, nullable=False)
    score_organic: Mapped[float] = mapped_column(Float, nullable=False)
    score_paper: Mapped[float] = mapped_column(Float, nullable=False)
    score_plastic: Mapped[float] = mapped_column(Float, nullable=False)
    source_type: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.current_timestamp(),
        index=True,
    )
