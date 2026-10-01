"""Persistence and aggregate queries for prediction history."""

from dataclasses import dataclass

from sqlalchemy import case, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models import PredictionHistory


CATEGORY_NAMES = ("general", "metal", "organic", "paper", "plastic")


@dataclass(frozen=True)
class DashboardSummaryData:
    total_scans: int
    average_confidence: float
    category_counts: dict[str, int]
    uncertain_scans: int
    recent_predictions: list[PredictionHistory]


def create_prediction_record(
    session: Session,
    *,
    predicted_label: str,
    confidence: float,
    is_uncertain: bool,
    scores: dict[str, float],
    source_type: str,
) -> PredictionHistory:
    """Persist one successful model result and return its database record."""

    record = PredictionHistory(
        predicted_label=predicted_label,
        confidence=confidence,
        is_uncertain=is_uncertain,
        score_general=scores["general"],
        score_metal=scores["metal"],
        score_organic=scores["organic"],
        score_paper=scores["paper"],
        score_plastic=scores["plastic"],
        source_type=source_type,
    )
    session.add(record)
    try:
        session.commit()
        session.refresh(record)
    except SQLAlchemyError:
        session.rollback()
        raise
    return record


def list_recent_predictions(session: Session, limit: int) -> list[PredictionHistory]:
    """Return recent prediction records in stable newest-first order."""

    statement = (
        select(PredictionHistory)
        .order_by(PredictionHistory.created_at.desc(), PredictionHistory.id.desc())
        .limit(limit)
    )
    return list(session.scalars(statement))


def get_dashboard_summary(session: Session, recent_limit: int = 10) -> DashboardSummaryData:
    """Calculate dashboard statistics in SQL and fetch only recent detail rows."""

    category_count_columns = [
        func.sum(case((PredictionHistory.predicted_label == category, 1), else_=0)).label(category)
        for category in CATEGORY_NAMES
    ]
    statement = select(
        func.count(PredictionHistory.id).label("total_scans"),
        func.coalesce(func.avg(PredictionHistory.confidence), 0.0).label(
            "average_confidence"
        ),
        func.sum(case((PredictionHistory.is_uncertain.is_(True), 1), else_=0)).label(
            "uncertain_scans"
        ),
        *category_count_columns,
    )
    row = session.execute(statement).one()
    row_values = row._mapping

    return DashboardSummaryData(
        total_scans=int(row_values["total_scans"] or 0),
        average_confidence=float(row_values["average_confidence"] or 0.0),
        category_counts={
            category: int(row_values[category] or 0) for category in CATEGORY_NAMES
        },
        uncertain_scans=int(row_values["uncertain_scans"] or 0),
        recent_predictions=list_recent_predictions(session, recent_limit),
    )
