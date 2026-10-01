"""Prediction history and dashboard summary endpoints."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.session import get_db_session
from app.schemas.history import DashboardSummaryResponse, PredictionHistoryResponse
from app.services.prediction_history import get_dashboard_summary, list_recent_predictions


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/predictions", response_model=list[PredictionHistoryResponse])
def get_prediction_history(
    session: Annotated[Session, Depends(get_db_session)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[PredictionHistoryResponse]:
    """Return a bounded list of the newest stored predictions."""

    try:
        records = list_recent_predictions(session, limit)
    except SQLAlchemyError as error:
        logger.exception("Unable to read prediction history")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction history is temporarily unavailable.",
        ) from error
    return [PredictionHistoryResponse.model_validate(record) for record in records]


@router.get("/dashboard/summary", response_model=DashboardSummaryResponse)
def get_summary(
    session: Annotated[Session, Depends(get_db_session)],
) -> DashboardSummaryResponse:
    """Return aggregate scan metrics plus the ten most recent records."""

    try:
        summary = get_dashboard_summary(session)
    except SQLAlchemyError as error:
        logger.exception("Unable to calculate prediction dashboard summary")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dashboard data is temporarily unavailable.",
        ) from error

    return DashboardSummaryResponse(
        total_scans=summary.total_scans,
        average_confidence=summary.average_confidence,
        category_counts=summary.category_counts,
        uncertain_scans=summary.uncertain_scans,
        recent_predictions=[
            PredictionHistoryResponse.model_validate(record)
            for record in summary.recent_predictions
        ],
    )
