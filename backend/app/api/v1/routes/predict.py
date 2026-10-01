"""Waste image prediction endpoint."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db_session
from app.ml.inference import ImageInferenceService, InvalidImageError, get_inference_service
from app.schemas.history import PredictionSource
from app.schemas.prediction import PredictionResponse
from app.services.prediction_history import create_prediction_record


logger = logging.getLogger(__name__)
router = APIRouter()
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/predict", response_model=PredictionResponse)
async def predict_waste(
    file: Annotated[UploadFile, File(description="JPEG, PNG, or WebP waste image")],
    inference_service: Annotated[ImageInferenceService, Depends(get_inference_service)],
    settings: Annotated[Settings, Depends(get_settings)],
    session: Annotated[Session, Depends(get_db_session)],
    source_type: Annotated[PredictionSource, Form()] = "upload",
) -> PredictionResponse:
    """Classify one uploaded waste image using the production checkpoint."""

    content_type = (file.content_type or "").partition(";")[0].strip().lower()
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported image type. Upload a JPEG, PNG, or WebP image.",
        )

    max_upload_bytes = settings.prediction_max_upload_bytes
    image_bytes = await file.read(max_upload_bytes + 1)
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image is empty.",
        )
    if len(image_bytes) > max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="The uploaded image exceeds the maximum allowed size.",
        )

    try:
        result = inference_service.predict_bytes(image_bytes)
    except InvalidImageError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    finally:
        await file.close()

    try:
        create_prediction_record(
            session,
            predicted_label=result.label,
            confidence=result.confidence,
            is_uncertain=result.is_uncertain,
            scores=result.scores,
            source_type=source_type,
        )
    except SQLAlchemyError as error:
        logger.exception("Prediction completed but its history record could not be saved")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction succeeded, but VANTA could not save the result. Please try again.",
        ) from error

    return PredictionResponse(
        label=result.label,
        confidence=result.confidence,
        is_uncertain=result.is_uncertain,
        scores=result.scores,
    )
