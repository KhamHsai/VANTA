"""Waste image prediction endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.config import Settings, get_settings
from app.ml.inference import ImageInferenceService, InvalidImageError, get_inference_service
from app.schemas.prediction import PredictionResponse


router = APIRouter()
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/predict", response_model=PredictionResponse)
async def predict_waste(
    file: Annotated[UploadFile, File(description="JPEG, PNG, or WebP waste image")],
    inference_service: Annotated[ImageInferenceService, Depends(get_inference_service)],
    settings: Annotated[Settings, Depends(get_settings)],
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

    return PredictionResponse(
        label=result.label,
        confidence=result.confidence,
        is_uncertain=result.is_uncertain,
        scores=result.scores,
    )
