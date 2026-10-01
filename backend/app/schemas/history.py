"""Schemas for prediction history and dashboard statistics."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer


WasteCategory = Literal["general", "metal", "organic", "paper", "plastic"]
PredictionSource = Literal["camera", "upload"]


class PredictionHistoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    predicted_label: WasteCategory
    confidence: float = Field(ge=0.0, le=1.0)
    is_uncertain: bool
    score_general: float = Field(ge=0.0, le=1.0)
    score_metal: float = Field(ge=0.0, le=1.0)
    score_organic: float = Field(ge=0.0, le=1.0)
    score_paper: float = Field(ge=0.0, le=1.0)
    score_plastic: float = Field(ge=0.0, le=1.0)
    source_type: PredictionSource
    created_at: datetime

    @field_serializer("created_at")
    def serialize_created_at(self, created_at: datetime) -> str:
        """Treat MySQL's timezone-naive timestamps as UTC in API responses."""

        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        return created_at.astimezone(UTC).isoformat().replace("+00:00", "Z")


class CategoryCounts(BaseModel):
    general: int = Field(ge=0)
    metal: int = Field(ge=0)
    organic: int = Field(ge=0)
    paper: int = Field(ge=0)
    plastic: int = Field(ge=0)


class DashboardSummaryResponse(BaseModel):
    total_scans: int = Field(ge=0)
    average_confidence: float = Field(ge=0.0, le=1.0)
    category_counts: CategoryCounts
    uncertain_scans: int = Field(ge=0)
    recent_predictions: list[PredictionHistoryResponse]
