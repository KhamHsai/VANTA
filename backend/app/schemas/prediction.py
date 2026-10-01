"""API schemas for waste image predictions."""

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """A five-class prediction and its confidence information."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    is_uncertain: bool
    scores: dict[str, float]
