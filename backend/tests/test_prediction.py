"""Integration tests for production model loading and image prediction."""

from __future__ import annotations

from io import BytesIO

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from torch import nn

from app.core.config import Settings, get_settings
from app.db.session import get_db_session
from app.main import app
from app.ml.inference import EXPECTED_CLASSES, ImageInferenceService, get_inference_service
from app.models import PredictionHistory


def create_test_image(image_format: str = "JPEG") -> bytes:
    image = Image.new("RGB", (256, 256), color=(35, 140, 75))
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def test_real_checkpoint_mapping_and_output_dimensions(
    inference_service: ImageInferenceService,
) -> None:
    output_layer = inference_service.model.classifier[-1]  # type: ignore[attr-defined]

    assert inference_service.class_names == EXPECTED_CLASSES
    assert inference_service.metadata["architecture"] == "mobilenet_v3_small"
    assert isinstance(output_layer, nn.Linear)
    assert output_layer.out_features == 5
    assert not inference_service.model.training
    assert get_inference_service() is inference_service


def test_valid_image_returns_ordered_probability_scores(
    client: TestClient,
    db_session: Session,
) -> None:
    response = client.post(
        "/api/v1/predict",
        files={"file": ("waste.jpg", create_test_image(), "image/jpeg")},
        data={"source_type": "camera"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] in EXPECTED_CLASSES
    assert 0.0 <= payload["confidence"] <= 1.0
    assert isinstance(payload["is_uncertain"], bool)
    assert tuple(payload["scores"]) == EXPECTED_CLASSES
    assert all(0.0 <= score <= 1.0 for score in payload["scores"].values())
    assert sum(payload["scores"].values()) == pytest.approx(1.0, abs=1e-5)
    assert payload["confidence"] == pytest.approx(max(payload["scores"].values()))

    saved_record = db_session.scalar(select(PredictionHistory))
    assert saved_record is not None
    assert saved_record.predicted_label == payload["label"]
    assert saved_record.source_type == "camera"
    assert saved_record.confidence == pytest.approx(payload["confidence"])


def test_confidence_threshold_marks_low_confidence_prediction(
    inference_service: ImageInferenceService,
) -> None:
    original_threshold = inference_service.confidence_threshold
    try:
        inference_service.confidence_threshold = 1.0
        result = inference_service.predict_bytes(create_test_image("PNG"))
    finally:
        inference_service.confidence_threshold = original_threshold

    assert result.confidence < 1.0
    assert result.is_uncertain is True


def test_rejects_empty_upload(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "The uploaded image is empty."


def test_rejects_unsupported_file_type(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415


def test_rejects_corrupt_image(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predict",
        files={"file": ("broken.png", b"not valid png data", "image/png")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "The uploaded file is not a valid image."


def test_rejects_oversized_upload(
    client: TestClient,
) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        prediction_max_upload_bytes=8
    )
    try:
        response = client.post(
            "/api/v1/predict",
            files={"file": ("large.webp", b"123456789", "image/webp")},
        )
    finally:
        app.dependency_overrides.pop(get_settings, None)

    assert response.status_code == 413


def test_model_runs_in_inference_mode(inference_service: ImageInferenceService) -> None:
    gradient_states: list[bool] = []
    hook = inference_service.model.register_forward_hook(
        lambda _model, _inputs, _output: gradient_states.append(torch.is_grad_enabled())
    )
    try:
        result = inference_service.predict_bytes(create_test_image("WEBP"))
    finally:
        hook.remove()

    assert gradient_states == [False]
    assert result.label in EXPECTED_CLASSES
    assert result.inference_time_ms > 0.0


def test_prediction_returns_safe_error_when_history_cannot_be_saved(
    client: TestClient,
    db_session: Session,
) -> None:
    class FailingSession:
        def add(self, _record: object) -> None:
            pass

        def commit(self) -> None:
            raise SQLAlchemyError("database unavailable")

        def rollback(self) -> None:
            pass

    app.dependency_overrides[get_db_session] = FailingSession
    try:
        response = client.post(
            "/api/v1/predict",
            files={"file": ("waste.jpg", create_test_image(), "image/jpeg")},
            data={"source_type": "upload"},
        )
    finally:
        app.dependency_overrides[get_db_session] = lambda: db_session

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Prediction succeeded, but VANTA could not save the result. Please try again."
    )
