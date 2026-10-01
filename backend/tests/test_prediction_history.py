"""Tests for prediction persistence, history, and dashboard aggregates."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models import PredictionHistory
from app.services.prediction_history import create_prediction_record, list_recent_predictions


def create_record(
    session: Session,
    *,
    label: str,
    confidence: float,
    uncertain: bool = False,
    source_type: str = "upload",
) -> None:
    remaining_score = (1.0 - confidence) / 4
    scores = {
        category: remaining_score
        for category in ("general", "metal", "organic", "paper", "plastic")
    }
    scores[label] = confidence
    create_prediction_record(
        session,
        predicted_label=label,
        confidence=confidence,
        is_uncertain=uncertain,
        scores=scores,
        source_type=source_type,
    )


def test_prediction_record_can_be_created(db_session: Session) -> None:
    create_record(
        db_session,
        label="plastic",
        confidence=0.92,
        source_type="camera",
    )

    response = get_history(db_session)
    assert len(response) == 1
    assert response[0].predicted_label == "plastic"
    assert response[0].source_type == "camera"
    assert response[0].score_plastic == 0.92


def test_history_endpoint_returns_newest_records_with_limit(
    client: TestClient,
    db_session: Session,
) -> None:
    create_record(db_session, label="general", confidence=0.70)
    create_record(db_session, label="paper", confidence=0.80)
    create_record(db_session, label="metal", confidence=0.90, source_type="camera")

    response = client.get("/api/v1/predictions", params={"limit": 2})

    assert response.status_code == 200
    payload = response.json()
    assert [record["predicted_label"] for record in payload] == ["metal", "paper"]
    assert payload[0]["source_type"] == "camera"
    assert payload[0]["created_at"].endswith("Z")


def test_dashboard_summary_returns_empty_state(client: TestClient) -> None:
    response = client.get("/api/v1/dashboard/summary")

    assert response.status_code == 200
    assert response.json() == {
        "total_scans": 0,
        "average_confidence": 0.0,
        "category_counts": {
            "general": 0,
            "metal": 0,
            "organic": 0,
            "paper": 0,
            "plastic": 0,
        },
        "uncertain_scans": 0,
        "recent_predictions": [],
    }


def test_dashboard_summary_uses_correct_aggregates(
    client: TestClient,
    db_session: Session,
) -> None:
    create_record(db_session, label="paper", confidence=0.90)
    create_record(db_session, label="paper", confidence=0.80, uncertain=True)
    create_record(db_session, label="plastic", confidence=0.70, source_type="camera")

    response = client.get("/api/v1/dashboard/summary")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_scans"] == 3
    assert payload["average_confidence"] == pytest.approx(0.8)
    assert payload["uncertain_scans"] == 1
    assert payload["category_counts"] == {
        "general": 0,
        "metal": 0,
        "organic": 0,
        "paper": 2,
        "plastic": 1,
    }
    assert [record["predicted_label"] for record in payload["recent_predictions"]] == [
        "plastic",
        "paper",
        "paper",
    ]


def get_history(session: Session) -> list[PredictionHistory]:
    return list_recent_predictions(session, limit=20)
