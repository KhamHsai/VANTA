import pytest
import torch

from vanta_training.evaluation import metrics_from_confusion_matrix


def test_metrics_are_calculated_from_confusion_matrix() -> None:
    confusion_matrix = torch.tensor(
        [
            [2, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 2, 0, 0],
            [0, 0, 0, 2, 0],
            [0, 0, 0, 0, 2],
        ]
    )

    results = metrics_from_confusion_matrix(confusion_matrix, 0.1, sample_count=10)

    assert results["accuracy"] == pytest.approx(0.9)
    assert results["evaluated_images"] == 10
    assert results["average_inference_latency_ms"] == pytest.approx(10.0)
    assert results["per_class"]["metal"]["recall"] == pytest.approx(0.5)
