"""Predict a VANTA waste category for one local image."""

import argparse
import json
from pathlib import Path

from vanta_training.prediction import ImagePredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pt"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    arguments = parser.parse_args()

    try:
        predictor = ImagePredictor(arguments.checkpoint, arguments.device)
        result = predictor.predict(arguments.image)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Prediction failed: {error}") from error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
