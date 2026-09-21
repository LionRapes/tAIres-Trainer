import json
import logging
from pathlib import Path

from taires.schemas.training import TrainingTelemetry

logger = logging.getLogger(__name__)


def save_telemetry_json(
    telemetry: TrainingTelemetry,
    output_dir: Path,
    filename: str = "evaluation_telemetry.json",
) -> Path:
    """Serialize and write a TrainingTelemetry model to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target_file = output_dir / filename
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(telemetry.model_dump(), f, indent=2)

    logger.info("Evaluation telemetry successfully written to %s", target_file)
    return target_file


def load_telemetry_json(filepath: Path) -> TrainingTelemetry:
    """Read telemetry JSON file and reconstruct the Pydantic schema."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return TrainingTelemetry.model_validate(data)