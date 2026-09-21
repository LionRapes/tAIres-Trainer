import argparse
import json
import logging
from pathlib import Path

from taires.ml.tester import run_automated_tests
from taires.ml.train import train
from taires.schemas.training import ModelVersionMetadata, TrainConfig
from taires.utils.io import save_telemetry_json

logger = logging.getLogger(__name__)


def create_model_version(
    config: TrainConfig,
    name: str,
    version: str,
    description: str,
    versions_root: str | Path = "versions",
) -> Path:
    csv_file = Path(config.data_path)
    if not csv_file.is_file():
        raise FileNotFoundError(f"Dataset CSV file not found: {csv_file.resolve()}")

    bundle_name = f"{name.strip()}-{version.strip()}"
    version_dir = Path(versions_root) / bundle_name

    if version_dir.exists():
        raise FileExistsError(f"Version directory '{version_dir.resolve()}' already exists.")

    version_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Initializing model bundle at: %s", version_dir.resolve())

    config.output_dir = version_dir / "model"
    config.checkpoints_dir = version_dir / "checkpoints"
    config.val_data_path = version_dir / "val_dataset.csv"
    config.train_data_path = version_dir / "train_dataset.csv"

    telemetry = train(config)

    save_telemetry_json(telemetry, output_dir=version_dir, filename="telemetry.json")

    optimal_thresh = telemetry.scalar_metrics.get("optimal_threshold", 0.5)
    logger.info("Applying calculated decision threshold: %.4f", optimal_thresh)

    test_report = run_automated_tests(
        model_dir=config.output_dir,
        dataset_path=config.val_data_path,
        samples_per_class=25,
        threshold=optimal_thresh,
    )
    with open(version_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(test_report.model_dump(), f, indent=2)

    metadata = ModelVersionMetadata(
        name=name,
        version=version,
        description=description,
        base_model_name=config.model_name,
        decision_threshold=optimal_thresh,
        metrics=telemetry.scalar_metrics,
        artifacts={
            "source_dataset": str(csv_file.resolve()),
            "train_dataset": "train_dataset.csv",
            "val_dataset": "val_dataset.csv",
            "model_dir": "model",
            "telemetry": "telemetry.json",
            "tests": "test_results.json",
        },
    )

    with open(version_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata.model_dump(mode="json"), f, indent=2)

    logger.info("Successfully packaged model %s. Directory: %s", bundle_name, version_dir.resolve())
    return version_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and package a versioned model bundle.")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--description", type=str, required=True)
    parser.add_argument("--dataset-path", dest="data_path", type=Path, required=True)
    parser.add_argument("--base-model", dest="model_name", type=str, default="DeepPavlov/rubert-base-cased")
    parser.add_argument("--epochs", dest="num_train_epochs", type=int, default=5)
    parser.add_argument("--batch-size", dest="per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--eval-batch-size", dest="per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=2e-5)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    train_config = TrainConfig.model_validate(args_dict)
    create_model_version(
        config=train_config,
        name=args.name,
        version=args.version,
        description=args.description,
    )