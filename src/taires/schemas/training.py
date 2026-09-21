from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TrainConfig(BaseModel):
    """Training pipeline and hyperparameters configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data_path: Path = Field()
    train_data_path: Path = Field(default=Path("train_dataset.csv"))
    val_data_path: Path = Field(default=Path("val_dataset.csv"))
    model_name: str = Field(default="DeepPavlov/rubert-base-cased")
    output_dir: Path = Field(default=Path("./model"))
    checkpoints_dir: Path = Field(default=Path("./checkpoints"))
    test_size: float = Field(default=0.15, gt=0.0, lt=1.0)
    random_state: int = Field(default=42, ge=0)
    num_train_epochs: int = Field(default=5, ge=1)
    per_device_train_batch_size: int = Field(default=16, ge=1)
    per_device_eval_batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=2e-5, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    max_length: int = Field(default=64, ge=16, le=512)
    logging_steps: int = Field(default=50, ge=1)


class CurvePoints(BaseModel):
    """Coordinates payload for ROC curve."""

    fpr: list[float]
    tpr: list[float]
    thresholds: list[float]


class PrecisionRecallPoints(BaseModel):
    """Coordinates payload for Precision-Recall curve."""

    precision: list[float]
    recall: list[float]
    thresholds: list[float]


class CurvesPayload(BaseModel):
    roc: CurvePoints
    precision_recall: PrecisionRecallPoints


class RawPredictionsPayload(BaseModel):
    true_labels: list[int]
    positive_probabilities: list[float]
    predictions: list[int]


class TrainingHistoryPayload(BaseModel):
    train: list[dict[str, Any]]
    eval: list[dict[str, Any]]
    full_log: list[dict[str, Any]]


class TrainingTelemetry(BaseModel):
    """Full telemetry payload for metrics and visualizations."""

    scalar_metrics: dict[str, float]
    confusion_matrix: list[list[int]]
    curves: CurvesPayload
    raw_predictions: RawPredictionsPayload
    training_history: TrainingHistoryPayload
    

class TestSampleResult(BaseModel):
    text1: str
    text2: str
    expected_match: bool
    predicted_match: bool
    confidence: float
    passed: bool


class AutomatedTestsReport(BaseModel):
    total_tests: int
    passed_tests: int
    accuracy: float
    results: list[TestSampleResult]


class ModelVersionMetadata(BaseModel):
    name: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    base_model_name: str
    decision_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    metrics: dict[str, float]
    artifacts: dict[str, str]