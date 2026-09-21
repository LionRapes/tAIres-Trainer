import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from transformers import EvalPrediction

from taires.schemas.training import (
    CurvePoints,
    CurvesPayload,
    PrecisionRecallPoints,
    RawPredictionsPayload,
    TrainingHistoryPayload,
    TrainingTelemetry,
)

logger = logging.getLogger(__name__)


def safe_roc_auc(labels: NDArray[np.int_], probs: NDArray[np.float64]) -> float:
    if len(np.unique(labels)) < 2:
        return 0.0
    try:
        return float(roc_auc_score(labels, probs))
    except ValueError as e:
        logger.debug("Metric calculation failed: %s", e)
        return 0.0


def safe_pr_auc(labels: NDArray[np.int_], probs: NDArray[np.float64]) -> float:
    if len(np.unique(labels)) < 2:
        return 0.0
    try:
        return float(average_precision_score(labels, probs))
    except ValueError as e:
        logger.debug("Metric calculation failed: %s", e)
        return 0.0


def safe_log_loss(labels: NDArray[np.int_], probs: NDArray[np.float64]) -> float:
    if len(np.unique(labels)) < 2:
        return 0.0
    try:
        return float(log_loss(labels, probs, labels=[0, 1]))
    except ValueError as e:
        logger.debug("Metric calculation failed: %s", e)
        return 0.0


def find_optimal_threshold(labels: NDArray[np.int_], probs: NDArray[np.float64]) -> tuple[float, float]:
    if len(np.unique(labels)) < 2:
        return 0.5, 0.0

    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    
    f1_scores = np.divide(
        2 * (precisions * recalls),
        (precisions + recalls),
        out=np.zeros_like(precisions),
        where=(precisions + recalls) != 0
    )
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return round(float(best_thresh), 4), round(float(f1_scores[best_idx]), 4)


def to_serializable_dict(metrics: dict[str, Any]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for key, val in metrics.items():
        if isinstance(val, (np.floating, np.integer, int, float)):
            cleaned[key] = float(val)
    return cleaned


def compute_metrics(pred: EvalPrediction) -> dict[str, float]:
    labels: NDArray[np.int_] = np.asarray(pred.label_ids, dtype=np.int_)
    logits: NDArray[np.float32] = np.asarray(pred.predictions, dtype=np.float32)

    probs: NDArray[np.float64] = softmax(logits, axis=-1)[:, 1]
    preds: NDArray[np.int_] = np.argmax(logits, axis=-1)

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": safe_roc_auc(labels, probs),
        "pr_auc": safe_pr_auc(labels, probs),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "cohen_kappa": float(cohen_kappa_score(labels, preds)),
        "log_loss": safe_log_loss(labels, probs),
    }


def build_telemetry_payload(
    predictions_output: Any,
    log_history: list[dict[str, Any]],
) -> TrainingTelemetry:
    y_true: NDArray[np.int_] = np.asarray(predictions_output.label_ids, dtype=np.int_)
    logits: NDArray[np.float32] = np.asarray(predictions_output.predictions, dtype=np.float32)

    probs: NDArray[np.float64] = softmax(logits, axis=-1)[:, 1]
    y_pred: NDArray[np.int_] = np.argmax(logits, axis=-1)

    best_thresh, best_f1 = find_optimal_threshold(y_true, probs)

    if len(np.unique(y_true)) >= 2:
        fpr, tpr, roc_thresh = roc_curve(y_true, probs)
        prec, rec, pr_thresh = precision_recall_curve(y_true, probs)
    else:
        fpr, tpr, roc_thresh = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.5])
        prec, rec, pr_thresh = np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5])

    scalar_metrics = to_serializable_dict(predictions_output.metrics)
    scalar_metrics["optimal_threshold"] = best_thresh
    scalar_metrics["optimal_f1"] = best_f1

    return TrainingTelemetry(
        scalar_metrics=scalar_metrics,
        confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
        curves=CurvesPayload(
            roc=CurvePoints(
                fpr=[float(x) for x in fpr],
                tpr=[float(x) for x in tpr],
                thresholds=[float(x) for x in roc_thresh],
            ),
            precision_recall=PrecisionRecallPoints(
                precision=[float(x) for x in prec],
                recall=[float(x) for x in rec],
                thresholds=[float(x) for x in pr_thresh],
            ),
        ),
        raw_predictions=RawPredictionsPayload(
            true_labels=[int(x) for x in y_true],
            positive_probabilities=[float(x) for x in probs],
            predictions=[int(x) for x in y_pred],
        ),
        training_history=TrainingHistoryPayload(
            train=[e for e in log_history if "loss" in e],
            eval=[e for e in log_history if "eval_loss" in e],
            full_log=log_history,
        ),
    )