"""Supervised classifier contract, metrics, thresholding, and safe bundle loading.

Joblib artifacts are executable Python objects and must only be loaded from a
trusted deployment channel.  The loader verifies the bundle's declared hashes
and feature schema before deserializing; this detects corruption or accidental
mix-ups, but is not a substitute for artifact signing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from backend.app.features.contract import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    feature_schema_manifest,
    validate_feature_frame,
)

BUNDLE_FORMAT_VERSION = "supervised-classifier-bundle/v1"
CLASSIFIER_FILENAME = "classifier.joblib"
FEATURE_SCHEMA_FILENAME = "feature_schema.json"
MANIFEST_FILENAME = "manifest.json"


class BundleValidationError(RuntimeError):
    """Raised when an artifact is incomplete, corrupt, or incompatible."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    """Convert numpy values and non-finite metric values to strict JSON values."""

    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def binary_classification_metrics(
    labels: np.ndarray | pd.Series | list[int],
    probabilities: np.ndarray | pd.Series | list[float],
    threshold: float,
    *,
    include_curves: bool = True,
) -> dict[str, Any]:
    """Evaluate binary probabilities without inventing single-class metrics."""

    y_true = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if y_true.ndim != 1 or scores.ndim != 1 or len(y_true) != len(scores):
        raise ValueError("labels and probabilities must be equal-length one-dimensional arrays")
    if len(y_true) == 0:
        raise ValueError("cannot evaluate an empty partition")
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("labels must contain only 0 and 1")
    if not np.isfinite(scores).all() or ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("probabilities must be finite and between 0 and 1")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    predicted = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    predicted_positive = tp + fp
    actual_positive = tp + fn
    actual_negative = tn + fp
    precision = float(tp / predicted_positive) if predicted_positive else None
    recall = float(tp / actual_positive) if actual_positive else None
    f1 = None
    if precision is not None and recall is not None:
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if precision + recall > 0.0
            else 0.0
        )
    fpr = float(fp / actual_negative) if actual_negative else None
    has_both_classes = actual_positive > 0 and actual_negative > 0
    metrics: dict[str, Any] = {
        "row_count": int(len(y_true)),
        "positive_count": int(actual_positive),
        "negative_count": int(actual_negative),
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": float(roc_auc_score(y_true, scores)) if has_both_classes else None,
        "pr_auc": float(average_precision_score(y_true, scores)) if has_both_classes else None,
        "false_positive_rate": fpr,
        "review_rate": float(predicted_positive / len(y_true)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "undefined_metrics": [],
    }
    undefined = [
        name
        for name in ("precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate")
        if metrics[name] is None
    ]
    metrics["undefined_metrics"] = undefined
    if include_curves:
        if has_both_classes:
            pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, scores)
            roc_fpr, roc_tpr, roc_thresholds = roc_curve(y_true, scores)
            metrics["precision_recall_curve"] = {
                "precision": pr_precision.tolist(),
                "recall": pr_recall.tolist(),
                "thresholds": pr_thresholds.tolist(),
            }
            metrics["roc_curve"] = {
                "false_positive_rate": roc_fpr.tolist(),
                "true_positive_rate": roc_tpr.tolist(),
                "thresholds": roc_thresholds.tolist(),
            }
        else:
            metrics["precision_recall_curve"] = None
            metrics["roc_curve"] = None
    return _json_value(metrics)


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    threshold: float
    metrics: dict[str, Any]
    table: tuple[dict[str, Any], ...]
    feasible: bool
    status: str


def tune_threshold(
    labels: np.ndarray | pd.Series | list[int],
    probabilities: np.ndarray | pd.Series | list[float],
    *,
    max_false_positive_rate: float | None = 0.10,
    max_review_rate: float | None = None,
) -> ThresholdSelection:
    """Maximize validation recall subject to FPR/review-capacity constraints."""

    for name, value in (
        ("max_false_positive_rate", max_false_positive_rate),
        ("max_review_rate", max_review_rate),
    ):
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    scores = np.asarray(probabilities, dtype=np.float64)
    candidates = sorted({0.0, 1.0, *(float(value) for value in scores)})
    rows: list[dict[str, Any]] = []
    for threshold in candidates:
        metrics = binary_classification_metrics(labels, scores, threshold, include_curves=False)
        fpr = metrics["false_positive_rate"]
        constraints_measurable = not (max_false_positive_rate is not None and fpr is None)
        feasible = constraints_measurable
        if max_false_positive_rate is not None and fpr is not None:
            feasible = feasible and fpr <= max_false_positive_rate + 1e-12
        if max_review_rate is not None:
            feasible = feasible and metrics["review_rate"] <= max_review_rate + 1e-12
        row = {**metrics, "feasible": bool(feasible)}
        rows.append(row)

    feasible_rows = [row for row in rows if row["feasible"]]
    positives_exist = any(int(value) == 1 for value in np.asarray(labels))

    def rank(row: dict[str, Any]) -> tuple[float, float, float, float]:
        # Recall dominates.  Among equally sensitive feasible operating points,
        # prefer the lower threshold: this spends only the explicitly allowed
        # false-positive/review budget and avoids selecting a brittle boundary
        # exactly on the weakest validation fraud score. Precision and review
        # burden then provide deterministic tie-breaks.
        return (
            -1.0 if row["recall"] is None else float(row["recall"]),
            -float(row["threshold"]),
            # With no predicted positives, precision is undefined. For
            # tie-breaking only, treat it like zero so a false-positive-only
            # queue is not preferred over an empty queue at equal recall.
            0.0 if row["precision"] is None else float(row["precision"]),
            -float(row["review_rate"]),
        )

    if feasible_rows:
        pool = feasible_rows
        feasible = True
        status = "constraints_satisfied"
    else:
        pool = rows
        feasible = False
        status = "constraints_not_measurable_or_unsatisfied"
    if not positives_exist:
        # Recall cannot guide selection. Choose the lowest review burden without
        # pretending that fraud sensitivity was observed.
        selected = max(pool, key=lambda row: (-float(row["review_rate"]), row["threshold"]))
        status = "validation_has_no_positive_labels"
    else:
        selected = max(pool, key=rank)
    return ThresholdSelection(
        threshold=float(selected["threshold"]),
        metrics=dict(selected),
        table=tuple(rows),
        feasible=feasible,
        status=status,
    )


@dataclass(slots=True)
class SupervisedModel:
    """Validated runtime wrapper around the selected sklearn-compatible model."""

    estimator: Any
    threshold: float
    model_name: str
    model_version: str
    manifest: dict[str, Any]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        validated = validate_feature_frame(frame)
        probabilities = np.asarray(self.estimator.predict_proba(validated), dtype=np.float64)
        if probabilities.shape != (len(validated), 2):
            raise BundleValidationError(
                f"classifier returned invalid probability shape {probabilities.shape}"
            )
        fraud = probabilities[:, 1]
        if not np.isfinite(fraud).all() or ((fraud < 0.0) | (fraud > 1.0)).any():
            raise BundleValidationError("classifier returned invalid probabilities")
        return fraud

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(frame) >= self.threshold).astype(np.int64)


def load_classifier_bundle(bundle_dir: str | Path) -> SupervisedModel:
    """Verify and load a compatible versioned classifier bundle."""

    root = Path(bundle_dir).resolve()
    if not root.is_dir():
        raise BundleValidationError(f"classifier bundle directory does not exist: {root}")
    manifest_path = root / MANIFEST_FILENAME
    schema_path = root / FEATURE_SCHEMA_FILENAME
    classifier_path = root / CLASSIFIER_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleValidationError("classifier manifest is missing or invalid") from exc
    if manifest.get("bundle_format_version") != BUNDLE_FORMAT_VERSION:
        raise BundleValidationError("unsupported classifier bundle format")
    if manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise BundleValidationError("classifier feature schema version is incompatible")
    if tuple(manifest.get("feature_names", ())) != FEATURE_NAMES:
        raise BundleValidationError("classifier feature order is incompatible")
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, dict):
        raise BundleValidationError("classifier manifest has no artifact hashes")
    for filename, path in (
        (CLASSIFIER_FILENAME, classifier_path),
        (FEATURE_SCHEMA_FILENAME, schema_path),
    ):
        expected = hashes.get(filename)
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or not path.is_file()
            or path.is_symlink()
        ):
            raise BundleValidationError(f"missing hash or artifact for {filename}")
        if sha256_file(path) != expected:
            raise BundleValidationError(f"artifact hash mismatch for {filename}")
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleValidationError("feature schema artifact is invalid") from exc
    if schema != feature_schema_manifest():
        raise BundleValidationError("feature schema artifact does not match runtime contract")
    try:
        payload = joblib.load(classifier_path)
    except Exception as exc:  # joblib surfaces several pickle/decompression errors
        raise BundleValidationError("classifier artifact could not be loaded") from exc
    if not isinstance(payload, dict):
        raise BundleValidationError("classifier artifact payload must be a mapping")
    required = {"estimator", "threshold", "model_name", "model_version", "feature_schema_version"}
    if not required.issubset(payload):
        raise BundleValidationError("classifier artifact payload is incomplete")
    if payload["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise BundleValidationError("classifier artifact schema metadata is incompatible")
    if payload["model_version"] != manifest.get("model_version"):
        raise BundleValidationError("classifier artifact version does not match manifest")
    threshold = float(payload["threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise BundleValidationError("classifier threshold is outside [0, 1]")
    estimator = payload["estimator"]
    if not callable(getattr(estimator, "predict_proba", None)):
        raise BundleValidationError("classifier does not expose predict_proba")
    return SupervisedModel(
        estimator=estimator,
        threshold=threshold,
        model_name=str(payload["model_name"]),
        model_version=str(payload["model_version"]),
        manifest=manifest,
    )
