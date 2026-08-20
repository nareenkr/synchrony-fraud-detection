"""Train, select, evaluate, and persist the supervised fraud classifier.

Examples:
    python -m training.train_classifier
    python -m training.train_classifier --max-fpr 0.05 --max-review-rate 0.20
"""

from __future__ import annotations

import argparse
import json
import platform
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from backend.app.features.contract import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    feature_schema_manifest,
)
from backend.app.fraud.supervised import (
    BUNDLE_FORMAT_VERSION,
    CLASSIFIER_FILENAME,
    FEATURE_SCHEMA_FILENAME,
    MANIFEST_FILENAME,
    ThresholdSelection,
    binary_classification_metrics,
    sha256_file,
    tune_threshold,
)
from training.prepare_data import DEFAULT_SEED, DatasetSplits
from training.replay import ModelMatrixSplits, entity_overlap_report, replay_splits

DEFAULT_BUNDLE_ID = "supervised-v1"
_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


@dataclass(frozen=True, slots=True)
class CandidateResult:
    name: str
    estimator: Any
    threshold: ThresholdSelection
    validation_metrics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TrainingResult:
    selected: CandidateResult
    candidates: tuple[CandidateResult, ...]
    test_metrics: dict[str, Any]
    feature_importance: tuple[dict[str, float | str], ...]


def load_prepared_splits(data_dir: str | Path) -> DatasetSplits:
    root = Path(data_dir)
    frames: list[pd.DataFrame] = []
    for partition in ("train", "validation", "test"):
        path = root / f"{partition}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"prepared partition does not exist: {path}")
        frame = pd.read_csv(path)
        if "fraud_label" not in frame:
            raise ValueError(f"prepared partition {partition!r} has no fraud_label column")
        frames.append(frame)
    return DatasetSplits(*frames)


def _validate_training_labels(labels: pd.Series) -> None:
    unique = set(labels.astype(int).unique())
    if not unique.issubset({0, 1}):
        raise ValueError("training labels must contain only 0 and 1")
    if unique != {0, 1}:
        raise ValueError(
            "training partition must contain both fraud and non-fraud examples; "
            "validation/test partitions may be single-class"
        )


def build_candidate_estimators(labels: pd.Series, *, seed: int) -> dict[str, Any]:
    """Return deterministic, imbalance-aware baseline and challenger models."""

    _validate_training_labels(labels)
    positive = int(labels.sum())
    negative = int(len(labels) - positive)
    imbalance_ratio = negative / positive
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2_000,
                        random_state=seed,
                        solver="liblinear",
                    ),
                ),
            ]
        ),
        "xgboost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            min_child_weight=2,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            scale_pos_weight=imbalance_ratio,
            random_state=seed,
            seed=seed,
            n_jobs=1,
            tree_method="hist",
        ),
    }


def _selection_key(candidate: CandidateResult) -> tuple[float, ...]:
    metrics = candidate.validation_metrics
    recall = metrics["recall"]
    pr_auc = metrics["pr_auc"]
    precision = metrics["precision"]
    fpr = metrics["false_positive_rate"]
    return (
        float(candidate.threshold.feasible),
        -1.0 if recall is None else float(recall),
        -1.0 if pr_auc is None else float(pr_auc),
        -1.0 if precision is None else float(precision),
        -1.0 if fpr is None else -float(fpr),
        float(candidate.name == "logistic_regression"),
    )


def _feature_importance(estimator: Any, model_name: str) -> tuple[dict[str, float | str], ...]:
    if model_name == "logistic_regression":
        values = np.abs(estimator.named_steps["classifier"].coef_[0])
    else:
        values = np.asarray(estimator.feature_importances_, dtype=np.float64)
    order = np.argsort(values)[::-1]
    return tuple(
        {"feature": FEATURE_NAMES[int(index)], "importance": float(values[int(index)])}
        for index in order
    )


def train_supervised_models(
    matrices: ModelMatrixSplits,
    *,
    seed: int = DEFAULT_SEED,
    max_false_positive_rate: float | None = 0.10,
    max_review_rate: float | None = None,
) -> TrainingResult:
    """Fit on train, tune/select on validation, then evaluate test exactly once."""

    _validate_training_labels(matrices.train.labels)
    candidates: list[CandidateResult] = []
    for name, estimator in build_candidate_estimators(matrices.train.labels, seed=seed).items():
        estimator.fit(matrices.train.features, matrices.train.labels)
        validation_probabilities = estimator.predict_proba(matrices.validation.features)[:, 1]
        threshold = tune_threshold(
            matrices.validation.labels,
            validation_probabilities,
            max_false_positive_rate=max_false_positive_rate,
            max_review_rate=max_review_rate,
        )
        validation_metrics = binary_classification_metrics(
            matrices.validation.labels,
            validation_probabilities,
            threshold.threshold,
        )
        candidates.append(
            CandidateResult(
                name=name,
                estimator=estimator,
                threshold=threshold,
                validation_metrics=validation_metrics,
            )
        )
    selected = max(candidates, key=_selection_key)
    test_probabilities = selected.estimator.predict_proba(matrices.test.features)[:, 1]
    test_metrics = binary_classification_metrics(
        matrices.test.labels,
        test_probabilities,
        selected.threshold.threshold,
    )
    return TrainingResult(
        selected=selected,
        candidates=tuple(candidates),
        test_metrics=test_metrics,
        feature_importance=_feature_importance(selected.estimator, selected.name),
    )


def _candidate_manifest(candidate: CandidateResult) -> dict[str, Any]:
    return {
        "model_name": candidate.name,
        "selected_threshold": candidate.threshold.threshold,
        "threshold_status": candidate.threshold.status,
        "constraints_satisfied": candidate.threshold.feasible,
        "validation_metrics": candidate.validation_metrics,
        "threshold_analysis": list(candidate.threshold.table),
    }


def persist_classifier_bundle(
    result: TrainingResult,
    output_dir: str | Path,
    *,
    model_version: str,
    seed: int,
    dataset_manifest: dict[str, Any],
    prepared_manifest_sha256: str | None,
    overlap_audit: dict[str, Any],
    max_false_positive_rate: float | None,
    max_review_rate: float | None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Persist one self-describing, hash-verified classifier bundle."""

    if not _VERSION_PATTERN.fullmatch(model_version):
        raise ValueError("model_version must be 1-80 URL/path-safe characters")
    root = Path(output_dir)
    if root.exists():
        if not root.is_dir():
            raise FileExistsError(f"bundle path is not a directory: {root}")
        if not overwrite:
            raise FileExistsError(f"bundle already exists: {root}; choose a new version")
        # Overwrite only files owned by this bundle writer. Unknown files in the
        # directory are preserved; broad recursive deletion is neither needed
        # nor appropriate for an artifact refresh.
        for filename in (CLASSIFIER_FILENAME, FEATURE_SCHEMA_FILENAME, MANIFEST_FILENAME):
            artifact = root / filename
            if artifact.exists():
                if not artifact.is_file() or artifact.is_symlink():
                    raise FileExistsError(f"refusing to overwrite unsafe artifact path: {artifact}")
                artifact.unlink()
    root.mkdir(parents=True, exist_ok=True)
    classifier_path = root / CLASSIFIER_FILENAME
    schema_path = root / FEATURE_SCHEMA_FILENAME
    payload = {
        "estimator": result.selected.estimator,
        "threshold": result.selected.threshold.threshold,
        "model_name": result.selected.name,
        "model_version": model_version,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": FEATURE_NAMES,
    }
    joblib.dump(payload, classifier_path, compress=3)
    schema_path.write_text(
        json.dumps(feature_schema_manifest(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest: dict[str, Any] = {
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "model_version": model_version,
        "model_name": result.selected.name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "seed": seed,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "selected_threshold": result.selected.threshold.threshold,
        "operating_constraints": {
            "max_false_positive_rate": max_false_positive_rate,
            "max_review_rate": max_review_rate,
        },
        "selection_policy": (
            "model selection uses validation constraint feasibility, recall, PR-AUC, precision, "
            "and lower FPR; threshold selection maximizes recall and then chooses the lowest "
            "feasible boundary for temporal robustness; test is not used for tuning or selection"
        ),
        "selection_status": result.selected.threshold.status,
        "candidate_evaluation": [_candidate_manifest(candidate) for candidate in result.candidates],
        "test_evaluation": result.test_metrics,
        "feature_importance": list(result.feature_importance),
        "dataset": {
            "manifest": dataset_manifest,
            "manifest_sha256": prepared_manifest_sha256,
            "entity_overlap_counts": overlap_audit,
        },
        "dependencies": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
        "artifact_hashes": {
            CLASSIFIER_FILENAME: sha256_file(classifier_path),
            FEATURE_SCHEMA_FILENAME: sha256_file(schema_path),
        },
    }
    (root / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def train_from_directory(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    model_version: str = DEFAULT_BUNDLE_ID,
    seed: int = DEFAULT_SEED,
    max_false_positive_rate: float | None = 0.10,
    max_review_rate: float | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    data_root = Path(data_dir)
    splits = load_prepared_splits(data_root)
    matrices = replay_splits(splits)
    result = train_supervised_models(
        matrices,
        seed=seed,
        max_false_positive_rate=max_false_positive_rate,
        max_review_rate=max_review_rate,
    )
    dataset_manifest_path = data_root / "manifest.json"
    if dataset_manifest_path.is_file():
        dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        dataset_manifest_hash = sha256_file(dataset_manifest_path)
    else:
        dataset_manifest = {}
        dataset_manifest_hash = None
    return persist_classifier_bundle(
        result,
        output_dir,
        model_version=model_version,
        seed=seed,
        dataset_manifest=dataset_manifest,
        prepared_manifest_sha256=dataset_manifest_hash,
        overlap_audit=entity_overlap_report(splits),
        max_false_positive_rate=max_false_positive_rate,
        max_review_rate=max_review_rate,
        overwrite=overwrite,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/supervised-v1"))
    parser.add_argument("--model-version", default=DEFAULT_BUNDLE_ID)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-fpr", type=float, default=0.10)
    parser.add_argument("--max-review-rate", type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    manifest = train_from_directory(
        args.data_dir,
        args.output_dir,
        model_version=args.model_version,
        seed=args.seed,
        max_false_positive_rate=args.max_fpr,
        max_review_rate=args.max_review_rate,
        overwrite=args.overwrite,
    )
    summary = {
        "bundle": str(args.output_dir),
        "model_version": manifest["model_version"],
        "selected_model": manifest["model_name"],
        "selected_threshold": manifest["selected_threshold"],
        "test_metrics": {
            key: manifest["test_evaluation"][key]
            for key in ("precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate")
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
