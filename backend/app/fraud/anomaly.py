"""Deterministic Isolation Forest anomaly scoring and artifact persistence."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.utils.validation import check_is_fitted

from backend.app.features.contract import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    validate_feature_frame,
    validate_feature_vector,
)

ANOMALY_ARTIFACT_VERSION = 1
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MAX_ARTIFACT_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class AnomalyConfig:
    """Centralized, reproducible Isolation Forest configuration."""

    n_estimators: int = 200
    max_samples: int = 256
    contamination: str | float = "auto"
    random_state: int = 20260819
    quantile_range: tuple[float, float] = (10.0, 90.0)

    def __post_init__(self) -> None:
        if self.n_estimators <= 0 or self.max_samples <= 1:
            raise ValueError("n_estimators must be positive and max_samples must exceed one")
        if self.contamination != "auto" and not (
            isinstance(self.contamination, (int, float)) and 0 < float(self.contamination) <= 0.5
        ):
            raise ValueError("contamination must be 'auto' or a number in (0, 0.5]")
        low, high = self.quantile_range
        if not 0 <= low < high <= 100:
            raise ValueError("quantile_range must be ordered within [0, 100]")


DEFAULT_ANOMALY_CONFIG = AnomalyConfig()


@dataclass(frozen=True, slots=True)
class EmpiricalQuantileCalibrator:
    """Monotonic empirical-CDF calibration fitted only on training scores."""

    reference_scores: tuple[float, ...]

    @classmethod
    def fit(cls, raw_scores: Sequence[float] | np.ndarray) -> Self:
        values = np.asarray(raw_scores, dtype=np.float64)
        if values.ndim != 1 or values.size < 2:
            raise ValueError("calibration requires at least two one-dimensional scores")
        if not np.isfinite(values).all():
            raise ValueError("calibration scores must be finite")
        return cls(tuple(float(value) for value in np.sort(values, kind="stable")))

    def transform(self, raw_scores: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(raw_scores, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("raw anomaly scores must be a finite one-dimensional array")
        reference = np.asarray(self.reference_scores, dtype=np.float64)
        if reference.ndim != 1 or reference.size < 2 or not np.isfinite(reference).all():
            raise ValueError("calibrator reference distribution is invalid")
        # Average ranks give tied training observations the same risk.  The
        # Mid-rank plotting positions avoid declaring an observed training
        # point absolutely anomalous while unseen values beyond the tail reach 1.
        left = np.searchsorted(reference, values, side="left")
        right = np.searchsorted(reference, values, side="right")
        ranks = (left + right) / 2.0
        return np.clip(ranks / reference.size, 0.0, 1.0)


def _as_feature_frame(
    features: pd.DataFrame | pd.Series | Sequence[float] | np.ndarray,
) -> pd.DataFrame:
    if isinstance(features, pd.DataFrame):
        return validate_feature_frame(features)
    if isinstance(features, pd.Series):
        missing = [name for name in FEATURE_NAMES if name not in features]
        if missing:
            raise ValueError(f"missing canonical features: {', '.join(missing)}")
        vector = validate_feature_vector([features[name] for name in FEATURE_NAMES])
        return pd.DataFrame([vector], columns=FEATURE_NAMES)
    values = np.asarray(features, dtype=np.float64)
    if values.ndim == 1:
        values = validate_feature_vector(values).reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"expected feature matrix with {len(FEATURE_NAMES)} columns, got {values.shape}"
        )
    return validate_feature_frame(pd.DataFrame(values, columns=FEATURE_NAMES))


class AnomalyScorer:
    """Isolation Forest whose calibrated outputs always lie in ``[0, 1]``."""

    def __init__(self, config: AnomalyConfig = DEFAULT_ANOMALY_CONFIG) -> None:
        self.config = config
        self.scaler: RobustScaler | None = None
        self.model: IsolationForest | None = None
        self.calibrator: EmpiricalQuantileCalibrator | None = None

    @property
    def is_fitted(self) -> bool:
        return self.scaler is not None and self.model is not None and self.calibrator is not None

    def fit(self, features: pd.DataFrame | np.ndarray) -> Self:
        frame = _as_feature_frame(features)
        if len(frame) < 2:
            raise ValueError("anomaly model requires at least two training rows")
        scaler = RobustScaler(quantile_range=self.config.quantile_range)
        transformed = scaler.fit_transform(frame)
        model = IsolationForest(
            n_estimators=self.config.n_estimators,
            max_samples=min(self.config.max_samples, len(frame)),
            contamination=self.config.contamination,
            random_state=self.config.random_state,
            n_jobs=1,
        )
        model.fit(transformed)
        # sklearn score_samples is larger for normal points; invert it so all
        # downstream component scores share the convention "higher is riskier".
        raw_scores = -model.score_samples(transformed)
        self.scaler = scaler
        self.model = model
        self.calibrator = EmpiricalQuantileCalibrator.fit(raw_scores)
        return self

    def score_samples(
        self, features: pd.DataFrame | pd.Series | Sequence[float] | np.ndarray
    ) -> np.ndarray:
        scaler, model, calibrator = self._fitted_components()
        frame = _as_feature_frame(features)
        raw_scores = -model.score_samples(scaler.transform(frame))
        return calibrator.transform(raw_scores)

    def score(self, features: pd.DataFrame | pd.Series | Sequence[float] | np.ndarray) -> float:
        scores = self.score_samples(features)
        if scores.size != 1:
            raise ValueError("score() requires exactly one row; use score_samples() for batches")
        return float(scores[0])

    def save(self, path: str | Path) -> str:
        """Atomically persist a versioned artifact and return its SHA-256 digest.

        joblib artifacts are pickle-based.  :meth:`load` hashes the bytes before
        deserialization and can require a caller-provided trusted digest.  It
        must still never be used for an artifact from an untrusted source.
        """

        scaler, model, calibrator = self._fitted_components()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.is_symlink():
            raise ValueError("refusing to overwrite a symlinked anomaly artifact")
        payload = {
            "artifact_version": ANOMALY_ARTIFACT_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": FEATURE_NAMES,
            "sklearn_version": sklearn.__version__,
            "config": asdict(self.config),
            "scaler": scaler,
            "model": model,
            "calibration_scores": calibrator.reference_scores,
        }
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(file_descriptor)
        temporary = Path(temporary_name)
        try:
            joblib.dump(payload, temporary, compress=3)
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            os.replace(temporary, destination)
            digest_path = self._digest_path(destination)
            digest_path.write_text(digest + "\n", encoding="ascii")
        finally:
            temporary.unlink(missing_ok=True)
        return digest

    @classmethod
    def load(cls, path: str | Path, *, expected_sha256: str | None = None) -> Self:
        """Verify integrity and compatibility before loading a trusted artifact."""

        source = Path(path)
        if source.is_symlink() or not source.is_file():
            raise ValueError("anomaly artifact must be an existing regular non-symlink file")
        size = source.stat().st_size
        if size <= 0 or size > _MAX_ARTIFACT_BYTES:
            raise ValueError("anomaly artifact size is invalid")
        expected = expected_sha256
        if expected is None:
            digest_path = cls._digest_path(source)
            if digest_path.is_symlink() or not digest_path.is_file():
                raise ValueError("anomaly artifact digest sidecar is missing or unsafe")
            expected = digest_path.read_text(encoding="ascii").strip().lower()
        if not _DIGEST_PATTERN.fullmatch(expected):
            raise ValueError("expected anomaly artifact SHA-256 is invalid")
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("anomaly artifact failed SHA-256 verification")

        payload: Any = joblib.load(source)
        required = {
            "artifact_version",
            "feature_schema_version",
            "feature_names",
            "sklearn_version",
            "config",
            "scaler",
            "model",
            "calibration_scores",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("anomaly artifact envelope is invalid")
        if payload["artifact_version"] != ANOMALY_ARTIFACT_VERSION:
            raise ValueError("unsupported anomaly artifact version")
        if payload["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
            raise ValueError("anomaly artifact feature schema is incompatible")
        if tuple(payload["feature_names"]) != FEATURE_NAMES:
            raise ValueError("anomaly artifact feature order is incompatible")
        if payload["sklearn_version"] != sklearn.__version__:
            raise ValueError("anomaly artifact scikit-learn version is incompatible")
        if not isinstance(payload["scaler"], RobustScaler) or not isinstance(
            payload["model"], IsolationForest
        ):
            raise ValueError("anomaly artifact contains unexpected estimator types")

        instance = cls(AnomalyConfig(**payload["config"]))
        instance.scaler = payload["scaler"]
        instance.model = payload["model"]
        instance.calibrator = EmpiricalQuantileCalibrator.fit(payload["calibration_scores"])
        instance._fitted_components()
        return instance

    def _fitted_components(
        self,
    ) -> tuple[RobustScaler, IsolationForest, EmpiricalQuantileCalibrator]:
        if self.scaler is None or self.model is None or self.calibrator is None:
            raise RuntimeError("anomaly scorer has not been fitted")
        check_is_fitted(self.scaler)
        check_is_fitted(self.model)
        return self.scaler, self.model, self.calibrator

    @staticmethod
    def _digest_path(path: Path) -> Path:
        return path.with_name(path.name + ".sha256")
