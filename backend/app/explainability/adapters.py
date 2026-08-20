"""Local supervised-model contribution adapters.

Attributions in this module are deliberately internal.  The public explanation
service converts them to a small, curated vocabulary and never serializes the
numeric contribution itself.
"""

from __future__ import annotations

import importlib
import io
from collections.abc import Sequence
from contextlib import redirect_stderr
from dataclasses import dataclass
from math import isfinite
from typing import Any, Protocol

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class FeatureContribution:
    """One local feature contribution, for internal ranking only."""

    feature: str
    contribution: float
    value: float | None = None

    def __post_init__(self) -> None:
        if not self.feature:
            raise ValueError("feature must not be empty")
        if not isfinite(self.contribution):
            raise ValueError("contribution must be finite")
        if self.value is not None and not isfinite(self.value):
            raise ValueError("feature value must be finite when supplied")


class ContributionAdapter(Protocol):
    """Boundary implemented by SHAP and equivalent local explainers."""

    def explain(self, model: Any, features: pd.DataFrame) -> list[FeatureContribution]: ...


def _fraud_output(values: Any) -> np.ndarray:
    """Normalize common SHAP output layouts to one row of feature values."""

    if isinstance(values, list):
        values = values[-1]
    array = np.asarray(values, dtype=float)
    if array.ndim == 3:  # rows, features, outputs
        array = array[:, :, -1]
    if array.ndim == 2:
        array = array[0]
    if array.ndim != 1:
        raise ValueError(f"unsupported attribution shape {array.shape}")
    return array


def _pipeline_parts(model: Any, features: pd.DataFrame) -> tuple[Any, np.ndarray, list[str]]:
    """Return final estimator, transformed values, and best-effort feature names."""

    estimator = model
    transformed: Any = features
    names = [str(name) for name in features.columns]
    if hasattr(model, "steps") and len(model.steps) > 1:
        preprocessing = model[:-1]
        estimator = model.steps[-1][1]
        transformed = preprocessing.transform(features)
        try:
            names = [str(name) for name in preprocessing.get_feature_names_out(features.columns)]
        except (AttributeError, TypeError, ValueError):
            pass
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    matrix = np.asarray(transformed, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != 1:
        raise ValueError("exactly one feature row is required for an explanation")
    if len(names) != matrix.shape[1]:
        names = [f"feature_{index}" for index in range(matrix.shape[1])]
    return estimator, matrix, names


class ShapContributionAdapter:
    """SHAP integration for logistic and tree estimators, including pipelines.

    SHAP is imported lazily so inference still works in lightweight deployments.
    A training background can be supplied for production artifacts.  With no
    background, a zero reference in transformed space is used for linear models.
    """

    def __init__(self, background: np.ndarray | None = None) -> None:
        self._background = background

    def explain(self, model: Any, features: pd.DataFrame) -> list[FeatureContribution]:
        # SHAP imports optional image packages eagerly. A broken unrelated
        # OpenCV installation can print ABI diagnostics even though SHAP
        # correctly degrades without it; keep that third-party noise out of
        # request logs while still surfacing an actual SHAP import exception.
        with redirect_stderr(io.StringIO()):
            shap = importlib.import_module("shap")

        estimator, matrix, names = _pipeline_parts(model, features)
        module = estimator.__class__.__module__.lower()
        class_name = estimator.__class__.__name__.lower()
        is_linear = hasattr(estimator, "coef_")
        is_tree = (
            hasattr(estimator, "tree_")
            or hasattr(estimator, "estimators_")
            or "xgboost" in module
            or "forest" in class_name
            or "tree" in class_name
        )
        if is_linear:
            background = self._background
            if background is None:
                background = np.zeros((1, matrix.shape[1]), dtype=float)
            explainer = shap.LinearExplainer(estimator, background)
            raw = explainer(matrix)
        elif is_tree:
            explainer = shap.TreeExplainer(estimator, data=self._background)
            raw = explainer(matrix)
        else:
            raise TypeError("SHAP adapter supports fitted linear and tree estimators")

        values = _fraud_output(getattr(raw, "values", raw))
        if values.shape[0] != len(names):
            raise ValueError("attribution count does not match transformed features")
        contributions: list[FeatureContribution] = []
        observed = features.iloc[0]
        for name, value in zip(names, values, strict=True):
            if not isfinite(float(value)):
                continue
            raw_name = name.rsplit("__", 1)[-1]
            observed_value = observed.get(raw_name)
            try:
                local_value = float(observed_value)
            except (TypeError, ValueError):
                local_value = None
            if local_value is not None and not isfinite(local_value):
                local_value = None
            contributions.append(FeatureContribution(name, float(value), local_value))
        return contributions


class PerturbationContributionAdapter:
    """Model-agnostic local occlusion fallback using the model's prediction API."""

    def __init__(self, reference: pd.Series | dict[str, float] | None = None) -> None:
        self._reference = reference

    @staticmethod
    def _score(model: Any, frame: pd.DataFrame) -> float:
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(frame), dtype=float)
            if probabilities.ndim == 2:
                return float(probabilities[0, -1])
            return float(probabilities.reshape(-1)[0])
        if hasattr(model, "decision_function"):
            raw = float(np.asarray(model.decision_function(frame), dtype=float).reshape(-1)[0])
            return float(1.0 / (1.0 + np.exp(-np.clip(raw, -50.0, 50.0))))
        prediction = np.asarray(model.predict(frame), dtype=float).reshape(-1)
        return float(prediction[0])

    def explain(self, model: Any, features: pd.DataFrame) -> list[FeatureContribution]:
        if len(features) != 1:
            raise ValueError("exactly one feature row is required for an explanation")
        observed = self._score(model, features)
        reference = dict(self._reference) if self._reference is not None else {}
        contributions: list[FeatureContribution] = []
        for feature in features.columns:
            comparison = features.copy()
            baseline = float(reference.get(str(feature), 0.0))
            comparison.loc[comparison.index[0], feature] = baseline
            delta = observed - self._score(model, comparison)
            value = float(features.iloc[0][feature])
            if isfinite(delta) and isfinite(value):
                contributions.append(FeatureContribution(str(feature), delta, value))
        return contributions


class ModelContributionAdapter:
    """Use fast exact-model perturbation for linear models and SHAP for trees.

    The explicit :class:`ShapContributionAdapter` remains the retained offline
    evaluation path. Per-request linear explanations use probability
    perturbation to avoid a costly first-request import while preserving local,
    directional feature attribution.
    """

    def __init__(
        self,
        *,
        background: np.ndarray | None = None,
        reference: pd.Series | dict[str, float] | None = None,
    ) -> None:
        self._shap = ShapContributionAdapter(background)
        self._fallback = PerturbationContributionAdapter(reference)

    def explain(self, model: Any, features: pd.DataFrame) -> list[FeatureContribution]:
        estimator = model.steps[-1][1] if hasattr(model, "steps") and model.steps else model
        if hasattr(estimator, "coef_"):
            return self._fallback.explain(model, features)
        try:
            return self._shap.explain(model, features)
        except (ImportError, OSError, AttributeError, TypeError, ValueError, RuntimeError):
            return self._fallback.explain(model, features)


def coerce_contributions(
    contributions: Sequence[FeatureContribution] | None,
) -> list[FeatureContribution]:
    """Make optional adapter output convenient for callers and tests."""

    return list(contributions or ())
