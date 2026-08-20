"""Safe composition of supervised, anomaly, rule, and graph explanations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

import pandas as pd

from backend.app.schemas import RiskSignal

from .adapters import ContributionAdapter, FeatureContribution, ModelContributionAdapter
from .reasons import (
    CODE_ALIASES,
    FEATURE_REASONS,
    MESSAGES,
    SOURCE_FALLBACK,
    canonical_feature_name,
)


@dataclass(frozen=True, slots=True)
class Explanation:
    """Only display-ready content crosses the explainability boundary."""

    # Lists intentionally match ``FraudAssessment``'s strict transport fields,
    # allowing the result to be passed into that Pydantic model unchanged.
    reasons: list[str]
    signals: list[RiskSignal]


class ExplanationService:
    def __init__(
        self,
        adapter: ContributionAdapter | None = None,
        *,
        max_reasons: int = 5,
        minimum_severity: float = 0.05,
    ) -> None:
        if not 1 <= max_reasons <= 20:
            raise ValueError("max_reasons must be between 1 and 20")
        if not 0.0 <= minimum_severity <= 1.0:
            raise ValueError("minimum_severity must be between 0 and 1")
        self._adapter = adapter or ModelContributionAdapter()
        self._max_reasons = max_reasons
        self._minimum_severity = minimum_severity

    def explain(
        self,
        *,
        features: pd.DataFrame | Mapping[str, float] | None = None,
        model: Any | None = None,
        contributions: Sequence[FeatureContribution] | None = None,
        component_signals: Iterable[RiskSignal | Mapping[str, Any]] = (),
    ) -> Explanation:
        frame = self._frame(features)
        local = list(contributions or ())
        if not local and model is not None and frame is not None:
            try:
                local = self._adapter.explain(model, frame)
            except (ImportError, OSError, AttributeError, TypeError, ValueError, RuntimeError):
                local = []

        candidates = self._supervised_signals(local, frame)
        candidates.extend(self._component_signals(component_signals))
        merged: dict[str, RiskSignal] = {}
        for signal in candidates:
            if signal.severity < self._minimum_severity:
                continue
            current = merged.get(signal.code)
            if current is None or signal.severity > current.severity:
                merged[signal.code] = signal
        ranked = sorted(merged.values(), key=lambda item: (-item.severity, item.code))
        selected = ranked[: self._max_reasons]
        return Explanation([signal.message for signal in selected], selected)

    @staticmethod
    def _frame(
        features: pd.DataFrame | Mapping[str, float] | None,
    ) -> pd.DataFrame | None:
        if features is None:
            return None
        if isinstance(features, pd.DataFrame):
            if len(features) != 1:
                raise ValueError("exactly one feature row is required")
            return features
        numeric: dict[str, float] = {}
        for key, value in features.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if isfinite(number):
                numeric[str(key)] = number
        return pd.DataFrame([numeric], dtype=float)

    @staticmethod
    def _supervised_signals(
        contributions: Sequence[FeatureContribution], frame: pd.DataFrame | None
    ) -> list[RiskSignal]:
        positive = [item for item in contributions if item.contribution > 0.0]
        scale = max((item.contribution for item in positive), default=0.0)
        if scale <= 0.0:
            return []
        values = frame.iloc[0].to_dict() if frame is not None else {}
        signals: list[RiskSignal] = []
        for item in positive:
            feature = canonical_feature_name(item.feature)
            template = FEATURE_REASONS.get(feature)
            value = item.value if item.value is not None else values.get(feature)
            if template is None or value is None:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if not isfinite(numeric_value) or not template.matches(numeric_value):
                continue
            signals.append(
                RiskSignal(
                    code=template.code,
                    message=template.message,
                    severity=min(max(item.contribution / scale, 0.0), 1.0),
                    source="supervised",
                )
            )
        return signals

    @staticmethod
    def _component_signals(
        signals: Iterable[RiskSignal | Mapping[str, Any]],
    ) -> list[RiskSignal]:
        safe: list[RiskSignal] = []
        for raw in signals:
            if isinstance(raw, RiskSignal):
                code, severity, source = raw.code, raw.severity, raw.source
            elif isinstance(raw, Mapping):
                code = str(raw.get("code", ""))
                source = str(raw.get("source", ""))
                try:
                    severity = float(raw.get("severity", 0.0))
                except (TypeError, ValueError):
                    continue
            else:
                continue
            if source not in SOURCE_FALLBACK or not isfinite(severity):
                continue
            severity = min(max(severity, 0.0), 1.0)
            canonical = CODE_ALIASES.get(code, code)
            if canonical in MESSAGES:
                message = MESSAGES[canonical]
            else:
                canonical, message = SOURCE_FALLBACK[source]
            safe.append(
                RiskSignal(code=canonical, message=message, severity=severity, source=source)
            )
        return safe


def explain_risk(
    *,
    features: pd.DataFrame | Mapping[str, float] | None = None,
    model: Any | None = None,
    contributions: Sequence[FeatureContribution] | None = None,
    component_signals: Iterable[RiskSignal | Mapping[str, Any]] = (),
    max_reasons: int = 5,
) -> Explanation:
    """Convenience entry point for the inference orchestration service."""

    return ExplanationService(max_reasons=max_reasons).explain(
        features=features,
        model=model,
        contributions=contributions,
        component_signals=component_signals,
    )
