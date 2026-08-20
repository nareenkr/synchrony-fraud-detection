"""Lightweight graph-proxy risk based on prior entity cardinalities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd

from backend.app.features.contract import FEATURE_NAMES, validate_feature_vector
from backend.app.schemas import RiskSignal


@dataclass(frozen=True, slots=True)
class GraphConfig:
    """Cardinality ramps and weights for device, IP, and bank associations.

    Counts are prior distinct users, so a value of one means the current user
    would be the second identity associated with that entity.
    """

    device_low: float = 0.0
    device_high: float = 5.0
    ip_low: float = 0.0
    ip_high: float = 7.0
    bank_low: float = 0.0
    bank_high: float = 3.0
    device_weight: float = 1.2
    ip_weight: float = 1.0
    bank_weight: float = 1.1

    def __post_init__(self) -> None:
        ranges = (
            (self.device_low, self.device_high),
            (self.ip_low, self.ip_high),
            (self.bank_low, self.bank_high),
        )
        if any(low < 0 or high <= low for low, high in ranges):
            raise ValueError("graph thresholds must be non-negative and strictly ordered")
        weights = self.weights().values()
        if any(weight < 0 for weight in weights) or sum(self.weights().values()) <= 0:
            raise ValueError("graph weights must be non-negative with a positive sum")

    def weights(self) -> dict[str, float]:
        return {
            "device": self.device_weight,
            "ip": self.ip_weight,
            "bank": self.bank_weight,
        }


DEFAULT_GRAPH_CONFIG = GraphConfig()


@dataclass(frozen=True, slots=True)
class GraphScore:
    risk: float
    signals: tuple[RiskSignal, ...]

    @property
    def score(self) -> float:
        return self.risk


def _ramp(value: float, low: float, high: float) -> float:
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _feature_values(
    features: Mapping[str, float] | pd.Series | pd.DataFrame | Sequence[float] | np.ndarray,
) -> dict[str, float]:
    if isinstance(features, pd.DataFrame):
        if len(features) != 1:
            raise ValueError("graph scoring requires exactly one feature row")
        if tuple(features.columns) != FEATURE_NAMES:
            raise ValueError("feature columns do not match the ordered feature contract")
        vector = validate_feature_vector(features.iloc[0].to_numpy(dtype=float))
    elif isinstance(features, (Mapping, pd.Series)):
        missing = [name for name in FEATURE_NAMES if name not in features]
        if missing:
            raise ValueError(f"missing canonical features: {', '.join(missing)}")
        vector = validate_feature_vector([features[name] for name in FEATURE_NAMES])
    else:
        vector = validate_feature_vector(features)
    return dict(zip(FEATURE_NAMES, vector, strict=True))


class GraphScorer:
    """Score graph-like linkage without constructing a heavyweight graph DB."""

    def __init__(self, config: GraphConfig = DEFAULT_GRAPH_CONFIG) -> None:
        self.config = config

    def score(
        self,
        features: Mapping[str, float] | pd.Series | pd.DataFrame | Sequence[float] | np.ndarray,
    ) -> GraphScore:
        values = _feature_values(features)
        config = self.config
        factors = {
            "device": _ramp(
                values["shared_device_user_count"], config.device_low, config.device_high
            ),
            "ip": _ramp(values["shared_ip_user_count"], config.ip_low, config.ip_high),
            "bank": _ramp(values["shared_bank_user_count"], config.bank_low, config.bank_high),
        }
        weights = config.weights()
        risk = sum(factors[name] * weights[name] for name in factors) / sum(weights.values())
        if not isfinite(risk):
            raise ValueError("graph risk is not finite")

        definitions = {
            "device": (
                "SHARED_DEVICE",
                "Device is associated with multiple applicant accounts.",
            ),
            "ip": ("SHARED_IP", "Network address is associated with multiple applicants."),
            "bank": (
                "SHARED_BANK_ACCOUNT",
                "Bank account is associated with multiple applicant identities.",
            ),
        }
        signals = tuple(
            sorted(
                (
                    RiskSignal(
                        code=definitions[name][0],
                        message=definitions[name][1],
                        severity=factor,
                        source="graph",
                    )
                    for name, factor in factors.items()
                    if factor > 0
                ),
                key=lambda signal: (-signal.severity, signal.code),
            )
        )
        return GraphScore(risk=float(np.clip(risk, 0.0, 1.0)), signals=signals)
