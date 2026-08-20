"""Model attribution and human-readable explanations."""

from .adapters import (
    ContributionAdapter,
    FeatureContribution,
    ModelContributionAdapter,
    PerturbationContributionAdapter,
    ShapContributionAdapter,
)
from .service import Explanation, ExplanationService, explain_risk

__all__ = [
    "ContributionAdapter",
    "Explanation",
    "ExplanationService",
    "FeatureContribution",
    "ModelContributionAdapter",
    "PerturbationContributionAdapter",
    "ShapContributionAdapter",
    "explain_risk",
]
