"""Shared, ordered feature engineering for training and inference."""

from .builder import FeatureBuilder, FeatureValidationError
from .contract import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    FEATURE_SPECS,
    FeatureSpec,
    feature_schema_manifest,
    validate_feature_frame,
    validate_feature_vector,
)

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "FEATURE_SPECS",
    "FeatureBuilder",
    "FeatureSpec",
    "FeatureValidationError",
    "feature_schema_manifest",
    "validate_feature_frame",
    "validate_feature_vector",
]
