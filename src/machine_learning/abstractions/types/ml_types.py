"""Machine learning type definitions."""

from enum import Enum


class ModelType(Enum):
    """Types of machine learning models."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"