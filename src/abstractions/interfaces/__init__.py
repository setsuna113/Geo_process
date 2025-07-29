"""Foundation interfaces - pure abstractions with no dependencies."""

from .processor import IProcessor
from .grid import IGrid, GridCell
from .validator import (
    BaseValidator, CompositeValidator, ConditionalValidator,
    ValidationResult, ValidationIssue, ValidationType, ValidationSeverity
)

__all__ = [
    'IProcessor', 'IGrid', 'GridCell',
    'BaseValidator', 'CompositeValidator', 'ConditionalValidator',
    'ValidationResult', 'ValidationIssue', 'ValidationType', 'ValidationSeverity'
]