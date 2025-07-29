"""Base validator interface for data integrity checking."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class ValidationType(Enum):
    """Types of validation checks."""
    BOUNDS_CONSISTENCY = "bounds_consistency"
    COORDINATE_TRANSFORM = "coordinate_transform"
    VALUE_RANGE = "value_range"
    DATA_COMPLETENESS = "data_completeness"
    SPATIAL_INTEGRITY = "spatial_integrity"


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"      # Critical issue that prevents processing
    WARNING = "warning"  # Issue that may affect quality but allows processing
    INFO = "info"       # Informational message


@dataclass
class ValidationIssue:
    """Single validation issue found during checking."""
    validator_name: str
    validation_type: ValidationType
    severity: ValidationSeverity
    message: str
    location: Optional[str] = None  # File path, cell ID, etc.
    details: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        """String representation of the issue."""
        loc = f" at {self.location}" if self.location else ""
        return f"[{self.severity.value.upper()}] {self.validator_name}: {self.message}{loc}"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    validator_name: str
    is_valid: bool
    issues: List[ValidationIssue]
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return sum(1 for issue in self.issues if issue.severity == ValidationSeverity.ERROR)
    
    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return sum(1 for issue in self.issues if issue.severity == ValidationSeverity.WARNING)
    
    @property
    def has_errors(self) -> bool:
        """Check if any error-level issues exist."""
        return self.error_count > 0
    
    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """Merge another validation result into this one."""
        return ValidationResult(
            validator_name=f"{self.validator_name}+{other.validator_name}",
            is_valid=self.is_valid and other.is_valid,
            issues=self.issues + other.issues,
            metadata={**(self.metadata or {}), **(other.metadata or {})}
        )


class BaseValidator(ABC):
    """
    Abstract base class for all validators.
    
    Validators check data integrity and consistency before, during,
    and after processing operations.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize validator.
        
        Args:
            name: Validator name (defaults to class name)
        """
        self.name = name or self.__class__.__name__
        self._config: Dict[str, Any] = {}
    
    def configure(self, **config) -> 'BaseValidator':
        """
        Configure validator with parameters.
        
        Args:
            **config: Configuration parameters
            
        Returns:
            Self for method chaining
        """
        self._config.update(config)
        return self
    
    @abstractmethod
    def validate(self, data: Any) -> ValidationResult:
        """
        Perform validation on data.
        
        Args:
            data: Data to validate
            
        Returns:
            ValidationResult with issues found
        """
        pass
    
    def validate_batch(self, data_items: List[Any]) -> List[ValidationResult]:
        """
        Validate a batch of data items.
        
        Args:
            data_items: List of data items to validate
            
        Returns:
            List of validation results
        """
        return [self.validate(item) for item in data_items]
    
    def create_issue(self, 
                    validation_type: ValidationType,
                    severity: ValidationSeverity,
                    message: str,
                    location: Optional[str] = None,
                    details: Optional[Dict[str, Any]] = None) -> ValidationIssue:
        """
        Helper to create a validation issue.
        
        Args:
            validation_type: Type of validation
            severity: Issue severity
            message: Issue description
            location: Optional location information
            details: Optional detailed information
            
        Returns:
            ValidationIssue instance
        """
        return ValidationIssue(
            validator_name=self.name,
            validation_type=validation_type,
            severity=severity,
            message=message,
            location=location,
            details=details
        )
    
    def create_result(self,
                     is_valid: bool,
                     issues: Optional[List[ValidationIssue]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Helper to create a validation result.
        
        Args:
            is_valid: Whether validation passed
            issues: List of issues found
            metadata: Optional metadata
            
        Returns:
            ValidationResult instance
        """
        return ValidationResult(
            validator_name=self.name,
            is_valid=is_valid,
            issues=issues or [],
            metadata=metadata
        )


class CompositeValidator(BaseValidator):
    """
    Validator that combines multiple validators.
    
    Runs all sub-validators and aggregates results.
    """
    
    def __init__(self, validators: List[BaseValidator], name: Optional[str] = None):
        """
        Initialize composite validator.
        
        Args:
            validators: List of validators to combine
            name: Composite validator name
        """
        super().__init__(name or "CompositeValidator")
        self.validators = validators
    
    def validate(self, data: Any) -> ValidationResult:
        """
        Run all validators and combine results.
        
        Args:
            data: Data to validate
            
        Returns:
            Combined validation result
        """
        all_issues = []
        all_metadata = {}
        is_valid = True
        
        for validator in self.validators:
            result = validator.validate(data)
            all_issues.extend(result.issues)
            if result.metadata:
                all_metadata.update(result.metadata)
            is_valid = is_valid and result.is_valid
        
        return self.create_result(
            is_valid=is_valid,
            issues=all_issues,
            metadata=all_metadata
        )


class ConditionalValidator(BaseValidator):
    """
    Validator that only runs if a condition is met.
    """
    
    def __init__(self, 
                 validator: BaseValidator,
                 condition: callable,
                 name: Optional[str] = None):
        """
        Initialize conditional validator.
        
        Args:
            validator: Validator to run conditionally
            condition: Function that takes data and returns bool
            name: Validator name
        """
        super().__init__(name or f"Conditional({validator.name})")
        self.validator = validator
        self.condition = condition
    
    def validate(self, data: Any) -> ValidationResult:
        """
        Run validator only if condition is met.
        
        Args:
            data: Data to validate
            
        Returns:
            Validation result or empty result if condition not met
        """
        if self.condition(data):
            return self.validator.validate(data)
        else:
            return self.create_result(is_valid=True, metadata={'skipped': True})