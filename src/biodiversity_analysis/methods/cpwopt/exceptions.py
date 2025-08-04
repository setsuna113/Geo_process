"""Custom exceptions for CP-WOPT module.

This module defines standardized exceptions for error handling across the
CP-WOPT implementation.
"""


class CPWOPTError(Exception):
    """Base exception for all CP-WOPT related errors."""
    pass


class TensorValidationError(CPWOPTError):
    """Raised when tensor data validation fails."""
    pass


class FactorValidationError(CPWOPTError):
    """Raised when factor matrix validation fails."""
    pass


class ConfigurationError(CPWOPTError):
    """Raised when configuration is invalid or inconsistent."""
    pass


class ConvergenceError(CPWOPTError):
    """Raised when algorithm fails to converge."""
    pass


class MemoryError(CPWOPTError):
    """Raised when memory requirements exceed available resources."""
    pass


class SecurityError(CPWOPTError):
    """Raised when security validation fails (e.g., path traversal)."""
    pass


class PreprocessingError(CPWOPTError):
    """Raised when data preprocessing fails."""
    pass


class ReporterError(CPWOPTError):
    """Raised when report generation fails."""
    pass