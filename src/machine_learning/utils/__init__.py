"""Utility modules for machine learning."""

from .julia_manager import JuliaManager

# Alias for backward compatibility during transition
H2OManager = JuliaManager

__all__ = ['JuliaManager', 'H2OManager']