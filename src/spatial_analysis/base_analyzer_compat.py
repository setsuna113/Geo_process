# src/spatial_analysis/base_analyzer_compat.py
"""Backward compatibility layer for BaseAnalyzer."""

import warnings
from src.infrastructure.analyzers.enhanced_analyzer import EnhancedAnalyzer
from src.foundations.interfaces.analyzer import IAnalyzer, AnalysisResult, AnalysisMetadata

# Export the data classes that other modules might import
__all__ = ['BaseAnalyzer', 'AnalysisResult', 'AnalysisMetadata']

class BaseAnalyzer(EnhancedAnalyzer):
    """
    Backward compatibility wrapper for BaseAnalyzer.
    
    This provides the same interface as the old BaseAnalyzer but uses the new
    architecture underneath. This will be deprecated in a future version.
    """
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "BaseAnalyzer is deprecated and will be removed in a future version. "
            "Use src.infrastructure.analyzers.enhanced_analyzer.EnhancedAnalyzer "
            "or src.foundations.interfaces.analyzer.IAnalyzer for interfaces. "
            "This compatibility layer will be removed in Phase 6.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)