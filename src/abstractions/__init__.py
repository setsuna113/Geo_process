"""
Abstractions module - only multi-use abstractions.

Following the principle: Abstract if used multiple times, don't if used once.
"""

from .interfaces import IAnalyzer, IMethodConfig
from .types import (
    TensorData,
    CPFactors,
    AnalysisResult,
    AnalysisMetadata,
    BiodiversityData,
    TensorMetrics,
    SOMConfig,
    SOMTrainingResult,
    TensorMode,
    AnalysisStatus
)

__all__ = [
    # Interfaces
    'IAnalyzer',
    'IMethodConfig',
    # Types
    'TensorData',
    'CPFactors', 
    'AnalysisResult',
    'AnalysisMetadata',
    'BiodiversityData',
    'TensorMetrics',
    'SOMConfig',
    'SOMTrainingResult',
    'TensorMode',
    'AnalysisStatus'
]