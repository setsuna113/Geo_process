"""Self-Organizing Map (SOM) analysis for biodiversity patterns.

This module implements GeoSOM + VLRSOM hybrid architecture for analyzing
biodiversity data with high sparsity (up to 70% missing values).

Key Features:
- Partial Bray-Curtis distance for sparse data
- Geographic coherence for spatial patterns
- Adaptive learning rate (VLRSOM)
- Optimized mini-batch training with momentum
- Parallel processing for large datasets
"""

# Core implementation
from .core.geo_som_core import GeoSOMVLRSOM, GeoSOMConfig

# Main analyzer interface
from .analyzer import GeoSOMAnalyzer
from .analyzer_adapter import SOMAnalyzer

# Training algorithms
from .trainers import ParallelBatchProcessor

# Calculators and metrics
from .calculators import (
    OptimizedDistanceCalculator,
    validate_missing_data_handling
)

# Utilities
from .utils import (
    SpatialBlockGenerator,
    GeographicCoherence,
    SOMProgressTracker,
    create_progress_callback
)

# Visualization
from .visualization import SOMVisualizer

__all__ = [
    # Core
    'GeoSOMVLRSOM',
    'GeoSOMConfig',
    
    # Analyzers
    'GeoSOMAnalyzer',
    'SOMAnalyzer',
    
    # Trainers
    'ParallelBatchProcessor',
    
    # Calculators
    'OptimizedDistanceCalculator',
    'validate_missing_data_handling',
    
    # Utils
    'SpatialBlockGenerator',
    'GeographicCoherence',
    'SOMProgressTracker',
    'create_progress_callback',
    
    # Visualization
    'SOMVisualizer'
]