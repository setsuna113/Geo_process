"""Self-Organizing Map (SOM) analysis for biodiversity patterns.

This module implements GeoSOM with pure Batch SOM for analyzing
biodiversity data with high sparsity (up to 70% missing values).

Key Features:
- Partial Bray-Curtis distance for sparse data
- Geographic coherence for spatial patterns
- Pure Batch SOM (no learning rate, momentum control)
- Direct parallel processing with Numba
- Handles 13M+ samples efficiently
"""

# Core implementation
from .core.geo_som_core import GeoSOM, GeoSOMConfig
# Backwards compatibility
GeoSOMVLRSOM = GeoSOM

# Main analyzer interface
from .analyzer import GeoSOMAnalyzer
from .analyzer_adapter import SOMAnalyzer

# Training algorithms
from .trainers import PureNumbaSOM

# Calculators and metrics
from .calculators import validate_missing_data_handling

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
    'PureNumbaSOM',
    
    # Calculators
    'validate_missing_data_handling',
    
    # Utils
    'SpatialBlockGenerator',
    'GeographicCoherence',
    'SOMProgressTracker',
    'create_progress_callback',
    
    # Visualization
    'SOMVisualizer'
]