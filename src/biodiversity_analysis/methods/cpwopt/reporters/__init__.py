"""CP-WOPT reporting modules for visualization and interpretation.

This package provides a clean, abstracted reporting system with:
- BaseReporter: Abstract base class for all reporters
- VisualizationReporter: Handles spatial, component, and biome visualizations
- AnalyticsReporter: Handles factor analysis, reconstruction metrics, and summaries
- UnifiedReporter: Main entry point that orchestrates all reporting

Legacy reporters are preserved for backward compatibility but are deprecated.
"""

# New consolidated reporters
from .base_reporter import BaseReporter
from .consolidated_reporters import (
    VisualizationReporter,
    AnalyticsReporter,
    UnifiedReporter
)

# Legacy reporters (deprecated but kept for compatibility)
from .factor_reporter import FactorReporter
from .reconstruction_reporter import ReconstructionReporter
from .spatial_reporter import SpatialReporter
from .component_reporter import ComponentReporter
from .summary_reporter import SummaryReporter
from .biome_reporter import BiomeReporter

__all__ = [
    # New abstracted reporters
    'BaseReporter',
    'VisualizationReporter',
    'AnalyticsReporter', 
    'UnifiedReporter',
    
    # Legacy (deprecated)
    'FactorReporter',
    'ReconstructionReporter',
    'SpatialReporter',
    'ComponentReporter',
    'SummaryReporter',
    'BiomeReporter'
]