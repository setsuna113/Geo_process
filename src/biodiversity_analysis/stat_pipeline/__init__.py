"""Statistical analysis pipeline for biodiversity data.

This module provides optimized data loading and preprocessing for statistical
analysis methods including correlation, trust maps, overlap analysis, coverage,
latitudinal patterns, and global maps.
"""

from src.biodiversity_analysis.stat_pipeline.utils import (
    StatDataConverter,
    StatDataPreprocessor,
    OptimizedStatLoader,
    load_parquet_for_correlation,
    load_parquet_for_spatial
)

from src.biodiversity_analysis.stat_pipeline.raster_loader import (
    StatRasterLoader,
    RasterMerger,
    load_rasters_for_trust_maps,
    load_rasters_for_global_maps
)

from src.biodiversity_analysis.stat_pipeline.orchestrator import (
    StatPipelineOrchestrator,
    run_stat_pipeline
)

__all__ = [
    # Utils
    'StatDataConverter',
    'StatDataPreprocessor', 
    'OptimizedStatLoader',
    'load_parquet_for_correlation',
    'load_parquet_for_spatial',
    
    # Raster loader
    'StatRasterLoader',
    'RasterMerger',
    'load_rasters_for_trust_maps',
    'load_rasters_for_global_maps',
    
    # Orchestrator
    'StatPipelineOrchestrator',
    'run_stat_pipeline'
]

__version__ = '1.0.0'