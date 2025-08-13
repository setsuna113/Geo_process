#!/usr/bin/env python
"""Test script for the statistical analysis pipeline."""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.biodiversity_analysis.stat_pipeline import (
    OptimizedStatLoader,
    StatRasterLoader,
    StatPipelineOrchestrator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_parquet_loader():
    """Test optimized parquet loading."""
    logger.info("Testing ParquetLoader with optimizations...")
    
    # Test with a sample parquet file
    parquet_path = "data/merged/biodiversity_climate_fine.parquet"
    
    if not Path(parquet_path).exists():
        logger.warning(f"Parquet file not found: {parquet_path}")
        return
    
    loader = OptimizedStatLoader()
    
    # Test 4-features variant
    logger.info("Loading 4-features variant...")
    df_4feat = loader.load_parquet_for_stats(
        parquet_path,
        variant='4_features',
        output_format='dataframe'
    )
    logger.info(f"Loaded DataFrame shape: {df_4feat.shape}")
    logger.info(f"Columns: {df_4feat.columns.tolist()}")
    
    # Test 3-features merged variant
    logger.info("Loading 3-features merged variant...")
    df_3feat = loader.load_parquet_for_stats(
        parquet_path,
        variant='3_features_merged',
        output_format='dataframe'
    )
    logger.info(f"Loaded DataFrame shape: {df_3feat.shape}")
    logger.info(f"Columns: {df_3feat.columns.tolist()}")
    
    # Test array output
    logger.info("Loading as arrays...")
    arrays = loader.load_parquet_for_stats(
        parquet_path,
        variant='4_features',
        output_format='arrays'
    )
    logger.info(f"Loaded {len(arrays)} arrays")
    for name, arr in arrays.items():
        logger.info(f"  {name}: shape {arr.shape}")


def test_raster_loader():
    """Test raster loading."""
    logger.info("Testing RasterLoader...")
    
    # Define test raster paths
    raster_paths = {
        'plant_richness': 'data/richness_maps/daru-plants-richness.tif',
        'animal_richness': 'data/biodiversity/local_animals.tif',
        'am_fungi_richness': 'data/fungal/spun/AM_fungi/AM_Fungi_Richness_Predicted.tif',
        'ecm_fungi_richness': 'data/fungal/spun/ECM_fungi/ECM_Fungi_Richness_Predicted.tif'
    }
    
    # Check which files exist
    existing = {}
    for name, path in raster_paths.items():
        if Path(path).exists():
            existing[name] = path
        else:
            logger.warning(f"Raster not found: {path}")
    
    if not existing:
        logger.warning("No raster files found for testing")
        return
    
    loader = StatRasterLoader()
    
    # Test aligned loading
    logger.info("Loading aligned rasters...")
    aligned = loader.load(
        existing,
        variant='4_features',
        sample_factor=10,
        align=True
    )
    
    for name, data in aligned.items():
        logger.info(f"  {name}: shape {data.shape}, "
                   f"range [{np.nanmin(data):.2f}, {np.nanmax(data):.2f}]")
    
    # Test with coordinates
    logger.info("Loading with coordinates...")
    first_path = list(existing.values())[0]
    coords_data = loader.load_with_coordinates(first_path, sample_factor=10)
    logger.info(f"  Data shape: {coords_data['data'].shape}")
    logger.info(f"  Lon range: {coords_data['lons'].min():.2f} to {coords_data['lons'].max():.2f}")
    logger.info(f"  Lat range: {coords_data['lats'].min():.2f} to {coords_data['lats'].max():.2f}")


def test_orchestrator():
    """Test the full pipeline orchestrator."""
    logger.info("Testing StatPipelineOrchestrator...")
    
    # Create test configuration
    config = {
        'parquet_path': 'data/merged/biodiversity_climate_fine.parquet',
        'raster_paths': {
            'plants': 'data/richness_maps/daru-plants-richness.tif',
            'animals': 'data/biodiversity/local_animals.tif',
        },
        'output_dir': 'outputs/test_stat_pipeline',
        'variant': '4_features',
        'sample_factor': 20,  # High sampling for fast test
        'methods': ['correlation']  # Just test one method
    }
    
    # Check if data exists
    if not Path(config['parquet_path']).exists():
        logger.warning("Parquet file not found, skipping orchestrator test")
        return
    
    # Create orchestrator
    orchestrator = StatPipelineOrchestrator(config)
    
    # Run analysis
    results = orchestrator.run_analysis()
    
    logger.info(f"Analysis results: {results}")


if __name__ == "__main__":
    import numpy as np
    
    logger.info("=" * 60)
    logger.info("STATISTICAL PIPELINE TEST")
    logger.info("=" * 60)
    
    # Run tests
    test_parquet_loader()
    logger.info("")
    
    test_raster_loader()
    logger.info("")
    
    test_orchestrator()
    
    logger.info("=" * 60)
    logger.info("TESTS COMPLETED")
    logger.info("=" * 60)