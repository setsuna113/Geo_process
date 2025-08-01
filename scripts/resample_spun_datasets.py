#!/usr/bin/env python3
"""Resample SPUN fungi datasets to match resolution of other datasets."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import config
from src.tools.rasterio_resampler import RasterioResampler, ResamplingConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Resample SPUN datasets to target resolution."""
    # Get SPUN dataset paths from config
    spun_datasets = []
    datasets_config = config.get('datasets', {}).get('target_datasets', [])
    for dataset in datasets_config:
        if 'fungi' in dataset['name'] and dataset.get('enabled', True):
            spun_datasets.append({
                'name': dataset['name'],
                'path': dataset['path'],
                'data_type': dataset.get('data_type', 'continuous_data')
            })
    
    if not spun_datasets:
        logger.error("No SPUN fungi datasets found in config")
        return 1
    
    logger.info(f"Found {len(spun_datasets)} SPUN datasets to resample")
    
    # Base config - will customize per dataset
    base_output_dir = Path(config.get('output_paths', {}).get('working_dir', '/scratch/yl998/geo_working')) / 'resampled'
    
    target_resolution = config.get('resampling', {}).get('target_resolution', 0.016666666666667)
    target_crs = config.get('resampling', {}).get('target_crs', 'EPSG:4326')
    
    logger.info(f"Target resolution: {target_resolution}")
    logger.info(f"Target CRS: {target_crs}")
    logger.info(f"Output directory: {base_output_dir}")
    
    # Process each dataset
    results = []
    for dataset in spun_datasets:
        logger.info(f"\nProcessing {dataset['name']}...")
        logger.info(f"Input: {dataset['path']}")
        
        try:
            # Check if input exists
            if not os.path.exists(dataset['path']):
                logger.error(f"Input file not found: {dataset['path']}")
                continue
            
            # Create dataset-specific config with unique progress file
            resample_config = ResamplingConfig(
                target_resolution=target_resolution,
                target_crs=target_crs,
                resampling_method='average',  # Area-weighted average for SDM predictions
                output_dir=str(base_output_dir),
                checkpoint_interval=5,
                log_level='INFO',
                compress='lzw',
                tiled=True,
                progress_file=f"resampling_progress_{dataset['name']}.json"
            )
            
            # Create resampler for this dataset
            resampler = RasterioResampler(resample_config)
            
            # Resample
            output_path = resampler.resample(dataset['path'])
            
            # Validate
            validation = resampler.validate_output(output_path, dataset['path'])
            
            results.append({
                'dataset': dataset['name'],
                'output': output_path,
                'valid': validation['valid'],
                'warnings': validation['warnings']
            })
            
            logger.info(f"✓ Completed: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to process {dataset['name']}: {e}")
            results.append({
                'dataset': dataset['name'],
                'output': None,
                'valid': False,
                'error': str(e)
            })
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("RESAMPLING SUMMARY")
    logger.info("="*60)
    
    for result in results:
        if result['output']:
            status = "✓ SUCCESS" if result['valid'] else "⚠ COMPLETE (with warnings)"
            logger.info(f"{result['dataset']}: {status}")
            logger.info(f"  Output: {result['output']}")
            if result.get('warnings'):
                for warning in result['warnings']:
                    logger.info(f"  Warning: {warning}")
        else:
            logger.info(f"{result['dataset']}: ✗ FAILED")
            if result.get('error'):
                logger.info(f"  Error: {result['error']}")
    
    # Return success if all completed
    return 0 if all(r['output'] is not None for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())