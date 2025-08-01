#!/usr/bin/env python3
"""Quick downsampling of all datasets for fast parquet generation."""

import sys
import os
from pathlib import Path
import time
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Datasets configuration
DATASETS = [
    {
        'name': 'plants-richness',
        'path': '/maps/mwd24/richness/daru-plants-richness.tif',
        'method': 'max',  # Max for richness counts
        'dtype': 'uint16'
    },
    {
        'name': 'terrestrial-richness', 
        'path': '/maps/mwd24/richness/iucn-terrestrial-richness.tif',
        'method': 'max',  # Max for richness counts
        'dtype': 'uint16'
    },
    {
        'name': 'am-fungi-richness',
        'path': '/scratch/yl998/resampled_rasters/AM_Fungi_Richness_Predicted_resampled_0.016667.tif',
        'method': 'average',  # Average for SDM predictions
        'dtype': 'float32'
    },
    {
        'name': 'ecm-fungi-richness',
        'path': '/scratch/yl998/resampled_rasters/EcM_Fungi_Richness_Predicted_resampled_0.016667.tif',
        'method': 'average',  # Average for SDM predictions
        'dtype': 'float32'
    }
]

# Target resolution - 10x coarser
TARGET_RESOLUTION = 0.16667  # ~18.5km
OUTPUT_DIR = Path('/scratch/yl998/downsampled_rasters')
OUTPUT_DIR.mkdir(exist_ok=True)

def downsample_dataset(dataset):
    """Downsample a single dataset."""
    start_time = time.time()
    name = dataset['name']
    input_path = dataset['path']
    method = dataset['method']
    dtype = dataset['dtype']
    
    output_path = OUTPUT_DIR / f"{name}_downsampled_{TARGET_RESOLUTION:.5f}.tif"
    
    logger.info(f"Processing {name}...")
    logger.info(f"  Input: {input_path}")
    logger.info(f"  Method: {method}")
    
    try:
        with rasterio.open(input_path) as src:
            # Calculate downsampling factor
            current_res = abs(src.transform[0])
            scale_factor = TARGET_RESOLUTION / current_res
            logger.info(f"  Current resolution: {current_res:.6f}")
            logger.info(f"  Scale factor: {scale_factor:.1f}x")
            
            # Calculate new dimensions
            new_width = int(src.width * current_res / TARGET_RESOLUTION)
            new_height = int(src.height * current_res / TARGET_RESOLUTION)
            
            # Calculate new transform
            transform, width, height = calculate_default_transform(
                src.crs, src.crs, 
                new_width, new_height,
                *src.bounds
            )
            
            logger.info(f"  New dimensions: {width} x {height} ({width*height:,} pixels)")
            
            # Update metadata
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': src.crs,
                'transform': transform,
                'width': width,
                'height': height,
                'dtype': dtype,
                'compress': 'lzw',
                'tiled': True,
                'blockxsize': 512,
                'blockysize': 512
            })
            
            # Choose resampling method
            if method == 'max':
                resample_method = Resampling.max
            elif method == 'average':
                resample_method = Resampling.average
            else:
                resample_method = Resampling.nearest
            
            # Perform resampling
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=src.crs,
                    resampling=resample_method,
                    num_threads=4
                )
            
            # Verify output
            with rasterio.open(output_path) as check:
                data = check.read(1)
                valid_pixels = np.sum(~np.isnan(data) & (data != check.nodata))
                logger.info(f"  Valid pixels: {valid_pixels:,} ({valid_pixels/data.size*100:.1f}%)")
            
            elapsed = time.time() - start_time
            logger.info(f"✓ Completed {name} in {elapsed:.1f} seconds")
            logger.info(f"  Output: {output_path}")
            
            return {
                'name': name,
                'success': True,
                'output': str(output_path),
                'pixels': width * height,
                'time': elapsed
            }
            
    except Exception as e:
        logger.error(f"✗ Failed {name}: {e}")
        return {
            'name': name,
            'success': False,
            'error': str(e),
            'time': time.time() - start_time
        }

def main():
    """Process all datasets in parallel."""
    logger.info("="*60)
    logger.info(f"QUICK DOWNSAMPLING TO {TARGET_RESOLUTION}° RESOLUTION")
    logger.info("="*60)
    
    # Process datasets in parallel
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(downsample_dataset, ds): ds for ds in DATASETS}
        results = []
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    
    total_time = sum(r['time'] for r in results)
    successful = [r for r in results if r['success']]
    
    for result in results:
        if result['success']:
            logger.info(f"✓ {result['name']}: {result['pixels']:,} pixels in {result['time']:.1f}s")
        else:
            logger.info(f"✗ {result['name']}: {result.get('error', 'Unknown error')}")
    
    logger.info(f"\nTotal time: {total_time:.1f} seconds")
    logger.info(f"Success rate: {len(successful)}/{len(results)}")
    
    if len(successful) == len(results):
        logger.info("\n✅ All datasets downsampled successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Update config.yml to point to downsampled files")
        logger.info("2. Run pipeline with --experiment-name 'downsampled_fast'")
        return 0
    else:
        logger.error("\n❌ Some datasets failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())