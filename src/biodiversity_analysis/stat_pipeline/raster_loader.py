"""Raster loader for statistical analysis pipeline.

This module provides efficient raster loading and preprocessing for statistical
analysis methods that require spatial data structure.
"""

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

from src.biodiversity_analysis.preprocessing.pipeline import SharedPreprocessor

logger = logging.getLogger(__name__)


class StatRasterLoader:
    """Efficient raster loading for statistical analysis."""
    
    def __init__(self):
        """Initialize raster loader."""
        self.preprocessor = None  # Created based on variant
        self._common_dimensions = None
        
    def load(self, 
             paths: Dict[str, str],
             variant: str = '4_features',
             sample_factor: int = 1,
             align: bool = True) -> Dict[str, np.ndarray]:
        """Load and preprocess raster data.
        
        Args:
            paths: Dictionary mapping dataset names to file paths
            variant: '4_features' or '3_features_merged'
            sample_factor: Spatial sampling factor (1 = full resolution)
            align: Whether to align all rasters to common dimensions
            
        Returns:
            Dictionary of preprocessed arrays
        """
        # Initialize preprocessor
        self.preprocessor = SharedPreprocessor(
            method_name='stat',
            normalize=False,
            missing_strategy='zero',
            combine_fungi=(variant == '3_features_merged'),
            feature_names=list(paths.keys())
        )
        
        # Load rasters
        if align:
            loaded = self._load_aligned(paths, sample_factor)
        else:
            loaded = self._load_individual(paths, sample_factor)
        
        # Apply preprocessing
        loaded = self._preprocess_rasters(loaded, variant)
        
        logger.info(f"Loaded {len(loaded)} rasters for variant '{variant}'")
        return loaded
    
    def load_with_coordinates(self, 
                            path: str,
                            sample_factor: int = 1) -> Dict:
        """Load single raster with coordinate information.
        
        Args:
            path: Path to raster file
            sample_factor: Spatial sampling factor
            
        Returns:
            Dictionary with data, coordinates, and metadata
        """
        with rasterio.open(path) as src:
            # Read data with optional sampling
            if sample_factor > 1:
                # Use decimated read for sampling
                out_shape = (
                    src.height // sample_factor,
                    src.width // sample_factor
                )
                data = src.read(
                    1,
                    out_shape=out_shape,
                    resampling=Resampling.average
                )
            else:
                data = src.read(1)
            
            # Get bounds and transform
            bounds = src.bounds
            transform = src.transform
            
            # Apply preprocessing
            data = self._clean_raster_data(data, path)
            
            # Generate coordinate arrays
            height, width = data.shape
            
            # Calculate longitude and latitude arrays
            lons = np.linspace(bounds.west, bounds.east, width)
            lats = np.linspace(bounds.north, bounds.south, height)
            
            # Create 2D coordinate grids if needed
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            
            return {
                'data': data,
                'lons': lons,
                'lats': lats,
                'lon_grid': lon_grid,
                'lat_grid': lat_grid,
                'bounds': bounds,
                'transform': transform,
                'shape': data.shape
            }
    
    def _load_aligned(self, 
                     paths: Dict[str, str],
                     sample_factor: int) -> Dict[str, np.ndarray]:
        """Load multiple rasters aligned to common dimensions.
        
        Args:
            paths: Dictionary of paths
            sample_factor: Sampling factor
            
        Returns:
            Dictionary of aligned arrays
        """
        # Find common dimensions
        min_height, min_width = self._find_common_dimensions(paths)
        
        # Apply sampling to dimensions
        if sample_factor > 1:
            min_height = min_height // sample_factor
            min_width = min_width // sample_factor
        
        logger.info(f"Aligning rasters to common dimensions: {min_height}x{min_width}")
        
        loaded = {}
        for name, path in paths.items():
            with rasterio.open(path) as src:
                # Read with cropping and optional sampling
                if sample_factor > 1:
                    # Decimated read with cropping
                    window = Window(0, 0, min_width * sample_factor, min_height * sample_factor)
                    data = src.read(
                        1,
                        window=window,
                        out_shape=(min_height, min_width),
                        resampling=Resampling.average
                    )
                else:
                    # Direct read with cropping
                    window = Window(0, 0, min_width, min_height)
                    data = src.read(1, window=window)
                
                # Convert to float32 for consistency
                data = data.astype(np.float32)
                
                # Clean the data
                data = self._clean_raster_data(data, name)
                
                loaded[name] = data
                logger.debug(f"Loaded {name}: shape {data.shape}, "
                           f"range [{np.nanmin(data):.2f}, {np.nanmax(data):.2f}]")
        
        return loaded
    
    def _load_individual(self, 
                        paths: Dict[str, str],
                        sample_factor: int) -> Dict[str, np.ndarray]:
        """Load rasters without alignment.
        
        Args:
            paths: Dictionary of paths
            sample_factor: Sampling factor
            
        Returns:
            Dictionary of arrays (may have different dimensions)
        """
        loaded = {}
        
        for name, path in paths.items():
            with rasterio.open(path) as src:
                if sample_factor > 1:
                    # Decimated read
                    out_shape = (
                        src.height // sample_factor,
                        src.width // sample_factor
                    )
                    data = src.read(
                        1,
                        out_shape=out_shape,
                        resampling=Resampling.average
                    )
                else:
                    data = src.read(1)
                
                # Convert to float32
                data = data.astype(np.float32)
                
                # Clean the data
                data = self._clean_raster_data(data, name)
                
                loaded[name] = data
                logger.debug(f"Loaded {name}: shape {data.shape}")
        
        return loaded
    
    def _find_common_dimensions(self, paths: Dict[str, str]) -> Tuple[int, int]:
        """Find minimum common dimensions across rasters.
        
        Args:
            paths: Dictionary of paths
            
        Returns:
            Tuple of (min_height, min_width)
        """
        if self._common_dimensions is not None:
            return self._common_dimensions
        
        min_height = float('inf')
        min_width = float('inf')
        
        for name, path in paths.items():
            with rasterio.open(path) as src:
                min_height = min(min_height, src.height)
                min_width = min(min_width, src.width)
                logger.debug(f"{name}: {src.height}x{src.width}")
        
        min_height = int(min_height)
        min_width = int(min_width)
        
        self._common_dimensions = (min_height, min_width)
        logger.info(f"Common dimensions: {min_height}x{min_width}")
        
        return min_height, min_width
    
    def _clean_raster_data(self, data: np.ndarray, name: str) -> np.ndarray:
        """Clean raster data for richness values.
        
        Args:
            data: Input array
            name: Dataset name for context
            
        Returns:
            Cleaned array
        """
        # Handle extreme negative values in fungi data
        if 'fungi' in name.lower():
            mask = data < -1e20
            data[mask] = np.nan
            if np.any(mask):
                logger.debug(f"Cleaned {np.sum(mask)} extreme negative values in {name}")
        
        # Negative richness values are invalid
        mask = data < 0
        data[mask] = np.nan
        if np.any(mask):
            logger.debug(f"Cleaned {np.sum(mask)} negative values in {name}")
        
        # Zero richness treated as no-data
        mask = data == 0
        data[mask] = np.nan
        if np.any(mask):
            logger.debug(f"Converted {np.sum(mask)} zeros to NaN in {name}")
        
        return data
    
    def _preprocess_rasters(self, 
                           data: Dict[str, np.ndarray],
                           variant: str) -> Dict[str, np.ndarray]:
        """Apply preprocessing including variant-specific merging.
        
        Args:
            data: Dictionary of arrays
            variant: Dataset variant
            
        Returns:
            Preprocessed dictionary
        """
        # Handle fungi merging for 3_features_merged variant
        if variant == '3_features_merged':
            if 'am_fungi_richness' in data and 'ecm_fungi_richness' in data:
                # Create total fungi by summing
                am_fungi = data['am_fungi_richness']
                ecm_fungi = data['ecm_fungi_richness']
                
                # Use nansum but handle all-nan cases
                with np.errstate(invalid='ignore'):
                    total_fungi = np.where(
                        np.isnan(am_fungi) & np.isnan(ecm_fungi),
                        np.nan,
                        np.nansum([am_fungi, ecm_fungi], axis=0)
                    )
                
                # Clean the merged data
                total_fungi[total_fungi == 0] = np.nan
                
                # Add to dictionary and remove individuals
                data['total_fungi_richness'] = total_fungi
                del data['am_fungi_richness']
                del data['ecm_fungi_richness']
                
                logger.info("Created merged total_fungi_richness")
        
        return data


class RasterMerger:
    """Utility for merging rasters efficiently."""
    
    @staticmethod
    def merge_and_save(input_paths: List[str],
                      output_path: str,
                      method: str = 'sum') -> str:
        """Merge multiple rasters and save to disk.
        
        Args:
            input_paths: List of input raster paths
            output_path: Output path for merged raster
            method: Merge method ('sum', 'mean', 'max', 'min')
            
        Returns:
            Path to saved merged raster
        """
        # Read first raster for metadata
        with rasterio.open(input_paths[0]) as src:
            meta = src.meta.copy()
            height = src.height
            width = src.width
            
        # Process in chunks for memory efficiency
        chunk_size = 1000  # Process 1000 rows at a time
        
        # Create output raster
        with rasterio.open(output_path, 'w', **meta) as dst:
            for row_start in range(0, height, chunk_size):
                row_end = min(row_start + chunk_size, height)
                window = Window(0, row_start, width, row_end - row_start)
                
                # Load chunk from all rasters
                chunks = []
                for path in input_paths:
                    with rasterio.open(path) as src:
                        chunk = src.read(1, window=window)
                        chunks.append(chunk)
                
                # Merge chunks
                stacked = np.stack(chunks, axis=0)
                
                if method == 'sum':
                    merged = np.nansum(stacked, axis=0)
                elif method == 'mean':
                    merged = np.nanmean(stacked, axis=0)
                elif method == 'max':
                    merged = np.nanmax(stacked, axis=0)
                elif method == 'min':
                    merged = np.nanmin(stacked, axis=0)
                else:
                    raise ValueError(f"Unknown merge method: {method}")
                
                # Write chunk
                dst.write(merged, 1, window=window)
        
        logger.info(f"Merged {len(input_paths)} rasters to {output_path}")
        return output_path


# Convenience functions
def load_rasters_for_trust_maps(paths: Dict[str, str], 
                               variant: str = '4_features',
                               sample_factor: int = 10) -> Dict[str, np.ndarray]:
    """Load rasters optimized for trust map analysis.
    
    Args:
        paths: Dictionary of raster paths
        variant: Dataset variant
        sample_factor: Sampling factor (default 10 for trust maps)
        
    Returns:
        Dictionary of aligned arrays
    """
    loader = StatRasterLoader()
    return loader.load(paths, variant, sample_factor, align=True)


def load_rasters_for_global_maps(paths: Dict[str, str],
                                variant: str = '4_features') -> List[Dict]:
    """Load rasters with coordinates for global map visualization.
    
    Args:
        paths: Dictionary of raster paths
        variant: Dataset variant
        
    Returns:
        List of dictionaries with data and coordinates
    """
    loader = StatRasterLoader()
    
    # Handle merging if needed
    if variant == '3_features_merged' and 'am_fungi_richness' in paths and 'ecm_fungi_richness' in paths:
        # Merge fungi rasters first
        merger = RasterMerger()
        merged_path = Path(paths['am_fungi_richness']).parent / 'total_fungi_richness.tif'
        
        if not merged_path.exists():
            merger.merge_and_save(
                [paths['am_fungi_richness'], paths['ecm_fungi_richness']],
                str(merged_path),
                method='sum'
            )
        
        # Update paths
        paths = paths.copy()
        paths['total_fungi_richness'] = str(merged_path)
        del paths['am_fungi_richness']
        del paths['ecm_fungi_richness']
    
    # Load each with coordinates
    results = []
    for name, path in paths.items():
        result = loader.load_with_coordinates(path)
        result['name'] = name
        results.append(result)
    
    return results