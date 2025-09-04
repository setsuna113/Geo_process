"""
Raster loader - Linus style.

"Talk is cheap. Show me the code."
Dead simple raster loading with rasterio.
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from typing import Dict, Union, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RasterLoader:
    """Simple raster loader. No inheritance, no magic."""
    
    def __init__(self):
        """Nothing to initialize. This class is basically stateless."""
        pass
    
    def load_single(self, path: str) -> np.ndarray:
        """
        Load a single raster file.
        
        Args:
            path: Path to raster file
            
        Returns:
            Numpy array with raster data
        """
        with rasterio.open(path) as src:
            data = src.read(1)  # Read first band
            # Replace nodata with NaN
            if src.nodata is not None:
                data = data.astype(np.float32)
                data[data == src.nodata] = np.nan
            return data
    
    def load(self, 
             raster_paths: Dict[str, Union[str, Dict]],
             align: bool = True,
             target_shape: Optional[Tuple[int, int]] = None) -> Dict[str, np.ndarray]:
        """
        Load multiple rasters, optionally aligning them.
        
        Args:
            raster_paths: Dict mapping names to raster paths
            align: Whether to align rasters to common grid
            target_shape: Force specific output shape
            
        Returns:
            Dict mapping names to numpy arrays
        """
        rasters = {}
        
        # First pass: load and get metadata
        metadata = {}
        for name, path_or_dict in raster_paths.items():
            if path_or_dict is None:
                continue
                
            # Handle nested dict (e.g., merged fungi)
            if isinstance(path_or_dict, dict):
                # This is for merged datasets
                sub_rasters = {}
                for sub_name, sub_path in path_or_dict.items():
                    if sub_path and isinstance(sub_path, str):
                        with rasterio.open(sub_path) as src:
                            sub_rasters[sub_name] = src.read(1)
                            if name not in metadata:  # Use first as reference
                                metadata[name] = {
                                    'transform': src.transform,
                                    'crs': src.crs,
                                    'shape': src.shape,
                                    'bounds': src.bounds
                                }
                # Simple average for merged data
                if sub_rasters:
                    rasters[name] = np.nanmean(list(sub_rasters.values()), axis=0)
                    
            elif isinstance(path_or_dict, str):
                # Single raster file
                with rasterio.open(path_or_dict) as src:
                    data = src.read(1)
                    # Replace nodata with NaN
                    if src.nodata is not None:
                        data = data.astype(np.float32)
                        data[data == src.nodata] = np.nan
                    rasters[name] = data
                    metadata[name] = {
                        'transform': src.transform,
                        'crs': src.crs,
                        'shape': src.shape,
                        'bounds': src.bounds
                    }
        
        # Align if requested
        if align and len(rasters) > 1:
            rasters = self._align_rasters(rasters, metadata, target_shape)
        
        return rasters
    
    def _align_rasters(self, 
                       rasters: Dict[str, np.ndarray],
                       metadata: Dict[str, dict],
                       target_shape: Optional[Tuple[int, int]] = None) -> Dict[str, np.ndarray]:
        """
        Align rasters to common grid.
        
        Uses the first raster as reference or target_shape if provided.
        """
        # Get reference (first raster)
        ref_name = list(rasters.keys())[0]
        ref_meta = metadata[ref_name]
        
        if target_shape:
            ref_shape = target_shape
        else:
            ref_shape = ref_meta['shape']
        
        aligned = {}
        
        for name, data in rasters.items():
            if metadata[name]['shape'] == ref_shape:
                # Already aligned
                aligned[name] = data
            else:
                # Need to reproject
                logger.info(f"Aligning {name} from {metadata[name]['shape']} to {ref_shape}")
                
                # Create destination array
                dst_data = np.empty(ref_shape, dtype=np.float32)
                dst_data.fill(np.nan)
                
                # Calculate transform for reprojection
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    metadata[name]['crs'], 
                    ref_meta['crs'],
                    metadata[name]['shape'][1], 
                    metadata[name]['shape'][0],
                    *metadata[name]['bounds'],
                    dst_width=ref_shape[1],
                    dst_height=ref_shape[0]
                )
                
                # Reproject
                reproject(
                    source=data,
                    destination=dst_data,
                    src_transform=metadata[name]['transform'],
                    src_crs=metadata[name]['crs'],
                    dst_transform=dst_transform,
                    dst_crs=ref_meta['crs'],
                    resampling=Resampling.bilinear
                )
                
                aligned[name] = dst_data
        
        return aligned
    
    def load_with_sampling(self,
                          path: str,
                          sample_factor: int = 1) -> np.ndarray:
        """
        Load raster with spatial sampling.
        
        Args:
            path: Path to raster
            sample_factor: Sample every Nth pixel (1 = no sampling)
            
        Returns:
            Sampled array
        """
        if sample_factor <= 1:
            return self.load_single(path)
        
        with rasterio.open(path) as src:
            # Read with decimation
            data = src.read(
                1,
                out_shape=(
                    src.height // sample_factor,
                    src.width // sample_factor
                ),
                resampling=Resampling.bilinear
            )
            
            # Handle nodata
            if src.nodata is not None:
                data = data.astype(np.float32)
                data[data == src.nodata] = np.nan
                
            return data
    
    def compute_stats_chunked(self,
                             path: str,
                             chunk_size: int = 1000) -> Dict[str, float]:
        """
        Compute statistics using chunked reading - for coverage analysis.
        Never loads full raster into memory.
        
        Args:
            path: Path to raster
            chunk_size: Rows per chunk
            
        Returns:
            Statistics dict
        """
        with rasterio.open(path) as src:
            stats = {
                'shape': src.shape,
                'total_pixels': src.height * src.width,
            }
            
            # Accumulate stats
            valid_count = 0
            value_sum = 0.0
            value_sum2 = 0.0
            value_min = float('inf')
            value_max = float('-inf')
            
            # Process in chunks
            for row_start in range(0, src.height, chunk_size):
                row_end = min(row_start + chunk_size, src.height)
                
                # Read chunk
                chunk = src.read(1, window=((row_start, row_end), (0, src.width)))
                
                # Handle nodata
                if src.nodata is not None:
                    valid_mask = (chunk != src.nodata) & ~np.isnan(chunk)
                else:
                    valid_mask = ~np.isnan(chunk)
                
                valid_chunk = chunk[valid_mask]
                
                if len(valid_chunk) > 0:
                    valid_count += len(valid_chunk)
                    value_sum += valid_chunk.sum()
                    value_sum2 += (valid_chunk ** 2).sum()
                    value_min = min(value_min, valid_chunk.min())
                    value_max = max(value_max, valid_chunk.max())
            
            # Calculate final stats
            stats['valid_pixels'] = valid_count
            stats['coverage_percent'] = 100 * valid_count / stats['total_pixels']
            
            if valid_count > 0:
                mean = value_sum / valid_count
                variance = (value_sum2 / valid_count) - mean ** 2
                stats.update({
                    'min': float(value_min),
                    'max': float(value_max),
                    'mean': float(mean),
                    'std': float(np.sqrt(max(0, variance)))
                })
            
            return stats