"""
Unified data utilities for biodiversity analysis.

This module eliminates redundant data loading, alignment, sampling, and validation
logic across all biodiversity analyzers.
"""

import numpy as np
import rasterio
from typing import Dict, Tuple, Optional, List, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RasterDataLoader:
    """Unified raster data loading with consistent preprocessing."""
    
    @staticmethod
    def load_single_raster(file_path: Union[str, Path], 
                          dtype: np.dtype = np.float32,
                          band: int = 1) -> np.ndarray:
        """
        Load a single raster file with consistent dtype conversion.
        
        Args:
            file_path: Path to raster file
            dtype: Target data type (default: float32)
            band: Band number to read (default: 1)
            
        Returns:
            Loaded raster data as numpy array
        """
        with rasterio.open(file_path) as src:
            data = src.read(band).astype(dtype)
            logger.debug(f"Loaded {file_path}: shape {data.shape}, dtype {data.dtype}")
            return data
    
    @staticmethod
    def get_raster_info(file_path: Union[str, Path]) -> Dict:
        """
        Get raster metadata without loading data.
        
        Args:
            file_path: Path to raster file
            
        Returns:
            Dictionary with raster metadata
        """
        with rasterio.open(file_path) as src:
            return {
                'height': src.height,
                'width': src.width,
                'bounds': src.bounds,
                'transform': src.transform,
                'crs': src.crs,
                'dtype': src.dtypes[0]
            }


class MultiRasterAligner:
    """Align multiple raster datasets to common dimensions and extent."""
    
    @staticmethod
    def find_common_dimensions(file_paths: Dict[str, Union[str, Path]]) -> Tuple[int, int]:
        """
        Find minimum common dimensions across multiple raster files.
        
        Args:
            file_paths: Dictionary of {name: file_path}
            
        Returns:
            Tuple of (min_height, min_width)
        """
        min_height, min_width = float('inf'), float('inf')
        
        for name, path in file_paths.items():
            info = RasterDataLoader.get_raster_info(path)
            min_height = min(min_height, info['height'])
            min_width = min(min_width, info['width'])
            logger.debug(f"{name}: {info['height']}x{info['width']}")
        
        min_height, min_width = int(min_height), int(min_width)
        logger.info(f"Common dimensions: {min_height}x{min_width}")
        
        return min_height, min_width
    
    @staticmethod
    def load_and_align_datasets(file_paths: Dict[str, Union[str, Path]],
                               dtype: np.dtype = np.float32,
                               band: int = 1) -> Dict[str, np.ndarray]:
        """
        Load multiple raster files and align them to common dimensions.
        
        Args:
            file_paths: Dictionary of {name: file_path}
            dtype: Target data type
            band: Band number to read
            
        Returns:
            Dictionary of {name: aligned_array}
        """
        # Find common dimensions
        min_height, min_width = MultiRasterAligner.find_common_dimensions(file_paths)
        
        # Load and crop all datasets
        aligned_data = {}
        for name, path in file_paths.items():
            # Load full data
            data = RasterDataLoader.load_single_raster(path, dtype, band)
            
            # Crop to common dimensions
            cropped_data = data[:min_height, :min_width]
            aligned_data[name] = cropped_data
            
            logger.debug(f"Aligned {name}: {data.shape} -> {cropped_data.shape}")
        
        logger.info(f"Aligned {len(aligned_data)} datasets to {min_height}x{min_width}")
        return aligned_data


class SpatialSampler:
    """Apply spatial sampling to reduce data size while maintaining spatial patterns."""
    
    @staticmethod
    def apply_sampling(data: np.ndarray, 
                      sample_factor: int,
                      method: str = 'regular') -> np.ndarray:
        """
        Apply spatial sampling to reduce data size.
        
        Args:
            data: Input 2D array
            sample_factor: Sampling factor (e.g., 5 = every 5th pixel)
            method: Sampling method ('regular' or 'random')
            
        Returns:
            Spatially sampled array
        """
        if sample_factor <= 1:
            return data
        
        if method == 'regular':
            # Regular grid sampling - every nth pixel
            sampled = data[::sample_factor, ::sample_factor]
        elif method == 'random':
            # Random sampling (maintaining approximate sample factor)
            height, width = data.shape
            n_samples = (height * width) // (sample_factor ** 2)
            
            # Generate random indices
            flat_indices = np.random.choice(height * width, n_samples, replace=False)
            row_indices = flat_indices // width
            col_indices = flat_indices % width
            
            # Create sampled array (sparse representation)
            sampled = np.full((height // sample_factor, width // sample_factor), np.nan)
            sampled_rows = row_indices // sample_factor
            sampled_cols = col_indices // sample_factor
            
            # Fill sampled array where we have data
            valid_mask = (sampled_rows < sampled.shape[0]) & (sampled_cols < sampled.shape[1])
            sampled[sampled_rows[valid_mask], sampled_cols[valid_mask]] = data[row_indices[valid_mask], col_indices[valid_mask]]
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        reduction_factor = (data.size / sampled.size)
        logger.info(f"Spatial sampling: {data.shape} -> {sampled.shape} "
                   f"(reduction: {reduction_factor:.1f}x)")
        
        return sampled
    
    @staticmethod
    def sample_multiple_datasets(datasets: Dict[str, np.ndarray],
                                sample_factor: int,
                                method: str = 'regular') -> Dict[str, np.ndarray]:
        """
        Apply consistent spatial sampling to multiple datasets.
        
        Args:
            datasets: Dictionary of {name: data_array}
            sample_factor: Sampling factor
            method: Sampling method
            
        Returns:
            Dictionary of {name: sampled_array}
        """
        sampled_datasets = {}
        
        for name, data in datasets.items():
            sampled_data = SpatialSampler.apply_sampling(data, sample_factor, method)
            sampled_datasets[name] = sampled_data
        
        return sampled_datasets


class DataValidator:
    """Data validation and cleaning utilities."""
    
    @staticmethod
    def create_valid_mask(data: np.ndarray,
                         min_value: float = 0.0,
                         exclude_nan: bool = True,
                         exclude_inf: bool = True) -> np.ndarray:
        """
        Create mask for valid data points.
        
        Args:
            data: Input array
            min_value: Minimum valid value (default: 0 to exclude zeros)
            exclude_nan: Whether to exclude NaN values
            exclude_inf: Whether to exclude infinite values
            
        Returns:
            Boolean mask where True = valid data
        """
        mask = np.ones(data.shape, dtype=bool)
        
        if exclude_nan:
            mask &= ~np.isnan(data)
        
        if exclude_inf:
            mask &= ~np.isinf(data)
        
        if min_value is not None:
            mask &= (data > min_value)
        
        valid_count = np.sum(mask)
        total_count = data.size
        valid_percent = (valid_count / total_count) * 100
        
        logger.debug(f"Valid data: {valid_count:,}/{total_count:,} ({valid_percent:.1f}%)")
        
        return mask
    
    @staticmethod
    def apply_multi_dataset_mask(datasets: Dict[str, np.ndarray],
                                min_value: float = 0.0) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Apply validation mask across multiple datasets (intersection of valid regions).
        
        Args:
            datasets: Dictionary of {name: data_array}
            min_value: Minimum valid value
            
        Returns:
            Tuple of (masked_datasets, combined_mask)
        """
        # Create individual masks
        masks = {}
        for name, data in datasets.items():
            masks[name] = DataValidator.create_valid_mask(data, min_value)
        
        # Combined mask (intersection - valid in ALL datasets)
        combined_mask = np.ones_like(list(masks.values())[0], dtype=bool)
        for mask in masks.values():
            combined_mask &= mask
        
        # Apply combined mask to all datasets
        masked_datasets = {}
        for name, data in datasets.items():
            masked_datasets[name] = data[combined_mask]
        
        valid_count = np.sum(combined_mask)
        total_count = combined_mask.size
        valid_percent = (valid_count / total_count) * 100
        
        logger.info(f"Multi-dataset valid mask: {valid_count:,}/{total_count:,} ({valid_percent:.1f}%)")
        
        return masked_datasets, combined_mask


class DataFlattener:
    """Convert 2D spatial data to 1D arrays for statistical analysis."""
    
    @staticmethod
    def flatten_single(data: np.ndarray, 
                      mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Flatten 2D array to 1D, optionally applying mask.
        
        Args:
            data: Input 2D array
            mask: Optional boolean mask
            
        Returns:
            Flattened 1D array
        """
        if mask is not None:
            return data[mask]
        else:
            return data.flatten()
    
    @staticmethod
    def flatten_multiple(datasets: Dict[str, np.ndarray],
                        mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Flatten multiple 2D arrays to 1D.
        
        Args:
            datasets: Dictionary of {name: 2D_array}
            mask: Optional boolean mask to apply to all
            
        Returns:
            Dictionary of {name: 1D_array}
        """
        flattened = {}
        
        for name, data in datasets.items():
            flattened[name] = DataFlattener.flatten_single(data, mask)
            logger.debug(f"Flattened {name}: {data.shape} -> {flattened[name].shape}")
        
        return flattened


# Convenience functions for common workflows
def load_align_and_sample(file_paths: Dict[str, Union[str, Path]],
                         sample_factor: int = 1,
                         dtype: np.dtype = np.float32,
                         sampling_method: str = 'regular') -> Dict[str, np.ndarray]:
    """
    Complete workflow: load, align, and sample multiple raster datasets.
    
    Args:
        file_paths: Dictionary of {name: file_path}
        sample_factor: Spatial sampling factor
        dtype: Target data type
        sampling_method: Sampling method
        
    Returns:
        Dictionary of processed arrays
    """
    logger.info(f"Loading and aligning {len(file_paths)} datasets...")
    
    # Load and align
    aligned_data = MultiRasterAligner.load_and_align_datasets(file_paths, dtype)
    
    # Apply spatial sampling if requested
    if sample_factor > 1:
        logger.info(f"Applying {sample_factor}x spatial sampling...")
        aligned_data = SpatialSampler.sample_multiple_datasets(
            aligned_data, sample_factor, sampling_method
        )
    
    return aligned_data


def prepare_for_correlation_analysis(file_paths: Dict[str, Union[str, Path]],
                                    sample_factor: int = 1,
                                    min_value: float = 0.0) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Prepare multiple datasets for correlation analysis.
    
    Args:
        file_paths: Dictionary of {name: file_path}
        sample_factor: Spatial sampling factor
        min_value: Minimum valid value
        
    Returns:
        Tuple of (flattened_data, metadata)
    """
    # Load, align, and sample
    spatial_data = load_align_and_sample(file_paths, sample_factor)
    
    # Apply validation and get combined mask
    masked_data, combined_mask = DataValidator.apply_multi_dataset_mask(
        spatial_data, min_value
    )
    
    # Flatten for correlation analysis
    flattened_data = DataFlattener.flatten_multiple(masked_data)
    
    # Collect metadata
    sample_shape = list(spatial_data.values())[0].shape
    metadata = {
        'original_files': len(file_paths),
        'spatial_shape': sample_shape,
        'total_pixels': np.prod(sample_shape),
        'valid_pixels': len(list(flattened_data.values())[0]),
        'coverage_percent': (len(list(flattened_data.values())[0]) / np.prod(sample_shape)) * 100,
        'sample_factor': sample_factor
    }
    
    return flattened_data, metadata