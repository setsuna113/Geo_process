# src/pipelines/stages/merge_stage.py
"""Dataset merging stage with memory-aware processing and passthrough support."""

from typing import List, Tuple, Dict, Optional
import logging
import xarray as xr
from datetime import datetime
import gc
import numpy as np
from pathlib import Path

from .base_stage import PipelineStage, StageResult

logger = logging.getLogger(__name__)


class MergeStage(PipelineStage):
    """Stage for merging resampled datasets with memory management."""
    
    @property
    def name(self) -> str:
        return "merge"
    
    @property
    def dependencies(self) -> List[str]:
        return ["resample"]
    
    @property
    def memory_requirements(self) -> float:
        # Dynamic based on processing config and chunking capability
        if self.processing_config and self.processing_config.enable_chunking:
            # Use memory limit from processing config for chunked mode
            return self.processing_config.memory_limit_mb / 1024.0  # Convert MB to GB
        # Fall back to config value or a sensible default
        from src.config import config
        return config.processing['subsampling']['memory_limit_gb']
    
    @property
    def supports_chunking(self) -> bool:
        return True
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate merge configuration."""
        return True, []
    
    def execute(self, context) -> StageResult:
        """Merge resampled datasets with memory management."""
        logger.info("Starting memory-aware merge stage")
        
        try:
            # Get resampled datasets
            resampled_datasets = context.get('resampled_datasets', [])
            
            if len(resampled_datasets) < 2:
                return StageResult(
                    success=False,
                    data={},
                    metrics={'datasets_merged': 0},
                    warnings=['Need at least 2 datasets for merging']
                )
            
            # Log dataset information
            for info in resampled_datasets:
                logger.info(f"Dataset {info.name}: bounds={info.bounds}, shape={info.shape}, resolution={info.target_resolution}")
            
            # Determine if we should use chunked merging
            total_size_estimate = sum(
                info.shape[0] * info.shape[1] * 4 / (1024**2)  # MB
                for info in resampled_datasets
            ) * len(resampled_datasets)
            
            use_chunked = (
                total_size_estimate > 1000 or  # More than 1GB estimated
                (self.processing_config and self.processing_config.enable_chunking)
            )
            
            if use_chunked:
                logger.info(f"Using chunked merging (estimated size: {total_size_estimate:.0f}MB)")
                merged_dataset = self._merge_chunked(context, resampled_datasets)
            else:
                logger.info(f"Using standard merging (estimated size: {total_size_estimate:.0f}MB)")
                merged_dataset = self._merge_standard(context, resampled_datasets)
            
            # Store in context
            context.set('merged_dataset', merged_dataset)
            
            # Save to file with compression
            output_path = context.output_dir / 'merged_dataset.nc'
            
            # Simple encoding without problematic parameters
            encoding = {
                var: {'zlib': True, 'complevel': 4} 
                for var in merged_dataset.data_vars
            }
            
            # Save with netcdf4 engine
            merged_dataset.to_netcdf(output_path, encoding=encoding, engine='netcdf4')
            
            # Store merged dataset metadata in database
            from src.database.schema import schema
            merged_metadata = {
                'experiment_id': context.experiment_id,
                'timestamp': datetime.now().isoformat(),
                'shape': dict(merged_dataset.sizes),
                'variables': list(merged_dataset.data_vars),
                'bounds': self._calculate_common_bounds(resampled_datasets),
                'resolution': resampled_datasets[0].target_resolution if resampled_datasets else None,
                'file_path': str(output_path),
                'file_size_mb': output_path.stat().st_size / (1024**2)
            }
            
            # Store metadata in context for later stages
            context.set('merge_metadata', merged_metadata)
            logger.info("✅ Merge metadata stored in context")
            
            # Clean up memory for chunked processing
            if use_chunked:
                del merged_dataset
                gc.collect()
                
                # Reload reference for context (lazy loading)
                merged_dataset = xr.open_dataset(output_path, chunks={'x': 100, 'y': 100})
                context.set('merged_dataset', merged_dataset)
            
            metrics = {
                'datasets_merged': len(resampled_datasets),
                'total_bands': len(merged_dataset.data_vars) if merged_dataset else 0,
                'output_shape': dict(merged_dataset.sizes) if merged_dataset else {},
                'output_size_mb': output_path.stat().st_size / (1024**2),
                'chunked_merge': use_chunked
            }
            
            return StageResult(
                success=True,
                data={'merged_dataset_path': str(output_path)},
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Merge stage failed: {e}")
            raise
    
    def _calculate_common_bounds(self, resampled_datasets) -> Tuple[float, float, float, float]:
        """Calculate the union of all dataset bounds."""
        min_x = min(info.bounds[0] for info in resampled_datasets)
        min_y = min(info.bounds[1] for info in resampled_datasets)
        max_x = max(info.bounds[2] for info in resampled_datasets)
        max_y = max(info.bounds[3] for info in resampled_datasets)
        
        logger.info(f"Common bounds: ({min_x}, {min_y}, {max_x}, {max_y})")
        return (min_x, min_y, max_x, max_y)
    
    def _create_common_coordinates(self, common_bounds: Tuple[float, float, float, float], 
                                  resolution: float) -> Dict[str, xr.DataArray]:
        """Create common coordinate arrays that encompass all datasets."""
        min_x, min_y, max_x, max_y = common_bounds
        
        # Calculate the number of pixels needed
        width = int(np.ceil((max_x - min_x) / resolution))
        height = int(np.ceil((max_y - min_y) / resolution))
        
        logger.info(f"Creating common grid: {width} x {height} pixels at resolution {resolution}")
        
        # Create coordinate arrays
        x_coords = xr.DataArray(
            data=[(min_x + (i + 0.5) * resolution) for i in range(width)],
            dims=['x'],
            attrs={'long_name': 'longitude', 'units': 'degrees_east'}
        )
        
        y_coords = xr.DataArray(
            data=[(max_y - (i + 0.5) * resolution) for i in range(height)],
            dims=['y'],
            attrs={'long_name': 'latitude', 'units': 'degrees_north'}
        )
        
        return {'x': x_coords, 'y': y_coords}
    
    def _align_data_to_common_grid(self, array_data: np.ndarray, 
                                  data_bounds: Tuple[float, float, float, float],
                                  common_bounds: Tuple[float, float, float, float],
                                  common_shape: Tuple[int, int],
                                  resolution: float) -> np.ndarray:
        """Align dataset to common coordinate grid."""
        # Create output array filled with appropriate value based on dtype
        if np.issubdtype(array_data.dtype, np.integer):
            # For integer data, use 0 or a specific nodata value instead of NaN
            fill_value = 0  # Could also use a specific nodata value from config
            aligned_data = np.full(common_shape, fill_value, dtype=array_data.dtype)
        else:
            # For floating point data, NaN is fine
            aligned_data = np.full(common_shape, np.nan, dtype=array_data.dtype)
        
        # Calculate offsets
        x_offset = int(np.round((data_bounds[0] - common_bounds[0]) / resolution))
        y_offset = int(np.round((common_bounds[3] - data_bounds[3]) / resolution))
        
        # Calculate the slice where this data should go
        y_slice = slice(y_offset, y_offset + array_data.shape[0])
        x_slice = slice(x_offset, x_offset + array_data.shape[1])
        
        # Place the data
        try:
            aligned_data[y_slice, x_slice] = array_data
        except ValueError as e:
            logger.error(f"Error aligning data: {e}")
            logger.error(f"Data shape: {array_data.shape}, target slice: y={y_slice}, x={x_slice}")
            logger.error(f"Common shape: {common_shape}")
            raise
        
        return aligned_data
    
    def _merge_standard(self, context, resampled_datasets):
        """Standard in-memory merge with proper coordinate handling."""
        from src.processors.data_preparation.resampling_processor import ResamplingProcessor
        processor = ResamplingProcessor(context.config, context.db)
        
        # Calculate common bounds and resolution
        common_bounds = self._calculate_common_bounds(resampled_datasets)
        
        # Use the first dataset's resolution (they should all be the same for passthrough)
        resolution = resampled_datasets[0].target_resolution
        
        # Create common coordinates
        coords = self._create_common_coordinates(common_bounds, resolution)
        common_shape = (len(coords['y']), len(coords['x']))
        
        data_vars = {}
        
        for info in resampled_datasets:
            logger.info(f"Loading data for: {info.name}")
            
            # Load array data based on whether it's passthrough or resampled
            if info.metadata.get('passthrough', False):
                logger.info(f"Loading passthrough data for {info.name}")
                array_data = processor.load_passthrough_data(info)
            else:
                logger.info(f"Loading resampled data for {info.name}")
                array_data = processor.load_resampled_data(info.name)
            
            if array_data is None:
                logger.warning(f"Failed to load data for {info.name}")
                continue
            
            logger.info(f"Loaded data shape: {array_data.shape}, bounds: {info.bounds}")
            
            # Align data to common grid
            aligned_data = self._align_data_to_common_grid(
                array_data, info.bounds, common_bounds, common_shape, resolution
            )
            
            # Create DataArray with common coordinates
            data_array = xr.DataArray(
                data=aligned_data,
                dims=['y', 'x'],
                coords=coords,
                attrs={
                    'long_name': f'{info.name} data',
                    'source_path': str(info.source_path),
                    'resampling_method': info.resampling_method,
                    'target_resolution': info.target_resolution,
                    'original_bounds': info.bounds,
                    'original_shape': info.shape
                }
            )
            
            data_vars[info.band_name] = data_array
            logger.info(f"Added band '{info.band_name}' to merged dataset")
        
        # Create merged dataset
        merged = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs={
                'title': 'Merged Resampled Dataset',
                'created_by': 'pipeline_merge_stage',
                'created_at': datetime.now().isoformat(),
                'common_bounds': common_bounds,
                'resolution': resolution
            }
        )
        
        logger.info(f"Created merged dataset with {len(data_vars)} bands, shape: {dict(merged.sizes)}")
        return merged
    
    def _merge_chunked(self, context, resampled_datasets):
        """Chunked merge for large datasets with proper coordinate handling."""
        from src.processors.data_preparation.resampling_processor import ResamplingProcessor
        processor = ResamplingProcessor(context.config, context.db)
        
        # Calculate common bounds and resolution
        common_bounds = self._calculate_common_bounds(resampled_datasets)
        resolution = resampled_datasets[0].target_resolution
        
        # Create common coordinates
        coords = self._create_common_coordinates(common_bounds, resolution)
        common_shape = (len(coords['y']), len(coords['x']))
        
        # Convert coordinates to lists for serialization
        x_values = coords['x'].values.tolist()
        y_values = coords['y'].values.tolist()
        
        # First, save each dataset as a separate chunked file
        temp_files = []
        
        for info in resampled_datasets:
            logger.info(f"Processing {info.name} for chunked merge")
            
            # Create temporary file path
            temp_path = context.output_dir / f"temp_{info.band_name}.nc"
            
            # Load array data based on whether it's passthrough or resampled
            if info.metadata.get('passthrough', False):
                logger.info(f"Loading passthrough data for {info.name}")
                array_data = processor.load_passthrough_data(info)
            else:
                logger.info(f"Loading resampled data for {info.name}")
                array_data = processor.load_resampled_data(info.name)
            
            if array_data is None:
                logger.warning(f"Failed to load data for {info.name}")
                continue
            
            logger.info(f"Loaded data shape: {array_data.shape}, bounds: {info.bounds}")
            
            # Align data to common grid
            aligned_data = self._align_data_to_common_grid(
                array_data, info.bounds, common_bounds, common_shape, resolution
            )
            
            # Create temporary dataset with aligned data
            temp_ds = xr.Dataset({
                info.band_name: xr.DataArray(
                    data=aligned_data,
                    dims=['y', 'x'],
                    coords={'x': x_values, 'y': y_values},
                    attrs={
                        'long_name': f'{info.name} data',
                        'source_path': str(info.source_path),
                        'resampling_method': info.resampling_method,
                        'original_bounds': info.bounds,
                        'original_shape': info.shape
                    }
                )
            })
            
            # Save without encoding to avoid issues
            temp_ds.to_netcdf(temp_path, engine='netcdf4')
            temp_files.append((info.band_name, temp_path))
            
            # Clean up memory
            del array_data
            del aligned_data
            del temp_ds
            gc.collect()
        
        # Now open all files lazily and merge
        datasets = []
        for band_name, temp_path in temp_files:
            ds = xr.open_dataset(temp_path, chunks={'x': 100, 'y': 100})
            datasets.append(ds)
        
        # Merge all datasets
        merged = xr.merge(datasets)
        
        # Clean up temp files
        for _, temp_path in temp_files:
            try:
                temp_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_path}: {e}")
        
        # Add attributes
        merged.attrs.update({
            'title': 'Merged Resampled Dataset (Chunked)',
            'created_by': 'pipeline_merge_stage',
            'created_at': datetime.now().isoformat(),
            'common_bounds': common_bounds,
            'resolution': resolution
        })
        
        logger.info(f"Created chunked merged dataset with shape: {dict(merged.sizes)}")
        return merged