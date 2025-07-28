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
            
            # Determine merging strategy based on data size
            total_size_estimate = sum(
                info.shape[0] * info.shape[1] * 4 / (1024**2)  # MB
                for info in resampled_datasets
            ) * len(resampled_datasets)
            
            # Check if we should use lazy chunked processing
            lazy_threshold_mb = context.config.get('processing.lazy_merge_threshold_mb', 500)
            use_lazy = total_size_estimate > lazy_threshold_mb
            
            if use_lazy:
                logger.info(f"Using lazy chunked merging (estimated size: {total_size_estimate:.0f}MB)")
                merged_dataset = self._merge_lazy_chunked(context, resampled_datasets)
            elif total_size_estimate > 1000 or (self.processing_config and self.processing_config.enable_chunking):
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
            
            # Clean up memory for lazy/chunked processing
            if use_lazy or (total_size_estimate > 1000):
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
                'chunked_merge': use_lazy or (total_size_estimate > 1000)
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
        
        # ADD validation and debugging
        logger.info(f"Aligning data: shape={array_data.shape}, "
                   f"data_bounds={data_bounds}, common_bounds={common_bounds}")
        
        # Create output array filled with appropriate value based on dtype
        if np.issubdtype(array_data.dtype, np.integer):
            # For integer data, use 0 or a specific nodata value instead of NaN
            fill_value = 0  # Could also use a specific nodata value from config
            aligned_data = np.full(common_shape, fill_value, dtype=array_data.dtype)
        else:
            # For floating point data, NaN is fine
            aligned_data = np.full(common_shape, np.nan, dtype=array_data.dtype)
        
        # Calculate offsets with validation
        x_offset = int(np.round((data_bounds[0] - common_bounds[0]) / resolution))
        y_offset = int(np.round((common_bounds[3] - data_bounds[3]) / resolution))
        
        # Validate offsets
        if x_offset < 0 or y_offset < 0:
            logger.error(f"Negative offsets detected: x={x_offset}, y={y_offset}")
            logger.error(f"This indicates bounds mismatch - data may be corrupted!")
        
        # Log for debugging
        logger.info(f"Calculated offsets: x={x_offset}, y={y_offset}")
        
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
    
    def _merge_lazy_chunked(self, context, resampled_datasets):
        """Main orchestrator for lazy chunked merging."""
        try:
            # Step 1: Prepare merge operation
            merge_config = self._prepare_merge_config(context, resampled_datasets)
            
            # Step 2: Initialize output file
            output_path = self._initialize_output_file(context, merge_config)
            
            # Step 3: Process chunks
            chunk_results = self._process_all_chunks(merge_config, output_path)
            
            # Step 4: Finalize and validate
            self._finalize_merge(output_path, chunk_results)
            
            return xr.open_dataset(output_path, chunks={'x': merge_config['chunk_size'], 'y': merge_config['chunk_size']})
            
        except Exception as e:
            logger.error(f"Merge failed: {e}")
            self._cleanup_failed_merge(context.output_dir / 'merged_dataset.nc')
            raise

    def _prepare_merge_config(self, context, resampled_datasets):
        """Prepare configuration for merge operation."""
        from src.processors.data_preparation.resampling_processor import ResamplingProcessor
        
        # Calculate common bounds and resolution
        common_bounds = self._calculate_common_bounds(resampled_datasets)
        resolution = resampled_datasets[0].target_resolution
        
        # Create common coordinates
        coords = self._create_common_coordinates(common_bounds, resolution)
        common_shape = (len(coords['y']), len(coords['x']))
        
        # Chunk size from config or default
        chunk_size = self._calculate_optimal_chunk_size(context, resampled_datasets)
        
        config = {
            'datasets': resampled_datasets,
            'processor': ResamplingProcessor(context.config, context.db),
            'chunk_size': chunk_size,
            'common_bounds': common_bounds,
            'common_shape': common_shape,
            'coords': coords,
            'resolution': resolution,
            'temp_files': []
        }
        
        # Validate configuration
        self._validate_merge_config(config)
        
        logger.info(f"Lazy merge: common shape {common_shape}, chunk size {chunk_size}")
        
        return config
    
    def _calculate_optimal_chunk_size(self, context, resampled_datasets):
        """Calculate optimal chunk size based on available memory."""
        from src.config import config
        
        # Get available memory (in GB)
        available_memory_gb = self._get_available_memory()
        
        # Calculate based on dataset sizes and memory config
        total_data_size = sum(getattr(d, 'size_gb', 1.0) for d in resampled_datasets)
        num_datasets = len(resampled_datasets)
        
        # Use configuration or defaults
        memory_factor = config.get('merge.memory_factor', 0.5)
        min_chunk_size = config.get('merge.min_chunk_size', 500)  # Updated default
        max_chunk_size = config.get('merge.max_chunk_size', 2000)
        
        # Calculate optimal size (in pixels)
        if available_memory_gb and available_memory_gb > 0:
            # Estimate memory per chunk (GB = pixels^2 * datasets * 4 bytes / 1GB)
            optimal_size = int(np.sqrt((available_memory_gb * memory_factor * 1024**3) / (num_datasets * 4)))
        else:
            optimal_size = context.config.get('processing.merge_chunk_size', 500)
        
        return max(min_chunk_size, min(optimal_size, max_chunk_size))
    
    def _get_available_memory(self):
        """Get available system memory in GB."""
        try:
            import psutil
            return psutil.virtual_memory().available / (1024**3)
        except:
            return None
    
    def _validate_merge_config(self, config):
        """Validate merge configuration."""
        if not config['datasets']:
            raise ValueError("No datasets provided for merge")
        
        if config['chunk_size'] <= 0:
            raise ValueError(f"Invalid chunk size: {config['chunk_size']}")
        
        if not all(hasattr(d, 'target_resolution') for d in config['datasets']):
            raise ValueError("All datasets must have target_resolution")
    
    def _initialize_output_file(self, context, merge_config):
        """Initialize the output NetCDF file with structure but no data."""
        output_path = context.output_dir / 'merged_dataset.nc'
        
        # Create file with safe atomic write (moved from original method)
        import netCDF4
        from src.pipelines.utils.file_utils import safe_write_file
        
        def write_nc(path):
            with netCDF4.Dataset(path, 'w', format='NETCDF4') as nc:
                # Create dimensions
                nc.createDimension('y', merge_config['common_shape'][0])
                nc.createDimension('x', merge_config['common_shape'][1])
                
                # Create coordinate variables
                y_var = nc.createVariable('y', 'f8', ('y',))
                y_var[:] = merge_config['coords']['y'].values
                y_var.units = 'degrees_north'
                y_var.standard_name = 'latitude'
                
                x_var = nc.createVariable('x', 'f8', ('x',))
                x_var[:] = merge_config['coords']['x'].values
                x_var.units = 'degrees_east'
                x_var.standard_name = 'longitude'
                
                # Create data variables (without allocating data)
                for info in merge_config['datasets']:
                    var = nc.createVariable(
                        info.band_name, 
                        'f4', 
                        ('y', 'x'),
                        chunksizes=(min(merge_config['chunk_size'], merge_config['common_shape'][0]), 
                                  min(merge_config['chunk_size'], merge_config['common_shape'][1])),
                        zlib=True,
                        complevel=4,
                        fill_value=np.nan
                    )
                    # Set attributes
                    var.long_name = f'{info.name} data'
                    var.source_path = str(info.source_path)
                    var.resampling_method = info.resampling_method
                    var.original_bounds = str(info.bounds)
                    var.original_shape = str(info.shape)
                
                # Set global attributes
                nc.title = 'Merged Resampled Dataset (Lazy Chunked)'
                nc.created_by = 'pipeline_merge_stage'
                nc.created_at = datetime.now().isoformat()
                nc.common_bounds = str(merge_config['common_bounds'])
                nc.resolution = merge_config['resolution']
        
        # Create file with safe atomic write
        safe_write_file(output_path, write_nc)
        
        return output_path
    
    def _process_all_chunks(self, merge_config, output_path):
        """Process all chunks with progress tracking."""
        chunk_results = []
        total_chunks = self._calculate_total_chunks(merge_config)
        
        chunk_count = 0
        for y_start in range(0, merge_config['common_shape'][0], merge_config['chunk_size']):
            y_end = min(y_start + merge_config['chunk_size'], merge_config['common_shape'][0])
            
            for x_start in range(0, merge_config['common_shape'][1], merge_config['chunk_size']):
                x_end = min(x_start + merge_config['chunk_size'], merge_config['common_shape'][1])
                chunk_count += 1
                
                logger.info(f"Processing chunk {chunk_count}/{total_chunks}: "
                           f"y[{y_start}:{y_end}], x[{x_start}:{x_end}]")
                
                result = self._process_single_chunk(
                    chunk_idx=chunk_count,
                    chunk_bounds=(y_start, y_end, x_start, x_end),
                    merge_config=merge_config,
                    output_path=output_path
                )
                chunk_results.append(result)
                
                # Memory management
                if chunk_count % 10 == 0:
                    self._cleanup_memory()
        
        return chunk_results
    
    def _calculate_total_chunks(self, merge_config):
        """Calculate total number of chunks to process."""
        y_chunks = (merge_config['common_shape'][0] + merge_config['chunk_size'] - 1) // merge_config['chunk_size']
        x_chunks = (merge_config['common_shape'][1] + merge_config['chunk_size'] - 1) // merge_config['chunk_size']
        return y_chunks * x_chunks
    
    def _process_single_chunk(self, chunk_idx, chunk_bounds, merge_config, output_path):
        """Process a single chunk of data."""
        try:
            y_start, y_end, x_start, x_end = chunk_bounds
            
            # Load chunk data from all datasets
            chunk_data = self._load_chunk_data(chunk_bounds, merge_config)
            
            # Write to output file
            self._write_chunk_to_output(chunk_data, chunk_bounds, output_path)
            
            return {
                'chunk_idx': chunk_idx,
                'bounds': chunk_bounds,
                'status': 'success',
                'records_written': len(chunk_data)
            }
            
        except Exception as e:
            logger.error(f"Failed to process chunk {chunk_idx}: {e}")
            return {
                'chunk_idx': chunk_idx,
                'bounds': chunk_bounds,
                'status': 'failed',
                'error': str(e)
            }
    
    def _load_chunk_data(self, chunk_bounds, merge_config):
        """Load data for a single chunk from all datasets."""
        y_start, y_end, x_start, x_end = chunk_bounds
        chunk_data = {}
        
        for info in merge_config['datasets']:
            # Calculate the bounds for this chunk in the dataset's coordinate system
            chunk_geographic_bounds = self._calculate_chunk_bounds(
                y_start, y_end, x_start, x_end,
                merge_config['common_bounds'], merge_config['resolution']
            )
            
            # Calculate corresponding indices in the source dataset
            src_indices = self._calculate_source_indices(
                chunk_geographic_bounds, info.bounds, info.shape, info.target_resolution
            )
            
            if src_indices is None:
                # This chunk doesn't overlap with this dataset
                chunk_data[info.band_name] = np.full(
                    (y_end - y_start, x_end - x_start), 
                    np.nan, 
                    dtype=np.float32
                )
                continue
                
            # Load only this chunk from the database
            src_y_start, src_y_end, src_x_start, src_x_end = src_indices
            
            logger.debug(f"Loading chunk from {info.name}: "
                        f"[{src_y_start}:{src_y_end}, {src_x_start}:{src_x_end}]")
            
            chunk_array = merge_config['processor'].load_resampled_data_chunk(
                info.name,
                src_y_start, src_y_end,
                src_x_start, src_x_end
            )
            
            if chunk_array is None:
                logger.warning(f"Failed to load chunk from {info.name}")
                chunk_data[info.band_name] = np.full(
                    (y_end - y_start, x_end - x_start),
                    np.nan,
                    dtype=np.float32
                )
                continue
            
            # Align this chunk to the common grid chunk
            aligned_chunk = self._align_chunk_to_common_grid(
                chunk_array, src_indices, info.bounds,
                chunk_geographic_bounds, (y_end - y_start, x_end - x_start),
                merge_config['resolution']
            )
            
            chunk_data[info.band_name] = aligned_chunk
            
        return chunk_data
    
    def _write_chunk_to_output(self, chunk_data, chunk_bounds, output_path):
        """Write chunk data to the output NetCDF file."""
        y_start, y_end, x_start, x_end = chunk_bounds
        
        # NetCDF doesn't support true append mode, so we need to use a different approach
        import netCDF4
        with netCDF4.Dataset(output_path, 'r+') as nc:
            for band_name, data in chunk_data.items():
                nc.variables[band_name][y_start:y_end, x_start:x_end] = data
    
    def _cleanup_memory(self):
        """Force garbage collection to free memory."""
        import gc
        gc.collect()
    
    def _finalize_merge(self, output_path, chunk_results):
        """Finalize merge and validate results."""
        # Check for failed chunks
        failed_chunks = [r for r in chunk_results if r['status'] == 'failed']
        if failed_chunks:
            raise RuntimeError(f"Merge failed for {len(failed_chunks)} chunks")
        
        # Add metadata
        logger.info(f"✅ Lazy chunked merge completed: {len(chunk_results)} chunks processed")
        
        # Validate output
        if not self._validate_merged_output(output_path):
            raise ValueError("Merged output validation failed")
    
    def _validate_merged_output(self, output_path):
        """Basic validation of merged output."""
        try:
            # Check if file exists and is readable
            import netCDF4
            with netCDF4.Dataset(output_path, 'r') as nc:
                # Check dimensions exist
                if 'x' not in nc.dimensions or 'y' not in nc.dimensions:
                    return False
                # Basic checks passed
                return True
        except Exception as e:
            logger.error(f"Output validation failed: {e}")
            return False
    
    def _cleanup_failed_merge(self, output_path):
        """Clean up after a failed merge."""
        if output_path and Path(output_path).exists():
            try:
                Path(output_path).unlink()
                logger.info(f"Cleaned up failed merge output: {output_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up {output_path}: {e}")
    
    # Refactoring complete - original 164-line method now broken into 8+ focused methods
    
    def _calculate_chunk_bounds(self, y_start: int, y_end: int, x_start: int, x_end: int,
                               common_bounds: Tuple[float, float, float, float],
                               resolution: float) -> Tuple[float, float, float, float]:
        """Calculate geographic bounds for a chunk given pixel indices."""
        min_x = common_bounds[0] + x_start * resolution
        max_x = common_bounds[0] + x_end * resolution
        max_y = common_bounds[3] - y_start * resolution
        min_y = common_bounds[3] - y_end * resolution
        return (min_x, min_y, max_x, max_y)
    
    def _calculate_source_indices(self, chunk_bounds: Tuple[float, float, float, float],
                                 dataset_bounds: Tuple[float, float, float, float],
                                 dataset_shape: Tuple[int, int],
                                 dataset_resolution: float) -> Optional[Tuple[int, int, int, int]]:
        """Calculate source dataset indices that correspond to the chunk bounds."""
        # Check if chunk overlaps with dataset
        if (chunk_bounds[2] <= dataset_bounds[0] or chunk_bounds[0] >= dataset_bounds[2] or
            chunk_bounds[3] <= dataset_bounds[1] or chunk_bounds[1] >= dataset_bounds[3]):
            return None  # No overlap
        
        # Calculate overlapping bounds
        overlap_min_x = max(chunk_bounds[0], dataset_bounds[0])
        overlap_max_x = min(chunk_bounds[2], dataset_bounds[2])
        overlap_min_y = max(chunk_bounds[1], dataset_bounds[1])
        overlap_max_y = min(chunk_bounds[3], dataset_bounds[3])
        
        # Convert to pixel indices in the dataset
        x_start = int(np.floor((overlap_min_x - dataset_bounds[0]) / dataset_resolution))
        x_end = int(np.ceil((overlap_max_x - dataset_bounds[0]) / dataset_resolution))
        y_start = int(np.floor((dataset_bounds[3] - overlap_max_y) / dataset_resolution))
        y_end = int(np.ceil((dataset_bounds[3] - overlap_min_y) / dataset_resolution))
        
        # Clamp to dataset bounds
        x_start = max(0, x_start)
        x_end = min(dataset_shape[1], x_end)
        y_start = max(0, y_start)
        y_end = min(dataset_shape[0], y_end)
        
        return (y_start, y_end, x_start, x_end)
    
    def _align_chunk_to_common_grid(self, chunk_data: np.ndarray,
                                   src_indices: Tuple[int, int, int, int],
                                   dataset_bounds: Tuple[float, float, float, float],
                                   chunk_bounds: Tuple[float, float, float, float],
                                   output_shape: Tuple[int, int],
                                   resolution: float) -> np.ndarray:
        """Align a data chunk to the common grid coordinates."""
        # This is a simplified version - in practice you'd need proper resampling
        # For now, just place the data in the correct position
        output = np.full(output_shape, np.nan, dtype=chunk_data.dtype)
        
        # Calculate where this data goes in the output chunk
        # This is simplified - real implementation would need proper coordinate mapping
        # Ensure we don't exceed output dimensions
        rows_to_copy = min(chunk_data.shape[0], output_shape[0])
        cols_to_copy = min(chunk_data.shape[1], output_shape[1])
        
        if rows_to_copy < chunk_data.shape[0] or cols_to_copy < chunk_data.shape[1]:
            logger.warning(f"Chunk data shape {chunk_data.shape} exceeds output shape {output_shape}, trimming to fit")
        
        output[:rows_to_copy, :cols_to_copy] = chunk_data[:rows_to_copy, :cols_to_copy]
        
        return output
    
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
        
        # Clean up temp files with proper error handling
        from src.pipelines.utils.file_utils import cleanup_temp_file, register_temp_file
        
        # Register temp files for cleanup on exit
        for _, temp_path in temp_files:
            register_temp_file(temp_path)
            
        # Attempt immediate cleanup
        for _, temp_path in temp_files:
            cleanup_temp_file(temp_path, ignore_errors=True)
        
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