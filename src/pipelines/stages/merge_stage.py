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
            
            # Skip NetCDF creation - go directly to ML-ready parquet
            logger.info("Creating ML-ready parquet dataset directly from database")
            
            # Create ML-ready parquet dataset from the database coordinate data
            ml_ready_path = self._create_ml_ready_parquet(context, resampled_datasets)
            
            # Store metadata in context for later stages  
            merged_metadata = {
                'experiment_id': context.experiment_id,
                'timestamp': datetime.now().isoformat(),
                'shape': dict(merged_dataset.sizes) if merged_dataset else {},
                'variables': list(merged_dataset.data_vars) if merged_dataset else [],
                'bounds': self._calculate_common_bounds(resampled_datasets),
                'resolution': resampled_datasets[0].target_resolution if resampled_datasets else None,
                'ml_ready_path': str(ml_ready_path),
                'file_size_mb': ml_ready_path.stat().st_size / (1024**2) if ml_ready_path.exists() else 0
            }
            
            context.set('merge_metadata', merged_metadata)
            context.set('ml_ready_path', str(ml_ready_path))
            logger.info(f"✅ ML-ready parquet created: {ml_ready_path}")
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
    
    def _merge_lazy_chunked(self, context, resampled_datasets):
        """True lazy chunked merge that processes data in spatial tiles without loading full arrays."""
        from src.processors.data_preparation.resampling_processor import ResamplingProcessor
        processor = ResamplingProcessor(context.config, context.db)
        
        # Calculate common bounds and resolution
        common_bounds = self._calculate_common_bounds(resampled_datasets)
        resolution = resampled_datasets[0].target_resolution
        
        # Create common coordinates
        coords = self._create_common_coordinates(common_bounds, resolution)
        common_shape = (len(coords['y']), len(coords['x']))
        
        logger.info(f"Lazy merge: common shape {common_shape}, processing in chunks")
        
        # Chunk size from config or default
        chunk_size = context.config.get('processing.merge_chunk_size', 500)
        
        # Create output NetCDF file with structure but no data yet
        output_path = context.output_dir / 'merged_dataset.nc'
        
        # Initialize the output dataset with proper dimensions and coordinates
        output_ds = xr.Dataset(
            coords={
                'x': coords['x'],
                'y': coords['y']
            },
            attrs={
                'title': 'Merged Resampled Dataset (Lazy Chunked)',
                'created_by': 'pipeline_merge_stage',
                'created_at': datetime.now().isoformat(),
                'common_bounds': common_bounds,
                'resolution': resolution
            }
        )
        
        # Use netCDF4 directly for better control over lazy allocation
        import netCDF4
        
        # Create the file and dimensions
        with netCDF4.Dataset(output_path, 'w', format='NETCDF4') as nc:
            # Create dimensions
            nc.createDimension('y', common_shape[0])
            nc.createDimension('x', common_shape[1])
            
            # Create coordinate variables
            y_var = nc.createVariable('y', 'f8', ('y',))
            y_var[:] = coords['y'].values
            y_var.units = 'degrees_north'
            y_var.standard_name = 'latitude'
            
            x_var = nc.createVariable('x', 'f8', ('x',))
            x_var[:] = coords['x'].values
            x_var.units = 'degrees_east'
            x_var.standard_name = 'longitude'
            
            # Create data variables (without allocating data)
            for info in resampled_datasets:
                var = nc.createVariable(
                    info.band_name, 
                    'f4', 
                    ('y', 'x'),
                    chunksizes=(min(chunk_size, common_shape[0]), min(chunk_size, common_shape[1])),
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
            nc.common_bounds = str(common_bounds)
            nc.resolution = resolution
        
        # Now process in chunks and write to the file
        total_chunks = ((common_shape[0] + chunk_size - 1) // chunk_size) * \
                      ((common_shape[1] + chunk_size - 1) // chunk_size)
        chunk_count = 0
        
        for y_start in range(0, common_shape[0], chunk_size):
            y_end = min(y_start + chunk_size, common_shape[0])
            
            for x_start in range(0, common_shape[1], chunk_size):
                x_end = min(x_start + chunk_size, common_shape[1])
                chunk_count += 1
                
                logger.info(f"Processing chunk {chunk_count}/{total_chunks}: "
                           f"y[{y_start}:{y_end}], x[{x_start}:{x_end}]")
                
                # Process this spatial chunk for all datasets
                chunk_data = {}
                
                for info in resampled_datasets:
                    # Calculate the bounds for this chunk in the dataset's coordinate system
                    chunk_bounds = self._calculate_chunk_bounds(
                        y_start, y_end, x_start, x_end,
                        common_bounds, resolution
                    )
                    
                    # Calculate corresponding indices in the source dataset
                    src_indices = self._calculate_source_indices(
                        chunk_bounds, info.bounds, info.shape, info.target_resolution
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
                    
                    chunk_array = processor.load_resampled_data_chunk(
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
                        chunk_bounds, (y_end - y_start, x_end - x_start),
                        resolution
                    )
                    
                    chunk_data[info.band_name] = aligned_chunk
                
                # Write this chunk to the output file
                # NetCDF doesn't support true append mode, so we need to use a different approach
                import netCDF4
                with netCDF4.Dataset(output_path, 'r+') as nc:
                    for band_name, data in chunk_data.items():
                        nc.variables[band_name][y_start:y_end, x_start:x_end] = data
                
                # Force garbage collection after each chunk
                del chunk_data
                gc.collect()
        
        logger.info(f"✅ Lazy chunked merge completed: {total_chunks} chunks processed")
        
        # Return reference to the output file
        return xr.open_dataset(output_path, chunks={'x': chunk_size, 'y': chunk_size})
    
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
    
    def _create_ml_ready_parquet(self, context, resampled_datasets):
        """Create ML-ready parquet directly from database coordinate data."""
        import pandas as pd
        
        logger.info("Loading coordinate data from database tables")
        
        # Collect all coordinate data from passthrough tables
        all_data = []
        
        for info in resampled_datasets:
            logger.info(f"Loading data for {info.name}")
            
            # Load coordinate data from database table
            table_name = f"passthrough_{info.name.replace('-', '_')}"
            
            with context.db.get_connection() as conn:
                with conn.cursor() as cur:
                    # Check if table uses row_idx/col_idx or x/y structure
                    cur.execute(f"""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = %s AND table_schema = 'public'
                    """, (table_name,))
                    columns = [row[0] for row in cur.fetchall()]
                    
                    if 'x' in columns and 'y' in columns:
                        # Standard coordinate structure
                        cur.execute(f"SELECT x, y, value FROM {table_name} WHERE value IS NOT NULL AND value != 0")
                    else:
                        # Passthrough structure - need to convert row_idx/col_idx to coordinates
                        logger.info(f"Converting passthrough table {table_name} to coordinates")
                        
                        # Get bounds and resolution from info
                        min_x, min_y, max_x, max_y = info.bounds
                        resolution = info.target_resolution
                        
                        cur.execute(f"SELECT row_idx, col_idx, value FROM {table_name} WHERE value IS NOT NULL AND value != 0")
                        rows = cur.fetchall()
                        
                        # Convert to coordinates
                        coord_data = []
                        for row_idx, col_idx, value in rows:
                            x = min_x + (col_idx + 0.5) * resolution
                            y = max_y - (row_idx + 0.5) * resolution
                            coord_data.append((x, y, value))
                        
                        rows = coord_data
                    
                    if 'x' in columns and 'y' in columns:
                        rows = cur.fetchall()
                    
                    logger.info(f"Loaded {len(rows):,} coordinate points for {info.name}")
                    
                    # Convert to DataFrame with renamed column
                    if rows:
                        df = pd.DataFrame(rows, columns=['x', 'y', info.name])
                        all_data.append(df)
        
        if not all_data:
            raise ValueError("No coordinate data found in any dataset")
        
        # Merge all datasets on coordinates
        logger.info("Merging coordinate datasets")
        merged_df = all_data[0]
        
        for df in all_data[1:]:
            merged_df = merged_df.merge(df, on=['x', 'y'], how='outer')
        
        # Sort by coordinates for consistent ordering
        merged_df = merged_df.sort_values(['y', 'x']).reset_index(drop=True)
        
        # Save as parquet
        output_path = context.output_dir / 'ml_ready_aligned_data.parquet'
        merged_df.to_parquet(output_path, index=False)
        
        logger.info(f"Created ML-ready parquet with {len(merged_df):,} rows and {len(merged_df.columns)} columns")
        logger.info(f"Columns: {list(merged_df.columns)}")
        
        return output_path