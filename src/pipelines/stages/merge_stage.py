# src/pipelines/stages/merge_stage.py (Updated)
"""Dataset merging stage with memory-aware processing."""

from typing import List, Tuple
import logging
import xarray as xr
from datetime import datetime
import gc

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
        # Dynamic based on number of datasets
        if self.processing_config and self.processing_config.enable_chunking:
            return 4.0  # GB - chunked mode
        return 16.0  # GB - full mode
    
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
            
            # Use chunked writing for large datasets
            if use_chunked:
                # Configure chunking for NetCDF
                encoding = {
                    var: {
                        'zlib': True, 
                        'complevel': 4,
                        'chunksizes': (100, 100)  # Chunk size for storage
                    } 
                    for var in merged_dataset.data_vars
                }
            else:
                encoding = {
                    var: {'zlib': True, 'complevel': 4} 
                    for var in merged_dataset.data_vars
                }
            
            merged_dataset.to_netcdf(output_path, encoding=encoding)
            
            # Clean up memory
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
    
    def _merge_standard(self, context, resampled_datasets):
        """Standard in-memory merge."""
        # Existing merge logic from original implementation
        from src.processors.data_preparation.resampling_processor import ResamplingProcessor
        processor = ResamplingProcessor(context.config, context.db)
        
        data_vars = {}
        coords = None
        
        for info in resampled_datasets:
            logger.info(f"Loading resampled data for: {info.name}")
            
            # Load array data
            array_data = processor.load_resampled_data(info.name)
            if array_data is None:
                logger.warning(f"Failed to load data for {info.name}")
                continue
            
            # Create coordinates if needed
            if coords is None:
                bounds = info.bounds
                height, width = array_data.shape
                
                x_coords = xr.DataArray(
                    data=[(bounds[0] + (i + 0.5) * info.target_resolution) for i in range(width)],
                    dims=['x'],
                    attrs={'long_name': 'longitude', 'units': 'degrees_east'}
                )
                
                y_coords = xr.DataArray(
                    data=[(bounds[3] - (i + 0.5) * info.target_resolution) for i in range(height)],
                    dims=['y'],
                    attrs={'long_name': 'latitude', 'units': 'degrees_north'}
                )
                
                coords = {'x': x_coords, 'y': y_coords}
            
            # Create DataArray
            data_array = xr.DataArray(
                data=array_data,
                dims=['y', 'x'],
                coords=coords,
                attrs={
                    'long_name': f'{info.name} data',
                    'source_path': info.source_path,
                    'resampling_method': info.resampling_method
                }
            )
            
            data_vars[info.band_name] = data_array
        
        # Create merged dataset
        return xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs={
                'title': 'Merged Resampled Dataset',
                'created_by': 'pipeline_merge_stage',
                'created_at': datetime.now().isoformat()
            }
        )
    
    def _merge_chunked(self, context, resampled_datasets):
        """Chunked merge for large datasets."""
        from src.processors.data_preparation.resampling_processor import ResamplingProcessor
        processor = ResamplingProcessor(context.config, context.db)
        
        # First, save each dataset as a separate chunked file
        temp_files = []
        coords = None
        
        for info in resampled_datasets:
            logger.info(f"Processing {info.name} for chunked merge")
            
            # Load data in chunks and save to temporary file
            temp_path = context.output_dir / f"temp_{info.band_name}.nc"
            
            # Load array data
            array_data = processor.load_resampled_data(info.name)
            if array_data is None:
                continue
            
            # Create coordinates if needed
            if coords is None:
                bounds = info.bounds
                height, width = array_data.shape
                
                x_coords = [(bounds[0] + (i + 0.5) * info.target_resolution) for i in range(width)]
                y_coords = [(bounds[3] - (i + 0.5) * info.target_resolution) for i in range(height)]
                coords = {'x': x_coords, 'y': y_coords}
            
            # Create temporary dataset with chunking
            temp_ds = xr.Dataset({
                info.band_name: xr.DataArray(
                    data=array_data,
                    dims=['y', 'x'],
                    coords=coords
                )
            })
            
            # Save with chunking
            temp_ds.to_netcdf(
                temp_path,
                encoding={
                    info.band_name: {
                        'chunks': (100, 100),
                        'zlib': True,
                        'complevel': 4
                    }
                }
            )
            
            temp_files.append((info.band_name, temp_path))
            
            # Clean up memory
            del array_data
            del temp_ds
            gc.collect()
        
        # Now open all files lazily and merge
        datasets = []
        for _, temp_path in temp_files:
            ds = xr.open_dataset(temp_path, chunks={'x': 100, 'y': 100})
            datasets.append(ds)
        
        # Merge all datasets
        merged = xr.merge(datasets)
        
        # Clean up temp files
        for _, temp_path in temp_files:
            temp_path.unlink()
        
        # Add attributes
        merged.attrs.update({
            'title': 'Merged Resampled Dataset (Chunked)',
            'created_by': 'pipeline_merge_stage',
            'created_at': datetime.now().isoformat()
        })
        
        return merged