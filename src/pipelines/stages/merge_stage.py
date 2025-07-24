# src/pipelines/stages/merge_stage.py
"""Dataset merging stage."""

from typing import List, Tuple
import logging
import xarray as xr
from datetime import datetime

from .base_stage import PipelineStage, StageResult

logger = logging.getLogger(__name__)


class MergeStage(PipelineStage):
    """Stage for merging resampled datasets."""
    
    @property
    def name(self) -> str:
        return "merge"
    
    @property
    def dependencies(self) -> List[str]:
        return ["resample"]
    
    @property
    def memory_requirements(self) -> float:
        return 16.0  # GB - needs memory for all datasets
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate merge configuration."""
        return True, []
    
    def execute(self, context) -> StageResult:
        """Merge resampled datasets into unified dataset."""
        logger.info("Starting merge stage")
        
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
            
            # Load and merge datasets
            logger.info(f"Merging {len(resampled_datasets)} datasets")
            
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
            merged_dataset = xr.Dataset(
                data_vars=data_vars,
                coords=coords,
                attrs={
                    'title': 'Merged Resampled Dataset',
                    'created_by': 'pipeline_merge_stage',
                    'created_at': datetime.now().isoformat()
                }
            )
            
            # Store in context
            context.set('merged_dataset', merged_dataset)
            
            # Save to file
            output_path = context.output_dir / 'merged_dataset.nc'
            merged_dataset.to_netcdf(output_path, encoding={
                var: {'zlib': True, 'complevel': 4} for var in merged_dataset.data_vars
            })
            
            metrics = {
                'datasets_merged': len(data_vars),
                'total_bands': len(merged_dataset.data_vars),
                'output_shape': dict(merged_dataset.sizes),
                'output_size_mb': output_path.stat().st_size / (1024**2)
            }
            
            return StageResult(
                success=True,
                data={'merged_dataset_path': str(output_path)},
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Merge stage failed: {e}")
            raise
