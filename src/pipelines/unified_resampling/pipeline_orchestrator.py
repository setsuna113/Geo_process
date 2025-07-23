# src/pipelines/unified_resampling/pipeline_orchestrator.py
"""Main orchestrator for unified resampling pipeline."""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import xarray as xr
import json
import numpy as np

from src.config.config import Config
from src.database.connection import DatabaseManager
from src.database.schema import schema
from src.processors.data_preparation.resampling_processor import ResamplingProcessor, ResampledDatasetInfo
from src.processors.data_preparation.raster_merger import RasterMerger
from src.spatial_analysis.som.som_trainer import SOMAnalyzer
from .dataset_processor import DatasetProcessor
from .resampling_workflow import ResamplingWorkflow
from .validation_checks import ValidationChecks

logger = logging.getLogger(__name__)


def clean_nan_for_json(obj):
    """Recursively clean NaN values from nested dictionaries and lists for JSON serialization."""
    if isinstance(obj, dict):
        return {key: clean_nan_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_for_json(item) for item in obj]
    elif isinstance(obj, (np.ndarray, np.generic)):
        # Convert numpy arrays/scalars to Python types
        if obj.ndim == 0:  # scalar
            val = float(obj)
            return None if np.isnan(val) or np.isinf(val) else val
        else:  # array
            cleaned = obj.tolist()
            return clean_nan_for_json(cleaned)
    elif isinstance(obj, float):
        return None if np.isnan(obj) or np.isinf(obj) else obj
    else:
        return obj


class UnifiedResamplingPipeline:
    """
    Main orchestrator for unified resampling pipeline.
    
    Pipeline flow:
    1. Load and validate dataset configurations
    2. Resample each dataset to target resolution
    3. Store resampled datasets in database
    4. Merge resampled datasets into unified multi-band dataset
    5. Perform SOM analysis on merged dataset
    6. Save results and generate reports
    """
    
    def __init__(self, config: Config, db_connection: DatabaseManager):
        self.config = config
        self.db = db_connection
        
        # Initialize pipeline components
        self.resampling_processor = ResamplingProcessor(config, db_connection)
        self.dataset_processor = DatasetProcessor(config, db_connection)
        self.resampling_workflow = ResamplingWorkflow(config)
        self.raster_merger = RasterMerger(config, db_connection)
        self.som_analyzer = SOMAnalyzer(config, db_connection)
        self.validator = ValidationChecks(config)
        
        # Pipeline state
        self.experiment_id = None
        self.resampled_datasets = []
        self.merged_dataset = None
        self.som_results = None
        
        logger.info("UnifiedResamplingPipeline initialized")
    
    def run_complete_pipeline(self, experiment_name: str, 
                             description: str = None,
                             skip_existing: bool = True) -> Dict[str, Any]:
        """
        Run the complete resampling pipeline.
        
        Args:
            experiment_name: Name for this experimental run
            description: Optional description
            skip_existing: Skip datasets that are already resampled
            
        Returns:
            Dictionary with pipeline results and metadata
        """
        logger.info(f"🚀 Starting unified resampling pipeline: {experiment_name}")
        
        try:
            # Create experiment record
            self.experiment_id = self._create_experiment(experiment_name, description)
            
            # Phase 1: Dataset resampling
            logger.info("📊 Phase 1: Dataset Resampling")
            resampled_info = self._run_resampling_phase(skip_existing)
            
            # Phase 2: Dataset merging
            logger.info("🔗 Phase 2: Dataset Merging")
            merged_data = self._run_merging_phase(resampled_info)
            
            # Phase 3: SOM analysis
            logger.info("🧠 Phase 3: SOM Analysis")
            som_results = self._run_analysis_phase(merged_data)
            
            # Phase 4: Results and cleanup
            logger.info("💾 Phase 4: Results Generation")
            final_results = self._finalize_results(som_results)
            
            # Update experiment status
            self._update_experiment_status('completed', final_results)
            
            logger.info("✅ Unified resampling pipeline completed successfully!")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            if self.experiment_id:
                self._update_experiment_status('failed', error_message=str(e))
            raise
    
    def _create_experiment(self, name: str, description: str = None) -> str:
        """Create experiment record."""
        config_dict = {
            'pipeline_type': 'unified_resampling',
            'target_resolution': self.config.get('resampling.target_resolution'),
            'datasets': self.config.get('datasets.target_datasets', []),
            'resampling_strategies': self.config.get('resampling.strategies', {}),
            'som_config': self.config.get('som_analysis', {}),
            'created_by': 'UnifiedResamplingPipeline'
        }
        
        experiment_id = schema.create_experiment(
            name=name,
            description=description or f"Unified resampling pipeline run at {datetime.now()}",
            config=config_dict
        )
        
        logger.info(f"Created experiment: {experiment_id}")
        return experiment_id
    
    def _run_resampling_phase(self, skip_existing: bool) -> List[ResampledDatasetInfo]:
        """Run dataset resampling phase."""
        logger.info("Starting resampling phase...")
        
        # Get dataset configurations
        dataset_configs = self.config.get('datasets.target_datasets', [])
        enabled_datasets = [ds for ds in dataset_configs if ds.get('enabled', True)]
        
        logger.info(f"Processing {len(enabled_datasets)} datasets")
        
        resampled_datasets = []
        
        for i, dataset_config in enumerate(enabled_datasets, 1):
            dataset_name = dataset_config['name']
            logger.info(f"({i}/{len(enabled_datasets)}) Processing: {dataset_name}")
            
            # Check if already resampled
            if skip_existing:
                existing = self.resampling_processor.get_resampled_dataset(dataset_name)
                if existing and existing.target_resolution == self.config.get('resampling.target_resolution'):
                    logger.info(f"✓ Using existing resampled dataset: {dataset_name}")
                    resampled_datasets.append(existing)
                    continue
            
            # Validate dataset configuration
            is_valid, error_msg = self.validator.validate_dataset_config(dataset_config)
            if not is_valid:
                logger.error(f"Dataset validation failed for {dataset_name}: {error_msg}")
                continue
            
            # Resample dataset
            try:
                resampled_info = self.resampling_processor.resample_dataset(dataset_config)
                resampled_datasets.append(resampled_info)
                logger.info(f"✅ Successfully resampled: {dataset_name}")
                
            except Exception as e:
                logger.error(f"Failed to resample {dataset_name}: {e}")
                continue
        
        self.resampled_datasets = resampled_datasets
        logger.info(f"Resampling phase completed: {len(resampled_datasets)} datasets processed")
        return resampled_datasets
    
    def _run_merging_phase(self, resampled_info: List[ResampledDatasetInfo]) -> xr.Dataset:
        """Run dataset merging phase."""
        logger.info("Starting merging phase...")
        
        if len(resampled_info) < 2:
            raise ValueError("Need at least 2 resampled datasets for merging")
        
        # Prepare raster names for merger
        raster_names = {}
        band_names = []
        
        for info in resampled_info:
            raster_names[info.band_name] = info.name
            band_names.append(info.band_name)
        
        logger.info(f"Merging {len(band_names)} bands: {band_names}")
        
        # Load and merge resampled datasets directly from database
        merged_data = self._merge_resampled_datasets(resampled_info)
        self.merged_dataset = merged_data
        
        logger.info(f"✅ Merged dataset shape: {dict(merged_data.sizes)}")
        logger.info(f"   Bands: {list(merged_data.data_vars)}")
        
        return merged_data
    
    def _run_analysis_phase(self, merged_data: xr.Dataset) -> Dict[str, Any]:
        """Run SOM analysis phase."""
        logger.info("Starting SOM analysis phase...")
        
        # Get SOM configuration
        som_config = self.config.get('som_analysis', {})
        
        som_params = {
            'grid_size': som_config.get('default_grid_size', [8, 8]),
            'iterations': som_config.get('iterations', 1000),
            'sigma': som_config.get('sigma', 1.5),
            'learning_rate': som_config.get('learning_rate', 0.5),
            'neighborhood_function': som_config.get('neighborhood_function', 'gaussian'),
            'random_seed': som_config.get('random_seed', 42)
        }
        
        logger.info(f"SOM parameters: {som_params}")
        
        # Run SOM analysis
        som_results = self.som_analyzer.analyze(
            data=merged_data,
            **som_params
        )
        
        self.som_results = som_results
        logger.info("✅ SOM analysis completed")
        
        return som_results
    
    def _finalize_results(self, som_results: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize and save results."""
        logger.info("Finalizing results...")
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f"UnifiedResampling_SOM_{timestamp}"
        
        # Save SOM results
        saved_path = self.som_analyzer.save_results(som_results, output_name)
        
        # Compile final results
        final_results = {
            'experiment_id': self.experiment_id,
            'resampled_datasets': [
                {
                    'name': info.name,
                    'resolution': info.target_resolution,
                    'shape': info.shape,
                    'method': info.resampling_method
                }
                for info in self.resampled_datasets
            ],
            'merged_dataset': {
                'shape': dict(self.merged_dataset.sizes) if self.merged_dataset else None,
                'bands': list(self.merged_dataset.data_vars) if self.merged_dataset else []
            },
            'som_analysis': {
                'saved_path': str(saved_path),
                'statistics': clean_nan_for_json(som_results.statistics) if hasattr(som_results, 'statistics') else {}
            },
            'pipeline_metadata': {
                'completed_at': timestamp,
                'target_resolution': self.config.get('resampling.target_resolution'),
                'total_datasets_processed': len(self.resampled_datasets)
            }
        }
        
        logger.info(f"✅ Results saved to: {saved_path}")
        return final_results
    
    def _update_experiment_status(self, status: str, results: Dict[str, Any] = None, 
                                 error_message: str = None):
        """Update experiment status in database."""
        if self.experiment_id:
            schema.update_experiment_status(
                self.experiment_id, 
                status, 
                results, 
                error_message
            )
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'experiment_id': self.experiment_id,
            'resampled_datasets_count': len(self.resampled_datasets),
            'merged_dataset_available': self.merged_dataset is not None,
            'som_results_available': self.som_results is not None
        }
    
    def cleanup_intermediate_data(self, keep_resampled: bool = True):
        """Clean up intermediate processing data."""
        logger.info("Cleaning up intermediate data...")
        
        if not keep_resampled:
            # Drop resampled data tables
            for info in self.resampled_datasets:
                table_name = f"resampled_{info.name.replace('-', '_')}"
                try:
                    schema.drop_resampled_data_table(table_name)
                except Exception as e:
                    logger.warning(f"Failed to drop table {table_name}: {e}")
        
        logger.info("Cleanup completed")
    
    def _merge_resampled_datasets(self, resampled_info: List[ResampledDatasetInfo]) -> xr.Dataset:
        """Merge resampled datasets from database into a single xarray Dataset."""
        logger.info(f"Loading and merging {len(resampled_info)} resampled datasets from database...")
        
        # Initialize resampling processor to load data
        from src.database.connection import db
        processor = ResamplingProcessor(self.config, db)
        data_vars = {}
        coords = None
        
        for info in resampled_info:
            logger.info(f"Loading resampled data for: {info.name}")
            
            # Load the actual array data from database
            array_data = processor.load_resampled_data(info.name)
            if array_data is None:
                raise RuntimeError(f"Failed to load resampled data for {info.name}")
            
            logger.info(f"Loaded array shape: {array_data.shape}")
            
            # Create coordinates if not already created (use first dataset)
            if coords is None:
                # Extract spatial extent from the first dataset
                bounds = info.bounds  # [west, south, east, north]
                height, width = array_data.shape
                
                # Create coordinate arrays
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
                logger.info(f"Created coordinates: x={len(x_coords)}, y={len(y_coords)}")
            
            # Create DataArray for this band
            data_array = xr.DataArray(
                data=array_data,
                dims=['y', 'x'],
                coords=coords,
                attrs={
                    'long_name': f'{info.name} data',
                    'source_path': info.source_path,
                    'resampling_method': info.resampling_method,
                    'target_resolution': info.target_resolution,
                    'target_crs': info.target_crs
                }
            )
            
            # Use band_name as the variable name
            data_vars[info.band_name] = data_array
            logger.info(f"Added band '{info.band_name}' to merged dataset")
        
        # Create the merged dataset
        merged_dataset = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs={
                'title': 'Merged Resampled Biodiversity Data',
                'target_resolution': resampled_info[0].target_resolution,
                'target_crs': resampled_info[0].target_crs,
                'created_by': 'unified_resampling_pipeline',
                'created_at': datetime.now().isoformat()
            }
        )
        
        logger.info(f"✅ Successfully merged {len(data_vars)} bands into dataset")
        logger.info(f"   Dataset shape: {dict(merged_dataset.sizes)}")
        logger.info(f"   Bands: {list(merged_dataset.data_vars)}")
        
        return merged_dataset