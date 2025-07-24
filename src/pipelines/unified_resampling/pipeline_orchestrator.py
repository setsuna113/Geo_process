# src/pipelines/unified_resampling/pipeline_orchestrator.py
"""Enhanced pipeline orchestrator with progress tracking and checkpoint support."""

import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import xarray as xr
import numpy as np
import time
import signal

from src.config.config import Config
from src.database.connection import DatabaseManager
from src.database.schema import schema
from src.processors.data_preparation.resampling_processor import ResamplingProcessor, ResampledDatasetInfo
from src.processors.data_preparation.raster_merger import RasterMerger
from src.spatial_analysis.som.som_trainer import SOMAnalyzer
from src.core.progress_manager import get_progress_manager
from src.core.checkpoint_manager import get_checkpoint_manager
from src.core.signal_handler import get_signal_handler
from src.base.memory_manager import get_memory_manager
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
    Enhanced orchestrator with progress tracking and checkpoint support.
    
    Pipeline phases:
    1. Resampling: Resample each dataset to target resolution
    2. Merging: Merge resampled datasets into unified dataset
    3. Analysis: Perform spatial analysis (SOM, etc.)
    4. Export: Generate outputs and reports
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
        
        # Enhanced infrastructure
        self.progress_manager = get_progress_manager()
        self.checkpoint_manager = get_checkpoint_manager()
        self.signal_handler = get_signal_handler()
        self.memory_manager = get_memory_manager()
        
        # Pipeline state
        self.experiment_id = None
        self.pipeline_id = None
        self.current_phase = None
        self.resampled_datasets = []
        self.merged_dataset = None
        self.som_results = None
        
        # Checkpoint data
        self._checkpoint_data = {
            'completed_phases': [],
            'phase_results': {}
        }
        
        # Register signal handlers
        self._register_signal_handlers()
        
        logger.info("Enhanced UnifiedResamplingPipeline initialized")
    
    def run_complete_pipeline(self, 
                             experiment_name: str, 
                             description: Optional[str] = None,
                             skip_existing: bool = True,
                             resume_from_checkpoint: Optional[str] = None,
                             phases_to_run: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the complete resampling pipeline with checkpoint support.
        
        Args:
            experiment_name: Name for this experimental run
            description: Optional description
            skip_existing: Skip datasets that are already resampled
            resume_from_checkpoint: Checkpoint ID to resume from
            phases_to_run: Specific phases to run (None = all)
            
        Returns:
            Dictionary with pipeline results and metadata
        """
        logger.info(f"🚀 Starting unified resampling pipeline: {experiment_name}")
        
        # Create pipeline progress node
        self.pipeline_id = f"pipeline_{experiment_name}_{int(time.time())}"
        self.progress_manager.create_pipeline(
            self.pipeline_id,
            total_phases=4,  # resampling, merging, analysis, export
            metadata={'experiment_name': experiment_name}
        )
        self.progress_manager.start(self.pipeline_id)
        
        try:
            # Load checkpoint if resuming
            if resume_from_checkpoint:
                self._load_checkpoint(resume_from_checkpoint)
                logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
                logger.info(f"Completed phases: {self._checkpoint_data['completed_phases']}")
            
            # Create experiment record if not resuming
            if not self.experiment_id:
                self.experiment_id = self._create_experiment(experiment_name, description)
            
            # Define phases
            all_phases = ['resampling', 'merging', 'analysis', 'export']
            phases = phases_to_run or all_phases
            
            # Filter out completed phases if resuming
            phases_to_execute = [
                p for p in phases 
                if p not in self._checkpoint_data['completed_phases']
            ]
            
            logger.info(f"Phases to execute: {phases_to_execute}")
            
            # Execute phases
            for phase in phases_to_execute:
                if phase == 'resampling' and 'resampling' in phases:
                    self._run_phase('resampling', self._run_resampling_phase, skip_existing)
                
                elif phase == 'merging' and 'merging' in phases:
                    resampled_info = self._checkpoint_data['phase_results'].get(
                        'resampling', self.resampled_datasets
                    )
                    self._run_phase('merging', self._run_merging_phase, resampled_info)
                
                elif phase == 'analysis' and 'analysis' in phases:
                    merged_data = self._checkpoint_data['phase_results'].get(
                        'merging', self.merged_dataset
                    )
                    self._run_phase('analysis', self._run_analysis_phase, merged_data)
                
                elif phase == 'export' and 'export' in phases:
                    som_results = self._checkpoint_data['phase_results'].get(
                        'analysis', self.som_results
                    )
                    self._run_phase('export', self._finalize_results, som_results)
            
            # Get final results
            final_results = self._checkpoint_data['phase_results'].get('export', {})
            
            # Update experiment status
            self._update_experiment_status('completed', final_results)
            
            # Complete pipeline progress
            self.progress_manager.complete(self.pipeline_id, metadata={
                'phases_completed': len(self._checkpoint_data['completed_phases']),
                'total_datasets': len(self.resampled_datasets)
            })
            
            logger.info("✅ Unified resampling pipeline completed successfully!")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            
            # Save error checkpoint
            self._save_checkpoint(f"error_{experiment_name}_{int(time.time())}")
            
            # Update statuses
            if self.experiment_id:
                self._update_experiment_status('failed', error_message=str(e))
            
            self.progress_manager.complete(
                self.pipeline_id, 
                status='failed',
                metadata={'error': str(e)}
            )
            
            raise
    
    def _run_phase(self, phase_name: str, phase_func: Callable, *args, **kwargs) -> Any:
        """Run a pipeline phase with progress and checkpoint support."""
        logger.info(f"📊 Starting phase: {phase_name}")
        
        # Create phase progress node
        phase_id = f"{self.pipeline_id}_{phase_name}"
        self.progress_manager.create_phase(
            phase_id,
            self.pipeline_id,
            total_steps=100,  # Will be updated by phase
            metadata={'phase': phase_name}
        )
        self.progress_manager.start(phase_id)
        
        # Set current phase
        self.current_phase = phase_name
        
        try:
            # Execute phase
            result = phase_func(*args, **kwargs)
            
            # Store result and mark complete
            self._checkpoint_data['phase_results'][phase_name] = result
            self._checkpoint_data['completed_phases'].append(phase_name)
            
            # Save checkpoint
            checkpoint_id = f"{self.experiment_id}_{phase_name}_{int(time.time())}"
            self._save_checkpoint(checkpoint_id)
            
            # Complete phase progress
            self.progress_manager.complete(phase_id, metadata={
                'checkpoint_id': checkpoint_id
            })
            
            logger.info(f"✅ Phase completed: {phase_name}")
            return result
            
        except Exception as e:
            logger.error(f"Phase {phase_name} failed: {e}")
            self.progress_manager.complete(phase_id, status='failed', metadata={
                'error': str(e)
            })
            raise
    
    def _create_experiment(self, name: str, description: Optional[str] = None) -> str:
        """Create experiment record."""
        config_dict = {
            'pipeline_type': 'unified_resampling',
            'target_resolution': self.config.get('resampling.target_resolution', 'unknown'),
            'datasets': self.config.get('datasets.target_datasets', []),
            'resampling_strategies': self.config.get('resampling.strategies', {}),
            'som_config': self.config.get('som_analysis', {}),
            'created_by': 'UnifiedResamplingPipeline',
            'checkpoint_enabled': True
        }
        
        experiment_id = schema.create_experiment(
            name=name,
            description=description or f"Unified resampling pipeline run at {datetime.now()}",
            config=config_dict
        )
        
        logger.info(f"Created experiment: {experiment_id}")
        return experiment_id
    
    def _run_resampling_phase(self, skip_existing: bool) -> List[ResampledDatasetInfo]:
        """Run dataset resampling phase with progress tracking."""
        logger.info("Starting resampling phase...")
        
        # Get dataset configurations
        dataset_configs = self.config.get('datasets.target_datasets', [])
        enabled_datasets = [ds for ds in dataset_configs if ds.get('enabled', True)]
        
        # Create step progress nodes
        resampling_step_id = f"{self.current_phase}_datasets"
        self.progress_manager.create_step(
            resampling_step_id,
            f"{self.pipeline_id}_{self.current_phase}",
            total_substeps=len(enabled_datasets),
            metadata={'total_datasets': len(enabled_datasets)}
        )
        self.progress_manager.start(resampling_step_id)
        
        logger.info(f"Processing {len(enabled_datasets)} datasets")
        
        # Check if we have partial results from checkpoint
        resampled_datasets = self._checkpoint_data.get('resampled_datasets', [])
        completed_names = {d.name for d in resampled_datasets}
        
        for i, dataset_config in enumerate(enabled_datasets, 1):
            dataset_name = dataset_config['name']
            
            # Skip if already completed in checkpoint
            if dataset_name in completed_names:
                logger.info(f"({i}/{len(enabled_datasets)}) Already completed: {dataset_name}")
                self.progress_manager.update(resampling_step_id, increment=1)
                continue
            
            logger.info(f"({i}/{len(enabled_datasets)}) Processing: {dataset_name}")
            
            # Check if already resampled
            if skip_existing:
                existing = self.resampling_processor.get_resampled_dataset(dataset_name)
                if existing and existing.target_resolution == self.config.get('resampling.target_resolution'):
                    logger.info(f"✓ Using existing resampled dataset: {dataset_name}")
                    resampled_datasets.append(existing)
                    self.progress_manager.update(resampling_step_id, increment=1)
                    continue
            
            # Validate dataset configuration
            is_valid, error_msg = self.validator.validate_dataset_config(dataset_config)
            if not is_valid:
                logger.error(f"Dataset validation failed for {dataset_name}: {error_msg}")
                self.progress_manager.update(resampling_step_id, increment=1)
                continue
            
            # Resample dataset with progress callback
            try:
                def dataset_progress(msg: str, pct: float):
                    self.progress_manager.update(
                        resampling_step_id,
                        metadata={
                            'current_dataset': dataset_name,
                            'dataset_progress': pct,
                            'status': msg
                        }
                    )
                
                resampled_info = self.resampling_processor.resample_dataset(
                    dataset_config,
                    progress_callback=dataset_progress
                )
                resampled_datasets.append(resampled_info)
                logger.info(f"✅ Successfully resampled: {dataset_name}")
                
                # Update checkpoint data
                self._checkpoint_data['resampled_datasets'] = resampled_datasets
                
                # Save intermediate checkpoint
                if i % 2 == 0:  # Every 2 datasets
                    self._save_checkpoint(f"resampling_progress_{i}")
                
            except Exception as e:
                logger.error(f"Failed to resample {dataset_name}: {e}")
            
            self.progress_manager.update(resampling_step_id, increment=1)
        
        self.resampled_datasets = resampled_datasets
        self.progress_manager.complete(resampling_step_id)
        
        logger.info(f"Resampling phase completed: {len(resampled_datasets)} datasets processed")
        return resampled_datasets
    
    def _run_merging_phase(self, resampled_info: List[ResampledDatasetInfo]) -> xr.Dataset:
        """Run dataset merging phase with progress tracking."""
        logger.info("Starting merging phase...")
        
        if len(resampled_info) < 2:
            raise ValueError("Need at least 2 resampled datasets for merging")
        
        # Create step progress
        merge_step_id = f"{self.current_phase}_merge"
        self.progress_manager.create_step(
            merge_step_id,
            f"{self.pipeline_id}_{self.current_phase}",
            total_substeps=100
        )
        self.progress_manager.start(merge_step_id)
        
        # Prepare raster names for merger
        raster_names = {}
        band_names = []
        
        for info in resampled_info:
            raster_names[info.band_name] = info.name
            band_names.append(info.band_name)
        
        logger.info(f"Merging {len(band_names)} bands: {band_names}")
        
        # Progress callback for merger
        def merge_progress(msg: str, pct: float):
            self.progress_manager.update(
                merge_step_id,
                completed_units=int(pct),
                metadata={'status': msg}
            )
        
        # Load and merge resampled datasets with progress
        merged_data = self._merge_resampled_datasets_with_progress(
            resampled_info, 
            merge_progress
        )
        self.merged_dataset = merged_data
        
        self.progress_manager.complete(merge_step_id)
        
        logger.info(f"✅ Merged dataset shape: {dict(merged_data.sizes)}")
        logger.info(f"   Bands: {list(merged_data.data_vars)}")
        
        return merged_data
    
    def _run_analysis_phase(self, merged_data: xr.Dataset) -> Dict[str, Any]:
        """Run SOM analysis phase with progress tracking."""
        logger.info("Starting SOM analysis phase...")
        
        # Create step progress
        analysis_step_id = f"{self.current_phase}_som"
        self.progress_manager.create_step(
            analysis_step_id,
            f"{self.pipeline_id}_{self.current_phase}",
            total_substeps=100
        )
        self.progress_manager.start(analysis_step_id)
        
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
        
        # Progress callback for SOM
        def som_progress(iteration: int, total: int):
            pct = (iteration / total) * 100
            self.progress_manager.update(
                analysis_step_id,
                completed_units=int(pct),
                metadata={
                    'iteration': iteration,
                    'total_iterations': total
                }
            )
        
        # Run SOM analysis with progress
        som_results = self.som_analyzer.analyze(
            data=merged_data,
            progress_callback=som_progress,
            **som_params
        )
        
        self.som_results = som_results
        self.progress_manager.complete(analysis_step_id)
        
        logger.info("✅ SOM analysis completed")
        
        return som_results
    
    def _finalize_results(self, som_results: Any) -> Dict[str, Any]:
        """Finalize and save results with progress tracking."""
        logger.info("Finalizing results...")
        
        # Create step progress
        export_step_id = f"{self.current_phase}_export"
        self.progress_manager.create_step(
            export_step_id,
            f"{self.pipeline_id}_{self.current_phase}",
            total_substeps=100
        )
        self.progress_manager.start(export_step_id)
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f"UnifiedResampling_SOM_{timestamp}"
        
        self.progress_manager.update(export_step_id, completed_units=20, 
                                   metadata={'status': 'Saving SOM results'})
        
        # Save SOM results
        saved_path = self.som_analyzer.save_results(som_results, output_name)
        
        self.progress_manager.update(export_step_id, completed_units=80,
                                   metadata={'status': 'Compiling final results'})
        
        # Compile final results
        final_results = {
            'experiment_id': self.experiment_id,
            'pipeline_id': self.pipeline_id,
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
                'target_resolution': self.config.get('resampling.target_resolution', 'unknown'),
                'total_datasets_processed': len(self.resampled_datasets),
                'checkpoints_created': len(self._checkpoint_data.get('checkpoint_ids', []))
            }
        }
        
        self.progress_manager.complete(export_step_id)
        
        logger.info(f"✅ Results saved to: {saved_path}")
        return final_results
    
    def _save_checkpoint(self, checkpoint_id: str) -> None:
        """Save pipeline checkpoint."""
        logger.info(f"Saving checkpoint: {checkpoint_id}")
        
        checkpoint_data = {
            'experiment_id': self.experiment_id,
            'pipeline_id': self.pipeline_id,
            'current_phase': self.current_phase,
            'completed_phases': self._checkpoint_data['completed_phases'],
            'phase_results': self._checkpoint_data['phase_results'],
            'resampled_datasets': [
                {
                    'name': d.name,
                    'source_path': str(d.source_path),
                    'target_resolution': d.target_resolution,
                    'target_crs': d.target_crs,
                    'bounds': d.bounds,
                    'shape': d.shape,
                    'data_type': d.data_type,
                    'resampling_method': d.resampling_method,
                    'band_name': d.band_name,
                    'metadata': d.metadata
                }
                for d in self.resampled_datasets
            ]
        }
        
        # Save checkpoint
        self.checkpoint_manager.save_checkpoint(
            checkpoint_id,
            checkpoint_data,
            level='pipeline',
            parent_id=self.experiment_id,
            metadata={'phase': self.current_phase}
        )
        
        # Track checkpoint IDs
        if 'checkpoint_ids' not in self._checkpoint_data:
            self._checkpoint_data['checkpoint_ids'] = []
        self._checkpoint_data['checkpoint_ids'].append(checkpoint_id)
    
    def _load_checkpoint(self, checkpoint_id: str) -> None:
        """Load pipeline checkpoint."""
        logger.info(f"Loading checkpoint: {checkpoint_id}")
        
        checkpoint_data = self.checkpoint_manager.load_checkpoint(checkpoint_id)
        
        # Restore state
        self.experiment_id = checkpoint_data.get('experiment_id')
        self.pipeline_id = checkpoint_data.get('pipeline_id')
        self._checkpoint_data['completed_phases'] = checkpoint_data.get('completed_phases', [])
        self._checkpoint_data['phase_results'] = checkpoint_data.get('phase_results', {})
        
        # Restore resampled datasets
        resampled_data = checkpoint_data.get('resampled_datasets', [])
        self.resampled_datasets = [
            ResampledDatasetInfo(**d) for d in resampled_data
        ]
        
        # Restore other state from phase results
        if 'merging' in self._checkpoint_data['phase_results']:
            self.merged_dataset = self._checkpoint_data['phase_results']['merging']
        
        if 'analysis' in self._checkpoint_data['phase_results']:
            self.som_results = self._checkpoint_data['phase_results']['analysis']
    
    def _merge_resampled_datasets_with_progress(self, 
                                              resampled_info: List[ResampledDatasetInfo],
                                              progress_callback: Callable) -> xr.Dataset:
        """Merge resampled datasets with progress reporting."""
        logger.info(f"Loading and merging {len(resampled_info)} resampled datasets...")
        
        progress_callback("Initializing merge", 0)
        
        # Initialize resampling processor to load data
        from src.database.connection import db
        processor = ResamplingProcessor(self.config, db)
        data_vars = {}
        coords = None
        
        total_bands = len(resampled_info)
        
        for i, info in enumerate(resampled_info):
            band_progress = (i / total_bands) * 90  # 90% for loading
            progress_callback(f"Loading band: {info.name}", band_progress)
            
            logger.info(f"Loading resampled data for: {info.name}")
            
            # Load the actual array data from database
            array_data = processor.load_resampled_data(info.name)
            if array_data is None:
                raise RuntimeError(f"Failed to load resampled data for {info.name}")
            
            logger.info(f"Loaded array shape: {array_data.shape}")
            
            # Create coordinates if not already created
            if coords is None:
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
            
            # Create DataArray for this band
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
        
        progress_callback("Creating merged dataset", 95)
        
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
        
        progress_callback("Merge complete", 100)
        
        logger.info(f"✅ Successfully merged {len(data_vars)} bands")
        return merged_dataset
    
    def _update_experiment_status(self, status: str, results: Optional[Dict[str, Any]] = None, 
                                 error_message: Optional[str] = None):
        """Update experiment status in database."""
        if self.experiment_id:
            # Store last checkpoint ID in results
            if results is None:
                results = {}
            
            if 'checkpoint_ids' in self._checkpoint_data:
                results['checkpoints'] = self._checkpoint_data['checkpoint_ids']
                results['last_checkpoint'] = self._checkpoint_data['checkpoint_ids'][-1]
            
            schema.update_experiment_status(
                self.experiment_id, 
                status, 
                results, 
                error_message
            )
    
    def _register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        def handle_shutdown():
            logger.info("Received shutdown signal, saving checkpoint...")
            self._save_checkpoint(f"shutdown_{self.experiment_id}_{int(time.time())}")
        
        self.signal_handler.register_shutdown_callback(handle_shutdown)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status with progress information."""
        # Get progress summary
        progress_summary = self.progress_manager.get_progress(self.pipeline_id) if self.pipeline_id else {}
        
        return {
            'experiment_id': self.experiment_id,
            'pipeline_id': self.pipeline_id,
            'current_phase': self.current_phase,
            'completed_phases': self._checkpoint_data.get('completed_phases', []),
            'resampled_datasets_count': len(self.resampled_datasets),
            'merged_dataset_available': self.merged_dataset is not None,
            'som_results_available': self.som_results is not None,
            'progress': progress_summary,
            'checkpoints': self._checkpoint_data.get('checkpoint_ids', [])
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
        
        # Clean up old checkpoints
        if self.experiment_id:
            self.checkpoint_manager.cleanup_old_checkpoints(
                days_old=7,
                keep_minimum={'pipeline': 2, 'phase': 1}
            )
        
        logger.info("Cleanup completed")