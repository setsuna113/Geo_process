"""
Database schema operations - decomposed god class facade.

This module provides a consolidated interface to the decomposed schema operations,
maintaining backward compatibility while improving maintainability.

The original DatabaseSchema god class (1,346 lines) has been decomposed into:
- GridOperations: Grid and cell management
- SpeciesOperations: Species range and intersection management  
- FeatureOperations: Feature and climate data management
- [Future] ExperimentTracking: Experiment and job management
- [Future] RasterCache: Raster processing and caching operations
"""

from .grid_operations import GridOperations
from .species_operations import SpeciesOperations
from .feature_operations import FeatureOperations
from .experiment_tracking import ExperimentTracking
from .raster_cache import RasterCache
from .schema_management import SchemaManagement
from .checkpoint_operations import CheckpointOperations


class DatabaseSchema:
    """
    Facade for decomposed database schema operations.
    
    This class provides the same interface as the original DatabaseSchema god class
    but delegates to focused, single-responsibility modules.
    
    The original god class (1,346 lines) has been fully decomposed into:
    - GridOperations: Grid and cell management (140 lines)
    - SpeciesOperations: Species range and intersection management (102 lines)  
    - FeatureOperations: Feature and climate data management (77 lines)
    - ExperimentTracking: Experiment and job management (108 lines)
    - RasterCache: Raster processing and caching operations (248 lines)
    - SchemaManagement: Schema creation and information (77 lines)
    - CheckpointOperations: Checkpoint database operations (115 lines)
    """
    
    def __init__(self):
        # Initialize operation modules
        self._grid_ops = GridOperations()
        self._species_ops = SpeciesOperations()
        self._feature_ops = FeatureOperations()
        self._experiment_ops = ExperimentTracking()
        self._raster_ops = RasterCache()
        self._schema_ops = SchemaManagement()
        self._checkpoint_ops = CheckpointOperations()
        
        # For backward compatibility - schema file access
        from pathlib import Path
        self.schema_file = Path(__file__).parent.parent / "schema.sql"
    
    # Grid Operations - delegated to GridOperations
    def store_grid_definition(self, name: str, grid_type: str, resolution: int,
                             bounds=None, metadata=None) -> str:
        """Store grid definition metadata."""
        return self._grid_ops.store_grid_definition(name, grid_type, resolution, bounds, metadata)
    
    def store_grid_cells_batch(self, grid_id: str, cells_data) -> int:
        """Bulk insert grid cells."""
        return self._grid_ops.store_grid_cells_batch(grid_id, cells_data)
    
    def get_grid_by_name(self, name: str):
        """Get grid by name."""
        return self._grid_ops.get_grid_by_name(name)
    
    def get_grid_cells(self, grid_id: str, limit=None):
        """Get grid cells for a grid."""
        return self._grid_ops.get_grid_cells(grid_id, limit)
    
    def delete_grid(self, name: str) -> bool:
        """Delete grid and all related data."""
        return self._grid_ops.delete_grid(name)
    
    def validate_grid_config(self, grid_type: str, resolution: int) -> bool:
        """Validate grid configuration parameters."""
        return self._grid_ops.validate_grid_config(grid_type, resolution)
    
    def get_grid_status(self, grid_name=None):
        """Get grid processing status."""
        return self._grid_ops.get_grid_status(grid_name)
    
    # Species Operations - delegated to SpeciesOperations
    def store_species_range(self, species_data) -> str:
        """Store species range data."""
        return self._species_ops.store_species_range(species_data)
    
    def store_species_intersections_batch(self, intersections) -> int:
        """Bulk store species-grid intersections."""
        return self._species_ops.store_species_intersections_batch(intersections)
    
    def get_species_ranges(self, category=None, source_file=None):
        """Get species ranges with optional filtering."""
        return self._species_ops.get_species_ranges(category, source_file)
    
    def get_species_richness(self, grid_id: str, category=None):
        """Get species richness summary."""
        return self._species_ops.get_species_richness(grid_id, category)
    
    # Feature Operations - delegated to FeatureOperations  
    def store_feature(self, grid_id: str, cell_id: str, feature_type: str,
                     feature_name: str, feature_value: float, metadata=None) -> str:
        """Store computed feature."""
        return self._feature_ops.store_feature(grid_id, cell_id, feature_type, 
                                               feature_name, feature_value, metadata)
    
    def store_features_batch(self, features) -> int:
        """Bulk store features."""
        return self._feature_ops.store_features_batch(features)
    
    def store_climate_data_batch(self, climate_data) -> int:
        """Bulk store climate data."""
        return self._feature_ops.store_climate_data_batch(climate_data)
    
    def get_features(self, grid_id: str, feature_type=None):
        """Get features for a grid."""
        return self._feature_ops.get_features(grid_id, feature_type)
    
    # Experiment Tracking - delegated to ExperimentTracking
    def create_experiment(self, name: str, description: str, config: dict) -> str:
        """Create new experiment."""
        return self._experiment_ops.create_experiment(name, description, config)
    
    def update_experiment_status(self, experiment_id: str, status: str, 
                                results=None, error_message=None):
        """Update experiment status."""
        return self._experiment_ops.update_experiment_status(experiment_id, status, results, error_message)
    
    def create_processing_job(self, job_type: str, job_name: str, parameters: dict,
                             parent_experiment_id=None) -> str:
        """Create processing job."""
        return self._experiment_ops.create_processing_job(job_type, job_name, parameters, parent_experiment_id)
    
    def update_job_progress(self, job_id: str, progress_percent: float, 
                           status=None, log_message=None):
        """Update job progress."""
        return self._experiment_ops.update_job_progress(job_id, progress_percent, status, log_message)
    
    def get_experiment(self, experiment_id: str):
        """Get experiment by ID."""
        return self._experiment_ops.get_experiment(experiment_id)
    
    def get_experiments(self, status=None):
        """Get experiments."""
        return self._experiment_ops.get_experiments(status)
    
    def get_processing_jobs(self, experiment_id=None):
        """Get processing jobs."""
        return self._experiment_ops.get_processing_jobs(experiment_id)
    
    # Raster Cache - delegated to RasterCache
    def store_resampled_dataset(self, name: str, source_path: str, target_resolution: float,
                               target_crs: str, bounds: list, shape: tuple,
                               data_type: str, resampling_method: str, band_name: str,
                               data_table_name: str, metadata: dict) -> int:
        """Store resampled dataset metadata."""
        return self._raster_ops.store_resampled_dataset(name, source_path, target_resolution,
                                                       target_crs, bounds, shape, data_type,
                                                       resampling_method, band_name,
                                                       data_table_name, metadata)
    
    def get_resampled_datasets(self, filters=None):
        """Get resampled datasets."""
        return self._raster_ops.get_resampled_datasets(filters)
    
    def create_resampled_data_table(self, table_name: str):
        """Create resampled data table."""
        return self._raster_ops.create_resampled_data_table(table_name)
    
    def drop_resampled_data_table(self, table_name: str):
        """Drop resampled data table."""
        return self._raster_ops.drop_resampled_data_table(table_name)
    
    def store_raster_source(self, raster_data: dict) -> str:
        """Store raster source metadata."""
        return self._raster_ops.store_raster_source(raster_data)
    
    def get_raster_sources(self, active_only=True, processing_status=None):
        """Get raster sources."""
        return self._raster_ops.get_raster_sources(active_only, processing_status)
    
    def update_raster_processing_status(self, raster_id: str, status: str, metadata=None):
        """Update raster processing status."""
        return self._raster_ops.update_raster_processing_status(raster_id, status, metadata)
    
    def store_raster_tiles_batch(self, raster_id: str, tiles_data: list) -> int:
        """Store raster tiles batch."""
        return self._raster_ops.store_raster_tiles_batch(raster_id, tiles_data)
    
    def get_raster_tiles_for_bounds(self, raster_id: str, bounds_wkt: str):
        """Get raster tiles for bounds."""
        return self._raster_ops.get_raster_tiles_for_bounds(raster_id, bounds_wkt)
    
    def store_resampling_cache_batch(self, cache_data: list) -> int:
        """Store resampling cache batch."""
        return self._raster_ops.store_resampling_cache_batch(cache_data)
    
    def get_cached_resampling_values(self, raster_id: str, grid_id: str, 
                                   cell_ids: list, method: str, band_number: int):
        """Get cached resampling values."""
        return self._raster_ops.get_cached_resampling_values(raster_id, grid_id, cell_ids, method, band_number)
    
    def add_processing_task(self, queue_type: str, raster_id=None, grid_id=None, 
                           tile_id=None, parameters=None, priority=0) -> str:
        """Add processing task."""
        return self._raster_ops.add_processing_task(queue_type, raster_id, grid_id, tile_id, parameters, priority)
    
    def get_next_processing_task(self, queue_type: str, worker_id: str):
        """Get next processing task."""
        return self._raster_ops.get_next_processing_task(queue_type, worker_id)
    
    def complete_processing_task(self, task_id: str, success=True, error_message=None):
        """Complete processing task."""
        return self._raster_ops.complete_processing_task(task_id, success, error_message)
    
    # Schema Management - delegated to SchemaManagement
    def create_schema(self) -> bool:
        """Create database schema."""
        return self._schema_ops.create_schema()
    
    def drop_schema(self, confirm: bool = False) -> bool:
        """Drop database schema."""
        return self._schema_ops.drop_schema(confirm)
    
    def get_schema_info(self):
        """Get schema information."""
        return self._schema_ops.get_schema_info()
    
    # Checkpoint Operations - delegated to CheckpointOperations
    def create_checkpoint(self, checkpoint_id: str, level: str, parent_id,
                         processor_name: str, data_summary: dict,
                         file_path: str, file_size_bytes: int,
                         compression_type=None) -> str:
        """Create checkpoint."""
        return self._checkpoint_ops.create_checkpoint(checkpoint_id, level, parent_id,
                                                     processor_name, data_summary,
                                                     file_path, file_size_bytes, compression_type)
    
    def update_checkpoint_status(self, checkpoint_id: str, status: str,
                               validation_checksum=None, validation_result=None,
                               error_message=None):
        """Update checkpoint status."""
        return self._checkpoint_ops.update_checkpoint_status(checkpoint_id, status,
                                                           validation_checksum, validation_result,
                                                           error_message)
    
    def get_checkpoint(self, checkpoint_id: str):
        """Get checkpoint."""
        return self._checkpoint_ops.get_checkpoint(checkpoint_id)
    
    def get_latest_checkpoint(self, processor_name: str, level: str):
        """Get latest checkpoint."""
        return self._checkpoint_ops.get_latest_checkpoint(processor_name, level)
    
    def list_checkpoints(self, processor_name=None, level=None, parent_id=None, status=None):
        """List checkpoints."""
        return self._checkpoint_ops.list_checkpoints(processor_name, level, parent_id, status)
    
    def cleanup_old_checkpoints(self, days_old: int, keep_minimum: dict) -> int:
        """Cleanup old checkpoints."""
        return self._checkpoint_ops.cleanup_old_checkpoints(days_old, keep_minimum)
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete checkpoint."""
        return self._checkpoint_ops.delete_checkpoint(checkpoint_id)
    
    def get_checkpoint_hierarchy(self, root_checkpoint_id: str):
        """Get checkpoint hierarchy."""
        return self._checkpoint_ops.get_checkpoint_hierarchy(root_checkpoint_id)