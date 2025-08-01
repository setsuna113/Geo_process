# src/database/schema/__init__.py
"""Modular database schema interface.

This module provides a fully modular database schema implementation,
breaking down database operations into logical components.
"""

import os
from typing import Optional
from ..interfaces import DatabaseInterface
from .grid_operations import GridOperations
from .species_operations import SpeciesOperations
from .feature_operations import FeatureOperations
from .experiment_tracking import ExperimentTracking
from .raster_cache import RasterCache
from .checkpoint_operations import CheckpointOperations
from .schema_management import SchemaManagement
from .windowed_storage import WindowedStorage
from .processing_status import ProcessingStatus
from .test_operations import TestOperations

# Export database exceptions for external use
from ..exceptions import (
    DatabaseError, DatabaseNotFoundError, DatabaseDuplicateError,
    DatabaseConnectionError, DatabaseIntegrityError
)

__all__ = [
    'DatabaseSchema', 'schema',
    'DatabaseError', 'DatabaseNotFoundError', 'DatabaseDuplicateError',
    'DatabaseConnectionError', 'DatabaseIntegrityError'
]


class DatabaseSchema:
    """Unified database schema interface with modular operations.
    
    This class aggregates all database operations from specialized modules,
    providing a single interface for database interactions.
    """
    
    def __init__(self, db: Optional[DatabaseInterface] = None):
        """Initialize with database connection and operation modules."""
        # Use provided db or get default from connection module
        if db is None:
            from ..connection import get_db
            self.db = get_db()
            if self.db is None:
                raise DatabaseConnectionError("Database connection not available")
        else:
            self.db = db
        
        # Initialize all operation modules
        # Only GridOperations takes a db_manager parameter
        self.grid_ops = GridOperations(self.db)
        
        # These modules use the global db from connection module
        self.species_ops = SpeciesOperations()
        self.feature_ops = FeatureOperations()
        self.experiment_ops = ExperimentTracking()
        self.raster_ops = RasterCache()
        self.checkpoint_ops = CheckpointOperations()
        self.schema_mgmt = SchemaManagement()
        self.windowed_ops = WindowedStorage()
        self.processing_ops = ProcessingStatus()
        self.test_ops = TestOperations()
    
    # Grid Operations
    def store_grid_definition(self, *args, **kwargs):
        return self.grid_ops.store_grid_definition(*args, **kwargs)
    
    def store_grid_cells_batch(self, *args, **kwargs):
        return self.grid_ops.store_grid_cells_batch(*args, **kwargs)
    
    def get_grid_by_name(self, *args, **kwargs):
        return self.grid_ops.get_grid_by_name(*args, **kwargs)
    
    def get_grid_cells(self, *args, **kwargs):
        return self.grid_ops.get_grid_cells(*args, **kwargs)
    
    def delete_grid(self, *args, **kwargs):
        return self.grid_ops.delete_grid(*args, **kwargs)
    
    def validate_grid_config(self, *args, **kwargs):
        return self.grid_ops.validate_grid_config(*args, **kwargs)
    
    def get_grid_status(self, *args, **kwargs):
        return self.grid_ops.get_grid_status(*args, **kwargs)
    
    # Species Operations
    def store_species_range(self, *args, **kwargs):
        return self.species_ops.store_species_range(*args, **kwargs)
    
    def store_species_intersections_batch(self, *args, **kwargs):
        return self.species_ops.store_species_intersections_batch(*args, **kwargs)
    
    def get_species_ranges(self, *args, **kwargs):
        return self.species_ops.get_species_ranges(*args, **kwargs)
    
    def get_species_richness(self, *args, **kwargs):
        return self.species_ops.get_species_richness(*args, **kwargs)
    
    # Feature Operations
    def store_feature(self, *args, **kwargs):
        return self.feature_ops.store_feature(*args, **kwargs)
    
    def store_features_batch(self, *args, **kwargs):
        return self.feature_ops.store_features_batch(*args, **kwargs)
    
    def store_climate_data_batch(self, *args, **kwargs):
        return self.feature_ops.store_climate_data_batch(*args, **kwargs)
    
    def get_features(self, *args, **kwargs):
        return self.feature_ops.get_features(*args, **kwargs)
    
    # Experiment Tracking
    def create_experiment(self, *args, **kwargs):
        return self.experiment_ops.create_experiment(*args, **kwargs)
    
    def update_experiment_status(self, *args, **kwargs):
        return self.experiment_ops.update_experiment_status(*args, **kwargs)
    
    def create_processing_job(self, *args, **kwargs):
        return self.experiment_ops.create_processing_job(*args, **kwargs)
    
    def update_job_progress(self, *args, **kwargs):
        return self.experiment_ops.update_job_progress(*args, **kwargs)
    
    def get_experiments(self, *args, **kwargs):
        return self.experiment_ops.get_experiments(*args, **kwargs)
    
    # Raster Cache Operations
    def store_resampled_dataset(self, *args, **kwargs):
        return self.raster_ops.store_resampled_dataset(*args, **kwargs)
    
    def get_resampled_datasets(self, *args, **kwargs):
        return self.raster_ops.get_resampled_datasets(*args, **kwargs)
    
    def create_resampled_data_table(self, *args, **kwargs):
        return self.raster_ops.create_resampled_data_table(*args, **kwargs)
    
    def drop_resampled_data_table(self, *args, **kwargs):
        return self.raster_ops.drop_resampled_data_table(*args, **kwargs)
    
    def store_raster_source(self, *args, **kwargs):
        return self.raster_ops.store_raster_source(*args, **kwargs)
    
    def get_raster_sources(self, *args, **kwargs):
        return self.raster_ops.get_raster_sources(*args, **kwargs)
    
    def update_raster_processing_status(self, *args, **kwargs):
        return self.raster_ops.update_raster_processing_status(*args, **kwargs)
    
    def store_raster_tiles_batch(self, *args, **kwargs):
        return self.raster_ops.store_raster_tiles_batch(*args, **kwargs)
    
    def get_raster_tiles_for_bounds(self, *args, **kwargs):
        return self.raster_ops.get_raster_tiles_for_bounds(*args, **kwargs)
    
    def store_resampling_cache_batch(self, *args, **kwargs):
        return self.raster_ops.store_resampling_cache_batch(*args, **kwargs)
    
    def get_cached_resampling_values(self, *args, **kwargs):
        return self.raster_ops.get_cached_resampling_values(*args, **kwargs)
    
    def add_processing_task(self, *args, **kwargs):
        return self.raster_ops.add_processing_task(*args, **kwargs)
    
    def get_next_processing_task(self, *args, **kwargs):
        return self.raster_ops.get_next_processing_task(*args, **kwargs)
    
    def complete_processing_task(self, *args, **kwargs):
        return self.raster_ops.complete_processing_task(*args, **kwargs)
    
    def get_passthrough_datasets(self, *args, **kwargs):
        return self.raster_ops.get_passthrough_datasets(*args, **kwargs)
    
    def get_dataset_by_type(self, *args, **kwargs):
        return self.raster_ops.get_dataset_by_type(*args, **kwargs)
    
    def update_dataset_metadata(self, *args, **kwargs):
        return self.raster_ops.update_dataset_metadata(*args, **kwargs)
    
    def get_raster_processing_status(self, *args, **kwargs):
        return self.raster_ops.get_raster_processing_status(*args, **kwargs)
    
    def get_cache_efficiency_summary(self, *args, **kwargs):
        return self.raster_ops.get_cache_efficiency_summary(*args, **kwargs)
    
    def cleanup_old_cache(self, *args, **kwargs):
        return self.raster_ops.cleanup_old_cache(*args, **kwargs)
    
    # Checkpoint Operations
    def create_checkpoint(self, *args, **kwargs):
        return self.checkpoint_ops.create_checkpoint(*args, **kwargs)
    
    def update_checkpoint_status(self, *args, **kwargs):
        return self.checkpoint_ops.update_checkpoint_status(*args, **kwargs)
    
    def get_checkpoint(self, *args, **kwargs):
        return self.checkpoint_ops.get_checkpoint(*args, **kwargs)
    
    def get_latest_checkpoint(self, *args, **kwargs):
        return self.checkpoint_ops.get_latest_checkpoint(*args, **kwargs)
    
    def list_checkpoints(self, *args, **kwargs):
        return self.checkpoint_ops.list_checkpoints(*args, **kwargs)
    
    def cleanup_old_checkpoints(self, *args, **kwargs):
        return self.checkpoint_ops.cleanup_old_checkpoints(*args, **kwargs)
    
    # Schema Management
    def create_schema(self, *args, **kwargs):
        return self.schema_mgmt.create_schema(*args, **kwargs)
    
    def drop_schema(self, *args, **kwargs):
        return self.schema_mgmt.drop_schema(*args, **kwargs)
    
    def get_schema_info(self, *args, **kwargs):
        return self.schema_mgmt.get_schema_info(*args, **kwargs)
    
    def create_all_tables(self, *args, **kwargs):
        return self.schema_mgmt.create_all_tables(*args, **kwargs)
    
    def drop_all_tables(self, *args, **kwargs):
        return self.schema_mgmt.drop_all_tables(*args, **kwargs)
    
    # Windowed Storage Operations
    def insert_raster_chunk(self, *args, **kwargs):
        return self.windowed_ops.insert_raster_chunk(*args, **kwargs)
    
    def create_windowed_storage_table(self, *args, **kwargs):
        return self.windowed_ops.create_windowed_storage_table(*args, **kwargs)
    
    def get_raster_chunk_bounds(self, *args, **kwargs):
        return self.windowed_ops.get_raster_chunk_bounds(*args, **kwargs)
    
    def migrate_legacy_table_to_coordinates(self, *args, **kwargs):
        return self.windowed_ops.migrate_legacy_table_to_coordinates(*args, **kwargs)
    
    def ensure_table_has_coordinates(self, *args, **kwargs):
        return self.windowed_ops.ensure_table_has_coordinates(*args, **kwargs)
    
    # Processing Status Operations
    def create_processing_step(self, *args, **kwargs):
        return self.processing_ops.create_processing_step(*args, **kwargs)
    
    def update_processing_step(self, *args, **kwargs):
        return self.processing_ops.update_processing_step(*args, **kwargs)
    
    def get_processing_steps(self, *args, **kwargs):
        return self.processing_ops.get_processing_steps(*args, **kwargs)
    
    def create_file_processing_status(self, *args, **kwargs):
        return self.processing_ops.create_file_processing_status(*args, **kwargs)
    
    def update_file_processing_status(self, *args, **kwargs):
        return self.processing_ops.update_file_processing_status(*args, **kwargs)
    
    def get_file_processing_status(self, *args, **kwargs):
        return self.processing_ops.get_file_processing_status(*args, **kwargs)
    
    def get_resumable_files(self, *args, **kwargs):
        return self.processing_ops.get_resumable_files(*args, **kwargs)
    
    def get_processing_queue_summary(self, *args, **kwargs):
        return self.processing_ops.get_processing_queue_summary(*args, **kwargs)
    
    # Test Operations
    def cleanup_test_data(self, *args, **kwargs):
        return self.test_ops.cleanup_test_data(*args, **kwargs)
    
    def mark_test_data(self, *args, **kwargs):
        return self.test_ops.mark_test_data(*args, **kwargs)


# LAZY INITIALIZATION - Do NOT create schema on import!
# The schema should only be created when explicitly requested.
_schema_instance = None

def get_schema():
    """Get the database schema instance (lazy initialization).
    
    This ensures the database connection is only created when actually needed,
    not as a side effect of importing the module.
    
    Returns:
        DatabaseSchema: The schema instance, or None if database is not available
    """
    global _schema_instance
    
    if _schema_instance is None:
        try:
            _schema_instance = DatabaseSchema()
        except:
            # If database is not available, return None
            return None
    
    return _schema_instance

# DEPRECATED: Direct access to 'schema' - use get_schema() instead
# This is set to None to prevent connection on import
schema = None