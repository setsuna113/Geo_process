# src/database/schema/__init__.py
"""Consolidated database schema interface."""

from typing import Optional
from ..interfaces import DatabaseInterface
from .grid_operations import GridOperations

# TODO: Import other operation modules as they are created
# from .species_operations import SpeciesOperations
# from .feature_operations import FeatureOperations
# from .experiment_tracking import ExperimentTracking
# from .raster_cache import RasterCache

# Export database exceptions for external use
from ..exceptions import (
    DatabaseError, DatabaseNotFoundError, DatabaseDuplicateError,
    DatabaseConnectionError, DatabaseIntegrityError
)


class DatabaseSchema:
    """
    Consolidated database schema interface.
    
    This serves as a facade that delegates operations to specialized modules,
    replacing the previous monolithic schema class.
    """
    
    def __init__(self, db_manager: Optional[DatabaseInterface] = None):
        """Initialize with database manager."""
        # Import only when needed to avoid circular dependency
        if db_manager:
            self.db = db_manager
        else:
            from ..connection import db
            self.db = db
            
        # Initialize operation modules
        self.grid_ops = GridOperations(self.db)
        # TODO: Initialize other operation modules
        # self.species_ops = SpeciesOperations(self.db)
        # self.feature_ops = FeatureOperations(self.db)
        # self.experiment_ops = ExperimentTracking(self.db)
        # self.raster_ops = RasterCache(self.db)
    
    # Delegate grid operations to grid_ops module
    def store_grid_definition(self, *args, **kwargs):
        """Store grid definition - delegated to grid operations."""
        return self.grid_ops.store_grid_definition(*args, **kwargs)
        
    def store_grid_cells_batch(self, *args, **kwargs):
        """Store grid cells batch - delegated to grid operations."""
        return self.grid_ops.store_grid_cells_batch(*args, **kwargs)
        
    def get_grid_by_name(self, *args, **kwargs):
        """Get grid by name - delegated to grid operations."""
        return self.grid_ops.get_grid_by_name(*args, **kwargs)
        
    def get_grid_cells(self, *args, **kwargs):
        """Get grid cells - delegated to grid operations."""
        return self.grid_ops.get_grid_cells(*args, **kwargs)
        
    def delete_grid(self, *args, **kwargs):
        """Delete grid - delegated to grid operations."""
        return self.grid_ops.delete_grid(*args, **kwargs)
        
    def get_grid_status(self, *args, **kwargs):
        """Get grid status - delegated to grid operations."""  
        return self.grid_ops.get_grid_status(*args, **kwargs)
        
    def validate_grid_config(self, *args, **kwargs):
        """Validate grid config - delegated to grid operations."""
        return self.grid_ops.validate_grid_config(*args, **kwargs)

    # Compatibility methods that delegate to monolithic schema
    # These will be replaced as modules are fully implemented
    def _get_monolithic_schema(self):
        """Get the working monolithic schema for delegation."""
        if not hasattr(self, '_monolithic_schema'):
            from .. import schema as monolithic_module
            self._monolithic_schema = monolithic_module.DatabaseSchema(self.db)
        return self._monolithic_schema
        
    def create_schema(self) -> bool:
        """Create database schema - delegates to monolithic implementation."""
        return self._get_monolithic_schema().create_schema()
        
    def drop_schema(self, confirm: bool = False) -> bool:
        """Drop database schema - delegates to monolithic implementation."""
        return self._get_monolithic_schema().drop_schema(confirm)
        
    def store_species_range(self, *args, **kwargs):
        """Store species range - delegates to monolithic implementation."""
        return self._get_monolithic_schema().store_species_range(*args, **kwargs)
        
    def store_species_intersections_batch(self, *args, **kwargs):
        """Store species intersections batch - delegates to monolithic implementation."""
        return self._get_monolithic_schema().store_species_intersections_batch(*args, **kwargs)
        
    def get_species_ranges(self, *args, **kwargs):
        """Get species ranges - delegates to monolithic implementation."""
        return self._get_monolithic_schema().get_species_ranges(*args, **kwargs)
        
    def store_feature(self, *args, **kwargs):
        """Store feature - delegates to monolithic implementation."""
        return self._get_monolithic_schema().store_feature(*args, **kwargs)
        
    def store_features_batch(self, *args, **kwargs):
        """Store features batch - delegates to monolithic implementation."""
        return self._get_monolithic_schema().store_features_batch(*args, **kwargs)
        
    def store_climate_data_batch(self, *args, **kwargs):
        """Store climate data batch - delegates to monolithic implementation."""
        return self._get_monolithic_schema().store_climate_data_batch(*args, **kwargs)
        
    def get_features(self, *args, **kwargs):
        """Get features - delegates to monolithic implementation."""
        return self._get_monolithic_schema().get_features(*args, **kwargs)
        
    def create_experiment(self, *args, **kwargs):
        """Create experiment - delegates to monolithic implementation."""
        return self._get_monolithic_schema().create_experiment(*args, **kwargs)
        
    def update_experiment_status(self, *args, **kwargs):
        """Update experiment status - delegates to monolithic implementation."""
        return self._get_monolithic_schema().update_experiment_status(*args, **kwargs)
        
    def get_experiment(self, *args, **kwargs):
        """Get experiment - delegates to monolithic implementation."""
        return self._get_monolithic_schema().get_experiment(*args, **kwargs)
        
    def store_resampling_cache_batch(self, *args, **kwargs):
        """Store resampling cache batch - delegates to monolithic implementation."""
        return self._get_monolithic_schema().store_resampling_cache_batch(*args, **kwargs)
        
    def get_cached_resampling_values(self, *args, **kwargs):
        """Get cached resampling values - delegates to monolithic implementation."""
        return self._get_monolithic_schema().get_cached_resampling_values(*args, **kwargs)


# Create default instance for backward compatibility
# This will be removed once all imports are updated to inject dependencies
def _create_default_schema():
    """Create default schema instance for backward compatibility."""
    try:
        return DatabaseSchema()
    except:
        # If database is not available, return a mock object
        return None

schema = _create_default_schema()