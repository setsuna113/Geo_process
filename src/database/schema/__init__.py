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

    # TODO: Add delegation methods for other operations as modules are created
    # For now, keep placeholder methods to avoid breaking existing code
    def create_schema(self) -> bool:
        """Create database schema - placeholder."""
        # TODO: Implement across all operation modules
        return True
        
    def drop_schema(self, confirm: bool = False) -> bool:
        """Drop database schema - placeholder."""
        # TODO: Implement across all operation modules
        return True
        
    def store_species_range(self, *args, **kwargs):
        """Store species range - placeholder for species operations."""
        # TODO: Delegate to species_ops when created
        pass
        
    def store_features_batch(self, *args, **kwargs):
        """Store features batch - placeholder for feature operations."""
        # TODO: Delegate to feature_ops when created
        pass
        
    def store_climate_data_batch(self, *args, **kwargs):
        """Store climate data batch - placeholder for feature operations."""
        # TODO: Delegate to feature_ops when created
        pass


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