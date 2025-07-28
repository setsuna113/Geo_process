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


class DatabaseSchema:
    """
    Facade for decomposed database schema operations.
    
    This class provides the same interface as the original DatabaseSchema god class
    but delegates to focused, single-responsibility modules.
    """
    
    def __init__(self):
        # Initialize operation modules
        self._grid_ops = GridOperations()
        self._species_ops = SpeciesOperations()
        self._feature_ops = FeatureOperations()
        
        # For now, import remaining methods from original implementation
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
    
    # TODO: Remaining methods from original schema.py
    # These will be gradually moved to appropriate operation modules:
    # - ExperimentTracking: create_experiment, update_experiment_status, etc.
    # - RasterCache: store_raster_source, get_cached_resampling_values, etc.
    # - SchemaManagement: create_schema, drop_schema, etc.
    # - CheckpointOperations: create_checkpoint, get_checkpoint, etc.
    
    def __getattr__(self, name):
        """
        Fallback for methods not yet decomposed.
        
        This allows gradual migration by falling back to the original implementation
        for methods that haven't been moved to operation modules yet.
        """
        # Import the original schema class for backward compatibility
        from .. import schema
        original_schema = schema.DatabaseSchema()
        
        if hasattr(original_schema, name):
            return getattr(original_schema, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")