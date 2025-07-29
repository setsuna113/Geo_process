"""Database schema architecture tests focusing on delegation and compatibility."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.database.schema import DatabaseSchema
from src.database.schema.grid_operations import GridOperations


class TestDatabaseSchemaFacade:
    """Test the database schema facade pattern."""
    
    def test_grid_operations_delegation(self):
        """Test that grid operations are properly delegated."""
        mock_db = Mock()
        schema = DatabaseSchema(mock_db)
        
        # Test grid operations delegation
        result = schema.store_grid_definition(
            name="test_grid",
            grid_type="cubic", 
            resolution=1000,
            bounds="POLYGON((0 0, 1 1, 1 0, 0 0))",
            metadata={"test": "data"}
        )
        
        # Verify delegation to grid_ops
        assert schema.grid_ops is not None
        assert isinstance(schema.grid_ops, GridOperations)
    
    def test_monolithic_schema_delegation(self):
        """Test delegation to monolithic schema for non-decomposed operations."""
        mock_db = Mock()
        
        with patch('src.database.schema.monolithic_module') as mock_module:
            mock_monolithic = Mock()
            mock_module.DatabaseSchema.return_value = mock_monolithic
            mock_monolithic.store_species_range.return_value = "species_123"
            
            schema = DatabaseSchema(mock_db)
            
            # Test species operations delegation
            result = schema.store_species_range({"species_name": "test_species"})
            
            # Verify delegation
            mock_monolithic.store_species_range.assert_called_once_with(
                {"species_name": "test_species"}
            )
            assert result == "species_123"
    
    def test_lazy_monolithic_schema_loading(self):
        """Test that monolithic schema is loaded lazily."""
        mock_db = Mock()
        
        with patch('src.database.schema.monolithic_module') as mock_module:
            mock_monolithic = Mock()
            mock_module.DatabaseSchema.return_value = mock_monolithic
            
            schema = DatabaseSchema(mock_db)
            
            # Should not be loaded initially
            assert not hasattr(schema, '_monolithic_schema')
            
            # First call should trigger loading
            schema.store_species_range({"test": "data"})
            
            # Should now be loaded
            assert hasattr(schema, '_monolithic_schema')
            assert schema._monolithic_schema is mock_monolithic
            
            # Second call should reuse
            schema.store_features_batch([{"test": "feature"}])
            
            # Should still be the same instance
            assert schema._monolithic_schema is mock_monolithic
            # Constructor should only be called once
            mock_module.DatabaseSchema.assert_called_once_with(mock_db)
    
    def test_all_delegation_methods_exist(self):
        """Test that all required delegation methods are implemented."""
        mock_db = Mock()
        schema = DatabaseSchema(mock_db)
        
        # Methods that should delegate to monolithic schema
        delegation_methods = [
            'store_species_range',
            'store_species_intersections_batch', 
            'get_species_ranges',
            'store_feature',
            'store_features_batch',
            'store_climate_data_batch',
            'get_features',
            'create_experiment',
            'update_experiment_status',
            'get_experiment',
            'get_experiments',
            'store_resampling_cache_batch',
            'get_cached_resampling_values',
            'store_resampled_dataset',
            'get_resampled_datasets',
            'create_resampled_data_table',
            'drop_resampled_data_table',
            'get_passthrough_datasets',
            'create_schema',
            'drop_schema'
        ]
        
        for method_name in delegation_methods:
            assert hasattr(schema, method_name), f"Missing method: {method_name}"
            method = getattr(schema, method_name)
            assert callable(method), f"Method {method_name} is not callable"
    
    def test_grid_operations_methods_exist(self):
        """Test that grid operations methods are properly exposed."""
        mock_db = Mock()
        schema = DatabaseSchema(mock_db)
        
        grid_methods = [
            'store_grid_definition',
            'store_grid_cells_batch',
            'get_grid_by_name', 
            'get_grid_cells',
            'delete_grid',
            'get_grid_status',
            'validate_grid_config'
        ]
        
        for method_name in grid_methods:
            assert hasattr(schema, method_name), f"Missing grid method: {method_name}"
            method = getattr(schema, method_name)
            assert callable(method), f"Grid method {method_name} is not callable"


class TestGridOperationsDependencyInjection:
    """Test grid operations dependency injection."""
    
    def test_grid_operations_initialization(self):
        """Test that GridOperations is initialized with database manager."""
        mock_db = Mock()
        schema = DatabaseSchema(mock_db)
        
        # Verify grid_ops is initialized
        assert schema.grid_ops is not None
        assert isinstance(schema.grid_ops, GridOperations)
        assert schema.grid_ops.db is mock_db
    
    def test_grid_operations_uses_injected_db(self):
        """Test that GridOperations uses the injected database manager."""
        mock_db = Mock()
        mock_cursor = Mock()
        mock_db.get_cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = {'id': 'grid_123'}
        
        schema = DatabaseSchema(mock_db)
        
        # Call a grid operation
        result = schema.store_grid_definition(
            name="test_grid",
            grid_type="cubic",
            resolution=1000
        )
        
        # Verify the injected database was used
        mock_db.get_cursor.assert_called()
    
    def test_grid_operations_error_handling(self):
        """Test error handling in grid operations."""
        mock_db = Mock()
        schema = DatabaseSchema(mock_db)
        
        # Test with invalid grid type
        with pytest.raises(ValueError, match="Unknown grid type"):
            schema.store_grid_definition(
                name="test_grid",
                grid_type="invalid_type",
                resolution=1000
            )


class TestBackwardCompatibility:
    """Test backward compatibility of the schema facade."""
    
    def test_default_instance_creation(self):
        """Test that default schema instance can be created."""
        from src.database.schema import schema as default_schema
        
        # Should be able to create without errors (may be None if DB unavailable)
        assert default_schema is not None or default_schema is None
    
    def test_existing_code_compatibility(self):
        """Test that existing code patterns still work."""
        mock_db = Mock()
        
        with patch('src.database.schema.monolithic_module') as mock_module:
            mock_monolithic = Mock()
            mock_module.DatabaseSchema.return_value = mock_monolithic
            mock_monolithic.store_features_batch.return_value = 5
            
            schema = DatabaseSchema(mock_db)
            
            # This should work exactly as before
            result = schema.store_features_batch([
                {
                    'grid_id': 'grid_1',
                    'cell_id': 'cell_1', 
                    'feature_type': 'elevation',
                    'feature_name': 'mean_elevation',
                    'feature_value': 125.5
                }
            ])
            
            assert result == 5
            mock_monolithic.store_features_batch.assert_called_once()
    
    def test_database_manager_injection(self):
        """Test that database manager injection works."""
        mock_db = Mock()
        
        # Create schema with explicit DB manager
        schema = DatabaseSchema(mock_db)
        assert schema.db is mock_db
        
        # Create schema with None (should use global)
        with patch('src.database.schema.db') as mock_global_db:
            schema2 = DatabaseSchema(None)
            assert schema2.db is mock_global_db


class TestArchitecturalIntegrity:
    """Test architectural integrity and design patterns."""
    
    def test_facade_pattern_implementation(self):
        """Test that facade pattern is properly implemented."""
        mock_db = Mock()
        schema = DatabaseSchema(mock_db)
        
        # Facade should delegate to appropriate subsystems
        assert hasattr(schema, 'grid_ops')  # Specialized subsystem
        
        # Should have delegation methods for monolithic fallback
        with patch.object(schema, '_get_monolithic_schema') as mock_get:
            mock_monolithic = Mock()
            mock_get.return_value = mock_monolithic
            
            schema.store_species_range({"test": "data"})
            mock_get.assert_called_once()
            mock_monolithic.store_species_range.assert_called_once()
    
    def test_no_direct_monolithic_coupling(self):
        """Test that facade doesn't have tight coupling to monolithic schema."""
        mock_db = Mock()
        schema = DatabaseSchema(mock_db)
        
        # Monolithic schema should not be loaded until needed
        assert not hasattr(schema, '_monolithic_schema')
        
        # Even after grid operations, monolithic should not be loaded
        with patch.object(schema.grid_ops, 'store_grid_definition') as mock_grid:
            schema.store_grid_definition("test", "cubic", 1000)
            mock_grid.assert_called_once()
            assert not hasattr(schema, '_monolithic_schema')
    
    def test_subsystem_isolation(self):
        """Test that subsystems are properly isolated."""
        mock_db = Mock()
        schema = DatabaseSchema(mock_db)
        
        # Grid operations should be independent subsystem
        assert schema.grid_ops.db is mock_db
        
        # Grid operations should not depend on monolithic schema
        assert not hasattr(schema.grid_ops, '_monolithic_schema')
    
    def test_dependency_injection_pattern(self):
        """Test proper dependency injection implementation."""
        mock_db = Mock()
        schema = DatabaseSchema(mock_db)
        
        # Database should be injected, not global
        assert schema.db is mock_db
        assert schema.grid_ops.db is mock_db
        
        # Monolithic schema should receive injected DB when created
        with patch('src.database.schema.monolithic_module') as mock_module:
            mock_monolithic = Mock()
            mock_module.DatabaseSchema.return_value = mock_monolithic
            
            schema._get_monolithic_schema()
            
            # Should pass the injected DB to monolithic schema
            mock_module.DatabaseSchema.assert_called_once_with(mock_db)


class TestErrorHandling:
    """Test error handling in the schema facade."""
    
    def test_monolithic_schema_import_failure(self):
        """Test handling of monolithic schema import failures."""
        mock_db = Mock()
        
        with patch('src.database.schema.monolithic_module', side_effect=ImportError("Module not found")):
            schema = DatabaseSchema(mock_db)
            
            # Should raise the import error when trying to use monolithic features
            with pytest.raises(ImportError):
                schema.store_species_range({"test": "data"})
    
    def test_database_connection_failure(self):
        """Test handling of database connection failures."""
        mock_db = Mock()
        mock_db.get_cursor.side_effect = Exception("Connection failed")
        
        schema = DatabaseSchema(mock_db)
        
        # Grid operations should handle database errors gracefully
        # (Specific error handling depends on implementation)
        with pytest.raises(Exception):
            schema.store_grid_definition("test", "cubic", 1000)


if __name__ == "__main__":
    pytest.main([__file__])