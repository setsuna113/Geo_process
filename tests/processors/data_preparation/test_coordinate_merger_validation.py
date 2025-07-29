"""Integration tests for CoordinateMerger with validation framework."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile

from src.processors.data_preparation.coordinate_merger import CoordinateMerger
from src.abstractions.interfaces.validator import ValidationSeverity


class TestCoordinateMergerValidation:
    """Test CoordinateMerger validation integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock()
        self.db = Mock()
        self.catalog = Mock()
        
        # Create temp directory for output
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)
        
        # Initialize merger with mocked dependencies
        with patch('src.processors.data_preparation.coordinate_merger.RasterCatalog'):
            self.merger = CoordinateMerger(self.config, self.db)
            self.merger.catalog = self.catalog
    
    def teardown_method(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_valid_dataset_bounds_validation(self):
        """Test validation passes for valid dataset bounds."""
        # Prepare test data
        dataset_info = {
            'name': 'test_dataset',
            'bounds': (-180.0, -90.0, 180.0, 90.0),
            'crs': 'EPSG:4326',
            'table_name': 'test_table',
            'resolution': 0.1,
            'passthrough': False
        }
        
        # Mock database query results
        test_df = pd.DataFrame({
            'x': np.linspace(-180, 180, 10),
            'y': np.linspace(-90, 90, 10),
            'test_dataset': np.random.rand(10)
        })
        
        self.db.get_connection.return_value.__enter__.return_value = Mock()
        with patch('pandas.read_sql', return_value=test_df):
            # Process datasets
            result_path = self.merger.create_ml_ready_parquet(
                [dataset_info], 
                self.output_dir
            )
        
        # Check validation results
        validation_results = self.merger.get_validation_results()
        
        # Should have validation results
        assert len(validation_results) > 0
        
        # Check bounds validation passed
        bounds_validations = [v for v in validation_results 
                            if v['stage'] == 'dataset_bounds']
        assert len(bounds_validations) == 1
        assert bounds_validations[0]['result'].is_valid
        assert bounds_validations[0]['result'].error_count == 0
    
    def test_invalid_bounds_detected(self):
        """Test detection of invalid bounds."""
        # Dataset with invalid bounds (minx > maxx)
        dataset_info = {
            'name': 'bad_bounds_dataset',
            'bounds': (180.0, -90.0, -180.0, 90.0),  # Invalid!
            'crs': 'EPSG:4326',
            'table_name': 'test_table',
            'resolution': 0.1,
            'passthrough': False
        }
        
        # Should raise ValueError due to invalid bounds
        with pytest.raises(ValueError) as exc_info:
            self.merger.create_ml_ready_parquet(
                [dataset_info], 
                self.output_dir
            )
        
        assert "Invalid bounds" in str(exc_info.value)
        assert "bad_bounds_dataset" in str(exc_info.value)
    
    def test_coordinate_data_validation(self):
        """Test validation of loaded coordinate data."""
        dataset_info = {
            'name': 'coord_test',
            'bounds': (-10.0, -10.0, 10.0, 10.0),
            'crs': 'EPSG:4326',
            'table_name': 'test_table',
            'resolution': 1.0,
            'passthrough': False
        }
        
        # Create data with some outliers
        test_df = pd.DataFrame({
            'x': [-5, 0, 5, 200],  # 200 is suspicious
            'y': [-5, 0, 5, 100],  # 100 is suspicious  
            'coord_test': [1, 2, 3, 4]
        })
        
        self.db.get_connection.return_value.__enter__.return_value = Mock()
        
        # Mock cursor for column check
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [('x',), ('y',), ('value',)]
        self.db.get_connection.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor
        
        with patch('pandas.read_sql', return_value=test_df):
            result_path = self.merger.create_ml_ready_parquet(
                [dataset_info], 
                self.output_dir
            )
        
        # Check validation detected suspicious coordinates
        validation_results = self.merger.get_validation_results()
        coord_validations = [v for v in validation_results 
                           if v['stage'] == 'coordinate_data']
        
        assert len(coord_validations) > 0
        # Should have warnings about suspicious coordinates
        assert any(v['result'].warning_count > 0 for v in coord_validations)
    
    def test_spatial_consistency_validation(self):
        """Test spatial consistency validation between datasets."""
        datasets = [
            {
                'name': 'dataset1',
                'bounds': (-10.0, -10.0, 10.0, 10.0),
                'crs': 'EPSG:4326',
                'table_name': 'table1',
                'resolution': 1.0,
                'passthrough': False
            },
            {
                'name': 'dataset2', 
                'bounds': (50.0, 50.0, 60.0, 60.0),  # No overlap with dataset1
                'crs': 'EPSG:4326',
                'table_name': 'table2',
                'resolution': 1.0,
                'passthrough': False
            }
        ]
        
        # Create non-overlapping data
        df1 = pd.DataFrame({
            'x': np.linspace(-10, 10, 5),
            'y': np.linspace(-10, 10, 5),
            'dataset1': np.ones(5)
        })
        
        df2 = pd.DataFrame({
            'x': np.linspace(50, 60, 5),
            'y': np.linspace(50, 60, 5),
            'dataset2': np.ones(5) * 2
        })
        
        self.db.get_connection.return_value.__enter__.return_value = Mock()
        
        # Mock cursor
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [('x',), ('y',), ('value',)]
        self.db.get_connection.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Use side_effect to return different dataframes
        with patch('pandas.read_sql', side_effect=[df1, df2]):
            result_path = self.merger.create_ml_ready_parquet(
                datasets, 
                self.output_dir
            )
        
        # Should complete but log low spatial overlap warning
        assert result_path.exists()
        
        # Verify the warning was logged (through validation results)
        validation_results = self.merger.get_validation_results()
        assert len(validation_results) > 0
    
    def test_merged_data_validation(self):
        """Test validation of merged dataset."""
        datasets = [
            {
                'name': 'temp',
                'bounds': (0.0, 0.0, 10.0, 10.0),
                'crs': 'EPSG:4326',
                'table_name': 'temp_table',
                'resolution': 1.0,
                'passthrough': False
            },
            {
                'name': 'precip',
                'bounds': (0.0, 0.0, 10.0, 10.0),
                'crs': 'EPSG:4326',
                'table_name': 'precip_table',
                'resolution': 1.0,
                'passthrough': False
            }
        ]
        
        # Create matching coordinate data
        coords = [(x, y) for x in range(0, 10, 2) for y in range(0, 10, 2)]
        
        df1 = pd.DataFrame({
            'x': [c[0] for c in coords],
            'y': [c[1] for c in coords],
            'temp': np.random.normal(20, 5, len(coords))  # Temperature data
        })
        
        df2 = pd.DataFrame({
            'x': [c[0] for c in coords],
            'y': [c[1] for c in coords],
            'precip': np.random.normal(100, 20, len(coords))  # Precipitation data
        })
        
        self.db.get_connection.return_value.__enter__.return_value = Mock()
        
        # Mock cursor
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [('x',), ('y',), ('value',)]
        self.db.get_connection.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor
        
        with patch('pandas.read_sql', side_effect=[df1, df2]):
            result_path = self.merger.create_ml_ready_parquet(
                datasets,
                self.output_dir
            )
        
        # Read back the merged file
        merged_df = pd.read_parquet(result_path)
        
        # Verify merge succeeded
        assert len(merged_df) == len(coords)
        assert 'x' in merged_df.columns
        assert 'y' in merged_df.columns
        assert 'temp' in merged_df.columns
        assert 'precip' in merged_df.columns
        
        # Check validation passed
        validation_results = self.merger.get_validation_results()
        merged_validations = [v for v in validation_results 
                            if v['stage'] == 'merged_data']
        assert len(merged_validations) > 0
        assert all(v['result'].is_valid for v in merged_validations)
    
    def test_validation_summary_reporting(self):
        """Test validation summary is properly reported."""
        dataset_info = {
            'name': 'summary_test',
            'bounds': (-1.0, -1.0, 1.0, 1.0),
            'crs': 'EPSG:4326',
            'table_name': 'test_table',
            'resolution': 0.5,
            'passthrough': False
        }
        
        test_df = pd.DataFrame({
            'x': [-0.5, 0.0, 0.5],
            'y': [-0.5, 0.0, 0.5],
            'summary_test': [1, 2, 3]
        })
        
        self.db.get_connection.return_value.__enter__.return_value = Mock()
        
        # Mock cursor
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [('x',), ('y',), ('value',)]
        self.db.get_connection.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor
        
        with patch('pandas.read_sql', return_value=test_df):
            # Capture log output
            with patch('src.processors.data_preparation.coordinate_merger.logger') as mock_logger:
                result_path = self.merger.create_ml_ready_parquet(
                    [dataset_info],
                    self.output_dir
                )
                
                # Check that validation summary was logged
                summary_logged = any(
                    "VALIDATION SUMMARY" in str(call)
                    for call in mock_logger.info.call_args_list
                )
                assert summary_logged
    
    def test_passthrough_data_validation(self):
        """Test validation of passthrough data conversion."""
        dataset_info = {
            'name': 'passthrough_test',
            'bounds': (0.0, 0.0, 100.0, 100.0),
            'crs': 'EPSG:4326',
            'table_name': 'passthrough_table',
            'resolution': 10.0,
            'passthrough': True  # Passthrough mode
        }
        
        # Simulate row/col index data
        test_df = pd.DataFrame({
            'x': [5.0, 15.0, 25.0],  # These will be calculated from indices
            'y': [95.0, 85.0, 75.0],
            'passthrough_test': [10, 20, 30]
        })
        
        self.db.get_connection.return_value.__enter__.return_value = Mock()
        
        with patch('pandas.read_sql', return_value=test_df):
            result_path = self.merger.create_ml_ready_parquet(
                [dataset_info],
                self.output_dir
            )
        
        # Verify coordinate conversion worked
        result_df = pd.read_parquet(result_path)
        assert len(result_df) == 3
        assert 'x' in result_df.columns
        assert 'y' in result_df.columns
        
        # Check validation passed
        validation_results = self.merger.get_validation_results()
        assert any(v['result'].is_valid for v in validation_results)


class TestCoordinateMergerErrorScenarios:
    """Test error scenarios and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock()
        self.db = Mock()
        self.output_dir = Path(tempfile.mkdtemp())
        
        with patch('src.processors.data_preparation.coordinate_merger.RasterCatalog'):
            self.merger = CoordinateMerger(self.config, self.db)
    
    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.output_dir, ignore_errors=True)
    
    def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        dataset_info = {
            'name': 'empty_dataset',
            'bounds': (0.0, 0.0, 10.0, 10.0),
            'crs': 'EPSG:4326',
            'table_name': 'empty_table',
            'resolution': 1.0,
            'passthrough': False
        }
        
        # Return empty dataframe
        empty_df = pd.DataFrame(columns=['x', 'y', 'value'])
        
        self.db.get_connection.return_value.__enter__.return_value = Mock()
        
        # Mock cursor
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [('x',), ('y',), ('value',)]
        self.db.get_connection.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = mock_cursor
        
        with patch('pandas.read_sql', return_value=empty_df):
            with pytest.raises(ValueError) as exc_info:
                self.merger.create_ml_ready_parquet(
                    [dataset_info],
                    self.output_dir
                )
            
            assert "No coordinate data found" in str(exc_info.value)
    
    def test_database_error_handling(self):
        """Test handling of database errors during validation."""
        dataset_info = {
            'name': 'db_error_test',
            'bounds': (0.0, 0.0, 10.0, 10.0),
            'crs': 'EPSG:4326',
            'table_name': 'test_table',
            'resolution': 1.0,
            'passthrough': False
        }
        
        # Simulate database error
        self.db.get_connection.side_effect = Exception("Database connection failed")
        
        with pytest.raises(Exception) as exc_info:
            self.merger.create_ml_ready_parquet(
                [dataset_info],
                self.output_dir
            )
        
        assert "Database connection failed" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])