"""Unit tests for CoordinateMerger with mixed storage formats."""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import tempfile

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.processors.data_preparation.coordinate_merger import CoordinateMerger
from src.database.connection import DatabaseManager
from src.config import config


class TestCoordinateMerger(unittest.TestCase):
    """Test CoordinateMerger handling of mixed storage formats."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.db_mock = Mock(spec=DatabaseManager)
        # Config needs to behave like a dict with _data attribute
        self.config_mock = Mock()
        self.config_mock._data = {
            'processors': {
                'CoordinateMerger': {
                    'batch_size': 1000,
                    'enable_progress': True
                }
            }
        }
        self.config_mock.get = Mock(side_effect=lambda k, d=None: self.config_mock._data.get(k, d))
        self.merger = CoordinateMerger(self.config_mock, self.db_mock)
        
    def test_table_has_coordinates_check(self):
        """Test _table_has_coordinates method."""
        # Mock database connection and cursor
        conn_mock = MagicMock()
        cursor_mock = MagicMock()
        conn_mock.cursor.return_value = cursor_mock
        self.db_mock.get_connection.return_value.__enter__.return_value = conn_mock
        
        # Test table with coordinates
        cursor_mock.fetchall.return_value = [('x_coord',), ('y_coord',)]
        has_coords = self.merger._table_has_coordinates('test_table')
        self.assertTrue(has_coords)
        
        # Test table without coordinates
        cursor_mock.fetchall.return_value = [('row_idx',)]
        has_coords = self.merger._table_has_coordinates('test_table')
        self.assertFalse(has_coords)
        
    def test_load_passthrough_with_coordinates(self):
        """Test loading passthrough data when table has coordinate columns."""
        # Mock database connection
        conn_mock = MagicMock()
        self.db_mock.get_connection.return_value.__enter__.return_value = conn_mock
        
        # Mock that table has coordinates
        with patch.object(self.merger, '_table_has_coordinates', return_value=True):
            # Mock data
            test_data = pd.DataFrame({
                'x': [-180, -179, -178],
                'y': [90, 90, 90],
                'test_dataset': [1.0, 2.0, 3.0]
            })
            
            # Mock pd.read_sql to return test data
            with patch('pandas.read_sql', return_value=test_data):
                result = self.merger._load_passthrough_coordinates(
                    'test_dataset',
                    'passthrough_test',
                    (-180, -90, 180, 90),
                    1.0
                )
                
                self.assertEqual(len(result), 3)
                self.assertIn('x', result.columns)
                self.assertIn('y', result.columns)
                self.assertIn('test_dataset', result.columns)
                
    def test_load_passthrough_without_coordinates(self):
        """Test loading passthrough data when table lacks coordinate columns."""
        # Mock database connection
        conn_mock = MagicMock()
        self.db_mock.get_connection.return_value.__enter__.return_value = conn_mock
        
        # Mock that table doesn't have coordinates
        with patch.object(self.merger, '_table_has_coordinates', return_value=False):
            # Mock data with indices that will be converted
            test_data = pd.DataFrame({
                'x': [-179.5, -178.5, -177.5],  # Calculated from indices
                'y': [89.5, 89.5, 89.5],
                'test_dataset': [1.0, 2.0, 3.0]
            })
            
            # Mock pd.read_sql to return test data
            with patch('pandas.read_sql', return_value=test_data):
                result = self.merger._load_passthrough_coordinates(
                    'test_dataset',
                    'passthrough_test', 
                    (-180, -90, 180, 90),
                    1.0
                )
                
                self.assertEqual(len(result), 3)
                # Should have calculated coordinates
                self.assertIn('x', result.columns)
                self.assertIn('y', result.columns)
                
    def test_load_resampled_mixed_formats(self):
        """Test loading resampled data with both storage formats."""
        # Mock database connection
        conn_mock = MagicMock()
        self.db_mock.get_connection.return_value.__enter__.return_value = conn_mock
        
        # Test new format (with coordinates)
        with patch.object(self.merger, '_table_has_coordinates', return_value=True):
            test_data = pd.DataFrame({
                'x': [-180, -179],
                'y': [90, 89],
                'test_data': [10.0, 20.0]
            })
            
            with patch('pandas.read_sql', return_value=test_data):
                result = self.merger._load_resampled_coordinates(
                    'test_data',
                    'resampled_test'
                )
                
                self.assertEqual(len(result), 2)
                self.assertIn('test_data', result.columns)
                
        # Test legacy format (without coordinates)
        with patch.object(self.merger, '_table_has_coordinates', return_value=False):
            # Should fall back to passthrough method
            with patch.object(self.merger, '_load_passthrough_coordinates') as mock_passthrough:
                mock_passthrough.return_value = test_data
                
                result = self.merger._load_resampled_coordinates(
                    'test_data',
                    'resampled_test',
                    bounds=(-180, -90, 180, 90),
                    resolution=1.0
                )
                
                mock_passthrough.assert_called_once()
                
    def test_merge_coordinate_datasets(self):
        """Test merging multiple coordinate datasets."""
        # Create test datasets
        df1 = pd.DataFrame({
            'x': [-180, -179, -178],
            'y': [90, 90, 90],
            'dataset1': [1.0, 2.0, 3.0]
        })
        
        df2 = pd.DataFrame({
            'x': [-180, -179, -177],  # Note: -178 missing, -177 added
            'y': [90, 90, 90],
            'dataset2': [10.0, 20.0, 30.0]
        })
        
        # Merge datasets
        merged = self.merger._merge_coordinate_datasets([df1, df2])
        
        # Check results
        self.assertEqual(len(merged), 4)  # Union of coordinates
        self.assertIn('dataset1', merged.columns)
        self.assertIn('dataset2', merged.columns)
        
        # Check specific values
        row_180 = merged[(merged['x'] == -180) & (merged['y'] == 90)]
        self.assertEqual(len(row_180), 1)
        self.assertEqual(row_180['dataset1'].values[0], 1.0)
        self.assertEqual(row_180['dataset2'].values[0], 10.0)
        
        # Check NaN for missing data
        row_178 = merged[(merged['x'] == -178) & (merged['y'] == 90)]
        self.assertTrue(pd.isna(row_178['dataset2'].values[0]))
        
    def test_chunked_merge_bounds_calculation(self):
        """Test overall bounds calculation for chunked processing."""
        datasets = [
            {'bounds': [-180, -90, 0, 0], 'resolution': 1.0},
            {'bounds': [-90, -45, 90, 45], 'resolution': 1.0},
            {'bounds': [0, 0, 180, 90], 'resolution': 1.0}
        ]
        
        overall_bounds = self.merger._get_overall_bounds(datasets)
        
        # Should encompass all datasets
        self.assertEqual(overall_bounds[0], -180)  # min_x
        self.assertEqual(overall_bounds[1], -90)   # min_y
        self.assertEqual(overall_bounds[2], 180)   # max_x
        self.assertEqual(overall_bounds[3], 90)    # max_y
        
    def test_load_bounded_coordinates(self):
        """Test loading coordinates within specific bounds."""
        # Mock database connection
        conn_mock = MagicMock()
        self.db_mock.get_connection.return_value.__enter__.return_value = conn_mock
        
        dataset_info = {
            'name': 'test_dataset',
            'table_name': 'test_table',
            'bounds': [-180, -90, 180, 90],
            'resolution': 1.0,
            'passthrough': True
        }
        
        query_bounds = (-10, -10, 10, 10)  # Small query region
        
        # Mock table has coordinates
        with patch.object(self.merger, '_table_has_coordinates', return_value=True):
            # Mock bounded data
            bounded_data = pd.DataFrame({
                'x': [-10, -9, -8],
                'y': [10, 10, 10],
                'test_dataset': [1.0, 2.0, 3.0]
            })
            
            with patch('pandas.read_sql', return_value=bounded_data):
                result = self.merger._load_dataset_coordinates_bounded(
                    dataset_info,
                    query_bounds
                )
                
                self.assertEqual(len(result), 3)
                # All coordinates should be within query bounds
                self.assertTrue(all(result['x'] >= -10))
                self.assertTrue(all(result['x'] <= 10))
                self.assertTrue(all(result['y'] >= -10))
                self.assertTrue(all(result['y'] <= 10))
                
    def test_no_overlap_bounded_load(self):
        """Test bounded load when datasets don't overlap."""
        dataset_info = {
            'name': 'test_dataset',
            'table_name': 'test_table',
            'bounds': [-180, -90, -170, -80],  # Far left
            'resolution': 1.0,
            'passthrough': True
        }
        
        query_bounds = (170, 80, 180, 90)  # Far right
        
        result = self.merger._load_dataset_coordinates_bounded(
            dataset_info,
            query_bounds
        )
        
        self.assertIsNone(result)  # No overlap
        
    @patch('pathlib.Path.mkdir')
    def test_create_ml_ready_parquet_inmemory(self, mock_mkdir):
        """Test in-memory parquet creation."""
        # Create test output directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            # Mock dataset loading
            test_df = pd.DataFrame({
                'x': [-180, -179],
                'y': [90, 90],
                'dataset1': [1.0, 2.0]
            })
            
            with patch.object(self.merger, '_load_dataset_coordinates', return_value=test_df):
                with patch.object(self.merger, '_merge_coordinate_datasets', return_value=test_df):
                    with patch.object(self.merger, '_validate_merged_data'):
                        with patch.object(pd.DataFrame, 'to_parquet'):
                            
                            datasets = [{
                                'name': 'dataset1',
                                'table_name': 'test_table',
                                'bounds': [-180, -90, 180, 90],
                                'resolution': 1.0
                            }]
                            
                            result = self.merger.create_ml_ready_parquet(
                                datasets,
                                output_dir,
                                chunk_size=None  # In-memory mode
                            )
                            
                            self.assertTrue(str(result).endswith('ml_ready_aligned_data.parquet'))


if __name__ == '__main__':
    unittest.main()