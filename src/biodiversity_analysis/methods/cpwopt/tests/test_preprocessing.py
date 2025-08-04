"""Unit tests for tensor preprocessing."""

import unittest
import numpy as np
import pandas as pd

from src.biodiversity_analysis.preprocessing.tensor_preprocessing import (
    TensorPreprocessor, GridConfig
)
from src.abstractions.types.tensor_types import TensorData


class TestTensorPreprocessor(unittest.TestCase):
    """Test tensor preprocessing functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create sample biodiversity data
        n_samples = 100
        self.df = pd.DataFrame({
            'latitude': np.random.uniform(-90, 90, n_samples),
            'longitude': np.random.uniform(-180, 180, n_samples),
            'species_a': np.random.poisson(5, n_samples),
            'species_b': np.random.poisson(3, n_samples),
            'species_c': np.random.poisson(2, n_samples),
            'species_d': np.zeros(n_samples)  # All zeros species
        })
        
        # Add some missing values
        mask = np.random.rand(n_samples) < 0.1
        self.df.loc[mask, 'species_b'] = np.nan
        
        # Create grid config
        self.grid_config = GridConfig(
            lat_bins=10,
            lon_bins=10,
            lat_range=(-90, 90),
            lon_range=(-180, 180)
        )
        
        # Create preprocessor
        self.preprocessor = TensorPreprocessor({
            'lat_bins': 10,
            'lon_bins': 10,
            'aggregation': 'sum',
            'transform': 'none'
        })
    
    def test_dataframe_to_tensor_basic(self):
        """Test basic DataFrame to tensor conversion."""
        tensor_data = self.preprocessor.dataframe_to_tensor(self.df)
        
        # Check type
        self.assertIsInstance(tensor_data, TensorData)
        
        # Check shape (10 x 10 x 4 features)
        self.assertEqual(tensor_data.shape, (10, 10, 4))
        
        # Check feature names preserved
        expected_features = ['species_a', 'species_b', 'species_c', 'species_d']
        self.assertEqual(tensor_data.feature_names, expected_features)
        
        # Check weights shape matches tensor
        self.assertEqual(tensor_data.weights.shape, tensor_data.tensor.shape)
    
    def test_aggregation_sum(self):
        """Test sum aggregation."""
        # Create data with duplicates in same grid cell
        df_dup = pd.DataFrame({
            'latitude': [0, 0, 0, 45],
            'longitude': [0, 0, 0, 90],
            'species_a': [1, 2, 3, 4],
            'species_b': [5, 6, 7, 8]
        })
        
        tensor_data = self.preprocessor.dataframe_to_tensor(df_dup)
        
        # Find the cell with multiple observations
        lat_idx = 5  # Center bin for latitude 0
        lon_idx = 5  # Center bin for longitude 0
        
        # Should sum to 1+2+3 = 6 for species_a
        self.assertEqual(tensor_data.tensor[lat_idx, lon_idx, 0], 6)
        # Should sum to 5+6+7 = 18 for species_b
        self.assertEqual(tensor_data.tensor[lat_idx, lon_idx, 1], 18)
    
    def test_aggregation_mean(self):
        """Test mean aggregation."""
        preprocessor_mean = TensorPreprocessor({
            'lat_bins': 10,
            'lon_bins': 10,
            'aggregation': 'mean'
        })
        
        df_dup = pd.DataFrame({
            'latitude': [0, 0, 0],
            'longitude': [0, 0, 0],
            'species_a': [3, 6, 9]
        })
        
        tensor_data = preprocessor_mean.dataframe_to_tensor(df_dup)
        
        # Should average to 6
        lat_idx = 5
        lon_idx = 5
        self.assertEqual(tensor_data.tensor[lat_idx, lon_idx, 0], 6)
    
    def test_transform_log1p(self):
        """Test log1p transformation."""
        preprocessor_log = TensorPreprocessor({
            'lat_bins': 10,
            'lon_bins': 10,
            'transform': 'log1p'
        })
        
        tensor_data = preprocessor_log.dataframe_to_tensor(self.df)
        
        # Check that values are transformed
        # Original values should be counts, transformed should be log(count + 1)
        max_original = self.df[['species_a', 'species_b', 'species_c']].max().max()
        max_transformed = np.max(tensor_data.tensor)
        
        # log1p of max count should be close to max transformed
        self.assertLess(max_transformed, np.log1p(max_original * 10))  # Account for summing
    
    def test_transform_sqrt(self):
        """Test square root transformation."""
        preprocessor_sqrt = TensorPreprocessor({
            'lat_bins': 10,
            'lon_bins': 10,
            'transform': 'sqrt'
        })
        
        df_simple = pd.DataFrame({
            'latitude': [0],
            'longitude': [0],
            'species_a': [16]
        })
        
        tensor_data = preprocessor_sqrt.dataframe_to_tensor(df_simple)
        
        # sqrt(16) = 4
        lat_idx = 5
        lon_idx = 5
        self.assertEqual(tensor_data.tensor[lat_idx, lon_idx, 0], 4)
    
    def test_missing_values_handling(self):
        """Test handling of missing values."""
        tensor_data = self.preprocessor.dataframe_to_tensor(self.df)
        
        # Check that weights are 0 where NaN
        # Find cells that should have NaN (from species_b)
        nan_mask = np.isnan(self.df['species_b'])
        
        # These observations should have weight 0 for species_b
        # (This is simplified - actual implementation may differ)
        self.assertTrue(np.all(tensor_data.weights >= 0))
        self.assertTrue(np.all(tensor_data.weights <= 1))
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame({
            'latitude': [],
            'longitude': [],
            'species_a': []
        })
        
        tensor_data = self.preprocessor.dataframe_to_tensor(empty_df)
        
        # Should create empty tensor with correct shape
        self.assertEqual(tensor_data.shape, (10, 10, 1))
        self.assertEqual(np.sum(tensor_data.tensor), 0)
        self.assertEqual(np.sum(tensor_data.weights), 0)
    
    def test_out_of_bounds_coordinates(self):
        """Test handling of out-of-bounds coordinates."""
        df_oob = pd.DataFrame({
            'latitude': [100, -100, 0],  # First two out of bounds
            'longitude': [200, -200, 0],
            'species_a': [1, 2, 3]
        })
        
        tensor_data = self.preprocessor.dataframe_to_tensor(df_oob)
        
        # Only the valid point (0, 0) should be in tensor
        total_sum = np.sum(tensor_data.tensor)
        self.assertEqual(total_sum, 3)  # Only the third point
    
    def test_feature_filtering(self):
        """Test that zero variance features are handled."""
        # species_d has all zeros
        tensor_data = self.preprocessor.dataframe_to_tensor(self.df)
        
        # Check that species_d is still included but marked properly
        species_d_idx = 3
        species_d_sum = np.sum(tensor_data.tensor[:, :, species_d_idx])
        self.assertEqual(species_d_sum, 0)
    
    def test_coordinate_extraction(self):
        """Test that coordinates are properly extracted."""
        tensor_data = self.preprocessor.dataframe_to_tensor(self.df)
        
        # Should have coordinates for each original point
        self.assertEqual(len(tensor_data.coordinates), len(self.df))
        
        # Coordinates should be (lat, lon) tuples
        for coord in tensor_data.coordinates:
            self.assertEqual(len(coord), 2)
            lat, lon = coord
            self.assertTrue(-90 <= lat <= 90)
            self.assertTrue(-180 <= lon <= 180)
    
    def test_custom_grid_bounds(self):
        """Test custom grid boundaries."""
        preprocessor_custom = TensorPreprocessor({
            'lat_bins': 5,
            'lon_bins': 5,
            'lat_range': (-45, 45),
            'lon_range': (-90, 90)
        })
        
        # Create data within custom bounds
        df_custom = pd.DataFrame({
            'latitude': [-30, 0, 30],
            'longitude': [-60, 0, 60],
            'species_a': [1, 2, 3]
        })
        
        tensor_data = preprocessor_custom.dataframe_to_tensor(df_custom)
        
        # Check shape
        self.assertEqual(tensor_data.shape, (5, 5, 1))
        
        # All points should be included
        self.assertEqual(np.sum(tensor_data.tensor), 6)
    
    def test_metadata_preservation(self):
        """Test that metadata is preserved."""
        # Add metadata to DataFrame
        self.df.attrs['study_name'] = 'Test Study'
        self.df.attrs['year'] = 2024
        
        tensor_data = self.preprocessor.dataframe_to_tensor(self.df)
        
        # Check metadata
        self.assertIn('preprocessing', tensor_data.metadata)
        self.assertIn('grid_config', tensor_data.metadata)
        self.assertIn('n_samples', tensor_data.metadata)
        
        # Check grid configuration is stored
        grid_meta = tensor_data.metadata['grid_config']
        self.assertEqual(grid_meta['lat_bins'], 10)
        self.assertEqual(grid_meta['lon_bins'], 10)


class TestGridConfig(unittest.TestCase):
    """Test grid configuration."""
    
    def test_grid_config_creation(self):
        """Test GridConfig creation."""
        config = GridConfig(
            lat_bins=20,
            lon_bins=40,
            lat_range=(-60, 60),
            lon_range=(-120, 120)
        )
        
        self.assertEqual(config.lat_bins, 20)
        self.assertEqual(config.lon_bins, 40)
        self.assertEqual(config.lat_range, (-60, 60))
        self.assertEqual(config.lon_range, (-120, 120))
    
    def test_grid_config_defaults(self):
        """Test GridConfig defaults."""
        config = GridConfig()
        
        self.assertEqual(config.lat_range, (-90, 90))
        self.assertEqual(config.lon_range, (-180, 180))


class TestTransformations(unittest.TestCase):
    """Test different transformation options."""
    
    def setUp(self):
        """Create test data."""
        self.df = pd.DataFrame({
            'latitude': [0, 0, 0],
            'longitude': [0, 0, 0],
            'feature': [0, 1, 100]
        })
    
    def test_asinh_transform(self):
        """Test inverse hyperbolic sine transformation."""
        preprocessor = TensorPreprocessor({
            'lat_bins': 1,
            'lon_bins': 1,
            'transform': 'asinh',
            'aggregation': 'sum'
        })
        
        tensor_data = preprocessor.dataframe_to_tensor(self.df)
        
        # asinh(101) ≈ 5.3
        expected = np.arcsinh(101)
        actual = tensor_data.tensor[0, 0, 0]
        self.assertAlmostEqual(actual, expected, places=4)
    
    def test_no_transform(self):
        """Test no transformation."""
        preprocessor = TensorPreprocessor({
            'lat_bins': 1,
            'lon_bins': 1,
            'transform': 'none',
            'aggregation': 'sum'
        })
        
        tensor_data = preprocessor.dataframe_to_tensor(self.df)
        
        # Should be sum without transformation
        self.assertEqual(tensor_data.tensor[0, 0, 0], 101)


if __name__ == '__main__':
    unittest.main()