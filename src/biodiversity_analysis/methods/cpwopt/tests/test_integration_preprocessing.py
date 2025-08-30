"""Integration tests for tensor preprocessing pipeline."""

import unittest
import numpy as np
import pandas as pd
import os
import tempfile

from src.biodiversity_analysis.preprocessing.tensor_preprocessing import TensorPreprocessor
from src.biodiversity_analysis.methods.cpwopt.core import CPWOPT
from src.core_abstractions import TensorData


class TestPreprocessingIntegration(unittest.TestCase):
    """Test preprocessing integration with CP-WOPT."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create realistic biodiversity data
        n_samples = 1000
        n_species = 20
        
        # Generate spatial coordinates
        self.latitudes = np.random.uniform(-45, 45, n_samples)
        self.longitudes = np.random.uniform(-180, 180, n_samples)
        
        # Generate species counts with realistic patterns
        self.species_data = {}
        for i in range(n_species):
            # Different species have different abundance patterns
            if i < 5:  # Common species
                counts = np.random.poisson(10, n_samples)
            elif i < 15:  # Uncommon species
                counts = np.random.poisson(3, n_samples)
            else:  # Rare species
                counts = np.random.poisson(0.5, n_samples)
            
            # Add spatial structure
            lat_effect = np.exp(-0.1 * np.abs(self.latitudes - np.random.uniform(-20, 20)))
            counts = (counts * lat_effect).astype(int)
            
            self.species_data[f'species_{i:02d}'] = counts
        
        # Create DataFrame
        self.df = pd.DataFrame({
            'latitude': self.latitudes,
            'longitude': self.longitudes,
            **self.species_data
        })
        
        # Add missing values (10% randomly)
        for col in self.species_data.keys():
            mask = np.random.rand(n_samples) < 0.1
            self.df.loc[mask, col] = np.nan
    
    def test_preprocessing_to_cpwopt_pipeline(self):
        """Test full pipeline from DataFrame to CP-WOPT."""
        # Configure preprocessor
        preprocessor = TensorPreprocessor({
            'lat_bins': 20,
            'lon_bins': 20,
            'aggregation': 'sum',
            'transform': 'log1p'
        })
        
        # Convert to tensor
        tensor_data = preprocessor.dataframe_to_tensor(self.df)
        
        # Check tensor properties
        self.assertEqual(tensor_data.shape, (20, 20, 20))
        self.assertGreater(np.sum(tensor_data.weights > 0), 0)
        
        # Fit CP-WOPT
        model = CPWOPT(rank=5, max_iterations=30, verbose=False)
        model.fit(tensor_data)
        
        # Check fitting succeeded
        self.assertIsNotNone(model.factors_)
        self.assertGreater(model.metrics_.fit, 0.1)
    
    def test_different_aggregations(self):
        """Test different aggregation methods."""
        aggregations = ['sum', 'mean', 'max']
        
        for agg in aggregations:
            with self.subTest(aggregation=agg):
                preprocessor = TensorPreprocessor({
                    'lat_bins': 10,
                    'lon_bins': 10,
                    'aggregation': agg
                })
                
                tensor_data = preprocessor.dataframe_to_tensor(self.df)
                
                # Fit model
                model = CPWOPT(rank=3, max_iterations=20, verbose=False)
                model.fit(tensor_data)
                
                # Should work for all aggregations
                self.assertIsNotNone(model.factors_)
    
    def test_different_transformations(self):
        """Test different data transformations."""
        transforms = ['none', 'log1p', 'sqrt', 'asinh']
        
        for transform in transforms:
            with self.subTest(transform=transform):
                preprocessor = TensorPreprocessor({
                    'lat_bins': 10,
                    'lon_bins': 10,
                    'transform': transform
                })
                
                tensor_data = preprocessor.dataframe_to_tensor(self.df)
                
                # Check no NaN or inf
                self.assertFalse(np.any(np.isnan(tensor_data.tensor)))
                self.assertFalse(np.any(np.isinf(tensor_data.tensor)))
                
                # Fit model
                model = CPWOPT(rank=3, max_iterations=20, verbose=False)
                model.fit(tensor_data)
                
                self.assertIsNotNone(model.factors_)
    
    def test_extreme_sparsity_handling(self):
        """Test handling of extremely sparse data."""
        # Create very sparse data
        sparse_df = self.df.copy()
        
        # Make 95% of entries zero
        for col in self.species_data.keys():
            mask = np.random.rand(len(sparse_df)) < 0.95
            sparse_df.loc[mask, col] = 0
        
        preprocessor = TensorPreprocessor({
            'lat_bins': 10,
            'lon_bins': 10,
            'aggregation': 'sum'
        })
        
        tensor_data = preprocessor.dataframe_to_tensor(sparse_df)
        
        # Calculate sparsity
        sparsity = np.sum(tensor_data.tensor == 0) / tensor_data.tensor.size
        self.assertGreater(sparsity, 0.9)
        
        # Should still be able to fit
        model = CPWOPT(rank=2, max_iterations=20, verbose=False)
        model.fit(tensor_data)
        
        self.assertIsNotNone(model.factors_)
    
    def test_geographic_binning_accuracy(self):
        """Test that geographic binning preserves spatial patterns."""
        # Create data with clear latitudinal gradient
        gradient_df = pd.DataFrame({
            'latitude': np.linspace(-45, 45, 100),
            'longitude': np.zeros(100),
            'gradient_species': np.linspace(0, 100, 100)
        })
        
        preprocessor = TensorPreprocessor({
            'lat_bins': 10,
            'lon_bins': 5,
            'aggregation': 'mean'
        })
        
        tensor_data = preprocessor.dataframe_to_tensor(gradient_df)
        
        # Check gradient is preserved
        # Average values should increase with latitude bin
        lat_averages = []
        for i in range(10):
            lat_slice = tensor_data.tensor[i, :, 0]
            if np.any(lat_slice > 0):
                lat_averages.append(np.mean(lat_slice[lat_slice > 0]))
        
        # Check monotonic increase
        for i in range(1, len(lat_averages)):
            self.assertGreater(lat_averages[i], lat_averages[i-1] * 0.9)
    
    def test_missing_value_propagation(self):
        """Test that missing values are properly handled through pipeline."""
        # Create data with structured missing values
        missing_df = self.df.copy()
        
        # Make entire region missing
        region_mask = (self.df['latitude'] > 0) & (self.df['longitude'] > 0)
        for col in self.species_data.keys():
            missing_df.loc[region_mask, col] = np.nan
        
        preprocessor = TensorPreprocessor({
            'lat_bins': 10,
            'lon_bins': 10
        })
        
        tensor_data = preprocessor.dataframe_to_tensor(missing_df)
        
        # Check weights are zero in missing region
        # Upper right quadrant should have low weights
        upper_right_weights = tensor_data.weights[5:, 5:, :].mean()
        other_weights = tensor_data.weights[:5, :5, :].mean()
        
        self.assertLess(upper_right_weights, other_weights * 0.5)
    
    def test_feature_name_preservation(self):
        """Test that feature names are preserved through pipeline."""
        preprocessor = TensorPreprocessor({
            'lat_bins': 5,
            'lon_bins': 5
        })
        
        tensor_data = preprocessor.dataframe_to_tensor(self.df)
        
        # All species should be in feature names
        for i in range(20):
            self.assertIn(f'species_{i:02d}', tensor_data.feature_names)
        
        # Fit model and check feature names are accessible
        model = CPWOPT(rank=3, max_iterations=10, verbose=False)
        model.fit(tensor_data)
        
        # Get top features
        top_features = model.get_top_features(component=0, n_top=5)
        feature_indices = top_features['features']['indices']
        
        # Should be able to map back to species names
        for idx in feature_indices:
            self.assertLess(idx, len(tensor_data.feature_names))
    
    def test_large_scale_performance(self):
        """Test performance with larger dataset."""
        # Create larger dataset
        n_samples = 10000
        large_df = pd.DataFrame({
            'latitude': np.random.uniform(-90, 90, n_samples),
            'longitude': np.random.uniform(-180, 180, n_samples)
        })
        
        # Add 50 species
        for i in range(50):
            large_df[f'sp_{i}'] = np.random.poisson(5, n_samples)
        
        preprocessor = TensorPreprocessor({
            'lat_bins': 30,
            'lon_bins': 30,
            'aggregation': 'sum'
        })
        
        # Time the preprocessing
        import time
        start_time = time.time()
        tensor_data = preprocessor.dataframe_to_tensor(large_df)
        preprocess_time = time.time() - start_time
        
        # Should be reasonably fast
        self.assertLess(preprocess_time, 5.0)  # Less than 5 seconds
        
        # Fit small model
        model = CPWOPT(rank=5, max_iterations=10, verbose=False)
        start_time = time.time()
        model.fit(tensor_data)
        fit_time = time.time() - start_time
        
        # Should complete in reasonable time
        self.assertLess(fit_time, 30.0)  # Less than 30 seconds
    
    def test_memory_efficiency(self):
        """Test memory efficiency of preprocessing."""
        # Create data that would be large if dense
        sparse_df = pd.DataFrame({
            'latitude': np.random.uniform(-90, 90, 100),
            'longitude': np.random.uniform(-180, 180, 100)
        })
        
        # Add many features but mostly zeros
        for i in range(100):
            # Only 5% non-zero
            values = np.zeros(100)
            non_zero_idx = np.random.choice(100, 5, replace=False)
            values[non_zero_idx] = np.random.poisson(5, 5)
            sparse_df[f'feature_{i}'] = values
        
        preprocessor = TensorPreprocessor({
            'lat_bins': 50,
            'lon_bins': 50
        })
        
        tensor_data = preprocessor.dataframe_to_tensor(sparse_df)
        
        # Check tensor is marked as sparse
        self.assertTrue(tensor_data.is_sparse)
        
        # Fit with memory-efficient settings
        model = CPWOPT(
            rank=3, 
            max_iterations=10,
            use_minibatch=True,
            minibatch_size=100,
            verbose=False
        )
        
        # Should handle without memory issues
        model.fit(tensor_data)
        self.assertIsNotNone(model.factors_)


class TestRealWorldScenarios(unittest.TestCase):
    """Test realistic biodiversity analysis scenarios."""
    
    def test_invasive_species_detection(self):
        """Test detection of invasive species patterns."""
        # Create data with native and invasive species
        n_samples = 500
        
        # Native species follow natural gradients
        lat = np.random.uniform(-30, 30, n_samples)
        lon = np.random.uniform(-60, 60, n_samples)
        
        native_sp1 = np.random.poisson(5 * np.exp(-0.1 * np.abs(lat)), n_samples)
        native_sp2 = np.random.poisson(3 * np.exp(-0.05 * np.abs(lon)), n_samples)
        
        # Invasive species has different pattern (urban areas)
        urban_centers = [(0, 0), (-20, 40), (15, -30)]
        invasive = np.zeros(n_samples)
        
        for center_lat, center_lon in urban_centers:
            dist = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2)
            invasive += np.random.poisson(10 * np.exp(-0.1 * dist), n_samples)
        
        df = pd.DataFrame({
            'latitude': lat,
            'longitude': lon,
            'native_sp1': native_sp1,
            'native_sp2': native_sp2,
            'invasive_sp': invasive
        })
        
        # Process and analyze
        preprocessor = TensorPreprocessor({
            'lat_bins': 15,
            'lon_bins': 15,
            'transform': 'log1p'
        })
        
        tensor_data = preprocessor.dataframe_to_tensor(df)
        
        model = CPWOPT(rank=3, max_iterations=50, verbose=False)
        model.fit(tensor_data)
        
        # Check that invasive species loads differently
        matrices = model.get_component_matrices()
        feature_loadings = matrices['features']
        
        # Invasive species should have distinct loading pattern
        invasive_idx = tensor_data.feature_names.index('invasive_sp')
        native1_idx = tensor_data.feature_names.index('native_sp1')
        
        # Calculate loading similarity
        invasive_loading = feature_loadings[invasive_idx, :]
        native_loading = feature_loadings[native1_idx, :]
        
        similarity = np.dot(invasive_loading, native_loading) / (
            np.linalg.norm(invasive_loading) * np.linalg.norm(native_loading)
        )
        
        # Should be dissimilar
        self.assertLess(similarity, 0.7)
    
    def test_biodiversity_hotspot_identification(self):
        """Test identification of biodiversity hotspots."""
        # Create data with hotspots
        n_samples = 1000
        
        lat = np.random.uniform(-45, 45, n_samples)
        lon = np.random.uniform(-90, 90, n_samples)
        
        # Define hotspot regions
        hotspots = [
            {'center': (0, 0), 'radius': 10, 'richness': 20},
            {'center': (-20, 50), 'radius': 15, 'richness': 15},
            {'center': (30, -60), 'radius': 12, 'richness': 25}
        ]
        
        df = pd.DataFrame({'latitude': lat, 'longitude': lon})
        
        # Generate species for each hotspot
        n_species = 30
        for i in range(n_species):
            species_counts = np.zeros(n_samples)
            
            # Each species associated with one or more hotspots
            for hotspot in hotspots:
                if np.random.rand() < 0.5:  # 50% chance for each hotspot
                    center = hotspot['center']
                    dist = np.sqrt((lat - center[0])**2 + (lon - center[1])**2)
                    in_hotspot = dist < hotspot['radius']
                    species_counts[in_hotspot] += np.random.poisson(
                        hotspot['richness'] / 10, np.sum(in_hotspot)
                    )
            
            df[f'species_{i}'] = species_counts
        
        # Analyze
        preprocessor = TensorPreprocessor({
            'lat_bins': 20,
            'lon_bins': 20,
            'transform': 'sqrt'
        })
        
        tensor_data = preprocessor.dataframe_to_tensor(df)
        
        model = CPWOPT(rank=5, max_iterations=100, verbose=False)
        model.fit(tensor_data)
        
        # Reconstruct to find hotspots
        reconstructed = model.predict()
        
        # Sum across species to get richness
        richness = np.sum(reconstructed, axis=2)
        
        # Hotspots should have high richness
        hotspot_threshold = np.percentile(richness[richness > 0], 80)
        high_richness_cells = richness > hotspot_threshold
        
        # Should identify multiple hotspot regions
        self.assertGreater(np.sum(high_richness_cells), 5)
        self.assertLess(np.sum(high_richness_cells), 50)


if __name__ == '__main__':
    unittest.main()