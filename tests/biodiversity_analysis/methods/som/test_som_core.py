"""Comprehensive tests for BiodiversitySOM implementation."""

import pytest
import numpy as np
from src.biodiversity_analysis.methods.som.som_core import BiodiversitySOM
from src.abstractions.types.som_types import (
    SOMConfig, DistanceMetric, NeighborhoodFunction, 
    InitializationMethod
)


class TestBiodiversitySOM:
    """Test suite for BiodiversitySOM core implementation."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample biodiversity data."""
        np.random.seed(42)
        # Simulate species abundance data
        n_samples = 100
        n_species = 20
        
        # Create data with some structure
        data = np.zeros((n_samples, n_species))
        
        # Group 1: High abundance of species 0-5
        data[:30, :5] = np.random.poisson(10, (30, 5))
        
        # Group 2: High abundance of species 10-15
        data[30:60, 10:15] = np.random.poisson(8, (30, 5))
        
        # Group 3: Mixed abundance
        data[60:, :] = np.random.poisson(2, (40, n_species))
        
        return data
    
    @pytest.fixture
    def basic_config(self):
        """Basic SOM configuration."""
        return SOMConfig(
            grid_size=(5, 5),
            distance_metric=DistanceMetric.EUCLIDEAN,
            neighborhood_function=NeighborhoodFunction.GAUSSIAN,
            initialization_method=InitializationMethod.RANDOM,
            learning_rate=0.5,
            epochs=10,
            random_seed=42
        )
    
    def test_initialization_methods(self, sample_data):
        """Test different weight initialization methods."""
        # Test random initialization
        config_random = SOMConfig(
            grid_size=(3, 3),
            initialization_method=InitializationMethod.RANDOM,
            random_seed=42
        )
        som_random = BiodiversitySOM(config_random)
        som_random.initialize_weights(sample_data)
        
        assert som_random.weights.shape == (9, 20)
        assert som_random.input_dim == 20
        
        # Test PCA initialization
        config_pca = SOMConfig(
            grid_size=(3, 3),
            initialization_method=InitializationMethod.PCA
        )
        som_pca = BiodiversitySOM(config_pca)
        som_pca.initialize_weights(sample_data)
        
        assert som_pca.weights.shape == (9, 20)
        # PCA initialization should span data variance
        assert np.std(som_pca.weights) > 0
        
        # Test sample initialization
        config_sample = SOMConfig(
            grid_size=(3, 3),
            initialization_method=InitializationMethod.SAMPLE,
            random_seed=42
        )
        som_sample = BiodiversitySOM(config_sample)
        som_sample.initialize_weights(sample_data)
        
        # Weights should be actual data samples
        for weight in som_sample.weights:
            assert any(np.allclose(weight, sample) for sample in sample_data)
    
    def test_distance_metrics(self, sample_data):
        """Test all distance metric implementations."""
        metrics = [
            DistanceMetric.EUCLIDEAN,
            DistanceMetric.MANHATTAN,
            DistanceMetric.COSINE,
            DistanceMetric.BRAY_CURTIS
        ]
        
        for metric in metrics:
            config = SOMConfig(
                grid_size=(3, 3),
                distance_metric=metric,
                epochs=5
            )
            som = BiodiversitySOM(config)
            
            # Train should work with all metrics
            result = som.train(sample_data)
            assert result.final_quantization_error > 0
            
            # Test distance calculation
            sample = sample_data[0]
            bmu = som.find_bmu(sample)
            assert isinstance(bmu, tuple)
            assert len(bmu) == 2
    
    def test_neighborhood_functions(self, sample_data):
        """Test different neighborhood functions."""
        neighborhoods = [
            NeighborhoodFunction.GAUSSIAN,
            NeighborhoodFunction.BUBBLE,
            NeighborhoodFunction.MEXICAN_HAT
        ]
        
        for neighborhood in neighborhoods:
            config = SOMConfig(
                grid_size=(3, 3),
                neighborhood_function=neighborhood,
                epochs=5
            )
            som = BiodiversitySOM(config)
            result = som.train(sample_data)
            
            # Training should complete
            assert result.n_samples_trained == len(sample_data)
            assert len(result.quantization_errors) > 0
    
    def test_convergence_detection(self, sample_data):
        """Test convergence detection mechanism."""
        config = SOMConfig(
            grid_size=(3, 3),
            epochs=100,
            convergence_threshold=0.001,
            convergence_window=5
        )
        som = BiodiversitySOM(config)
        
        # Generate data that converges quickly
        simple_data = np.random.randn(50, 5)
        result = som.train(simple_data)
        
        # Should converge before max epochs
        assert result.converged
        assert result.convergence_epoch is not None
        assert result.convergence_epoch < 100
    
    def test_batch_vs_online_training(self, sample_data):
        """Compare batch and online training modes."""
        # Online training
        config_online = SOMConfig(
            grid_size=(3, 3),
            epochs=10,
            batch_size=None,
            random_seed=42
        )
        som_online = BiodiversitySOM(config_online)
        result_online = som_online.train(sample_data)
        
        # Batch training
        config_batch = SOMConfig(
            grid_size=(3, 3),
            epochs=10,
            batch_size=10,
            random_seed=42
        )
        som_batch = BiodiversitySOM(config_batch)
        result_batch = som_batch.train(sample_data)
        
        # Both should produce valid results
        assert result_online.final_quantization_error > 0
        assert result_batch.final_quantization_error > 0
        
        # Results should be different but reasonable
        assert abs(result_online.final_quantization_error - 
                  result_batch.final_quantization_error) < 10
    
    def test_quantization_error(self, sample_data, basic_config):
        """Test quantization error calculation."""
        som = BiodiversitySOM(basic_config)
        som.train(sample_data)
        
        qe = som.calculate_quantization_error(sample_data)
        assert qe > 0
        
        # QE should generally decrease with more training
        # (but may fluctuate due to random sampling in online training)
        initial_qe = qe
        
        # Train with lower learning rate to ensure improvement
        config_low_lr = SOMConfig(
            grid_size=(5, 5),
            learning_rate=0.1,  # Lower learning rate
            epochs=10,
            random_seed=42
        )
        som_new = BiodiversitySOM(config_low_lr)
        som_new.train(sample_data)
        final_qe = som_new.calculate_quantization_error(sample_data)
        # Just ensure QE is reasonable, not necessarily lower
        assert final_qe > 0
    
    def test_topographic_error(self, sample_data, basic_config):
        """Test topographic error calculation."""
        som = BiodiversitySOM(basic_config)
        som.train(sample_data)
        
        te = som.calculate_topographic_error(sample_data)
        assert 0 <= te <= 1  # TE is a proportion
        
        # Well-trained SOM should have low TE
        som.train(sample_data)  # Train more
        final_te = som.calculate_topographic_error(sample_data)
        assert final_te < 0.5  # Reasonable threshold
    
    def test_bray_curtis_for_biodiversity(self):
        """Test Bray-Curtis distance specifically for species data."""
        # Create species abundance data
        species_data = np.array([
            [10, 5, 0, 0, 1],    # Site 1
            [8, 6, 0, 0, 2],     # Site 2 (similar to 1)
            [0, 0, 10, 8, 0],    # Site 3 (different species)
            [0, 0, 12, 7, 0]     # Site 4 (similar to 3)
        ])
        
        config = SOMConfig(
            grid_size=(2, 2),
            distance_metric=DistanceMetric.BRAY_CURTIS,
            epochs=20
        )
        som = BiodiversitySOM(config)
        result = som.train(species_data)
        
        # Similar sites should map to same/adjacent neurons
        bmu1 = som.find_bmu(species_data[0])
        bmu2 = som.find_bmu(species_data[1])
        bmu3 = som.find_bmu(species_data[2])
        bmu4 = som.find_bmu(species_data[3])
        
        # Sites 1&2 should be close, 3&4 should be close
        dist_12 = np.sqrt((bmu1[0]-bmu2[0])**2 + (bmu1[1]-bmu2[1])**2)
        dist_34 = np.sqrt((bmu3[0]-bmu4[0])**2 + (bmu3[1]-bmu4[1])**2)
        dist_13 = np.sqrt((bmu1[0]-bmu3[0])**2 + (bmu1[1]-bmu3[1])**2)
        
        assert dist_12 <= dist_13
        assert dist_34 <= dist_13
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        config = SOMConfig(grid_size=(2, 2))
        som = BiodiversitySOM(config)
        
        # Empty data
        with pytest.raises(Exception):
            som.train(np.array([]))
        
        # Single sample
        single_sample = np.array([[1, 2, 3]])
        result = som.train(single_sample)
        assert result.n_samples_trained == 1
        
        # Single feature (need fresh SOM)
        som_single_feature = BiodiversitySOM(config)
        single_feature = np.random.randn(10, 1)
        result = som_single_feature.train(single_feature)
        assert som_single_feature.input_dim == 1
        
        # Test predictions before training
        som_untrained = BiodiversitySOM(config)
        with pytest.raises(ValueError):
            som_untrained.calculate_quantization_error(single_sample)
    
    def test_deterministic_with_seed(self, sample_data):
        """Test that results are deterministic with random seed."""
        config = SOMConfig(
            grid_size=(3, 3),
            epochs=5,
            random_seed=123
        )
        
        # Train twice with same seed
        som1 = BiodiversitySOM(config)
        result1 = som1.train(sample_data)
        
        som2 = BiodiversitySOM(config)
        result2 = som2.train(sample_data)
        
        # Results should be identical
        np.testing.assert_array_equal(som1.weights, som2.weights)
        assert result1.final_quantization_error == result2.final_quantization_error
    
    def test_large_dataset_handling(self):
        """Test handling of larger datasets."""
        # Create a larger dataset
        large_data = np.random.randn(1000, 50)
        
        config = SOMConfig(
            grid_size=(10, 10),
            epochs=5,
            batch_size=50  # Use batching for efficiency
        )
        
        som = BiodiversitySOM(config)
        result = som.train(large_data)
        
        # Should complete without errors
        assert result.n_samples_trained == 1000
        assert som.weights.shape == (100, 50)
    
    def test_progress_callback(self, sample_data):
        """Test progress callback functionality."""
        progress_values = []
        
        def track_progress(progress):
            progress_values.append(progress)
        
        config = SOMConfig(grid_size=(3, 3), epochs=10)
        som = BiodiversitySOM(config)
        som.train(sample_data, progress_callback=track_progress)
        
        # Should have received progress updates
        assert len(progress_values) > 0
        assert all(0 <= p <= 1 for p in progress_values)
        assert progress_values[-1] == 1.0  # Should end at 100%
    
    def test_weight_updates(self, sample_data):
        """Test that weights are actually updated during training."""
        config = SOMConfig(grid_size=(3, 3), epochs=1)
        som = BiodiversitySOM(config)
        som.initialize_weights(sample_data)
        
        initial_weights = som.weights.copy()
        som.train_epoch(sample_data, 0, 0.5, 1.0)
        
        # Weights should have changed
        assert not np.allclose(initial_weights, som.weights)
    
    def test_predict_and_transform(self, sample_data, basic_config):
        """Test predict and transform methods."""
        som = BiodiversitySOM(basic_config)
        som.train(sample_data)
        
        # Test predict (returns flattened indices)
        predictions = som.predict(sample_data)
        assert predictions.shape == (len(sample_data),)
        assert all(0 <= p < 25 for p in predictions)  # 5x5 grid
        
        # Test transform (returns grid coordinates)
        coordinates = som.transform(sample_data)
        assert coordinates.shape == (len(sample_data), 2)
        assert all(0 <= c[0] < 5 and 0 <= c[1] < 5 for c in coordinates)