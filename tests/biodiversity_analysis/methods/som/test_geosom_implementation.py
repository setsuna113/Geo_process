"""Tests for the new GeoSOM + VLRSOM implementation."""

import pytest
import numpy as np
from src.biodiversity_analysis.methods.som.geo_som_core import GeoSOMVLRSOM, GeoSOMConfig
from src.biodiversity_analysis.methods.som.preprocessing import BiodiversityPreprocessor
from src.biodiversity_analysis.methods.som.partial_metrics import partial_bray_curtis_numba
from src.biodiversity_analysis.methods.som.spatial_utils import haversine_distance_numba


class TestGeoSOMCore:
    """Test suite for GeoSOM + VLRSOM core implementation."""
    
    def test_partial_bray_curtis(self):
        """Test partial Bray-Curtis distance with missing values."""
        # Test vectors with NaN
        u = np.array([1.0, 2.0, np.nan, 4.0])
        v = np.array([2.0, 3.0, 5.0, np.nan])
        
        # Should only compare first two elements
        dist = partial_bray_curtis_numba(u, v, min_valid=2)
        expected = (abs(1-2) + abs(2-3)) / (1+2 + 2+3)
        assert np.isclose(dist, expected)
        
        # Test with insufficient valid pairs
        u_sparse = np.array([np.nan, np.nan, np.nan, 1.0])
        v_sparse = np.array([np.nan, np.nan, np.nan, 2.0])
        dist_sparse = partial_bray_curtis_numba(u_sparse, v_sparse, min_valid=2)
        assert np.isnan(dist_sparse)
    
    def test_haversine_distance(self):
        """Test haversine distance calculation."""
        # Test known distance (roughly London to Paris)
        lat1, lon1 = 51.5074, -0.1278  # London
        lat2, lon2 = 48.8566, 2.3522   # Paris
        
        dist = haversine_distance_numba(lat1, lon1, lat2, lon2)
        # Should be approximately 344 km
        assert 340 < dist < 350
    
    def test_preprocessing_pipeline(self):
        """Test the log1p + separate z-score preprocessing."""
        # Create test data with observed and predicted columns
        np.random.seed(42)
        data = np.random.poisson(5, (100, 4)).astype(float)
        
        # Add some NaN values
        data[10:20, 0] = np.nan
        data[30:35, 2] = np.nan
        
        # Define column types
        observed_cols = [0, 1]
        predicted_cols = [2, 3]
        
        # Apply preprocessing
        preprocessor = BiodiversityPreprocessor()
        data_transformed = preprocessor.fit_transform(data, observed_cols, predicted_cols)
        
        # Check that NaN values are preserved
        assert np.isnan(data_transformed[10:20, 0]).all()
        assert np.isnan(data_transformed[30:35, 2]).all()
        
        # Check that non-NaN values are transformed
        assert not np.array_equal(data[~np.isnan(data)], data_transformed[~np.isnan(data_transformed)])
        
        # Check that observed and predicted are standardized separately
        obs_mean = np.nanmean(data_transformed[:, observed_cols])
        pred_mean = np.nanmean(data_transformed[:, predicted_cols])
        
        # Should be close to 0 (z-score)
        assert abs(obs_mean) < 0.1
        assert abs(pred_mean) < 0.1
    
    def test_geosom_initialization(self):
        """Test GeoSOM initialization and basic functionality."""
        config = GeoSOMConfig(
            grid_size=(3, 3),
            spatial_weight=0.3,
            initial_learning_rate=0.5,
            max_epochs=10,
            patience=5
        )
        
        som = GeoSOMVLRSOM(config)
        
        assert som.n_neurons == 9
        assert som.current_lr == 0.5
        assert len(som._grid_positions) == 9
    
    def test_geosom_training_with_missing_data(self):
        """Test GeoSOM training with realistic missing data."""
        # Create data with 70% missing values
        np.random.seed(42)
        n_samples = 50
        n_features = 10
        
        data = np.random.randn(n_samples, n_features)
        
        # Make 70% missing
        missing_mask = np.random.random((n_samples, n_features)) < 0.7
        data[missing_mask] = np.nan
        
        # Create random coordinates
        coordinates = np.random.uniform(-90, 90, (n_samples, 2))
        
        # Configure and train
        config = GeoSOMConfig(
            grid_size=(3, 3),
            spatial_weight=0.3,
            initial_learning_rate=0.5,
            max_epochs=5,  # Quick test
            patience=3
        )
        
        som = GeoSOMVLRSOM(config)
        result = som.train_batch(data, coordinates)
        
        # Check that training completed
        assert result.weights is not None
        assert result.weights.shape == (3, 3, n_features)
        assert len(result.quantization_errors) > 0
        
        # Check that QE is reasonable (not inf or nan)
        assert not np.isnan(result.quantization_errors[-1])
        assert not np.isinf(result.quantization_errors[-1])
    
    def test_adaptive_learning_rate(self):
        """Test VLRSOM adaptive learning rate behavior."""
        config = GeoSOMConfig(
            grid_size=(2, 2),
            initial_learning_rate=0.5,
            lr_increase_factor=1.1,
            lr_decrease_factor=0.85,
            max_epochs=20
        )
        
        som = GeoSOMVLRSOM(config)
        
        # Create simple data
        data = np.random.randn(20, 5)
        
        # Train and check learning rate adaptation
        result = som.train_batch(data)
        
        # Learning rate should have changed during training
        lr_history = som.training_history['learning_rates']
        assert len(lr_history) > 1
        assert not all(lr == lr_history[0] for lr in lr_history)
        
        # Final LR should be different from initial
        assert result.final_learning_rate != config.initial_learning_rate


class TestGeoSOMAnalyzer:
    """Test suite for GeoSOM analyzer integration."""
    
    def test_analyzer_initialization(self):
        """Test GeoSOM analyzer initialization."""
        from src.biodiversity_analysis.methods.som.analyzer import GeoSOMAnalyzer
        
        analyzer = GeoSOMAnalyzer()
        
        assert analyzer.method_name == 'geosom'
        assert analyzer.config is not None
        assert analyzer.config['architecture_config']['spatial_weight'] == 0.3
    
    def test_default_configuration_matches_spec(self):
        """Test that default configuration matches the specification."""
        from src.biodiversity_analysis.methods.som.analyzer import GeoSOMAnalyzer
        
        analyzer = GeoSOMAnalyzer()
        config = analyzer.config
        
        # Check key parameters from specification
        assert config['distance_config']['input_space'] == 'bray_curtis'
        assert config['distance_config']['min_valid_features'] == 2
        assert config['preprocessing_config']['transformation'] == 'log1p'
        assert config['preprocessing_config']['standardization'] == 'z_score_by_type'
        assert config['architecture_config']['spatial_weight'] == 0.3
        assert config['architecture_config']['initial_learning_rate'] == 0.5
        assert config['validation_config']['block_size'] == '750km'