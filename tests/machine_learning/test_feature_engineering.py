"""Tests for feature engineering components."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.machine_learning import (
    RichnessFeatureBuilder,
    SpatialFeatureBuilder,
    EcologicalFeatureBuilder,
    CompositeFeatureBuilder
)
from src.config import config


class TestRichnessFeatureBuilder:
    """Test RichnessFeatureBuilder."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample biodiversity data."""
        np.random.seed(42)
        n_samples = 100
        
        data = pd.DataFrame({
            'latitude': np.random.uniform(-60, 60, n_samples),
            'longitude': np.random.uniform(-180, 180, n_samples),
            'plants_richness': np.random.poisson(50, n_samples),
            'animals_richness': np.random.poisson(30, n_samples),
            'birds_richness': np.random.poisson(20, n_samples)
        })
        
        return data
    
    @pytest.fixture
    def builder(self):
        """Create feature builder instance."""
        return RichnessFeatureBuilder(config)
    
    def test_fit_transform(self, builder, sample_data):
        """Test basic fit_transform functionality."""
        features = builder.fit_transform(sample_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_data)
        
        # Check expected features exist
        expected_features = [
            'richness_total_richness',
            'richness_plants_ratio',
            'richness_animals_ratio',
            'richness_shannon_diversity',
            'richness_simpson_diversity'
        ]
        
        for feat in expected_features:
            assert feat in features.columns
    
    def test_richness_calculations(self, builder, sample_data):
        """Test specific richness calculations."""
        features = builder.fit_transform(sample_data)
        
        # Check total richness
        expected_total = (sample_data['plants_richness'] + 
                         sample_data['animals_richness'] + 
                         sample_data['birds_richness'])
        np.testing.assert_array_equal(
            features['richness_total_richness'].values,
            expected_total.values
        )
        
        # Check ratios
        plants_ratio = sample_data['plants_richness'] / expected_total
        np.testing.assert_array_almost_equal(
            features['richness_plants_ratio'].values,
            plants_ratio.fillna(0).values
        )
    
    def test_diversity_indices(self, builder, sample_data):
        """Test diversity index calculations."""
        features = builder.fit_transform(sample_data)
        
        # Shannon diversity should be non-negative
        assert (features['richness_shannon_diversity'] >= 0).all()
        
        # Simpson diversity should be between 0 and 1
        assert (features['richness_simpson_diversity'] >= 0).all()
        assert (features['richness_simpson_diversity'] <= 1).all()
    
    def test_log_transform(self, builder, sample_data):
        """Test log transformation of richness values."""
        features = builder.fit_transform(sample_data)
        
        # Check log-transformed features exist
        log_features = [col for col in features.columns if 'log_' in col]
        assert len(log_features) > 0
        
        # Log values should handle zeros properly (log1p)
        for col in log_features:
            assert not features[col].isna().any()
            assert (features[col] >= 0).all()
    
    def test_anomaly_detection(self, builder, sample_data):
        """Test richness anomaly detection."""
        features = builder.fit_transform(sample_data)
        
        # Check anomaly features exist
        anomaly_features = [col for col in features.columns if 'anomaly' in col]
        assert len(anomaly_features) > 0
        
        # Anomalies should be z-scores (roughly mean 0, std 1)
        for col in anomaly_features:
            assert abs(features[col].mean()) < 0.1
            assert 0.9 < features[col].std() < 1.1
    
    def test_empty_data(self, builder):
        """Test handling of empty data."""
        empty_data = pd.DataFrame()
        features = builder.fit_transform(empty_data)
        assert len(features) == 0
    
    def test_missing_richness_columns(self, builder):
        """Test handling when richness columns are missing."""
        data = pd.DataFrame({
            'latitude': [1, 2, 3],
            'longitude': [4, 5, 6]
        })
        
        features = builder.fit_transform(data)
        # Should return original data when no richness columns found
        assert len(features.columns) == 2


class TestSpatialFeatureBuilder:
    """Test SpatialFeatureBuilder."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample spatial data."""
        np.random.seed(42)
        n_samples = 100
        
        data = pd.DataFrame({
            'latitude': np.random.uniform(-60, 60, n_samples),
            'longitude': np.random.uniform(-180, 180, n_samples),
            'some_value': np.random.randn(n_samples)
        })
        
        return data
    
    @pytest.fixture
    def builder(self):
        """Create feature builder instance."""
        return SpatialFeatureBuilder(config)
    
    def test_fit_transform(self, builder, sample_data):
        """Test basic fit_transform functionality."""
        features = builder.fit_transform(sample_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_data)
        
        # Check expected features exist
        expected_features = [
            'spatial_distance_to_equator',
            'spatial_lat_squared',
            'spatial_lon_squared',
            'spatial_lat_lon_interaction',
            'spatial_hemisphere'
        ]
        
        for feat in expected_features:
            assert feat in features.columns
    
    def test_distance_to_equator(self, builder, sample_data):
        """Test distance to equator calculation."""
        features = builder.fit_transform(sample_data)
        
        # Distance should be absolute latitude
        expected_distance = np.abs(sample_data['latitude'])
        np.testing.assert_array_almost_equal(
            features['spatial_distance_to_equator'].values,
            expected_distance.values
        )
    
    def test_polynomial_features(self, builder, sample_data):
        """Test polynomial feature generation."""
        features = builder.fit_transform(sample_data)
        
        # Check squared terms
        np.testing.assert_array_almost_equal(
            features['spatial_lat_squared'].values,
            sample_data['latitude'].values ** 2
        )
        
        # Check interaction term
        np.testing.assert_array_almost_equal(
            features['spatial_lat_lon_interaction'].values,
            sample_data['latitude'].values * sample_data['longitude'].values
        )
    
    def test_spatial_bins(self, builder, sample_data):
        """Test spatial binning features."""
        features = builder.fit_transform(sample_data)
        
        # Check lat/lon bins exist
        assert 'spatial_lat_bin' in features.columns
        assert 'spatial_lon_bin' in features.columns
        
        # Bins should be categorical
        assert features['spatial_lat_bin'].dtype == 'object'
        assert features['spatial_lon_bin'].dtype == 'object'
    
    def test_hemisphere_encoding(self, builder, sample_data):
        """Test hemisphere encoding."""
        features = builder.fit_transform(sample_data)
        
        # Hemisphere should be binary
        assert features['spatial_hemisphere'].isin([0, 1]).all()
        
        # Northern hemisphere (lat > 0) should be 1
        northern_mask = sample_data['latitude'] > 0
        assert (features.loc[northern_mask, 'spatial_hemisphere'] == 1).all()
    
    def test_missing_coordinates(self, builder):
        """Test handling of missing coordinate columns."""
        data = pd.DataFrame({
            'value': [1, 2, 3]
        })
        
        features = builder.fit_transform(data)
        # Should return original data when coordinates missing
        assert len(features.columns) == 1


class TestEcologicalFeatureBuilder:
    """Test EcologicalFeatureBuilder."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return pd.DataFrame({
            'latitude': [10, 20, 30],
            'longitude': [40, 50, 60],
            'elevation': [100, 200, 300]
        })
    
    @pytest.fixture
    def builder(self):
        """Create feature builder instance."""
        return EcologicalFeatureBuilder(config)
    
    def test_default_behavior(self, builder, sample_data):
        """Test default behavior without data sources."""
        features = builder.fit_transform(sample_data)
        
        # Should pass through original data
        pd.testing.assert_frame_equal(features, sample_data)
    
    def test_add_data_source(self, builder, sample_data):
        """Test adding custom data source."""
        # Add mock data source
        climate_config = {
            'type': 'climate',
            'path': '/fake/path',
            'variables': ['temperature', 'precipitation']
        }
        
        builder.add_data_source('climate', climate_config)
        
        assert 'climate' in builder._data_sources
        assert builder._data_sources['climate'] == climate_config
    
    def test_extensibility(self, builder):
        """Test that builder is designed for extension."""
        # Should have methods for extension
        assert hasattr(builder, 'add_data_source')
        assert hasattr(builder, '_load_ecological_data')


class TestCompositeFeatureBuilder:
    """Test CompositeFeatureBuilder."""
    
    @pytest.fixture
    def sample_data(self):
        """Create comprehensive sample data."""
        np.random.seed(42)
        n_samples = 50
        
        data = pd.DataFrame({
            'latitude': np.random.uniform(-60, 60, n_samples),
            'longitude': np.random.uniform(-180, 180, n_samples),
            'plants_richness': np.random.poisson(50, n_samples),
            'animals_richness': np.random.poisson(30, n_samples)
        })
        
        return data
    
    @pytest.fixture
    def builder(self):
        """Create composite builder instance."""
        return CompositeFeatureBuilder(config)
    
    def test_auto_discovery(self, builder):
        """Test automatic discovery of feature builders."""
        # Should find registered builders
        assert len(builder._builders) > 0
        
        # Should include at least richness and spatial
        builder_types = [type(b).__name__ for b in builder._builders]
        assert 'RichnessFeatureBuilder' in builder_types
        assert 'SpatialFeatureBuilder' in builder_types
    
    def test_fit_transform(self, builder, sample_data):
        """Test composite feature generation."""
        features = builder.fit_transform(sample_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_data)
        
        # Should have more columns than original
        assert len(features.columns) > len(sample_data.columns)
        
        # Should include features from multiple builders
        richness_features = [col for col in features.columns if 'richness_' in col]
        spatial_features = [col for col in features.columns if 'spatial_' in col]
        
        assert len(richness_features) > 0
        assert len(spatial_features) > 0
    
    def test_feature_summary(self, builder, sample_data):
        """Test feature summary generation."""
        builder.fit_transform(sample_data)
        summary = builder.get_feature_summary()
        
        assert isinstance(summary, dict)
        assert 'total_features' in summary
        assert 'categories' in summary
        assert 'feature_names' in summary
        
        # Categories should match builders
        assert 'richness' in summary['categories']
        assert 'spatial' in summary['categories']
        
        # Each category should have count
        for category, info in summary['categories'].items():
            assert 'count' in info
            assert info['count'] > 0
    
    def test_transform_consistency(self, builder, sample_data):
        """Test that transform is consistent after fit."""
        # Fit on data
        features1 = builder.fit_transform(sample_data)
        
        # Transform should give same result
        features2 = builder.transform(sample_data)
        
        pd.testing.assert_frame_equal(features1, features2)
    
    def test_empty_data(self, builder):
        """Test handling of empty data."""
        empty_data = pd.DataFrame()
        features = builder.fit_transform(empty_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == 0
    
    def test_partial_data(self, builder):
        """Test with data that only works for some builders."""
        # Data with only coordinates (no richness)
        data = pd.DataFrame({
            'latitude': [10, 20, 30],
            'longitude': [40, 50, 60]
        })
        
        features = builder.fit_transform(data)
        
        # Should still generate spatial features
        spatial_features = [col for col in features.columns if 'spatial_' in col]
        assert len(spatial_features) > 0
        
        # But no richness features
        richness_features = [col for col in features.columns if 'richness_' in col]
        assert len(richness_features) == 0