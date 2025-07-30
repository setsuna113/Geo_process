"""Tests for missing value imputation strategies."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.machine_learning import SpatialKNNImputer


class TestSpatialKNNImputer:
    """Test SpatialKNNImputer."""
    
    @pytest.fixture
    def sample_data_with_missing(self):
        """Create sample data with missing values."""
        np.random.seed(42)
        
        # Create spatial data with clear pattern
        n_samples = 100
        lat = np.linspace(-30, 30, n_samples)
        lon = np.linspace(-60, 60, n_samples)
        
        # Values correlate with latitude (higher near equator)
        value1 = 50 - np.abs(lat) + np.random.randn(n_samples) * 2
        value2 = 30 - np.abs(lat) * 0.5 + np.random.randn(n_samples) * 2
        
        data = pd.DataFrame({
            'latitude': lat,
            'longitude': lon,
            'value1': value1,
            'value2': value2
        })
        
        # Add missing values with spatial pattern
        # More missing at high latitudes
        missing_prob = np.abs(lat) / 60
        
        mask1 = np.random.random(n_samples) < missing_prob * 0.3
        mask2 = np.random.random(n_samples) < missing_prob * 0.2
        
        data.loc[mask1, 'value1'] = np.nan
        data.loc[mask2, 'value2'] = np.nan
        
        return data
    
    @pytest.fixture
    def imputer(self):
        """Create imputer instance."""
        return SpatialKNNImputer(n_neighbors=5, spatial_weight=0.7)
    
    def test_fit_transform(self, imputer, sample_data_with_missing):
        """Test basic imputation functionality."""
        original_missing = sample_data_with_missing.isna().sum().sum()
        assert original_missing > 0  # Ensure we have missing values
        
        # Impute
        imputed_data = imputer.fit_transform(sample_data_with_missing)
        
        # Check no missing values remain
        assert imputed_data.isna().sum().sum() == 0
        
        # Check shape preserved
        assert imputed_data.shape == sample_data_with_missing.shape
        
        # Check column names preserved
        assert list(imputed_data.columns) == list(sample_data_with_missing.columns)
    
    def test_spatial_consistency(self, imputer, sample_data_with_missing):
        """Test that imputed values respect spatial patterns."""
        imputed_data = imputer.fit_transform(sample_data_with_missing)
        
        # Get indices where values were imputed
        imputed_mask = sample_data_with_missing['value1'].isna()
        imputed_indices = imputed_mask[imputed_mask].index
        
        # Check that imputed values are reasonable
        for idx in imputed_indices:
            imputed_val = imputed_data.loc[idx, 'value1']
            lat = imputed_data.loc[idx, 'latitude']
            
            # Expected value based on latitude pattern
            expected_range = (40 - np.abs(lat), 60 - np.abs(lat))
            
            # Imputed value should be in reasonable range
            assert expected_range[0] < imputed_val < expected_range[1]
    
    def test_spatial_weighting(self, sample_data_with_missing):
        """Test effect of spatial weighting parameter."""
        # High spatial weight - prioritize nearby points
        imputer_spatial = SpatialKNNImputer(n_neighbors=5, spatial_weight=0.9)
        imputed_spatial = imputer_spatial.fit_transform(sample_data_with_missing)
        
        # Low spatial weight - prioritize feature similarity
        imputer_feature = SpatialKNNImputer(n_neighbors=5, spatial_weight=0.1)
        imputed_feature = imputer_feature.fit_transform(sample_data_with_missing)
        
        # Results should be different
        assert not imputed_spatial.equals(imputed_feature)
        
        # Get a missing value location
        missing_idx = sample_data_with_missing['value1'].isna().idxmax()
        
        # Spatial imputation should be smoother (closer to local average)
        # This is a weak test, but ensures different behavior
        assert imputed_spatial.loc[missing_idx, 'value1'] != \
               imputed_feature.loc[missing_idx, 'value1']
    
    def test_no_missing_values(self, imputer):
        """Test behavior when no missing values present."""
        data = pd.DataFrame({
            'latitude': [1, 2, 3],
            'longitude': [4, 5, 6],
            'value': [7, 8, 9]
        })
        
        imputed = imputer.fit_transform(data)
        
        # Should return unchanged
        pd.testing.assert_frame_equal(imputed, data)
    
    def test_all_missing_column(self, imputer):
        """Test handling of column with all missing values."""
        data = pd.DataFrame({
            'latitude': [1, 2, 3, 4, 5],
            'longitude': [1, 2, 3, 4, 5],
            'value1': [1, 2, 3, 4, 5],
            'value2': [np.nan, np.nan, np.nan, np.nan, np.nan]
        })
        
        # Should handle gracefully
        imputed = imputer.fit_transform(data)
        
        # All missing column might be filled with mean or median
        assert not imputed['value2'].isna().any()
    
    def test_single_row(self, imputer):
        """Test handling of single row data."""
        data = pd.DataFrame({
            'latitude': [1],
            'longitude': [2],
            'value': [np.nan]
        })
        
        # Should handle edge case
        imputed = imputer.fit_transform(data)
        
        # Might use global statistics or zero
        assert not pd.isna(imputed.loc[0, 'value'])
    
    def test_transform_new_data(self, imputer, sample_data_with_missing):
        """Test transform on new data after fitting."""
        # Fit on training data
        train_data = sample_data_with_missing.iloc[:80]
        imputer.fit(train_data)
        
        # Transform new data
        test_data = sample_data_with_missing.iloc[80:].copy()
        imputed_test = imputer.transform(test_data)
        
        # Should handle new data
        assert imputed_test.isna().sum().sum() == 0
        assert len(imputed_test) == len(test_data)
    
    def test_preserve_non_numeric(self, imputer):
        """Test that non-numeric columns are preserved."""
        data = pd.DataFrame({
            'latitude': [1, 2, 3],
            'longitude': [4, 5, 6],
            'value': [7, np.nan, 9],
            'category': ['A', 'B', 'C']
        })
        
        imputed = imputer.fit_transform(data)
        
        # Category column should be unchanged
        assert list(imputed['category']) == ['A', 'B', 'C']
        
        # Numeric column should be imputed
        assert not imputed['value'].isna().any()
    
    def test_different_missing_indicators(self, imputer):
        """Test handling of different missing value indicators."""
        data = pd.DataFrame({
            'latitude': [1, 2, 3],
            'longitude': [4, 5, 6],
            'value': [7, -999, 9]  # -999 as missing indicator
        })
        
        # Create imputer with custom missing value
        custom_imputer = SpatialKNNImputer(
            n_neighbors=3, 
            spatial_weight=0.5,
            missing_values=-999
        )
        
        imputed = custom_imputer.fit_transform(data)
        
        # -999 should be replaced
        assert -999 not in imputed['value'].values
        assert imputed.loc[1, 'value'] != -999
    
    def test_haversine_distance_calculation(self, imputer):
        """Test that imputer uses proper geographic distance."""
        # Create data with known geographic pattern
        data = pd.DataFrame({
            'latitude': [0, 0, 45, 45],  # Equator and mid-latitude
            'longitude': [0, 90, 0, 90],  # 90 degrees apart
            'value': [10, 20, np.nan, np.nan]
        })
        
        imputed = imputer.fit_transform(data)
        
        # Points at same latitude should influence each other more
        # due to shorter distance at higher latitudes
        assert imputed.loc[2, 'value'] != imputed.loc[3, 'value']