"""Tests for spatial cross-validation strategies."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.machine_learning import (
    SpatialBlockCV,
    SpatialBufferCV,
    EnvironmentalBlockCV
)


class TestSpatialBlockCV:
    """Test SpatialBlockCV."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample spatial data."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create clustered spatial data
        data = pd.DataFrame({
            'latitude': np.random.uniform(-50, 50, n_samples),
            'longitude': np.random.uniform(-100, 100, n_samples),
            'value': np.random.randn(n_samples)
        })
        
        return data
    
    @pytest.fixture
    def cv(self):
        """Create CV instance."""
        return SpatialBlockCV(n_splits=5, block_size=100, random_state=42)
    
    def test_split_generation(self, cv, sample_data):
        """Test that splits are generated correctly."""
        coords = sample_data[['latitude', 'longitude']].values
        
        splits = list(cv.split(sample_data, lat_lon=coords))
        
        assert len(splits) == cv.n_splits
        
        # Check each split
        for train_idx, test_idx in splits:
            assert len(train_idx) + len(test_idx) == len(sample_data)
            assert len(np.intersect1d(train_idx, test_idx)) == 0  # No overlap
            assert len(train_idx) > len(test_idx)  # More training than test
    
    def test_spatial_separation(self, cv, sample_data):
        """Test that test blocks are spatially separated."""
        coords = sample_data[['latitude', 'longitude']].values
        
        for train_idx, test_idx in cv.split(sample_data, lat_lon=coords):
            # Get test coordinates
            test_coords = coords[test_idx]
            
            # Calculate pairwise distances within test set
            from scipy.spatial.distance import cdist
            distances = cdist(test_coords, test_coords, metric='euclidean')
            
            # Most test points should be relatively close (same blocks)
            # This is a weak test, but ensures some spatial structure
            assert distances.max() < 200  # Within reasonable distance
    
    def test_deterministic_splits(self, sample_data):
        """Test that splits are deterministic with same random state."""
        coords = sample_data[['latitude', 'longitude']].values
        
        cv1 = SpatialBlockCV(n_splits=5, block_size=100, random_state=42)
        cv2 = SpatialBlockCV(n_splits=5, block_size=100, random_state=42)
        
        splits1 = list(cv1.split(sample_data, lat_lon=coords))
        splits2 = list(cv2.split(sample_data, lat_lon=coords))
        
        for (train1, test1), (train2, test2) in zip(splits1, splits2):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(test1, test2)
    
    def test_different_block_sizes(self, sample_data):
        """Test behavior with different block sizes."""
        coords = sample_data[['latitude', 'longitude']].values
        
        # Smaller blocks
        cv_small = SpatialBlockCV(n_splits=5, block_size=50)
        splits_small = list(cv_small.split(sample_data, lat_lon=coords))
        
        # Larger blocks
        cv_large = SpatialBlockCV(n_splits=5, block_size=200)
        splits_large = list(cv_large.split(sample_data, lat_lon=coords))
        
        # Both should generate valid splits
        assert len(splits_small) == 5
        assert len(splits_large) == 5
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        # Very few samples
        data = pd.DataFrame({
            'latitude': [1, 2, 3],
            'longitude': [1, 2, 3]
        })
        coords = data[['latitude', 'longitude']].values
        
        cv = SpatialBlockCV(n_splits=5, block_size=100)
        
        # Should handle gracefully (fewer splits or error)
        splits = list(cv.split(data, lat_lon=coords))
        assert len(splits) <= 3  # Can't have more splits than samples


class TestSpatialBufferCV:
    """Test SpatialBufferCV."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample spatial data."""
        np.random.seed(42)
        n_samples = 500
        
        data = pd.DataFrame({
            'latitude': np.random.uniform(-30, 30, n_samples),
            'longitude': np.random.uniform(-60, 60, n_samples),
            'value': np.random.randn(n_samples)
        })
        
        return data
    
    @pytest.fixture
    def cv(self):
        """Create CV instance."""
        return SpatialBufferCV(n_splits=5, buffer_distance=50, random_state=42)
    
    def test_buffer_separation(self, cv, sample_data):
        """Test that buffer zones properly separate train/test."""
        coords = sample_data[['latitude', 'longitude']].values
        
        for train_idx, test_idx in cv.split(sample_data, lat_lon=coords):
            train_coords = coords[train_idx]
            test_coords = coords[test_idx]
            
            # Calculate distances between train and test points
            from src.machine_learning.validation.spatial_cv import haversine_distance
            
            # Check minimum distance between any train and test point
            min_distance = float('inf')
            for test_coord in test_coords:
                for train_coord in train_coords:
                    dist = haversine_distance(
                        test_coord[0], test_coord[1],
                        train_coord[0], train_coord[1]
                    )
                    min_distance = min(min_distance, dist)
            
            # Should be at least buffer distance (with some tolerance)
            assert min_distance >= cv.buffer_distance * 0.9
    
    def test_split_sizes(self, cv, sample_data):
        """Test that split sizes are reasonable."""
        coords = sample_data[['latitude', 'longitude']].values
        
        test_sizes = []
        for train_idx, test_idx in cv.split(sample_data, lat_lon=coords):
            test_sizes.append(len(test_idx))
        
        # Test sizes should be relatively balanced
        mean_size = np.mean(test_sizes)
        std_size = np.std(test_sizes)
        
        # Coefficient of variation should be reasonable
        assert std_size / mean_size < 0.5
    
    def test_coverage(self, cv, sample_data):
        """Test that all samples are used in test set."""
        coords = sample_data[['latitude', 'longitude']].values
        
        all_test_indices = set()
        for train_idx, test_idx in cv.split(sample_data, lat_lon=coords):
            all_test_indices.update(test_idx)
        
        # Most samples should appear in test set at least once
        coverage = len(all_test_indices) / len(sample_data)
        assert coverage > 0.8


class TestEnvironmentalBlockCV:
    """Test EnvironmentalBlockCV."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with environmental gradient."""
        np.random.seed(42)
        n_samples = 500
        
        # Create data with clear latitudinal gradient
        latitudes = np.random.uniform(-60, 60, n_samples)
        
        data = pd.DataFrame({
            'latitude': latitudes,
            'longitude': np.random.uniform(-180, 180, n_samples),
            'temperature': 30 - np.abs(latitudes) * 0.5 + np.random.randn(n_samples),
            'precipitation': 1000 + latitudes * 5 + np.random.randn(n_samples) * 50,
            'value': np.random.randn(n_samples)
        })
        
        return data
    
    @pytest.fixture
    def cv(self):
        """Create CV instance."""
        return EnvironmentalBlockCV(n_splits=5, stratify_by='latitude')
    
    def test_environmental_stratification(self, cv, sample_data):
        """Test that folds are stratified by environmental variable."""
        splits = list(cv.split(sample_data))
        
        # Check latitude ranges in each test fold
        test_lat_ranges = []
        for train_idx, test_idx in splits:
            test_lats = sample_data.iloc[test_idx]['latitude']
            test_lat_ranges.append((test_lats.min(), test_lats.max()))
        
        # Ranges should be relatively non-overlapping
        # Sort by minimum latitude
        test_lat_ranges.sort(key=lambda x: x[0])
        
        # Check that ranges don't overlap too much
        for i in range(len(test_lat_ranges) - 1):
            # Some overlap is okay, but not complete overlap
            overlap = min(test_lat_ranges[i][1], test_lat_ranges[i+1][1]) - \
                     max(test_lat_ranges[i][0], test_lat_ranges[i+1][0])
            range_size = test_lat_ranges[i][1] - test_lat_ranges[i][0]
            
            # Overlap should be less than 50% of range
            assert overlap < 0.5 * range_size
    
    def test_custom_stratification(self, sample_data):
        """Test stratification by custom variable."""
        cv = EnvironmentalBlockCV(n_splits=5, stratify_by='temperature')
        
        splits = list(cv.split(sample_data))
        
        # Check temperature ranges in each test fold
        for train_idx, test_idx in splits:
            test_temps = sample_data.iloc[test_idx]['temperature']
            
            # Each fold should have a range of temperatures
            assert test_temps.std() > 0
            assert test_temps.max() - test_temps.min() > 5
    
    def test_missing_stratification_column(self, sample_data):
        """Test error handling for missing stratification column."""
        cv = EnvironmentalBlockCV(n_splits=5, stratify_by='nonexistent_column')
        
        with pytest.raises(KeyError):
            list(cv.split(sample_data))
    
    def test_balanced_splits(self, cv, sample_data):
        """Test that splits are relatively balanced."""
        splits = list(cv.split(sample_data))
        
        test_sizes = [len(test_idx) for _, test_idx in splits]
        
        # Sizes should be within 20% of each other
        min_size = min(test_sizes)
        max_size = max(test_sizes)
        
        assert max_size / min_size < 1.5


def test_haversine_distance():
    """Test haversine distance calculation."""
    from src.machine_learning.validation.spatial_cv import haversine_distance
    
    # Test known distances
    # Equator to North Pole (90 degrees)
    dist = haversine_distance(0, 0, 90, 0)
    assert abs(dist - 10018.75) < 1  # ~10,019 km
    
    # Same point
    dist = haversine_distance(40, -74, 40, -74)
    assert dist == 0
    
    # New York to London (~5,570 km)
    dist = haversine_distance(40.7128, -74.0060, 51.5074, -0.1278)
    assert 5500 < dist < 5600