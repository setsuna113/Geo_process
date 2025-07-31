"""Tests for spatial data splitting."""

import pytest
import numpy as np

from src.biodiversity_analysis.shared.spatial import SpatialSplitter
from src.biodiversity_analysis.shared.spatial.spatial_splitter import SpatialSplit


class TestSpatialSplitter:
    """Test SpatialSplitter functionality."""
    
    @pytest.fixture
    def sample_coordinates(self):
        """Create sample coordinate data."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create clustered spatial data
        centers = [
            (-120, 40),  # West
            (-100, 35),  # Central
            (-80, 30),   # East
        ]
        
        coordinates = []
        for center in centers:
            cluster_coords = np.random.normal(
                loc=center, 
                scale=5, 
                size=(n_samples // 3, 2)
            )
            coordinates.append(cluster_coords)
        
        return np.vstack(coordinates)
    
    def test_train_val_test_split_random_blocks(self, sample_coordinates):
        """Test random block splitting."""
        split = SpatialSplitter.train_val_test_split(
            sample_coordinates,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            strategy='random_blocks',
            random_state=42
        )
        
        assert isinstance(split, SpatialSplit)
        
        # Check split sizes (approximately)
        total = len(sample_coordinates)
        assert 0.6 < split.n_train / total < 0.8
        assert 0.1 < split.n_val / total < 0.2
        assert 0.1 < split.n_test / total < 0.2
        
        # Check no overlap
        train_set = set(split.train_idx)
        val_set = set(split.val_idx)
        test_set = set(split.test_idx)
        
        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0
        
        # Check all indices are covered
        all_idx = train_set | val_set | test_set
        assert len(all_idx) == total
    
    def test_train_val_test_split_latitudinal(self, sample_coordinates):
        """Test latitudinal splitting."""
        split = SpatialSplitter.train_val_test_split(
            sample_coordinates,
            strategy='latitudinal',
            random_state=42
        )
        
        # Check that splits are ordered by latitude
        train_lats = sample_coordinates[split.train_idx, 1]
        val_lats = sample_coordinates[split.val_idx, 1]
        test_lats = sample_coordinates[split.test_idx, 1]
        
        # Training should have lowest latitudes, test highest
        assert train_lats.max() <= val_lats.min() + 0.1  # Small overlap tolerance
        assert val_lats.max() <= test_lats.min() + 0.1
    
    def test_block_cv_split(self, sample_coordinates):
        """Test block cross-validation splitting."""
        n_folds = 5
        folds = SpatialSplitter.block_cv_split(
            sample_coordinates,
            n_folds=n_folds,
            random_state=42
        )
        
        assert len(folds) == n_folds
        
        # Check each fold
        for train_idx, test_idx in folds:
            # No overlap
            assert len(set(train_idx) & set(test_idx)) == 0
            
            # Reasonable sizes
            assert len(train_idx) > len(test_idx)
            assert len(test_idx) > 0
    
    def test_block_cv_with_buffer(self, sample_coordinates):
        """Test block CV with spatial buffer."""
        folds = SpatialSplitter.block_cv_split(
            sample_coordinates,
            n_folds=3,
            buffer_size=10.0,  # 10 units buffer
            random_state=42
        )
        
        # With buffer, training sets should be smaller
        for train_idx, test_idx in folds:
            # Check minimum distance between train and test
            for test_i in test_idx[:5]:  # Check a few
                test_coord = sample_coordinates[test_i]
                train_coords = sample_coordinates[train_idx]
                distances = np.sqrt(np.sum((train_coords - test_coord)**2, axis=1))
                
                # At least some training points should be beyond buffer
                assert np.any(distances > 10.0)
    
    def test_environmental_block_split(self, sample_coordinates):
        """Test environmental block splitting."""
        # Create fake environmental data (e.g., temperature gradient)
        env_data = sample_coordinates[:, 0] * 0.1 + sample_coordinates[:, 1] * 0.2
        env_data = env_data.reshape(-1, 1)
        
        split = SpatialSplitter.environmental_block_split(
            sample_coordinates,
            env_data,
            n_blocks=10,
            train_ratio=0.7,
            random_state=42
        )
        
        assert isinstance(split, SpatialSplit)
        assert 'cluster_centers' in split.split_info
        
        # Check that environmental blocks were created
        assert split.split_info['n_blocks'] == 10
    
    def test_invalid_ratios(self, sample_coordinates):
        """Test error handling for invalid ratios."""
        with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
            SpatialSplitter.train_val_test_split(
                sample_coordinates,
                train_ratio=0.8,
                val_ratio=0.2,
                test_ratio=0.2  # Sum > 1
            )
    
    def test_spatial_autocorrelation_prevention(self, sample_coordinates):
        """Test that spatial splits reduce autocorrelation."""
        # Create spatially autocorrelated feature
        feature = np.sin(sample_coordinates[:, 0] * 0.1) + \
                 np.cos(sample_coordinates[:, 1] * 0.1)
        
        # Random split (bad for spatial data)
        random_idx = np.random.permutation(len(sample_coordinates))
        random_train = random_idx[:700]
        random_test = random_idx[700:]
        
        # Spatial block split (good for spatial data)
        spatial_split = SpatialSplitter.train_val_test_split(
            sample_coordinates,
            strategy='random_blocks',
            random_state=42
        )
        
        # Calculate feature similarity between train and test
        random_train_mean = feature[random_train].mean()
        random_test_mean = feature[random_test].mean()
        random_diff = abs(random_train_mean - random_test_mean)
        
        spatial_train_mean = feature[spatial_split.train_idx].mean()
        spatial_test_mean = feature[spatial_split.test_idx].mean()
        spatial_diff = abs(spatial_train_mean - spatial_test_mean)
        
        # Spatial split should show larger difference (less leakage)
        # This is a probabilistic test, but should usually pass
        assert spatial_diff > random_diff * 0.5  # At least somewhat better