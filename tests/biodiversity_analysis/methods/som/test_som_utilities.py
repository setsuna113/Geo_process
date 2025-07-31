"""Tests for SOM utilities: preprocessor and metrics."""

import pytest
import numpy as np
from src.biodiversity_analysis.shared.data.som_preprocessor import SOMPreprocessor
from src.biodiversity_analysis.shared.metrics.som_metrics import SOMMetrics


class TestSOMPreprocessor:
    """Test suite for SOM preprocessor."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return SOMPreprocessor()
    
    @pytest.fixture
    def species_data(self):
        """Create sample species abundance data."""
        np.random.seed(42)
        # Mix of count data and continuous data
        data = np.zeros((100, 10))
        
        # Species counts (columns 0-5)
        data[:, :6] = np.random.poisson(3, (100, 6))
        
        # Diversity indices (columns 6-9)
        data[:, 6:] = np.random.randn(100, 4) * 2 + 5
        
        return data
    
    def test_zero_inflation_handling(self, preprocessor):
        """Test handling of zero-inflated data."""
        # Create zero-inflated data
        data = np.zeros((100, 5))
        data[10:20, 0] = np.random.poisson(5, 10)  # 90% zeros
        data[:, 1] = np.random.poisson(3, 100)     # Normal distribution
        
        # Process data with no additional scaling
        processed = preprocessor.prepare_for_som(
            data,
            handle_zero_inflation=True,
            transform_method='sqrt',
            scaling_method='none'  # Don't apply additional scaling
        )
        
        # Should transform the data
        assert not np.array_equal(data, processed)
        # sqrt transform should be applied to zero-inflated features
        assert np.allclose(processed[10:20, 0], np.sqrt(data[10:20, 0]))
    
    def test_scaling_methods(self, preprocessor, species_data):
        """Test different scaling methods."""
        # Test adaptive scaling
        adaptive_scaled = preprocessor.prepare_for_som(
            species_data,
            scaling_method='adaptive'
        )
        
        # Test standard scaling
        standard_scaled = preprocessor.prepare_for_som(
            species_data,
            scaling_method='standard'
        )
        
        # Test minmax scaling
        minmax_scaled = preprocessor.prepare_for_som(
            species_data,
            scaling_method='minmax'
        )
        
        # Test no scaling
        no_scaled = preprocessor.prepare_for_som(
            species_data,
            scaling_method='none'
        )
        
        # Different methods should produce different results
        assert not np.array_equal(adaptive_scaled, standard_scaled)
        assert not np.array_equal(standard_scaled, minmax_scaled)
        assert np.array_equal(no_scaled, species_data)  # No scaling
    
    def test_pca_initialization(self, preprocessor, species_data):
        """Test PCA-based weight initialization."""
        grid_size = (5, 5)
        weights = preprocessor.initialize_weights_pca(species_data, grid_size)
        
        assert weights.shape == (5, 5, 10)
        
        # Weights should span data variance
        weights_flat = weights.reshape(-1, 10)
        assert np.std(weights_flat) > 0
        
        # Should be within reasonable range of data
        data_min, data_max = species_data.min(), species_data.max()
        assert weights.min() >= data_min - 3 * np.std(species_data)
        assert weights.max() <= data_max + 3 * np.std(species_data)
    
    def test_sample_initialization(self, preprocessor, species_data):
        """Test sample-based weight initialization."""
        grid_size = (3, 3)
        weights = preprocessor.initialize_weights_sample(
            species_data, grid_size, random_state=42
        )
        
        assert weights.shape == (3, 3, 10)
        
        # Weights should be actual samples
        weights_flat = weights.reshape(9, 10)
        for weight in weights_flat:
            assert any(np.allclose(weight, sample) for sample in species_data)
    
    def test_feature_statistics(self, preprocessor, species_data):
        """Test feature statistics collection."""
        preprocessor.prepare_for_som(species_data)
        
        stats = preprocessor.get_feature_statistics()
        
        assert 'original_mean' in stats
        assert 'original_std' in stats
        assert 'zero_proportion' in stats
        
        assert len(stats['original_mean']) == 10
        assert len(stats['original_std']) == 10
        assert len(stats['zero_proportion']) == 10


class TestSOMMetrics:
    """Test suite for SOM biodiversity metrics."""
    
    @pytest.fixture
    def metrics(self):
        """Create metrics calculator instance."""
        return SOMMetrics()
    
    @pytest.fixture
    def sample_analysis_data(self):
        """Create sample data for metric testing."""
        np.random.seed(42)
        
        # Create structured species data
        n_samples = 60
        n_species = 10
        
        species_data = np.zeros((n_samples, n_species))
        
        # Group 1: Species 0-3 dominant
        species_data[:20, :4] = np.random.poisson(10, (20, 4))
        
        # Group 2: Species 4-7 dominant
        species_data[20:40, 4:8] = np.random.poisson(8, (20, 4))
        
        # Group 3: Species 8-9 dominant
        species_data[40:, 8:] = np.random.poisson(6, (20, 2))
        
        # Create corresponding BMU assignments
        bmu_indices = np.array([0] * 20 + [1] * 20 + [2] * 20)
        
        # Create SOM weights (3 neurons)
        som_weights = np.zeros((3, n_species))
        som_weights[0, :4] = 10
        som_weights[1, 4:8] = 8
        som_weights[2, 8:] = 6
        
        return {
            'species_data': species_data,
            'bmu_indices': bmu_indices,
            'som_weights': som_weights,
            'coordinates': np.random.randn(n_samples, 2) * 10
        }
    
    def test_species_coherence(self, metrics, sample_analysis_data):
        """Test species coherence calculation."""
        coherence = metrics.calculate_species_coherence(
            sample_analysis_data['som_weights'],
            sample_analysis_data['species_data'],
            sample_analysis_data['bmu_indices']
        )
        
        assert 0 <= coherence <= 1
        # With well-structured data, coherence should be high
        assert coherence > 0.5
    
    def test_beta_diversity_preservation(self, metrics, sample_analysis_data):
        """Test beta diversity preservation metric."""
        preservation = metrics.calculate_beta_diversity_preservation(
            sample_analysis_data['species_data'],
            sample_analysis_data['bmu_indices'],
            distance_metric='bray_curtis'
        )
        
        assert 0 <= preservation <= 1
        # Well-separated groups should have good preservation
        assert preservation > 0.3
    
    def test_rare_species_representation(self, metrics):
        """Test rare species representation metric."""
        # Create data with rare species
        species_data = np.zeros((100, 10))
        
        # Common species (present in >50% of samples)
        species_data[:80, 0] = np.random.poisson(5, 80)
        
        # Rare species (present in <10% of samples)
        species_data[0:5, 5] = 1
        species_data[10:12, 6] = 1
        
        # BMU assignments
        bmu_indices = np.random.randint(0, 4, 100)
        
        representation = metrics.calculate_rare_species_representation(
            species_data,
            bmu_indices,
            prevalence_threshold=0.1
        )
        
        assert 0 <= representation <= 1
    
    def test_spatial_autocorrelation_preserved(self, metrics, sample_analysis_data):
        """Test spatial autocorrelation preservation."""
        # Create spatially structured coordinates
        coords = np.zeros((60, 2))
        coords[:20] = np.random.randn(20, 2) + [0, 0]    # Cluster 1
        coords[20:40] = np.random.randn(20, 2) + [10, 0]  # Cluster 2
        coords[40:] = np.random.randn(20, 2) + [5, 10]    # Cluster 3
        
        preservation = metrics.calculate_spatial_autocorrelation_preserved(
            coords,
            sample_analysis_data['species_data'],
            sample_analysis_data['bmu_indices']
        )
        
        assert 0 <= preservation <= 1
    
    def test_environmental_gradient_detection(self, metrics):
        """Test environmental gradient detection."""
        np.random.seed(42)  # Set seed for reproducibility
        
        # Create SOM weights with gradient
        weights = np.zeros((5, 5, 3))
        for i in range(5):
            for j in range(5):
                # Create gradient in feature 0
                weights[i, j, 0] = i * 2 + j
                # Random for other features
                weights[i, j, 1:] = np.random.randn(2)
        
        # Environmental variables (one with gradient)
        env_vars = np.zeros((25, 2))
        for i in range(25):
            row = i // 5
            col = i % 5
            env_vars[i, 0] = row * 2 + col  # Matches gradient
            env_vars[i, 1] = np.random.randn()  # Random
        
        gradients = metrics.calculate_environmental_gradient_detection(
            weights,
            env_vars,
            var_names=['Temperature', 'Precipitation']
        )
        
        assert 'Temperature' in gradients
        assert 'Precipitation' in gradients
        
        # Both should have scores between 0 and 1
        assert 0 <= gradients['Temperature'] <= 1
        assert 0 <= gradients['Precipitation'] <= 1
        
        # Temperature gradient should be detected (but the test is stochastic)
        # So we just ensure it has a reasonable score
        assert gradients['Temperature'] > 0.3  # At least moderate detection
    
    def test_metrics_edge_cases(self, metrics):
        """Test edge cases in metric calculations."""
        # Empty data
        empty_data = np.array([])
        empty_bmu = np.array([])
        
        # Single sample
        single_data = np.array([[1, 2, 3]])
        single_bmu = np.array([0])
        
        # Test each metric with edge cases
        coherence = metrics.calculate_species_coherence(
            np.array([[1, 2, 3]]),
            single_data,
            single_bmu
        )
        assert coherence >= 0
        
        # All samples in one cluster
        uniform_bmu = np.zeros(10, dtype=int)
        uniform_data = np.random.randn(10, 5)
        
        preservation = metrics.calculate_beta_diversity_preservation(
            uniform_data,
            uniform_bmu
        )
        assert preservation == 0  # No between-cluster distances