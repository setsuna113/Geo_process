"""Tests for SOM Analyzer integration with biodiversity framework."""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch
from src.biodiversity_analysis.methods.som.analyzer import SOMAnalyzer
from src.abstractions.types.biodiversity_types import BiodiversityData
from src.abstractions.interfaces.analyzer import AnalysisResult


class TestSOMAnalyzer:
    """Test suite for SOMAnalyzer."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            'som_analysis': {
                'grid_size': [3, 3],
                'distance_metric': 'euclidean',
                'epochs': 5,
                'learning_rate': 0.5,
                'neighborhood_function': 'gaussian'
            },
            'random_seed': 42
        }
    
    @pytest.fixture
    def sample_biodiversity_data(self):
        """Create sample biodiversity data."""
        np.random.seed(42)
        n_samples = 50
        n_species = 10
        
        # Create species abundance data
        features = np.random.poisson(3, (n_samples, n_species)).astype(float)
        
        # Create spatial coordinates
        coordinates = np.random.randn(n_samples, 2) * 10
        
        # Create BiodiversityData object
        biodiv_data = BiodiversityData(
            features=features,
            coordinates=coordinates,
            feature_names=[f'Species_{i}' for i in range(n_species)]
        )
        biodiv_data.species_names = [f'Species_{i}' for i in range(n_species)]
        biodiv_data.has_abundance = True
        biodiv_data.zero_inflated = True
        return biodiv_data
    
    @pytest.fixture
    def mock_dataset(self, sample_biodiversity_data):
        """Mock dataset object."""
        dataset = Mock()
        dataset.file_path = 'test_data.parquet'
        return dataset
    
    def test_analyzer_initialization(self, mock_config):
        """Test SOMAnalyzer initialization."""
        analyzer = SOMAnalyzer(mock_config)
        
        assert analyzer.method_name == 'som'
        assert analyzer.version == '2.0.0'
        assert analyzer.config == mock_config
        assert analyzer.som is None
    
    @patch('src.biodiversity_analysis.methods.som.analyzer.SOMAnalyzer.load_data')
    def test_analyze_basic(self, mock_load_data, mock_config, 
                          sample_biodiversity_data, mock_dataset):
        """Test basic analyze functionality."""
        mock_load_data.return_value = sample_biodiversity_data
        
        analyzer = SOMAnalyzer(mock_config)
        result = analyzer.analyze(
            mock_dataset,
            spatial_validation=False,
            save_results=False
        )
        
        # Check result structure
        assert isinstance(result, AnalysisResult)
        assert result.metadata.analysis_type == 'SOM'
        assert result.labels is not None
        assert result.labels.shape == (50, 2)  # Grid coordinates
        
        # Check metadata
        assert 'training_metrics' in result.statistics
        assert 'validation_metrics' in result.statistics
        assert 'biodiversity_metrics' in result.statistics
    
    @patch('src.biodiversity_analysis.methods.som.analyzer.SOMAnalyzer.load_data')
    def test_spatial_validation(self, mock_load_data, mock_config,
                              sample_biodiversity_data, mock_dataset):
        """Test spatial validation functionality."""
        mock_load_data.return_value = sample_biodiversity_data
        
        analyzer = SOMAnalyzer(mock_config)
        result = analyzer.analyze(
            mock_dataset,
            spatial_validation=True,
            spatial_strategy='random_blocks'
        )
        
        # Should have validation metrics for all splits
        val_metrics = result.statistics['validation_metrics']
        assert 'train_qe' in val_metrics
        assert 'val_qe' in val_metrics
        assert 'test_qe' in val_metrics
        
        # Metrics should be reasonable
        assert val_metrics['train_qe'] > 0
        assert val_metrics['val_qe'] > 0
        assert val_metrics['test_qe'] > 0
    
    def test_biodiversity_metrics(self, mock_config):
        """Test biodiversity-specific metrics calculation."""
        analyzer = SOMAnalyzer(mock_config)
        
        # Create simple test data
        features = np.array([
            [10, 0, 0],   # Only species 1
            [10, 0, 0],   # Only species 1
            [0, 10, 0],   # Only species 2
            [0, 10, 0],   # Only species 2
            [0, 0, 10],   # Only species 3
            [0, 0, 10]    # Only species 3
        ])
        
        biodiv_data = BiodiversityData(
            features=features,
            coordinates=np.random.randn(6, 2),
            feature_names=['sp1', 'sp2', 'sp3']
        )
        
        # Mock the necessary methods
        analyzer.load_data = Mock(return_value=biodiv_data)
        analyzer.preprocess_data = Mock(return_value=biodiv_data)
        
        dataset = Mock(file_path='test.parquet')
        result = analyzer.analyze(dataset, spatial_validation=False)
        
        # Check biodiversity metrics
        biodiv_metrics = result.statistics['biodiversity_metrics']
        assert 'species_coherence' in biodiv_metrics
        assert 'beta_diversity_preservation' in biodiv_metrics
        assert 'rare_species_representation' in biodiv_metrics
    
    def test_feature_importance(self, mock_config, sample_biodiversity_data):
        """Test feature importance calculation."""
        analyzer = SOMAnalyzer(mock_config)
        
        # Train a simple SOM
        from src.biodiversity_analysis.methods.som.som_core import BiodiversitySOM
        from src.abstractions.types.som_types import SOMConfig
        
        som_config = SOMConfig(grid_size=(3, 3), epochs=5)
        analyzer.som = BiodiversitySOM(som_config)
        analyzer.som.train(sample_biodiversity_data.features)
        
        # Calculate feature importance
        importance = analyzer._calculate_feature_importance()
        
        assert importance is not None
        assert len(importance) == sample_biodiversity_data.features.shape[1]
        assert np.allclose(np.sum(importance), 1.0)  # Should sum to 1
    
    def test_save_results(self, mock_config, sample_biodiversity_data):
        """Test result saving functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = SOMAnalyzer(mock_config)
            
            # Create mock result
            from src.abstractions.interfaces.analyzer import AnalysisMetadata
            
            metadata = AnalysisMetadata(
                analysis_type='SOM',
                input_shape=(10, 5),
                input_bands=['feat1', 'feat2', 'feat3', 'feat4', 'feat5'],
                parameters={'grid_size': [3, 3]},
                processing_time=10.5,
                timestamp='2024-01-01 00:00:00'
            )
            
            result = AnalysisResult(
                labels=np.random.randint(0, 3, (10, 2)),
                metadata=metadata,
                statistics={
                    'training_metrics': {'convergence_epoch': 5},
                    'data_info': {'n_samples': 10, 'n_features': 5}
                },
                additional_outputs={'feature_importance': np.random.rand(5)}
            )
            
            # Save results
            analyzer._save_results(result, tmpdir)
            
            # Check files were created
            assert os.path.exists(os.path.join(tmpdir, 'som_analysis_metadata.json'))
            assert os.path.exists(os.path.join(tmpdir, 'som_cluster_assignments.npy'))
            assert os.path.exists(os.path.join(tmpdir, 'feature_importance.npy'))
    
    def test_parameter_validation(self, mock_config):
        """Test parameter validation."""
        analyzer = SOMAnalyzer(mock_config)
        
        # Valid parameters
        is_valid, issues = analyzer.validate_parameters({'grid_size': [5, 5]})
        assert is_valid
        assert len(issues) == 0
        
        # Invalid parameters (missing required)
        is_valid, issues = analyzer.validate_parameters({'epochs': 100})
        assert not is_valid
        assert len(issues) > 0
        assert any('grid_size' in issue for issue in issues)
    
    def test_different_distance_metrics(self, mock_config, 
                                      sample_biodiversity_data, mock_dataset):
        """Test different distance metrics."""
        metrics = ['euclidean', 'manhattan', 'bray_curtis']
        
        for metric in metrics:
            config = mock_config.copy()
            config['som_analysis']['distance_metric'] = metric
            
            analyzer = SOMAnalyzer(config)
            analyzer.load_data = Mock(return_value=sample_biodiversity_data)
            analyzer.preprocess_data = Mock(return_value=sample_biodiversity_data)
            
            result = analyzer.analyze(mock_dataset, spatial_validation=False)
            
            # Should complete successfully
            assert result is not None
            assert result.statistics['validation_metrics']['train_qe'] > 0
    
    def test_interface_methods(self, mock_config):
        """Test ISOMAnalyzer interface methods."""
        analyzer = SOMAnalyzer(mock_config)
        
        # Create some test data
        data = np.random.randn(20, 5)
        
        # Train using interface method
        analyzer.train(data, epochs=5, learning_rate=0.5, 
                      neighborhood_radius=1.0)
        
        # Test other interface methods
        sample = data[0]
        bmu = analyzer.find_bmu(sample)
        assert isinstance(bmu, tuple)
        assert len(bmu) == 2
        
        weights = analyzer.get_weights()
        assert weights.shape[2] == 5  # n_features
        
        qe = analyzer.calculate_quantization_error(data)
        assert qe > 0
        
        te = analyzer.calculate_topographic_error(data)
        assert 0 <= te <= 1
    
    def test_error_handling(self, mock_config):
        """Test error handling."""
        analyzer = SOMAnalyzer(mock_config)
        
        # Try to use methods before training
        with pytest.raises(ValueError):
            analyzer.find_bmu(np.array([1, 2, 3]))
        
        with pytest.raises(ValueError):
            analyzer.get_weights()
        
        with pytest.raises(ValueError):
            analyzer.calculate_quantization_error(np.array([[1, 2, 3]]))
    
    @patch('src.biodiversity_analysis.methods.som.analyzer.SOMAnalyzer.load_data')
    def test_preprocessing_integration(self, mock_load_data, mock_config,
                                     sample_biodiversity_data, mock_dataset):
        """Test integration with preprocessing pipeline."""
        # Add zero-inflated features
        sample_biodiversity_data.features[:, :3] = 0  # Make first 3 features mostly zero
        sample_biodiversity_data.features[0, 0] = 1
        sample_biodiversity_data.features[5, 1] = 2
        sample_biodiversity_data.features[10, 2] = 1
        
        mock_load_data.return_value = sample_biodiversity_data
        
        analyzer = SOMAnalyzer(mock_config)
        
        # Spy on preprocessing
        original_preprocess = analyzer.preprocess_data
        preprocess_called = False
        
        def track_preprocess(*args, **kwargs):
            nonlocal preprocess_called
            preprocess_called = True
            return original_preprocess(*args, **kwargs)
        
        analyzer.preprocess_data = track_preprocess
        
        result = analyzer.analyze(mock_dataset, spatial_validation=False)
        
        # Preprocessing should have been called
        assert preprocess_called
        assert result is not None