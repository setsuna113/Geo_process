# tests/test_spatial_analysis/test_som_analyzer.py
"""Tests for SOM analyzer."""

import pytest
import numpy as np
import xarray as xr
from unittest.mock import Mock, patch, MagicMock

from src.spatial_analysis.som.som_trainer import SOMAnalyzer
from src.spatial_analysis.som.som_visualizer import SOMVisualizer
from src.spatial_analysis.som.som_reporter import SOMReporter


class TestSOMAnalyzerUnit:
    """Unit tests for SOM analyzer."""
    
    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.get.return_value = {
            'som': {
                'grid_size': [3, 3],
                'iterations': 100,
                'sigma': 1.0,
                'learning_rate': 0.5,
                'neighborhood_function': 'gaussian'
            }
        }
        return config
    
    @pytest.fixture
    def analyzer(self, mock_config):
        with patch('src.spatial_analysis.som.som_trainer.ArrayConverter'), \
             patch('src.spatial_analysis.som.som_trainer.DataNormalizer'):
            return SOMAnalyzer(mock_config)
    
    def test_get_default_parameters(self, analyzer):
        """Test default parameters."""
        params = analyzer.get_default_parameters()
        assert params['grid_size'] == [3, 3]
        assert params['iterations'] == 100
        assert params['random_seed'] == 42
    
    def test_validate_parameters(self, analyzer):
        """Test parameter validation."""
        # Valid parameters
        valid_params = {
            'grid_size': [3, 3],
            'iterations': 100,
            'sigma': 1.0,
            'learning_rate': 0.5,
            'neighborhood_function': 'gaussian'
        }
        is_valid, issues = analyzer.validate_parameters(valid_params)
        assert is_valid
        assert len(issues) == 0
        
        # Invalid grid size
        invalid_params = valid_params.copy()
        invalid_params['grid_size'] = [0, 3]
        is_valid, issues = analyzer.validate_parameters(invalid_params)
        assert not is_valid
        assert any('grid_size' in issue for issue in issues)
        
        # Invalid learning rate
        invalid_params = valid_params.copy()
        invalid_params['learning_rate'] = 1.5
        is_valid, issues = analyzer.validate_parameters(invalid_params)
        assert not is_valid
        assert any('learning_rate' in issue for issue in issues)
    
    @patch('src.spatial_analysis.som.som_trainer.MiniSom')
    def test_analyze_basic(self, mock_minisom, analyzer):
        """Test basic SOM analysis."""
        # Setup mock SOM
        mock_som = Mock()
        mock_som.winner.side_effect = lambda x: (0, 0)  # All samples to same cluster
        mock_som.quantization_error.return_value = 0.1
        mock_som.topographic_error.return_value = 0.05
        mock_som.get_weights.return_value = np.random.rand(3, 3, 3)
        mock_som.distance_map.return_value = np.random.rand(3, 3)
        mock_minisom.return_value = mock_som
        
        # Mock data preparation
        test_data = np.random.rand(10, 3)  # 10 samples, 3 features
        with patch.object(analyzer, 'prepare_data') as mock_prepare:
            mock_prepare.return_value = (test_data, {
                'original_shape': (3, 2, 5),
                'bands': ['P', 'A', 'F'],
                'normalized': False
            })
            
            with patch.object(analyzer, 'restore_spatial_structure') as mock_restore:
                mock_restore.return_value = xr.DataArray(np.zeros((2, 5)))
                
                # Run analysis
                result = analyzer.analyze(np.random.rand(3, 2, 5), grid_size=[3, 3])
                
                assert result.metadata.analysis_type == 'SOM'
                assert result.labels.shape == (10,)
                assert result.statistics['n_clusters'] == 9  # 3x3 grid
                assert result.statistics['quantization_error'] == 0.1
    
    def test_calculate_statistics(self, analyzer):
        """Test statistics calculation."""
        data = np.random.rand(20, 3)
        labels = np.array([0, 0, 1, 1, 2] * 4)
        
        # Create mock SOM
        mock_som = Mock()
        mock_som.quantization_error.return_value = 0.15
        mock_som.topographic_error.return_value = 0.08
        
        stats = analyzer._calculate_statistics(data, labels, mock_som, {'grid_size': [3, 3]})
        
        assert stats['n_clusters'] == 3
        assert stats['quantization_error'] == 0.15
        assert stats['topographic_error'] == 0.08
        assert 0 in stats['cluster_statistics']
        assert stats['cluster_statistics'][0]['count'] == 8


class TestSOMVisualizerUnit:
    """Unit tests for SOM visualizer."""
    
    @pytest.fixture
    def visualizer(self):
        return SOMVisualizer(figsize=(8, 6))
    
    @pytest.fixture
    def mock_result(self):
        from src.spatial_analysis.base_analyzer import AnalysisResult, AnalysisMetadata
        
        return AnalysisResult(
            labels=np.array([0, 1, 0, 1, 2, 2, 1, 0, 2]),
            metadata=AnalysisMetadata(
                analysis_type='SOM',
                input_shape=(3, 3, 3),
                input_bands=['P', 'A', 'F'],
                parameters={'grid_size': [3, 3]},
                processing_time=0.1,
                timestamp='2025-01-01'
            ),
            statistics={
                'n_clusters': 3,
                'quantization_error': 0.1,
                'topographic_error': 0.05,
                'empty_neurons': 6,
                'cluster_balance': 0.3,
                'cluster_statistics': {
                    0: {'count': 3, 'percentage': 33.3, 'mean': [0.5, 0.5, 0.5]},
                    1: {'count': 3, 'percentage': 33.3, 'mean': [0.6, 0.4, 0.5]},
                    2: {'count': 3, 'percentage': 33.3, 'mean': [0.4, 0.6, 0.5]}
                }
            },
            spatial_output=xr.DataArray(
                np.array([[0, 1, 0], [1, 2, 2], [1, 0, 2]]),
                dims=['y', 'x']
            ),
            additional_outputs={
                'distance_map': np.random.rand(3, 3),
                'activation_map': np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
                'component_planes': {
                    'P': np.random.rand(3, 3),
                    'A': np.random.rand(3, 3),
                    'F': np.random.rand(3, 3)
                }
            }
        )
    
    def test_plot_cluster_map(self, visualizer, mock_result):
        """Test cluster map plotting."""
        fig = visualizer.plot_cluster_map(mock_result)
        assert fig is not None
        assert len(fig.axes) == 2  # Main plot + colorbar
        plt.close(fig)
    
    def test_plot_umatrix(self, visualizer, mock_result):
        """Test U-matrix plotting."""
        fig = visualizer.plot_umatrix(mock_result)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_component_planes(self, visualizer, mock_result):
        """Test component planes plotting."""
        fig = visualizer.plot_component_planes(mock_result)
        assert fig is not None
        assert len(fig.axes) >= 3  # One for each component
        plt.close(fig)


class TestSOMReporter:
    """Tests for SOM reporter."""
    
    @pytest.fixture
    def reporter(self):
        return SOMReporter()
    
    def test_generate_cluster_summary(self, reporter, mock_result):
        """Test cluster summary generation."""
        df = reporter.generate_cluster_summary(mock_result)
        
        assert len(df) == 3  # 3 clusters
        assert 'cluster_id' in df.columns
        assert 'pixel_count' in df.columns
        assert 'percentage' in df.columns
        assert df['pixel_count'].sum() == 9  # Total pixels


class TestSOMIntegration:
    """Integration tests for SOM workflow."""
    
    @pytest.fixture
    def config(self):
        from src.core.config import Config
        config = Config()
        config.config = {
            'spatial_analysis': {
                'som': {
                    'grid_size': [2, 2],
                    'iterations': 10,  # Fast for testing
                    'sigma': 1.0,
                    'learning_rate': 0.5
                },
                'normalize_data': False,
                'save_results': False
            },
            'processors': {
                'data_preparation': {}
            }
        }
        return config
    
    def test_som_full_workflow(self, config):
        """Test complete SOM workflow."""
        analyzer = SOMAnalyzer(config)
        
        # Create small test data
        np.random.seed(42)
        data = xr.Dataset({
            'P': xr.DataArray(np.random.rand(4, 4), dims=['y', 'x']),
            'A': xr.DataArray(np.random.rand(4, 4), dims=['y', 'x']),
            'F': xr.DataArray(np.random.rand(4, 4), dims=['y', 'x'])
        })
        
        # Run analysis
        result = analyzer.analyze(data, iterations=10)
        
        # Verify results
        assert result.metadata.analysis_type == 'SOM'
        assert result.labels.shape == (16,)  # 4x4 pixels
        assert 'quantization_error' in result.statistics
        assert 'component_planes' in result.additional_outputs
        
        # Test visualization
        visualizer = SOMVisualizer()
        fig = visualizer.plot_cluster_map(result)
        assert fig is not None
        plt.close(fig)
        
        # Test reporting
        reporter = SOMReporter()
        summary = reporter.generate_cluster_summary(result)
        assert len(summary) <= 4  # Maximum 2x2 = 4 clusters