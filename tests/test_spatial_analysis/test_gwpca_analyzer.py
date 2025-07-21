# tests/test_spatial_analysis/test_gwpca_analyzer.py
"""Tests for GWPCA analyzer."""

import pytest
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch, MagicMock

from src.spatial_analysis.gwpca.gwpca_analyzer import GWPCAAnalyzer
from src.spatial_analysis.gwpca.bandwidth_selector import BandwidthSelector
from src.spatial_analysis.gwpca.local_stats_mapper import LocalStatsMapper


class TestGWPCAAnalyzerUnit:
    """Unit tests for GWPCA analyzer."""
    
    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.get.return_value = {
            'gwpca': {
                'bandwidth_method': 'cv',
                'adaptive': True,
                'n_components': 2,
                'kernel': 'bisquare',
                'block_size_km': 10,
                'pixel_size_km': 1.8,
                'use_block_aggregation': True
            }
        }
        return config
    
    @pytest.fixture
    def analyzer(self, mock_config):
        with patch('src.spatial_analysis.base_analyzer.ArrayConverter'):
            return GWPCAAnalyzer(mock_config)
    
    def test_aggregate_to_blocks(self, analyzer):
        """Test block aggregation."""
        # Create 6x6 pixel data
        data = np.random.rand(2, 6, 6)  # 2 bands
        metadata = {
            'original_shape': (2, 6, 6),
            'bands': ['P', 'A']
        }
        params = {
            'block_size_km': 5.4,  # 3 pixels at 1.8km each
            'aggregation_method': 'mean'
        }
        
        block_data, block_meta = analyzer._aggregate_to_blocks(data, metadata, params)
        
        # Should create 2x2 blocks
        assert block_data.shape == (2, 2, 2)
        assert block_meta['n_blocks'] == 4
        assert block_meta['block_size_pixels'] == 3
        assert np.all(block_meta['block_counts'] == 9)  # 3x3 pixels per block
    
    @patch('src.spatial_analysis.gwpca.gwpca_analyzer.Sel_BW')
    def test_bandwidth_selection(self, mock_sel_bw, analyzer):
        """Test bandwidth selection."""
        # Mock bandwidth selector
        mock_selector = Mock()
        mock_selector.search.return_value = 30
        mock_sel_bw.return_value = mock_selector
        
        X = np.random.rand(25, 3)  # 25 points, 3 features
        coords = np.random.rand(25, 2)
        params = {'adaptive': True, 'bandwidth_method': 'cv', 
                 'kernel': 'bisquare', 'use_block_aggregation': True}
        
        bw = analyzer._select_bandwidth(X, coords, params)
        
        assert bw == 30
        mock_selector.search.assert_called_once()
    
    def test_compute_gwpca_small(self, analyzer):
        """Test GWPCA computation on small data."""
        # 3x3 grid, 2 features
        X = np.random.rand(9, 2)
        coords = np.array([[i, j] for i in range(3) for j in range(3)])
        params = {
            'bandwidth': 3,  # 3 nearest neighbors
            'adaptive': True,
            'kernel': 'bisquare',
            'n_components': 1
        }
        
        results = analyzer._compute_gwpca(X, coords, params)
        
        assert results['local_r2'].shape == (9,)
        assert results['local_eigenvalues'].shape == (9, 1)
        assert results['local_loadings'].shape == (9, 2, 1)
        assert np.all(results['local_r2'] >= 0)
        assert np.all(results['local_r2'] <= 1)


class TestBandwidthSelector:
    """Tests for bandwidth selector."""
    
    @pytest.fixture
    def selector(self):
        coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        return BandwidthSelector(coords, kernel='bisquare')
    
    def test_golden_section_search(self, selector):
        """Test golden section optimization."""
        # Simple quadratic function
        def objective(x):
            return (x - 5) ** 2
        
        optimal = selector.golden_section_search(objective, 0, 10, tolerance=0.01)
        assert abs(optimal - 5) < 0.1
    
    def test_kernel_functions(self, selector):
        """Test different kernel functions."""
        distances = np.array([0, 0.5, 1.0, 1.5, 2.0])
        bandwidth = 1.0
        
        # Bisquare kernel
        weights = selector._kernel_function(distances, bandwidth)
        assert weights[0] == 1.0  # Distance 0
        assert weights[2] == 0.0  # Distance = bandwidth
        assert weights[4] == 0.0  # Distance > bandwidth
    
    def test_suggest_bandwidth_range(self, selector):
        """Test bandwidth range suggestion."""
        lower, upper = selector.suggest_bandwidth_range(adaptive=True)
        assert lower > 0
        assert upper > lower
        assert upper <= selector.n_points


class TestLocalStatsMapper:
    """Tests for local stats mapper."""
    
    @pytest.fixture
    def mapper(self):
        return LocalStatsMapper()
    
    @pytest.fixture
    def mock_result_blocks(self):
        from src.spatial_analysis.base_analyzer import AnalysisResult, AnalysisMetadata
        
        # Create block-based result
        return AnalysisResult(
            labels=np.random.rand(4),  # 2x2 blocks
            metadata=AnalysisMetadata(
                analysis_type='GWPCA',
                input_shape=(2, 6, 6),
                input_bands=['P', 'A'],
                parameters={
                    'block_size_km': 10,
                    'use_block_aggregation': True,
                    'kernel': 'bisquare',
                    'adaptive': True
                },
                processing_time=0.1,
                timestamp='2025-01-01'
            ),
            statistics={
                'local_r2_mean': 0.7,
                'local_r2_std': 0.1,
                'local_r2_min': 0.5,
                'local_r2_max': 0.9,
                'bandwidth': 2,
                'analysis_scale': '10km blocks',
                'n_blocks_analyzed': 4,
                'data_reduction_factor': 9,
                'spatial_heterogeneity': 0.15
            },
            spatial_output=xr.DataArray(
                np.array([[0.7, 0.8], [0.6, 0.9]]),
                dims=['y', 'x']
            ),
            additional_outputs={
                'block_pixel_counts': xr.DataArray(
                    np.array([[9, 9], [9, 9]]),
                    dims=['y', 'x']
                )
            }
        )
    
    def test_plot_local_r2_map(self, mapper, mock_result_blocks):
        """Test R² map plotting."""
        fig = mapper.plot_local_r2_map(mock_result_blocks, show_blocks=True)
        assert fig is not None
        plt.close(fig)
    
    def test_plot_block_summary(self, mapper, mock_result_blocks):
        """Test block analysis summary."""
        fig = mapper.plot_block_analysis_summary(mock_result_blocks)
        assert fig is not None
        assert len(fig.axes) >= 5  # Multiple subplots
        plt.close(fig)


class TestGWPCAIntegration:
    """Integration tests for GWPCA workflow."""
    
    @pytest.fixture
    def config(self):
        from src.config.config import Config
        config = Config()
        config.config = {
            'spatial_analysis': {
                'gwpca': {
                    'block_size_km': 3.6,  # 2 pixels
                    'pixel_size_km': 1.8,
                    'use_block_aggregation': True,
                    'n_components': 2,
                    'adaptive': True,
                    'iterations': 10
                },
                'normalize_data': False,
                'save_results': False
            },
            'processors': {
                'data_preparation': {}
            }
        }
        return config
    
    def test_gwpca_full_workflow(self, config):
        """Test complete GWPCA workflow with blocks."""
        analyzer = GWPCAAnalyzer(config)
        
        # Create small test data (6x6 grid)
        np.random.seed(42)
        data = xr.Dataset({
            'P': xr.DataArray(
                np.random.rand(6, 6) + np.arange(6).reshape(6, 1) * 0.1,
                dims=['y', 'x']
            ),
            'A': xr.DataArray(
                np.random.rand(6, 6) + np.arange(6).reshape(1, 6) * 0.1,
                dims=['y', 'x']
            ),
            'F': xr.DataArray(
                np.random.rand(6, 6),
                dims=['y', 'x']
            )
        })
        
        # Run analysis
        result = analyzer.analyze(
            data,
            bandwidth=2,  # Fixed bandwidth for testing
            use_block_aggregation=True,
            block_size_km=3.6
        )
        
        # Verify results
        assert result.metadata.analysis_type == 'GWPCA'
        assert result.labels.shape == (9,)  # 3x3 blocks from 6x6 pixels
        assert 'local_r2_mean' in result.statistics
        assert result.statistics['n_blocks_analyzed'] == 9
        assert 0 <= result.statistics['local_r2_mean'] <= 1
        
        # Test visualization
        mapper = LocalStatsMapper()
        fig = mapper.plot_local_r2_map(result)
        assert fig is not None
        plt.close(fig)