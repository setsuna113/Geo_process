# tests/test_spatial_analysis/test_maxp_analyzer.py
"""Tests for Max-p regions analyzer."""

import pytest
import numpy as np
import xarray as xr
from unittest.mock import Mock, patch, MagicMock
import libpysal

from src.spatial_analysis.maxp_regions.region_optimizer import MaxPAnalyzer
from src.spatial_analysis.maxp_regions.contiguity_builder import ContiguityBuilder
from src.spatial_analysis.maxp_regions.region_reporter import RegionReporter


class TestMaxPAnalyzerUnit:
    """Unit tests for Max-p analyzer."""
    
    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.get.return_value = {
            'maxp': {
                'pixel_size_km': 1.8,
                'min_area_km2': 100,
                'ecological_scale': 'landscape',
                'contiguity': 'queen',
                'min_regions': 2,
                'iterations': 5
            }
        }
        return config
    
    @pytest.fixture
    def analyzer(self, mock_config):
        with patch('src.spatial_analysis.base_analyzer.ArrayConverter'):
            return MaxPAnalyzer(mock_config)
    
    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.pixel_area_km2 == 1.8 ** 2
        assert analyzer.default_min_area_km2 == 100
        assert 'landscape' in analyzer.ecological_scales
    
    def test_validate_parameters(self, analyzer):
        """Test parameter validation."""
        # Valid parameters
        valid_params = {
            'min_area_km2': 100,
            'ecological_scale': 'landscape',
            'contiguity': 'queen',
            'iterations': 5
        }
        is_valid, issues = analyzer.validate_parameters(valid_params)
        assert is_valid
        
        # Invalid area
        invalid_params = valid_params.copy()
        invalid_params['min_area_km2'] = -10
        is_valid, issues = analyzer.validate_parameters(invalid_params)
        assert not is_valid
        assert any('positive' in issue for issue in issues)
        
        # Too small area
        invalid_params = valid_params.copy()
        invalid_params['min_area_km2'] = 1  # Less than 4 pixels
        is_valid, issues = analyzer.validate_parameters(invalid_params)
        assert not is_valid
        assert any('too small' in issue for issue in issues)
    
    @patch('src.spatial_analysis.maxp_regions.region_optimizer.MaxPHeuristic')
    def test_analyze_basic(self, mock_maxp, analyzer):
        """Test basic Max-p analysis."""
        # Setup mock Max-p
        mock_model = Mock()
        mock_model.labels_ = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        mock_maxp.return_value = mock_model
        
        # Mock data preparation
        test_data = np.random.rand(4, 2)  # 4x2 grid
        with patch.object(analyzer, 'prepare_data') as mock_prepare:
            mock_prepare.return_value = (test_data, {
                'original_shape': (1, 4, 2),
                'bands': ['richness'],
                'normalized': False
            })
            
            with patch.object(analyzer, '_build_spatial_weights') as mock_weights:
                # Create a proper weights mock with neighbors attribute
                mock_w = Mock()
                mock_w.neighbors = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}  # Simple chain
                mock_weights.return_value = mock_w
                
                with patch.object(analyzer, 'restore_spatial_structure') as mock_restore:
                    mock_restore.return_value = xr.DataArray(np.array([[0, 0], [1, 1], [0, 0], [1, 1]]))
                    
                    # Run analysis - use min_area_km2 that satisfies validation
                    # With pixel_area = 3.24 km², need at least 3.7 pixels = 11.988 km²
                    result = analyzer.analyze(
                        np.random.rand(4, 2),
                        min_area_km2=13,  # > 11.988 km² to satisfy validation
                        run_perturbation=False
                    )
                    
                    assert result.metadata.analysis_type == 'MaxP'
                    assert result.labels.shape == (8,)
                    assert len(np.unique(result.labels)) == 2
    
    def test_calculate_boundary_stability(self, analyzer):
        """Test boundary stability calculation."""
        labels1 = np.array([0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]).reshape(4, 4)
        labels2 = np.array([0, 0, 1, 1, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]).reshape(4, 4)
        
        stability = analyzer._calculate_boundary_stability(
            labels1.flatten(), labels2.flatten()
        )
        
        assert 0 <= stability <= 1
        assert stability > 0.5  # Most boundaries preserved
    
    def test_region_areas(self, analyzer):
        """Test region area calculation."""
        labels = np.array([0, 0, 1, 1, 1, 2])
        areas = analyzer._get_region_areas(labels)
        
        assert areas[0] == 2 * analyzer.pixel_area_km2
        assert areas[1] == 3 * analyzer.pixel_area_km2
        assert areas[2] == 1 * analyzer.pixel_area_km2


class TestContiguityBuilder:
    """Tests for contiguity builder."""
    
    def test_rook_contiguity(self):
        """Test rook contiguity construction."""
        w = ContiguityBuilder.build_rook_contiguity(3, 3)
        
        # Check center pixel (index 4) has 4 neighbors
        assert len(w.neighbors[4]) == 4
        assert set(w.neighbors[4]) == {1, 3, 5, 7}
    
    def test_queen_contiguity(self):
        """Test queen contiguity construction."""
        w = ContiguityBuilder.build_queen_contiguity(3, 3)
        
        # Check center pixel (index 4) has 8 neighbors
        assert len(w.neighbors[4]) == 8
    
    def test_custom_kernel(self):
        """Test custom kernel contiguity."""
        # Cross-shaped kernel
        kernel = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ], dtype=bool)
        
        w = ContiguityBuilder.build_custom_contiguity(3, 3, kernel)
        
        # Should be same as rook
        assert len(w.neighbors[4]) == 4


class TestRegionReporter:
    """Tests for region reporter."""
    
    @pytest.fixture
    def reporter(self):
        return RegionReporter()
    
    @pytest.fixture
    def mock_result(self):
        from src.spatial_analysis.base_analyzer import AnalysisResult, AnalysisMetadata
        
        return AnalysisResult(
            labels=np.array([0, 0, 1, 1]),
            metadata=AnalysisMetadata(
                analysis_type='MaxP',
                input_shape=(1, 2, 2),
                input_bands=['richness'],
                parameters={
                    'min_area_km2': 10,
                    'ecological_scale': 'landscape'
                },
                processing_time=0.1,
                timestamp='2025-01-01'
            ),
            statistics={
                'n_regions': 2,
                'min_area_threshold_km2': 10,
                'ecological_scale': 'landscape',  # Add missing key
                'mean_region_area_km2': 12,  # Add missing key
                'region_statistics': {
                    0: {'area_km2': 12, 'pixel_count': 2, 'percentage_of_total': 50,
                        'mean': [0.5], 'std': [0.1], 'within_variance': 0.01},
                    1: {'area_km2': 12, 'pixel_count': 2, 'percentage_of_total': 50,
                        'mean': [0.6], 'std': [0.1], 'within_variance': 0.01}
                },
                'variance_explained': 0.8,
                'smallest_region_km2': 12,
                'largest_region_km2': 12,
                'threshold_satisfied': True
            },
            additional_outputs={
                'compactness_scores': {0: 0.7, 1: 0.8},
                'homogeneity_scores': {0: 0.9, 1: 0.85}
            }
        )
    
    def test_generate_region_summary(self, reporter, mock_result):
        """Test region summary generation."""
        df = reporter.generate_region_summary(mock_result)
        
        assert len(df) == 2
        assert 'area_km2' in df.columns
        assert df['area_km2'].sum() == 24
    
    def test_quality_report(self, reporter, mock_result):
        """Test quality report generation."""
        quality = reporter.generate_quality_report(mock_result)
        
        assert quality['variance_explained'] == 0.8
        assert quality['threshold_satisfied'] == True
        assert 0 < quality['avg_compactness'] < 1


class TestMaxPIntegration:
    """Integration tests for Max-p workflow."""
    
    @pytest.fixture
    def config(self):
        from src.config.config import Config
        config = Config()
        config.config = {
            'spatial_analysis': {
                'maxp': {
                    'pixel_size_km': 1.8,
                    'min_area_km2': 20,
                    'iterations': 2
                },
                'normalize_data': False,
                'save_results': False
            },
            'processors': {
                'data_preparation': {}
            }
        }
        return config
    
    def test_maxp_full_workflow(self, config):
        """Test complete Max-p workflow."""
        analyzer = MaxPAnalyzer(config)
        
        # Create small test data (4x4 grid)
        np.random.seed(42)
        data = xr.DataArray(
            np.random.rand(4, 4),
            dims=['y', 'x'],
            name='richness'
        )
        
        # Run analysis - use min_area that allows for reasonable regions
        # With 4x4 grid (16 pixels) and pixel_area=3.24 km²/pixel
        # Use min_area = 13 km² ≈ 4 pixels to allow multiple regions
        result = analyzer.analyze(
            data,
            min_area_km2=13,  # About 4 pixels (13/3.24 = 4.01)
            run_perturbation=False
        )
        
        # Verify results
        assert result.metadata.analysis_type == 'MaxP'
        assert result.labels.shape == (16,)
        assert 'region_statistics' in result.statistics
        
        # Each region should have at least min_pixels (using actual analyzer pixel area)
        min_pixels = 13 / analyzer.pixel_area_km2  # About 4 pixels
        for region_stats in result.statistics['region_statistics'].values():
            assert region_stats['pixel_count'] >= int(min_pixels)
        
        # Test reporting
        reporter = RegionReporter()
        summary = reporter.generate_region_summary(result)
        assert len(summary) > 0