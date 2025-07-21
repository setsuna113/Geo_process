# tests/test_spatial_analysis/test_base_analyzer.py
"""Tests for base spatial analyzer."""

import pytest
import numpy as np
import xarray as xr
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import tempfile

from src.spatial_analysis.base_analyzer import BaseAnalyzer, AnalysisResult, AnalysisMetadata


class ConcreteAnalyzer(BaseAnalyzer):
    """Concrete implementation for testing."""
    
    def analyze(self, data, **kwargs):
        # Simple implementation
        labels = np.zeros(data.size, dtype=int)
        metadata = AnalysisMetadata(
            analysis_type='Test',
            input_shape=data.shape,
            input_bands=['test'],
            parameters={},
            processing_time=0.1,
            timestamp='2025-01-01 00:00:00'
        )
        return AnalysisResult(
            labels=labels,
            metadata=metadata,
            statistics={'test': True}
        )
    
    def get_default_parameters(self):
        return {'test_param': 1}
    
    def validate_parameters(self, parameters):
        if parameters.get('test_param', 1) < 0:
            return False, ['test_param must be positive']
        return True, []


class TestBaseAnalyzerUnit:
    """Unit tests with mocked dependencies."""
    
    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.get.return_value = {
            'normalize_data': True,
            'save_results': True,
            'output_dir': 'test_output'
        }
        return config
    
    @pytest.fixture
    def mock_db(self):
        return Mock()
    
    @pytest.fixture
    def analyzer(self, mock_config, mock_db):
        with patch('src.spatial_analysis.base_analyzer.ArrayConverter'), \
             patch('src.spatial_analysis.base_analyzer.DataNormalizer'):
            return ConcreteAnalyzer(mock_config, mock_db)
    
    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.normalize_data == True
        assert analyzer.save_results == True
        assert analyzer.output_dir.name == 'test_output'
    
    def test_prepare_data_numpy(self, analyzer):
        """Test data preparation with numpy array."""
        # 3D array (bands, height, width)
        data = np.random.rand(3, 4, 5)
        
        with patch.object(analyzer.array_converter, 'xarray_to_numpy') as mock_convert:
            mock_convert.return_value = {
                'array': data.reshape(3, -1).T,
                'coords_info': {'test': True}
            }
            
            prepared, metadata = analyzer.prepare_data(data, flatten=True)
            
            assert prepared.shape == (20, 3)  # (pixels, bands)
            assert metadata['original_shape'] == (3, 4, 5)
            assert len(metadata['bands']) == 3
    
    def test_prepare_data_xarray(self, analyzer):
        """Test data preparation with xarray."""
        data = xr.DataArray(
            np.random.rand(3, 4, 5),
            dims=['band', 'y', 'x'],
            coords={'band': ['P', 'A', 'F']}
        )
        
        with patch.object(analyzer.array_converter, 'xarray_to_numpy') as mock_convert:
            mock_convert.return_value = {
                'array': data.values.reshape(3, -1).T,
                'coords_info': {'test': True}
            }
            
            prepared, metadata = analyzer.prepare_data(data, flatten=True)
            
            assert metadata['bands'] == ['P', 'A', 'F']
            assert metadata['original_type'] == 'DataArray'
    
    def test_validate_input_data(self, analyzer):
        """Test input validation."""
        # Valid data
        valid_data = np.random.rand(3, 10, 10)
        is_valid, issues = analyzer.validate_input_data(valid_data)
        assert is_valid
        assert len(issues) == 0
        
        # Empty data
        empty_data = np.array([])
        is_valid, issues = analyzer.validate_input_data(empty_data)
        assert not is_valid
        assert 'Empty array provided' in issues
        
        # All NaN
        nan_data = np.full((3, 10, 10), np.nan)
        is_valid, issues = analyzer.validate_input_data(nan_data)
        assert not is_valid
        assert 'All values are NaN' in issues
    
    def test_restore_spatial_structure(self, analyzer):
        """Test spatial structure restoration."""
        labels = np.arange(20)
        metadata = {
            'original_shape': (1, 4, 5),
            'coords': {
                'lat': list(range(4)),
                'lon': list(range(5))
            }
        }
        
        with patch.object(analyzer.array_converter, 'unflatten_spatial') as mock_unflatten:
            mock_unflatten.return_value = xr.DataArray(
                labels.reshape(4, 5),
                dims=['lat', 'lon']
            )
            
            result = analyzer.restore_spatial_structure(labels, metadata)
            
            assert isinstance(result, xr.DataArray)
            assert result.shape == (4, 5)
    
    def test_save_results(self, analyzer):
        """Test saving results to disk."""
        # Create test result
        result = AnalysisResult(
            labels=np.array([0, 1, 0, 1]),
            metadata=AnalysisMetadata(
                analysis_type='Test',
                input_shape=(2, 2),
                input_bands=['test'],
                parameters={'param': 1},
                processing_time=0.1,
                timestamp='2025-01-01'
            ),
            statistics={'stat': 1},
            spatial_output=xr.DataArray([1, 2, 3, 4])
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer.output_dir = Path(tmpdir)
            output_path = analyzer.save_results(result, 'test_run')
            
            assert output_path.exists()
            assert (output_path / 'labels.npy').exists()
            assert (output_path / 'metadata.json').exists()
            assert (output_path / 'statistics.json').exists()
            assert (output_path / 'spatial_output.nc').exists()
    
    def test_load_results(self, analyzer):
        """Test loading saved results."""
        # First save a result
        result = AnalysisResult(
            labels=np.array([0, 1, 0, 1]),
            metadata=AnalysisMetadata(
                analysis_type='Test',
                input_shape=(2, 2),
                input_bands=['test'],
                parameters={'param': 1},
                processing_time=0.1,
                timestamp='2025-01-01'
            ),
            statistics={'stat': 1}
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer.output_dir = Path(tmpdir)
            output_path = analyzer.save_results(result, 'test_run')
            
            # Load it back
            loaded_result = analyzer.load_results(output_path)
            
            assert np.array_equal(loaded_result.labels, result.labels)
            assert loaded_result.metadata.analysis_type == 'Test'
            assert loaded_result.statistics['stat'] == 1


class TestBaseAnalyzerIntegration:
    """Integration tests with real dependencies."""
    
    @pytest.fixture
    def config(self):
        from src.core.config import Config
        config = Config()
        config.config = {
            'spatial_analysis': {
                'normalize_data': True,
                'save_results': False,
                'output_dir': 'test_output'
            },
            'processors': {
                'data_preparation': {}
            }
        }
        return config
    
    @pytest.fixture
    def analyzer(self, config):
        return ConcreteAnalyzer(config)
    
    def test_full_workflow(self, analyzer):
        """Test complete analysis workflow."""
        # Create test data
        data = xr.Dataset({
            'P': xr.DataArray(np.random.rand(5, 5), dims=['y', 'x']),
            'A': xr.DataArray(np.random.rand(5, 5), dims=['y', 'x']),
            'F': xr.DataArray(np.random.rand(5, 5), dims=['y', 'x'])
        })
        
        # Run analysis
        result = analyzer.analyze(data)
        
        assert isinstance(result, AnalysisResult)
        assert result.labels.shape == (75,)  # 3 bands * 5 * 5
        assert result.metadata.analysis_type == 'Test'