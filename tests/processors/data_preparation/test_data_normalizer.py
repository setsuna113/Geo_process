# tests/processors/data_preparation/test_data_normalizer.py
import pytest
import numpy as np
import xarray as xr
from unittest.mock import Mock, patch, MagicMock
from sklearn.preprocessing import StandardScaler

from src.processors.data_preparation.data_normalizer import DataNormalizer

class TestDataNormalizer:
    """Test DataNormalizer with mocked dependencies."""
    
    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.get.return_value = {'chunk_size': 1000}
        return config
    
    @pytest.fixture
    def mock_db(self):
        db = Mock()
        mock_conn = MagicMock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=None)
        db.get_connection.return_value = mock_conn
        return db
    
    @pytest.fixture
    def normalizer(self, mock_config, mock_db):
        return DataNormalizer(mock_config, mock_db)
    
    @pytest.fixture
    def sample_dataarray(self):
        # Create sample data with spatial dimensions
        data = np.random.randn(50, 100) * 100 + 500  # Mean ~500, std ~100
        
        return xr.DataArray(
            data,
            coords={
                'lat': np.linspace(40, 60, 50),
                'lon': np.linspace(-10, 10, 100)
            },
            dims=['lat', 'lon'],
            attrs={'units': 'species', 'description': 'test data'}
        )
    
    @pytest.fixture
    def sample_dataset(self, sample_dataarray):
        # Create dataset with multiple variables
        return xr.Dataset({
            'plants': sample_dataarray,
            'animals': sample_dataarray * 0.5,
            'fungi': sample_dataarray * 0.2
        })
    
    def test_initialization(self, normalizer):
        assert normalizer.chunk_size == 1000
        assert 'standard' in normalizer.scalers
        assert 'minmax' in normalizer.scalers
        assert 'robust' in normalizer.scalers
    
    def test_normalize_dataarray_standard(self, normalizer, sample_dataarray):
        result = normalizer.normalize(
            sample_dataarray,
            method='standard',
            save_params=False
        )
        
        assert 'data' in result
        assert 'parameters' in result
        assert result['method'] == 'standard'
        
        # Check normalized data properties
        normalized = result['data']
        assert isinstance(normalized, xr.DataArray)
        assert normalized.shape == sample_dataarray.shape
        
        # Check approximate standard normalization (mean ~0, std ~1)
        valid_data = normalized.values[~np.isnan(normalized.values)]
        assert abs(np.mean(valid_data)) < 0.1
        assert abs(np.std(valid_data) - 1.0) < 0.1
    
    def test_normalize_dataarray_minmax(self, normalizer, sample_dataarray):
        result = normalizer.normalize(
            sample_dataarray,
            method='minmax',
            feature_range=(0, 1),
            save_params=False
        )
        
        normalized = result['data']
        valid_data = normalized.values[~np.isnan(normalized.values)]
        
        # Check range is [0, 1]
        assert np.min(valid_data) >= -0.01  # Small tolerance
        assert np.max(valid_data) <= 1.01
    
    def test_normalize_dataset(self, normalizer, sample_dataset):
        result = normalizer.normalize(
            sample_dataset,
            method='standard',
            by_band=True,
            save_params=False
        )
        
        assert isinstance(result['data'], xr.Dataset)
        assert set(result['data'].data_vars) == set(sample_dataset.data_vars)
        
        # Check each variable is normalized
        for var in result['data'].data_vars:
            data = result['data'][var].values
            valid_data = data[~np.isnan(data)]
            assert abs(np.mean(valid_data)) < 0.1
    
    def test_normalize_preserve_metadata(self, normalizer, sample_dataarray):
        result = normalizer.normalize(sample_dataarray, save_params=False)
        
        normalized = result['data']
        
        # Check coordinates preserved
        assert np.array_equal(normalized.lat.values, sample_dataarray.lat.values)
        assert np.array_equal(normalized.lon.values, sample_dataarray.lon.values)
        
        # Check attributes updated
        assert normalized.attrs['normalized'] == True
        assert 'normalization_method' in normalized.attrs
    
    def test_denormalize_dataarray(self, normalizer, sample_dataarray):
        # First normalize
        norm_result = normalizer.normalize(sample_dataarray, save_params=False)
        normalized = norm_result['data']
        params = norm_result['parameters']
        
        # Then denormalize
        denorm_result = normalizer.denormalize(normalized, parameters=params)
        
        # Check values are approximately restored
        original = sample_dataarray.values
        restored = denorm_result.values
        
        # Allow for small numerical errors
        assert np.allclose(original, restored, rtol=1e-5, atol=1e-5)
    
    def test_save_load_parameters(self, normalizer, sample_dataarray):
        # Mock database save
        normalizer._save_normalization_params = Mock(return_value=123)
        
        # Normalize with save
        result = normalizer.normalize(sample_dataarray, save_params=True)
        
        assert result['parameter_id'] == 123
        assert normalizer._save_normalization_params.called
    
    def test_normalize_with_nan_values(self, normalizer):
        # Create data with NaN values
        data = np.random.randn(10, 10)
        data[3:5, 3:5] = np.nan
        
        da = xr.DataArray(
            data,
            coords={'lat': range(10), 'lon': range(10)},
            dims=['lat', 'lon']
        )
        
        result = normalizer.normalize(da, save_params=False)
        normalized = result['data']
        
        # Check NaN values preserved
        assert np.isnan(normalized.values[3, 3])
        assert np.isnan(normalized.values[4, 4])
        
        # Check non-NaN values are normalized
        valid_data = normalized.values[~np.isnan(normalized.values)]
        assert len(valid_data) > 0
        assert abs(np.mean(valid_data)) < 0.2
    
    def test_invalid_normalization_method(self, normalizer, sample_dataarray):
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalizer.normalize(sample_dataarray, method='invalid')
    
    def test_parameters_extraction(self, normalizer, sample_dataarray):
        result = normalizer.normalize(sample_dataarray, method='standard', save_params=False)
        params = result['parameters']
        
        assert 'method' in params
        assert 'shape' in params
        assert 'dims' in params
        assert 'mean' in params
        assert 'scale' in params