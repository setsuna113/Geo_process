# tests/processors/data_preparation/test_array_converter.py
import pytest
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
from unittest.mock import Mock

from src.processors.data_preparation.array_converter import ArrayConverter

class TestArrayConverter:
    """Test ArrayConverter functionality."""
    
    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.get.return_value = {'chunk_size': 100}
        return config
    
    @pytest.fixture
    def converter(self, mock_config):
        return ArrayConverter(mock_config)
    
    @pytest.fixture
    def sample_dataarray(self):
        data = np.arange(20).reshape(4, 5)
        return xr.DataArray(
            data,
            coords={
                'lat': [40, 41, 42, 43],
                'lon': [-5, -4, -3, -2, -1]
            },
            dims=['lat', 'lon'],
            attrs={'units': 'count', 'crs': 'EPSG:4326'}
        )
    
    @pytest.fixture
    def sample_dataset(self, sample_dataarray):
        return xr.Dataset({
            'var1': sample_dataarray,
            'var2': sample_dataarray * 2
        })
    
    def test_xarray_to_numpy_flatten(self, converter, sample_dataarray):
        result = converter.xarray_to_numpy(sample_dataarray, flatten=True)
        
        assert 'array' in result
        assert 'shape' in result
        assert 'dims' in result
        assert 'coords_info' in result
        
        # Check flattened shape
        assert result['array'].shape == (20,)
        assert result['shape'] == (4, 5)
        assert result['dims'] == ['lat', 'lon']
    
    def test_xarray_to_numpy_no_flatten(self, converter, sample_dataarray):
        result = converter.xarray_to_numpy(sample_dataarray, flatten=False)
        
        assert result['array'].shape == (4, 5)
        assert np.array_equal(result['array'], sample_dataarray.values)
    
    def test_numpy_to_xarray(self, converter, sample_dataarray):
        # Convert to numpy first
        numpy_result = converter.xarray_to_numpy(sample_dataarray)
        
        # Convert back to xarray
        restored = converter.numpy_to_xarray(
            numpy_result['array'].reshape(4, 5),
            numpy_result['coords_info'],
            dims=['lat', 'lon'],
            attrs=numpy_result['attrs']
        )
        
        assert isinstance(restored, xr.DataArray)
        assert restored.shape == sample_dataarray.shape
        assert np.array_equal(restored.values, sample_dataarray.values)
        assert list(restored.dims) == list(sample_dataarray.dims)
    
    def test_xarray_to_geopandas_dataarray(self, converter, sample_dataarray):
        gdf = converter.xarray_to_geopandas(sample_dataarray)
        
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 20  # 4x5 grid points
        assert 'value' in gdf.columns
        assert gdf.crs == 'EPSG:4326'
        
        # Check geometries
        assert all(isinstance(geom, Point) for geom in gdf.geometry)
    
    def test_xarray_to_geopandas_dataset(self, converter, sample_dataset):
        gdf = converter.xarray_to_geopandas(sample_dataset)
        
        assert 'var1' in gdf.columns
        assert 'var2' in gdf.columns
        assert len(gdf) == 20
    
    def test_geopandas_to_xarray(self, converter):
        # Create sample GeoDataFrame
        points = []
        values = []
        for lat in [40, 41, 42]:
            for lon in [-2, -1, 0]:
                points.append(Point(lon, lat))
                values.append(lat + lon)
        
        gdf = gpd.GeoDataFrame(
            {'value': values},
            geometry=points,
            crs='EPSG:4326'
        )
        
        # Convert to xarray
        da = converter.geopandas_to_xarray(gdf, resolution=1.0)
        
        assert isinstance(da, xr.DataArray)
        assert 'lat' in da.dims
        assert 'lon' in da.dims
        assert da.attrs['crs'] == 'EPSG:4326'
    
    def test_flatten_spatial(self, converter, sample_dataarray):
        result = converter.flatten_spatial(sample_dataarray)
        
        assert 'array' in result
        assert 'original_shape' in result
        assert 'spatial_dims' in result
        assert 'indices' in result
        
        assert result['array'].shape == (20,)
        assert result['original_shape'] == (4, 5)
        assert result['spatial_dims'] == ['lat', 'lon']
    
    def test_unflatten_spatial(self, converter, sample_dataarray):
        # Flatten first
        flattened = converter.flatten_spatial(sample_dataarray)
        
        # Unflatten
        restored = converter.unflatten_spatial(
            flattened['array'],
            flattened
        )
        
        assert isinstance(restored, xr.DataArray)
        assert restored.shape == sample_dataarray.shape
        assert np.array_equal(restored.values, sample_dataarray.values)
    
    def test_process_chunked_to_numpy(self, converter, sample_dataarray):
        # Create larger array that would be chunked
        large_data = np.random.rand(200, 200)
        large_da = xr.DataArray(
            large_data,
            coords={
                'lat': np.linspace(40, 60, 200),
                'lon': np.linspace(-10, 10, 200)
            },
            dims=['lat', 'lon']
        )
        
        result = converter.process_chunked(
            large_da,
            operation='to_numpy',
            flatten=True
        )
        
        assert 'array' in result
        assert result['array'].shape == (40000,)  # 200x200 flattened
    
    def test_extract_coord_info(self, converter, sample_dataarray):
        coord_info = converter._extract_coord_info(sample_dataarray)
        
        assert 'lat' in coord_info
        assert 'lon' in coord_info
        
        assert np.array_equal(coord_info['lat']['values'], sample_dataarray.lat.values)
        assert np.array_equal(coord_info['lon']['values'], sample_dataarray.lon.values)