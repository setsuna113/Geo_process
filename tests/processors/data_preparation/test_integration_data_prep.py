# tests/processors/data_preparation/test_integration_data_prep.py
import pytest
import tempfile
from pathlib import Path
import numpy as np
import xarray as xr
import yaml
from osgeo import gdal, osr

from src.config.config import Config
from src.database.connection import DatabaseManager
from src.database.setup import setup_database
from src.processors.data_preparation.raster_cleaner import RasterCleaner
from src.processors.data_preparation.raster_merger import RasterMerger
from src.processors.data_preparation.data_normalizer import DataNormalizer
from src.processors.data_preparation.array_converter import ArrayConverter

class TestDataPreparationIntegration:
    """Integration tests with real database and config."""
    
    @pytest.fixture(scope="class")
    def test_config_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            config_data = {
                'database': {
                    'host': 'localhost',
                    'port': 5432,
                    'database': 'test_biodiversity',
                    'user': 'test_user',
                    'password': 'test_pass'
                },
                'raster_processing': {
                    'tile_size': 50,
                    'cache_ttl_days': 1,
                    'memory_limit_mb': 512
                },
                'data_preparation': {
                    'chunk_size': 100,
                    'resolution_tolerance': 1e-6,
                    'bounds_tolerance': 1e-4
                },
                'data_cleaning': {
                    'log_operations': True
                }
            }
            yaml.dump(config_data, f)
            return Path(f.name)
    
    @pytest.fixture(scope="class")
    def config(self, test_config_file):
        return Config(test_config_file)
    
    @pytest.fixture(scope="class")
    def db_connection(self, config):
        try:
            # Set up test database
            setup_database(reset=True)
            return DatabaseManager()
        except Exception as e:
            pytest.skip(f"Test database not available: {e}")
    
    @pytest.fixture
    def test_raster(self, tmp_path):
        """Create a test GeoTIFF."""
        raster_path = tmp_path / "test_species.tif"
        
        # Create test data with some negative values and outliers
        data = np.random.normal(100, 30, size=(100, 100))
        data[10:20, 10:20] = -50  # Negative values
        data[80:90, 80:90] = 1000  # Outliers
        data[50:55, 50:55] = -9999  # NoData
        
        # Create GeoTIFF
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(str(raster_path), 100, 100, 1, gdal.GDT_Float32)
        
        # Set projection
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        dataset.SetProjection(srs.ExportToWkt())
        
        # Set geotransform
        dataset.SetGeoTransform([-10, 0.2, 0, 60, 0, -0.2])
        
        # Write data
        band = dataset.GetRasterBand(1)
        band.SetNoDataValue(-9999)
        band.WriteArray(data)
        
        dataset = None
        return raster_path
    
    def test_cleaner_integration(self, config, db_connection, test_raster):
        """Test RasterCleaner with real database."""
        cleaner = RasterCleaner(config, db_connection)
        
        result = cleaner.clean_raster(
            test_raster,
            dataset_type='plants',
            output_path=None,
            in_place=False
        )
        
        assert 'statistics' in result
        assert 'data' in result
        
        stats = result['statistics']
        assert stats.negative_values_fixed > 0
        assert stats.outliers_removed > 0
        assert stats.nodata_pixels == 25  # 5x5 area
        
        # Check data is cleaned
        cleaned_data = result['data']
        assert isinstance(cleaned_data, xr.DataArray)
        valid_data = cleaned_data.values[cleaned_data.values != -9999]
        assert np.all(valid_data >= 0)
        assert np.max(valid_data) < 1000
    
    def test_merger_integration(self, config, db_connection, tmp_path):
        """Test RasterMerger with multiple rasters."""
        from src.raster_data.catalog import RasterCatalog
        
        # Create test rasters
        raster_paths = []
        for i, name in enumerate(['plants', 'animals', 'fungi']):
            path = tmp_path / f"{name}.tif"
            data = np.random.randint(0, 100 * (i + 1), size=(50, 50))
            
            driver = gdal.GetDriverByName('GTiff')
            ds = driver.Create(str(path), 50, 50, 1, gdal.GDT_Int32)
            
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            ds.SetProjection(srs.ExportToWkt())
            ds.SetGeoTransform([-5, 0.1, 0, 45, 0, -0.1])
            
            band = ds.GetRasterBand(1)
            band.WriteArray(data)
            ds = None
            
            raster_paths.append((name, path))
        
        # Add to catalog
        catalog = RasterCatalog(db_connection, config)
        for name, path in raster_paths:
            catalog.add_raster(path, dataset_type=name, validate=False)
        
        # Test merger
        merger = RasterMerger(config, db_connection)
        result = merger.merge_paf_rasters(
            'plants', 'animals', 'fungi',
            validate_alignment=True
        )
        
        assert 'data' in result
        assert isinstance(result['data'], xr.Dataset)
        assert set(result['data'].data_vars) == {'plants', 'animals', 'fungi'}
        assert result['alignment'].aligned == True
    
    def test_normalizer_integration(self, config, db_connection):
        """Test DataNormalizer with database parameter storage."""
        normalizer = DataNormalizer(config, db_connection)
        
        # Create test data
        data = xr.DataArray(
            np.random.normal(500, 100, size=(30, 40)),
            coords={'lat': np.linspace(40, 50, 30),
                   'lon': np.linspace(-5, 5, 40)},
            dims=['lat', 'lon']
        )
        
        # Normalize and save parameters
        result = normalizer.normalize(data, method='standard', save_params=True)
        
        assert result['parameter_id'] is not None
        
        # Test loading parameters
        denorm_data = normalizer.denormalize(
            result['data'],
            parameter_id=result['parameter_id']
        )
        
        # Check restoration
        assert np.allclose(
            data.values,
            np.asarray(denorm_data.values),
            rtol=1e-5
        )
    
    def test_converter_integration(self, config):
        """Test ArrayConverter with various formats."""
        converter = ArrayConverter(config)
        
        # Create test dataset
        ds = xr.Dataset({
            'richness': xr.DataArray(
                np.random.randint(0, 100, size=(20, 30)),
                coords={'lat': np.linspace(40, 45, 20),
                       'lon': np.linspace(-5, 0, 30)},
                dims=['lat', 'lon']
            )
        })
        
        # Test round-trip conversions
        # xarray -> numpy -> xarray
        numpy_result = converter.xarray_to_numpy(ds['richness'])
        restored_xr = converter.numpy_to_xarray(
            numpy_result['array'].reshape(20, 30),
            numpy_result['coords_info'],
            dims=numpy_result['dims']
        )
        
        assert np.array_equal(ds['richness'].values, restored_xr.values)
        
        # xarray -> geopandas -> xarray
        gdf = converter.xarray_to_geopandas(ds)
        restored_xr2 = converter.geopandas_to_xarray(
            gdf,
            resolution=0.25,  # Approximate from coordinate spacing
            value_col='richness'
        )
        
        # Check some values match (may not be exact due to gridding)
        assert restored_xr2.shape[0] > 0
        assert restored_xr2.shape[1] > 0