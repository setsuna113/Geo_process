# tests/raster_data/test_value_validator.py
import pytest
import numpy as np
from pathlib import Path

from src.domain.raster.loaders.geotiff_loader import GeoTIFFLoader
from src.domain.raster.validators.value_validator import ValueValidator

class TestValueValidator:
    """Test value validation functionality."""
    
    def test_detect_dataset_type(self, real_config):
        loader = GeoTIFFLoader(real_config)
        validator = ValueValidator(loader)
        
        test_cases = [
            ("daru-plants-richness.tif", "plants"),
            ("iucn-vertebrate-richness.tif", "vertebrates"),
            ("global-terrestrial.tif", "terrestrial"),
            ("marine-species.tif", "marine"),
            ("unknown-data.tif", "unknown")
        ]
        
        for filename, expected in test_cases:
            path = Path(filename)
            detected = validator._detect_dataset_type(path)
            assert detected == expected
            
    def test_validate_plant_values(self, real_config, raster_helper, test_data_dir):
        # Create raster with plant-like values
        plant_raster = test_data_dir / "plants.tif"
        raster_helper.create_test_raster(
            plant_raster,
            pattern="hotspots",
            width=100,
            height=100
        )
        
        loader = GeoTIFFLoader(real_config)
        validator = ValueValidator(loader)
        
        result = validator.validate_values(plant_raster, "plants")
        
        assert result['dataset_type'] == 'plants'
        assert result['validation']['valid'] == True
        assert result['sample_stats']['has_valid_data'] == True
        assert 0 <= result['sample_stats']['min'] <= result['sample_stats']['max'] <= 20000
        
    def test_validate_out_of_range(self, real_config, raster_helper, test_data_dir):
        # Create raster with out-of-range values
        bad_raster = test_data_dir / "bad_values.tif"
        data = np.random.randint(50000, 100000, size=(100, 100), dtype=np.int32)
        
        # Manual creation to inject bad values
        from osgeo import gdal, osr
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(str(bad_raster), 100, 100, 1, gdal.GDT_Int32)
        
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        dataset.SetProjection(srs.ExportToWkt())
        dataset.SetGeoTransform([-10, 0.2, 0, 60, 0, -0.2])
        
        band = dataset.GetRasterBand(1)
        band.WriteArray(data)
        dataset = None
        
        loader = GeoTIFFLoader(real_config)
        validator = ValueValidator(loader)
        
        result = validator.validate_values(bad_raster, "plants")
        
        assert result['validation']['valid'] == False
        assert len(result['validation']['issues']) > 0
        assert 'higher than expected' in result['validation']['issues'][0]
        
    def test_detect_anomalies(self, real_config, raster_helper, test_data_dir):
        # Create raster with uniform values
        uniform_raster = test_data_dir / "uniform.tif"
        raster_helper.create_test_raster(
            uniform_raster,
            pattern="uniform",
            width=100,
            height=100
        )
        
        loader = GeoTIFFLoader(real_config)
        validator = ValueValidator(loader)
        
        result = validator.validate_values(uniform_raster)
        
        anomalies = result['anomalies']
        assert len(anomalies) > 0
        
        # Should detect uniform data
        uniform_anomaly = next((a for a in anomalies if a['type'] == 'uniform_data'), None)
        assert uniform_anomaly is not None
        
    def test_sample_statistics(self, real_config, sample_raster):
        loader = GeoTIFFLoader(real_config)
        validator = ValueValidator(loader)
        
        stats = validator._sample_values(sample_raster, sample_size=1000)
        
        assert stats['count'] > 0
        assert stats['has_valid_data'] == True
        assert 'percentiles' in stats
        assert stats['min'] <= stats['mean'] <= stats['max']
        assert stats['std'] >= 0
        
    def test_nodata_handling(self, real_config, raster_helper, test_data_dir):
        # Create raster with lots of nodata
        nodata_raster = test_data_dir / "nodata.tif"
        data = np.full((100, 100), -9999, dtype=np.int32)
        data[40:60, 40:60] = 100  # Small valid region
        
        from osgeo import gdal, osr
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(str(nodata_raster), 100, 100, 1, gdal.GDT_Int32)
        
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        dataset.SetProjection(srs.ExportToWkt())
        dataset.SetGeoTransform([-10, 0.2, 0, 60, 0, -0.2])
        
        band = dataset.GetRasterBand(1)
        band.SetNoDataValue(-9999)
        band.WriteArray(data)
        dataset = None
        
        loader = GeoTIFFLoader(real_config)
        validator = ValueValidator(loader)
        
        result = validator.validate_values(nodata_raster)
        
        # Should detect excessive nodata
        anomalies = result['anomalies']
        nodata_anomaly = next((a for a in anomalies if a['type'] == 'excessive_nodata'), None)
        assert nodata_anomaly is not None