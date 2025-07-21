# tests/integration/test_workflow_simulation.py
import pytest
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any

from tests.fixtures.data_generator import TestDataGenerator
from tests.integration.validators import SystemStateValidator
from src.core.registry import Registry
from src.database.connection import DatabaseConnection
from src.database.setup import DatabaseSetup
from src.config.config import Config

class TestWorkflowSimulation:
    """Test complete workflow integration."""
    
    @pytest.fixture(scope="class")
    def test_env(self):
        """Set up test environment."""
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create test config
        config_data = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'database': 'test_biodiversity',
                'user': 'test_user',
                'password': 'test_pass'
            },
            'raster_processing': {
                'tile_size': 100,
                'cache_ttl_days': 1,
                'memory_limit_mb': 512,
                'parallel_workers': 2
            },
            'output_formats': {
                'csv': True,
                'parquet': True,
                'geojson': False
            }
        }
        
        config_path = temp_dir / "config.yml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Initialize components
        config = Config(config_path)
        db_setup = DatabaseSetup(config)
        db_setup.initialize()
        
        generator = TestDataGenerator(temp_dir)
        validator = SystemStateValidator(config.database.dict())
        
        yield {
            'temp_dir': temp_dir,
            'config': config,
            'generator': generator,
            'validator': validator,
            'db_setup': db_setup
        }
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_mini_pipeline(self, test_env):
        """Test minimal end-to-end pipeline."""
        generator = test_env['generator']
        validator = test_env['validator']
        config = test_env['config']
        
        # Capture initial state
        validator.capture_initial_state()
        
        # Step 1: Create test data
        test_raster = generator.create_test_raster(
            width=100, 
            height=100,
            pattern="gradient"
        )
        
        test_grid = generator.create_test_grid(
            grid_type="cubic",
            resolution=25.0,
            bounds=(-10, 40, 10, 60)
        )
        
        # Step 2: Register components
        from src.raster.loaders.geotiff_loader import GeoTIFFLoader
        from src.resampling.engines.gdal_resampler import GDALResampler
        from src.features.ml_formatters.regression_formatter import RegressionFormatter
        
        registry = Registry()
        registry.register('raster_loader', 'geotiff', GeoTIFFLoader)
        registry.register('resampler', 'gdal', GDALResampler)
        registry.register('formatter', 'regression', RegressionFormatter)
        
        # Step 3: Execute pipeline
        # Load raster
        loader = registry.get('raster_loader', 'geotiff')(config)
        raster_data = loader.load(test_raster)
        
        # Resample to grid
        resampler = registry.get('resampler', 'gdal')(config)
        resampled_data = resampler.resample(raster_data, test_grid)
        
        # Format for ML
        formatter = registry.get('formatter', 'regression')(config)
        output_path = test_env['temp_dir'] / "output.csv"
        formatter.format(resampled_data, output_path)
        
        # Verify output
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        # Validate system state
        issues = validator.validate_final_state()
        assert len(issues) == 0, f"System state issues: {issues}"
    
    def test_multi_resolution_consistency(self, test_env):
        """Test consistency across multiple resolutions."""
        generator = test_env['generator']
        config = test_env['config']
        
        # Create test raster with known sum
        test_raster = generator.create_test_raster(
            width=200,
            height=200,
            pattern="checkerboard"
        )
        
        # Test multiple resolutions
        resolutions = [10.0, 25.0, 50.0]
        results = {}
        
        for resolution in resolutions:
            grid = generator.create_test_grid(
                grid_type="cubic",
                resolution=resolution,
                bounds=(-10, 40, 10, 60)
            )
            
            # Process at this resolution
            from src.processors.richness_processor import RichnessProcessor
            processor = RichnessProcessor(config)
            result = processor.process_raster(test_raster, grid)
            
            # Store total richness
            results[resolution] = result['total_richness']
        
        # Verify conservation of total richness (within tolerance)
        base_total = results[10.0]
        for res, total in results.items():
            relative_diff = abs(total - base_total) / base_total
            assert relative_diff < 0.01, f"Resolution {res} has {relative_diff:.2%} difference"
    
    def test_registry_component_selection(self, test_env):
        """Test dynamic component selection based on data format."""
        config = test_env['config']
        registry = Registry()
        
        # Register multiple loaders
        from src.raster.loaders.geotiff_loader import GeoTIFFLoader
        # Note: NetCDFLoader not yet implemented
        
        registry.register('raster_loader', 'geotiff', GeoTIFFLoader)
        # registry.register('raster_loader', 'netcdf', NetCDFLoader)  # Not implemented yet
        
        # Test format detection and selection
        test_files = {
            'test.tif': 'geotiff',
            'test.nc': 'netcdf',
            'test.tiff': 'geotiff'
        }
        
        for filename, expected_format in test_files.items():
            detected_format = registry.detect_format(filename)
            assert detected_format == expected_format
            
            loader = registry.get_by_format('raster_loader', detected_format)
            assert loader is not None