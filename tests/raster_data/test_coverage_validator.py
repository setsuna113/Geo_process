# tests/raster_data/test_coverage_validator.py
import pytest
import geopandas as gpd
from shapely.geometry import box

from src.domain.raster.loaders.geotiff_loader import GeoTIFFLoader
from src.domain.raster.validators.coverage_validator import CoverageValidator
from src.grid_systems.grid_factory import GridFactory

class TestCoverageValidator:
    """Test coverage validation functionality."""
    
    @pytest.fixture
    def test_grid(self):
        """Create a test grid."""
        from src.grid_systems.grid_factory import GridSpecification
        from src.grid_systems.bounds_manager import BoundsDefinition
        factory = GridFactory()
        
        bounds_def = BoundsDefinition(
            name="test_bounds",
            bounds=(-5, 45, 5, 55),  # Smaller than raster
            crs="EPSG:4326"
        )
        
        spec = GridSpecification(
            grid_type='cubic',
            resolution=25000,  # 25km in meters
            bounds=bounds_def,
            crs='EPSG:4326'
        )
        
        grid = factory.create_grid(spec)
        # Grid creation should be sufficient for testing
        return grid
    
    def test_validate_full_coverage(self, real_config, sample_raster, test_grid):
        loader = GeoTIFFLoader(real_config)
        validator = CoverageValidator(loader)
        
        result = validator.validate_coverage(sample_raster, test_grid)
        
        assert result['fully_covers'] == True
        assert result['coverage_ratio'] >= 0.99
        assert result['resolution_adequate'] == True
        # Sample data includes ~5% random nodata values, so some gaps are expected
        # Accept up to 15% of sampled cells having gaps (reasonable tolerance)
        total_cells = len(test_grid.get_cells())
        sample_size = max(1, int(total_cells * 0.1))  # 10% sample rate
        max_expected_gaps = int(sample_size * 0.15)  # 15% of sample
        assert len(result['coverage_gaps']) <= max_expected_gaps
        
    def test_validate_partial_coverage(self, real_config, sample_raster, raster_helper, test_data_dir):
        # Create grid that extends beyond raster
        from src.grid_systems.grid_factory import GridSpecification
        from src.grid_systems.bounds_manager import BoundsDefinition
        factory = GridFactory()
        
        bounds_def = BoundsDefinition(
            name="extended_bounds",
            bounds=(-20, 30, 20, 70),  # Extends beyond raster
            crs="EPSG:4326"
        )
        
        spec = GridSpecification(
            grid_type='cubic',
            resolution=25000,  # 25km in meters
            bounds=bounds_def,
            crs='EPSG:4326'
        )
        
        grid = factory.create_grid(spec)
        
        loader = GeoTIFFLoader(real_config)
        validator = CoverageValidator(loader)
        
        result = validator.validate_coverage(sample_raster, grid)
        
        assert result['fully_covers'] == False
        assert result['coverage_ratio'] < 1.0
        assert 'Consider using a larger raster' in result['recommendations'][0]
        
    def test_resolution_compatibility(self, real_config, test_grid, raster_helper, test_data_dir):
        # Create coarse resolution raster
        coarse_raster = test_data_dir / "coarse.tif"
        raster_helper.create_test_raster(
            coarse_raster,
            width=20,  # Very coarse
            height=20,
            bounds=(-10, 40, 10, 60)
        )
        
        loader = GeoTIFFLoader(real_config)
        validator = CoverageValidator(loader)
        
        result = validator.validate_coverage(coarse_raster, test_grid)
        
        assert result['resolution_adequate'] == False
        assert result['resolution_ratio'] > 1.0
        assert 'resolution is' in result['recommendations'][0]
        
    def test_find_coverage_gaps(self, real_config, raster_helper, test_data_dir, test_grid):
        # Create raster with nodata edges
        gappy_raster = test_data_dir / "gappy.tif"
        raster_helper.create_test_raster(
            gappy_raster,
            pattern="edge_effects",
            bounds=(-10, 40, 10, 60)
        )
        
        loader = GeoTIFFLoader(real_config)
        validator = CoverageValidator(loader)
        
        # Increase sample rate for this test
        gaps = validator._find_coverage_gaps(
            gappy_raster, 
            test_grid, 
            loader.extract_metadata(gappy_raster),
            sample_rate=0.5
        )
        
        # Should find some gaps
        assert len(gaps) > 0
        assert all(gap['type'] == 'nodata' for gap in gaps)
        
    def test_validate_multiple_rasters(self, real_config, raster_helper, test_data_dir):
        # Create multiple rasters
        raster1 = test_data_dir / "raster1.tif"
        raster2 = test_data_dir / "raster2.tif"
        
        raster_helper.create_test_raster(
            raster1,
            bounds=(-10, 40, 0, 60),  # Western half
            pixel_size=0.0166667
        )
        
        raster_helper.create_test_raster(
            raster2,
            bounds=(0, 40, 10, 60),   # Eastern half
            pixel_size=0.0166667      # Same resolution
        )
        
        loader = GeoTIFFLoader(real_config)
        validator = CoverageValidator(loader)
        
        result = validator.validate_multiple_rasters([raster1, raster2])
        
        assert result['consistency']['resolution_consistent'] == True
        assert result['consistency']['crs_consistent'] == True
        assert result['consistency']['spatial_overlap'] == False  # Adjacent, not overlapping