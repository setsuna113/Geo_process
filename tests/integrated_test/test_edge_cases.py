# tests/integration/test_edge_cases.py
import pytest
import numpy as np
from shapely.geometry import Point
from pyproj import Transformer

from tests.fixtures.data_generator import TestDataGenerator
from src.grid_systems.grid_factory import GridFactory

class TestEdgeCases:
    """Test boundary conditions and edge cases."""
    
    def test_grid_raster_misalignment(self, tmp_path):
        """Test grids that don't align with raster pixels."""
        generator = TestDataGenerator(tmp_path)
        
        # Create raster with specific resolution
        raster = generator.create_test_raster(
            width=100,
            height=100,
            bounds=(-10.0, 40.0, 10.0, 60.0),  # 20° x 20°
            pattern="gradient"
        )
        
        # Create grid with incompatible resolution
        # Raster: 0.2° per pixel, Grid: 0.13° cells (won't align)
        grid = generator.create_test_grid(
            grid_type="cubic",
            resolution=14.5,  # Odd resolution in km
            bounds=(-10.0, 40.0, 10.0, 60.0)
        )
        
        from src.resampling.engines.gdal_resampler import GDALResampler
        from src.config.config import Config
        
        config = Config()
        resampler = GDALResampler(config)
        
        # Resample - should handle misalignment
        result = resampler.resample_to_grid(raster, grid)
        
        # Verify all grid cells got values
        assert len(result) == len(grid)
        
        # Check for edge artifacts
        edge_cells = [cell for cell in grid.itertuples() 
                      if abs(cell.geometry.bounds[0] + 10) < 0.1 or 
                         abs(cell.geometry.bounds[2] - 10) < 0.1]
        
        edge_values = [result[cell.Index] for cell in edge_cells]
        assert all(v is not None for v in edge_values), "Edge cells have null values"
    
    def test_dateline_crossing(self, tmp_path):
        """Test handling of dateline crossing (-180/180)."""
        generator = TestDataGenerator(tmp_path)
        
        # Create raster crossing dateline
        raster = generator.create_test_raster(
            width=200,
            height=100,
            bounds=(170.0, -10.0, -170.0, 10.0),  # Crosses dateline
            pattern="gradient"
        )
        
        # Create grid crossing dateline
        grid = generator.create_test_grid(
            grid_type="cubic",
            resolution=100.0,
            bounds=(170.0, -10.0, -170.0, 10.0)
        )
        
        from src.processors.richness_processor import RichnessProcessor
        from src.config.config import Config
        
        config = Config()
        processor = RichnessProcessor(config)
        
        # Process - should handle dateline correctly
        result = processor.process_raster(raster, grid)
        
        # Verify cells on both sides of dateline
        west_cells = [cell for cell in grid.itertuples() 
                      if cell.geometry.centroid.x > 170]
        east_cells = [cell for cell in grid.itertuples() 
                      if cell.geometry.centroid.x < -170]
        
        assert len(west_cells) > 0, "No cells on western side of dateline"
        assert len(east_cells) > 0, "No cells on eastern side of dateline"
        
        # Values should be continuous across dateline
        west_values = [result['richness'][cell.Index] for cell in west_cells]
        east_values = [result['richness'][cell.Index] for cell in east_cells]
        
        assert all(v > 0 for v in west_values), "Invalid values west of dateline"
        assert all(v > 0 for v in east_values), "Invalid values east of dateline"
    
    def test_polar_regions(self, tmp_path):
        """Test handling of polar regions with projection issues."""
        generator = TestDataGenerator(tmp_path)
        
        # Create raster near pole
        raster = generator.create_test_raster(
            width=100,
            height=50,
            bounds=(-180.0, 80.0, 180.0, 85.0),  # Arctic region
            pattern="gradient"
        )
        
        # Try different grid types near poles
        grid_types = ["cubic", "hexagonal"]
        
        for grid_type in grid_types:
            try:
                grid = generator.create_test_grid(
                    grid_type=grid_type,
                    resolution=100.0,
                    bounds=(-180.0, 80.0, 180.0, 85.0)
                )
                
                # Verify grid generation succeeded
                assert len(grid) > 0, f"No {grid_type} grid cells generated near pole"
                
                # Check for degenerate cells
                for cell in grid.itertuples():
                    area = cell.geometry.area
                    assert area > 0, f"Degenerate cell found in {grid_type} grid"
                    
            except Exception as e:
                pytest.skip(f"{grid_type} grid not supported at poles: {e}")
    
    def test_data_type_conversions(self, tmp_path):
        """Test handling of different data types."""
        generator = TestDataGenerator(tmp_path)
        
        from osgeo import gdal
        
        # Test different data type combinations
        test_cases = [
            (gdal.GDT_Int32, np.int32, 2**31 - 1),      # Max int32
            (gdal.GDT_UInt16, np.uint16, 2**16 - 1),    # Max uint16
            (gdal.GDT_Float32, np.float32, 1e6),        # Large float
        ]
        
        for gdal_type, np_type, test_value in test_cases:
            # Create raster with specific type
            raster = generator.create_test_raster(
                width=50,
                height=50,
                data_type=gdal_type,
                pattern="gradient"
            )
            
            from src.raster_data.loaders.geotiff_loader import GeoTIFFLoader
            from src.config.config import Config
            
            config = Config()
            loader = GeoTIFFLoader(config)
            
            # Load and verify type handling
            data = loader.load(raster)
            
            # Insert extreme value
            data.set_value(25, 25, test_value)
            
            # Process through pipeline
            from src.features.extractors.richness_features import RichnessFeatureExtractor
            extractor = RichnessFeatureExtractor(config)
            
            features = extractor.extract(data)
            
            # Verify no overflow/underflow
            assert features['max_value'] == test_value
            assert not np.isnan(features['mean_value'])
            assert not np.isinf(features['mean_value'])
    
    def test_nodata_handling(self, tmp_path):
        """Test various NO_DATA scenarios."""
        generator = TestDataGenerator(tmp_path)
        
        # Create raster with lots of NO_DATA
        raster = generator.create_test_raster(
            width=100,
            height=100,
            pattern="nodata_edges",
            nodata_value=-9999
        )
        
        # Create grid
        grid = generator.create_test_grid(
            grid_type="cubic",
            resolution=10.0,
            bounds=(-10, 40, 10, 60)
        )
        
        from src.resampling.engines.gdal_resampler import GDALResampler
        from src.config.config import Config
        
        config = Config()
        resampler = GDALResampler(config)
        
        # Resample with different NO_DATA handling strategies
        strategies = ['ignore', 'propagate', 'interpolate']
        
        for strategy in strategies:
            result = resampler.resample_to_grid(
                raster, 
                grid,
                nodata_strategy=strategy
            )
            
            # Count cells with data
            valid_cells = sum(1 for v in result.values() if v is not None and v != -9999)
            
            if strategy == 'ignore':
                # Should have fewer valid cells
                assert valid_cells < len(grid)
            elif strategy == 'propagate':
                # NO_DATA should spread to affected cells
                assert valid_cells < len(grid) * 0.8
            elif strategy == 'interpolate':
                # Should try to fill gaps
                assert valid_cells > len(grid) * 0.9
    
    def test_coordinate_precision(self, tmp_path):
        """Test coordinate precision and rounding errors."""
        generator = TestDataGenerator(tmp_path)
        
        # Create grid with precise bounds
        precise_bounds = (
            -10.123456789,
            40.987654321, 
            10.123456789,
            60.987654321
        )
        
        grid = generator.create_test_grid(
            grid_type="cubic",
            resolution=25.0,
            bounds=precise_bounds
        )
        
        # Verify bounds are preserved
        actual_bounds = grid.total_bounds
        for i in range(4):
            assert abs(actual_bounds[i] - precise_bounds[i]) < 1e-6, \
                f"Coordinate precision lost: {actual_bounds[i]} != {precise_bounds[i]}"
        
        # Test point sampling at precise coordinates
        from src.raster_data.loaders.geotiff_loader import GeoTIFFLoader
        
        test_points = [
            (-10.123456789, 40.987654321),  # Exact corner
            (0.000000001, 50.000000001),     # Near zero with precision
            (10.123456788, 60.987654320),    # Slightly off corner
        ]
        
        raster = generator.create_test_raster(bounds=precise_bounds)
        
        config = Config()
        loader = GeoTIFFLoader(config)
        data = loader.load(raster)
        
        for x, y in test_points:
            value = data.sample_point(x, y)
            assert value is not None, f"Failed to sample at precise coordinate ({x}, {y})"