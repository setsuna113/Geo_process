# tests/integration/test_real_data_smoke.py
import pytest
from pathlib import Path
import numpy as np
import rasterio
from osgeo import gdal

from src.config.config import Config
from src.domain.raster.loaders.geotiff_loader import GeoTIFFLoader
from src.grid_systems.grid_factory import GridFactory
from src.processors.richness_processor import RichnessProcessor

class TestRealDataSmoke:
    """Quick tests with real data samples."""
    
    @pytest.fixture
    def real_data_paths(self):
        """Get paths to Michael's rasters."""
        base_path = Path("/home/jason/geo/data/richness_maps")
        return {
            'plants': base_path / "daru-plants-richness.tif",
            'vertebrates': base_path / "iucn-terrestrial-richness.tif"
        }
    
    def test_inspect_real_rasters(self, real_data_paths):
        """Inspect and validate real raster properties."""
        for name, path in real_data_paths.items():
            if not path.exists():
                pytest.skip(f"Real data not available: {path}")
            
            with rasterio.open(path) as src:
                print(f"\n{name.upper()} Raster Properties:")
                print(f"  Size: {src.width} x {src.height}")
                print(f"  Bounds: {src.bounds}")
                print(f"  CRS: {src.crs}")
                print(f"  Resolution: {src.res}")
                print(f"  Data type: {src.dtypes[0]}")
                
                # Sample statistics
                data_sample = src.read(1, window=((0, 100), (0, 100)))
                valid_data = data_sample[data_sample != src.nodata]
                
                if len(valid_data) > 0:
                    print(f"  Sample stats: min={valid_data.min()}, "
                          f"max={valid_data.max()}, "
                          f"mean={valid_data.mean():.1f}")
                
                # Verify expected properties
                assert src.crs.to_epsg() == 4326, f"Unexpected CRS: {src.crs}"
                assert abs(src.res[0] - 0.0166667) < 0.00001, f"Unexpected resolution: {src.res}"
    
    def test_small_region_processing(self, real_data_paths, tmp_path):
        """Process a small region from real data."""
        plants_path = real_data_paths['plants']
        
        if not plants_path.exists():
            pytest.skip("Real plant data not available")
        
        # Define small test region (1° x 1° in Europe)
        test_bounds = (5.0, 45.0, 6.0, 46.0)  # Small area in Alps
        
        # Create config
        config = Config()
        
        # Generate grid for test region
        factory = GridFactory()
        grid = factory.create_grid(
            grid_type="cubic",
            resolution=10.0,  # 10km
            bounds=test_bounds,
            epsg=4326
        )
        
        print(f"\nGenerated {len(grid)} grid cells")
        
        # Load raster data for region
        loader = GeoTIFFLoader(config)
        raster_data = loader.load_window(plants_path, test_bounds)
        
        # Process richness
        processor = RichnessProcessor(config)
        result = processor.process_raster(raster_data, grid)
        
        # Validate results
        assert 'richness' in result
        assert len(result['richness']) == len(grid)
        
        # Check value ranges
        richness_values = list(result['richness'].values())
        valid_values = [v for v in richness_values if v is not None and v > 0]
        
        print(f"\nRichness statistics:")
        print(f"  Cells with data: {len(valid_values)}/{len(grid)}")
        if valid_values:
            print(f"  Min: {min(valid_values)}")
            print(f"  Max: {max(valid_values)}")
            print(f"  Mean: {np.mean(valid_values):.1f}")
            
            # Sanity checks
            assert max(valid_values) < 10000, "Unreasonably high richness"
            assert min(valid_values) > 0, "Invalid richness values"
    
    def test_format_comparison(self, real_data_paths):
        """Compare data formats between plant and vertebrate rasters."""
        results = {}
        
        for name, path in real_data_paths.items():
            if not path.exists():
                continue
                
            # Use GDAL for detailed inspection
            dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
            band = dataset.GetRasterBand(1)
            
            # Get data type info
            dtype_name = gdal.GetDataTypeName(band.DataType)
            dtype_size = gdal.GetDataTypeSize(band.DataType)
            
            # Compute statistics (sampling for speed)
            stats = band.ComputeStatistics(False)
            
            results[name] = {
                'dtype': dtype_name,
                'dtype_size': dtype_size,
                'min': stats[0],
                'max': stats[1],
                'mean': stats[2],
                'std': stats[3],
                'nodata': band.GetNoDataValue()
            }
            
            dataset = None
        
        # Compare formats
        if len(results) == 2:
            print("\nFormat comparison:")
            for key in ['dtype', 'dtype_size', 'nodata']:
                plant_val = results['plants'][key]
                vert_val = results['vertebrates'][key]
                match = "✓" if plant_val == vert_val else "✗"
                print(f"  {key}: plants={plant_val}, vertebrates={vert_val} {match}")
            
            # Verify handling requirements
            assert results['plants']['dtype'] in ['Int32', 'Float32']
            assert results['vertebrates']['dtype'] in ['UInt16', 'Int16', 'Int32']
    
    def test_memory_efficient_loading(self, real_data_paths, tmp_path):
        """Test memory-efficient loading of large rasters."""
        plants_path = real_data_paths['plants']
        
        if not plants_path.exists():
            pytest.skip("Real plant data not available")
        
        config = Config()
        config.raster_processing.memory_limit_mb = 100  # Low limit
        
        loader = GeoTIFFLoader(config)
        
        # Test lazy loading
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Open raster lazily
        with loader.open_lazy(plants_path) as raster:
            # Memory should not spike
            open_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = open_memory - initial_memory
            
            assert memory_increase < 50, f"Memory spike on open: {memory_increase:.1f}MB"
            
            # Read small window
            window_data = raster.read_window((0, 0, 100, 100))
            
            window_memory = process.memory_info().rss / 1024 / 1024
            total_increase = window_memory - initial_memory
            
            assert total_increase < 100, f"Memory limit exceeded: {total_increase:.1f}MB"
            
            # Verify data
            assert window_data.shape == (100, 100)
            assert window_data.dtype in [np.int32, np.uint16]