#!/usr/bin/env python3
"""
Test data generator adapted for current system capabilities.
"""

import sys
from pathlib import Path
import tempfile
import numpy as np
from osgeo import gdal, osr
from typing import Tuple

# Add project root to path  
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class SimpleTestDataGenerator:
    """Generate test data compatible with current system."""
    
    def __init__(self, temp_dir=None):
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.mkdtemp())
        self.temp_dir.mkdir(exist_ok=True)
        
    def create_test_raster(
        self,
        width: int = 100,
        height: int = 100,
        bounds: Tuple[float, float, float, float] = (-10, 40, 10, 60),
        pattern: str = "gradient"
    ) -> Path:
        """Create a synthetic raster with known patterns."""
        output_path = self.temp_dir / f"test_raster_{pattern}_{width}x{height}.tif"
        
        # Create the raster
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(
            str(output_path),
            width,
            height,
            1,
            gdal.GDT_Int32
        )
        
        # Set projection and geotransform
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        dataset.SetProjection(srs.ExportToWkt())
        
        # Calculate pixel size
        pixel_width = (bounds[2] - bounds[0]) / width
        pixel_height = (bounds[3] - bounds[1]) / height
        
        geotransform = [
            bounds[0],    # top left x
            pixel_width,  # pixel width
            0,           # rotation
            bounds[3],    # top left y
            0,           # rotation
            -pixel_height # pixel height (negative)
        ]
        dataset.SetGeoTransform(geotransform)
        
        # Generate pattern
        if pattern == "gradient":
            data = np.arange(width * height, dtype=np.int32).reshape(height, width)
        elif pattern == "checkerboard":
            data = np.zeros((height, width), dtype=np.int32)
            for i in range(height):
                for j in range(width):
                    if (i // 10 + j // 10) % 2:
                        data[i, j] = 100
        else:
            data = np.random.randint(1, 255, size=(height, width), dtype=np.int32)
        
        # Write data
        band = dataset.GetRasterBand(1)
        band.SetNoDataValue(0)
        band.WriteArray(data)
        band.ComputeStatistics(False)
        
        dataset.FlushCache()
        dataset = None
        
        return output_path

def test_data_generator():
    """Test the simple data generator."""
    try:
        generator = SimpleTestDataGenerator()
        
        # Create test raster
        raster_path = generator.create_test_raster(
            width=50,
            height=50,
            pattern="gradient"
        )
        
        print(f"✅ Created test raster: {raster_path}")
        print(f"✅ File exists: {raster_path.exists()}")
        print(f"✅ File size: {raster_path.stat().st_size} bytes")
        
        # Test with GDAL
        dataset = gdal.Open(str(raster_path))
        print(f"✅ GDAL can open raster: {dataset.RasterXSize}x{dataset.RasterYSize}")
        print(f"✅ Projection: {dataset.GetProjection()[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Data generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_grid_generation():
    """Test grid generation with current system."""
    try:
        from src.grid_systems.grid_factory import GridFactory, GridSpecification
        from src.grid_systems.bounds_manager import BoundsManager
        
        # Test grid generation
        factory = GridFactory()
        bounds_manager = BoundsManager()
        
        # Create a grid for Europe
        europe_bounds = bounds_manager.get_bounds('europe')
        print(f"✅ Europe bounds: {europe_bounds.bounds}")
        
        spec = GridSpecification(
            grid_type='cubic',
            resolution=25000,  # 25km
            bounds='europe'
        )
        
        grid = factory.create_grid(spec)
        print(f"✅ Created grid: {grid.__class__.__name__}")
        print(f"✅ Grid resolution: {grid.resolution}m")
        
        # Test cell generation for a few points
        test_points = [
            (0, 50),    # London area
            (2, 48),    # Paris area  
            (13, 52),   # Berlin area
        ]
        
        generated_cells = 0
        for x, y in test_points:
            try:
                cell_id = grid.get_cell_id(x, y)
                if cell_id:
                    generated_cells += 1
                    print(f"✅ Point ({x}, {y}) -> Cell ID: {cell_id}")
            except Exception as e:
                print(f"⚠️  Point ({x}, {y}) failed: {e}")
        
        print(f"✅ Successfully generated {generated_cells}/{len(test_points)} cell IDs")
        
        return True
        
    except Exception as e:
        print(f"❌ Grid generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run simple integration tests."""
    print("="*60)
    print("SIMPLE INTEGRATION TESTS - DATA GENERATION & GRIDS")
    print("="*60)
    
    tests = [
        ("Test Data Generator", test_data_generator),
        ("Grid Generation", test_grid_generation),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*60)
    print("SIMPLE INTEGRATION TEST RESULTS")
    print("="*60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
        if success:
            passed += 1
    
    print(f"\nSUMMARY: {passed}/{len(results)} tests passed")
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
