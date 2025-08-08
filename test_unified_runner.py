#!/usr/bin/env python
"""
Test script to verify the unified biodiversity analysis runner works correctly.
"""

import sys
from pathlib import Path
import tempfile
import numpy as np
import rasterio

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.run_biodiversity_analysis import BiodiversityAnalysisRunner

def create_test_raster(path: Path, data: np.ndarray, crs='EPSG:4326'):
    """Create a test raster file."""
    height, width = data.shape
    transform = rasterio.transform.from_bounds(-180, -90, 180, 90, width, height)
    
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)

def test_unified_runner():
    """Test the unified runner with synthetic data."""
    print("🧪 Testing unified biodiversity analysis runner...")
    
    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create synthetic test datasets
        print("Creating synthetic test data...")
        np.random.seed(42)  # Reproducible results
        
        # Small test rasters (100x50)
        shape = (50, 100)
        
        test_datasets = {}
        for name in ['plants', 'animals', 'am_fungi', 'ecm_fungi']:
            # Create synthetic richness data
            data = np.random.exponential(20, shape).astype(np.float32)
            # Add some NaN regions (ocean areas)
            mask = np.random.random(shape) < 0.3
            data[mask] = np.nan
            
            # Create test file
            test_file = temp_path / f"{name}_test.tif"
            create_test_raster(test_file, data)
            test_datasets[name] = str(test_file)
            print(f"  Created {name}: {data.shape}, {np.sum(~np.isnan(data))} valid pixels")
        
        # Test output directory
        output_dir = temp_path / "test_output"
        
        # Initialize runner
        runner = BiodiversityAnalysisRunner(str(output_dir))
        
        print(f"\n📋 Available analyzers: {list(runner.available_analyzers.keys())}")
        
        # Test individual analyzers
        print("\n🔍 Testing individual analyzers...")
        results = {}
        
        for analyzer_name in runner.available_analyzers.keys():
            print(f"\n--- Testing {analyzer_name} ---")
            try:
                success = runner.run_analysis(
                    analyzer_name,
                    test_datasets,
                    sample_factor=2,  # Use light sampling for faster testing
                )
                results[analyzer_name] = success
                status = "✅ SUCCESS" if success else "❌ FAILED"
                print(f"{analyzer_name}: {status}")
                
                # Check if output was created
                analyzer_output_dir = output_dir / analyzer_name
                if analyzer_output_dir.exists():
                    files = list(analyzer_output_dir.glob("*"))
                    print(f"  Created {len(files)} output files")
                
            except Exception as e:
                print(f"❌ {analyzer_name} failed with error: {e}")
                results[analyzer_name] = False
        
        # Summary
        print(f"\n{'='*50}")
        print("UNIFIED RUNNER TEST RESULTS:")
        print(f"{'='*50}")
        
        successful = sum(results.values())
        total = len(results)
        
        for analyzer_name, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {analyzer_name:<15}: {status}")
        
        print(f"\nOverall: {successful}/{total} analyzers passed")
        
        if successful == total:
            print("\n🎉 All analyzers working correctly with unified runner!")
            return True
        else:
            print(f"\n⚠️  {total - successful} analyzers failed")
            return False

if __name__ == "__main__":
    success = test_unified_runner()
    sys.exit(0 if success else 1)