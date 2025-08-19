#!/usr/bin/env python3
"""
Test if COG output format improves merge performance or memory usage.

This test compares:
1. Standard GTiff output with ZSTD
2. COG output with ZSTD
3. Memory usage patterns during merge
"""

import rasterio
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from pathlib import Path
import time
import psutil
import os
import tempfile
import numpy as np
from contextlib import contextmanager
import tracemalloc

# Test configuration
TEST_SIZE = 2000  # Use 2000x2000 rasters for testing
OVERLAP = 500     # 25% overlap between rasters


@contextmanager
def measure_memory():
    """Context manager to measure memory usage."""
    process = psutil.Process()
    
    # Start memory tracking
    tracemalloc.start()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    yield
    
    # Get memory stats
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"  Memory before: {mem_before:.1f} MB")
    print(f"  Memory after: {mem_after:.1f} MB")
    print(f"  Memory increase: {mem_after - mem_before:.1f} MB")
    print(f"  Peak traced: {peak / 1024 / 1024:.1f} MB")


def create_test_rasters(temp_dir):
    """Create test rasters with overlap."""
    raster_paths = []
    
    for i in range(3):
        # Create data with some pattern
        data = np.random.randint(0, 255, (TEST_SIZE, TEST_SIZE), dtype=np.uint8)
        data = data + i * 50  # Different values per raster
        
        # Create transform with overlap
        from rasterio.transform import from_origin
        x_offset = i * (TEST_SIZE - OVERLAP) * 0.01  # 0.01 degree pixels
        transform = from_origin(x_offset, 50, 0.01, 0.01)
        
        # Write raster
        path = temp_dir / f"test_{i}.tif"
        with rasterio.open(
            path, 'w',
            driver='GTiff',
            height=TEST_SIZE,
            width=TEST_SIZE,
            count=1,
            dtype='uint8',
            crs='EPSG:4326',
            transform=transform,
            compress='LZW'  # Light compression for test data
        ) as dst:
            dst.write(data, 1)
        
        raster_paths.append(path)
    
    return raster_paths


def test_merge_with_gtiff(raster_paths, output_path):
    """Test merge with standard GTiff output."""
    print("\n=== Testing GTiff Merge ===")
    
    with measure_memory():
        start_time = time.time()
        
        # Open sources and create VRTs
        sources = [rasterio.open(p) for p in raster_paths]
        
        # Create VRTs for alignment
        vrts = []
        for src in sources:
            vrt = WarpedVRT(src, crs=src.crs)
            vrts.append(vrt)
        
        # Merge with GTiff output
        dst_kwds = {
            "driver": "GTiff",
            "tiled": True,
            "compress": "ZSTD",
            "zstd_level": 9,
            "blockxsize": 512,
            "blockysize": 512,
            "NUM_THREADS": "ALL_CPUS",
            "BIGTIFF": "IF_SAFER"
        }
        
        merge(
            vrts,
            method='first',
            dst_path=str(output_path),
            dst_kwds=dst_kwds
        )
        
        # Close sources
        for src in sources:
            src.close()
        
        elapsed = time.time() - start_time
        print(f"  Time: {elapsed:.2f} seconds")
        
    # Check output file size
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Output size: {size_mb:.2f} MB")
    
    return elapsed, size_mb


def test_merge_with_cog(raster_paths, output_path):
    """Test merge with COG output."""
    print("\n=== Testing COG Merge ===")
    
    with measure_memory():
        start_time = time.time()
        
        # Open sources and create VRTs
        sources = [rasterio.open(p) for p in raster_paths]
        
        # Create VRTs for alignment
        vrts = []
        for src in sources:
            vrt = WarpedVRT(src, crs=src.crs)
            vrts.append(vrt)
        
        # Merge with COG output
        dst_kwds = {
            "driver": "COG",
            "BLOCKSIZE": 512,  # COG uses BLOCKSIZE not blockxsize
            "COMPRESS": "ZSTD",
            "LEVEL": 9,  # COG uses LEVEL not zstd_level
            "NUM_THREADS": "ALL_CPUS",
            "BIGTIFF": "IF_SAFER",
            "OVERVIEW_RESAMPLING": "AVERAGE"
        }
        
        merge(
            vrts,
            method='first',
            dst_path=str(output_path),
            dst_kwds=dst_kwds
        )
        
        # Close sources
        for src in sources:
            src.close()
        
        elapsed = time.time() - start_time
        print(f"  Time: {elapsed:.2f} seconds")
        
    # Check output file size
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Output size: {size_mb:.2f} MB")
    
    return elapsed, size_mb


def test_cog_two_pass(raster_paths, output_path):
    """Test traditional two-pass COG creation (merge to GTiff, then convert)."""
    print("\n=== Testing Two-Pass COG Creation ===")
    
    with measure_memory():
        start_time = time.time()
        
        # First pass: merge to GTiff
        temp_tiff = output_path.parent / "temp_merge.tif"
        
        sources = [rasterio.open(p) for p in raster_paths]
        vrts = [WarpedVRT(src, crs=src.crs) for src in sources]
        
        dst_kwds = {
            "driver": "GTiff",
            "tiled": True,
            "compress": "LZW",  # Fast compression for temp file
            "blockxsize": 512,
            "blockysize": 512
        }
        
        merge(vrts, method='first', dst_path=str(temp_tiff), dst_kwds=dst_kwds)
        
        for src in sources:
            src.close()
        
        # Second pass: convert to COG
        from rasterio.shutil import copy
        with rasterio.open(temp_tiff) as src:
            copy(
                src, str(output_path),
                driver='COG',
                BLOCKSIZE=512,
                COMPRESS='ZSTD',
                LEVEL=9
            )
        
        # Clean up temp file
        temp_tiff.unlink()
        
        elapsed = time.time() - start_time
        print(f"  Time: {elapsed:.2f} seconds")
        
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  Output size: {size_mb:.2f} MB")
    
    return elapsed, size_mb


def main():
    """Run performance comparison tests."""
    print("=" * 60)
    print("COG vs GTiff Merge Performance Test")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test rasters
        print("\nCreating test rasters...")
        raster_paths = create_test_rasters(temp_path)
        print(f"Created {len(raster_paths)} test rasters of {TEST_SIZE}x{TEST_SIZE} pixels")
        
        # Test different approaches
        results = {}
        
        # 1. Standard GTiff
        output_gtiff = temp_path / "merged_gtiff.tif"
        time_gtiff, size_gtiff = test_merge_with_gtiff(raster_paths, output_gtiff)
        results['GTiff'] = {'time': time_gtiff, 'size': size_gtiff}
        
        # 2. Direct COG
        output_cog = temp_path / "merged_cog.tif"
        time_cog, size_cog = test_merge_with_cog(raster_paths, output_cog)
        results['COG'] = {'time': time_cog, 'size': size_cog}
        
        # 3. Two-pass COG
        output_twopass = temp_path / "merged_twopass.tif"
        time_twopass, size_twopass = test_cog_two_pass(raster_paths, output_twopass)
        results['TwoPass'] = {'time': time_twopass, 'size': size_twopass}
        
        # Print summary
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        
        print("\nTime Comparison:")
        baseline = results['GTiff']['time']
        for method, metrics in results.items():
            ratio = metrics['time'] / baseline
            print(f"  {method:10}: {metrics['time']:6.2f}s ({ratio:.2f}x)")
        
        print("\nSize Comparison:")
        for method, metrics in results.items():
            print(f"  {method:10}: {metrics['size']:6.2f} MB")
        
        print("\n" + "=" * 60)
        print("CONCLUSION")
        print("=" * 60)
        
        # Analyze results
        if results['COG']['time'] < results['GTiff']['time'] * 0.95:
            print("✅ COG is significantly faster (>5% improvement)")
        elif results['COG']['time'] > results['GTiff']['time'] * 1.05:
            print("❌ COG is significantly slower (>5% worse)")
        else:
            print("➖ COG and GTiff have similar performance (±5%)")
        
        print("\nKey Insights:")
        print("1. COG driver creates Cloud Optimized GeoTIFF in single pass")
        print("2. But it may do extra work for overview generation")
        print("3. Memory usage should be similar (both use tiled writing)")
        print("4. Two-pass approach adds overhead but ensures true COG format")


if __name__ == "__main__":
    main()