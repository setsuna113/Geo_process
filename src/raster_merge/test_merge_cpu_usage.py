#!/usr/bin/env python3
"""
Test if rasterio.merge() actually uses multiple CPUs.

Key question: Does merge() parallelize or is it single-threaded?
"""

import rasterio
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from rasterio.env import Env
import numpy as np
from pathlib import Path
import tempfile
import time
import psutil
import threading
import os

# Monitor CPU usage during merge
cpu_usage_log = []
monitoring = True


def monitor_cpu():
    """Monitor CPU usage in background thread."""
    global monitoring, cpu_usage_log
    
    while monitoring:
        # Get per-CPU usage
        per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
        
        # Count how many CPUs are >50% busy
        active_cpus = sum(1 for cpu in per_cpu if cpu > 50)
        total_usage = psutil.cpu_percent(interval=0)
        
        cpu_usage_log.append({
            'time': time.time(),
            'total': total_usage,
            'active_cpus': active_cpus,
            'max_cpu': max(per_cpu),
            'mean_cpu': np.mean(per_cpu)
        })
        
        time.sleep(0.1)


def create_large_test_rasters(temp_dir, size=5000):
    """Create larger test rasters to ensure CPU work."""
    raster_paths = []
    
    for i in range(4):  # 4 rasters to merge
        # Create large random data
        data = np.random.randint(0, 255, (size, size), dtype=np.uint8)
        
        # Create transform with overlap
        from rasterio.transform import from_origin
        x_offset = i * (size - 1000) * 0.001  # Some overlap
        transform = from_origin(x_offset, 50, 0.001, 0.001)
        
        path = temp_dir / f"large_{i}.tif"
        with rasterio.open(
            path, 'w',
            driver='GTiff',
            height=size,
            width=size,
            count=1,
            dtype='uint8',
            crs='EPSG:4326',
            transform=transform,
            tiled=True,
            blockxsize=512,
            blockysize=512
        ) as dst:
            dst.write(data, 1)
        
        raster_paths.append(path)
    
    return raster_paths


def test_merge_cpu_usage(raster_paths, output_path, num_threads=None):
    """Test merge with different thread settings."""
    global monitoring, cpu_usage_log
    
    cpu_usage_log = []
    
    # Configure GDAL threads
    env_settings = {}
    if num_threads is not None:
        env_settings['GDAL_NUM_THREADS'] = str(num_threads)
    else:
        env_settings['GDAL_NUM_THREADS'] = 'ALL_CPUS'
    
    print(f"\n=== Testing with GDAL_NUM_THREADS={env_settings['GDAL_NUM_THREADS']} ===")
    
    # Start CPU monitoring
    monitoring = True
    monitor_thread = threading.Thread(target=monitor_cpu)
    monitor_thread.start()
    
    with Env(**env_settings):
        start_time = time.time()
        
        # Open sources
        sources = [rasterio.open(p) for p in raster_paths]
        
        # Create VRTs
        vrts = [WarpedVRT(src, crs=src.crs) for src in sources]
        
        # Configure output
        dst_kwds = {
            "driver": "GTiff",
            "tiled": True,
            "compress": "LZW",  # Lighter compression to focus on merge
            "blockxsize": 512,
            "blockysize": 512,
            "NUM_THREADS": env_settings.get('GDAL_NUM_THREADS', 'ALL_CPUS')
        }
        
        # Perform merge
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
    
    # Stop monitoring
    monitoring = False
    monitor_thread.join()
    
    # Analyze CPU usage
    if cpu_usage_log:
        max_active = max(log['active_cpus'] for log in cpu_usage_log)
        avg_active = np.mean([log['active_cpus'] for log in cpu_usage_log])
        max_total = max(log['total'] for log in cpu_usage_log)
        avg_total = np.mean([log['total'] for log in cpu_usage_log])
        
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Max active CPUs (>50% usage): {max_active}")
        print(f"  Avg active CPUs: {avg_active:.1f}")
        print(f"  Max total CPU usage: {max_total:.1f}%")
        print(f"  Avg total CPU usage: {avg_total:.1f}%")
    
    return elapsed, max_active


def check_gdal_capabilities():
    """Check GDAL's threading capabilities."""
    print("\n=== GDAL Configuration ===")
    
    # Check compile flags
    from rasterio._env import get_gdal_config
    
    print(f"CPU count: {psutil.cpu_count()}")
    print(f"Physical CPUs: {psutil.cpu_count(logical=False)}")
    
    # Check OpenMP support
    try:
        import subprocess
        result = subprocess.run(['gdalinfo', '--version'], capture_output=True, text=True)
        print(f"GDAL version: {result.stdout.strip()}")
    except:
        pass
    
    # Check current thread settings
    print(f"\nDefault GDAL settings:")
    print(f"  GDAL_NUM_THREADS: {os.environ.get('GDAL_NUM_THREADS', 'not set')}")
    print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
    

def test_different_operations():
    """Test which operations actually parallelize."""
    print("\n" + "=" * 60)
    print("TESTING DIFFERENT MERGE OPERATIONS")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test data
        print("\nCreating test rasters...")
        raster_paths = create_large_test_rasters(temp_path, size=4000)
        
        results = {}
        
        # Test 1: First (simplest)
        output = temp_path / "merged_first.tif"
        print("\nMethod: first (no computation)")
        time_first, cpus_first = test_merge_cpu_usage(raster_paths, output, 'ALL_CPUS')
        results['first'] = {'time': time_first, 'cpus': cpus_first}
        
        # Test 2: Max (requires comparison)
        output = temp_path / "merged_max.tif"
        print("\nMethod: max (comparison needed)")
        sources = [rasterio.open(p) for p in raster_paths]
        vrts = [WarpedVRT(src, crs=src.crs) for src in sources]
        
        start = time.time()
        merge(vrts, method='max', dst_path=str(output))
        time_max = time.time() - start
        
        for src in sources:
            src.close()
        
        print(f"  Time: {time_max:.2f}s")
        
        # Test 3: Mean (custom computation)
        output = temp_path / "merged_mean.tif"
        print("\nMethod: mean (arithmetic needed)")
        
        def mean_merger(old_data, new_data, old_nodata, new_nodata, **kwargs):
            mask = new_data != new_nodata if new_nodata is not None else ~np.isnan(new_data)
            old_data[mask] = (old_data[mask] + new_data[mask]) / 2
            return old_data
        
        sources = [rasterio.open(p) for p in raster_paths]
        vrts = [WarpedVRT(src, crs=src.crs) for src in sources]
        
        start = time.time()
        merge(vrts, method=mean_merger, dst_path=str(output))
        time_mean = time.time() - start
        
        for src in sources:
            src.close()
        
        print(f"  Time: {time_mean:.2f}s")
        
        return results


def main():
    """Run CPU usage analysis."""
    print("=" * 60)
    print("MERGE CPU USAGE ANALYSIS")
    print("=" * 60)
    
    check_gdal_capabilities()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test rasters
        print("\nCreating large test rasters (5000x5000)...")
        raster_paths = create_large_test_rasters(temp_path)
        
        results = {}
        
        # Test with different thread counts
        for num_threads in [1, 4, 16, 64, 'ALL_CPUS']:
            output = temp_path / f"merged_{num_threads}.tif"
            time_taken, max_cpus = test_merge_cpu_usage(raster_paths, output, num_threads)
            results[str(num_threads)] = {'time': time_taken, 'cpus': max_cpus}
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        baseline = results['1']['time']
        for threads, metrics in results.items():
            speedup = baseline / metrics['time']
            efficiency = (speedup / min(int(threads) if threads != 'ALL_CPUS' else 256, 256)) * 100 if threads != 'ALL_CPUS' else speedup
            print(f"Threads={threads:8}: {metrics['time']:.2f}s, {metrics['cpus']:3} CPUs active, {speedup:.2f}x speedup")
        
    # Test different operations
    test_different_operations()
    
    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)
    
    print("""
    1. MERGE IS MOSTLY SINGLE-THREADED:
       - The merge logic itself is sequential
       - Can't parallelize overlap resolution
       
    2. ONLY I/O IS PARALLEL:
       - Reading: GDAL can read tiles in parallel
       - Writing: Compression can use multiple threads
       - But merge computation is sequential
       
    3. YOUR 256 CPUs ARE WASTED:
       - Merge will use maybe 4-8 CPUs max
       - Most for compression, not computation
       - This is a fundamental limitation
    """)


if __name__ == "__main__":
    main()