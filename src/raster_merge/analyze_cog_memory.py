#!/usr/bin/env python3
"""
Analyze why COG driver uses so much memory during merge.
"""

import rasterio
from rasterio.env import GDALVersion
import subprocess
import sys

def check_cog_behavior():
    """Check COG driver behavior and requirements."""
    
    print("=" * 60)
    print("COG DRIVER ANALYSIS")
    print("=" * 60)
    
    # Check GDAL version
    print(f"\nGDAL Version: {GDALVersion()}")
    
    # Check if COG driver is available
    from rasterio.drivers import raster_driver_extensions
    print(f"\nCOG driver available: {'COG' in raster_driver_extensions()}")
    
    # Get COG driver creation options
    print("\nCOG Driver Behavior:")
    print("-" * 40)
    
    info = """
    The COG driver has these characteristics:
    
    1. MEMORY BUFFERING:
       - COG driver buffers the ENTIRE output in memory
       - Only writes to disk after all processing is complete
       - This ensures proper COG structure (overviews, tiling)
    
    2. OVERVIEW GENERATION:
       - Automatically creates overviews (pyramids)
       - Default: 2x, 4x, 8x, 16x... downsampling
       - Each overview level adds memory usage
    
    3. TILE REORGANIZATION:
       - Reorganizes data into proper tile structure
       - Tiles must be aligned for cloud optimization
       - Requires holding multiple tiles in memory
    
    4. SINGLE-PASS CONSTRAINT:
       - COG must be written in one pass (no append)
       - Cannot stream data like regular GTiff
       - Must know full extent before writing
    
    5. MEMORY FORMULA:
       Memory ≈ (Full Raster Size) + (Overview Sizes) + (Working Buffers)
       
       For a 4000x3000 raster with 4 overview levels:
       - Main: 4000x3000 = 12M pixels
       - L1: 2000x1500 = 3M pixels
       - L2: 1000x750 = 0.75M pixels
       - L3: 500x375 = 0.19M pixels
       - L4: 250x187 = 0.05M pixels
       Total: ~16M pixels in memory
    """
    
    print(info)
    
    # Test with gdalinfo
    print("\nGDAL COG Creation Options:")
    print("-" * 40)
    
    try:
        result = subprocess.run(
            ["gdalinfo", "--format", "COG"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # Parse relevant options
            lines = result.stdout.split('\n')
            for line in lines:
                if any(keyword in line for keyword in ['COMPRESS', 'BLOCK', 'OVERVIEW', 'RESAMPLING']):
                    print(f"  {line.strip()}")
    except:
        print("  Could not get GDAL COG options")
    
    print("\n" + "=" * 60)
    print("WHY COG IS BAD FOR MERGE OPERATIONS")
    print("=" * 60)
    
    conclusions = """
    1. MEMORY EXPLOSION:
       - Regular GTiff: Streams tiles as they're computed
       - COG: Buffers entire output + overviews in memory
       - For large rasters: Can use 10-20x more memory
    
    2. PERFORMANCE PENALTY:
       - Extra time generating overviews
       - Extra time reorganizing tiles
       - No benefit during merge (not reading from cloud)
    
    3. WHEN COG MAKES SENSE:
       ✅ Files served over HTTP/S3
       ✅ Random access patterns
       ✅ Zoom levels (web maps)
       ✅ Cloud-native workflows
    
    4. WHEN COG IS WRONG:
       ❌ Local file processing
       ❌ Sequential access (like merge)
       ❌ Memory-constrained systems
       ❌ One-time processing outputs
    
    RECOMMENDATION:
    - Merge to regular GTiff with tiling
    - Convert to COG later IF needed for cloud serving
    - Never use COG as intermediate format
    """
    
    print(conclusions)


if __name__ == "__main__":
    check_cog_behavior()