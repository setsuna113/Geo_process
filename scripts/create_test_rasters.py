#!/usr/bin/env python3
"""Create test raster files for pipeline testing."""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def create_test_raster(output_path: Path, 
                      bounds: tuple, 
                      resolution: float, 
                      value_range: tuple,
                      seed: int = None):
    """
    Create a test raster file with specified properties.
    
    Args:
        output_path: Output file path
        bounds: (west, south, east, north) in degrees
        resolution: Pixel size in degrees
        value_range: (min_value, max_value) for random data
        seed: Random seed for reproducibility
    """
    west, south, east, north = bounds
    
    # Calculate dimensions
    width = int((east - west) / resolution)
    height = int((north - south) / resolution)
    
    print(f"Creating raster: {output_path.name}")
    print(f"  Bounds: {bounds}")
    print(f"  Resolution: {resolution}° ({resolution * 111} km at equator)")
    print(f"  Dimensions: {width} x {height}")
    
    # Create transform
    transform = from_bounds(west, south, east, north, width, height)
    
    # Generate data - use seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Create realistic richness data (integer counts)
    min_val, max_val = value_range
    data = np.random.randint(min_val, max_val + 1, size=(height, width), dtype=np.int32)
    
    # Add some spatial autocorrelation to make it more realistic
    # Apply gaussian smoothing
    from scipy.ndimage import gaussian_filter
    data = gaussian_filter(data.astype(float), sigma=2.0)
    data = np.clip(data, min_val, max_val).astype(np.int32)
    
    # Create raster
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype='int32',
        crs=CRS.from_epsg(4326),
        transform=transform,
        compress='deflate'
    ) as dst:
        dst.write(data, 1)
        
        # Add metadata
        dst.update_tags(
            description=f"Test richness data for {output_path.stem}",
            units="species_count",
            min_value=int(data.min()),
            max_value=int(data.max()),
            mean_value=float(data.mean())
        )
    
    print(f"  Data range: {data.min()} - {data.max()} (mean: {data.mean():.1f})")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    print()

def main():
    """Create test rasters with overlapping bounds."""
    
    # Create test data directory
    test_dir = project_root / "test_data" / "rasters"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print("🌍 Creating Test Rasters for Pipeline Testing")
    print("=" * 50)
    print(f"Output directory: {test_dir}")
    print()
    
    # Define test rasters with overlapping bounds
    # Using small areas around different regions
    test_rasters = [
        {
            "name": "test-plants-richness.tif",
            "bounds": (-10.0, 40.0, -5.0, 45.0),  # Western Europe
            "resolution": 0.1,  # ~11km
            "value_range": (0, 150),  # Plant species counts
            "seed": 42
        },
        {
            "name": "test-birds-richness.tif", 
            "bounds": (-8.0, 42.0, -3.0, 47.0),  # Overlaps with plants
            "resolution": 0.1,
            "value_range": (0, 80),  # Bird species counts
            "seed": 43
        },
        {
            "name": "test-mammals-richness.tif",
            "bounds": (-12.0, 38.0, -7.0, 43.0),  # Partial overlap
            "resolution": 0.1,
            "value_range": (0, 50),  # Mammal species counts
            "seed": 44
        },
        {
            "name": "test-amphibians-richness.tif",
            "bounds": (-9.0, 41.0, -4.0, 46.0),  # Central overlap
            "resolution": 0.1,
            "value_range": (0, 30),  # Amphibian species counts
            "seed": 45
        }
    ]
    
    # Create each raster
    for raster_config in test_rasters:
        output_path = test_dir / raster_config["name"]
        create_test_raster(
            output_path=output_path,
            bounds=raster_config["bounds"],
            resolution=raster_config["resolution"],
            value_range=raster_config["value_range"],
            seed=raster_config["seed"]
        )
    
    # Calculate overlap region
    all_bounds = [r["bounds"] for r in test_rasters]
    overlap_west = max(b[0] for b in all_bounds)
    overlap_south = max(b[1] for b in all_bounds)
    overlap_east = min(b[2] for b in all_bounds)
    overlap_north = min(b[3] for b in all_bounds)
    
    print("📊 Summary:")
    print(f"Created {len(test_rasters)} test rasters")
    print(f"Overlap region: ({overlap_west}, {overlap_south}, {overlap_east}, {overlap_north})")
    print(f"Overlap area: {(overlap_east - overlap_west) * (overlap_north - overlap_south):.1f} deg²")
    print()
    print("✅ Test rasters created successfully!")
    print()
    print("Next steps:")
    print("1. Update config.yml to point to these test files")
    print("2. Run: ./run_pipeline.sh --experiment test_integration")

if __name__ == "__main__":
    main()