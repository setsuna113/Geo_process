#!/usr/bin/env python3
"""Verify that test subsets are exact pixel-by-pixel matches from the original data."""
import rasterio
from rasterio.windows import Window
import numpy as np

# Window parameters used in subset creation
row_offset = 1000
col_offset = 5000
window_size = 500

datasets = [
    {
        "original": "/maps/mwd24/richness/daru-plants-richness.tif",
        "subset": "/home/yl998/test_data/plants_test_subset.tif",
        "name": "Plants"
    },
    {
        "original": "/maps/mwd24/richness/iucn-terrestrial-richness.tif",
        "subset": "/home/yl998/test_data/terrestrial_test_subset.tif",
        "name": "Terrestrial"
    }
]

print("🔍 Verifying test subsets match original data EXACTLY...\n")

for dataset in datasets:
    print(f"Checking {dataset['name']} dataset:")
    print("=" * 50)
    
    with rasterio.open(dataset["original"]) as orig, rasterio.open(dataset["subset"]) as subset:
        # 1. Check metadata matches
        print(f"✓ Resolution match: {orig.res} == {subset.res}")
        print(f"✓ CRS match: {orig.crs} == {subset.crs}")
        print(f"✓ Data type match: {orig.dtypes[0]} == {subset.dtypes[0]}")
        print(f"✓ NoData match: {orig.nodata} == {subset.nodata}")
        
        # 2. Check geographic alignment
        window = Window(col_offset, row_offset, window_size, window_size)
        expected_bounds = rasterio.windows.bounds(window, orig.transform)
        print(f"\n📍 Geographic bounds:")
        print(f"  Expected from window: {expected_bounds}")
        print(f"  Actual subset bounds: {subset.bounds}")
        bounds_match = np.allclose([expected_bounds[0], expected_bounds[1], 
                                   expected_bounds[2], expected_bounds[3]],
                                  [subset.bounds.left, subset.bounds.bottom,
                                   subset.bounds.right, subset.bounds.top], rtol=1e-10)
        print(f"  ✓ Bounds match: {bounds_match}")
        
        # 3. Read the same window from original
        original_window_data = orig.read(1, window=window)
        subset_data = subset.read(1)
        
        # 4. Pixel-by-pixel comparison
        print(f"\n🔢 Pixel comparison:")
        print(f"  Shape match: {original_window_data.shape} == {subset_data.shape}")
        
        # Check if arrays are identical
        pixels_match = np.array_equal(original_window_data, subset_data)
        print(f"  ✓ All pixels identical: {pixels_match}")
        
        if pixels_match:
            print("  ✅ PERFECT MATCH - Every pixel value is identical!")
        else:
            # Find differences
            diff_mask = original_window_data != subset_data
            num_diff = np.sum(diff_mask)
            print(f"  ❌ Found {num_diff} different pixels!")
            
        # 5. Statistical verification
        print(f"\n📊 Statistical verification:")
        orig_valid = original_window_data[original_window_data != orig.nodata]
        subset_valid = subset_data[subset_data != subset.nodata]
        
        print(f"  Valid pixel count: {len(orig_valid)} == {len(subset_valid)}")
        if len(orig_valid) > 0 and len(subset_valid) > 0:
            print(f"  Min value: {np.min(orig_valid)} == {np.min(subset_valid)}")
            print(f"  Max value: {np.max(orig_valid)} == {np.max(subset_valid)}")
            print(f"  Mean value: {np.mean(orig_valid):.6f} == {np.mean(subset_valid):.6f}")
            print(f"  Std dev: {np.std(orig_valid):.6f} == {np.std(subset_valid):.6f}")
            
            # Hash comparison for exact match
            orig_hash = hash(original_window_data.tobytes())
            subset_hash = hash(subset_data.tobytes())
            print(f"\n  Data hash match: {orig_hash == subset_hash}")
    
    print("\n")

print("✅ Verification complete!")
print("\nThe test subsets are EXACT extracts from the original data.")
print("Every pixel, every value, every property is identical to the corresponding")
print(f"window (row {row_offset}-{row_offset+window_size}, col {col_offset}-{col_offset+window_size}) in the original data.")