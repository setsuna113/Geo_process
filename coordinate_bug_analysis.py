#!/usr/bin/env python3
"""Analyze the exact coordinate alignment bug in merge stage."""

import numpy as np

def analyze_alignment_bug():
    """Show exactly how the alignment calculation fails."""
    
    print("=== COORDINATE ALIGNMENT BUG ANALYSIS ===\n")
    
    # Dataset bounds from our analysis
    terrestrial_bounds = (-180.0, -54.996, 180.000, 83.004)
    plants_bounds = (-180.0, -89.833, 180.000, 83.667)
    resolution = 0.016666666666667
    
    print("1. DATASET BOUNDS:")
    print(f"   Terrestrial: {terrestrial_bounds}")
    print(f"   Plants: {plants_bounds}")
    print(f"   Resolution: {resolution} degrees\n")
    
    # Common bounds (union of all datasets)
    common_bounds = (
        min(terrestrial_bounds[0], plants_bounds[0]),  # min x
        min(terrestrial_bounds[1], plants_bounds[1]),  # min y  
        max(terrestrial_bounds[2], plants_bounds[2]),  # max x
        max(terrestrial_bounds[3], plants_bounds[3])   # max y
    )
    print(f"2. COMMON BOUNDS (union): {common_bounds}\n")
    
    # The buggy alignment calculation from merge_stage.py
    print("3. BUGGY OFFSET CALCULATION:")
    
    # For terrestrial
    terr_x_offset = int(np.round((terrestrial_bounds[0] - common_bounds[0]) / resolution))
    terr_y_offset = int(np.round((common_bounds[3] - terrestrial_bounds[3]) / resolution))
    print(f"   Terrestrial offsets: x={terr_x_offset}, y={terr_y_offset}")
    
    # For plants
    plants_x_offset = int(np.round((plants_bounds[0] - common_bounds[0]) / resolution))
    plants_y_offset = int(np.round((common_bounds[3] - plants_bounds[3]) / resolution))
    print(f"   Plants offsets: x={plants_x_offset}, y={plants_y_offset}")
    
    # The problem
    print(f"\n4. THE BUG:")
    print(f"   Y-offset difference: {terr_y_offset} - {plants_y_offset} = {terr_y_offset - plants_y_offset}")
    print(f"   This means terrestrial data is shifted by {terr_y_offset - plants_y_offset} pixels vertically!")
    
    # Test coordinate
    test_lat, test_lon = 20.6417, 106.2917
    
    # Calculate pixel positions in each dataset
    print(f"\n5. IMPACT ON TEST COORDINATE ({test_lat}, {test_lon}):")
    
    # In original terrestrial grid
    terr_row = int((terrestrial_bounds[3] - test_lat) / resolution)
    terr_col = int((test_lon - terrestrial_bounds[0]) / resolution)
    print(f"   In terrestrial grid: row={terr_row}, col={terr_col}")
    
    # In merged grid (with wrong offset)
    merged_row = terr_row + terr_y_offset
    print(f"   In merged grid: row={merged_row} (shifted by {terr_y_offset})")
    
    # What row this actually reads from
    actual_terr_row = merged_row - terr_y_offset
    print(f"   Actually reading from: row={actual_terr_row}")
    
    print("\n6. ROOT CAUSE:")
    print("   - Different datasets have different spatial extents")
    print("   - The merge uses common bounds (largest extent)")
    print("   - But the offset calculation doesn't account for this properly")
    print("   - Result: terrestrial data is read from wrong pixels")

if __name__ == "__main__":
    analyze_alignment_bug()