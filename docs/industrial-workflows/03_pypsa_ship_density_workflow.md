# PyPSA-EUR Ship Density Raster Processing Workflow

## Repository Structure
```
PyPSA/pypsa-eur/
├── scripts/
│   ├── build_ship_raster.py         # Ship density raster processing
│   ├── build_renewable_profiles.py  # Wind/solar resource processing
│   ├── build_cutout.py             # Geographic extent definition
│   └── retrieve_cutout.py          # Data download utilities
├── envs/
│   └── environment.yaml            # Dependencies (includes rioxarray)
└── data/
    └── bundle/                     # Downloaded data storage
```

## 1. LOADING RASTERS

### Simple Raster Loading from ZIP
```python
# From scripts/build_ship_raster.py (Lines 60-75)
import zipfile
from pathlib import Path
import rioxarray

def load_ship_density_from_zip(zip_path: str) -> xarray.DataArray:
    """
    Load global ship density raster from World Bank dataset
    File is ~20GB uncompressed
    """
    # Extract from ZIP archive
    with zipfile.ZipFile(zip_path) as zip_f:
        resources = Path(snakemake.output[0]).parent
        fn = "shipdensity_global.tif"
        
        # Extract to temporary location
        zip_f.extract(fn, resources)
        
    # Load with rioxarray
    with rioxarray.open_rasterio(resources / fn) as ship_density:
        # Drop unnecessary band dimension
        ship_density = ship_density.drop_vars(["band"])
        
        # Keep as xarray DataArray for coordinate-based slicing
        return ship_density
```

### Cutout (Study Area) Definition
```python
# From scripts/build_cutout.py (conceptual - referenced in build_ship_raster.py)
def determine_cutout_xXyY(cutout):
    """
    Determine the geographic extent for European study area
    Returns bounds in the raster's coordinate system
    """
    # Cutout defines the study area (e.g., Europe)
    # Format: [min_lon, max_lon, min_lat, max_lat]
    bounds = cutout.bounds
    
    # Convert to raster coordinate system if needed
    # PyPSA uses standard lat/lon (EPSG:4326)
    x = bounds[0]  # min longitude
    X = bounds[1]  # max longitude  
    y = bounds[2]  # min latitude
    Y = bounds[3]  # max latitude
    
    return x, X, y, Y
```

## 2. SPATIAL SUBSETTING (NO MERGE NEEDED)

### Direct Coordinate-Based Slicing
```python
# From scripts/build_ship_raster.py (Lines 68-72)
def clip_to_european_extent(
    global_raster: xarray.DataArray,
    cutout
) -> xarray.DataArray:
    """
    Clip global raster to European extent
    This is NOT merging - just spatial subsetting
    """
    # Get European bounds
    x, X, y, Y = determine_cutout_xXyY(cutout)
    
    # Use xarray's coordinate-based slicing
    # This is extremely efficient - no resampling needed
    european_subset = global_raster.sel(
        x=slice(x, X),  # Longitude slice
        y=slice(Y, y)   # Latitude slice (note: Y to y because of coordinate direction)
    )
    
    return european_subset
```

## 3. NO ALIGNMENT OR RESAMPLING REQUIRED

```python
# PyPSA doesn't need alignment because:
# 1. Only processing ONE raster (no merge)
# 2. Keeping original resolution (1km global grid)
# 3. Just cropping to smaller area

# The beauty of this approach:
def process_ship_density_simple():
    """
    Entire processing pipeline in minimal code
    No merge, no resample, no alignment needed
    """
    # Load
    global_data = rioxarray.open_rasterio("shipdensity_global.tif")
    
    # Crop
    europe_data = global_data.sel(
        x=slice(-12, 35),  # Western Europe to Eastern Europe
        y=slice(72, 35)    # Scandinavia to Mediterranean
    )
    
    # Save
    europe_data.rio.to_raster("ship_density_europe.tif")
    
    # That's it! 3 lines for the entire workflow
```

## 4. EXPORT OPERATIONS

### Direct Export with rioxarray
```python
# From scripts/build_ship_raster.py (Line 73)
def export_cropped_raster(
    subset_data: xarray.DataArray,
    output_path: str
) -> None:
    """
    Export the cropped European ship density
    """
    # rioxarray makes this trivial
    subset_data.rio.to_raster(
        output_path,
        compress='deflate',  # Compression
        tiled=True,         # Tiled for GIS software
        dtype='float32'     # Maintain precision
    )
```

### Complete Workflow Implementation
```python
# From scripts/build_ship_raster.py (Lines 65-75)
if __name__ == "__main__":
    """
    Complete PyPSA ship density processing
    """
    # Input/output from Snakemake workflow
    input_zip = snakemake.input.ship_density
    output_raster = snakemake.output[0]
    cutout = snakemake.input.cutout
    
    # Step 1: Extract from ZIP
    with zipfile.ZipFile(input_zip) as zip_f:
        resources = Path(output_raster).parent
        fn = "shipdensity_global.tif"
        zip_f.extract(fn, resources)
    
    # Step 2: Load and crop in one operation
    with rioxarray.open_rasterio(resources / fn) as ship_density:
        # Remove band dimension
        ship_density = ship_density.drop_vars(["band"])
        
        # Get European bounds
        x, X, y, Y = determine_cutout_xXyY(cutout)
        
        # Crop to Europe
        ship_density_europe = ship_density.sel(
            x=slice(x, X),
            y=slice(Y, y)
        )
        
        # Step 3: Export
        ship_density_europe.rio.to_raster(output_raster)
    
    # Step 4: Clean up temporary file
    (resources / fn).unlink()  # Delete extracted global file
```

## 5. INTEGRATION WITH ENERGY MODELING

### How Ship Density Affects Wind Farm Planning
```python
# Conceptual usage in PyPSA energy modeling
def exclude_shipping_lanes_from_offshore_wind(
    potential_sites: geopandas.GeoDataFrame,
    ship_density_path: str,
    threshold: float = 100  # ships per km² per year
) -> geopandas.GeoDataFrame:
    """
    Remove potential wind farm sites that conflict with shipping
    """
    # Load ship density for Europe
    ship_density = rioxarray.open_rasterio(ship_density_path)
    
    # For each potential wind farm site
    valid_sites = []
    for idx, site in potential_sites.iterrows():
        # Sample ship density at site location
        density = ship_density.sel(
            x=site.geometry.x,
            y=site.geometry.y,
            method='nearest'
        ).values
        
        # Keep site if shipping traffic is low
        if density < threshold:
            valid_sites.append(site)
    
    return geopandas.GeoDataFrame(valid_sites)
```

## KEY CHARACTERISTICS

1. **No Merging**: Only processing single raster
2. **No Resampling**: Keeping original 1km resolution
3. **No Alignment**: Single source, no coordination needed
4. **Simple Slicing**: Coordinate-based subsetting
5. **Minimal Code**: Entire workflow in <20 lines
6. **Memory Efficient**: Never loads full 20GB global file

## MEMORY OPTIMIZATION

```python
# From scripts/build_ship_raster.py
# Smart memory handling with context managers

def memory_efficient_processing():
    """
    Process without loading entire dataset
    """
    # Use context manager - auto cleanup
    with rioxarray.open_rasterio("shipdensity_global.tif") as src:
        # Slice BEFORE loading to memory
        # This only reads the required portion
        europe = src.sel(x=slice(-12, 35), y=slice(72, 35))
        
        # Now load just European data
        europe.rio.to_raster("europe.tif")
    # src automatically closed and memory freed
```

## COMPARISON WITH OTHER WORKFLOWS

| Aspect | PyPSA | FarmVibes | NOAA |
|--------|-------|-----------|------|
| Input Files | 1 | Multiple | Multiple |
| Operation | Crop | Merge | Correct |
| Resampling | No | Yes | No |
| Alignment | No | Yes | Yes (point-to-raster) |
| Output | Subset | Mosaic | Difference |
| Code Lines | ~15 | ~200 | ~400 |

## COMPLETE MINIMAL EXAMPLE

```python
#!/usr/bin/env python3
"""
Minimal PyPSA ship density workflow
Total processing in 10 lines
"""

import rioxarray
import zipfile
from pathlib import Path

# Extract
with zipfile.ZipFile("ship_density.zip") as z:
    z.extract("shipdensity_global.tif", ".")

# Load, crop, save
with rioxarray.open_rasterio("shipdensity_global.tif") as src:
    europe = src.sel(x=slice(-12, 35), y=slice(72, 35))
    europe.rio.to_raster("ship_density_europe.tif")

# Cleanup
Path("shipdensity_global.tif").unlink()
```

## WHY THIS APPROACH WORKS

1. **Single Source**: No coordination complexity
2. **Regular Grid**: Ship density already in lat/lon grid
3. **One-Time Processing**: Not operational/streaming
4. **Simple Requirement**: Just need European subset
5. **Coordinate Slicing**: xarray makes this trivial

## LESSONS FOR GENERAL WORKFLOWS

1. **Avoid Complexity**: If you don't need merge, don't merge
2. **Use Coordinates**: xarray slicing is powerful
3. **Context Managers**: Automatic memory cleanup
4. **Lazy Loading**: Slice before loading
5. **Original Resolution**: Resampling loses information