# Microsoft FarmVibes-AI Raster Processing Workflow

## Repository Structure
```
microsoft/farmvibes-ai/
├── ops/
│   ├── merge_rasters/           # Generic raster merging
│   │   └── merge_rasters.py
│   ├── merge_sentinel2_orbits/  # Sentinel-2 specific merging
│   │   └── merge_sentinel2_orbits.py
│   ├── merge_sentinel1_orbits/  # Sentinel-1 SAR merging
│   │   └── merge_sentinel1.py
│   ├── chunk_raster/            # Raster chunking for parallel processing
│   │   └── chunk_raster.py
│   └── download_cdl_data/       # Cropland data layer download & merge
│       └── download_cdl_data.py
└── vibe_lib/
    └── raster.py                # Core raster utilities

```

## 1. LOADING RASTERS

### Basic Loading Pattern
```python
# From ops/chunk_raster/chunk_raster.py (Line 50)
import rioxarray

# Simple loading with rioxarray
raster_data = rioxarray.open_rasterio(
    rasters[0].raster_asset.path_or_url,  # Can be local path or URL
    masked=True,  # Handle nodata values
    chunks={'band': 1, 'x': 2048, 'y': 2048}  # Dask chunking for memory efficiency
)
```

### Loading Multiple Rasters for Merging
```python
# From ops/merge_rasters/merge_rasters.py
import rasterio
from typing import List, Tuple

def load_rasters_for_merge(raster_paths: List[str]) -> List[rasterio.DatasetReader]:
    """Load multiple rasters ensuring they're compatible for merging"""
    rasters = []
    for path in raster_paths:
        src = rasterio.open(path)
        rasters.append(src)
    return rasters
```

## 2. ALIGNMENT AND RESAMPLING

### CRS and Resolution Validation
```python
# From ops/merge_rasters/merge_rasters.py
def validate_rasters_compatibility(rasters: List[rasterio.DatasetReader]) -> None:
    """Ensure all rasters have same CRS before merging"""
    crs_set = set()
    dtype_set = set()
    
    for raster in rasters:
        crs_set.add(raster.crs.to_string())
        dtype_set.add(raster.dtypes[0])
    
    if len(crs_set) > 1:
        raise ValueError(f"All rasters must have the same CRS. Found: {crs_set}")
    
    if len(dtype_set) > 1:
        # Check if mixing int and float
        if any('int' in str(dt) for dt in dtype_set) and any('float' in str(dt) for dt in dtype_set):
            raise ValueError("Cannot mix integer and float rasters")
```

### Resolution Strategy Selection
```python
# From ops/merge_rasters/merge_rasters.py
from rasterio.enums import Resampling

def determine_target_resolution(
    rasters: List[rasterio.DatasetReader], 
    resolution_strategy: str
) -> Tuple[float, float]:
    """
    Compute target resolution based on strategy:
    - 'equal': All must be same (fails if different)
    - 'average': Mean of all resolutions
    - 'lowest': Minimum resolution (finest detail)
    - 'highest': Maximum resolution (coarsest)
    """
    resolutions = [(r.res[0], r.res[1]) for r in rasters]
    
    if resolution_strategy == "equal":
        if len(set(resolutions)) > 1:
            raise ValueError(f"Resolutions must be equal. Found: {resolutions}")
        return resolutions[0]
    
    elif resolution_strategy == "average":
        avg_x = sum(r[0] for r in resolutions) / len(resolutions)
        avg_y = sum(r[1] for r in resolutions) / len(resolutions)
        return (avg_x, avg_y)
    
    elif resolution_strategy == "lowest":
        return (
            min(r[0] for r in resolutions),
            min(r[1] for r in resolutions)
        )
    
    elif resolution_strategy == "highest":
        return (
            max(r[0] for r in resolutions),
            max(r[1] for r in resolutions)
        )
```

### Virtual Warping for Alignment (Sentinel-1 specific)
```python
# From ops/merge_sentinel1_orbits/merge_sentinel1.py (Lines 80-100)
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

def align_to_utm(raster: rasterio.DatasetReader, utm_crs) -> WarpedVRT:
    """Create virtual aligned raster without loading data"""
    return WarpedVRT(
        raster,
        crs=utm_crs,
        resampling=Resampling.bilinear,  # Smooth resampling for continuous data
        # Transform and dimensions calculated automatically
    )

# Usage pattern
aligned_vrts = []
for raster_path in raster_paths:
    with rasterio.open(raster_path) as src:
        # Determine appropriate UTM zone
        utm_crs = tile_to_utm(src.bounds, src.crs)
        
        # Create virtual aligned raster
        vrt = WarpedVRT(
            src,
            crs=utm_crs,
            resampling=Resampling.bilinear
        )
        aligned_vrts.append(vrt)
```

## 3. MERGING RASTERS

### Core Merge Implementation
```python
# From ops/merge_rasters/merge_rasters.py
from rasterio.merge import merge
import numpy as np

def merge_rasters(
    raster_sequence: RasterSequence,
    output_dir: str,
    resampling: Resampling = Resampling.nearest,
    resolution: str = "average"
) -> Dict[str, Raster]:
    """
    Main merge function combining multiple rasters
    """
    # Open all rasters
    rasters = [rasterio.open(r.raster_asset.path_or_url) for r in raster_sequence.rasters]
    
    # Validate compatibility
    validate_rasters_compatibility(rasters)
    
    # Determine target resolution
    target_res = determine_target_resolution(rasters, resolution)
    
    # Perform merge with rasterio
    merged_array, merged_transform = merge(
        rasters,
        res=target_res,
        resampling=resampling,
        method='first',  # How to handle overlaps: 'first', 'last', 'min', 'max', 'mean'
        target_aligned_pixels=True,  # Align to pixel grid
        dst_path=None,  # In-memory merge
        dst_kwds=None
    )
    
    # Get output metadata from first raster
    out_meta = rasters[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": merged_array.shape[1],
        "width": merged_array.shape[2],
        "transform": merged_transform,
        "compress": "deflate",  # Compression for output
        "tiled": True,  # Tiled for efficient access
        "blockxsize": 512,
        "blockysize": 512
    })
    
    # Handle data type specific compression
    if 'int' in str(out_meta['dtype']):
        out_meta.update(INT_COMPRESSION_KWARGS)
    else:
        out_meta.update(FLOAT_COMPRESSION_KWARGS)
    
    return merged_array, out_meta
```

### Specialized Sentinel-2 Orbit Merging
```python
# From ops/merge_sentinel2_orbits/merge_sentinel2_orbits.py
from rasterio.merge import merge

def merge_sentinel2_orbits(orbit_group: Sentinel2CloudMaskOrbitGroup) -> Sentinel2CloudMask:
    """
    Merge multiple Sentinel-2 orbits covering same area
    Special handling for cloud masks
    """
    rasters = orbit_group.rasters
    
    # Sort by acquisition time (newest first)
    rasters = sorted(rasters, key=lambda x: x.time_range[0], reverse=True)
    
    # Open all rasters
    src_rasters = [rasterio.open(r.raster_asset.path_or_url) for r in rasters]
    
    # Merge with 'first' method (prefer newest data)
    merged, transform = merge(
        src_rasters,
        method='first',  # Use first valid pixel (from newest image)
        target_aligned_pixels=True,
        dst_path=None
    )
    
    # Close source rasters
    for src in src_rasters:
        src.close()
    
    return merged, transform
```

## 4. MEMORY OPTIMIZATION - CHUNKING STRATEGY

### Raster Chunking for Large Files
```python
# From ops/chunk_raster/chunk_raster.py
from typing import List, Tuple

def make_chunks(
    shape: Tuple[int, int, int],  # (bands, height, width)
    step_y: int,
    step_x: int,
    rasters: List[Raster]
) -> List[RasterChunk]:
    """
    Split large rasters into chunks for parallel processing
    """
    _, height, width = shape
    chunks = []
    
    for y in range(0, height, step_y):
        for x in range(0, width, step_x):
            # Calculate chunk boundaries
            y_end = min(y + step_y, height)
            x_end = min(x + step_x, width)
            
            # Create chunk metadata
            chunk = RasterChunk(
                limits=[y, y_end, x, x_end],
                rasters=rasters,
                geometry=calculate_chunk_geometry(y, y_end, x, x_end, rasters[0])
            )
            chunks.append(chunk)
    
    return chunks

# Usage for parallel processing
chunks = make_chunks(
    shape=(1, 10000, 10000),  # 10k x 10k raster
    step_y=2048,  # Process in 2048x2048 chunks
    step_x=2048,
    rasters=input_rasters
)

# Process chunks in parallel
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=8) as executor:
    results = executor.map(process_chunk, chunks)
```

## 5. EXPORT OPERATIONS

### Basic Export
```python
# From ops/merge_rasters/merge_rasters.py
def export_merged_raster(
    merged_array: np.ndarray,
    output_meta: dict,
    output_path: str
) -> None:
    """Export merged raster to GeoTIFF"""
    with rasterio.open(output_path, 'w', **output_meta) as dst:
        if len(merged_array.shape) == 3:
            # Multi-band
            for band_idx in range(merged_array.shape[0]):
                dst.write(merged_array[band_idx], band_idx + 1)
        else:
            # Single band
            dst.write(merged_array, 1)
        
        # Add overviews for pyramids (fast zoom)
        dst.build_overviews([2, 4, 8, 16], Resampling.average)
        dst.update_tags(ns='rio_overview', resampling='average')
```

### Cloud-Optimized GeoTIFF (COG) Export
```python
# From vibe_lib/raster.py
FLOAT_COMPRESSION_KWARGS = {
    "compress": "deflate",
    "predictor": 3,  # Floating point predictor
    "zlevel": 9,  # Maximum compression
    "tiled": True,
    "blockxsize": 512,
    "blockysize": 512,
    "interleave": "band"
}

INT_COMPRESSION_KWARGS = {
    "compress": "deflate", 
    "predictor": 2,  # Horizontal differencing
    "zlevel": 9,
    "tiled": True,
    "blockxsize": 512,
    "blockysize": 512,
    "interleave": "pixel"
}

def export_as_cog(
    data: np.ndarray,
    transform,
    crs,
    output_path: str,
    dtype: str
) -> None:
    """Export as Cloud-Optimized GeoTIFF"""
    profile = {
        'driver': 'GTiff',
        'dtype': dtype,
        'width': data.shape[-1],
        'height': data.shape[-2],
        'count': 1 if len(data.shape) == 2 else data.shape[0],
        'crs': crs,
        'transform': transform,
        'BIGTIFF': 'IF_SAFER',  # Use BigTIFF for large files
        'COPY_SRC_OVERVIEWS': True,
        'COMPRESS': 'DEFLATE',
        'TILED': True
    }
    
    # Add appropriate compression
    if 'int' in dtype:
        profile.update(INT_COMPRESSION_KWARGS)
    else:
        profile.update(FLOAT_COMPRESSION_KWARGS)
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data)
        # Build overviews
        factors = [2, 4, 8, 16, 32]
        dst.build_overviews(factors, Resampling.average)
```

## KEY OPTIMIZATIONS

1. **Virtual Rasters (VRT)**: Use WarpedVRT to align without loading data
2. **Chunked Processing**: Split large rasters into 2048x2048 chunks
3. **Parallel Execution**: ProcessPoolExecutor for chunk processing
4. **Compression**: Separate strategies for int vs float data
5. **Overviews**: Pyramid levels for fast zoom operations
6. **Method Selection**: 'first', 'last', 'mean' for overlap handling
7. **Memory Management**: Never load full raster when chunking available
8. **Direct URL Access**: Can read from Azure Blob URLs directly

## COMPLETE EXAMPLE WORKFLOW

```python
# Complete FarmVibes workflow
from ops.merge_rasters import merge_rasters
from rasterio.enums import Resampling

# Input: Multiple Sentinel-2 images from different dates/orbits
raster_sequence = RasterSequence(
    rasters=[
        Raster(path="s2_orbit1_20230601.tif"),
        Raster(path="s2_orbit2_20230601.tif"),
        Raster(path="s2_orbit1_20230608.tif")
    ]
)

# Merge with average resolution strategy
result = merge_rasters(
    raster_sequence=raster_sequence,
    output_dir="/output",
    resampling=Resampling.bilinear,  # Smooth for continuous data
    resolution="average"  # Use mean resolution of all inputs
)

# Output: Single merged COG file optimized for cloud storage
```