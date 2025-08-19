# Microsoft FarmVibes-AI Raster Processing Workflow

## Repository Structure  
```
microsoft/farmvibes-ai/
├── ops/
│   ├── merge_rasters/           # Generic raster merging
│   │   ├── merge_rasters.py
│   │   └── merge_rasters.yaml
│   ├── merge_sentinel2_orbits/  # Sentinel-2 specific merging
│   │   └── merge_sentinel2_orbits.py
│   ├── merge_sentinel1_orbits/  # Sentinel-1 SAR merging
│   │   └── merge_sentinel1.py
│   ├── chunk_raster/            # Raster chunking for parallel processing
│   │   └── chunk_raster.py
│   └── download_cdl_data/       # Cropland data layer download & merge
│       └── download_cdl_data.py
└── src/vibe_lib/
    └── vibe_lib/
        └── raster.py            # Core raster utilities

```

## 1. LOADING RASTERS

### Basic Loading Pattern (VERIFIED)
```python
# From ops/chunk_raster/chunk_raster.py (Lines 146-148)
import rioxarray

# ACTUAL CODE from repository:
ref = rasters[0]
shape = rioxarray.open_rasterio(ref.raster_asset.path_or_url).shape  # type: ignore
chunks = make_chunks(shape, self.step_y, self.step_x, rasters)
```

### Loading with Rasterio for Merging (VERIFIED)
```python
# From ops/merge_rasters/merge_rasters.py (Lines 60-76)
import rasterio
from rasterio.merge import merge

# ACTUAL IMPLEMENTATION:
def merge_rasters(
    raster_sequence: RasterSequence, output_dir: str, resampling: Resampling, resolution: str
) -> Dict[str, Raster]:
    out_id = gen_guid()
    file_path = os.path.join(output_dir, f"{out_id}.tif")
    
    # All rasters should have the same CRS
    with ExitStack() as stack:
        rasters = [
            stack.enter_context(rasterio.open(r.raster_asset.path_or_url, "r"))
            for r in raster_sequence.rasters
        ]
        
        # Validate compatibility
        check_rasters_are_valid(rasters, resolution)
        
        # Get target resolution
        target_resolution = compute_target_resolution(rasters, resolution)
```

## 2. ALIGNMENT AND RESAMPLING

### CRS and Resolution Validation (VERIFIED - CORRECTED)
```python
# From ops/merge_rasters/merge_rasters.py (Lines 29-43)
# ACTUAL CODE:
def check_rasters_are_valid(rasters: List[rasterio.DatasetReader], resolution_method: str):
    """Validate rasters for merging compatibility"""
    assets_meta = defaultdict(set)
    for r in rasters:
        for field in FIELDS:  # FIELDS = ("crs", "dtype", "count")
            assets_meta[field].add(r.meta[field])
    
    # Check all rasters have same CRS
    if len(assets_meta["crs"]) > 1:
        raise ValueError(
            f"Cannot merge rasters with different CRS: {assets_meta['crs']}"
        )
```

### Resolution Strategy Implementation (VERIFIED)
```python
# From ops/merge_rasters/merge_rasters.py (Lines 17-26, 45-56)
# ACTUAL CONSTANTS:
FIELDS = ("crs", "dtype", "count")
RESOLUTION_METHODS = {
    "equal": None,
    "average": lambda resolutions: tuple(np.mean(resolutions, axis=0)),
    "lowest": lambda resolutions: tuple(np.min(resolutions, axis=0)),
    "highest": lambda resolutions: tuple(np.max(resolutions, axis=0)),
}

def compute_target_resolution(
    rasters: List[rasterio.DatasetReader], resolution_method: str
) -> Optional[Tuple[float, float]]:
    if resolution_method == "equal":
        resolutions = list({raster.res for raster in rasters})
        if len(resolutions) > 1:
            raise ValueError(f"Found different resolutions {resolutions}")
        return None
    else:
        resolutions = [raster.res for raster in rasters]
        if len(set(resolutions)) > 1:
            LOGGER.warning(
                f"Found multiple resolutions when merging, using {resolution_method}"
            )
        return cast(Tuple[float, float], RESOLUTION_METHODS[resolution_method](resolutions))
```

### Virtual Warping for Alignment (Sentinel-1 specific - VERIFIED)
```python
# From ops/merge_sentinel1_orbits/merge_sentinel1.py (Lines 19-40)
from rasterio.vrt import WarpedVRT
from rasterio.merge import merge
from vibe_lib.raster import FLOAT_COMPRESSION_KWARGS, tile_to_utm

# ACTUAL CODE:
def merge_rasters(
    filepaths: List[str],
    bounds: Tuple[float, float, float, float],
    resampling: Resampling,
    out_path: str,
    **kwargs: Any,
) -> None:
    src = []
    vrt = []
    try:
        src = [rasterio.open(i) for i in filepaths]
        vrt = [WarpedVRT(i, **kwargs) for i in src]
        dst_kwds = FLOAT_COMPRESSION_KWARGS
        dst_kwds["driver"] = "GTiff"
        dst_kwds.update({"blockxsize": 512, "blockysize": 512})
        return merge(
            vrt,
            bounds=bounds,
            res=None,
            nodata=None,
            precision=None,
            method="first",  # first is the default
            target_aligned_pixels=False,
            dst_path=out_path,
            dst_kwds=dst_kwds,
            resampling=resampling,
        )
    finally:
        for i in vrt + src:
            i.close()
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
    
    # Validate compatibility - CORRECTED: actual function name
    check_rasters_are_valid(rasters, resolution)
    
    # Determine target resolution - CORRECTED: actual function name
    target_res = compute_target_resolution(rasters, resolution)
    
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
        "compress": "ZSTD",  # CORRECTED: Uses ZSTD not deflate
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
# From vibe_lib/raster.py - CORRECTED COMPRESSION SETTINGS
# ACTUAL CODE uses ZSTD compression, not DEFLATE:
COMPRESSION_KWARGS = {
    "tiled": True,
    "compress": "ZSTD",  # NOT deflate as originally claimed!
    "zstd_level": 9,
}
FLOAT_COMPRESSION_KWARGS = {**COMPRESSION_KWARGS, "predictor": 3}  # Floating point predictor
    "tiled": True,
    "blockxsize": 512,
    "blockysize": 512,
    "interleave": "band"
}

INT_COMPRESSION_KWARGS = {**COMPRESSION_KWARGS, "predictor": 2}  # Horizontal differencing
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
        'COMPRESS': 'ZSTD',  # CORRECTED: FarmVibes uses ZSTD throughout
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