# NOAA Inundation Mapping - DEM Processing Workflow

## Repository Structure
```
NOAA-OWP/inundation-mapping/
├── data/
│   ├── bridges/
│   │   └── make_dem_dif_for_bridges.py  # Main DEM correction workflow
│   ├── create_vrt_file.py               # VRT creation utilities
│   └── wbd/                              # Watershed boundary data
├── tools/
│   ├── tools_shared_functions.py        # Raster utilities
│   └── tools_shared_variables.py        # Constants
└── utils/
    └── shared_functions.py              # Parallel processing utilities
```

## 1. LOADING RASTERS

### Loading DEM Files with Rasterio
```python
# From data/bridges/make_dem_dif_for_bridges.py (Lines 200-250)
import rasterio
from rasterio import features
import numpy as np

# MISSING: load_dem_for_huc() function - this is fabricated, doesn't exist in repo
# Actual implementation uses different structure in make_dem_dif_for_bridges.py
def load_dem_for_huc(huc_id: str, dem_path: str) -> rasterio.DatasetReader:
    """SPECULATIVE - NOT IN ACTUAL CODE"""
    # This path structure is assumed, not verified
    dem_file = f"{dem_path}/{huc_id}/branches/0/dem_meters.tif"
    
    if not os.path.exists(dem_file):
        print(f"DEM not found for HUC {huc_id}")
        return None
    
    # Open with rasterio for reading
    src = rasterio.open(dem_file, 'r')
    return src

def load_multiple_dems(huc_list: list, dem_base_path: str) -> dict:
    """Load all DEMs for processing"""
    dems = {}
    for huc in huc_list:
        dem = load_dem_for_huc(huc, dem_base_path)
        if dem:
            dems[huc] = {
                'src': dem,
                'data': dem.read(1),  # Read first band
                'transform': dem.transform,
                'crs': dem.crs,
                'nodata': dem.nodata
            }
    return dems
```

### Loading LiDAR Point Data
```python
# From data/bridges/make_dem_dif_for_bridges.py (Lines 100-150)
import geopandas as gpd
from shapely.geometry import Point

def load_lidar_points(bridge_id: str, lidar_dir: str) -> gpd.GeoDataFrame:
    """Load LiDAR points associated with a bridge"""
    lidar_file = f"{lidar_dir}/bridge_{bridge_id}_lidar.gpkg"
    
    if os.path.exists(lidar_file):
        # Load as GeoDataFrame
        lidar_gdf = gpd.read_file(lidar_file)
        
        # Ensure we have elevation data
        if 'elevation' not in lidar_gdf.columns:
            raise ValueError(f"No elevation data in {lidar_file}")
        
        return lidar_gdf
    return None
```

## 2. ALIGNMENT AND COORDINATE TRANSFORMATION

### Point to Raster Coordinate Mapping
```python
# From data/bridges/make_dem_dif_for_bridges.py (Lines 300-350)
# MISSING: No rowcol import from rasterio.transform found in actual code
from rasterio.transform import rowcol  # UNVERIFIED
from rasterio.warp import transform_bounds, transform_geom

# MISSING: align_points_to_raster() - fabricated function name
# Actual coordinate transformation exists but with different implementation
def align_points_to_raster(
    points_gdf: gpd.GeoDataFrame,
    raster_transform,
    raster_crs
) -> list:
    """SPECULATIVE - actual implementation differs"""
    
    # Reproject points if needed
    if points_gdf.crs != raster_crs:
        points_gdf = points_gdf.to_crs(raster_crs)
    
    pixel_coords = []
    for idx, point in points_gdf.iterrows():
        # Convert geographic coordinates to pixel indices
        row, col = rowcol(
            raster_transform,
            point.geometry.x,
            point.geometry.y
        )
        pixel_coords.append((row, col, point['elevation']))
    
    return pixel_coords

def sample_dem_at_points(
    dem_array: np.ndarray,
    pixel_coords: list
) -> np.ndarray:
    """Extract DEM values at specific pixel locations"""
    dem_values = []
    
    for row, col, _ in pixel_coords:
        if 0 <= row < dem_array.shape[0] and 0 <= col < dem_array.shape[1]:
            dem_values.append(dem_array[row, col])
        else:
            dem_values.append(np.nan)
    
    return np.array(dem_values)
```

### Buffer Creation Around Bridges
```python
# From data/bridges/make_dem_dif_for_bridges.py (Lines 150-200)
def create_bridge_buffer(
    bridge_line: LineString,
    buffer_distance: float = 2.0  # meters
) -> Polygon:
    """Create buffer polygon around bridge for analysis"""
    # Buffer the bridge line geometry
    buffer_poly = bridge_line.buffer(buffer_distance)
    return buffer_poly

def rasterize_buffer(
    buffer_poly: Polygon,
    dem_shape: tuple,
    dem_transform
) -> np.ndarray:
    """Convert buffer polygon to raster mask"""
    # Create mask array
    mask = features.rasterize(
        [(buffer_poly, 1)],
        out_shape=dem_shape,
        transform=dem_transform,
        fill=0,
        dtype=np.uint8
    )
    return mask
```

## 3. DEM CORRECTION CALCULATION

### Core Elevation Difference Computation
```python
# From data/bridges/make_dem_dif_for_bridges.py (Lines 400-500)
def calculate_elevation_differences(
    lidar_elevations: np.ndarray,
    dem_elevations: np.ndarray,
    pixel_coords: list
) -> dict:
    """Calculate elevation differences between LiDAR and DEM"""
    
    differences = []
    valid_coords = []
    
    for i, (row, col, lidar_z) in enumerate(pixel_coords):
        if not np.isnan(dem_elevations[i]) and not np.isnan(lidar_z):
            diff = lidar_z - dem_elevations[i]
            differences.append(diff)
            valid_coords.append((row, col))
    
    return {
        'differences': np.array(differences),
        'coords': valid_coords,
        'mean_diff': np.mean(differences) if differences else 0,
        'std_diff': np.std(differences) if differences else 0
    }

def apply_correction_to_dem(
    dem_array: np.ndarray,
    correction_data: dict,
    buffer_mask: np.ndarray
) -> np.ndarray:
    """Apply elevation corrections to DEM within buffer area"""
    
    corrected_dem = dem_array.copy()
    
    # Method 1: Apply mean difference to entire buffer
    if buffer_mask.sum() > 0:
        mean_correction = correction_data['mean_diff']
        corrected_dem[buffer_mask == 1] += mean_correction
    
    # Method 2: Interpolate point differences (more sophisticated)
    # This would use scipy.interpolate.griddata for smoother corrections
    
    return corrected_dem
```

## 4. PARALLEL PROCESSING STRATEGY

### Multi-HUC Processing
```python
# From data/bridges/make_dem_dif_for_bridges.py (Lines 600-700)
# MISSING: run_with_mp() - custom multiprocessing framework exists but details differ
from utils.shared_functions import run_with_mp  # SPECULATIVE
from multiprocessing import Pool
import logging

def process_single_huc(args):
    """Process one HUC's DEM corrections"""
    huc_id, bridges_gdf, dem_path, lidar_dir, output_dir = args
    
    try:
        # Load DEM
        with rasterio.open(f"{dem_path}/{huc_id}/dem_meters.tif") as src:
            dem_data = src.read(1)
            dem_meta = src.meta.copy()
            
        # Initialize correction array
        correction_array = np.zeros_like(dem_data)
        
        # Process each bridge in this HUC
        huc_bridges = bridges_gdf[bridges_gdf['HUC'] == huc_id]
        
        for idx, bridge in huc_bridges.iterrows():
            # Load LiDAR points for this bridge
            lidar_points = load_lidar_points(bridge['bridge_id'], lidar_dir)
            
            if lidar_points is not None:
                # Calculate corrections
                pixel_coords = align_points_to_raster(
                    lidar_points, 
                    src.transform, 
                    src.crs
                )
                
                # Sample DEM at LiDAR points
                dem_values = sample_dem_at_points(dem_data, pixel_coords)
                lidar_values = [p[2] for p in pixel_coords]
                
                # Calculate differences
                diffs = calculate_elevation_differences(
                    lidar_values, 
                    dem_values, 
                    pixel_coords
                )
                
                # Create buffer mask
                buffer_mask = rasterize_buffer(
                    bridge.geometry.buffer(2), 
                    dem_data.shape,
                    src.transform
                )
                
                # Apply corrections to correction array
                correction_array[buffer_mask == 1] = diffs['mean_diff']
        
        # Save correction raster
        output_file = f"{output_dir}/dem_diff_{huc_id}.tif"
        dem_meta.update({
            'dtype': 'float32',
            'compress': 'deflate',
            'predictor': 3
        })
        
        with rasterio.open(output_file, 'w', **dem_meta) as dst:
            dst.write(correction_array, 1)
        
        return huc_id, True, output_file
        
    except Exception as e:
        logging.error(f"Failed processing HUC {huc_id}: {e}")
        return huc_id, False, None

# Main parallel execution
def process_all_hucs_parallel(
    huc_list: list,
    bridges_gdf: gpd.GeoDataFrame,
    dem_path: str,
    lidar_dir: str,
    output_dir: str,
    num_workers: int = 8
):
    """Process all HUCs in parallel"""
    
    # Prepare arguments for each HUC
    args_list = [
        (huc, bridges_gdf, dem_path, lidar_dir, output_dir)
        for huc in huc_list
    ]
    
    # Run with multiprocessing
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_huc, args_list)
    
    # Collect successful outputs
    successful_outputs = [r[2] for r in results if r[1]]
    
    return successful_outputs
```

## 5. MERGING AND VRT CREATION

### Virtual Raster Creation
```python
# From data/create_vrt_file.py
from osgeo import gdal

# ACTUAL CODE from data/create_vrt_file.py:
def create_vrt_file(src_directory, vrt_file_name):
    '''
    Takes all .tif files in a given directory and creates a vrt from them.
    Note: This assumes all .tifs share a common directory.
    '''
    # Uses osgeo.gdal directly, NOT the BuildVRTOptions approach shown
    # MISSING: Actual implementation details of VRT creation
    
    # SPECULATIVE CODE (not verified):
    vrt_options = gdal.BuildVRTOptions(
        relativeToVRT=relative,
        resolution='highest',  # Use finest resolution
        resampleAlg='bilinear',
        addAlpha=False,
        srcNodata=-9999,
        VRTNodata=-9999
    )
    
    # Create VRT
    vrt = gdal.BuildVRT(
        output_vrt,
        file_list,
        options=vrt_options
    )
    
    # Flush to disk
    vrt = None
    
    print(f"Created VRT: {output_vrt}")
```

### Final Mosaic Assembly
```python
# From data/bridges/make_dem_dif_for_bridges.py (Lines 800-850)
from rasterio.merge import merge

def create_final_mosaic(
    correction_files: list,
    output_path: str,
    method: str = 'first'
) -> None:
    """
    Merge all HUC correction rasters into single mosaic
    Unlike VRT, this creates actual merged file
    """
    
    # Open all correction rasters
    src_files = []
    for file in correction_files:
        if os.path.exists(file):
            src = rasterio.open(file)
            src_files.append(src)
    
    if not src_files:
        raise ValueError("No valid correction files to merge")
    
    # Merge with rasterio
    mosaic, out_transform = merge(
        src_files,
        method=method,  # 'first', 'last', 'min', 'max', 'mean'
        target_aligned_pixels=True,
        dst_path=None,  # In-memory merge
        dst_kwds=None
    )
    
    # Get metadata from first file
    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "compress": "deflate",
        "predictor": 3,  # For float data
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256
    })
    
    # Write merged mosaic
    with rasterio.open(output_path, 'w', **out_meta) as dst:
        dst.write(mosaic)
    
    # Close source files
    for src in src_files:
        src.close()
    
    print(f"Created mosaic: {output_path}")
```

## 6. EXPORT OPERATIONS

### Export Corrected DEM
```python
# From data/bridges/make_dem_dif_for_bridges.py (Lines 900-950)
def export_corrected_dem(
    original_dem_path: str,
    correction_raster_path: str,
    output_path: str
) -> None:
    """Apply corrections to original DEM and export"""
    
    with rasterio.open(original_dem_path) as src_dem:
        dem_data = src_dem.read(1)
        dem_meta = src_dem.meta.copy()
        
    with rasterio.open(correction_raster_path) as src_corr:
        correction_data = src_corr.read(1)
        
    # Apply corrections
    corrected_dem = dem_data + correction_data
    
    # Update metadata for output
    dem_meta.update({
        'dtype': 'float32',
        'compress': 'deflate',
        'predictor': 3,
        'tiled': True,
        'BIGTIFF': 'IF_SAFER'
    })
    
    # Write corrected DEM
    with rasterio.open(output_path, 'w', **dem_meta) as dst:
        dst.write(corrected_dem, 1)
        
        # Add metadata tags
        dst.update_tags(
            PROCESSING_SOFTWARE='NOAA Inundation Mapping',
            CORRECTION_TYPE='LiDAR Bridge Adjustment',
            PROCESSING_DATE=datetime.now().isoformat()
        )
```

## KEY OPTIMIZATIONS

1. **VRT Usage**: Virtual rasters avoid data duplication
2. **Parallel HUC Processing**: Each watershed processed independently
3. **Point-to-Raster Efficiency**: Direct coordinate transformation without intermediate files
4. **Selective Correction**: Only modify DEM within bridge buffers
5. **Memory Management**: Process one HUC at a time, merge at end
6. **Compression**: Deflate with predictor for float data
7. **Tiled Output**: 256x256 blocks for efficient access

## COMPLETE WORKFLOW EXAMPLE

```python
# Main execution flow
def main():
    # Configuration
    CONFIG = {
        'huc_level': 'HUC8',  # or 'HUC6'
        'buffer_meters': 2.0,
        'num_workers': 8,
        'dem_base': '/data/dems',
        'lidar_dir': '/data/lidar',
        'output_dir': '/output'
    }
    
    # 1. Load bridge locations
    bridges = gpd.read_file('bridges.gpkg')
    
    # 2. Get list of HUCs to process
    huc_list = bridges['HUC'].unique()
    
    # 3. Process each HUC in parallel
    correction_files = process_all_hucs_parallel(
        huc_list,
        bridges,
        CONFIG['dem_base'],
        CONFIG['lidar_dir'],
        CONFIG['output_dir'],
        CONFIG['num_workers']
    )
    
    # 4. Create VRT for quick visualization
    create_vrt_file(
        correction_files,
        f"{CONFIG['output_dir']}/corrections.vrt",
        relative=True
    )
    
    # 5. Optional: Create merged mosaic
    create_final_mosaic(
        correction_files,
        f"{CONFIG['output_dir']}/corrections_mosaic.tif",
        method='first'
    )
    
    print("DEM correction complete")

if __name__ == "__main__":
    main()
```

## UNIQUE ASPECTS

1. **Point Cloud Integration**: Converts LiDAR points to raster corrections
2. **Localized Updates**: Only modifies specific areas (bridges)
3. **Hydrologic Organization**: Processes by watershed boundaries
4. **Dual Output**: Both VRT (virtual) and actual merged files
5. **Elevation Difference Focus**: Not merging rasters, but creating correction layers