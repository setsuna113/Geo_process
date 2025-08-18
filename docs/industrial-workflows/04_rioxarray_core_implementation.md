# rioxarray Core Implementation - The Foundation Library

## Repository Structure
```
corteva/rioxarray/
├── rioxarray/
│   ├── raster_array.py          # Core DataArray operations
│   ├── raster_dataset.py        # Dataset operations
│   ├── merge.py                 # Merge functionality
│   ├── _io.py                   # I/O operations
│   ├── crs.py                   # CRS handling
│   └── _reproject.py            # Reprojection engine
├── test/
│   ├── integration/
│   │   ├── test_integration_merge.py
│   │   └── test_integration__io.py
│   └── unit/
└── docs/
```

## 1. LOADING RASTERS - THE FOUNDATION

### Core Loading Implementation
```python
# From rioxarray/_io.py (Lines 800-900)
def open_rasterio(
    filename,
    parse_coordinates=None,
    chunks=None,
    cache=None,
    lock=None,
    masked=False,
    mask_and_scale=True,
    variable=None,
    group=None,
    default_name=None,
    decode_times=True,
    decode_timedelta=None,
    band_as_variable=False,
    **open_kwargs,
):
    """
    Open a rasterio dataset as xarray object
    
    Parameters
    ----------
    filename : str, Path, or rasterio.DatasetReader
        Path to the file to open. Or already open rasterio dataset.
    parse_coordinates : bool, optional
        Whether to parse coordinate system info from the file.
    chunks : int, tuple or dict, optional
        Chunk sizes for dask array. E.g., chunks={'band': 1, 'x': 2048, 'y': 2048}
    cache : bool, optional
        Cache data in memory. Default: True for small files, False for large.
    lock : threading.Lock, optional
        Lock for thread safety.
    masked : bool, optional
        If True, mask nodata values as NaN.
    mask_and_scale : bool, optional
        Apply scale and offset attributes.
    """
    
    # Handle different input types
    if isinstance(filename, rasterio.DatasetReader):
        riods = filename
        filename = riods.name
    else:
        riods = rasterio.open(filename, "r", **open_kwargs)
    
    # Parse shape and coordinates
    shape = (riods.count, riods.height, riods.width)
    
    # Build coordinate arrays
    if parse_coordinates is None:
        parse_coordinates = True if riods.count < 10 else False
    
    # Create coordinates from transform
    coords = _make_coords(
        riods.transform,
        riods.width,
        riods.height,
        riods.crs
    )
    
    # Handle chunks for dask
    if chunks is not None:
        from dask.array import from_delayed
        import dask
        
        if isinstance(chunks, dict):
            chunks = (
                chunks.get('band', 1),
                chunks.get('y', 'auto'),
                chunks.get('x', 'auto')
            )
        
        # Create dask array with lazy loading
        def read_block(block_info=None):
            # Determine block slices
            if block_info is not None:
                block_id = block_info[0]['block-id']
                band_slice = slice(
                    block_id[0] * chunks[0],
                    min((block_id[0] + 1) * chunks[0], shape[0])
                )
                row_slice = slice(
                    block_id[1] * chunks[1],
                    min((block_id[1] + 1) * chunks[1], shape[1])
                )
                col_slice = slice(
                    block_id[2] * chunks[2],
                    min((block_id[2] + 1) * chunks[2], shape[2])
                )
                
                # Read window from file
                window = Window.from_slices(row_slice, col_slice)
                data = riods.read(
                    indexes=range(band_slice.start + 1, band_slice.stop + 1),
                    window=window,
                    masked=masked
                )
            return data
        
        # Create dask array
        data = dask.array.from_delayed(
            read_block,
            shape=shape,
            dtype=riods.dtypes[0],
            chunks=chunks
        )
    else:
        # Read entire array
        data = riods.read(masked=masked)
    
    # Handle nodata masking
    if masked and riods.nodata is not None:
        if np.isnan(riods.nodata):
            data = np.where(np.isnan(data), np.nan, data)
        else:
            data = np.where(data == riods.nodata, np.nan, data)
    
    # Create xarray DataArray
    da = xr.DataArray(
        data,
        dims=('band', 'y', 'x'),
        coords=coords,
        attrs=riods.tags()
    )
    
    # Add CRS information
    da.rio.write_crs(riods.crs, inplace=True)
    da.rio.write_transform(riods.transform, inplace=True)
    da.rio.write_nodata(riods.nodata, inplace=True)
    
    return da
```

### Coordinate Generation
```python
# From rioxarray/rioxarray.py
def _make_coords(transform, width, height, crs):
    """
    Generate coordinate arrays from transform
    """
    # Generate 1D coordinate arrays
    x_coords = np.arange(width) * transform[0] + transform[2] + transform[0] / 2
    y_coords = np.arange(height) * transform[4] + transform[5] + transform[4] / 2
    
    # Create coordinate dictionary
    coords = {
        'x': xr.IndexVariable('x', x_coords),
        'y': xr.IndexVariable('y', y_coords)
    }
    
    # Add spatial_ref coordinate for CRS
    if crs is not None:
        coords['spatial_ref'] = xr.Variable((), 0, attrs=crs.to_cf())
    
    return coords
```

## 2. REPROJECTION AND ALIGNMENT

### Core reproject_match Implementation
```python
# From rioxarray/raster_array.py (Lines 540-650)
def reproject_match(
    self,
    match_data_array: Union[xarray.DataArray, xarray.Dataset],
    resampling: Resampling = Resampling.nearest,
    **reproject_kwargs,
) -> xarray.DataArray:
    """
    Reproject to match another dataset's resolution and extent
    
    This is THE KEY FUNCTION for alignment!
    """
    # Get target CRS and transform from match dataset
    dst_crs = match_data_array.rio.crs
    dst_transform = match_data_array.rio.transform()
    dst_shape = match_data_array.rio.shape
    
    # Call main reproject function
    return self.reproject(
        dst_crs,
        transform=dst_transform,
        shape=dst_shape,
        resampling=resampling,
        **reproject_kwargs
    )

def reproject(
    self,
    dst_crs,
    resolution=None,
    shape=None,
    transform=None,
    resampling=Resampling.nearest,
    nodata=None,
    **kwargs
) -> xarray.DataArray:
    """
    Core reprojection using rasterio.warp
    """
    from rasterio.warp import reproject as _reproject
    from rasterio.warp import calculate_default_transform
    
    # Get source CRS
    src_crs = self._obj.rio.crs
    if src_crs is None:
        raise MissingCRS("Source CRS not found")
    
    # Calculate transform if not provided
    if transform is None:
        transform, width, height = calculate_default_transform(
            src_crs,
            dst_crs,
            self._obj.rio.width,
            self._obj.rio.height,
            *self._obj.rio.bounds(),
            resolution=resolution
        )
        dst_shape = (height, width)
    else:
        dst_shape = shape
    
    # Create output array
    dst_data = np.empty(
        (self._obj.shape[0], *dst_shape),
        dtype=self._obj.dtype
    )
    
    # Handle nodata
    if nodata is None:
        nodata = self._obj.rio.nodata
    
    # Reproject each band
    for band_idx in range(self._obj.shape[0]):
        _reproject(
            source=self._obj.values[band_idx],
            destination=dst_data[band_idx],
            src_transform=self._obj.rio.transform(),
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=resampling,
            src_nodata=self._obj.rio.nodata,
            dst_nodata=nodata,
            **kwargs
        )
    
    # Create new coordinates
    new_coords = _make_coords(
        transform,
        dst_shape[1],  # width
        dst_shape[0],  # height
        dst_crs
    )
    
    # Build output DataArray
    out_da = xr.DataArray(
        dst_data,
        dims=self._obj.dims,
        coords={**self._obj.coords, **new_coords},
        attrs=self._obj.attrs
    )
    
    # Set CRS and transform
    out_da.rio.write_crs(dst_crs, inplace=True)
    out_da.rio.write_transform(transform, inplace=True)
    out_da.rio.write_nodata(nodata, inplace=True)
    
    return out_da
```

## 3. MERGING MULTIPLE RASTERS

### Core Merge Implementation
```python
# From rioxarray/merge.py (Lines 84-200)
def merge_arrays(
    dataarrays: Sequence[DataArray],
    bounds: Optional[tuple] = None,
    res: Optional[tuple] = None,
    nodata: Optional[float] = None,
    precision: Optional[float] = None,
    method: Union[str, Callable] = "first",
    crs: Optional[CRS] = None,
    parse_coordinates: bool = True,
) -> DataArray:
    """
    Merge multiple raster DataArrays into single DataArray
    
    Parameters
    ----------
    dataarrays : list of DataArray
        DataArrays to merge
    bounds : tuple, optional
        Output bounds (minx, miny, maxx, maxy)
    res : tuple, optional
        Output resolution (x_res, y_res)
    nodata : float, optional
        NoData value for output
    method : str or callable
        How to handle overlaps:
        - 'first': first valid pixel
        - 'last': last valid pixel  
        - 'min': minimum value
        - 'max': maximum value
        - 'mean': average value
        - callable: custom function
    """
    
    # Input validation
    if not dataarrays:
        raise ValueError("No data arrays to merge")
    
    # Get CRS from first array if not provided
    if crs is None:
        crs = dataarrays[0].rio.crs
    
    # Check all arrays have same CRS
    for da in dataarrays:
        if da.rio.crs != crs:
            raise ValueError(f"CRS mismatch: {da.rio.crs} != {crs}")
    
    # Prepare rasterio-compatible datasets
    class _XarrayReader:
        """Adapter to make xarray look like rasterio dataset"""
        
        def __init__(self, xarray_obj):
            self._xarray_obj = xarray_obj
            self.bounds = xarray_obj.rio.bounds()
            self.transform = xarray_obj.rio.transform()
            self.height = xarray_obj.rio.height
            self.width = xarray_obj.rio.width
            self.crs = xarray_obj.rio.crs
            self.count = xarray_obj.shape[0] if len(xarray_obj.shape) == 3 else 1
            self.dtypes = [xarray_obj.dtype]
            self.nodatavals = [xarray_obj.rio.nodata]
            self.res = (self.transform[0], abs(self.transform[4]))
        
        def read(self, indexes=None, window=None):
            """Read data window"""
            if window is not None:
                row_slice = slice(window.row_off, window.row_off + window.height)
                col_slice = slice(window.col_off, window.col_off + window.width)
                
                if len(self._xarray_obj.shape) == 3:
                    data = self._xarray_obj.values[:, row_slice, col_slice]
                else:
                    data = self._xarray_obj.values[row_slice, col_slice]
            else:
                data = self._xarray_obj.values
            
            return data
    
    # Create readers
    readers = [_XarrayReader(da) for da in dataarrays]
    
    # Use rasterio's merge function
    from rasterio.merge import merge as _rio_merge
    
    merged, out_transform = _rio_merge(
        readers,
        bounds=bounds,
        res=res,
        nodata=nodata,
        precision=precision,
        method=method,
        target_aligned_pixels=True
    )
    
    # Calculate output shape
    if bounds and res:
        width = int((bounds[2] - bounds[0]) / res[0])
        height = int((bounds[3] - bounds[1]) / res[1])
    else:
        height, width = merged.shape[-2:]
    
    # Create coordinates
    coords = _make_coords(out_transform, width, height, crs)
    
    # Add non-spatial coordinates from input
    for coord_name in dataarrays[0].coords:
        if coord_name not in ['x', 'y', 'spatial_ref']:
            coords[coord_name] = dataarrays[0].coords[coord_name]
    
    # Create output DataArray
    out_da = DataArray(
        merged,
        coords=coords,
        dims=dataarrays[0].dims if len(merged.shape) == 3 else ('y', 'x'),
        attrs=dataarrays[0].attrs
    )
    
    # Set metadata
    out_da.rio.write_crs(crs, inplace=True)
    out_da.rio.write_transform(out_transform, inplace=True)
    out_da.rio.write_nodata(nodata or dataarrays[0].rio.nodata, inplace=True)
    
    return out_da
```

### Merge Method Implementations
```python
# From rasterio/merge.py (conceptual - used by rioxarray)
def _merge_method_first(old_data, new_data, old_nodata, new_nodata, **kwargs):
    """First valid pixel"""
    mask = np.isnan(old_data) if np.isnan(old_nodata) else old_data == old_nodata
    np.copyto(old_data, new_data, where=mask)

def _merge_method_last(old_data, new_data, old_nodata, new_nodata, **kwargs):
    """Last valid pixel"""
    mask = ~(np.isnan(new_data) if np.isnan(new_nodata) else new_data == new_nodata)
    np.copyto(old_data, new_data, where=mask)

def _merge_method_min(old_data, new_data, old_nodata, new_nodata, **kwargs):
    """Minimum value"""
    mask = ~(np.isnan(new_data) if np.isnan(new_nodata) else new_data == new_nodata)
    np.copyto(old_data, np.minimum(old_data, new_data), where=mask)

def _merge_method_max(old_data, new_data, old_nodata, new_nodata, **kwargs):
    """Maximum value"""
    mask = ~(np.isnan(new_data) if np.isnan(new_nodata) else new_data == new_nodata)
    np.copyto(old_data, np.maximum(old_data, new_data), where=mask)
```

## 4. EXPORT OPERATIONS

### Core to_raster Implementation
```python
# From rioxarray/raster_array.py (Lines 1800-1900)
def to_raster(
    self,
    raster_path: str,
    driver: str = "GTiff",
    dtype: Optional[str] = None,
    tags: Optional[dict] = None,
    windowed: bool = False,
    recalc_transform: bool = True,
    lock: Optional[Lock] = None,
    compress: Optional[str] = None,
    **profile_kwargs
) -> None:
    """
    Export DataArray to raster file
    
    Parameters
    ----------
    raster_path : str
        Output file path
    driver : str
        GDAL driver name (GTiff, COG, NetCDF, etc.)
    dtype : str, optional
        Output data type
    windowed : bool
        Use windowed writing for large files
    compress : str, optional
        Compression method (deflate, lzw, packbits, etc.)
    """
    
    # Get array data
    data = self._obj.values
    
    # Handle data type
    if dtype is None:
        dtype = str(data.dtype)
    
    # Build output profile
    profile = {
        'driver': driver,
        'dtype': dtype,
        'width': self.width,
        'height': self.height,
        'count': 1 if data.ndim == 2 else data.shape[0],
        'crs': self.crs,
        'transform': self.transform(),
        'nodata': self.nodata
    }
    
    # Add compression
    if compress:
        profile['compress'] = compress
        if compress == 'deflate':
            profile['predictor'] = 3 if 'float' in dtype else 2
            profile['zlevel'] = 9
    
    # Add user kwargs
    profile.update(profile_kwargs)
    
    # Handle tiling for GTiff
    if driver == 'GTiff':
        if 'tiled' not in profile:
            profile['tiled'] = True
        if 'blockxsize' not in profile:
            profile['blockxsize'] = min(512, self.width)
        if 'blockysize' not in profile:
            profile['blockysize'] = min(512, self.height)
    
    # Handle COG driver
    if driver == 'COG':
        profile['driver'] = 'GTiff'
        profile.update({
            'TILED': 'YES',
            'COPY_SRC_OVERVIEWS': 'YES',
            'COMPRESS': compress or 'DEFLATE',
            'BLOCKSIZE': 512
        })
    
    # Write raster
    with rasterio.open(raster_path, 'w', **profile) as dst:
        if windowed:
            # Windowed writing for memory efficiency
            for window, data_window in self._iter_windows():
                if data.ndim == 2:
                    dst.write(data_window, 1, window=window)
                else:
                    for band in range(data.shape[0]):
                        dst.write(data_window[band], band + 1, window=window)
        else:
            # Write all at once
            if data.ndim == 2:
                dst.write(data, 1)
            else:
                for band in range(data.shape[0]):
                    dst.write(data[band], band + 1)
        
        # Add tags
        if tags:
            dst.update_tags(**tags)
        
        # Build overviews for COG
        if driver == 'COG' or profile_kwargs.get('TILED') == 'YES':
            factors = [2, 4, 8, 16, 32]
            dst.build_overviews(factors, Resampling.average)

def _iter_windows(self, block_shapes=(512, 512)):
    """Iterate over windows for windowed writing"""
    height, width = self.shape
    
    for row in range(0, height, block_shapes[0]):
        row_size = min(block_shapes[0], height - row)
        
        for col in range(0, width, block_shapes[1]):
            col_size = min(block_shapes[1], width - col)
            
            window = Window(col, row, col_size, row_size)
            
            # Extract data for window
            if self._obj.ndim == 2:
                data_window = self._obj.values[
                    row:row + row_size,
                    col:col + col_size
                ]
            else:
                data_window = self._obj.values[
                    :,
                    row:row + row_size,
                    col:col + col_size
                ]
            
            yield window, data_window
```

## 5. MEMORY OPTIMIZATION STRATEGIES

### Dask Integration
```python
# From rioxarray/_io.py
def _load_dask_chunks(filename, chunks):
    """
    Load raster with dask for out-of-core computation
    """
    import dask.array as da
    
    # Open dataset
    with rasterio.open(filename) as src:
        # Create reading tasks
        def read_window(window):
            with rasterio.open(filename) as src:
                return src.read(window=window)
        
        # Generate windows
        windows = []
        for row in range(0, src.height, chunks[1]):
            for col in range(0, src.width, chunks[2]):
                window = Window(
                    col, row,
                    min(chunks[2], src.width - col),
                    min(chunks[1], src.height - row)
                )
                windows.append(window)
        
        # Create dask array from delayed reads
        arrays = [da.from_delayed(
            dask.delayed(read_window)(w),
            shape=(src.count, w.height, w.width),
            dtype=src.dtypes[0]
        ) for w in windows]
        
        # Concatenate into single array
        return da.concatenate(arrays)
```

## KEY DESIGN PATTERNS

1. **Adapter Pattern**: `_XarrayReader` makes xarray look like rasterio
2. **Lazy Loading**: Dask integration for out-of-core processing
3. **Windowed I/O**: Process large files in chunks
4. **Transform-based Coordinates**: Direct from GDAL transform matrix
5. **Method Polymorphism**: Multiple merge strategies
6. **COG Support**: Built-in Cloud-Optimized GeoTIFF generation

## COMPLETE EXAMPLE WORKFLOW

```python
import rioxarray
from rasterio.enums import Resampling

# 1. Load multiple rasters with dask
rasters = [
    rioxarray.open_rasterio(
        f"image_{i}.tif",
        chunks={'band': 1, 'x': 2048, 'y': 2048}
    )
    for i in range(5)
]

# 2. Align all to first raster's grid
aligned = [rasters[0]]  # First is reference
for raster in rasters[1:]:
    aligned.append(
        raster.rio.reproject_match(
            rasters[0],
            resampling=Resampling.bilinear
        )
    )

# 3. Merge with mean method
from rioxarray.merge import merge_arrays
merged = merge_arrays(
    aligned,
    method='mean',  # Average overlapping pixels
    nodata=-9999
)

# 4. Export as COG
merged.rio.to_raster(
    "merged_output.tif",
    driver="COG",
    compress="deflate",
    windowed=True  # Memory efficient writing
)
```

## WHY RIOXARRAY IS THE STANDARD

1. **Seamless Integration**: xarray + rasterio = best of both worlds
2. **Coordinate Awareness**: Preserves geographic information
3. **Dask Support**: Built-in distributed computing
4. **GDAL Power**: Full GDAL functionality through rasterio
5. **Pythonic API**: Clean, intuitive interface
6. **Production Ready**: Used by Microsoft, NASA, NOAA, etc.