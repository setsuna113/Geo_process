# src/processors/data_preparation/array_converter.py
"""Convert between different array formats while preserving spatial information."""

import logging
from typing import Union, Tuple, Dict, Any, Optional, List, Callable
from typing import Union, Tuple, Dict, Any, Optional, List
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point, box
import pandas as pd
from pathlib import Path

from src.config import config
from src.base.processor import BaseProcessor

logger = logging.getLogger(__name__)

class ArrayConverter(BaseProcessor):
    """Convert between xarray, numpy, and geopandas formats."""
    
    def __init__(self, config: Config):
        super().__init__(batch_size=1000, config=config)
        self.chunk_size = config.get('data_preparation', {}).get('chunk_size', 1000)
    
    def xarray_to_numpy(self, 
                       data: Union[xr.Dataset, xr.DataArray],
                       flatten: bool = True,
                       preserve_coords: bool = True) -> Dict[str, Any]:
        """
        Convert xarray to numpy array.
        
        Args:
            data: Input xarray object
            flatten: Whether to flatten spatial dimensions
            preserve_coords: Whether to preserve coordinate information
            
        Returns:
            Dict with numpy array and metadata
        """
        if isinstance(data, xr.Dataset):
            # Convert to stacked array
            data_array = data.to_array()
        else:
            data_array = data
        
        # Get coordinate information
        coords_info = self._extract_coord_info(data_array)
        
        # Convert to numpy
        if flatten:
            # Handle different cases for flattening
            if isinstance(data, xr.Dataset):
                # For Dataset, we want (n_pixels, n_variables) shape
                # The to_array() creates (variable, lat, lon)
                data_array = data.to_array()
                n_vars = data_array.shape[0]
                # Reshape to (n_vars, n_pixels) then transpose to (n_pixels, n_vars)
                np_array = data_array.values.reshape(n_vars, -1).T
                pixel_coords = None
            else:
                # For DataArray, stack spatial dimensions only
                spatial_dims = [d for d in data_array.dims if d in ['lat', 'lon', 'x', 'y']]
                if len(spatial_dims) == 2:
                    stacked = data_array.stack(pixel=spatial_dims)
                    np_array = stacked.values
                    pixel_coords = np.array([
                        stacked.pixel.values[i] for i in range(len(stacked.pixel))
                    ])
                else:
                    np_array = data_array.values.flatten()
                    pixel_coords = None
        else:
            np_array = data_array.values
            pixel_coords = None
        
        result = {
            'array': np_array,
            'shape': data_array.shape,
            'dims': list(data_array.dims),
            'attrs': dict(data_array.attrs)
        }
        
        if preserve_coords:
            result['coords_info'] = coords_info
            if pixel_coords is not None:
                result['pixel_coords'] = pixel_coords
        
        return result
    
    def numpy_to_xarray(self,
                       array: np.ndarray,
                       coords_info: Dict[str, Any],
                       dims: Optional[List[str]] = None,
                       attrs: Optional[Dict] = None) -> xr.DataArray:
        """
        Convert numpy array back to xarray.
        
        Args:
            array: Numpy array
            coords_info: Coordinate information from original conversion
            dims: Dimension names
            attrs: Attributes to add
            
        Returns:
            xarray DataArray
        """
        # Reconstruct coordinates
        coords = {}
        for coord_name, coord_data in coords_info.items():
            if isinstance(coord_data, dict) and 'values' in coord_data:
                coords[coord_name] = coord_data['values']
            else:
                coords[coord_name] = coord_data
        
        # Use provided dims or extract from coords_info
        if dims is None:
            dims = list(coords.keys())
        
        # Create DataArray
        da = xr.DataArray(
            array,
            coords=coords,
            dims=dims,
            attrs=attrs or {}
        )
        
        return da
    
    def xarray_to_geopandas(self,
                          data: Union[xr.Dataset, xr.DataArray],
                          variable_names: Optional[List[str]] = None,
                          crs: str = 'EPSG:4326') -> gpd.GeoDataFrame:
        """
        Convert xarray to GeoDataFrame with point geometries.
        
        Args:
            data: Input xarray
            variable_names: Variables to include (for Dataset)
            crs: Coordinate reference system
            
        Returns:
            GeoDataFrame with point geometries
        """
        logger.info("Converting xarray to GeoDataFrame")
        
        # Handle Dataset vs DataArray
        if isinstance(data, xr.Dataset):
            if variable_names is None:
                variable_names = list(data.data_vars)
            
            # Stack to create long format
            stacked_data = []
            for var in variable_names:
                da = data[var]
                df = da.to_dataframe(name=var).reset_index()
                stacked_data.append(df)
            
            # Merge all variables
            if len(stacked_data) > 1:
                df = stacked_data[0]
                for other_df in stacked_data[1:]:
                    df = df.merge(other_df, on=['lat', 'lon'], how='outer')
            else:
                df = stacked_data[0]
        else:
            # DataArray
            df = data.to_dataframe(name='value').reset_index()
        
        # Create point geometries
        geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
        
        # Drop redundant lat/lon columns (now in geometry)
        # Drop redundant lat/lon columns (now in geometry)
        gdf = gpd.GeoDataFrame(gdf.drop(columns=["lat", "lon"]), geometry=gdf.geometry, crs=gdf.crs)
        return gdf
    
    def geopandas_to_xarray(self,
                          gdf: gpd.GeoDataFrame,
                          resolution: Optional[float] = None,
                          bounds: Optional[Tuple[float, float, float, float]] = None,
                          value_col: str = 'value') -> xr.DataArray:
        """
        Convert GeoDataFrame points to xarray grid.
        
        Args:
            gdf: Input GeoDataFrame with point geometries
            resolution: Grid resolution (inferred if None)
            bounds: Output bounds (inferred if None)
            value_col: Column containing values
            
        Returns:
            xarray DataArray on regular grid
        """
        logger.info("Converting GeoDataFrame to xarray")
        
        # Extract coordinates
        gdf['lon'] = gdf.geometry.x
        gdf['lat'] = gdf.geometry.y
        
        # Infer resolution if not provided
        if resolution is None:
            lon_diff = np.diff(sorted(gdf['lon'].unique()))
            lat_diff = np.diff(sorted(gdf['lat'].unique()))
            
            if len(lon_diff) > 0 and len(lat_diff) > 0:
                resolution = float(min(float(np.median(lon_diff)), float(np.median(lat_diff))))
            else:
                resolution = 0.1  # Default
            
            logger.info(f"Inferred resolution: {resolution}")
        
        # Infer bounds if not provided
        if bounds is None:
            bounds = (
                float(gdf["lon"].min()) - resolution/2,
                float(gdf["lat"].min()) - resolution/2,
                float(gdf["lon"].max()) + resolution/2,
                float(gdf["lat"].max()) + resolution/2
            )
        
        # Create regular grid
        west, south, east, north = bounds
        lons = np.arange(west + resolution/2, east, resolution)
        lats = np.arange(south + resolution/2, north, resolution)
        
        # Create empty grid
        grid = np.full((len(lats), len(lons)), np.nan)
        
        # Fill grid with values
        for _, row in gdf.iterrows():
            lon_idx = np.argmin(np.abs(lons - row['lon']))
            lat_idx = np.argmin(np.abs(lats - row['lat']))
            
            if 0 <= lon_idx < len(lons) and 0 <= lat_idx < len(lats):
                grid[lat_idx, lon_idx] = row[value_col]
        
        # Create DataArray
        da = xr.DataArray(
            grid,
            coords={'lat': lats, 'lon': lons},
            dims=['lat', 'lon'],
            attrs={'crs': gdf.crs.to_string() if gdf.crs else 'EPSG:4326'}
        )
        
        return da
    
    def flatten_spatial(self, 
                       data: xr.DataArray,
                       return_indices: bool = True) -> Dict[str, Any]:
        """
        Flatten spatial dimensions of DataArray.
        
        Args:
            data: Input DataArray
            return_indices: Whether to return pixel indices
            
        Returns:
            Dict with flattened array and metadata
        """
        spatial_dims = [d for d in data.dims if d in ['lat', 'lon', 'x', 'y']]
        
        if len(spatial_dims) != 2:
            raise ValueError(f"Expected 2 spatial dimensions, got {len(spatial_dims)}")
        
        # Stack spatial dimensions
        stacked = data.stack(pixel=spatial_dims)
        
        # Get coordinate arrays
        if return_indices:
            indices = np.array([
                (i, j) for i, j in stacked.pixel.values
            ])
        else:
            indices = None
        
        return {
            'array': stacked.values,
            'original_shape': data.shape,
            'spatial_dims': spatial_dims,
            'other_dims': [d for d in data.dims if d not in spatial_dims],
            'indices': indices,
            'coords': {dim: data[dim].values for dim in spatial_dims}
        }
    
    def unflatten_spatial(self,
                         flat_array: np.ndarray,
                         metadata: Dict[str, Any]) -> xr.DataArray:
        """
        Unflatten array back to spatial dimensions.
        
        Args:
            flat_array: Flattened array
            metadata: Metadata from flatten operation
            
        Returns:
            Reconstructed DataArray
        """
        # Reconstruct coordinates
        coords = {}
        for dim, values in metadata['coords'].items():
            coords[dim] = values
        
        # Add other dimensions if present
        for dim in metadata.get('other_dims', []):
            if dim in metadata['coords']:
                coords[dim] = metadata['coords'][dim]
        
        # Reshape array
        shape = metadata['original_shape']
        reshaped = flat_array.reshape(shape)
        
        # Create DataArray
        dims = metadata.get('other_dims', []) + metadata['spatial_dims']
        da = xr.DataArray(
            reshaped,
            coords=coords,
            dims=dims
        )
        
        return da
    
    def process_chunked(self,
                       data: Union[xr.Dataset, xr.DataArray],
                       operation: str,
                       **kwargs) -> Any:
        """
        Process large arrays in chunks.
        
        Args:
            data: Input data
            operation: Operation to perform ('to_numpy', 'to_geopandas', etc.)
            **kwargs: Additional arguments for operation
            
        Returns:
            Processed result
        """
        logger.info(f"Processing in chunks with operation: {operation}")
        
        # Define chunk size based on memory constraints
        if hasattr(data, 'chunk'):
            # Already chunked
            chunked_data = data
        else:
            # Chunk the data
            chunk_dict = {}
            for dim in data.dims:
                if dim in ['lat', 'lon', 'x', 'y']:
                    chunk_dict[dim] = self.chunk_size
                else:
                    chunk_dict[dim] = -1  # Don't chunk non-spatial dims
            
            chunked_data = data.chunk(chunk_dict)
        
        # Map operation
        operation_map: Dict[str, Callable] = {
            'to_numpy': self.xarray_to_numpy,
            'to_geopandas': self.xarray_to_geopandas,
            'flatten': self.flatten_spatial
        }
        
        if operation not in operation_map:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Process chunks
        results = []
        if hasattr(chunked_data.data, "blocks"):
            # Data is chunked with Dask
            for chunk in chunked_data.data.blocks:
                # Convert chunk back to xarray for processing
                chunk_coords = {}
                for dim, coord in chunked_data.coords.items():
                    if dim in chunked_data.dims:
                        chunk_coords[dim] = coord
                
                chunk_da = xr.DataArray(chunk, coords=chunk_coords, dims=chunked_data.dims)
                result = operation_map[operation](chunk_da, **kwargs)
                results.append(result)
        else:
            # Data is not chunked or small enough to process as single chunk
            result = operation_map[operation](chunked_data, **kwargs)
            results.append(result)
        
        # Combine results based on operation type
        if operation == 'to_numpy':
            # Concatenate numpy arrays
            combined = np.concatenate([r['array'] for r in results])
            return {
                'array': combined,
                'metadata': results[0]  # Use metadata from first chunk
            }
        elif operation == 'to_geopandas':
            # Concatenate GeoDataFrames
            return pd.concat(results, ignore_index=True)
        else:
            return results
    
    def _extract_coord_info(self, data_array: xr.DataArray) -> Dict[str, Any]:
        """Extract coordinate information from DataArray."""
        coord_info = {}
        
        for coord_name, coord in data_array.coords.items():
            coord_info[str(coord_name)] = {
                'values': coord.values,
                'dims': coord.dims,
                'attrs': dict(coord.attrs)
            }
        
        return coord_info

    def process_single(self, item: Any) -> Any:
        """Process a single item - implementation depends on specific use case."""
        # This is a utility class, specific processing should be handled by calling methods directly
        return item
    
    def validate_input(self, item: Any) -> Tuple[bool, Optional[str]]:
        """Validate input item."""
        return True, None

