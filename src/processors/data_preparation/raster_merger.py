# src/processors/data_preparation/raster_merger.py
"""Merge multiple rasters into multi-band datasets."""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
import xarray as xr
import rioxarray
from dataclasses import dataclass
from datetime import datetime

from src.config.config import Config
from src.base.processor import BaseProcessor
from src.raster_data.catalog import RasterCatalog, RasterEntry
from src.raster_data.loaders.geotiff_loader import GeoTIFFLoader
from src.database.connection import DatabaseManager

logger = logging.getLogger(__name__)

@dataclass
class AlignmentCheck:
    """Results of spatial alignment check."""
    aligned: bool
    same_resolution: bool
    same_bounds: bool
    same_crs: bool
    resolution_diff: Optional[float]
    bounds_diff: Optional[Tuple[float, float, float, float]]
    crs_mismatch: Optional[Tuple[str, str]]

class RasterMerger(BaseProcessor):
    """Merge multiple rasters into multi-band datasets."""
    
    def __init__(self, config: Config, db_connection: DatabaseManager):
        super().__init__(batch_size=1000, config=config)
        self.db = db_connection
        self.catalog = RasterCatalog(db_connection, config)
        self.loader = GeoTIFFLoader(config)
        
        # Tolerance for alignment checks
        self.resolution_tolerance = config.get('data_preparation', {}).get(
            'resolution_tolerance', 1e-6
        )
        self.bounds_tolerance = config.get('data_preparation', {}).get(
            'bounds_tolerance', 1e-4
        )
    
    def merge_paf_rasters(self, 
                         plants_name: str,
                         animals_name: str,
                         fungi_name: str,
                         output_path: Optional[Path] = None,
                         validate_alignment: bool = True) -> Dict[str, Any]:
        """
        Merge P, A, F rasters into a multi-band dataset.
        
        Args:
            plants_name: Name of plants raster in catalog
            animals_name: Name of animals/vertebrates raster in catalog
            fungi_name: Name of fungi raster in catalog
            output_path: Optional path to save merged raster
            validate_alignment: Whether to validate spatial alignment
            
        Returns:
            Dictionary with merged data and metadata
        """
        logger.info(f"Merging rasters: P={plants_name}, A={animals_name}, F={fungi_name}")
        
        # Load raster entries from catalog
        rasters = self._load_raster_entries({
            'plants': plants_name,
            'animals': animals_name,
            'fungi': fungi_name
        })
        
        alignment = None
        # Check alignment
        if validate_alignment:
            alignment = self._check_alignment(list(rasters.values()))
            if not alignment.aligned:
                if not self._can_fix_alignment(alignment):
                    raise ValueError(f"Rasters are not aligned: {alignment}")
                else:
                    logger.warning("Rasters misaligned but fixable, proceeding with adjustment")
        
        # Load and merge data
        merged_data = self._merge_raster_data(rasters)
        
        # Save if requested
        if output_path:
            self._save_merged_raster(merged_data, output_path)
        
        # Store merge metadata in database
        merge_id = self._log_merge_operation(rasters, merged_data)
        
        return {
            'data': merged_data,
            'merge_id': merge_id,
            'alignment': alignment if validate_alignment else None,
            'metadata': {
                'bands': ['plants', 'animals', 'fungi'],
                'sources': {k: v.path for k, v in rasters.items()},
                'shape': merged_data.shape,
                'dtype': str(merged_data.dtype)
            }
        }
    
    def merge_custom_rasters(self,
                           raster_names: Dict[str, str],
                           band_names: Optional[List[str]] = None,
                           output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Merge arbitrary rasters into multi-band dataset.
        
        Args:
            raster_names: Dict mapping band names to raster catalog names
            band_names: Optional custom band names (defaults to dict keys)
            output_path: Optional output path
            
        Returns:
            Dictionary with merged data and metadata
        """
        logger.info(f"Merging {len(raster_names)} rasters")
        
        # Load raster entries
        rasters = self._load_raster_entries(raster_names)
        
        alignment = None
        # Check alignment
        alignment = self._check_alignment(list(rasters.values()))
        if not alignment.aligned:
            raise ValueError(f"Rasters are not aligned: {alignment}")
        
        # Use provided band names or dict keys
        if band_names is None:
            band_names = list(raster_names.keys())
        
        # Load and merge
        merged_data = self._merge_raster_data(rasters, band_names)
        
        # Save if requested
        if output_path:
            self._save_merged_raster(merged_data, output_path)
        
        return {
            'data': merged_data,
            'alignment': alignment,
            'metadata': {
                'bands': band_names,
                'sources': {k: v.path for k, v in rasters.items()}
            }
        }
    
    def _load_raster_entries(self, raster_names: Dict[str, str]) -> Dict[str, RasterEntry]:
        """Load raster entries from catalog."""
        entries = {}
        
        for band_name, raster_name in raster_names.items():
            entry = self.catalog.get_raster(raster_name)
            if entry is None:
                raise ValueError(f"Raster '{raster_name}' not found in catalog")
            entries[band_name] = entry
            
        return entries
    
    def _check_alignment(self, rasters: List[RasterEntry]) -> AlignmentCheck:
        """Check if rasters are spatially aligned."""
        if len(rasters) < 2:
            return AlignmentCheck(
                aligned=True,
                same_resolution=True,
                same_bounds=True,
                same_crs=True,
                resolution_diff=None,
                bounds_diff=None,
                crs_mismatch=None
            )
        
        ref_raster = rasters[0]
        ref_metadata = self.loader.extract_metadata(ref_raster.path)
        
        for raster in rasters[1:]:
            metadata = self.loader.extract_metadata(raster.path)
            
            # Check resolution
            res_diff = abs(metadata.resolution_degrees - ref_metadata.resolution_degrees)
            same_resolution = res_diff < self.resolution_tolerance
            
            # Check bounds
            # Calculate bounds difference
            bounds_diff_calc = [
                abs(m - r) for m, r in zip(metadata.bounds, ref_metadata.bounds)
            ]
            bounds_diff = (bounds_diff_calc[0], bounds_diff_calc[1], bounds_diff_calc[2], bounds_diff_calc[3]) if len(bounds_diff_calc) >= 4 else (0.0, 0.0, 0.0, 0.0)
            same_bounds = all(d < self.bounds_tolerance for d in bounds_diff)
            
            # Check CRS
            same_crs = metadata.crs == ref_metadata.crs
            
            if not (same_resolution and same_bounds and same_crs):
                return AlignmentCheck(
                    aligned=False,
                    same_resolution=same_resolution,
                    same_bounds=same_bounds,
                    same_crs=same_crs,
                    resolution_diff=res_diff if not same_resolution else None,
                    bounds_diff=bounds_diff if not same_bounds else None,
                    crs_mismatch=(metadata.crs, ref_metadata.crs) if not same_crs else None
                )
        
        return AlignmentCheck(
            aligned=True,
            same_resolution=True,
            same_bounds=True,
            same_crs=True,
            resolution_diff=None,
            bounds_diff=None,
            crs_mismatch=None
        )
    
    def _can_fix_alignment(self, alignment: AlignmentCheck) -> bool:
        """Check if alignment issues can be automatically fixed."""
        # Can fix small bounds differences
        if not alignment.same_bounds and alignment.bounds_diff:
            max_diff = max(alignment.bounds_diff)
            if max_diff < 0.1:  # Less than 0.1 degrees
                return True
        
        # Cannot fix CRS or resolution differences automatically
        return False
    
    def _merge_raster_data(self, 
                          rasters: Dict[str, RasterEntry],
                          band_names: Optional[List[str]] = None) -> xr.Dataset:
        """Load and merge raster data into xarray Dataset."""
        if band_names is None:
            band_names = list(rasters.keys())
        
        data_arrays = {}
        
        # Load first raster to get dimensions
        first_name = band_names[0]
        first_entry = rasters[first_name]
        
        # Load as xarray
        logger.info(f"Loading {first_name} from {first_entry.path}")
        first_da = self._load_as_xarray(first_entry.path, first_name)
        data_arrays[first_name] = first_da
        
        # Get reference coordinates
        ref_coords = first_da.coords
        
        # Load remaining rasters
        for band_name in band_names[1:]:
            entry = rasters[band_name]
            logger.info(f"Loading {band_name} from {entry.path}")
            
            da = self._load_as_xarray(entry.path, band_name)
            
            # Align to reference coordinates if needed
            if not da.coords.equals(ref_coords):
                logger.warning(f"Aligning {band_name} to reference coordinates")
                da = da.interp_like(first_da, method='nearest')
            
            data_arrays[band_name] = da
        
        # Merge into dataset
        ds = xr.Dataset(data_arrays)
        
        # Add metadata
        ds.attrs.update({
            'created_by': 'RasterMerger',
            'merge_date': str(datetime.now()),
            'bands': band_names,
            'crs': first_da.attrs.get('crs', 'EPSG:4326')
        })
        
        return ds
    
    def _load_as_xarray(self, raster_path: Path, band_name: str) -> xr.DataArray:
        """Load raster as xarray DataArray."""
        # rioxarray imported at top
        
        # Use rioxarray for better CRS handling
        da = rioxarray.open_rasterio(raster_path, chunks={'x': 1000, 'y': 1000})
        
        # If multi-band, select first band
        if 'band' in da.dims:
            da = da.sel(band=1)
        
        # Rename to standard names
        if 'x' in da.dims:
            da = da.rename({'x': 'lon', 'y': 'lat'})
        
        # Set name
        da.name = band_name
        
        return da
    
    def _save_merged_raster(self, dataset: xr.Dataset, output_path: Path):
        """Save merged dataset to file."""
        # Convert to multi-band DataArray
        bands = list(dataset.data_vars)
        
        # Stack bands
        stacked = xr.concat([dataset[band] for band in bands], dim='band')
        stacked = stacked.assign_coords(band=bands)
        
        # Save using rioxarray
        stacked.rio.to_raster(
            output_path,
            compress='lzw',
            tiled=True,
            blockxsize=256,
            blockysize=256
        )
        
        logger.info(f"Merged raster saved to {output_path}")
    
    def _log_merge_operation(self, rasters: Dict[str, RasterEntry], 
                           merged_data: xr.Dataset) -> int:
        """Log merge operation to database."""
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            
            # Create merge record
            cur.execute("""
                INSERT INTO raster_merge_log
                (source_rasters, band_names, output_shape, merge_date, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (
                {k: v.id for k, v in rasters.items()},
                list(rasters.keys()),
                list(merged_data.dims.values()),
                datetime.now(),
                {
                    'sources': {k: str(v.path) for k, v in rasters.items()},
                    'alignment_checked': True,
                    'dtype': str(merged_data.to_array().dtype)
                }
            ))
            
            merge_id = cur.fetchone()[0]
            conn.commit()
            
        return merge_id
    def process_single(self, item: Any) -> Any:
        """Process a single item - implementation depends on specific use case."""
        return item
    
    def validate_input(self, item: Any) -> Tuple[bool, Optional[str]]:
        """Validate input item."""
        return True, None

