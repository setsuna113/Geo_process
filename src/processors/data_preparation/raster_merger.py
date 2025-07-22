# src/processors/data_preparation/raster_merger.py
"""Merge multiple rasters into multi-band datasets."""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
import xarray as xr
import rioxarray
from datetime import datetime

from src.config.config import Config
from src.base.processor import BaseProcessor
from src.raster_data.catalog import RasterCatalog, RasterEntry
from src.raster_data.loaders.geotiff_loader import GeoTIFFLoader
from src.database.connection import DatabaseManager
from src.processors.data_preparation.raster_alignment import RasterAligner, AlignmentConfig, AlignmentStrategy
import json

logger = logging.getLogger(__name__)

class RasterMerger(BaseProcessor):
    """Merge multiple rasters into multi-band datasets."""
    
    def __init__(self, config: Config, db_connection: DatabaseManager):
        super().__init__(batch_size=1000, config=config)
        self.db = db_connection
        self.catalog = RasterCatalog(db_connection, config)
        self.loader = GeoTIFFLoader(config)
        
        # Configure alignment handling
        alignment_config = AlignmentConfig(
            resolution_tolerance=config.get('data_preparation', {}).get('resolution_tolerance', 1e-6),
            bounds_tolerance=config.get('data_preparation', {}).get('bounds_tolerance', 1e-4),
            strategy=AlignmentStrategy.REPROJECT  # Default strategy for merging
        )
        self.aligner = RasterAligner(alignment_config)
    
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
        
        alignment_report = None
        # Check alignment using robust aligner
        if validate_alignment:
            raster_paths = [raster.path for raster in rasters.values()]
            alignment_report = self.aligner.analyze_alignment(raster_paths)
            
            if not alignment_report.aligned:
                if alignment_report.has_issues("critical"):
                    # Try to fix alignment issues automatically
                    logger.info(f"Found {len(alignment_report.issues)} alignment issues, attempting fixes...")
                    
                    # Create temporary aligned versions  
                    import tempfile
                    import shutil
                    temp_dir = tempfile.mkdtemp()
                    temp_dir_path = Path(temp_dir)
                    
                    try:
                        aligned_paths = self.aligner.fix_alignment(raster_paths, temp_dir_path)
                        
                        # Reload entries with aligned rasters
                        temp_rasters = {}
                        for key, original_raster in rasters.items():
                            aligned_path = aligned_paths[str(original_raster.path)]
                            # Create temporary entry with aligned path
                            temp_rasters[key] = type(original_raster)(
                                id=original_raster.id,
                                name=original_raster.name,
                                path=aligned_path,
                                dataset_type=original_raster.dataset_type,
                                resolution_degrees=original_raster.resolution_degrees,
                                bounds=original_raster.bounds,
                                data_type=original_raster.data_type,
                                nodata_value=original_raster.nodata_value,
                                file_size_mb=original_raster.file_size_mb,
                                last_validated=original_raster.last_validated,
                                is_active=original_raster.is_active,
                                metadata=original_raster.metadata
                            )
                        rasters = temp_rasters
                        logger.info("✅ Successfully aligned rasters for merging")
                        
                        # Store temp_dir for cleanup after merge
                        self._temp_alignment_dir = temp_dir_path
                        
                    except Exception as e:
                        # Clean up temp dir if alignment fails
                        shutil.rmtree(temp_dir)
                        raise e
                else:
                    logger.warning("Minor alignment issues found, proceeding with xarray interpolation")
        
        # Load and merge data
        merged_data = self._merge_raster_data(rasters)
        
        # Clean up temporary alignment files if they exist
        if hasattr(self, '_temp_alignment_dir') and self._temp_alignment_dir.exists():
            import shutil
            shutil.rmtree(self._temp_alignment_dir)
            delattr(self, '_temp_alignment_dir')
            logger.info("🧹 Cleaned up temporary alignment files")
        
        # Save if requested
        if output_path:
            self._save_merged_raster(merged_data, output_path)
        
        # Store merge metadata in database
        merge_id = self._log_merge_operation(rasters, merged_data)
        
        return {
            'data': merged_data,
            'merge_id': merge_id,
            'alignment_report': alignment_report if validate_alignment else None,
            'metadata': {
                'bands': ['plants', 'animals', 'fungi'],
                'sources': {k: v.path for k, v in rasters.items()},
                'shape': dict(merged_data.sizes),
                'dtype': str(merged_data.to_array().dtype)
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
        
        # Check alignment using robust aligner
        raster_paths = [raster.path for raster in rasters.values()]
        alignment_report = self.aligner.analyze_alignment(raster_paths)
        
        if not alignment_report.aligned:
            if alignment_report.has_issues("critical"):
                # Try to fix alignment issues automatically
                logger.info(f"Found {len(alignment_report.issues)} alignment issues, attempting fixes...")
                
                import tempfile
                import shutil
                temp_dir = tempfile.mkdtemp()
                temp_dir_path = Path(temp_dir)
                
                try:
                    aligned_paths = self.aligner.fix_alignment(raster_paths, temp_dir_path)
                    
                    # Reload entries with aligned rasters
                    temp_rasters = {}
                    for key, original_raster in rasters.items():
                        aligned_path = aligned_paths[str(original_raster.path)]
                        temp_rasters[key] = type(original_raster)(
                            id=original_raster.id,
                            name=original_raster.name,
                            path=aligned_path,
                            dataset_type=original_raster.dataset_type,
                            resolution_degrees=original_raster.resolution_degrees,
                            bounds=original_raster.bounds,
                            data_type=original_raster.data_type,
                            nodata_value=original_raster.nodata_value,
                            file_size_mb=original_raster.file_size_mb,
                            last_validated=original_raster.last_validated,
                            is_active=original_raster.is_active,
                            metadata=original_raster.metadata
                        )
                    rasters = temp_rasters
                    logger.info("✅ Successfully aligned rasters for merging")
                    
                    # Store temp_dir for cleanup after merge
                    self._temp_alignment_dir = temp_dir_path
                    
                except Exception as e:
                    # Clean up temp dir if alignment fails
                    shutil.rmtree(temp_dir)
                    raise e
            else:
                logger.warning("Minor alignment issues found, proceeding with xarray interpolation")
        
        # Use provided band names or dict keys
        if band_names is None:
            band_names = list(raster_names.keys())
        
        # Load and merge
        merged_data = self._merge_raster_data(rasters, band_names)
        
        # Clean up temporary alignment files if they exist
        if hasattr(self, '_temp_alignment_dir') and self._temp_alignment_dir.exists():
            import shutil
            shutil.rmtree(self._temp_alignment_dir)
            delattr(self, '_temp_alignment_dir')
            logger.info("🧹 Cleaned up temporary alignment files")
        
        # Save if requested
        if output_path:
            self._save_merged_raster(merged_data, output_path)
        
        return {
            'data': merged_data,
            'alignment_report': alignment_report,
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
        da_temp = rioxarray.open_rasterio(raster_path, chunks={'x': 1000, 'y': 1000})
        if isinstance(da_temp, xr.DataArray):
            da: xr.DataArray = da_temp
        elif isinstance(da_temp, xr.Dataset):
            da = da_temp.to_array().squeeze()
        elif isinstance(da_temp, list) and da_temp:
            # Handle list case - take first dataset and convert
            da = da_temp[0].to_array().squeeze()
        else:
            raise TypeError(f"Unexpected type from rioxarray.open_rasterio: {type(da_temp)}")
        
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
                json.dumps({k: v.id for k, v in rasters.items()}),
                list(rasters.keys()),
                list(merged_data.dims.values()),
                datetime.now(),
                json.dumps({
                    'sources': {k: str(v.path) for k, v in rasters.items()},
                    'alignment_checked': True,
                    'dtype': str(merged_data.to_array().dtype)
                })
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

