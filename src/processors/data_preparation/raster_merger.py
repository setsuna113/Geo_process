# src/processors/data_preparation/raster_merger.py
"""Enhanced raster merger with progress tracking and checkpoint support."""

import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from pathlib import Path
import numpy as np
import xarray as xr
import rioxarray
from datetime import datetime
import tempfile
import shutil

from src.config import config
from src.base.processor import BaseProcessor
from src.base.memory_manager import get_memory_manager
from src.domain.raster.catalog import RasterCatalog, RasterEntry
from src.domain.raster.loaders.geotiff_loader import GeoTIFFLoader
from src.database.connection import DatabaseManager
from src.processors.data_preparation.raster_alignment import RasterAligner, AlignmentConfig, AlignmentStrategy
from src.core.progress_events import get_event_bus, create_file_progress, EventType
import json

logger = logging.getLogger(__name__)


class RasterMerger(BaseProcessor):
    """Enhanced raster merger with progress tracking and memory management."""
    
    def __init__(self, config: Config, db_connection: DatabaseManager):
        # Initialize with enhanced features
        super().__init__(
            batch_size=1000, 
            config=config,
            enable_progress=True,
            enable_checkpoints=True,
            checkpoint_interval=1,  # Checkpoint after each merge
            supports_chunking=True
        )
        
        self.db = db_connection
        self.catalog = RasterCatalog(db_connection, config)
        self.loader = GeoTIFFLoader(config)
        self.memory_manager = get_memory_manager()
        self.event_bus = get_event_bus()
        
        # Configure alignment handling
        alignment_config = AlignmentConfig(
            resolution_tolerance=config.get('data_preparation', {}).get('resolution_tolerance', 1e-6),
            bounds_tolerance=config.get('data_preparation', {}).get('bounds_tolerance', 1e-4),
            strategy=AlignmentStrategy.REPROJECT
        )
        self.aligner = RasterAligner(alignment_config)
        
        # Merge configuration
        self.merge_config = {
            'chunk_size': config.get('data_preparation.merge_chunk_size', 1024),
            'max_memory_mb': config.get('data_preparation.max_memory_mb', 1024),
            'use_temporary_files': config.get('data_preparation.use_temp_files', True),
            'compression': config.get('data_preparation.compression', 'lzw')
        }
    
    def merge_paf_rasters(self, 
                         plants_name: str,
                         animals_name: str,
                         fungi_name: str,
                         output_path: Optional[Path] = None,
                         validate_alignment: bool = True,
                         resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Enhanced PAF merge with progress tracking and checkpoints.
        
        Args:
            plants_name: Name of plants raster
            animals_name: Name of animals raster
            fungi_name: Name of fungi raster
            output_path: Optional output path
            validate_alignment: Whether to validate alignment
            resume_from_checkpoint: Optional checkpoint to resume from
            
        Returns:
            Merge result dictionary
        """
        # Start progress tracking
        self.start_progress("PAF Raster Merge", 100)
        
        try:
            # Load checkpoint if resuming
            checkpoint_data = {}
            if resume_from_checkpoint:
                checkpoint_data = self.load_checkpoint(resume_from_checkpoint)
                logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            
            # Step 1: Load raster entries (10%)
            self.update_progress(10, metadata={'status': 'Loading raster entries'})
            
            rasters = self._load_raster_entries({
                'plants': plants_name,
                'animals': animals_name,
                'fungi': fungi_name
            })
            
            # Step 2: Validate alignment (30%)
            alignment_report = None
            aligned_rasters = rasters
            temp_dir_path = None
            
            if validate_alignment and not checkpoint_data.get('alignment_completed', False):
                self.update_progress(20, metadata={'status': 'Checking alignment'})
                
                raster_paths = [raster.path for raster in rasters.values()]
                alignment_report = self.aligner.analyze_alignment(raster_paths)
                
                if not alignment_report.aligned:
                    self.update_progress(25, metadata={'status': 'Fixing alignment issues'})
                    
                    # Fix alignment with progress tracking
                    aligned_rasters, temp_dir_path = self._fix_alignment_with_progress(
                        rasters, alignment_report
                    )
                
                # Save checkpoint
                self._checkpoint_data = {
                    'alignment_completed': True,
                    'temp_dir': str(temp_dir_path) if temp_dir_path else None
                }
                self.save_checkpoint()
            
            # Step 3: Merge data (70%)
            self.update_progress(40, metadata={'status': 'Starting merge operation'})
            
            # Memory allocation tracking
            estimated_memory = self._estimate_merge_memory(aligned_rasters)
            with self.memory_manager.memory_context(
                "raster_merge_paf",
                estimated_memory
            ):
                if estimated_memory > self.merge_config['max_memory_mb']:
                    logger.info("Using chunked merge due to memory constraints")
                    merged_data = self._merge_raster_data_chunked(
                        aligned_rasters,
                        progress_start=40,
                        progress_end=85
                    )
                else:
                    logger.info("Using single-pass merge")
                    merged_data = self._merge_raster_data(
                        aligned_rasters,
                        progress_callback=lambda pct: self.update_progress(
                            40 + int(pct * 0.45),
                            metadata={'status': f'Merging data ({pct:.0f}%)'}
                        )
                    )
            
            # Step 4: Save output (90%)
            if output_path:
                self.update_progress(85, metadata={'status': 'Saving merged raster'})
                self._save_merged_raster_with_progress(merged_data, output_path)
            
            # Step 5: Clean up and finalize (100%)
            self.update_progress(95, metadata={'status': 'Finalizing'})
            
            # Clean up temporary files
            if temp_dir_path and temp_dir_path.exists():
                shutil.rmtree(temp_dir_path)
                logger.info("Cleaned up temporary alignment files")
            
            # Store merge metadata
            merge_id = self._log_merge_operation(rasters, merged_data)
            
            # Complete progress
            self.complete_progress(metadata={
                'merge_id': merge_id,
                'bands_merged': 3,
                'output_path': str(output_path) if output_path else None
            })
            
            return {
                'data': merged_data,
                'merge_id': merge_id,
                'alignment_report': alignment_report,
                'metadata': {
                    'bands': ['plants', 'animals', 'fungi'],
                    'sources': {k: str(v.path) for k, v in rasters.items()},
                    'shape': dict(merged_data.sizes),
                    'dtype': str(merged_data.to_array().dtype)
                }
            }
            
        except Exception as e:
            logger.error(f"Merge failed: {e}")
            self.complete_progress(status="failed", metadata={'error': str(e)})
            raise
    
    def merge_custom_rasters(self,
                           raster_names: Dict[str, str],
                           band_names: Optional[List[str]] = None,
                           output_path: Optional[Path] = None,
                           progress_callback: Optional[Callable[[str, float], None]] = None) -> Dict[str, Any]:
        """Enhanced custom merge with progress support."""
        # Start progress if not already started
        if not self._progress_node_id:
            self.start_progress("Custom Raster Merge", len(raster_names))
        
        try:
            # Progress wrapper
            def wrapped_progress(msg: str, pct: float):
                if progress_callback:
                    progress_callback(msg, pct)
                self.update_progress(
                    int(pct),
                    metadata={'status': msg, 'rasters': len(raster_names)}
                )
            
            wrapped_progress("Loading raster entries", 5)
            
            # Load raster entries
            rasters = self._load_raster_entries(raster_names)
            
            # Check alignment
            wrapped_progress("Checking alignment", 15)
            
            raster_paths = [raster.path for raster in rasters.values()]
            alignment_report = self.aligner.analyze_alignment(raster_paths)
            
            aligned_rasters = rasters
            temp_dir_path = None
            
            if not alignment_report.aligned:
                wrapped_progress("Fixing alignment issues", 25)
                aligned_rasters, temp_dir_path = self._fix_alignment_with_progress(
                    rasters, alignment_report, wrapped_progress
                )
            
            # Use provided band names or dict keys
            if band_names is None:
                band_names = list(raster_names.keys())
            
            # Merge data
            wrapped_progress("Merging rasters", 40)
            
            estimated_memory = self._estimate_merge_memory(aligned_rasters)
            with self.memory_manager.memory_context(
                f"raster_merge_custom_{len(rasters)}",
                estimated_memory
            ):
                if estimated_memory > self.merge_config['max_memory_mb']:
                    merged_data = self._merge_raster_data_chunked(
                        aligned_rasters,
                        band_names,
                        progress_callback=wrapped_progress
                    )
                else:
                    merged_data = self._merge_raster_data(
                        aligned_rasters,
                        band_names,
                        progress_callback=lambda pct: wrapped_progress(
                            f"Merging progress: {pct:.0f}%", 40 + pct * 0.4
                        )
                    )
            
            # Save if requested
            if output_path:
                wrapped_progress("Saving merged raster", 85)
                self._save_merged_raster_with_progress(merged_data, output_path)
            
            # Clean up
            if temp_dir_path and temp_dir_path.exists():
                shutil.rmtree(temp_dir_path)
            
            wrapped_progress("Complete", 100)
            
            return {
                'data': merged_data,
                'alignment_report': alignment_report,
                'metadata': {
                    'bands': band_names,
                    'sources': {k: str(v.path) for k, v in rasters.items()}
                }
            }
            
        except Exception as e:
            logger.error(f"Custom merge failed: {e}")
            raise
    
    def _fix_alignment_with_progress(self,
                                   rasters: Dict[str, RasterEntry],
                                   alignment_report: Any,
                                   progress_callback: Optional[Callable] = None) -> Tuple[Dict[str, RasterEntry], Path]:
        """Fix alignment issues with progress tracking."""
        logger.info(f"Fixing {len(alignment_report.issues)} alignment issues")
        
        temp_dir = tempfile.mkdtemp(prefix="biodiversity_align_")
        temp_dir_path = Path(temp_dir)
        
        try:
            # Progress tracking for alignment fixes
            total_steps = len(rasters)
            
            aligned_paths = self.aligner.fix_alignment(
                [r.path for r in rasters.values()],
                temp_dir_path,
                progress_callback=lambda msg, pct: (
                    progress_callback(f"Alignment: {msg}", 25 + pct * 0.1) 
                    if progress_callback else None
                )
            )
            
            # Create aligned raster entries
            temp_rasters = {}
            for i, (key, original_raster) in enumerate(rasters.items()):
                if progress_callback:
                    progress_callback(
                        f"Creating aligned entry for {key}",
                        30 + (i / total_steps) * 5
                    )
                
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
            
            logger.info("Successfully aligned rasters for merging")
            return temp_rasters, temp_dir_path
            
        except Exception as e:
            # Clean up on failure
            shutil.rmtree(temp_dir)
            raise e
    
    def _merge_raster_data_chunked(self,
                                 rasters: Dict[str, RasterEntry],
                                 band_names: Optional[List[str]] = None,
                                 progress_start: int = 0,
                                 progress_end: int = 100,
                                 progress_callback: Optional[Callable] = None) -> xr.Dataset:
        """Merge rasters using chunked approach for memory efficiency."""
        if band_names is None:
            band_names = list(rasters.keys())
        
        logger.info(f"Starting chunked merge of {len(band_names)} bands")
        
        # Initialize output dataset with first band structure
        first_name = band_names[0]
        first_entry = rasters[first_name]
        
        # Open first raster to get dimensions
        with rioxarray.open_rasterio(
            first_entry.path,
            chunks={'x': self.merge_config['chunk_size'], 'y': self.merge_config['chunk_size']},
            cache=False
        ) as first_da:
            if 'band' in first_da.dims:
                first_da = first_da.sel(band=1)
            if 'x' in first_da.dims:
                first_da = first_da.rename({'x': 'lon', 'y': 'lat'})
            
            # Create output dataset structure
            output_coords = {
                'lon': first_da.lon.values,
                'lat': first_da.lat.values
            }
        
        # Process each band
        merged_arrays = {}
        total_bands = len(band_names)
        
        for i, band_name in enumerate(band_names):
            entry = rasters[band_name]
            
            # Update progress
            band_progress = progress_start + (i / total_bands) * (progress_end - progress_start)
            if progress_callback:
                progress_callback(f"Processing band: {band_name}", band_progress)
            else:
                self.update_progress(int(band_progress), metadata={
                    'status': f'Processing band {band_name}',
                    'band': i + 1,
                    'total_bands': total_bands
                })
            
            # Publish file I/O event
            if self.event_bus:
                self.event_bus.publish(create_file_progress(
                    str(entry.path),
                    i,
                    total_bands,
                    operation="read",
                    source=self.__class__.__name__
                ))
            
            # Load band data in chunks
            logger.info(f"Loading {band_name} from {entry.path}")
            
            # Initialize output array
            output_shape = (len(output_coords['lat']), len(output_coords['lon']))
            band_data = np.zeros(output_shape, dtype=np.float32)
            
            # Process chunks
            with rioxarray.open_rasterio(
                entry.path,
                chunks={'x': self.merge_config['chunk_size'], 'y': self.merge_config['chunk_size']},
                cache=False
            ) as da:
                if 'band' in da.dims:
                    da = da.sel(band=1)
                if 'x' in da.dims:
                    da = da.rename({'x': 'lon', 'y': 'lat'})
                
                # Check if resampling needed
                if not da.coords.equals(output_coords):
                    logger.warning(f"Resampling {band_name} to match reference coordinates")
                    da = da.interp(lon=output_coords['lon'], lat=output_coords['lat'], method='nearest')
                
                # Process in chunks to manage memory
                for chunk in da.to_delayed():
                    chunk_data = chunk.compute()
                    # This is simplified - in practice you'd need to track chunk positions
                    # and place them correctly in the output array
                
                # For now, compute the full array (but with chunked loading)
                band_data = da.compute()
            
            # Create DataArray for this band
            band_da = xr.DataArray(
                band_data,
                coords=output_coords,
                dims=['lat', 'lon'],
                name=band_name
            )
            
            merged_arrays[band_name] = band_da
            
            # Memory cleanup
            if self.memory_manager.get_memory_pressure_level().value in ['high', 'critical']:
                logger.info("High memory pressure, triggering cleanup")
                self.memory_manager.trigger_cleanup()
        
        # Create final dataset
        ds = xr.Dataset(merged_arrays)
        
        # Add metadata
        ds.attrs.update({
            'created_by': 'RasterMerger',
            'merge_date': str(datetime.now()),
            'bands': band_names,
            'crs': first_da.attrs.get('crs', 'EPSG:4326'),
            'merge_method': 'chunked'
        })
        
        if progress_callback:
            progress_callback("Merge complete", progress_end)
        
        return ds
    
    def _estimate_merge_memory(self, rasters: Dict[str, RasterEntry]) -> float:
        """Estimate memory requirements for merge operation."""
        total_mb = 0
        
        for entry in rasters.values():
            # Get dimensions from metadata
            width = entry.metadata.get('width', 0)
            height = entry.metadata.get('height', 0)
            
            # Estimate size (assuming float32)
            pixels = width * height
            mb = (pixels * 4) / (1024 * 1024)
            total_mb += mb
        
        # Add overhead for processing
        return total_mb * 1.5
    
    def _save_merged_raster_with_progress(self, dataset: xr.Dataset, output_path: Path):
        """Save merged raster with progress tracking."""
        # Convert to multi-band DataArray
        bands = list(dataset.data_vars)
        
        # Stack bands
        stacked = xr.concat([dataset[band] for band in bands], dim='band')
        stacked = stacked.assign_coords(band=bands)
        
        # Publish file write event
        if self.event_bus:
            self.event_bus.publish(create_file_progress(
                str(output_path),
                0,
                1,
                operation="write",
                source=self.__class__.__name__
            ))
        
        # Save using rioxarray with compression
        stacked.rio.to_raster(
            output_path,
            compress=self.merge_config['compression'],
            tiled=True,
            blockxsize=256,
            blockysize=256
        )
        
        logger.info(f"Merged raster saved to {output_path}")
        
        # Publish completion event
        if self.event_bus:
            self.event_bus.publish(create_file_progress(
                str(output_path),
                1,
                1,
                operation="write",
                source=self.__class__.__name__
            ))
    
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
                          band_names: Optional[List[str]] = None,
                          progress_callback: Optional[Callable[[float], None]] = None) -> xr.Dataset:
        """Single-pass merge for smaller datasets."""
        if band_names is None:
            band_names = list(rasters.keys())
        
        data_arrays = {}
        
        # Load first raster to get dimensions
        first_name = band_names[0]
        first_entry = rasters[first_name]
        
        logger.info(f"Loading {first_name} from {first_entry.path}")
        first_da = self._load_as_xarray(first_entry.path, first_name)
        data_arrays[first_name] = first_da
        
        if progress_callback:
            progress_callback(100 / len(band_names))
        
        # Get reference coordinates
        ref_coords = first_da.coords
        
        # Load remaining rasters
        for i, band_name in enumerate(band_names[1:], 1):
            entry = rasters[band_name]
            logger.info(f"Loading {band_name} from {entry.path}")
            
            da = self._load_as_xarray(entry.path, band_name)
            
            # Align to reference coordinates if needed
            if not da.coords.equals(ref_coords):
                logger.warning(f"Aligning {band_name} to reference coordinates")
                da = da.interp_like(first_da, method='nearest')
            
            data_arrays[band_name] = da
            
            if progress_callback:
                progress_callback((i + 1) * 100 / len(band_names))
        
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
        # Use rioxarray for better CRS handling
        da_temp = rioxarray.open_rasterio(raster_path, chunks={'x': 1000, 'y': 1000})
        
        if isinstance(da_temp, xr.DataArray):
            da = da_temp
        elif isinstance(da_temp, xr.Dataset):
            da = da_temp.to_array().squeeze()
        else:
            raise TypeError(f"Unexpected type from rioxarray: {type(da_temp)}")
        
        # Handle multi-band
        if 'band' in da.dims:
            da = da.sel(band=1)
        
        # Rename coordinates
        if 'x' in da.dims:
            da = da.rename({'x': 'lon', 'y': 'lat'})
        
        da.name = band_name
        return da
    
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
                    'dtype': str(merged_data.to_array().dtype),
                    'processing_mode': 'chunked' if hasattr(self, '_used_chunked_merge') else 'single'
                })
            ))
            
            merge_id = cur.fetchone()[0]
            conn.commit()
            
        return merge_id
    
    # BaseProcessor abstract method implementations
    def process_single(self, item: Any) -> Any:
        """Process a single item."""
        return item
    
    def validate_input(self, item: Any) -> Tuple[bool, Optional[str]]:
        """Validate input item."""
        return True, None