"""Raster data management coordinator."""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime
import json

from .metadata import RasterMetadataExtractor
from .processor import RasterProcessor
from ..database.schema import schema
from ..config import config

logger = logging.getLogger(__name__)

class RasterManager:
    """Coordinates raster operations between processing and database layers."""
    
    def __init__(self):
        self.config = config.raster_processing
        self.metadata_extractor = RasterMetadataExtractor()
        self.processor = RasterProcessor(
            tile_size=self.config['tile_size'],
            memory_limit_mb=self.config['memory_limit_mb']
        )
        
    def register_raster_source(self, file_path: Union[str, Path], 
                              name: Optional[str] = None,
                              metadata: Optional[Dict] = None) -> str:
        """Register a new raster data source."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Raster file not found: {file_path}")
        
        # Generate name if not provided
        if not name:
            name = file_path.stem
            
        # Validate file format
        if not self.metadata_extractor.validate_file_format(file_path):
            raise ValueError(f"Unsupported raster format: {file_path.suffix}")
        
        try:
            # Extract basic metadata
            raster_info = self.metadata_extractor.extract_metadata(file_path)
            
            # Calculate file checksum for integrity checking
            checksum = self.metadata_extractor.calculate_file_checksum(file_path)
            
            # Prepare raster data for storage
            raster_data = {
                'name': name,
                'file_path': str(file_path.absolute()),
                'data_type': raster_info['data_type'],
                'pixel_size_degrees': raster_info['pixel_size_degrees'],
                'spatial_extent_wkt': raster_info['spatial_extent_wkt'],
                'nodata_value': raster_info.get('nodata_value'),
                'band_count': raster_info['band_count'],
                'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                'checksum': checksum,
                'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime),
                'source_dataset': metadata.get('source_dataset') if metadata else None,
                'variable_name': metadata.get('variable_name') if metadata else None,
                'units': metadata.get('units') if metadata else None,
                'description': metadata.get('description') if metadata else None,
                'temporal_info': metadata.get('temporal_info', {}) if metadata else {},
                'metadata': metadata or {}
            }
            
            # Store in database via schema layer
            raster_id = schema.store_raster_source(raster_data)
            
            # Queue for tiling if large enough
            if raster_data['file_size_mb'] > self.config.get('auto_tile_threshold_mb', 100):
                schema.add_processing_task(
                    queue_type='raster_tiling',
                    raster_id=raster_id,
                    parameters={'tile_size': self.config['tile_size']},
                    priority=1
                )
            
            logger.info(f"✅ Registered raster source: {name}")
            return raster_id
            
        except Exception as e:
            logger.error(f"❌ Failed to register raster source {name}: {e}")
            raise
    
    def create_raster_tiles(self, raster_id: str, force_retile: bool = False) -> int:
        """Create spatial tiles for a raster source."""
        try:
            # Get raster source info
            raster_sources = schema.get_raster_sources(active_only=True)
            raster_source = next((rs for rs in raster_sources if rs['id'] == raster_id), None)
            
            if not raster_source:
                raise ValueError(f"Raster source not found: {raster_id}")
            
            # Check if already tiled
            if not force_retile and raster_source['processing_status'] in ['ready', 'tiling']:
                logger.info(f"Raster {raster_source['name']} already tiled or in progress")
                return 0
            
            # Update status
            schema.update_raster_processing_status(raster_id, 'tiling')
            
            # Generate tiles using processor
            tiles_data = self.processor.generate_tile_metadata(raster_source)
            
            # Store tiles in database
            count = schema.store_raster_tiles_batch(raster_id, tiles_data)
            
            # Update status to ready
            schema.update_raster_processing_status(raster_id, 'ready', {
                'tiling_completed_at': datetime.now().isoformat(),
                'tile_count': count
            })
            
            logger.info(f"✅ Created {count} tiles for raster {raster_source['name']}")
            return count
            
        except Exception as e:
            schema.update_raster_processing_status(raster_id, 'error', {
                'error_message': str(e),
                'error_at': datetime.now().isoformat()
            })
            logger.error(f"❌ Failed to create tiles for raster {raster_id}: {e}")
            raise
    
    def resample_to_grid(self, raster_id: str, grid_id: str, 
                        cell_ids: Optional[List[str]] = None,
                        method: str = 'bilinear',
                        band_number: int = 1,
                        use_cache: bool = True) -> Dict[str, float]:
        """Resample raster values to grid cells."""
        try:
            # Validate resampling method
            if not self.processor.validate_resampling_method(method):
                raise ValueError(f"Unsupported resampling method: {method}")
            
            # Check cache first if enabled
            if use_cache and cell_ids:
                cached_values = schema.get_cached_resampling_values(
                    raster_id, grid_id, cell_ids, method, band_number
                )
                if cached_values:
                    logger.info(f"Found {len(cached_values)} cached values for resampling")
                    # Return cached values and compute missing ones
                    missing_cells = [cid for cid in cell_ids if cid not in cached_values]
                    if not missing_cells:
                        return cached_values
                    cell_ids = missing_cells
            else:
                cached_values = {}
            
            # Get grid cells that need resampling
            if not cell_ids:
                grid_cells = schema.get_grid_cells(grid_id)
                cell_ids = [cell['cell_id'] for cell in grid_cells]
            
            # Perform resampling using processor
            resampled_values = self.processor.perform_resampling(
                raster_id, grid_id, cell_ids, method, band_number
            )
            
            # Cache results if enabled
            if use_cache and resampled_values:
                cache_data = []
                for cell_id, value in resampled_values.items():
                    cache_data.append({
                        'source_raster_id': raster_id,
                        'target_grid_id': grid_id,
                        'cell_id': cell_id,
                        'method': method,
                        'band_number': band_number,
                        'value': value,
                        'confidence_score': 1.0,  # Would be calculated based on actual resampling
                        'source_tiles_used': [],  # Would track which tiles were used
                        'computation_metadata': {
                            'resampled_at': datetime.now().isoformat(),
                            'method_used': method
                        }
                    })
                
                schema.store_resampling_cache_batch(cache_data)
                logger.info(f"Cached {len(cache_data)} resampling results")
            
            # Combine cached and newly computed values
            all_values = {**cached_values, **resampled_values}
            
            return all_values
            
        except Exception as e:
            logger.error(f"❌ Failed to resample raster {raster_id} to grid {grid_id}: {e}")
            raise
    
    def process_queue_task(self, queue_type: str, worker_id: str) -> bool:
        """Process next task from the queue."""
        task = schema.get_next_processing_task(queue_type, worker_id)
        
        if not task:
            logger.debug(f"No {queue_type} tasks available for worker {worker_id}")
            return False
        
        try:
            logger.info(f"Processing {queue_type} task {task['id']} with worker {worker_id}")
            
            if queue_type == 'raster_tiling':
                # Process tiling task
                self.create_raster_tiles(task['raster_source_id'])
                
            elif queue_type == 'resampling':
                # Process resampling task
                params = json.loads(task['parameters']) if task['parameters'] else {}
                self.resample_to_grid(
                    task['raster_source_id'],
                    task['grid_id'],
                    cell_ids=params.get('cell_ids'),
                    method=params.get('method', 'bilinear')
                )
            
            # Mark task as completed
            schema.complete_processing_task(task['id'], success=True)
            logger.info(f"✅ Completed task {task['id']}")
            return True
            
        except Exception as e:
            # Mark task as failed
            schema.complete_processing_task(task['id'], success=False, error_message=str(e))
            logger.error(f"❌ Failed to process task {task['id']}: {e}")
            return False
    
    # Convenience methods that delegate to schema
    def get_processing_status(self, raster_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get raster processing status."""
        return schema.get_raster_processing_status(raster_id)
    
    def get_cache_statistics(self, raster_id: Optional[str] = None, 
                           grid_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get cache efficiency statistics."""
        return schema.get_cache_efficiency_summary(raster_id, grid_id)
    
    def cleanup_cache(self, days_old: Optional[int] = None, 
                     min_access_count: Optional[int] = None) -> int:
        """Clean up old cache entries."""
        actual_days_old = days_old if days_old is not None else self.config['cache_ttl_days']
        actual_min_access = min_access_count if min_access_count is not None else 1
            
        return schema.cleanup_old_cache(actual_days_old, actual_min_access)

# Global raster manager instance
raster_manager = RasterManager()
