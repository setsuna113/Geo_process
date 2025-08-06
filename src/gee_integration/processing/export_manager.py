"""GEE Export Manager

Handles exporting merged data from GEE.
"""

import ee
from typing import Dict, Optional, List, Tuple
import time
import json

from src.infrastructure.logging import get_logger
from src.infrastructure.logging.decorators import log_operation

logger = get_logger(__name__)


class ExportManager:
    """Manages GEE export tasks"""
    
    def __init__(self, config: Dict):
        """
        Initialize export manager.
        
        Args:
            config: Export configuration
        """
        self.config = config
        self.logger = logger
        self.active_tasks = {}
    
    @log_operation("export_to_csv")
    def export_sampled_data(self,
                          merged_image: ee.Image,
                          sampling_points: ee.FeatureCollection,
                          resolution_name: str,
                          description: Optional[str] = None) -> str:
        """
        Export sampled data as CSV for parquet conversion.
        
        Args:
            merged_image: Merged multi-band image
            sampling_points: Points to sample at
            resolution_name: Resolution identifier (e.g., '1_8km')
            description: Export description
            
        Returns:
            Task ID
        """
        # Sample the image at points
        self.logger.info(f"Sampling merged image")
        
        sampled = merged_image.sampleRegions(
            collection=sampling_points,
            scale=0.016667 * 111320,  # Default ~1.8km resolution in meters
            projection='EPSG:4326',
            tileScale=self.config.get('tile_scale', 16)
        )
        
        # Prepare export
        if not description:
            description = f'biodiversity_climate_{resolution_name}'
        
        export_path = f"{self.config.get('export_folder', 'merged_outputs')}/{resolution_name}/merged_data"
        
        # Create export task
        if self.config.get('export_destination', 'cloud_storage') == 'cloud_storage':
            task = ee.batch.Export.table.toCloudStorage(
                collection=sampled,
                description=description,
                bucket=self.config.get('gcs_bucket', 'biodiv-gee-yl998'),
                fileNamePrefix=export_path,
                fileFormat='CSV'
            )
        else:
            # Export to Drive
            task = ee.batch.Export.table.toDrive(
                collection=sampled,
                description=description,
                folder=self.config.get('export_folder', 'merged_outputs'),
                fileNamePrefix=f'merged_data_{resolution_name}',
                fileFormat='CSV'
            )
        
        # Start task
        task.start()
        task_id = task.id
        
        # Track task
        self.active_tasks[task_id] = {
            'task': task,
            'description': description,
            'resolution': resolution_name,
            'export_path': export_path,
            'start_time': time.time()
        }
        
        self.logger.info(f"🚀 Export task started: {task_id}")
        self.logger.info(f"   Resolution: {resolution_name}")
        self.logger.info(f"   Export path: {export_path}")
        
        return task_id
    
    def _export_as_geotiff(self,
                          merged_image: ee.Image,
                          bounds: ee.Geometry,
                          resolution_name: str,
                          resolution_config: Dict,
                          description: Optional[str] = None) -> str:
        """
        Export merged image as GeoTIFF.
        
        Args:
            merged_image: Merged multi-band image
            bounds: Export bounds as ee.Geometry
            resolution_name: Resolution identifier
            resolution_config: Resolution configuration dict
            description: Export description
            
        Returns:
            Task ID or list of task IDs if hemisphere splitting is enabled
        """
        if not description:
            description = f'biodiversity_climate_{resolution_name}'
            
        export_path = f"{self.config.get('export_folder', 'merged_outputs')}/{resolution_name}/merged_data"
        
        # Get scale in meters
        scale_meters = resolution_config.get('resolution', 0.016667) * 111320  # degrees to meters at equator
        
        # Check if hemisphere splitting is needed/enabled
        hemisphere_config = self.config.get('hemisphere_split', {})
        use_hemisphere_split = hemisphere_config.get('enabled', False)
        
        # Auto-detect if enabled (not recommended)
        if hemisphere_config.get('auto_detect', False) and not use_hemisphere_split:
            # Check if this is a global export at coarse resolution
            coords = bounds.coordinates().getInfo()[0]
            min_lon = min(c[0] for c in coords)
            max_lon = max(c[0] for c in coords)
            
            # Auto-enable for global exports at coarse resolution
            if (max_lon - min_lon) > 300 and resolution_name == 'coarse':
                use_hemisphere_split = True
                self.logger.info("Auto-enabled hemisphere splitting for global coarse export")
        
        # Handle hemisphere splitting if enabled
        if use_hemisphere_split:
            # For hemisphere splitting, use fixed global bounds
            # Don't try to extract from geometry which can have coordinate issues
            # Use latitude bounds from masking config if available
            if hasattr(self.config, 'masking') and hasattr(self.config.masking, 'enabled') and self.config.masking.enabled:
                if hasattr(self.config.masking, 'latitude_bounds'):
                    min_lat = self.config.masking.latitude_bounds.get('min', -60)
                    max_lat = self.config.masking.latitude_bounds.get('max', 75)
                else:
                    min_lat = -60
                    max_lat = 75
            else:
                min_lat = -90
                max_lat = 90
            
            min_lon = -180
            max_lon = 180
            
            # Always do hemisphere split when enabled (config already controls this)
            if True:  # Simplified - config controls whether we get here
                self.logger.info("Using hemisphere split for global export (enabled in config)")
                
                # Export in two hemispheres
                task_ids = []
                
                # Add timestamp to make descriptions unique
                timestamp = int(time.time())
                
                # Western hemisphere - use non-geodesic polygon to avoid dateline issues
                # Using -179.999 instead of -180 to avoid boundary issues
                west_bounds = ee.Geometry.Polygon(
                    [[[-179.999, min_lat], [-179.999, max_lat], [0, max_lat], [0, min_lat], [-179.999, min_lat]]],
                    None, False  # geodesic=False is crucial for global exports
                )
                west_path = f"{export_path}_west"
                
                task_west = ee.batch.Export.image.toCloudStorage(
                    image=merged_image,
                    description=f"{description}_west_{timestamp}",
                    bucket=self.config.get('gcs_bucket', 'biodiv-gee-yl998'),
                    fileNamePrefix=west_path,
                    region=west_bounds,
                    scale=scale_meters,
                    crs='EPSG:4326',
                    fileFormat='GeoTIFF',
                    formatOptions=self.config.get('file_format_options', {
                        'cloudOptimized': True,
                        'noData': 0  # Use 0 to match original biodiversity data NoData value
                    }),
                    maxPixels=self.config.get('export_max_pixels', 1e13),
                    shardSize=256  # Add sharding to help with large exports
                )
                task_west.start()
                task_ids.append(task_west.id)
                self.logger.info(f"   Started western hemisphere export: {task_west.id}")
                
                # Eastern hemisphere - use non-geodesic polygon  
                # Using 179.999 instead of 180 to avoid boundary issues
                east_bounds = ee.Geometry.Polygon(
                    [[[0, min_lat], [0, max_lat], [179.999, max_lat], [179.999, min_lat], [0, min_lat]]],
                    None, False  # geodesic=False is crucial for global exports
                )
                east_path = f"{export_path}_east"
                
                task_east = ee.batch.Export.image.toCloudStorage(
                    image=merged_image,
                    description=f"{description}_east_{timestamp}",
                    bucket=self.config.get('gcs_bucket', 'biodiv-gee-yl998'),
                    fileNamePrefix=east_path,
                    region=east_bounds,
                    scale=scale_meters,
                    crs='EPSG:4326',
                    fileFormat='GeoTIFF',
                    formatOptions=self.config.get('file_format_options', {
                        'cloudOptimized': True,
                        'noData': 0  # Use 0 to match original biodiversity data NoData value
                    }),
                    maxPixels=self.config.get('export_max_pixels', 1e13),
                    shardSize=256  # Add sharding to help with large exports
                )
                task_east.start()
                task_ids.append(task_east.id)
                self.logger.info(f"   Started eastern hemisphere export: {task_east.id}")
                
                # Store both task IDs
                task_id = ','.join(task_ids)  # Return comma-separated task IDs
                
                # Track both tasks
                for tid in task_ids:
                    self.active_tasks[tid] = {
                        'task': task_west if tid == task_west.id else task_east,
                        'description': description,
                        'resolution': resolution_name,
                        'export_path': export_path,
                        'start_time': time.time(),
                        'format': 'GeoTIFF',
                        'hemisphere': 'west' if tid == task_west.id else 'east'
                    }
                
                return task_id  # Return the comma-separated task IDs
        
        # Standard single export (no hemisphere splitting)
        self.logger.info(f"Exporting as single GeoTIFF: {resolution_name}")
        task = ee.batch.Export.image.toCloudStorage(
            image=merged_image,
            description=description,
            bucket=self.config.get('gcs_bucket', 'biodiv-gee-yl998'),
            fileNamePrefix=export_path,
            region=bounds,
            scale=scale_meters,
            crs='EPSG:4326',
            fileFormat='GeoTIFF',
            formatOptions=self.config.get('file_format_options', {
                'cloudOptimized': True,
                'noData': 0  # Use 0 to match original biodiversity data NoData value
            }),
            maxPixels=self.config.get('export_max_pixels', 1e13),
            shardSize=256  # Add sharding to help with large exports
        )
        
        task.start()
        task_id = task.id
        
        # Track task
        self.active_tasks[task_id] = {
            'task': task,
            'description': description,
            'resolution': resolution_name,
            'export_path': export_path,
            'start_time': time.time(),
            'format': 'GeoTIFF'
        }
        
        self.logger.info(f"🚀 GeoTIFF export task started: {task_id}")
        self.logger.info(f"   Resolution: {resolution_name} ({scale_meters:.1f}m)")
        self.logger.info(f"   Export path: {export_path}")
        
        return task_id
    
    def export_for_both_resolutions(self,
                                  merged_image: ee.Image,
                                  bounds: Tuple[float, float, float, float],
                                  resolutions: Dict[str, Dict]) -> Dict[str, str]:
        """
        Export data at both configured resolutions.
        
        Args:
            merged_image: Merged image to export
            bounds: Export bounds
            resolutions: Resolution configurations
            
        Returns:
            Dict mapping resolution name to task ID
        """
        task_ids = {}
        export_format = self.config.get('export_format', 'CSV')
        
        for res_name, res_config in resolutions.items():
            self.logger.info(f"Preparing {export_format} export for {res_name} ({res_config['description']})")
            
            if export_format == 'GeoTIFF':
                # Convert bounds tuple to ee.Geometry
                ee_bounds = ee.Geometry.Rectangle(bounds)
                
                # Export as GeoTIFF
                task_id = self._export_as_geotiff(
                    merged_image=merged_image,
                    bounds=ee_bounds,
                    resolution_name=res_name,
                    resolution_config=res_config,
                    description=f'biodiv_climate_{res_name}_{int(time.time())}'
                )
            else:
                # Original CSV export with sampling
                from .merger import GEEMerger
                merger = GEEMerger()
                
                sampling_grid = merger.create_sampling_grid(
                    bounds=bounds,
                    resolution=res_config['resolution']
                )
                
                self.logger.info(f"   Created sampling grid for resolution {res_name}")
                
                task_id = self.export_sampled_data(
                    merged_image=merged_image,
                    sampling_points=sampling_grid,
                    resolution_name=res_name,
                    description=f'biodiv_climate_{res_name}_{int(time.time())}'
                )
            
            task_ids[res_name] = task_id
        
        return task_ids
    
    @log_operation("monitor_export_tasks")
    def monitor_tasks(self, task_ids: List[str], 
                     check_interval: int = 30,
                     timeout: int = 7200) -> Dict[str, Dict]:
        """
        Monitor multiple export tasks.
        
        Args:
            task_ids: List of task IDs to monitor (may contain comma-separated IDs for hemisphere exports)
            check_interval: Check interval in seconds
            timeout: Overall timeout in seconds
            
        Returns:
            Dict mapping task ID to final status
        """
        start_time = time.time()
        
        # Expand comma-separated task IDs
        all_task_ids = []
        for task_id_str in task_ids:
            if ',' in task_id_str:
                all_task_ids.extend(task_id_str.split(','))
            else:
                all_task_ids.append(task_id_str)
        
        task_statuses = {tid: {'status': 'RUNNING'} for tid in all_task_ids}
        
        self.logger.info(f"Monitoring {len(all_task_ids)} export tasks...")
        
        while True:
            # Check each task
            all_complete = True
            
            for task_id in all_task_ids:
                if task_statuses[task_id]['status'] in ['COMPLETED', 'FAILED', 'CANCELLED']:
                    continue
                
                all_complete = False
                
                # Get task info
                if task_id in self.active_tasks:
                    task = self.active_tasks[task_id]['task']
                    status = task.status()
                    
                    task_statuses[task_id].update({
                        'status': status['state'],
                        'progress': status.get('progress', 0),
                        'error_message': status.get('error_message'),
                        'duration': time.time() - self.active_tasks[task_id]['start_time']
                    })
                    
                    # Log progress
                    if status['state'] == 'RUNNING':
                        progress = status.get('progress', 0) * 100
                        hemisphere = self.active_tasks[task_id].get('hemisphere', '')
                        hemisphere_str = f" ({hemisphere})" if hemisphere else ""
                        self.logger.debug(f"Task {task_id}{hemisphere_str}: {progress:.1f}% complete")
                    elif status['state'] == 'COMPLETED':
                        duration = task_statuses[task_id]['duration']
                        hemisphere = self.active_tasks[task_id].get('hemisphere', '')
                        hemisphere_str = f" ({hemisphere})" if hemisphere else ""
                        self.logger.info(f"✅ Task {task_id}{hemisphere_str} completed in {duration/60:.1f} minutes")
                    elif status['state'] == 'FAILED':
                        self.logger.error(f"❌ Task {task_id} failed: {status.get('error_message')}")
                    elif status['state'] == 'CANCELLED':
                        self.logger.warning(f"⚠️ Task {task_id} was cancelled")
            
            if all_complete:
                break
            
            # Check timeout
            if time.time() - start_time > timeout:
                self.logger.error(f"Export monitoring timed out after {timeout}s")
                for task_id in all_task_ids:
                    if task_statuses[task_id]['status'] == 'RUNNING':
                        task_statuses[task_id]['status'] = 'TIMEOUT'
                break
            
            time.sleep(check_interval)
        
        # Summary
        completed = sum(1 for s in task_statuses.values() if s['status'] == 'COMPLETED')
        failed = sum(1 for s in task_statuses.values() if s['status'] in ['FAILED', 'TIMEOUT'])
        
        self.logger.info(f"Export tasks complete: {completed}/{len(all_task_ids)} successful, {failed} failed")
        
        return task_statuses
    
    def get_export_paths(self, task_ids: Dict[str, str]) -> Dict[str, str]:
        """
        Get export paths for completed tasks.
        
        Args:
            task_ids: Dict mapping resolution name to task ID
            
        Returns:
            Dict mapping resolution name to export path
        """
        export_paths = {}
        
        for res_name, task_id_str in task_ids.items():
            # Handle comma-separated task IDs for hemisphere exports
            if ',' in task_id_str:
                # Hemisphere export - just return the base path
                # The fetcher will look for _west and _east files
                task_ids_list = task_id_str.split(',')
                if task_ids_list[0] in self.active_tasks:
                    task_info = self.active_tasks[task_ids_list[0]]
                    base_path = task_info['export_path'].replace('_west', '').replace('_east', '')
                    if self.config.get('export_destination', 'cloud_storage') == 'cloud_storage':
                        export_paths[res_name] = f"gs://{self.config.get('gcs_bucket', 'biodiv-gee-yl998')}/{base_path}"
                    else:
                        export_paths[res_name] = base_path
            else:
                # Single task
                if task_id_str in self.active_tasks:
                    task_info = self.active_tasks[task_id_str]
                    if self.config.get('export_destination', 'cloud_storage') == 'cloud_storage':
                        export_paths[res_name] = f"gs://{self.config.get('gcs_bucket', 'biodiv-gee-yl998')}/{task_info['export_path']}"
                    else:
                        export_paths[res_name] = task_info['export_path']
        
        return export_paths