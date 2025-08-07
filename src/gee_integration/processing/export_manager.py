"""GEE Export Manager

Handles exporting merged data from GEE with support for multiple resolutions
and hemisphere splitting for large exports.
"""

import ee
from typing import Dict, Optional, List, Tuple, Union
import time

from src.infrastructure.logging import get_logger
from src.infrastructure.logging.decorators import log_operation
from .task_monitor import GEETaskMonitor
from config.gee.constants import (
    DEGREES_TO_METERS,
    MAX_PIXELS,
    NODATA_ZERO,
    EXPORT_TILE_SIZE,
    DEFAULT_LAT_MIN,
    DEFAULT_LAT_MAX,
    GLOBAL_LON_MIN,
    GLOBAL_LON_MAX,
    GLOBAL_LAT_MAX
)

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
        self.task_monitor = GEETaskMonitor()
    
    def export_image(self,
                    merged_image: ee.Image,
                    bounds: ee.Geometry,
                    resolution_name: str,
                    resolution_config: Dict,
                    description: Optional[str] = None,
                    export_base_path: Optional[str] = None) -> Union[str, List[str]]:
        """
        Export merged image as GeoTIFF.
        
        This is the main public API for exporting images.
        
        Args:
            merged_image: Merged multi-band image
            bounds: Export bounds as ee.Geometry
            resolution_name: Resolution identifier (e.g., '1x', '4x')
            resolution_config: Resolution configuration dict with 'resolution' and 'description'
            description: Export description
            export_base_path: Base path for export
            
        Returns:
            Task ID string or list of task IDs if hemisphere splitting is used
        """
        if not description:
            description = f'export_{resolution_name}_{int(time.time())}'
            
        # Use experiment-specific path if provided, otherwise fall back to config
        base_folder = export_base_path or self.config.get('export_folder', 'merged_outputs')
        export_path = f"{base_folder}/{resolution_name}/merged_data"
        
        # Get scale in meters
        scale_meters = resolution_config.get('resolution', 0.016667) * DEGREES_TO_METERS  # degrees to meters at equator
        
        # Check if hemisphere splitting is needed/enabled
        hemisphere_config = self.config.get('hemisphere_split', {})
        use_hemisphere_split = hemisphere_config.get('enabled', False)
        
        # Handle hemisphere splitting if enabled
        if use_hemisphere_split:
            return self._export_with_hemisphere_split(
                merged_image, bounds, resolution_name, resolution_config,
                description, export_path, scale_meters
            )
        else:
            return self._export_single_region(
                merged_image, bounds, resolution_name,
                description, export_path, scale_meters
            )
    
    def _export_with_hemisphere_split(self,
                                     merged_image: ee.Image,
                                     bounds: ee.Geometry,
                                     resolution_name: str,
                                     resolution_config: Dict,
                                     description: str,
                                     export_path: str,
                                     scale_meters: float) -> List[str]:
        """
        Export image split into hemispheres to avoid antimeridian issues.
        
        Returns:
            List of task IDs [west_task_id, east_task_id]
        """
        # Get latitude bounds from masking config if available
        if hasattr(self.config, 'masking') and hasattr(self.config.masking, 'enabled') and self.config.masking.enabled:
            if hasattr(self.config.masking, 'latitude_bounds'):
                min_lat = self.config.masking.latitude_bounds.get('min', DEFAULT_LAT_MIN)
                max_lat = self.config.masking.latitude_bounds.get('max', DEFAULT_LAT_MAX)
            else:
                min_lat, max_lat = DEFAULT_LAT_MIN, DEFAULT_LAT_MAX
        else:
            min_lat, max_lat = -90, 90
        
        self.logger.info(f"Using hemisphere split for export (lat: [{min_lat}, {max_lat}])")
        
        task_ids = []
        timestamp = int(time.time())
        
        # Calculate crsTransform for consistent pixel alignment
        scale_degrees = scale_meters / DEGREES_TO_METERS
        crs_transform = [scale_degrees, 0, GLOBAL_LON_MIN, 0, -scale_degrees, GLOBAL_LAT_MAX]
        
        # Western hemisphere (GLOBAL_LON_MIN to 0)
        west_bounds = ee.Geometry.Polygon(
            [[[-179.999, min_lat], [-179.999, max_lat], [0, max_lat], [0, min_lat], [-179.999, min_lat]]],
            None, False  # geodesic=False is crucial for global exports
        )
        
        task_west = self._create_export_task(
            image=merged_image,
            description=f"{description}_west_{timestamp}",
            file_prefix=f"{export_path}_west",
            region=west_bounds,
            scale=scale_meters,
            crs_transform=crs_transform
        )
        task_west.start()
        task_ids.append(task_west.id)
        self.logger.info(f"   Started western hemisphere: {task_west.id}")
        
        # Eastern hemisphere (0 to GLOBAL_LON_MAX)
        east_bounds = ee.Geometry.Polygon(
            [[[0, min_lat], [0, max_lat], [179.999, max_lat], [179.999, min_lat], [0, min_lat]]],
            None, False
        )
        
        task_east = self._create_export_task(
            image=merged_image,
            description=f"{description}_east_{timestamp}",
            file_prefix=f"{export_path}_east",
            region=east_bounds,
            scale=scale_meters,
            crs_transform=crs_transform
        )
        task_east.start()
        task_ids.append(task_east.id)
        self.logger.info(f"   Started eastern hemisphere: {task_east.id}")
        
        # Track tasks
        for task_id, hemisphere in zip(task_ids, ['west', 'east']):
            self.active_tasks[task_id] = {
                'description': f"{description}_{hemisphere}",
                'resolution': resolution_name,
                'hemisphere': hemisphere,
                'export_path': f"{export_path}_{hemisphere}",
                'start_time': time.time()
            }
        
        return task_ids
    
    def _export_single_region(self,
                             merged_image: ee.Image,
                             bounds: ee.Geometry,
                             resolution_name: str,
                             description: str,
                             export_path: str,
                             scale_meters: float) -> str:
        """
        Export image as a single region.
        
        Returns:
            Task ID string
        """
        # Calculate crsTransform for consistent pixel alignment
        scale_degrees = scale_meters / DEGREES_TO_METERS
        crs_transform = [scale_degrees, 0, GLOBAL_LON_MIN, 0, -scale_degrees, GLOBAL_LAT_MAX]
        
        task = self._create_export_task(
            image=merged_image,
            description=description,
            file_prefix=export_path,
            region=bounds,
            scale=scale_meters,
            crs_transform=crs_transform
        )
        task.start()
        task_id = task.id
        
        # Track task
        self.active_tasks[task_id] = {
            'description': description,
            'resolution': resolution_name,
            'export_path': export_path,
            'start_time': time.time()
        }
        
        self.logger.info(f"   Started export task: {task_id}")
        return task_id
    
    def _create_export_task(self,
                           image: ee.Image,
                           description: str,
                           file_prefix: str,
                           region: ee.Geometry,
                           scale: float,
                           crs_transform: List[float]) -> ee.batch.Task:
        """
        Create an export task with consistent settings.
        
        Returns:
            Unstarted export task
        """
        export_params = {
            'image': image,
            'description': description,
            'bucket': self.config.get('gcs_bucket', 'biodiv-gee-yl998'),
            'fileNamePrefix': file_prefix,
            'region': region,
            'scale': scale,
            'crs': 'EPSG:4326',
            'crsTransform': crs_transform,
            'fileFormat': 'GeoTIFF',
            'formatOptions': self.config.get('file_format_options', {
                'cloudOptimized': True,
                'noData': NODATA_ZERO  # Use 0 to match original biodiversity data NoData value
            }),
            'maxPixels': self.config.get('export_max_pixels', MAX_PIXELS)
        }
        
        return ee.batch.Export.image.toCloudStorage(**export_params)
    
    @log_operation("monitor_export_tasks")
    def monitor_tasks(self,
                     task_ids: Union[Dict[str, Union[str, List[str]]], List[str]],
                     timeout: Optional[int] = None) -> Dict[str, Dict]:
        """
        Monitor export tasks until completion using centralized GEETaskMonitor.
        
        Args:
            task_ids: Either a dict mapping resolution names to task IDs,
                     or a list of task IDs
            timeout: Overall timeout in seconds (default from config)
            
        Returns:
            Dict mapping task ID to final status
        """
        # Normalize input to dict format
        if isinstance(task_ids, list):
            task_dict = {f"task_{i}": tid for i, tid in enumerate(task_ids)}
        else:
            task_dict = task_ids
        
        # Use config timeout if not provided
        if timeout is None:
            timeout = self.config.get('export_timeout', 7200)
        
        check_interval = self.config.get('export_check_interval', 30)
        
        self.logger.info(f"⏳ Monitoring {len(task_dict)} export tasks (timeout: {timeout}s)...")
        
        # Create task objects for monitoring
        tasks_to_monitor = {}
        for res_name, task_id in task_dict.items():
            if isinstance(task_id, list):
                # Hemisphere split - monitor both
                for i, tid in enumerate(task_id):
                    hemisphere = ['west', 'east'][i]
                    desc = f"{res_name} ({hemisphere})"
                    # Create a mock task object that GEETaskMonitor can use
                    task = type('Task', (), {
                        'id': tid,
                        'status': lambda: ee.data.getTaskStatus(tid)[0]
                    })()
                    tasks_to_monitor[desc] = task
            else:
                # Create a mock task object
                task = type('Task', (), {
                    'id': task_id,
                    'status': lambda tid=task_id: ee.data.getTaskStatus(tid)[0]
                })()
                tasks_to_monitor[res_name] = task
        
        # Use GEETaskMonitor's batch monitoring
        start_time = time.time()
        try:
            final_statuses = self.task_monitor.batch_monitor(
                tasks_to_monitor,
                check_interval=check_interval
            )
            
            # Check for failures
            for desc, status in final_statuses.items():
                if status.get('state') == 'FAILED':
                    error_msg = status.get('error_message', 'Unknown error')
                    raise RuntimeError(f"Export failed for {desc}: {error_msg}")
            
            # Convert to expected format
            result = {}
            for desc, task in tasks_to_monitor.items():
                status = final_statuses.get(desc, {})
                result[task.id] = {
                    'state': status.get('state', 'UNKNOWN'),
                    'duration': int(time.time() - start_time),
                    'resolution': desc.split(' (')[0] if ' (' in desc else desc,
                    'hemisphere': desc.split('(')[1].rstrip(')') if '(' in desc else None
                }
            
            return result
            
        except TimeoutError:
            raise RuntimeError(f"Export monitoring timeout after {timeout}s")
    
    def get_export_paths(self, task_ids: Dict[str, Union[str, List[str]]]) -> Dict[str, str]:
        """
        Get the export paths for completed tasks.
        
        Args:
            task_ids: Dict mapping resolution name to task ID(s)
            
        Returns:
            Dict mapping resolution name to export path
        """
        export_paths = {}
        
        for res_name, task_id in task_ids.items():
            if isinstance(task_id, list):
                # Hemisphere split - both parts share same base path
                if task_id[0] in self.active_tasks:
                    base_path = self.active_tasks[task_id[0]].get('export_path', '')
                    # Remove hemisphere suffix to get base path
                    if base_path.endswith('_west'):
                        base_path = base_path[:-5]
                    export_paths[res_name] = base_path
            else:
                if task_id in self.active_tasks:
                    export_paths[res_name] = self.active_tasks[task_id].get('export_path', '')
        
        return export_paths