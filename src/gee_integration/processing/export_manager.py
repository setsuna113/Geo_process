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
        self.logger.info(f"Sampling merged image at {sampling_points.size().getInfo()} points")
        
        sampled = merged_image.sampleRegions(
            collection=sampling_points,
            scale=self.config['processing']['scale'],
            projection='EPSG:4326',
            tileScale=self.config['processing']['tile_scale']
        )
        
        # Prepare export
        if not description:
            description = f'biodiversity_climate_{resolution_name}'
        
        export_path = f"{self.config['export']['folder']}/{resolution_name}/merged_data"
        
        # Create export task
        if self.config['export']['destination'] == 'cloud_storage':
            task = ee.batch.Export.table.toCloudStorage(
                collection=sampled,
                description=description,
                bucket=self.config['export']['bucket'],
                fileNamePrefix=export_path,
                fileFormat='CSV'
            )
        else:
            # Export to Drive
            task = ee.batch.Export.table.toDrive(
                collection=sampled,
                description=description,
                folder=self.config['export']['folder'],
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
        
        for res_name, res_config in resolutions.items():
            self.logger.info(f"Preparing export for {res_name} ({res_config['description']})")
            
            # Create sampling grid for this resolution
            from .merger import GEEMerger
            merger = GEEMerger()
            
            sampling_grid = merger.create_sampling_grid(
                bounds=bounds,
                resolution=res_config['resolution']
            )
            
            # Get sample count
            sample_count = sampling_grid.size().getInfo()
            self.logger.info(f"   Sample points: {sample_count:,}")
            
            # Export
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
            task_ids: List of task IDs to monitor
            check_interval: Check interval in seconds
            timeout: Overall timeout in seconds
            
        Returns:
            Dict mapping task ID to final status
        """
        start_time = time.time()
        task_statuses = {tid: {'status': 'RUNNING'} for tid in task_ids}
        
        self.logger.info(f"Monitoring {len(task_ids)} export tasks...")
        
        while True:
            # Check each task
            all_complete = True
            
            for task_id in task_ids:
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
                        self.logger.debug(f"Task {task_id}: {progress:.1f}% complete")
                    elif status['state'] == 'COMPLETED':
                        duration = task_statuses[task_id]['duration']
                        self.logger.info(f"✅ Task {task_id} completed in {duration/60:.1f} minutes")
                    elif status['state'] == 'FAILED':
                        self.logger.error(f"❌ Task {task_id} failed: {status.get('error_message')}")
                    elif status['state'] == 'CANCELLED':
                        self.logger.warning(f"⚠️ Task {task_id} was cancelled")
            
            if all_complete:
                break
            
            # Check timeout
            if time.time() - start_time > timeout:
                self.logger.error(f"Export monitoring timed out after {timeout}s")
                for task_id in task_ids:
                    if task_statuses[task_id]['status'] == 'RUNNING':
                        task_statuses[task_id]['status'] = 'TIMEOUT'
                break
            
            time.sleep(check_interval)
        
        # Summary
        completed = sum(1 for s in task_statuses.values() if s['status'] == 'COMPLETED')
        failed = sum(1 for s in task_statuses.values() if s['status'] in ['FAILED', 'TIMEOUT'])
        
        self.logger.info(f"Export tasks complete: {completed}/{len(task_ids)} successful, {failed} failed")
        
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
        
        for res_name, task_id in task_ids.items():
            if task_id in self.active_tasks:
                task_info = self.active_tasks[task_id]
                if self.config['export']['destination'] == 'cloud_storage':
                    export_paths[res_name] = f"gs://{self.config['export']['bucket']}/{task_info['export_path']}"
                else:
                    export_paths[res_name] = task_info['export_path']
        
        return export_paths