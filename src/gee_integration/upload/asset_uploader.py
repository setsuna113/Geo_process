"""Upload assets to Google Earth Engine via GCS"""

import time
from pathlib import Path
from typing import Optional, Dict, List, Union

import ee
from src.infrastructure.logging import get_logger
from src.infrastructure.logging.decorators import log_operation
from ..utils import GCSClient

logger = get_logger(__name__)


class AssetUploader:
    """Handles uploading files to GEE via GCS"""
    
    def __init__(self, gcs_bucket: str, asset_prefix: str, config: Optional[Dict] = None):
        """
        Initialize uploader.
        
        Args:
            gcs_bucket: GCS bucket name (without gs://)
            asset_prefix: GEE asset prefix (e.g., users/username/biodiversity)
            config: Optional config dict with service account info
        """
        self.gcs_bucket = gcs_bucket
        self.asset_prefix = asset_prefix
        self.logger = logger
        self.config = config or {}
        self._ee_initialized = False
        self.gcs_client = GCSClient(bucket=gcs_bucket)
        
    def _ensure_ee_initialized(self):
        """Ensure Earth Engine is initialized"""
        if not self._ee_initialized:
            # Use the authenticator module if ee is not already initialized
            try:
                # Check if already initialized
                ee.Number(1).getInfo()
                self._ee_initialized = True
            except:
                # Need to initialize
                from ..auth.authenticator import GEEAuthenticator
                auth = GEEAuthenticator(self.config)
                auth.authenticate()
                self._ee_initialized = True
        
    @log_operation("upload_to_gcs")
    def upload_to_gcs(self, local_path: str, gcs_path: Optional[str] = None) -> bool:
        """
        Upload file to Google Cloud Storage.
        
        Args:
            local_path: Local file path
            gcs_path: GCS path (relative to bucket), defaults to filename
            
        Returns:
            bool: Success status
        """
        local_path = Path(local_path)
        if not local_path.exists():
            self.logger.error(f"File not found: {local_path}")
            return False
        
        # Default GCS path
        if not gcs_path:
            gcs_path = local_path.name
        
        # Use GCS client for upload
        try:
            success = self.gcs_client.upload(local_path, gcs_path)
            if success:
                full_gcs_path = f"gs://{self.gcs_bucket}/{gcs_path}"
                self.logger.info(f"✅ Uploaded to {full_gcs_path}")
            return success
        except Exception as e:
            self.logger.error(f"GCS upload failed: {e}")
            raise RuntimeError(f"GCS upload failed: {e}")
    
    @log_operation("create_gee_asset")
    def create_gee_asset(self, gcs_path: str, asset_name: str, 
                        no_data_value: Optional[float] = None,
                        bands: Optional[List[Dict]] = None) -> Union[bool, str]:
        """
        Create GEE asset from GCS file using Python API.
        
        Args:
            gcs_path: Full GCS path (gs://bucket/path)
            asset_name: Asset name (without prefix)
            no_data_value: Optional nodata value
            bands: Optional band names list
            
        Returns:
            Task ID string if started, True if exists, False if failed
        """
        self._ensure_ee_initialized()
        
        full_asset_id = f"{self.asset_prefix}/{asset_name}"
        
        # Ensure parent folder exists
        self._ensure_asset_folder_exists()
        
        # Check if asset already exists
        if self.check_asset_exists(asset_name):
            self.logger.info(f"Asset already exists, skipping: {asset_name}")
            return True
        
        try:
            self.logger.info(f"Creating GEE asset: {asset_name}")
            
            # Build the image manifest for ingestion
            image_manifest = {
                'name': full_asset_id,
                'tilesets': [{
                    'sources': [{
                        'uris': [gcs_path]
                    }]
                }]
            }
            
            # Add band names if provided
            if bands:
                image_manifest['bands'] = bands
            
            # Add missing data value if specified
            # Skip nodata for biodiversity datasets to preserve ocean/land distinction
            if no_data_value is not None:
                # Only apply nodata to climate/elevation datasets
                if any(keyword in asset_name.lower() for keyword in ['climate', 'elevation', 'worldclim']):
                    image_manifest['missingData'] = {
                        'values': [no_data_value]
                    }
                    self.logger.info(f"Using nodata value {no_data_value} for {asset_name}")
                else:
                    self.logger.info(f"Skipping nodata flag for {asset_name} to preserve all pixels as data")
            
            # Start the ingestion
            result = ee.data.startExternalImageIngestion(
                image_manifest=image_manifest,
                allow_overwrite=False
            )
            
            # Extract task ID from result
            if isinstance(result, dict) and 'name' in result:
                # The 'name' field contains the operation ID
                task_id = result['name'].split('/')[-1] if '/' in result['name'] else result['name']
                self.logger.info(f"📤 GEE upload task started: {task_id}")
                return task_id
            elif isinstance(result, str):
                # Sometimes it returns just the task ID
                self.logger.info(f"📤 GEE upload task started: {result}")
                return result
            else:
                self.logger.warning(f"Unexpected response from ingestion: {result}")
                return True
                
        except Exception as e:
            self.logger.error(f"GEE asset creation failed: {e}")
            return False
    
    def _ensure_asset_folder_exists(self) -> None:
        """Ensure the parent asset folder exists, create if needed"""
        self._ensure_ee_initialized()
        
        try:
            # Check if folder exists
            try:
                asset_info = ee.data.getAsset(self.asset_prefix)
                if asset_info:
                    return  # Folder exists
            except ee.EEException:
                pass  # Folder doesn't exist, need to create
            
            # Create the folder
            self.logger.info(f"Creating GEE asset folder: {self.asset_prefix}")
            
            # Parse the path to create parent folders if needed
            parts = self.asset_prefix.split('/')
            current_path = ''
            
            for i, part in enumerate(parts):
                if i == 0:
                    current_path = part
                else:
                    current_path = f"{current_path}/{part}"
                
                # Skip 'projects' and 'assets' parts
                if part in ['projects', 'assets'] or not current_path:
                    continue
                
                try:
                    # Check if this level exists
                    ee.data.getAsset(current_path)
                except:
                    # Create folder at this level
                    try:
                        ee.data.createFolder(current_path)
                        self.logger.debug(f"Created folder: {current_path}")
                    except ee.EEException as e:
                        if 'already exists' not in str(e):
                            self.logger.warning(f"Could not create folder {current_path}: {e}")
            
            self.logger.info(f"✅ Asset folder ready: {self.asset_prefix}")
            
        except Exception as e:
            self.logger.warning(f"Error ensuring asset folder: {e}")
    
    def check_asset_exists(self, asset_name: str) -> bool:
        """Check if GEE asset already exists"""
        self._ensure_ee_initialized()
        
        full_asset_id = f"{self.asset_prefix}/{asset_name}"
        
        try:
            asset_info = ee.data.getAsset(full_asset_id)
            return asset_info is not None
        except ee.EEException:
            return False
        except Exception as e:
            self.logger.debug(f"Error checking asset {full_asset_id}: {e}")
            return False
    
    def wait_for_task(self, task_id: str, timeout: int = 3600) -> bool:
        """
        Wait for GEE task to complete.
        
        Args:
            task_id: GEE task ID or operation name
            timeout: Maximum wait time in seconds
            
        Returns:
            bool: True if successful
        """
        self._ensure_ee_initialized()
        
        start_time = time.time()
        check_interval = 30  # seconds
        
        while time.time() - start_time < timeout:
            try:
                # Get task status using Python API
                tasks = ee.data.getTaskList()
                
                # Find our task
                task_info = None
                for task in tasks:
                    if task.get('id') == task_id or task.get('name', '').endswith(task_id):
                        task_info = task
                        break
                
                if not task_info:
                    # Task might have completed and been removed from list
                    self.logger.debug(f"Task {task_id} not found in task list, may be complete")
                    return True
                
                state = task_info.get('state', 'UNKNOWN')
                
                if state == 'COMPLETED' or state == 'SUCCEEDED':
                    self.logger.info(f"✅ Task {task_id} completed")
                    return True
                elif state == 'FAILED' or state == 'CANCELLED':
                    error_msg = task_info.get('error_message', 'No error message')
                    self.logger.error(f"❌ Task {task_id} failed: {error_msg}")
                    return False
                elif state in ['RUNNING', 'PENDING', 'READY']:
                    elapsed = int(time.time() - start_time)
                    self.logger.debug(f"Task {task_id} {state} ({elapsed}s)")
                else:
                    self.logger.warning(f"Unknown task state: {state}")
                
            except Exception as e:
                self.logger.error(f"Error checking task status: {e}")
            
            time.sleep(check_interval)
        
        self.logger.error(f"Task {task_id} timed out after {timeout}s")
        return False
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running GEE task.
        
        Args:
            task_id: GEE task ID
            
        Returns:
            bool: True if cancelled successfully
        """
        self._ensure_ee_initialized()
        
        try:
            ee.data.cancelTask(task_id)
            self.logger.info(f"Cancelled task {task_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel task {task_id}: {e}")
            return False
    
    def list_tasks(self, limit: int = 10) -> List[Dict]:
        """
        List recent GEE tasks.
        
        Args:
            limit: Maximum number of tasks to return
            
        Returns:
            List of task info dictionaries
        """
        self._ensure_ee_initialized()
        
        try:
            tasks = ee.data.getTaskList()
            return tasks[:limit]
        except Exception as e:
            self.logger.error(f"Failed to list tasks: {e}")
            return []