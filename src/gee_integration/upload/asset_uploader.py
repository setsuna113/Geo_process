"""Upload assets to Google Earth Engine via GCS"""

import subprocess
import time
import json
from pathlib import Path
from typing import Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.infrastructure.logging import get_logger
from src.infrastructure.logging.decorators import log_operation

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
        
        full_gcs_path = f"gs://{self.gcs_bucket}/{gcs_path}"
        
        # Use gsutil with optimizations
        cmd = [
            'gsutil',
            '-m',  # Parallel mode
            '-o', 'GSUtil:parallel_composite_upload_threshold=150M',  # Parallel for large files
            'cp',
            str(local_path),
            full_gcs_path
        ]
        
        try:
            self.logger.info(f"Uploading {local_path.name} to GCS...")
            
            # Run upload
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            self.logger.info(f"✅ Uploaded to {full_gcs_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"GCS upload failed: {e.stderr}")
            raise RuntimeError(f"GCS upload failed: {e.stderr}")
        except FileNotFoundError:
            self.logger.error("gsutil not found. Please install Google Cloud SDK")
            raise RuntimeError("gsutil not found. Please install Google Cloud SDK")
    
    @log_operation("create_gee_asset")
    def create_gee_asset(self, gcs_path: str, asset_name: str, 
                        no_data_value: Optional[float] = None) -> bool:
        """
        Create GEE asset from GCS file.
        
        Args:
            gcs_path: Full GCS path (gs://bucket/path)
            asset_name: Asset name (without prefix)
            no_data_value: Optional nodata value
            
        Returns:
            bool: Success status
        """
        full_asset_id = f"{self.asset_prefix}/{asset_name}"
        
        # Check if asset already exists
        if self.check_asset_exists(asset_name):
            self.logger.info(f"Asset already exists, skipping: {asset_name}")
            return True
        
        # Build earthengine command
        cmd = [
            'earthengine',
            '--service_account_file', self.config.get('service_account_key', '.gee/service-account-key.json'),
            '--project', self.config.get('project_id', 'triple-carrier-441622-t1'),
            'upload',
            'image',
            '--asset_id=' + full_asset_id,
            gcs_path
        ]
        
        # Add nodata if specified
        if no_data_value is not None:
            cmd.extend(['--nodata_value=' + str(no_data_value)])
        
        try:
            self.logger.info(f"Creating GEE asset: {asset_name}")
            
            # Start upload task
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # Extract task ID from output
            task_id = None
            for line in result.stdout.split('\n'):
                if 'Started upload task' in line:
                    # Extract task ID from output
                    parts = line.split()
                    if len(parts) > 0:
                        task_id = parts[-1]
                        break
            
            if task_id:
                self.logger.info(f"📤 GEE upload task started: {task_id}")
                return task_id
            else:
                self.logger.warning("Could not extract task ID from earthengine output")
                return True
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"GEE asset creation failed: {e.stderr}")
            return False
        except FileNotFoundError:
            self.logger.error("earthengine CLI not found. Please install with: pip install earthengine-api")
            return False
    
    def check_asset_exists(self, asset_name: str) -> bool:
        """Check if GEE asset already exists"""
        full_asset_id = f"{self.asset_prefix}/{asset_name}"
        
        cmd = [
            'earthengine',
            '--service_account_file', self.config.get('service_account_key', '.gee/service-account-key.json'),
            '--project', self.config.get('project_id', 'triple-carrier-441622-t1'),
            'asset', 'info', full_asset_id
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            return result.returncode == 0
            
        except:
            return False
    
    def wait_for_task(self, task_id: str, timeout: int = 3600) -> bool:
        """
        Wait for GEE task to complete.
        
        Args:
            task_id: GEE task ID
            timeout: Maximum wait time in seconds
            
        Returns:
            bool: True if successful
        """
        start_time = time.time()
        check_interval = 30  # seconds
        
        while time.time() - start_time < timeout:
            try:
                # Check task status
                cmd = ['earthengine', 'task', 'info', task_id]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    # Parse status from output
                    output = result.stdout.lower()
                    
                    if 'completed' in output:
                        self.logger.info(f"✅ Task {task_id} completed")
                        return True
                    elif 'failed' in output or 'error' in output:
                        self.logger.error(f"❌ Task {task_id} failed")
                        return False
                    elif 'cancelled' in output:
                        self.logger.warning(f"⚠️ Task {task_id} cancelled")
                        return False
                    else:
                        # Still running
                        elapsed = int(time.time() - start_time)
                        self.logger.debug(f"Task {task_id} still running ({elapsed}s)")
                
            except Exception as e:
                self.logger.error(f"Error checking task status: {e}")
            
            time.sleep(check_interval)
        
        self.logger.error(f"Task {task_id} timed out after {timeout}s")
        return False