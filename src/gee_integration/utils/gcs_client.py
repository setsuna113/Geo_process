"""Google Cloud Storage Client

Centralized GCS operations using gsutil subprocess calls.
This provides a clean abstraction over gsutil commands.
"""

import subprocess
from pathlib import Path
from typing import Optional, List, Dict
import time

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class GCSClient:
    """Handles GCS operations via gsutil"""
    
    def __init__(self, bucket: Optional[str] = None):
        """
        Initialize GCS client.
        
        Args:
            bucket: Default bucket name (without gs://)
        """
        self.bucket = bucket
        self.logger = logger
        self._check_gsutil()
    
    def _check_gsutil(self) -> None:
        """Check if gsutil is available"""
        try:
            subprocess.run(['gsutil', 'version'], 
                         capture_output=True, check=True)
        except FileNotFoundError:
            raise RuntimeError("gsutil not found. Please install Google Cloud SDK")
        except subprocess.CalledProcessError:
            raise RuntimeError("gsutil check failed. Ensure gcloud is configured")
    
    def upload(self, 
               local_path: Path, 
               gcs_path: str,
               parallel_threshold: str = "150M") -> bool:
        """
        Upload file to GCS.
        
        Args:
            local_path: Local file path
            gcs_path: GCS path (can be full gs:// URL or relative to bucket)
            parallel_threshold: Threshold for parallel composite uploads
            
        Returns:
            True if successful
        """
        # Ensure full GCS path
        if not gcs_path.startswith('gs://'):
            if not self.bucket:
                raise ValueError("No bucket specified and path is not absolute")
            gcs_path = f"gs://{self.bucket}/{gcs_path}"
        
        cmd = [
            'gsutil',
            '-m',  # Parallel mode
            '-o', f'GSUtil:parallel_composite_upload_threshold={parallel_threshold}',
            'cp',
            str(local_path),
            gcs_path
        ]
        
        try:
            self.logger.info(f"Uploading {local_path.name} to {gcs_path}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.info(f"✅ Upload complete: {gcs_path}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Upload failed: {e.stderr}")
            return False
    
    def download(self,
                gcs_path: str,
                local_dir: Path,
                pattern: Optional[str] = None) -> List[Path]:
        """
        Download file(s) from GCS.
        
        Args:
            gcs_path: GCS path (full gs:// URL)
            local_dir: Local directory to download to
            pattern: Optional pattern for multiple files (e.g., "*.tif")
            
        Returns:
            List of downloaded file paths
        """
        if not gcs_path.startswith('gs://'):
            if not self.bucket:
                raise ValueError("No bucket specified and path is not absolute")
            gcs_path = f"gs://{self.bucket}/{gcs_path}"
        
        # Add pattern if specified
        if pattern:
            gcs_path = f"{gcs_path}/{pattern}" if not gcs_path.endswith('/') else f"{gcs_path}{pattern}"
        
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            'gsutil',
            '-m',  # Parallel mode
            'cp',
            gcs_path,
            str(local_dir) + '/'
        ]
        
        try:
            self.logger.info(f"Downloading from {gcs_path} to {local_dir}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse downloaded files from output
            downloaded = []
            for line in result.stderr.split('\n'):
                if 'Copying gs://' in line:
                    # Extract filename from gsutil output
                    parts = line.split('...')
                    if len(parts) > 1:
                        filename = Path(parts[-1].strip()).name
                        downloaded.append(local_dir / filename)
            
            if not downloaded:
                # If parsing failed, list directory
                downloaded = list(local_dir.glob('*'))
            
            self.logger.info(f"✅ Downloaded {len(downloaded)} files")
            return downloaded
            
        except subprocess.CalledProcessError as e:
            if 'CommandException: No URLs matched' in e.stderr:
                self.logger.warning(f"No files found at {gcs_path}")
                return []
            else:
                self.logger.error(f"Download failed: {e.stderr}")
                raise
    
    def list(self, gcs_prefix: str) -> List[str]:
        """
        List objects in GCS.
        
        Args:
            gcs_prefix: GCS prefix to list (full gs:// URL or relative)
            
        Returns:
            List of GCS paths
        """
        if not gcs_prefix.startswith('gs://'):
            if not self.bucket:
                raise ValueError("No bucket specified and path is not absolute")
            gcs_prefix = f"gs://{self.bucket}/{gcs_prefix}"
        
        cmd = ['gsutil', 'ls', gcs_prefix]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            paths = [line.strip() for line in result.stdout.split('\n') if line.strip()]
            return paths
        except subprocess.CalledProcessError as e:
            if 'CommandException: No URLs matched' in e.stderr:
                return []
            else:
                self.logger.error(f"List failed: {e.stderr}")
                raise
    
    def exists(self, gcs_path: str) -> bool:
        """
        Check if a GCS object exists.
        
        Args:
            gcs_path: GCS path to check
            
        Returns:
            True if exists
        """
        if not gcs_path.startswith('gs://'):
            if not self.bucket:
                raise ValueError("No bucket specified and path is not absolute")
            gcs_path = f"gs://{self.bucket}/{gcs_path}"
        
        cmd = ['gsutil', 'stat', gcs_path]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def delete(self, gcs_path: str, recursive: bool = False) -> bool:
        """
        Delete GCS object(s).
        
        Args:
            gcs_path: GCS path to delete
            recursive: Delete recursively
            
        Returns:
            True if successful
        """
        if not gcs_path.startswith('gs://'):
            if not self.bucket:
                raise ValueError("No bucket specified and path is not absolute")
            gcs_path = f"gs://{self.bucket}/{gcs_path}"
        
        cmd = ['gsutil', '-m', 'rm']
        if recursive:
            cmd.append('-r')
        cmd.append(gcs_path)
        
        try:
            self.logger.info(f"Deleting {gcs_path}")
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.info(f"✅ Deleted: {gcs_path}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Delete failed: {e.stderr}")
            return False
    
    def copy(self, source: str, dest: str) -> bool:
        """
        Copy objects within GCS.
        
        Args:
            source: Source GCS path
            dest: Destination GCS path
            
        Returns:
            True if successful
        """
        # Ensure full paths
        if not source.startswith('gs://'):
            if not self.bucket:
                raise ValueError("No bucket specified")
            source = f"gs://{self.bucket}/{source}"
        
        if not dest.startswith('gs://'):
            if not self.bucket:
                raise ValueError("No bucket specified")
            dest = f"gs://{self.bucket}/{dest}"
        
        cmd = ['gsutil', '-m', 'cp', source, dest]
        
        try:
            self.logger.info(f"Copying {source} to {dest}")
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.info(f"✅ Copied successfully")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Copy failed: {e.stderr}")
            return False