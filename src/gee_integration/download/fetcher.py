"""GCS Fetcher Module

Downloads exported data from Google Cloud Storage using the centralized GCS client.
"""

from pathlib import Path
from typing import List, Dict, Optional

from src.infrastructure.logging import get_logger
from src.infrastructure.logging.decorators import log_operation, retry_with_logging
from ..utils import GCSClient

logger = get_logger(__name__)


class GCSFetcher:
    """Fetches exported data from GCS"""
    
    def __init__(self, temp_dir: Path):
        """
        Initialize fetcher.
        
        Args:
            temp_dir: Temporary directory for downloads
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        # Initialize GCS client without default bucket (we get full gs:// paths)
        self.gcs_client = GCSClient()
    
    @log_operation("download_from_gcs")
    @retry_with_logging(max_attempts=3, delay=5.0)
    def download_export(self, gcs_path: str, local_dir: Optional[Path] = None, 
                       export_format: str = 'CSV') -> List[Path]:
        """
        Download exported files from GCS.
        
        Args:
            gcs_path: GCS path (gs://bucket/path)
            local_dir: Local directory to download to
            export_format: Export format ('CSV' or 'GeoTIFF')
            
        Returns:
            List of downloaded file paths
        """
        if not gcs_path.startswith('gs://'):
            raise ValueError(f"Invalid GCS path: {gcs_path}")
        
        # Use temp dir if not specified
        if local_dir is None:
            local_dir = self.temp_dir
        else:
            local_dir = Path(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Downloading from {gcs_path}...")
        
        downloaded_files = []
        
        if export_format == 'GeoTIFF':
            # Check if this is a hemisphere export (look for _west and _east files)
            base_path = gcs_path.rstrip('/')
            
            # Try to download hemisphere files first
            hemisphere_files = []
            for suffix in ['_west.tif', '_east.tif']:
                hemisphere_path = f"{base_path}{suffix}"
                self.logger.debug(f"Trying hemisphere file: {hemisphere_path}")
                
                files = self.gcs_client.download(hemisphere_path, local_dir)
                if files:
                    hemisphere_files.extend(files)
                    self.logger.info(f"Downloaded hemisphere file: {files[0].name}")
            
            if len(hemisphere_files) == 2:
                # We have both hemispheres
                downloaded_files = hemisphere_files
            else:
                # Try single GeoTIFF file
                single_path = f"{gcs_path}.tif"
                self.logger.debug(f"Trying single GeoTIFF: {single_path}")
                
                files = self.gcs_client.download(single_path, local_dir)
                if files:
                    downloaded_files = files
                    self.logger.info(f"Downloaded single GeoTIFF: {files[0].name}")
                else:
                    # Try tiled GeoTIFF files (GEE splits large exports)
                    self.logger.info("Single file not found, checking for tiled export...")
                    files = self.gcs_client.download(gcs_path, local_dir, pattern="*.tif")
                    
                    if files:
                        downloaded_files = files
                        self.logger.info(f"Downloaded {len(files)} tile files")
                        for f in files[:3]:  # Show first 3 as examples
                            self.logger.info(f"  - {f.name}")
                        if len(files) > 3:
                            self.logger.info(f"  ... and {len(files) - 3} more")
                    else:
                        raise RuntimeError(f"No GeoTIFF files found at {gcs_path}")
        else:
            # Handle CSV exports
            # First try numbered CSV files (for large exports)
            self.logger.debug("Checking for numbered CSV files...")
            files = self.gcs_client.download(gcs_path, local_dir, pattern="-*.csv")
            
            if files:
                downloaded_files = files
                self.logger.info(f"Downloaded {len(files)} numbered CSV files")
            else:
                # Try single CSV file (for small exports)
                self.logger.debug("No numbered files found, trying single CSV...")
                single_path = f"{gcs_path}.csv"
                files = self.gcs_client.download(single_path, local_dir)
                
                if files:
                    downloaded_files = files
                    self.logger.info(f"Downloaded single CSV: {files[0].name}")
                else:
                    self.logger.warning(f"No CSV files found at {gcs_path}")
        
        # Log file sizes
        if downloaded_files:
            total_size = sum(f.stat().st_size for f in downloaded_files)
            self.logger.info(f"✅ Downloaded {len(downloaded_files)} files ({total_size/(1024*1024):.1f} MB)")
            
            for path in downloaded_files:
                self.logger.debug(f"   {path.name}: {path.stat().st_size/(1024*1024):.1f} MB")
        
        return downloaded_files
    
    def download_multiple_exports(self, export_paths: Dict[str, str], 
                                 export_format: str = 'CSV') -> Dict[str, List[Path]]:
        """
        Download multiple export results.
        
        Args:
            export_paths: Dict mapping resolution name to GCS path
            export_format: Export format ('CSV' or 'GeoTIFF')
            
        Returns:
            Dict mapping resolution name to list of downloaded files
        """
        downloads = {}
        
        for res_name, gcs_path in export_paths.items():
            self.logger.info(f"Downloading {res_name} data...")
            
            # Create resolution-specific directory
            res_dir = self.temp_dir / res_name
            res_dir.mkdir(exist_ok=True)
            
            try:
                downloaded = self.download_export(gcs_path, res_dir, export_format)
                downloads[res_name] = downloaded
            except Exception as e:
                self.logger.error(f"Failed to download {res_name}: {e}")
                downloads[res_name] = []
        
        return downloads
    
    def cleanup_temp_files(self, files: List[Path]) -> None:
        """Clean up temporary files"""
        for file_path in files:
            try:
                if file_path.exists():
                    file_path.unlink()
                    self.logger.debug(f"Deleted temp file: {file_path}")
            except Exception as e:
                self.logger.warning(f"Failed to delete {file_path}: {e}")
    
    def get_total_size(self, files: List[Path]) -> float:
        """Get total size of files in MB"""
        total_bytes = sum(f.stat().st_size for f in files if f.exists())
        return total_bytes / (1024 * 1024)