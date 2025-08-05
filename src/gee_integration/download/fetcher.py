"""GCS Fetcher Module

Downloads files from Google Cloud Storage.
"""

import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import os
import glob

from src.infrastructure.logging import get_logger
from src.infrastructure.logging.decorators import log_operation, retry_with_logging

logger = get_logger(__name__)


class GCSFetcher:
    """Fetches files from Google Cloud Storage"""
    
    def __init__(self, temp_dir: Path):
        """
        Initialize fetcher.
        
        Args:
            temp_dir: Temporary directory for downloads
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
    
    @log_operation("download_from_gcs")
    @retry_with_logging(max_attempts=3, delay=5.0)
    def download_export(self, gcs_path: str, local_dir: Optional[Path] = None, export_format: str = 'CSV') -> List[Path]:
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
        
        try:
            self.logger.info(f"Downloading from {gcs_path}...")
            
            if export_format == 'GeoTIFF':
                # Check if this is a hemisphere export (look for _west and _east files)
                base_path = gcs_path.rstrip('/')
                
                # Try to download hemisphere files first
                hemisphere_files = []
                for suffix in ['_west.tif', '_east.tif']:
                    try:
                        cmd = [
                            'gsutil',
                            '-m',
                            'cp',
                            f"{base_path}{suffix}",
                            str(local_dir)
                        ]
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            check=False
                        )
                        if result.returncode == 0:
                            hemisphere_file = local_dir / f"{Path(base_path).name}{suffix}"
                            if hemisphere_file.exists():
                                hemisphere_files.append(hemisphere_file)
                                self.logger.info(f"Downloaded hemisphere file: {hemisphere_file.name}")
                    except subprocess.CalledProcessError as e:
                        self.logger.warning(f"Failed to download hemisphere file {suffix}: {e}")
                    except Exception as e:
                        self.logger.warning(f"Unexpected error downloading hemisphere file {suffix}: {e}")
                
                if len(hemisphere_files) == 2:
                    # We have both hemispheres
                    downloaded_files = hemisphere_files
                else:
                    # Try downloading single GeoTIFF file
                    cmd = [
                        'gsutil',
                        '-m',
                        'cp',
                        f"{gcs_path}.tif",
                        str(local_dir)
                    ]
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    
                    # Find downloaded file
                    tif_file = local_dir / f"{Path(gcs_path).name}.tif"
                    if tif_file.exists():
                        downloaded_files = [tif_file]
                    else:
                        downloaded_files = []
            else:
                # First try to download numbered CSV files (for large exports)
                cmd = [
                    'gsutil',
                    '-m',  # Parallel mode
                    'cp',
                    f"{gcs_path}-*.csv",  # GEE adds numbers to CSV exports
                    str(local_dir)
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False  # Don't raise on error
                )
                
                if result.returncode != 0:
                    # Try single CSV file (for small exports)
                    self.logger.debug("No numbered CSV files found, trying single file...")
                    cmd = [
                        'gsutil',
                        '-m',
                        'cp',
                        f"{gcs_path}.csv",
                        str(local_dir)
                    ]
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        check=True
                    )
                
                # Find downloaded CSV files
                csv_pattern = str(local_dir / f"{Path(gcs_path).name}-*.csv")
                downloaded_files = glob.glob(csv_pattern)
                
                if not downloaded_files:
                    # Check if it's a single CSV file (for small exports)
                    single_file = local_dir / f"{Path(gcs_path).name}.csv"
                    if single_file.exists():
                        downloaded_files = [single_file]
            
            downloaded_paths = [Path(f) for f in downloaded_files]
            
            # Log file sizes
            total_size = sum(f.stat().st_size for f in downloaded_paths)
            self.logger.info(f"✅ Downloaded {len(downloaded_paths)} files ({total_size/(1024*1024):.1f} MB)")
            
            for path in downloaded_paths:
                self.logger.debug(f"   {path.name}: {path.stat().st_size/(1024*1024):.1f} MB")
            
            return downloaded_paths
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Download failed: {e.stderr}")
            raise
        except FileNotFoundError:
            self.logger.error("gsutil not found. Please install Google Cloud SDK")
            raise
    
    def download_multiple_exports(self, export_paths: Dict[str, str], export_format: str = 'CSV') -> Dict[str, List[Path]]:
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