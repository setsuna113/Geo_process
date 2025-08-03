"""Batch upload manager for multiple files"""

from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from src.infrastructure.logging import get_logger
from src.infrastructure.logging.decorators import log_operation
from .asset_uploader import AssetUploader
from .validators import TIFValidator

logger = get_logger(__name__)


class BatchUploadManager:
    """Manages batch uploads of multiple TIF files to GEE"""
    
    def __init__(self, config: Dict):
        """
        Initialize batch manager.
        
        Args:
            config: Upload configuration
        """
        self.config = config
        self.logger = logger
        self.uploader = AssetUploader(
            gcs_bucket=config['gcs_bucket'],
            asset_prefix=config['asset_prefix'],
            config=config
        )
        self.validator = TIFValidator()
        
    @log_operation("batch_upload_to_gee")
    def upload_datasets(self, datasets: List[Dict], 
                       parallel_uploads: int = 4,
                       skip_existing: bool = True) -> Dict[str, Dict]:
        """
        Upload multiple datasets to GEE.
        
        Args:
            datasets: List of dataset configs
            parallel_uploads: Number of parallel uploads
            skip_existing: Skip if asset already exists
            
        Returns:
            Dict mapping dataset name to upload result
        """
        results = {}
        
        # First, validate all files
        self.logger.info(f"Validating {len(datasets)} datasets...")
        validation_results = {}
        
        for dataset in datasets:
            validation = self.validator.validate_file(dataset['path'])
            validation_results[dataset['name']] = validation
            
            if not validation['valid']:
                self.logger.error(f"❌ {dataset['name']}: {validation['issues']}")
                results[dataset['name']] = {
                    'success': False,
                    'error': f"Validation failed: {validation['issues']}"
                }
        
        # Filter valid datasets
        valid_datasets = [
            d for d in datasets 
            if validation_results[d['name']]['valid']
        ]
        
        self.logger.info(f"Valid datasets after validation: {len(valid_datasets)}/{len(datasets)}")
        
        if not valid_datasets:
            self.logger.error("No valid datasets to upload")
            return results
        
        # Check existing assets
        if skip_existing:
            remaining_datasets = []
            for dataset in valid_datasets:
                if self.uploader.check_asset_exists(dataset['name']):
                    self.logger.info(f"⏭️ Skipping {dataset['name']} (already exists)")
                    results[dataset['name']] = {
                        'success': True,
                        'skipped': True,
                        'asset_id': f"{self.config['asset_prefix']}/{dataset['name']}"
                    }
                else:
                    remaining_datasets.append(dataset)
            valid_datasets = remaining_datasets
        
        if not valid_datasets:
            self.logger.info("All assets already exist")
            return results
        
        # Upload to GCS first
        self.logger.info(f"Uploading {len(valid_datasets)} files to GCS...")
        gcs_uploads = {}
        
        with ThreadPoolExecutor(max_workers=parallel_uploads) as executor:
            # Submit GCS upload tasks
            future_to_dataset = {
                executor.submit(
                    self._upload_to_gcs,
                    dataset
                ): dataset
                for dataset in valid_datasets
            }
            
            # Process completed uploads
            for future in as_completed(future_to_dataset):
                dataset = future_to_dataset[future]
                try:
                    gcs_path = future.result()
                    if gcs_path:
                        gcs_uploads[dataset['name']] = {
                            'dataset': dataset,
                            'gcs_path': gcs_path
                        }
                    else:
                        results[dataset['name']] = {
                            'success': False,
                            'error': 'GCS upload failed'
                        }
                except Exception as e:
                    self.logger.error(f"Upload failed for {dataset['name']}: {e}")
                    results[dataset['name']] = {
                        'success': False,
                        'error': str(e)
                    }
        
        # Create GEE assets
        if gcs_uploads:
            self.logger.info(f"Creating {len(gcs_uploads)} GEE assets...")
            task_ids = {}
            
            for name, upload_info in gcs_uploads.items():
                dataset = upload_info['dataset']
                task_id = self.uploader.create_gee_asset(
                    gcs_path=upload_info['gcs_path'],
                    asset_name=dataset['name'],
                    no_data_value=dataset.get('no_data')
                )
                
                if task_id:
                    # Handle both new upload (task_id string) and existing asset (True)
                    if isinstance(task_id, str):
                        task_ids[name] = task_id
                    results[name] = {
                        'success': True,
                        'task_id': task_id if isinstance(task_id, str) else None,
                        'asset_id': f"{self.config['asset_prefix']}/{dataset['name']}",
                        'gcs_path': upload_info['gcs_path']
                    }
                else:
                    results[name] = {
                        'success': False,
                        'error': 'Failed to create GEE asset'
                    }
        
        # Summary
        success_count = sum(1 for r in results.values() if r.get('success', False))
        self.logger.info(f"Upload complete: {success_count}/{len(datasets)} successful")
        
        return results
    
    def _upload_to_gcs(self, dataset: Dict) -> Optional[str]:
        """Upload single file to GCS"""
        local_path = Path(dataset['path'])
        gcs_path = f"biodiversity/{dataset['name']}/{local_path.name}"
        
        if self.uploader.upload_to_gcs(local_path, gcs_path):
            return f"gs://{self.config['gcs_bucket']}/{gcs_path}"
        return None
    
    def wait_for_all_tasks(self, results: Dict[str, Dict], timeout: int = 7200) -> Dict[str, bool]:
        """
        Wait for all GEE tasks to complete.
        
        Args:
            results: Upload results with task IDs
            timeout: Total timeout in seconds
            
        Returns:
            Dict mapping dataset name to completion status
        """
        task_statuses = {}
        tasks_to_wait = {
            name: info['task_id']
            for name, info in results.items()
            if info.get('success') and info.get('task_id') and not info.get('skipped')
        }
        
        if not tasks_to_wait:
            self.logger.info("No tasks to wait for")
            return {}
        
        self.logger.info(f"Waiting for {len(tasks_to_wait)} GEE tasks...")
        
        # Check tasks periodically
        for name, task_id in tasks_to_wait.items():
            self.logger.info(f"Checking task for {name}...")
            success = self.uploader.wait_for_task(task_id, timeout=timeout // len(tasks_to_wait))
            task_statuses[name] = success
        
        # Summary
        success_count = sum(1 for s in task_statuses.values() if s)
        self.logger.info(f"Tasks complete: {success_count}/{len(task_statuses)} successful")
        
        return task_statuses