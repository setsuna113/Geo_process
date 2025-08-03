"""GEE Workflow Orchestrator

Main controller for the complete GEE integration workflow.
"""

from pathlib import Path
from typing import Dict, Optional, List, Tuple
import time

from src.infrastructure.logging import get_logger, LoggingContext
from src.infrastructure.logging.decorators import log_operation

from ..config import GEEConfig
from ..auth import GEEAuthenticator
from ..upload import BatchUploadManager
from ..processing import (
    GEEMerger, GEEResampler, GEEAlignmentValidator,
    ExportManager, GEETaskMonitor
)
from ..download import GCSFetcher, ParquetConverter
from .state_manager import WorkflowState, WorkflowStep

logger = get_logger(__name__)


class GEEWorkflowOrchestrator:
    """Orchestrates the complete GEE workflow"""
    
    def __init__(self, config_path: str, experiment_name: Optional[str] = None):
        """
        Initialize orchestrator.
        
        Args:
            config_path: Path to GEE config file
            experiment_name: Optional experiment name for logging
        """
        self.config = GEEConfig.from_yaml(config_path)
        self.state = WorkflowState.load_or_create()
        self.logger = logger
        self.experiment_name = experiment_name or f"gee_merge_{int(time.time())}"
        self.logging_context = LoggingContext(experiment_id=self.experiment_name)
        
        # Initialize components (lazy loading)
        self._authenticator = None
        self._upload_manager = None
        self._export_manager = None
        self._task_monitor = None
    
    @log_operation("gee_workflow_complete")
    def run(self, resume: bool = True, force_step: Optional[WorkflowStep] = None) -> Dict[str, Path]:
        """
        Execute the complete workflow.
        
        Args:
            resume: Resume from last completed step
            force_step: Force start from specific step
            
        Returns:
            Dict mapping resolution name to final parquet path
        """
        self.logger.info("=" * 60)
        self.logger.info(f"Starting GEE Workflow: {self.experiment_name}")
        self.logger.info("=" * 60)
        
        # Check if resuming
        if resume and self.state.current_step != WorkflowStep.NOT_STARTED:
            self.logger.info(f"Resuming from step: {self.state.current_step.value}")
        elif force_step:
            self.state.update_step(force_step)
            self.logger.info(f"Forcing start from step: {force_step.value}")
        
        # Execute workflow steps
        try:
            # Step 1: Validate prerequisites
            if not self.state.is_step_complete(WorkflowStep.PREREQUISITES_VALIDATED):
                self._validate_prerequisites()
            
            # Step 2: Authenticate
            if not self.state.is_step_complete(WorkflowStep.AUTHENTICATED):
                self._authenticate()
            
            # Step 3: Upload assets
            if not self.state.is_step_complete(WorkflowStep.ASSETS_UPLOADED):
                self._upload_assets()
            
            # Step 4: Process in GEE
            if not self.state.is_step_complete(WorkflowStep.PROCESSING_COMPLETE):
                self._process_in_gee()
            
            # Step 5: Export from GEE
            if not self.state.is_step_complete(WorkflowStep.EXPORT_COMPLETE):
                self._export_from_gee()
            
            # Step 6: Download and convert
            if not self.state.is_step_complete(WorkflowStep.DOWNLOAD_COMPLETE):
                final_outputs = self._download_and_convert()
            else:
                # Get outputs from state
                final_outputs = self.state.get_metadata('final_outputs', {})
            
            # Mark complete
            self.state.update_step(WorkflowStep.WORKFLOW_COMPLETE)
            
            self.logger.info("=" * 60)
            self.logger.info("✅ GEE Workflow completed successfully!")
            self.logger.info("=" * 60)
            
            # Summary
            for res_name, parquet_path in final_outputs.items():
                self.logger.info(f"   {res_name}: {parquet_path}")
                
            return final_outputs
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {e}")
            self.logger.error(f"Current step: {self.state.current_step.value}")
            self.logger.info("Run with resume=True to continue from last successful step")
            raise
    
    @log_operation("validate_prerequisites")
    def _validate_prerequisites(self) -> None:
        """Step 1: Validate prerequisites"""
        with self.logging_context.stage("validate_prerequisites"):
            self.logger.info("\n📋 Step 1/6: Validating prerequisites...")
            
            issues = self.config.validate()
            if issues:
                for issue in issues:
                    self.logger.error(f"   ❌ {issue}")
                raise ValueError(f"Configuration validation failed: {len(issues)} issues")
            
            # Check files exist
            self.logger.info(f"   Found {len(self.config.biodiversity_datasets)} biodiversity datasets")
            for dataset in self.config.biodiversity_datasets:
                size_mb = Path(dataset.path).stat().st_size / (1024 * 1024)
                self.logger.info(f"   ✅ {dataset.name}: {size_mb:.1f} MB")
            
            # Check gsutil
            import subprocess
            try:
                subprocess.run(['gsutil', 'version'], capture_output=True, check=True)
                self.logger.info("   ✅ gsutil available")
            except:
                raise RuntimeError("gsutil not found. Please install Google Cloud SDK")
            
            self.state.update_step(WorkflowStep.PREREQUISITES_VALIDATED)
    
    @log_operation("authenticate_gee")
    def _authenticate(self) -> None:
        """Step 2: Authenticate with GEE"""
        with self.logging_context.stage("authenticate"):
            self.logger.info("\n🔐 Step 2/6: Authenticating with GEE...")
            
            self._authenticator = GEEAuthenticator(self.config.__dict__)
            
            success = self._authenticator.authenticate(
                method=self.config.auth_method,
                service_account_key=self.config.service_account_key,
                project_id=self.config.project_id
            )
            
            if not success:
                raise RuntimeError("GEE authentication failed")
            
            # Initialize ee module
            import ee
            ee = self._authenticator.get_ee_module()
            
            self.state.update_step(WorkflowStep.AUTHENTICATED)
    
    @log_operation("upload_assets")
    def _upload_assets(self) -> None:
        """Step 3: Upload biodiversity data to GEE"""
        with self.logging_context.stage("upload_assets"):
            self.logger.info("\n📤 Step 3/6: Uploading biodiversity data...")
            
            self._upload_manager = BatchUploadManager({
                'gcs_bucket': self.config.gcs_bucket,
                'asset_prefix': self.config.asset_prefix,
                'parallel_uploads': self.config.parallel_uploads
            })
            
            # Upload all datasets
            # Convert DatasetConfig objects to dicts
            dataset_dicts = [
                {
                    'path': ds.path,
                    'name': ds.name,
                    'band_name': ds.band_name,
                    'data_type': ds.data_type,
                    'resampling': ds.resampling,
                    'no_data': ds.no_data
                }
                for ds in self.config.biodiversity_datasets
            ]
            upload_results = self._upload_manager.upload_datasets(
                datasets=dataset_dicts,
                parallel_uploads=self.config.parallel_uploads
            )
            
            # Save upload results
            self.logger.info(f"Upload results to save: {upload_results}")
            self.state.set_metadata('upload_results', upload_results)
            
            # Wait for uploads to complete
            if any(r.get('task_id') for r in upload_results.values()):
                self.logger.info("   ⏳ Waiting for GEE ingestion...")
                task_statuses = self._upload_manager.wait_for_all_tasks(upload_results)
                
                # Check all succeeded
                failed = [name for name, success in task_statuses.items() if not success]
                if failed:
                    raise RuntimeError(f"Asset upload failed for: {failed}")
            
            self.state.update_step(WorkflowStep.ASSETS_UPLOADED)
    
    @log_operation("process_in_gee")
    def _process_in_gee(self) -> None:
        """Step 4: Process data in GEE (resample, align, merge)"""
        with self.logging_context.stage("process_in_gee"):
            self.logger.info("\n⚙️ Step 4/6: Processing in GEE...")
            
            # Get upload results
            upload_results = self.state.get_metadata('upload_results', {})
            self.logger.info(f"Upload results: {upload_results}")
            
            # Load biodiversity images
            biodiv_images = []
            biodiv_configs = []
            
            import ee
            
            # Initialize EE if not already done
            if not self._authenticator:
                self._authenticator = GEEAuthenticator(self.config.__dict__)
                self._authenticator.authenticate(
                    method=self.config.auth_method,
                    service_account_key=self.config.service_account_key,
                    project_id=self.config.project_id
                )
            
            for dataset in self.config.biodiversity_datasets:
                if dataset.name in upload_results and upload_results[dataset.name].get('success'):
                    asset_id = upload_results[dataset.name]['asset_id']
                    image = ee.Image(asset_id)
                    # Rename band to expected name
                    image = image.rename([dataset.band_name])
                    biodiv_images.append(image)
                    biodiv_configs.append(dataset.__dict__)
            
            if not biodiv_images:
                raise RuntimeError("No biodiversity images loaded")
            
            # Validate alignment
            validator = GEEAlignmentValidator(self.config.tolerance)
            validation = validator.validate_images(biodiv_images, biodiv_configs)
            
            if not validation['valid']:
                self.logger.warning(f"Alignment issues: {validation['issues']}")
            
            # Load WorldClim
            merger = GEEMerger()
            worldclim = merger.load_worldclim_v2(self.config.climate_variables)
            
            # Process for each resolution
            processed_images = {}
            
            for res_name, res_config in self.config.resolutions.items():
                self.logger.info(f"\n   Processing {res_name} ({res_config.description})...")
                
                # Resample biodiversity images
                resampler = GEEResampler(
                    target_resolution=res_config.resolution,
                    target_crs='EPSG:4326'
                )
                
                resampled_biodiv = resampler.resample_collection(
                    biodiv_images, biodiv_configs
                )
                
                # Resample WorldClim
                worldclim_config = {
                    'name': 'worldclim',
                    'band_name': 'climate',
                    'data_type': 'continuous',
                    'resampling': 'bilinear'
                }
                resampled_climate = resampler.resample_image(worldclim, worldclim_config)
                
                # Merge
                merged = merger.merge_datasets(
                    resampled_biodiv,
                    resampled_climate,
                    self.config.merge_strategy
                )
                
                processed_images[res_name] = merged
            
            # Save processed images info
            self.state.set_metadata('processed_images', {
                res: {'bands': img.bandNames().getInfo()}
                for res, img in processed_images.items()
            })
            
            self.state.update_step(WorkflowStep.PROCESSING_COMPLETE)
    
    @log_operation("export_from_gee")
    def _export_from_gee(self) -> None:
        """Step 5: Export processed data from GEE"""
        with self.logging_context.stage("export_from_gee"):
            self.logger.info("\n📥 Step 5/6: Exporting from GEE...")
            
            # Re-authenticate if needed (for resume cases)
            import ee
            if not self._authenticator:
                self._authenticator = GEEAuthenticator(self.config.__dict__)
                self._authenticator.authenticate(
                    method=self.config.auth_method,
                    service_account_key=self.config.service_account_key,
                    project_id=self.config.project_id
                )
            
            # Initialize components
            self._export_manager = ExportManager(self.config.__dict__)
            self._task_monitor = GEETaskMonitor()
            
            # Get processed images
            processed_info = self.state.get_metadata('processed_images', {})
            
            # Recreate images (since we can't serialize ee.Image)
            merger = GEEMerger()
            
            # Get bounds for sampling
            bounds = (-180, -90, 180, 90)  # Default global
            if self.config.biodiversity_datasets:
                # Could get from parquet metadata if available
                pass
            
            # Export both resolutions
            task_ids = {}
            
            for res_name, res_config in self.config.resolutions.items():
                self.logger.info(f"\n   Exporting {res_name}...")
                
                # Create sampling grid
                sampling_grid = merger.create_sampling_grid(
                    bounds=bounds,
                    resolution=res_config.resolution
                )
                
                # Recreate merged image for export
                self.logger.info("   Recreating merged image for export...")
                
                # Re-authenticate if needed
                import ee
                if not self._authenticator:
                    self._authenticator = GEEAuthenticator(self.config.__dict__)
                    self._authenticator.authenticate(
                        method=self.config.auth_method,
                        service_account_key=self.config.service_account_key,
                        project_id=self.config.project_id
                    )
                
                # Reload biodiversity images
                upload_results = self.state.get_metadata('upload_results', {})
                biodiv_images = []
                
                for dataset in self.config.biodiversity_datasets:
                    if dataset.name in upload_results and upload_results[dataset.name].get('success'):
                        asset_id = upload_results[dataset.name]['asset_id']
                        image = ee.Image(asset_id)
                        # Rename band to expected name
                        image = image.rename([dataset.band_name])
                        biodiv_images.append(image)
                
                # Reload WorldClim
                worldclim = merger.load_worldclim_v2(self.config.climate_variables)
                
                # Resample to target resolution
                resampler = GEEResampler(
                    target_resolution=res_config.resolution,
                    target_crs='EPSG:4326'
                )
                
                # Resample and merge
                if len(biodiv_images) == 1:
                    biodiv_combined = biodiv_images[0]
                else:
                    biodiv_combined = biodiv_images[0]
                    for img in biodiv_images[1:]:
                        biodiv_combined = biodiv_combined.addBands(img)
                
                # Resample biodiversity
                biodiv_resampled = resampler.resample_image(biodiv_combined, {
                    'name': 'biodiversity',
                    'data_type': 'continuous',
                    'resampling': 'bilinear'
                })
                
                # Resample climate
                climate_resampled = resampler.resample_image(worldclim, {
                    'name': 'climate', 
                    'data_type': 'continuous',
                    'resampling': 'bilinear'
                })
                
                # Merge
                merged = biodiv_resampled.addBands(climate_resampled)
                
                # Export the merged image
                export_path = f"{self.config.export_folder}/{res_name}/merged_data"
                
                # Create export task
                task = ee.batch.Export.table.toCloudStorage(
                    collection=merged.sampleRegions(
                        collection=sampling_grid,
                        scale=res_config.resolution * 111320,  # Convert degrees to meters
                        tileScale=self.config.tile_scale
                    ),
                    description=f'export_{res_name}_{self.experiment_name}',
                    bucket=self.config.gcs_bucket,
                    fileNamePrefix=export_path,
                    fileFormat=self.config.export_format
                )
                
                # Start the task
                task.start()
                task_id = task.id
                self.logger.info(f"   Started export task: {task_id}")
                task_ids[res_name] = task_id
            
            # Save task IDs
            self.state.set_metadata('export_task_ids', task_ids)
            
            # Monitor export tasks
            self.logger.info("\n   ⏳ Monitoring export tasks...")
            all_completed = False
            start_time = time.time()
            timeout = 3600  # 1 hour timeout
            
            while not all_completed and (time.time() - start_time) < timeout:
                statuses = {}
                all_completed = True
                
                for res_name, task_id in task_ids.items():
                    status = self._task_monitor.get_task_status(task_id)
                    statuses[res_name] = status
                    
                    if status['state'] in ['READY', 'RUNNING']:
                        all_completed = False
                    elif status['state'] == 'FAILED':
                        raise RuntimeError(f"Export failed for {res_name}: {status.get('error_message', 'Unknown error')}")
                    elif status['state'] == 'COMPLETED':
                        self.logger.info(f"   ✅ {res_name} export completed")
                
                if not all_completed:
                    time.sleep(30)  # Check every 30 seconds
            
            if not all_completed:
                raise RuntimeError("Export tasks timed out")
            
            # Get export paths
            export_paths = {
                res: f"gs://{self.config.gcs_bucket}/{self.config.export_folder}/{res}/merged_data"
                for res in task_ids.keys()
            }
            self.state.set_metadata('export_paths', export_paths)
            
            self.state.update_step(WorkflowStep.EXPORT_COMPLETE)
    
    @log_operation("download_and_convert")
    def _download_and_convert(self) -> Dict[str, Path]:
        """Step 6: Download and convert to parquet"""
        with self.logging_context.stage("download_and_convert"):
            self.logger.info("\n💾 Step 6/6: Downloading and converting to parquet...")
            
            # Get export paths
            export_paths = self.state.get_metadata('export_paths', {})
            
            if not export_paths:
                # For testing, create dummy paths
                self.logger.warning("No export paths found, using test mode")
                final_outputs = {
                    res: self.config.parquet_dir / f"biodiversity_climate_{res}.parquet"
                    for res in self.config.resolutions.keys()
                }
            else:
                # Download from GCS
                fetcher = GCSFetcher(self.config.temp_dir)
                downloads = fetcher.download_multiple_exports(export_paths)
                
                # Convert to parquet
                converter = ParquetConverter(self.config.parquet_dir)
                final_outputs = converter.convert_all_resolutions(downloads)
                
                # Cleanup temp files if requested
                if self.config.cleanup_temp:
                    self.logger.info("   🧹 Cleaning up temporary files...")
                    for csv_files in downloads.values():
                        fetcher.cleanup_temp_files(csv_files)
            
            # Save final outputs
            self.state.set_metadata('final_outputs', {
                k: str(v) for k, v in final_outputs.items()
            })
            
            self.state.update_step(WorkflowStep.DOWNLOAD_COMPLETE)
            
            return final_outputs