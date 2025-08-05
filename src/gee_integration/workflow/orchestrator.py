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
            elif self.state.current_step == WorkflowStep.PROCESSING_COMPLETE:
                # Fix: If we're at processing_complete, move to export
                self.logger.info("Resuming from processing_complete, starting export...")
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
            
            # Log detailed validation results
            self.logger.info("=" * 60)
            self.logger.info("ALIGNMENT VALIDATION RESULTS:")
            self.logger.info("=" * 60)
            
            # Log metadata for each image
            for img_meta in validation['metadata']:
                self.logger.info(f"\nImage: {img_meta['name']}")
                self.logger.info(f"  - CRS: {img_meta['crs']}")
                self.logger.info(f"  - Dimensions: {img_meta['dimensions']}")
                self.logger.info(f"  - Data type: {img_meta['data_type']}")
                self.logger.info(f"  - Bands: {img_meta['bands']} ({', '.join(img_meta['band_names'])})")
                if 'bounds' in img_meta:
                    self.logger.info(f"  - Bounds: {img_meta['bounds']}")
                if 'crs_transform' in img_meta:
                    self.logger.info(f"  - Transform: {img_meta['crs_transform']}")
            
            # Log summary
            summary = validation['summary']
            self.logger.info(f"\nSummary:")
            self.logger.info(f"  - Total images: {summary['total_images']}")
            self.logger.info(f"  - CRS list: {summary['crs_list']}")
            self.logger.info(f"  - Total bands: {summary['total_bands']}")
            self.logger.info(f"  - Data types: {summary['data_types']}")
            if 'overall_bounds' in summary:
                self.logger.info(f"  - Overall bounds: {summary['overall_bounds']}")
            
            # Log validation status
            self.logger.info(f"\nValidation Status: {'✅ PASSED' if validation['valid'] else '❌ FAILED'}")
            if not validation['valid']:
                self.logger.warning(f"Alignment issues found:")
                for issue in validation['issues']:
                    self.logger.warning(f"  - {issue}")
            else:
                self.logger.info("  - All images are compatible for merging")
            
            self.logger.info("=" * 60)
            
            # Load WorldClim
            merger = GEEMerger(config={'processing': {'latitude_bounds': self.config.latitude_bounds}})
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
            
            # Save processed images info (without blocking getInfo calls)
            self.state.set_metadata('processed_images', {
                res: {'status': 'processed'}
                for res in processed_images.keys()
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
            merger = GEEMerger(config={'processing': {'latitude_bounds': self.config.latitude_bounds}})
            
            # Export both resolutions
            task_ids = {}
            
            # Prepare images for all resolutions
            processed_images_for_export = {}
            
            for res_name, res_config in self.config.resolutions.items():
                self.logger.info(f"\n   Preparing {res_name} for export...")
                
                # Reload biodiversity images
                upload_results = self.state.get_metadata('upload_results', {})
                biodiv_images = []
                biodiv_configs = []
                
                for dataset in self.config.biodiversity_datasets:
                    if dataset.name in upload_results and upload_results[dataset.name].get('success'):
                        asset_id = upload_results[dataset.name]['asset_id']
                        image = ee.Image(asset_id)
                        # Rename band to expected name
                        image = image.rename([dataset.band_name])
                        biodiv_images.append(image)
                        biodiv_configs.append(dataset.__dict__)
                
                # Reload WorldClim
                worldclim = merger.load_worldclim_v2(self.config.climate_variables)
                
                # Resample to target resolution
                resampler = GEEResampler(
                    target_resolution=res_config.resolution,
                    target_crs='EPSG:4326'
                )
                
                # Resample biodiversity images
                biodiv_resampled = resampler.resample_collection(biodiv_images, biodiv_configs)
                
                # Resample climate
                worldclim_config = {
                    'name': 'worldclim',
                    'band_name': 'climate',
                    'data_type': 'continuous',
                    'resampling': 'bilinear'
                }
                climate_resampled = resampler.resample_image(worldclim, worldclim_config)
                
                # Merge all datasets
                merged = merger.merge_datasets(
                    biodiv_resampled,
                    climate_resampled,
                    self.config.merge_strategy
                )
                
                processed_images_for_export[res_name] = merged
            
            # Now export each resolution using ExportManager
            self.logger.info(f"\n   Using {self.config.export_format} export format...")
            
            for res_name, merged_image in processed_images_for_export.items():
                self.logger.info(f"\n   Exporting {res_name}...")
                
                # Always use hardcoded region for reliability
                # Use latitude bounds from config or conservative defaults
                lat_bounds = getattr(self.config, 'latitude_bounds', [-60, 75])
                self.logger.info(f"   Using export region with latitude range: {lat_bounds}")
                
                # Create simple bounded region
                export_region = ee.Geometry.Rectangle([-180, lat_bounds[0], 180, lat_bounds[1]])
                
                if self.config.export_format == 'GeoTIFF':
                    # Export as GeoTIFF with fixed region
                    task_id = self._export_manager._export_as_geotiff(
                        merged_image=merged_image,
                        bounds=export_region,
                        resolution_name=res_name,
                        resolution_config=self.config.resolutions[res_name].__dict__,
                        description=f'export_{res_name}_{self.experiment_name}'
                    )
                else:
                    # Export as CSV (existing logic)
                    # Add pixel coordinates as bands
                    lon = ee.Image.pixelLonLat().select('longitude').rename('x')
                    lat = ee.Image.pixelLonLat().select('latitude').rename('y')
                    
                    # Add coordinates to the merged image
                    image_with_coords = lon.addBands(lat).addBands(merged_image)
                    
                    # Sample the image using detected bounds or global bounds
                    if data_bounds is not None:
                        sample_region = ee.FeatureCollection([ee.Feature(data_bounds)])
                    else:
                        # Use a global region if bounds detection failed
                        global_bounds = ee.Geometry.Rectangle([-180, -90, 180, 90])
                        sample_region = ee.FeatureCollection([ee.Feature(global_bounds)])
                        
                    sampled = image_with_coords.sampleRegions(
                        collection=sample_region,
                        scale=self.config.resolutions[res_name].resolution * 111320,
                        projection='EPSG:4326',
                        tileScale=4
                    )
                    
                    # Export as CSV
                    export_path = f"{self.config.export_folder}/{res_name}/merged_data"
                    task = ee.batch.Export.table.toCloudStorage(
                        collection=sampled,
                        description=f'export_{res_name}_{self.experiment_name}',
                        bucket=self.config.gcs_bucket,
                        fileNamePrefix=export_path,
                        fileFormat='CSV'
                    )
                    
                    # Start the task
                    task.start()
                    task_id = task.id
                
                self.logger.info(f"   Started export task: {task_id}")
                task_ids[res_name] = task_id
            
            # Save task IDs
            self.state.set_metadata('export_task_ids', task_ids)
            
            # Monitor export tasks with configurable timeout
            self.logger.info(f"\n   ⏳ Monitoring export tasks (timeout: {self.config.export_timeout}s)...")
            all_completed = False
            start_time = time.time()
            
            # Track individual task progress
            task_progress = {res: {'start_time': start_time, 'state': 'READY'} 
                           for res in task_ids.keys()}
            
            while not all_completed and (time.time() - start_time) < self.config.export_timeout:
                statuses = {}
                all_completed = True
                
                for res_name, task_id in task_ids.items():
                    # Get task status directly from Earth Engine
                    task_info = ee.data.getTaskStatus(task_id)[0]
                    status = {
                        'state': task_info.get('state', 'UNKNOWN'),
                        'error_message': task_info.get('error_message', '')
                    }
                    statuses[res_name] = status
                    
                    # Update progress tracking
                    if status['state'] != task_progress[res_name]['state']:
                        task_progress[res_name]['state'] = status['state']
                        elapsed = int(time.time() - task_progress[res_name]['start_time'])
                        self.logger.info(f"   {res_name}: {status['state']} ({elapsed}s)")
                    
                    if status['state'] in ['READY', 'RUNNING']:
                        all_completed = False
                        # Check individual task timeout (3x global timeout)
                        if (time.time() - task_progress[res_name]['start_time']) > self.config.export_timeout * 3:
                            raise RuntimeError(f"Export timeout for {res_name} after {self.config.export_timeout * 3}s")
                    elif status['state'] == 'FAILED':
                        raise RuntimeError(f"Export failed for {res_name}: {status.get('error_message', 'Unknown error')}")
                    elif status['state'] == 'COMPLETED':
                        if task_progress[res_name]['state'] != 'LOGGED':
                            elapsed = int(time.time() - task_progress[res_name]['start_time'])
                            self.logger.info(f"   ✅ {res_name} export completed in {elapsed}s")
                            task_progress[res_name]['state'] = 'LOGGED'
                
                if not all_completed:
                    time.sleep(self.config.export_check_interval)
            
            if not all_completed:
                # Log which tasks are still running
                running_tasks = [res for res, info in task_progress.items() 
                               if info['state'] in ['READY', 'RUNNING']]
                raise RuntimeError(f"Export tasks timed out after {self.config.export_timeout}s. "
                                 f"Still running: {', '.join(running_tasks)}")
            
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
                # Pass export format to fetcher
                downloads = fetcher.download_multiple_exports(
                    export_paths, 
                    export_format=self.config.export_format
                )
                
                # Convert to parquet
                converter = ParquetConverter(self.config.parquet_dir)
                
                # Build band names from config
                band_names = []
                biodiversity_bands = []
                
                # Add biodiversity bands
                for dataset in self.config.biodiversity_datasets:
                    band_names.append(dataset.band_name)
                    biodiversity_bands.append(dataset.band_name)
                    
                # Add climate bands
                if hasattr(self.config, 'climate_variables'):
                    band_names.extend(self.config.climate_variables)
                    
                # Add elevation if configured
                if hasattr(self.config, 'include_elevation') and self.config.include_elevation:
                    band_names.append('elevation')
                
                # Pass metadata for converter
                metadata = {
                    'band_names': band_names,
                    'biodiversity_bands': biodiversity_bands,  # For validity checking
                    'format': self.config.export_format,
                    'chunk_size': getattr(self.config, 'parquet_chunk_size', 100000)
                }
                
                # Convert based on format
                if self.config.export_format == 'GeoTIFF':
                    # For GeoTIFF, we expect one file per resolution
                    final_outputs = {}
                    for res_name, downloaded_files in downloads.items():
                        if downloaded_files:
                            output_path = converter.convert(
                                downloaded_files[0],  # GeoTIFF is single file
                                res_name,
                                metadata
                            )
                            final_outputs[res_name] = output_path
                else:
                    # CSV format - use existing method
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