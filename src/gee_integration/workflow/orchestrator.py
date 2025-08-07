"""GEE Workflow Orchestrator

Main controller for the complete GEE integration workflow.
"""

from pathlib import Path
from typing import Dict, Optional, List, Tuple
import time

from src.infrastructure.logging import get_logger, LoggingContext
from src.infrastructure.logging.decorators import log_operation
from config.gee.constants import (
    BASE_RESOLUTION_DEGREES,
    DEGREES_TO_METERS,
    DEFAULT_LAT_MIN,
    DEFAULT_LAT_MAX,
    GLOBAL_LON_MIN,
    GLOBAL_LON_MAX,
    GLOBAL_LAT_MIN,
    GLOBAL_LAT_MAX
)

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
        self.experiment_name = experiment_name or f"gee_merge_{int(time.time())}"
        
        # CRITICAL: Use experiment-specific export path to avoid conflicts
        self.export_base_path = self.experiment_name
        
        # Use experiment-specific state file
        state_file = Path.home() / '.biodiversity' / f'gee_workflow_state_{self.experiment_name}.json'
        self.state = WorkflowState.load_or_create(state_file)
        self.logger = logger
        self.logging_context = LoggingContext(experiment_id=self.experiment_name)
        
        # Clear logging for experiment tracking
        self.logger.info("=" * 60)
        self.logger.info(f"EXPERIMENT: {self.experiment_name}")
        self.logger.info(f"Export Path: gs://{self.config.gcs_bucket}/{self.export_base_path}/")
        self.logger.info(f"State File: {state_file.name}")
        self.logger.info("=" * 60)
        
        # Initialize components (lazy loading)
        self._authenticator = None
        self._upload_manager = None
        self._export_manager = None
        self._task_monitor = None
        
        # Clean up old tasks on initialization to prevent clutter
        self.state.cleanup_old_tasks()
    
    def _get_resolution_outputs(self) -> Dict[str, bool]:
        """Get the resolution outputs configuration."""
        outputs = self.config.resolution_settings.get('outputs', {})
        self.logger.info(f"Resolution outputs configured: {outputs}")
        return outputs
    
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
                    self.logger.error(f"Asset upload failed for: {failed}")
                    # Try to provide helpful error details
                    for name in failed:
                        if name in upload_results and 'task_id' in upload_results[name]:
                            task_id = upload_results[name]['task_id']
                            self.logger.error(f"   {name}: Check task {task_id} with 'earthengine task info {task_id}'")
                    
                    # Check if this is likely a folder issue
                    self.logger.info("\n💡 Common issues:")
                    self.logger.info("   - Asset folder may not exist (now auto-created)")
                    self.logger.info("   - Permissions issue with service account")
                    self.logger.info("   - Asset name conflicts")
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
            
            # Initialize merger with masking config
            merger_config = {
                'masking': {
                    'enabled': self.config.masking.enabled,
                    'latitude_bounds': self.config.masking.latitude_bounds
                }
            }
            merger = GEEMerger(config=merger_config)
            
            # Load climate if enabled
            climate_image = None
            if 'climate' in self.config.additional_layers and self.config.additional_layers['climate'].enabled:
                climate_layer = self.config.additional_layers['climate']
                self.logger.info(f"Loading climate data from {climate_layer.source}")
                climate_image = merger.load_worldclim_v2(climate_layer.variables)
            else:
                self.logger.info("Climate data disabled in config")
            
            # Check elevation setting
            include_elevation = False
            if 'elevation' in self.config.additional_layers and self.config.additional_layers['elevation'].enabled:
                include_elevation = True
                self.logger.info("Elevation will be included during merge")
            
            # Use recursive resampler for multi-resolution processing
            resolution_outputs = self._get_resolution_outputs()
            base_resolution = self.config.resolution_settings.get('base_resolution', BASE_RESOLUTION_DEGREES)
            
            self.logger.info(f"\n   Using recursive 2x downsampling with base resolution: {base_resolution}°")
            self.logger.info(f"   Outputs configured: {resolution_outputs}")
            
            # Initialize recursive resampler
            resampler = GEEResampler(
                base_resolution=base_resolution,
                outputs=resolution_outputs,
                target_crs='EPSG:4326'
            )
            
            # Process biodiversity datasets through recursive downsampling
            biodiv_resolution_sets = resampler.process_collection(
                biodiv_images, biodiv_configs
            )
            
            # Climate doesn't need recursive resampling - it has native pyramids
            # GEE will automatically use the appropriate pyramid level for the requested scale
            
            # Merge datasets at each resolution
            processed_images = {}
            
            for res_name in biodiv_resolution_sets.keys():
                self.logger.info(f"\n   Merging datasets for {res_name}...")
                
                # Get biodiversity images for this resolution
                biodiv_images_at_res = biodiv_resolution_sets[res_name]
                
                # Climate uses native pyramids - no manual resampling needed
                climate_at_res = climate_image
                
                # Merge datasets
                if climate_at_res is not None:
                    # Calculate scale from resolution name
                    factor = int(res_name[:-1]) if res_name.endswith('x') else 1
                    scale_meters = base_resolution * factor * DEGREES_TO_METERS
                    
                    # Get ocean mask config for this resolution
                    ocean_mask_config = None
                    if self.config.masking.enabled and res_name in resolution_outputs:
                        output_config = resolution_outputs[res_name]
                        if isinstance(output_config, dict) and 'ocean_mask' in output_config:
                            ocean_mask_config = output_config['ocean_mask']
                    
                    merged = merger.merge_datasets(
                        biodiv_images_at_res,
                        climate_at_res,
                        self.config.merge_strategy,
                        mask_ocean=self.config.masking.enabled,
                        include_elevation=include_elevation,
                        scale=scale_meters,
                        ocean_mask_config=ocean_mask_config
                    )
                else:
                    # Climate disabled - just combine biodiversity
                    if len(biodiv_images_at_res) == 1:
                        merged = biodiv_images_at_res[0]
                    else:
                        merged = biodiv_images_at_res[0]
                        for img in biodiv_images_at_res[1:]:
                            merged = merged.addBands(img)
                    
                    # Apply masking if enabled
                    if self.config.masking.enabled:
                        lat_bounds = (self.config.masking.latitude_bounds['min'], 
                                     self.config.masking.latitude_bounds['max'])
                        merged = merger.apply_land_mask(merged, latitude_bounds=lat_bounds)
                
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
            merger_config = {
                'masking': {
                    'enabled': self.config.masking.enabled,
                    'latitude_bounds': self.config.masking.latitude_bounds
                }
            }
            merger = GEEMerger(config=merger_config)
            
            # Export selected resolutions
            task_ids = {}
            
            # Get resolution outputs configuration
            resolution_outputs = self._get_resolution_outputs()
            base_resolution = self.config.resolution_settings.get('base_resolution', BASE_RESOLUTION_DEGREES)
            
            # Prepare images for selected resolutions
            processed_images_for_export = {}
            
            # Process each enabled resolution
            for res_name, res_config in resolution_outputs.items():
                # Expect dict format with 'enabled' key
                if not isinstance(res_config, dict):
                    raise ValueError(f"Resolution config for {res_name} must be a dict with 'enabled' key")
                    
                if not res_config.get('enabled', False):
                    continue
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
                        
                        # No need to unmask since we're uploading without nodata flag
                        # This preserves the distinction between ocean mask and missing data
                        
                        biodiv_images.append(image)
                        biodiv_configs.append(dataset.__dict__)
                
                # Reload climate if enabled
                climate_image = None
                if 'climate' in self.config.additional_layers and self.config.additional_layers['climate'].enabled:
                    climate_layer = self.config.additional_layers['climate']
                    climate_image = merger.load_worldclim_v2(climate_layer.variables)
                
                # Check elevation setting
                include_elevation = False
                if 'elevation' in self.config.additional_layers and self.config.additional_layers['elevation'].enabled:
                    include_elevation = True
                
                # Use recursive resampler
                resampler = GEEResampler(
                    base_resolution=base_resolution,
                    outputs=resolution_outputs,
                    target_crs='EPSG:4326'
                )
                
                # Process biodiversity images for this specific resolution
                biodiv_results = resampler.process_collection(biodiv_images, biodiv_configs)
                biodiv_resampled = biodiv_results.get(res_name, [])
                
                # Climate doesn't need resampling - use native pyramids
                # GEE will automatically select the appropriate pyramid level
                climate_resampled = climate_image
                
                # Merge all datasets - handle case where climate might be disabled
                if climate_resampled is not None:
                    # Calculate scale from resolution name
                    factor = int(res_name[:-1]) if res_name.endswith('x') else 1
                    scale_meters = base_resolution * factor * DEGREES_TO_METERS
                    merged = merger.merge_datasets(
                        biodiv_resampled,
                        climate_resampled,
                        self.config.merge_strategy,
                        mask_ocean=self.config.masking.enabled,
                        include_elevation=include_elevation,
                        scale=scale_meters
                    )
                else:
                    # Climate disabled - just combine biodiversity
                    if len(biodiv_resampled) == 1:
                        merged = biodiv_resampled[0]
                    else:
                        merged = biodiv_resampled[0]
                        for img in biodiv_resampled[1:]:
                            merged = merged.addBands(img)
                    
                    # Apply masking if enabled
                    if self.config.masking.enabled:
                        lat_bounds = (self.config.masking.latitude_bounds['min'], 
                                     self.config.masking.latitude_bounds['max'])
                        merged = merger.apply_land_mask(merged, latitude_bounds=lat_bounds)
                
                processed_images_for_export[res_name] = merged
            
            # Now export each resolution using ExportManager
            self.logger.info(f"\n   Using {self.config.export_format} export format...")
            
            for res_name, merged_image in processed_images_for_export.items():
                self.logger.info(f"\n   Exporting {res_name}...")
                
                # Check if explicit bounds are configured
                export_bounds_config = self.config.export.get('bounds') if self.config.export else None
                if export_bounds_config and all(k in export_bounds_config for k in ['min_lon', 'max_lon', 'min_lat', 'max_lat']):
                    # Use explicit bounds from config
                    self.logger.info(f"   Using explicit export bounds from config")
                    export_region = ee.Geometry.Rectangle([
                        export_bounds_config['min_lon'],
                        export_bounds_config['min_lat'],
                        export_bounds_config['max_lon'],
                        export_bounds_config['max_lat']
                    ])
                else:
                    # For production data, skip expensive bounds detection and use configured bounds
                    # The detection often fails for large global datasets anyway
                    
                    # Use latitude bounds from masking config
                    if self.config.masking.enabled:
                        lat_min = self.config.masking.latitude_bounds.get('min', DEFAULT_LAT_MIN)
                        lat_max = self.config.masking.latitude_bounds.get('max', DEFAULT_LAT_MAX)
                    else:
                        lat_min, lat_max = GLOBAL_LAT_MIN, GLOBAL_LAT_MAX  # Full global if masking disabled
                    
                    self.logger.info(f"   Using configured region with latitude range: [{lat_min}, {lat_max}]")
                    
                    # Create global rectangle by using BBox instead of Rectangle
                    # This avoids the antimeridian collapse issue
                    # Using BBox with specific coordinates ensures proper global coverage
                    export_region = ee.Geometry.BBox(
                        GLOBAL_LON_MIN, lat_min, GLOBAL_LON_MAX, lat_max
                    )
                    self.logger.info("   Using non-geodesic rectangle to avoid antimeridian collapse")
                
                # We only support GeoTIFF format - CSV export was removed due to bugs
                if self.config.export_format != 'GeoTIFF':
                    raise ValueError(f"Unsupported export format: {self.config.export_format}. Only 'GeoTIFF' is supported.")
                
                # Calculate resolution config for this level
                factor = int(res_name[:-1]) if res_name.endswith('x') else 1
                resolution_config = {
                    'resolution': base_resolution * factor,
                    'description': f'{factor}x downsampled ({base_resolution * factor:.6f}°)'
                }
                
                # Export using the public API
                task_id = self._export_manager.export_image(
                    merged_image=merged_image,
                    bounds=export_region,
                    resolution_name=res_name,
                    resolution_config=resolution_config,
                    description=f'export_{res_name}_{self.experiment_name}',
                    export_base_path=self.export_base_path
                )
                
                task_ids[res_name] = task_id
            
            # Save task IDs
            self.state.set_metadata('export_task_ids', task_ids)
            
            # Monitor export tasks using export manager
            self.logger.info(f"\n   Monitoring export tasks...")
            try:
                final_statuses = self._export_manager.monitor_tasks(
                    task_ids=task_ids,
                    timeout=self.config.export_timeout
                )
            except RuntimeError as e:
                # Log which tasks are still running if timeout
                if "timeout" in str(e).lower():
                    # Could extract running tasks from exception if needed
                    pass
                raise  # Re-raise the original exception
            
            # Get export paths - MUST use experiment-specific paths
            export_paths = {
                res: f"gs://{self.config.gcs_bucket}/{self.export_base_path}/{res}/merged_data"
                for res in task_ids.keys()
            }
            self.state.set_metadata('export_paths', export_paths)
            
            # Log the paths for clarity
            self.logger.info("Export paths for this experiment:")
            for res, path in export_paths.items():
                self.logger.info(f"  {res}: {path}")
            
            self.state.update_step(WorkflowStep.EXPORT_COMPLETE)
    
    @log_operation("download_and_convert")
    def _download_and_convert(self) -> Dict[str, Path]:
        """Step 6: Download and convert to parquet"""
        with self.logging_context.stage("download_and_convert"):
            self.logger.info("\n💾 Step 6/6: Downloading and converting to parquet...")
            self.logger.info(f"Experiment: {self.experiment_name}")
            
            # Get export paths
            export_paths = self.state.get_metadata('export_paths', {})
            
            # CRITICAL VALIDATION: Ensure paths match current experiment
            if export_paths:
                for res, path in export_paths.items():
                    if self.experiment_name not in path:
                        self.logger.error(f"⚠️ Path mismatch detected!")
                        self.logger.error(f"  Expected experiment: {self.experiment_name}")
                        self.logger.error(f"  Found path: {path}")
                        self.logger.error(f"  This suggests the state file contains data from a different experiment!")
                        raise ValueError(f"Export paths don't match current experiment '{self.experiment_name}'")
                
                self.logger.info("✅ Export paths validated for current experiment")
                for res, path in export_paths.items():
                    self.logger.info(f"  {res}: {path}")
            
            if not export_paths:
                # For testing, create dummy paths
                self.logger.warning("No export paths found, using test mode")
                resolution_outputs = self._get_resolution_outputs()
                final_outputs = {
                    res: self.config.parquet_dir / f"biodiversity_climate_{res}.parquet"
                    for res in resolution_outputs.keys() if resolution_outputs[res]
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
                
                # Use centralized band configuration
                from ..utils.band_config import BandConfig
                
                # Create band config from current configuration
                band_config = BandConfig.from_config(self.config.__dict__)
                
                # Get metadata from centralized config
                metadata = band_config.to_metadata()
                metadata.update({
                    'format': self.config.export_format,
                    'chunk_size': getattr(self.config, 'parquet_chunk_size', 100000)
                })
                
                self.logger.info(f"Using band order: {metadata['band_names']}")
                
                # Convert based on format
                if self.config.export_format == 'GeoTIFF':
                    # For GeoTIFF, handle multiple scenarios:
                    # 1. Single file (small export)
                    # 2. Multiple tiles (large export automatically tiled by GEE)
                    # 3. Two hemisphere files (_west.tif, _east.tif)
                    final_outputs = {}
                    for res_name, downloaded_files in downloads.items():
                        if downloaded_files:
                            self.logger.info(f"   Processing {len(downloaded_files)} files for {res_name}")
                            
                            # Check what type of files we have
                            file_names = [f.name for f in downloaded_files]
                            
                            # Detect hemisphere files
                            has_hemispheres = any('_west' in name for name in file_names) and \
                                            any('_east' in name for name in file_names)
                            
                            if has_hemispheres:
                                self.logger.info(f"   Detected hemisphere split files")
                            elif len(downloaded_files) > 2:
                                self.logger.info(f"   Detected {len(downloaded_files)} tile files")
                            else:
                                self.logger.info(f"   Processing single file")
                            
                            # Convert all files (converter handles merging internally)
                            output_path = converter.convert(
                                downloaded_files,  # Pass all files
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