"""Simplified orchestrator without logging context issues"""

import ee
import time
import logging
from typing import Dict, Optional, List
from pathlib import Path

from ..config import GEEConfig
from ..auth import GEEAuthenticator
from ..processing import GEEMerger
from ..processing.recursive_resampler import RecursiveResampler
from ..processing.export_manager import ExportManager
from .state_manager import WorkflowState, WorkflowStep

logger = logging.getLogger(__name__)


class SimpleOrchestrator:
    """Simplified workflow orchestrator without complex logging contexts"""
    
    def __init__(self, config_path: str, experiment_name: str):
        """Initialize simplified orchestrator"""
        self.config = GEEConfig.from_yaml(config_path)
        self.experiment_name = experiment_name
        self.export_base_path = experiment_name
        
        # Setup state management
        state_file = Path.home() / '.biodiversity' / f'gee_workflow_state_{experiment_name}.json'
        self.state = WorkflowState.load_or_create(state_file)
        
        logger.info(f"Initialized SimpleOrchestrator for experiment: {experiment_name}")
    
    def run(self, resume: bool = True) -> Dict:
        """Run the workflow"""
        
        logger.info("=" * 70)
        logger.info(f"Starting Simple Workflow: {self.experiment_name}")
        logger.info("=" * 70)
        
        # Check resume
        if resume and self.state.current_step != WorkflowStep.NOT_STARTED:
            logger.info(f"Resuming from step: {self.state.current_step.value}")
        
        try:
            # Step 1: Authenticate
            if not self.state.is_step_complete(WorkflowStep.AUTHENTICATED):
                self._authenticate()
            
            # Step 2: Process and Export (combined for simplicity)
            if not self.state.is_step_complete(WorkflowStep.EXPORT_COMPLETE):
                self._process_and_export()
            
            logger.info("✅ Workflow completed successfully!")
            return {'success': True, 'experiment': self.experiment_name}
            
        except Exception as e:
            logger.error(f"Workflow failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _authenticate(self):
        """Authenticate with GEE"""
        logger.info("\n🔐 Authenticating with GEE...")
        
        auth = GEEAuthenticator(self.config.__dict__)
        success = auth.authenticate(
            method=self.config.auth_method,
            service_account_key=self.config.service_account_key,
            project_id=self.config.project_id
        )
        
        if success:
            self.state.update_step(WorkflowStep.AUTHENTICATED)
            logger.info("   ✓ Authentication successful")
        else:
            raise RuntimeError("Authentication failed")
    
    def _process_and_export(self):
        """Process data and export (combined step)"""
        logger.info("\n⚙️ Processing and exporting data...")
        
        # Get resolution settings
        resolution_outputs = self.config.resolution_settings.get('outputs', {})
        base_resolution = self.config.resolution_settings.get('base_resolution', 0.016666666666667)
        
        logger.info(f"Resolution outputs: {resolution_outputs}")
        logger.info(f"Base resolution: {base_resolution}")
        
        # Load biodiversity datasets
        logger.info("\nLoading biodiversity datasets...")
        biodiv_images = []
        biodiv_configs = []
        
        datasets = [
            ('animals', 'animal_richness', 'continuous', 'max'),
            ('plants', 'plant_richness', 'continuous', 'max'),
            ('am_fungi', 'am_fungi_richness', 'continuous', 'max'),
            ('am_fungi_cv', 'am_fungi_cv', 'cv', 'mean'),
            ('ecm_fungi', 'ecm_fungi_richness', 'continuous', 'max'),
            ('ecm_fungi_cv', 'ecm_fungi_cv', 'cv', 'mean')
        ]
        
        for name, band_name, data_type, resampling in datasets:
            try:
                asset_id = f'projects/earthengine-legacy/assets/users/geebiodiversity/production/{name}'
                image = ee.Image(asset_id)
                image = image.rename([band_name])
                biodiv_images.append(image)
                biodiv_configs.append({
                    'name': name,
                    'band_name': band_name,
                    'data_type': data_type,
                    'resampling': resampling
                })
                logger.info(f"   ✓ Loaded {name}")
            except Exception as e:
                logger.warning(f"   ⚠️ Skipping {name}: {str(e)[:50]}")
        
        # Setup merger
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
            climate_image = merger.load_worldclim_v2(climate_layer.variables)
            logger.info(f"   ✓ Loaded climate data: {climate_layer.variables}")
        
        # Check elevation
        include_elevation = False
        if 'elevation' in self.config.additional_layers and self.config.additional_layers['elevation'].enabled:
            include_elevation = True
            logger.info("   ✓ Will include elevation")
        
        # Process each resolution
        for res_name, should_export in resolution_outputs.items():
            if not should_export:
                continue
            
            logger.info(f"\nProcessing {res_name} resolution...")
            
            # Use recursive resampler
            resampler = RecursiveResampler(
                base_resolution=base_resolution,
                outputs={res_name: True},  # Only process this resolution
                target_crs='EPSG:4326'
            )
            
            # Process biodiversity
            biodiv_results = resampler.process_collection(biodiv_images, biodiv_configs)
            biodiv_resampled = biodiv_results.get(res_name, [])
            
            if not biodiv_resampled:
                logger.error(f"   Failed to get {res_name} biodiversity images!")
                continue
            
            logger.info(f"   ✓ Got {len(biodiv_resampled)} biodiversity images at {res_name}")
            
            # Process climate if enabled
            climate_resampled = None
            if climate_image is not None:
                climate_config = {
                    'name': 'climate',
                    'band_name': 'climate',
                    'data_type': 'continuous',
                    'resampling': 'bilinear'
                }
                climate_results = resampler.process_dataset(climate_image, climate_config)
                climate_resampled = climate_results.get(res_name)
                if climate_resampled:
                    logger.info(f"   ✓ Climate processed to {res_name}")
            
            # Merge all datasets
            factor = int(res_name[:-1]) if res_name.endswith('x') else 1
            scale_meters = base_resolution * factor * 111320
            
            if climate_resampled is not None:
                merged = merger.merge_datasets(
                    biodiv_resampled,
                    climate_resampled,
                    self.config.merge_strategy,
                    mask_ocean=self.config.masking.enabled,
                    include_elevation=include_elevation,
                    scale=scale_meters
                )
            else:
                # Just combine biodiversity
                merged = biodiv_resampled[0]
                for img in biodiv_resampled[1:]:
                    merged = merged.addBands(img)
            
            logger.info(f"   ✓ Merged datasets for {res_name}")
            
            # Export
            self._export_resolution(merged, res_name, base_resolution * factor, scale_meters)
        
        self.state.update_step(WorkflowStep.EXPORT_COMPLETE)
        logger.info("\n✅ All exports started successfully!")
    
    def _export_resolution(self, image: ee.Image, res_name: str, resolution: float, scale_meters: float):
        """Export a single resolution"""
        
        # Get export bounds
        if self.config.masking.enabled:
            lat_min = self.config.masking.latitude_bounds.get('min', -60)
            lat_max = self.config.masking.latitude_bounds.get('max', 75)
        else:
            lat_min, lat_max = -90, 90
        
        export_region = ee.Geometry.BBox(-180, lat_min, 180, lat_max)
        
        # Start export
        task = ee.batch.Export.image.toCloudStorage(
            image=image,
            description=f'{self.experiment_name}_{res_name}_export',
            bucket=self.config.gcs_bucket,
            fileNamePrefix=f'{self.export_base_path}/{res_name}/merged_data',
            region=export_region,
            scale=scale_meters,
            crs='EPSG:4326',
            fileFormat='GeoTIFF',
            formatOptions={
                'cloudOptimized': True,
                'noData': 0
            },
            maxPixels=1e13
        )
        
        task.start()
        logger.info(f"   Export task started for {res_name}: {task.id}")
        
        # Store task ID
        export_tasks = self.state.get_metadata('export_tasks', {})
        export_tasks[res_name] = task.id
        self.state.set_metadata('export_tasks', export_tasks)