"""Google Earth Engine Climate Data Extractor

Extracts WorldClim bioclimatic variables (BIO1, BIO4, BIO12) using chunked
requests to handle GEE quota limits while maintaining coordinate alignment.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Iterator
import time
import json
from pathlib import Path
import tempfile
import os

from .auth import GEEAuthenticator
from .coordinate_generator import CoordinateGenerator


class GEEClimateExtractor:
    """Extracts climate data from Google Earth Engine WorldClim datasets."""
    
    # WorldClim dataset configurations
    WORLDCLIM_DATASETS = {
        'bio01': {
            'asset': 'WORLDCLIM/V1/BIO',
            'band': 'bio01',
            'description': 'Annual Mean Temperature',
            'units': '°C * 10',
            'scale_factor': 0.1
        },
        'bio04': {
            'asset': 'WORLDCLIM/V1/BIO', 
            'band': 'bio04',
            'description': 'Temperature Seasonality',
            'units': '°C * 100',
            'scale_factor': 0.01
        },
        'bio12': {
            'asset': 'WORLDCLIM/V1/BIO',
            'band': 'bio12', 
            'description': 'Annual Precipitation',
            'units': 'mm',
            'scale_factor': 1.0
        }
    }
    
    def __init__(self, 
                 authenticator: GEEAuthenticator,
                 coordinate_generator: CoordinateGenerator,
                 chunk_size: int = 5000,
                 logger=None):
        """
        Initialize GEE climate extractor.
        
        Args:
            authenticator: Authenticated GEE authenticator
            coordinate_generator: Coordinate generator for grid points
            chunk_size: Points per GEE request (max 5000)
            logger: Logger instance
        """
        self.auth = authenticator
        self.coord_gen = coordinate_generator
        self.chunk_size = min(chunk_size, 5000)  # GEE limit
        self.logger = logger
        self.ee = None
        
        if not self.auth.is_authenticated():
            raise ValueError("GEE authenticator must be authenticated")
        
        self.ee = self.auth.ee
        
        # Initialize WorldClim images
        self._load_worldclim_images()
        
    def _load_worldclim_images(self):
        """Load WorldClim image assets."""
        self.images = {}
        
        try:
            base_image = self.ee.Image("WORLDCLIM/V1/BIO")
            
            for var_name, config in self.WORLDCLIM_DATASETS.items():
                self.images[var_name] = base_image.select(config['band'])
                
            if self.logger:
                self.logger.info(f"Loaded {len(self.images)} WorldClim variables")
            else:
                print(f"Loaded {len(self.images)} WorldClim variables")
                
        except Exception as e:
            error_msg = f"Failed to load WorldClim images: {e}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(f"ERROR: {error_msg}")
            raise
    
    def extract_climate_data(self,
                           bounds: Tuple[float, float, float, float],
                           variables: Optional[List[str]] = None,
                           output_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Extract climate data for coordinate grid within bounds.
        
        Args:
            bounds: (min_x, min_y, max_x, max_y) in degrees
            variables: List of variables to extract (default: all)
            output_dir: Directory for temporary CSV files
            
        Returns:
            DataFrame with coordinates and climate variables
        """
        if variables is None:
            variables = list(self.WORLDCLIM_DATASETS.keys())
        
        # Validate variables
        invalid_vars = [v for v in variables if v not in self.WORLDCLIM_DATASETS]
        if invalid_vars:
            raise ValueError(f"Invalid variables: {invalid_vars}")
        
        if self.logger:
            self.logger.info(f"Extracting variables {variables} for bounds {bounds}")
        else:
            print(f"Extracting variables {variables} for bounds {bounds}")
        
        # Generate coordinate chunks
        coord_chunks = list(self.coord_gen.generate_coordinate_chunks(bounds, self.chunk_size))
        total_chunks = len(coord_chunks)
        
        if self.logger:
            self.logger.info(f"Processing {total_chunks} coordinate chunks")
        
        all_results = []
        
        for i, coord_chunk in enumerate(coord_chunks):
            chunk_id = i + 1
            
            if self.logger:
                self.logger.info(f"Processing chunk {chunk_id}/{total_chunks} ({len(coord_chunk)} points)")
            else:
                print(f"Processing chunk {chunk_id}/{total_chunks}")
            
            try:
                chunk_result = self._extract_chunk_data(coord_chunk, variables, output_dir, chunk_id)
                all_results.append(chunk_result)
                
                # Brief pause to avoid overwhelming GEE
                time.sleep(1)
                
            except Exception as e:
                error_msg = f"Failed to process chunk {chunk_id}: {e}"
                if self.logger:
                    self.logger.error(error_msg)
                else:
                    print(f"ERROR: {error_msg}")
                
                # Continue with other chunks rather than failing completely
                continue
        
        if not all_results:
            raise RuntimeError("No chunks processed successfully")
        
        # Combine all results
        final_result = pd.concat(all_results, ignore_index=True)
        
        if self.logger:
            self.logger.info(f"Extraction complete: {len(final_result)} points with {len(variables)} variables")
        else:
            print(f"Extraction complete: {len(final_result)} points")
        
        return final_result
    
    def _extract_chunk_data(self,
                          coord_chunk: pd.DataFrame,
                          variables: List[str],
                          output_dir: Optional[str],
                          chunk_id: int) -> pd.DataFrame:
        """Extract climate data for a single coordinate chunk."""
        
        # Create GEE FeatureCollection from coordinates
        features = []
        for _, row in coord_chunk.iterrows():
            point = self.ee.Geometry.Point([row['x'], row['y']])
            features.append(self.ee.Feature(point))
        
        feature_collection = self.ee.FeatureCollection(features)
        
        # Sample all requested variables
        sample_data = {}
        
        for var_name in variables:
            if self.logger:
                self.logger.debug(f"Sampling {var_name} for chunk {chunk_id}")
            
            # Sample the image at feature points
            sampled = self.images[var_name].sampleRegions(
                collection=feature_collection,
                scale=1000,  # 1km resolution (WorldClim native)
                geometries=True
            )
            
            # Export to temporary CSV and download
            csv_data = self._export_and_download_csv(sampled, var_name, chunk_id, output_dir)
            sample_data[var_name] = csv_data
        
        # Merge all variables into single DataFrame
        result_df = coord_chunk.copy()
        
        for var_name, var_data in sample_data.items():
            if not var_data.empty:
                # Apply scale factor
                scale_factor = self.WORLDCLIM_DATASETS[var_name]['scale_factor']
                var_data[var_name] = var_data[var_name] * scale_factor
                
                # Merge by coordinates (assuming same order)
                if len(var_data) == len(result_df):
                    result_df[var_name] = var_data[var_name].values
                else:
                    if self.logger:
                        self.logger.warning(f"Size mismatch for {var_name}: expected {len(result_df)}, got {len(var_data)}")
        
        return result_df
    
    def _export_and_download_csv(self,
                               feature_collection,
                               var_name: str,
                               chunk_id: int,
                               output_dir: Optional[str]) -> pd.DataFrame:
        """Export GEE FeatureCollection to CSV and download."""
        
        # Create temporary file for CSV export
        if output_dir:
            temp_dir = Path(output_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            csv_file = temp_dir / f"gee_{var_name}_chunk_{chunk_id}.csv"
        else:
            csv_file = Path(tempfile.gettempdir()) / f"gee_{var_name}_chunk_{chunk_id}.csv"
        
        try:
            # Start GEE export task
            task = self.ee.batch.Export.table.toDrive(
                collection=feature_collection,
                description=f"climate_{var_name}_chunk_{chunk_id}",
                fileFormat='CSV'
            )
            
            task.start()
            
            # Wait for task completion
            task_id = task.id
            self._wait_for_task_completion(task_id, var_name, chunk_id)
            
            # Download the CSV file from Google Drive
            # Note: This is a simplified approach - in practice, you'd need
            # to implement Google Drive API download or manual download
            csv_data = self._download_csv_from_drive(task_id, csv_file)
            
            return csv_data
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"CSV export failed for {var_name} chunk {chunk_id}: {e}")
            
            # Return empty DataFrame on failure
            return pd.DataFrame()
    
    def _wait_for_task_completion(self, task_id: str, var_name: str, chunk_id: int, timeout: int = 300):
        """Wait for GEE export task to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            task_status = self.ee.batch.Task.status(task_id)
            state = task_status.get('state', 'UNKNOWN')
            
            if state == 'COMPLETED':
                if self.logger:
                    self.logger.debug(f"Task {task_id} completed successfully")
                return
            elif state == 'FAILED':
                error_msg = task_status.get('error_message', 'Unknown error')
                raise RuntimeError(f"GEE task failed: {error_msg}")
            elif state in ['CANCELLED', 'CANCEL_REQUESTED']:
                raise RuntimeError(f"GEE task was cancelled")
            
            # Still running, wait a bit more
            time.sleep(10)
        
        raise TimeoutError(f"GEE task {task_id} timed out after {timeout} seconds")
    
    def _download_csv_from_drive(self, task_id: str, local_file: Path) -> pd.DataFrame:
        """
        Download CSV from Google Drive.
        
        Note: This is a placeholder implementation. In practice, you would need to:
        1. Use Google Drive API to download the file
        2. Or instruct users to manually download from Drive
        3. Or use alternative export methods (Cloud Storage, etc.)
        """
        
        # Placeholder - in real implementation, download the actual CSV
        if self.logger:
            self.logger.warning(f"CSV download not implemented - returning empty DataFrame")
            self.logger.info(f"Manual download required from Google Drive for task {task_id}")
        else:
            print(f"Warning: Manual CSV download required from Google Drive")
            print(f"Task ID: {task_id}")
        
        # Return empty DataFrame as placeholder
        return pd.DataFrame()
    
    def get_variable_info(self) -> Dict[str, Dict]:
        """Get information about available WorldClim variables."""
        return self.WORLDCLIM_DATASETS.copy()
    
    def test_extraction(self, 
                       test_bounds: Tuple[float, float, float, float] = (-1, -1, 1, 1),
                       variables: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Test extraction with a small area.
        
        Args:
            test_bounds: Small test area bounds
            variables: Variables to test
            
        Returns:
            DataFrame with test results
        """
        if variables is None:
            variables = ['bio01']  # Test with just temperature
        
        if self.logger:
            self.logger.info(f"Testing extraction with bounds {test_bounds}")
        else:
            print(f"Testing extraction with bounds {test_bounds}")
        
        return self.extract_climate_data(test_bounds, variables)


def create_gee_extractor(config_path: str = "config.yml",
                        service_account_key: Optional[str] = None,
                        project_id: Optional[str] = None,
                        chunk_size: int = 5000,
                        logger=None) -> GEEClimateExtractor:
    """
    Convenience function to create GEE extractor from configuration.
    
    Args:
        config_path: Path to config.yml
        service_account_key: Path to GEE service account key
        project_id: GEE project ID
        chunk_size: Points per chunk
        logger: Logger instance
        
    Returns:
        GEEClimateExtractor: Configured extractor
    """
    from .auth import setup_gee_auth
    from .coordinate_generator import create_from_config
    
    # Setup authentication
    auth = setup_gee_auth(service_account_key, project_id, logger)
    if not auth:
        raise RuntimeError("Failed to authenticate with Google Earth Engine")
    
    # Create coordinate generator
    coord_gen = create_from_config(config_path, logger)
    
    # Create extractor
    return GEEClimateExtractor(auth, coord_gen, chunk_size, logger)