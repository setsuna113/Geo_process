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
        """Extract climate data for a single coordinate chunk using efficient batch sampling."""
        
        # Create list of coordinate points for sampling
        points = [[row['x'], row['y']] for _, row in coord_chunk.iterrows()]
        
        # Initialize result DataFrame
        result_df = coord_chunk.copy()
        
        for var_name in variables:
            if self.logger:
                self.logger.debug(f"Sampling {var_name} for chunk {chunk_id} ({len(points)} points)")
            
            try:
                # Use efficient batch sampling
                raw_values = self._sample_image_batch(
                    self.images[var_name], 
                    points, 
                    scale=1000  # 1km resolution (WorldClim native)
                )
                
                # Apply scale factor to valid values
                scale_factor = self.WORLDCLIM_DATASETS[var_name]['scale_factor']
                scaled_values = []
                
                for raw_value in raw_values:
                    if raw_value is not None and not np.isnan(raw_value):
                        scaled_values.append(raw_value * scale_factor)
                    else:
                        scaled_values.append(np.nan)
                
                # Add values to result DataFrame
                result_df[var_name] = scaled_values
                
                if self.logger:
                    valid_count = sum(1 for v in scaled_values if not np.isnan(v))
                    self.logger.debug(f"Extracted {valid_count}/{len(scaled_values)} valid values for {var_name}")
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to sample {var_name} for chunk {chunk_id}: {e}")
                # Fill with NaN values on failure
                result_df[var_name] = np.nan
        
        return result_df
    
    def _sample_image_batch(self, 
                          image: any, 
                          points: List[List[float]], 
                          scale: int = 1000) -> List[float]:
        """
        Efficiently sample image at multiple points using batch processing.
        
        Args:
            image: GEE Image to sample
            points: List of [lon, lat] coordinates
            scale: Sampling scale in meters
            
        Returns:
            List of sampled values (may contain NaN for invalid pixels)
        """
        try:
            # Create FeatureCollection from points for batch sampling
            features = []
            for i, point in enumerate(points):
                geom = self.ee.Geometry.Point(point)
                features.append(self.ee.Feature(geom, {'point_id': i}))
            
            feature_collection = self.ee.FeatureCollection(features)
            
            # Sample the image at all points
            sampled = image.sampleRegions(
                collection=feature_collection,
                scale=scale,
                geometries=False
            )
            
            # Get the results as a list
            sampled_list = sampled.getInfo()
            
            # Extract values in correct order
            values = [np.nan] * len(points)  # Initialize with NaN
            
            if sampled_list and 'features' in sampled_list:
                for feature in sampled_list['features']:
                    if 'properties' in feature:
                        props = feature['properties']
                        point_id = props.get('point_id')
                        
                        # Get the band value (first band in image)
                        band_names = list(props.keys())
                        band_names = [b for b in band_names if b != 'point_id']
                        
                        if band_names and point_id is not None and 0 <= point_id < len(values):
                            values[point_id] = props.get(band_names[0])
            
            return values
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Batch sampling failed, falling back to individual sampling: {e}")
            # Fallback to individual sampling
            return self._sample_image_individual(image, points, scale)
    
    def _sample_image_individual(self, 
                               image: any, 
                               points: List[List[float]], 
                               scale: int = 1000) -> List[float]:
        """Fallback method for individual point sampling."""
        values = []
        
        for point in points:
            try:
                geom = self.ee.Geometry.Point(point)
                sampled = image.sample(geom, scale=scale).first()
                value_info = sampled.getInfo()
                
                if value_info and 'properties' in value_info:
                    # Get first property value (should be the band value)
                    props = value_info['properties']
                    band_values = [v for k, v in props.items() if k != 'system:index']
                    if band_values:
                        values.append(band_values[0])
                    else:
                        values.append(np.nan)
                else:
                    values.append(np.nan)
                    
            except Exception:
                values.append(np.nan)
        
        return values
    
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