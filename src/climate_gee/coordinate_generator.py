"""Coordinate Generator for GEE Climate Data Extraction

Generates coordinate grids that exactly match the existing pipeline
coordinate system for perfect alignment.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Iterator
from pathlib import Path
import yaml


class CoordinateGenerator:
    """Generates coordinate grids matching the existing pipeline."""
    
    def __init__(self, target_resolution: float = 0.016667, logger=None):
        """
        Initialize coordinate generator.
        
        Args:
            target_resolution: Target resolution in degrees (default: ~5km)
            logger: Logger instance
        """
        self.target_resolution = target_resolution
        self.logger = logger
        
    def generate_coordinate_grid(self, 
                               bounds: Tuple[float, float, float, float],
                               chunk_size: Optional[int] = None) -> pd.DataFrame:
        """
        Generate coordinate grid matching pipeline logic.
        
        Args:
            bounds: (min_x, min_y, max_x, max_y) in degrees
            chunk_size: Optional chunk size for memory management
            
        Returns:
            DataFrame with 'x' and 'y' columns
        """
        min_x, min_y, max_x, max_y = bounds
        
        # Create coordinate arrays with improved precision handling
        # Calculate number of points to avoid floating point precision errors
        n_x_points = int(np.round((max_x - min_x) / self.target_resolution))
        n_y_points = int(np.round((max_y - min_y) / self.target_resolution))
        
        # Generate coordinates using linspace for better precision
        x_coords = np.linspace(min_x, min_x + n_x_points * self.target_resolution, 
                              n_x_points, endpoint=False)
        y_coords = np.linspace(min_y, min_y + n_y_points * self.target_resolution, 
                              n_y_points, endpoint=False)
        
        total_points = len(x_coords) * len(y_coords)
        
        if self.logger:
            self.logger.info(f"Coordinate grid: {len(x_coords)} x {len(y_coords)} = {total_points} points")
            self.logger.info(f"Resolution: {self.target_resolution} degrees")
            self.logger.info(f"Bounds: {bounds}")
        else:
            print(f"Coordinate grid: {len(x_coords)} x {len(y_coords)} = {total_points} points")
        
        if chunk_size and total_points > chunk_size:
            if self.logger:
                self.logger.info(f"Using chunked generation with chunk_size={chunk_size}")
            return self._generate_chunked_grid(x_coords, y_coords, chunk_size)
        else:
            return self._generate_full_grid(x_coords, y_coords)
    
    def _generate_full_grid(self, x_coords: np.ndarray, y_coords: np.ndarray) -> pd.DataFrame:
        """Generate full coordinate grid at once."""
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        coords_df = pd.DataFrame({
            'x': xx.flatten(),
            'y': yy.flatten()
        })
        
        return coords_df
    
    def _generate_chunked_grid(self, 
                              x_coords: np.ndarray, 
                              y_coords: np.ndarray,
                              chunk_size: int) -> pd.DataFrame:
        """Generate coordinate grid in chunks to manage memory."""
        all_chunks = []
        
        # Calculate y-chunks based on chunk_size
        total_points = len(x_coords) * len(y_coords)
        rows_per_chunk = max(1, chunk_size // len(x_coords))
        
        for i in range(0, len(y_coords), rows_per_chunk):
            y_chunk = y_coords[i:i + rows_per_chunk]
            
            xx, yy = np.meshgrid(x_coords, y_chunk)
            chunk_df = pd.DataFrame({
                'x': xx.flatten(),
                'y': yy.flatten()
            })
            
            all_chunks.append(chunk_df)
            
            if self.logger:
                self.logger.debug(f"Generated chunk {len(all_chunks)} with {len(chunk_df)} points")
        
        return pd.concat(all_chunks, ignore_index=True)
    
    def generate_coordinate_chunks(self,
                                 bounds: Tuple[float, float, float, float],
                                 chunk_size: int = 5000) -> Iterator[pd.DataFrame]:
        """
        Generate coordinate grid in chunks (iterator for memory efficiency).
        
        Args:
            bounds: (min_x, min_y, max_x, max_y) in degrees
            chunk_size: Points per chunk
            
        Yields:
            DataFrame chunks with 'x' and 'y' columns
        """
        min_x, min_y, max_x, max_y = bounds
        
        # Use same precision-safe coordinate generation
        n_x_points = int(np.round((max_x - min_x) / self.target_resolution))
        n_y_points = int(np.round((max_y - min_y) / self.target_resolution))
        
        x_coords = np.linspace(min_x, min_x + n_x_points * self.target_resolution, 
                              n_x_points, endpoint=False)
        y_coords = np.linspace(min_y, min_y + n_y_points * self.target_resolution, 
                              n_y_points, endpoint=False)
        
        # Calculate rows per chunk
        rows_per_chunk = max(1, chunk_size // len(x_coords))
        
        total_chunks = int(np.ceil(len(y_coords) / rows_per_chunk))
        
        if self.logger:
            self.logger.info(f"Generating {total_chunks} chunks of ~{chunk_size} points each")
        
        for i in range(0, len(y_coords), rows_per_chunk):
            y_chunk = y_coords[i:i + rows_per_chunk]
            
            xx, yy = np.meshgrid(x_coords, y_chunk)
            chunk_df = pd.DataFrame({
                'x': xx.flatten(),
                'y': yy.flatten()
            })
            
            # Add chunk metadata
            chunk_df.attrs['chunk_id'] = i // rows_per_chunk
            chunk_df.attrs['total_chunks'] = total_chunks
            
            if self.logger:
                chunk_id = i // rows_per_chunk + 1
                self.logger.debug(f"Generated chunk {chunk_id}/{total_chunks} with {len(chunk_df)} points")
            
            yield chunk_df
    
    def validate_coordinate_alignment(self, 
                                    coords_df: pd.DataFrame,
                                    expected_bounds: Tuple[float, float, float, float],
                                    tolerance: float = 1e-6) -> bool:
        """
        Validate that generated coordinates align with expected bounds.
        
        Args:
            coords_df: DataFrame with coordinates
            expected_bounds: Expected (min_x, min_y, max_x, max_y)
            tolerance: Tolerance for floating point comparison
            
        Returns:
            bool: True if alignment is valid
        """
        if coords_df.empty:
            if self.logger:
                self.logger.error("Empty coordinate DataFrame")
            return False
        
        actual_bounds = (
            coords_df['x'].min(),
            coords_df['y'].min(), 
            coords_df['x'].max(),
            coords_df['y'].max()
        )
        
        expected_min_x, expected_min_y, expected_max_x, expected_max_y = expected_bounds
        actual_min_x, actual_min_y, actual_max_x, actual_max_y = actual_bounds
        
        # Check alignment within tolerance
        x_min_ok = abs(actual_min_x - expected_min_x) < tolerance
        y_min_ok = abs(actual_min_y - expected_min_y) < tolerance
        
        # Max bounds might be slightly different due to grid alignment
        x_range_ok = actual_max_x <= expected_max_x + self.target_resolution
        y_range_ok = actual_max_y <= expected_max_y + self.target_resolution
        
        is_valid = x_min_ok and y_min_ok and x_range_ok and y_range_ok
        
        if self.logger:
            self.logger.info(f"Coordinate validation:")
            self.logger.info(f"  Expected bounds: {expected_bounds}")
            self.logger.info(f"  Actual bounds: {actual_bounds}")
            self.logger.info(f"  Resolution: {self.target_resolution}")
            self.logger.info(f"  Valid: {is_valid}")
        
        return is_valid


def load_config_resolution(config_path: str = "config.yml") -> float:
    """
    Load target resolution from config.yml file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        float: Target resolution from config
    """
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            print(f"Warning: Config file {config_path} not found, using default resolution")
            return 0.016667
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        resolution = config.get('resampling', {}).get('target_resolution', 0.016667)
        print(f"Loaded target resolution from config: {resolution}")
        return resolution
        
    except Exception as e:
        print(f"Warning: Failed to load config {config_path}: {e}")
        return 0.016667


def get_processing_bounds(config_path: str = "config.yml", 
                         bounds_key: str = "global") -> Tuple[float, float, float, float]:
    """
    Load processing bounds from config.yml file.
    
    Args:
        config_path: Path to config file
        bounds_key: Key for bounds in processing_bounds section
        
    Returns:
        Tuple: (min_x, min_y, max_x, max_y) bounds
    """
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            print(f"Warning: Config file {config_path} not found, using global bounds")
            return (-180, -90, 180, 90)
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        bounds = config.get('processing_bounds', {}).get(bounds_key, [-180, -90, 180, 90])
        print(f"Loaded processing bounds '{bounds_key}': {bounds}")
        return tuple(bounds)
        
    except Exception as e:
        print(f"Warning: Failed to load bounds from config {config_path}: {e}")
        return (-180, -90, 180, 90)


# Convenience function for creating coordinate generator from config
def create_from_config(config_path: str = "config.yml", logger=None) -> CoordinateGenerator:
    """
    Create CoordinateGenerator instance from config file.
    
    Args:
        config_path: Path to config file
        logger: Logger instance
        
    Returns:
        CoordinateGenerator: Configured instance
    """
    resolution = load_config_resolution(config_path)
    return CoordinateGenerator(target_resolution=resolution, logger=logger)