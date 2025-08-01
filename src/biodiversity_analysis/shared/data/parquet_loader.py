"""Parquet data loader for biodiversity analysis."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

from src.abstractions.types.biodiversity_types import BiodiversityData

logger = logging.getLogger(__name__)


class ParquetLoader:
    """Load biodiversity data from parquet files."""
    
    def __init__(self):
        """Initialize parquet loader."""
        self.last_loaded_path = None
        self.last_loaded_data = None
    
    def load(self, 
             data_path: str,
             coordinate_cols: Optional[List[str]] = None,
             feature_cols: Optional[List[str]] = None,
             species_cols: Optional[List[str]] = None,
             validate: bool = True) -> BiodiversityData:
        """Load biodiversity data from parquet file.
        
        Args:
            data_path: Path to parquet file
            coordinate_cols: Names of coordinate columns (auto-detect if None)
            feature_cols: Feature columns to use (all numeric if None)
            species_cols: Species column names if applicable
            validate: Whether to validate the loaded data
            
        Returns:
            BiodiversityData object
            
        Raises:
            FileNotFoundError: If parquet file doesn't exist
            ValueError: If data validation fails
        """
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {data_path}")
        
        logger.info(f"Loading parquet file: {data_path}")
        
        # Load parquet file
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Auto-detect coordinate columns if not specified
        if coordinate_cols is None:
            coordinate_cols = self._detect_coordinate_columns(df)
        
        # Extract coordinates
        if len(coordinate_cols) != 2:
            raise ValueError(f"Expected 2 coordinate columns, got {len(coordinate_cols)}: {coordinate_cols}")
        
        coordinates = df[coordinate_cols].values
        
        # Auto-detect feature columns if not specified
        if feature_cols is None:
            # Use all numeric columns except coordinates
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col not in coordinate_cols]
            logger.info(f"Auto-detected {len(feature_cols)} feature columns")
        
        # Extract features
        features = df[feature_cols].values
        feature_names = feature_cols
        
        # Extract species data if specified
        species_data = None
        if species_cols:
            species_data = df[species_cols].values
        
        # Check for zero inflation
        zero_inflated = self._check_zero_inflation(features)
        
        # Create metadata
        metadata = {
            'source_file': str(path),
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'coordinate_columns': coordinate_cols,
            'feature_columns': feature_cols,
            'species_columns': species_cols,
            'missing_proportion': np.isnan(features).mean(),
            'zero_proportion': (features == 0).mean()
        }
        
        # Create BiodiversityData object
        biodiv_data = BiodiversityData(
            features=features,
            coordinates=coordinates,
            feature_names=feature_names,
            coordinate_system='WGS84',  # Assume WGS84
            metadata=metadata,
            zero_inflated=zero_inflated,
            species_names=species_cols  # Use species column names if provided
        )
        
        # Validate if requested
        if validate:
            self._validate_data(biodiv_data)
        
        # Cache for potential reuse
        self.last_loaded_path = str(path)
        self.last_loaded_data = biodiv_data
        
        return biodiv_data
    
    def _detect_coordinate_columns(self, df: pd.DataFrame) -> List[str]:
        """Auto-detect coordinate columns from dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            List of coordinate column names [longitude, latitude]
        """
        # Common coordinate column names
        lon_names = ['longitude', 'lon', 'x', 'lng', 'long']
        lat_names = ['latitude', 'lat', 'y']
        
        lon_col = None
        lat_col = None
        
        # Case-insensitive search
        df_cols_lower = {col.lower(): col for col in df.columns}
        
        # Find longitude column
        for name in lon_names:
            if name in df_cols_lower:
                lon_col = df_cols_lower[name]
                break
        
        # Find latitude column
        for name in lat_names:
            if name in df_cols_lower:
                lat_col = df_cols_lower[name]
                break
        
        if lon_col is None or lat_col is None:
            raise ValueError(
                f"Could not auto-detect coordinate columns. "
                f"Please specify coordinate_cols parameter. "
                f"Available columns: {list(df.columns)}"
            )
        
        logger.info(f"Auto-detected coordinate columns: [{lon_col}, {lat_col}]")
        return [lon_col, lat_col]
    
    def _check_zero_inflation(self, features: np.ndarray, threshold: float = 0.5) -> bool:
        """Check if features show zero inflation.
        
        Args:
            features: Feature array
            threshold: Proportion of zeros to consider zero-inflated
            
        Returns:
            True if zero-inflated
        """
        zero_prop = (features == 0).mean()
        return zero_prop > threshold
    
    def _validate_data(self, data: BiodiversityData) -> None:
        """Validate loaded biodiversity data.
        
        Args:
            data: BiodiversityData to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Check shapes
        n_samples = data.n_samples
        if data.coordinates.shape[0] != n_samples:
            raise ValueError(
                f"Coordinate samples ({data.coordinates.shape[0]}) "
                f"don't match feature samples ({n_samples})"
            )
        
        # Check coordinate bounds
        lon_min, lon_max = data.coordinates[:, 0].min(), data.coordinates[:, 0].max()
        lat_min, lat_max = data.coordinates[:, 1].min(), data.coordinates[:, 1].max()
        
        if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
            logger.warning(f"Longitude values outside [-180, 180]: [{lon_min}, {lon_max}]")
        
        if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
            logger.warning(f"Latitude values outside [-90, 90]: [{lat_min}, {lat_max}]")
        
        # Check for non-finite values in features
        if not np.isfinite(data.features).any():
            raise ValueError("No finite values found in features")
        
        # Check feature names match feature count
        if len(data.feature_names) != data.n_features:
            raise ValueError(
                f"Feature names count ({len(data.feature_names)}) "
                f"doesn't match feature count ({data.n_features})"
            )
        
        logger.info("Data validation passed")