"""
Data schema validation for statistical analysis.

Linus: "Bad data in, bad results out. Check your inputs."
Simple validation, no over-engineering.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DataSchema:
    """Validate data against expected schema."""
    
    @staticmethod
    def validate_biodiversity_data(data: Dict[str, np.ndarray]) -> bool:
        """Validate biodiversity data dictionary.
        
        Args:
            data: Dictionary with biodiversity arrays
            
        Returns:
            True if valid, raises ValueError if not
        """
        required_keys = {'plants', 'animals'}
        optional_keys = {'am_fungi', 'ecm_fungi', 'fungi', 'merged_fungi'}
        
        # Check required keys
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required dataset: {key}")
            if not isinstance(data[key], np.ndarray):
                raise ValueError(f"{key} must be numpy array")
            if data[key].ndim != 2:
                raise ValueError(f"{key} must be 2D array")
        
        # Check all arrays have same shape
        reference_shape = data['plants'].shape
        for key, arr in data.items():
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                if arr.shape != reference_shape:
                    raise ValueError(f"Shape mismatch: {key} has shape {arr.shape}, "
                                   f"expected {reference_shape}")
        
        # Check data ranges
        for key, arr in data.items():
            if isinstance(arr, np.ndarray):
                valid_data = arr[~np.isnan(arr)]
                if len(valid_data) > 0:
                    if np.any(valid_data < 0):
                        logger.warning(f"{key} has negative values")
                    if np.all(valid_data == 0):
                        logger.warning(f"{key} has all zeros")
        
        return True
    
    @staticmethod
    def validate_climate_data(data: Dict[str, np.ndarray]) -> bool:
        """Validate climate data.
        
        Args:
            data: Dictionary with climate arrays
            
        Returns:
            True if valid
        """
        # Temperature should be in reasonable range
        if 'temperature' in data:
            temp = data['temperature']
            valid_temp = temp[~np.isnan(temp)]
            if len(valid_temp) > 0:
                if np.any(valid_temp < -100) or np.any(valid_temp > 60):
                    raise ValueError("Temperature out of range [-100, 60]°C")
        
        # Precipitation should be non-negative
        if 'precipitation' in data:
            precip = data['precipitation']
            valid_precip = precip[~np.isnan(precip)]
            if len(valid_precip) > 0:
                if np.any(valid_precip < 0):
                    raise ValueError("Precipitation cannot be negative")
                if np.any(valid_precip > 20000):
                    logger.warning("Precipitation > 20000mm/year seems unrealistic")
        
        return True
    
    @staticmethod
    def validate(df, method_name: str):
        """Validate and complete DataFrame for analysis.
        
        Simple validation - ensure required columns exist.
        Add missing columns with defaults.
        
        Args:
            df: DataFrame to validate
            method_name: Method calling validation
            
        Returns:
            Validated DataFrame
        """
        import pandas as pd
        
        # Ensure it's a DataFrame
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{method_name} requires DataFrame")
        
        # Required columns for any analysis
        required = ['x', 'y']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Climate columns for biome analysis
        if method_name == 'biome_analysis':
            if 'bio01' not in df.columns:
                raise ValueError("biome_analysis requires bio01 (temperature)")
            if 'bio12' not in df.columns:
                raise ValueError("biome_analysis requires bio12 (precipitation)")
        
        # Add elevation if missing
        if 'elevation' not in df.columns:
            logger.info("No elevation data, using sea level (0m)")
            df['elevation'] = 0
        
        return df
    
    @staticmethod
    def get_biodiversity_columns(df) -> List[str]:
        """Get biodiversity columns from DataFrame.
        
        Args:
            df: DataFrame
            
        Returns:
            List of biodiversity column names
        """
        # Known biodiversity columns
        biodiv_names = {
            'plants', 'animals', 
            'am_fungi', 'ecm_fungi', 'fungi', 'merged_fungi',
            'plant_richness', 'animal_richness', 'fungal_richness'
        }
        
        # Find which ones exist in the DataFrame
        biodiv_cols = []
        for col in df.columns:
            if col.lower() in biodiv_names or 'richness' in col.lower():
                biodiv_cols.append(col)
        
        if not biodiv_cols:
            logger.warning("No biodiversity columns found")
        
        return biodiv_cols
    
    @staticmethod
    def get_data_summary(data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Get summary statistics for data.
        
        Args:
            data: Dictionary with arrays
            
        Returns:
            Summary statistics
        """
        summary = {}
        
        for key, arr in data.items():
            if isinstance(arr, np.ndarray):
                valid_data = arr[~np.isnan(arr)]
                summary[key] = {
                    'shape': arr.shape,
                    'dtype': str(arr.dtype),
                    'min': float(np.min(valid_data)) if len(valid_data) > 0 else None,
                    'max': float(np.max(valid_data)) if len(valid_data) > 0 else None,
                    'mean': float(np.mean(valid_data)) if len(valid_data) > 0 else None,
                    'std': float(np.std(valid_data)) if len(valid_data) > 0 else None,
                    'nan_count': int(np.isnan(arr).sum()),
                    'zero_count': int((arr == 0).sum()),
                    'coverage': float((~np.isnan(arr)).mean())
                }
        
        return summary