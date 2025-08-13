"""Utilities for stat pipeline data processing.

This module provides utilities to convert BiodiversityData objects to DataFrames
and handle data preprocessing for statistical analysis methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path

from src.abstractions.types.biodiversity_types import BiodiversityData
from src.biodiversity_analysis.shared.data.parquet_loader import ParquetLoader
from src.biodiversity_analysis.preprocessing.pipeline import SharedPreprocessor

logger = logging.getLogger(__name__)


class StatDataConverter:
    """Convert BiodiversityData to formats needed for statistical analysis."""
    
    @staticmethod
    def biodiv_to_dataframe(biodiv_data: BiodiversityData, 
                           include_coords: bool = True) -> pd.DataFrame:
        """Convert BiodiversityData object to pandas DataFrame.
        
        Args:
            biodiv_data: BiodiversityData object from ParquetLoader
            include_coords: Whether to include coordinate columns
            
        Returns:
            DataFrame with features and optionally coordinates
        """
        data_dict = {}
        
        # Add coordinates if requested
        if include_coords:
            data_dict['latitude'] = biodiv_data.coordinates[:, 0]
            data_dict['longitude'] = biodiv_data.coordinates[:, 1]
        
        # Add features with their names
        for i, name in enumerate(biodiv_data.feature_names):
            data_dict[name] = biodiv_data.features[:, i]
        
        return pd.DataFrame(data_dict)
    
    @staticmethod
    def biodiv_to_arrays(biodiv_data: BiodiversityData) -> Dict[str, np.ndarray]:
        """Convert BiodiversityData to dictionary of numpy arrays.
        
        Args:
            biodiv_data: BiodiversityData object
            
        Returns:
            Dictionary mapping feature names to arrays
        """
        arrays = {}
        for i, name in enumerate(biodiv_data.feature_names):
            arrays[name] = biodiv_data.features[:, i]
        
        # Add coordinates as separate arrays
        arrays['latitude'] = biodiv_data.coordinates[:, 0]
        arrays['longitude'] = biodiv_data.coordinates[:, 1]
        
        return arrays


class StatDataPreprocessor:
    """Preprocessing specifically for statistical analysis of richness data."""
    
    def __init__(self, variant: str = '4_features'):
        """Initialize preprocessor.
        
        Args:
            variant: '4_features' or '3_features_merged'
        """
        self.variant = variant
        self.preprocessor = SharedPreprocessor(
            method_name='stat',
            normalize=False,  # Stats methods handle their own normalization
            missing_strategy='zero',  # Replace NaN with 0 for richness
            combine_fungi=(variant == '3_features_merged'),
            feature_names=['plant_richness', 'animal_richness', 
                          'am_fungi_richness', 'ecm_fungi_richness']
        )
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess DataFrame for statistical analysis.
        
        Args:
            df: Input DataFrame with richness columns
            
        Returns:
            Cleaned DataFrame
        """
        # Apply richness-specific cleaning
        for col in df.columns:
            if 'richness' in col.lower():
                # Handle extreme negative values in fungi
                if 'fungi' in col.lower():
                    df.loc[df[col] < -1e20, col] = np.nan
                
                # Negative richness is invalid
                df.loc[df[col] < 0, col] = np.nan
                
                # Zero richness treated as no-data
                df.loc[df[col] == 0, col] = np.nan
        
        # Handle variant-specific processing
        if self.variant == '3_features_merged':
            if 'am_fungi_richness' in df.columns and 'ecm_fungi_richness' in df.columns:
                # Create total fungi column
                df['total_fungi_richness'] = (
                    df['am_fungi_richness'].fillna(0) + 
                    df['ecm_fungi_richness'].fillna(0)
                )
                # Clean the merged column
                df.loc[df['total_fungi_richness'] == 0, 'total_fungi_richness'] = np.nan
                
                # Drop individual fungi columns
                df = df.drop(['am_fungi_richness', 'ecm_fungi_richness'], axis=1)
        
        # Remove rows with all NaN in richness columns
        richness_cols = [col for col in df.columns if 'richness' in col.lower()]
        df = df.dropna(subset=richness_cols, how='all')
        
        return df
    
    def preprocess_arrays(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Preprocess dictionary of arrays.
        
        Args:
            data: Dictionary mapping names to arrays
            
        Returns:
            Cleaned dictionary of arrays
        """
        # Convert to numpy array for preprocessing
        feature_names = []
        arrays = []
        
        for name in ['plant_richness', 'animal_richness', 
                    'am_fungi_richness', 'ecm_fungi_richness']:
            if name in data:
                feature_names.append(name)
                arrays.append(data[name])
        
        if not arrays:
            return data
        
        # Stack into 2D array
        stacked = np.column_stack(arrays)
        
        # Apply cleaning
        cleaned = self.preprocessor.clean_richness_data(stacked, feature_names)
        
        # Handle fungi merging if needed
        if self.variant == '3_features_merged':
            cleaned = self.preprocessor._combine_fungi_features(cleaned)
            # Update feature names
            feature_names = ['plant_richness', 'animal_richness', 'total_fungi_richness']
        
        # Convert back to dictionary
        result = {}
        for i, name in enumerate(feature_names):
            result[name] = cleaned[:, i]
        
        # Add back non-feature arrays (coordinates)
        for name in data:
            if name not in result and 'richness' not in name:
                result[name] = data[name]
        
        return result


class OptimizedStatLoader:
    """Optimized loader for statistical analysis using existing infrastructure."""
    
    def __init__(self):
        """Initialize with base loader."""
        self.base_loader = ParquetLoader()
        self.converter = StatDataConverter()
        self.preprocessor = None  # Created based on variant
    
    def load_parquet_for_stats(self, 
                              path: str, 
                              variant: str = '4_features',
                              output_format: str = 'dataframe') -> Union[pd.DataFrame, Dict[str, np.ndarray]]:
        """Load parquet file optimized for statistical analysis.
        
        Args:
            path: Path to parquet file
            variant: '4_features' or '3_features_merged'
            output_format: 'dataframe' or 'arrays'
            
        Returns:
            Preprocessed data in requested format
        """
        # Initialize preprocessor for variant
        self.preprocessor = StatDataPreprocessor(variant)
        
        # Define feature columns based on variant
        # Always load all 4 features, then merge if needed
        feature_cols = ['plant_richness', 'animal_richness',
                       'am_fungi_richness', 'ecm_fungi_richness']
        coord_cols = ['latitude', 'longitude']
        
        # Load with optimizations
        logger.info(f"Loading parquet file for stats: {path}")
        biodiv_data = self.base_loader.load(
            data_path=path,
            coordinate_cols=coord_cols,
            feature_cols=feature_cols,
            species_cols=None,  # Not needed for richness analysis
            validate=False,  # Skip validation for speed
            spatial_grid_step=1,  # No sampling
            skip_auto_detection=True,  # We provide explicit columns
            skip_coordinate_validation=True,  # Trust the data
            skip_zero_inflation_check=True,  # Not needed for stats
            skip_thread_management=False  # Keep for memory management
        )
        
        # Convert to requested format
        if output_format == 'dataframe':
            df = self.converter.biodiv_to_dataframe(biodiv_data, include_coords=True)
            # Apply preprocessing
            df = self.preprocessor.preprocess_dataframe(df)
            logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
            return df
        else:
            arrays = self.converter.biodiv_to_arrays(biodiv_data)
            # Apply preprocessing
            arrays = self.preprocessor.preprocess_arrays(arrays)
            logger.info(f"Loaded {len(arrays)} arrays")
            return arrays


def load_parquet_for_correlation(path: str, variant: str = '4_features') -> pd.DataFrame:
    """Convenience function to load parquet for correlation analysis.
    
    Args:
        path: Path to parquet file
        variant: Dataset variant
        
    Returns:
        DataFrame ready for correlation analysis
    """
    loader = OptimizedStatLoader()
    return loader.load_parquet_for_stats(path, variant, 'dataframe')


def load_parquet_for_spatial(path: str, variant: str = '4_features') -> Dict[str, np.ndarray]:
    """Convenience function to load parquet for spatial analysis.
    
    Args:
        path: Path to parquet file
        variant: Dataset variant
        
    Returns:
        Dictionary of arrays for spatial methods
    """
    loader = OptimizedStatLoader()
    return loader.load_parquet_for_stats(path, variant, 'arrays')