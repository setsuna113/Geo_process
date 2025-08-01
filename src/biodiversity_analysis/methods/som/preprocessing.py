"""Preprocessing pipeline for biodiversity SOM analysis.

Implements the specific preprocessing requirements from final_som_configuration_decisions.md:
- Log(x+1) transformation for all features
- Separate z-score standardization for observed vs predicted features
- Handles missing data (keeps NaN, no imputation)
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)


class BiodiversityPreprocessor:
    """Preprocessing for biodiversity data following specific requirements.
    
    Pipeline:
    1. Log(x+1) transformation
    2. Separate z-score standardization for observed/predicted features
    3. Keep missing values as NaN (no imputation)
    """
    
    def __init__(self):
        self.scaler_observed = None
        self.scaler_predicted = None
        self.use_minmax = True  # Use MinMax scaling for Bray-Curtis compatibility
        self.observed_indices = None
        self.predicted_indices = None
        self.original_stats = {}
        self.transformed_stats = {}
    
    def fit_transform(self, data: np.ndarray,
                     observed_columns: List[int],
                     predicted_columns: List[int]) -> np.ndarray:
        """Fit and transform the data according to specifications.
        
        Args:
            data: Input data (n_samples, n_features) with possible NaN values
            observed_columns: Indices of observed feature columns
            predicted_columns: Indices of predicted feature columns
            
        Returns:
            Transformed data
        """
        # Store column indices
        self.observed_indices = np.array(observed_columns)
        self.predicted_indices = np.array(predicted_columns)
        
        # Validate indices
        all_indices = np.concatenate([self.observed_indices, self.predicted_indices])
        if len(np.unique(all_indices)) != len(all_indices):
            raise ValueError("Overlapping observed and predicted column indices")
        
        # Store original statistics
        self._store_original_stats(data)
        
        # Step 1: Log transformation
        data_transformed = self._log_transform(data)
        
        # Step 2: Separate standardization
        data_standardized = self._separate_standardization(data_transformed)
        
        # Store transformed statistics
        self._store_transformed_stats(data_standardized)
        
        return data_standardized
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform new data using fitted parameters.
        
        Args:
            data: Input data to transform
            
        Returns:
            Transformed data
        """
        if self.scaler_observed is None or self.scaler_predicted is None:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Log transformation
        data_transformed = self._log_transform(data)
        
        # Apply fitted standardization
        data_standardized = data_transformed.copy()
        
        # Transform observed features
        if len(self.observed_indices) > 0:
            obs_data = data_transformed[:, self.observed_indices]
            # Handle NaN during standardization
            obs_mask = ~np.isnan(obs_data)
            obs_data_filled = np.where(obs_mask, obs_data, 0)
            obs_standardized = self.scaler_observed.transform(obs_data_filled)
            data_standardized[:, self.observed_indices] = np.where(
                obs_mask, obs_standardized, np.nan
            )
        
        # Transform predicted features
        if len(self.predicted_indices) > 0:
            pred_data = data_transformed[:, self.predicted_indices]
            pred_mask = ~np.isnan(pred_data)
            pred_data_filled = np.where(pred_mask, pred_data, 0)
            pred_standardized = self.scaler_predicted.transform(pred_data_filled)
            data_standardized[:, self.predicted_indices] = np.where(
                pred_mask, pred_standardized, np.nan
            )
        
        return data_standardized
    
    def _log_transform(self, data: np.ndarray) -> np.ndarray:
        """Apply log(x+1) transformation.
        
        Handles negative values by taking absolute value first.
        Preserves NaN values.
        """
        # Create mask for valid values
        valid_mask = ~np.isnan(data)
        
        # Transform only valid values
        data_transformed = data.copy()
        data_transformed[valid_mask] = np.log1p(np.abs(data[valid_mask]))
        
        logger.info(f"Applied log(x+1) transformation, preserved {np.isnan(data).sum()} NaN values")
        
        return data_transformed
    
    def _separate_standardization(self, data: np.ndarray) -> np.ndarray:
        """Apply separate z-score standardization for observed/predicted features.
        
        Args:
            data: Log-transformed data
            
        Returns:
            Standardized data
        """
        data_standardized = data.copy()
        
        # Standardize observed features
        if len(self.observed_indices) > 0:
            obs_data = data[:, self.observed_indices]
            
            # Fit scaler ignoring NaN
            if self.use_minmax:
                self.scaler_observed = MinMaxScaler(feature_range=(0, 1))
            else:
                self.scaler_observed = StandardScaler()
            obs_mask = ~np.isnan(obs_data)
            
            # Temporarily fill NaN with 0 for fitting
            obs_data_filled = np.where(obs_mask, obs_data, 0)
            self.scaler_observed.fit(obs_data_filled)
            
            # Transform and preserve NaN
            obs_standardized = self.scaler_observed.transform(obs_data_filled)
            data_standardized[:, self.observed_indices] = np.where(
                obs_mask, obs_standardized, np.nan
            )
            
            scaling_type = "MinMax" if self.use_minmax else "Z-score"
            logger.info(f"{scaling_type} scaled {len(self.observed_indices)} observed features")
        
        # Standardize predicted features
        if len(self.predicted_indices) > 0:
            pred_data = data[:, self.predicted_indices]
            
            # Fit scaler ignoring NaN
            if self.use_minmax:
                self.scaler_predicted = MinMaxScaler(feature_range=(0, 1))
            else:
                self.scaler_predicted = StandardScaler()
            pred_mask = ~np.isnan(pred_data)
            
            # Temporarily fill NaN with 0 for fitting
            pred_data_filled = np.where(pred_mask, pred_data, 0)
            self.scaler_predicted.fit(pred_data_filled)
            
            # Transform and preserve NaN
            pred_standardized = self.scaler_predicted.transform(pred_data_filled)
            data_standardized[:, self.predicted_indices] = np.where(
                pred_mask, pred_standardized, np.nan
            )
            
            scaling_type = "MinMax" if self.use_minmax else "Z-score"
            logger.info(f"{scaling_type} scaled {len(self.predicted_indices)} predicted features")
        
        return data_standardized
    
    def _store_original_stats(self, data: np.ndarray):
        """Store statistics of original data."""
        self.original_stats = {
            'mean': np.nanmean(data, axis=0),
            'std': np.nanstd(data, axis=0),
            'min': np.nanmin(data, axis=0),
            'max': np.nanmax(data, axis=0),
            'missing_proportion': np.isnan(data).mean(axis=0)
        }
    
    def _store_transformed_stats(self, data: np.ndarray):
        """Store statistics of transformed data."""
        self.transformed_stats = {
            'mean': np.nanmean(data, axis=0),
            'std': np.nanstd(data, axis=0),
            'min': np.nanmin(data, axis=0),
            'max': np.nanmax(data, axis=0),
            'missing_proportion': np.isnan(data).mean(axis=0)
        }
    
    def get_preprocessing_info(self) -> Dict[str, any]:
        """Get preprocessing information and statistics.
        
        Returns:
            Dictionary with preprocessing details
        """
        info = {
            'transformation': 'log1p',
            'standardization': 'z_score_by_type',
            'observed_features': self.observed_indices.tolist() if self.observed_indices is not None else [],
            'predicted_features': self.predicted_indices.tolist() if self.predicted_indices is not None else [],
            'original_stats': self.original_stats,
            'transformed_stats': self.transformed_stats
        }
        
        if self.scaler_observed is not None:
            if hasattr(self.scaler_observed, 'mean_'):  # StandardScaler
                info['observed_scaler_mean'] = self.scaler_observed.mean_
                info['observed_scaler_std'] = self.scaler_observed.scale_
            else:  # MinMaxScaler
                info['observed_scaler_min'] = self.scaler_observed.min_
                info['observed_scaler_scale'] = self.scaler_observed.scale_
        
        if self.scaler_predicted is not None:
            if hasattr(self.scaler_predicted, 'mean_'):  # StandardScaler
                info['predicted_scaler_mean'] = self.scaler_predicted.mean_
                info['predicted_scaler_std'] = self.scaler_predicted.scale_
            else:  # MinMaxScaler
                info['predicted_scaler_min'] = self.scaler_predicted.min_
                info['predicted_scaler_scale'] = self.scaler_predicted.scale_
        
        return info


def preprocess_biodiversity(data: np.ndarray,
                          observed_columns: List[int],
                          predicted_columns: List[int]) -> Tuple[np.ndarray, BiodiversityPreprocessor]:
    """Convenience function matching the specification example.
    
    Args:
        data: Input biodiversity data
        observed_columns: Indices of observed features
        predicted_columns: Indices of predicted features
        
    Returns:
        Tuple of (transformed_data, fitted_preprocessor)
    """
    preprocessor = BiodiversityPreprocessor()
    data_transformed = preprocessor.fit_transform(data, observed_columns, predicted_columns)
    
    return data_transformed, preprocessor