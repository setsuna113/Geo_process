"""Feature preprocessing utilities for biodiversity analysis."""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)


class FeaturePreprocessor:
    """Handle feature preprocessing tasks."""
    
    def handle_missing_values(self, 
                            features: np.ndarray,
                            feature_names: List[str],
                            strategy: str = 'keep') -> Tuple[np.ndarray, List[str]]:
        """Handle missing values in features.
        
        Args:
            features: Feature array with possible NaN values
            feature_names: List of feature names
            strategy: How to handle missing values:
                - 'keep': Keep NaN values (default)
                - 'drop_samples': Remove samples with any NaN
                - 'drop_features': Remove features with any NaN
                - 'mean': Impute with mean
                - 'median': Impute with median
                - 'zero': Replace with zero
                
        Returns:
            Tuple of (processed features, updated feature names)
        """
        if strategy == 'keep':
            return features, feature_names
            
        elif strategy == 'drop_samples':
            mask = ~np.any(np.isnan(features), axis=1)
            return features[mask], feature_names
            
        elif strategy == 'drop_features':
            mask = ~np.any(np.isnan(features), axis=0)
            return features[:, mask], [name for i, name in enumerate(feature_names) if mask[i]]
            
        elif strategy == 'mean':
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            return imputer.fit_transform(features), feature_names
            
        elif strategy == 'median':
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            return imputer.fit_transform(features), feature_names
            
        elif strategy == 'zero':
            return np.nan_to_num(features, nan=0.0), feature_names
            
        else:
            raise ValueError(f"Unknown missing value strategy: {strategy}")
    
    def remove_constant_features(self,
                               features: np.ndarray,
                               feature_names: List[str],
                               threshold: float = 1e-10) -> Tuple[np.ndarray, List[str]]:
        """Remove features with constant or near-constant values.
        
        Args:
            features: Feature array
            feature_names: List of feature names
            threshold: Variance threshold below which features are removed
            
        Returns:
            Tuple of (filtered features, filtered feature names)
        """
        # Calculate variance ignoring NaN
        variances = np.nanvar(features, axis=0)
        
        # Keep features with variance above threshold
        mask = variances > threshold
        
        n_removed = (~mask).sum()
        if n_removed > 0:
            logger.info(f"Removed {n_removed} constant features")
        
        return features[:, mask], [name for i, name in enumerate(feature_names) if mask[i]]
    
    def normalize_features(self,
                         features: np.ndarray,
                         method: str = 'standard',
                         feature_range: Tuple[float, float] = (0, 1)) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Normalize features using specified method.
        
        Args:
            features: Feature array
            method: Normalization method:
                - 'standard': Z-score standardization
                - 'minmax': Min-max scaling
                - 'robust': Robust scaling (median and IQR)
            feature_range: Range for minmax scaling
            
        Returns:
            Tuple of (normalized features, scaler info)
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler(feature_range=feature_range)
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Handle NaN during normalization
        mask = ~np.isnan(features)
        features_filled = np.where(mask, features, 0)
        
        # Fit and transform
        features_normalized = scaler.fit_transform(features_filled)
        
        # Restore NaN values
        features_normalized = np.where(mask, features_normalized, np.nan)
        
        # Store scaler info
        scaler_info = {
            'method': method,
            'scaler': scaler,
            'feature_range': feature_range if method == 'minmax' else None
        }
        
        return features_normalized, scaler_info


class ZeroInflationHandler:
    """Handle zero-inflated data transformations."""
    
    def transform_zero_inflated(self,
                              features: np.ndarray,
                              method: str = 'log1p',
                              epsilon: float = 1e-8) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Transform zero-inflated features.
        
        Args:
            features: Feature array with many zeros
            method: Transformation method:
                - 'log1p': log(x + 1) transformation
                - 'sqrt': Square root transformation
                - 'arcsinh': Inverse hyperbolic sine
                - 'binary': Convert to presence/absence
            epsilon: Small value to add before log for 'log' method
            
        Returns:
            Tuple of (transformed features, transform info)
        """
        if method == 'log1p':
            # Ensure non-negative values
            features_clean = np.maximum(features, 0)
            transformed = np.log1p(features_clean)
            
        elif method == 'sqrt':
            # Ensure non-negative values
            features_clean = np.maximum(features, 0)
            transformed = np.sqrt(features_clean)
            
        elif method == 'arcsinh':
            # Works with negative values too
            transformed = np.arcsinh(features)
            
        elif method == 'binary':
            # Convert to presence/absence
            transformed = (features > 0).astype(float)
            
        else:
            raise ValueError(f"Unknown zero inflation method: {method}")
        
        # Calculate transformation statistics
        transform_info = {
            'method': method,
            'epsilon': epsilon if method == 'log' else None,
            'zero_proportion_before': (features == 0).mean(),
            'zero_proportion_after': (transformed == 0).mean()
        }
        
        return transformed, transform_info
    
    def inverse_transform(self,
                         transformed: np.ndarray,
                         transform_info: Dict[str, Any]) -> np.ndarray:
        """Inverse transform zero-inflated features.
        
        Args:
            transformed: Transformed feature array
            transform_info: Information from forward transform
            
        Returns:
            Original scale features
        """
        method = transform_info['method']
        
        if method == 'log1p':
            return np.expm1(transformed)
            
        elif method == 'sqrt':
            return transformed ** 2
            
        elif method == 'arcsinh':
            return np.sinh(transformed)
            
        elif method == 'binary':
            # Cannot truly inverse binary transform
            logger.warning("Binary transform cannot be fully inverted")
            return transformed
            
        else:
            raise ValueError(f"Unknown zero inflation method: {method}")