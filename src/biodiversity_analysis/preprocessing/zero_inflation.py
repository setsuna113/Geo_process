"""Zero-inflation and compositional data preprocessing for biodiversity data.

This module handles the extreme sparsity (95% zeros) in biodiversity data through:
- Zero-inflated transformations
- Compositional data analysis (CLR, ILR)
- Variance stabilization
- Adaptive pseudo-counts
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.special import logit, expit
import logging

logger = logging.getLogger(__name__)


class ZeroInflationPreprocessor:
    """Handle zero-inflated biodiversity data."""
    
    def __init__(self,
                 structural_zero_threshold: float = 0.95,
                 pseudo_count_method: str = 'adaptive',
                 min_pseudo_count: float = 0.001,
                 compositional_transform: str = 'clr'):
        """Initialize zero-inflation preprocessor.
        
        Args:
            structural_zero_threshold: Proportion of zeros to consider structural
            pseudo_count_method: 'fixed', 'adaptive', or 'gmean'
            min_pseudo_count: Minimum pseudo-count value
            compositional_transform: Transform method:
                - 'clr': Centered log-ratio
                - 'ilr': Isometric log-ratio  
                - 'alr': Additive log-ratio
                - 'hurdle': Two-stage hurdle model
                - 'log1p': Simple log(x+1)
                - 'sqrt': Square root
                - 'arcsinh': Inverse hyperbolic sine
                - 'binary': Presence/absence
        """
        self.structural_zero_threshold = structural_zero_threshold
        self.pseudo_count_method = pseudo_count_method
        self.min_pseudo_count = min_pseudo_count
        self.compositional_transform = compositional_transform
        self._fitted = False
        
    def fit(self, data: np.ndarray, feature_names: List[str] = None):
        """Fit preprocessor to data.
        
        Args:
            data: Feature matrix (n_samples, n_features)
            feature_names: Names of features
        """
        self.n_features_ = data.shape[1]
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(self.n_features_)]
        self.feature_names_ = feature_names
        
        # Analyze zero patterns
        self.zero_proportions_ = np.zeros(self.n_features_)
        self.structural_zeros_ = np.zeros(self.n_features_, dtype=bool)
        
        for i in range(self.n_features_):
            valid_mask = ~np.isnan(data[:, i])
            if np.any(valid_mask):
                zero_prop = np.mean(data[valid_mask, i] == 0)
                self.zero_proportions_[i] = zero_prop
                self.structural_zeros_[i] = zero_prop >= self.structural_zero_threshold
            else:
                self.zero_proportions_[i] = 1.0
                self.structural_zeros_[i] = True
        
        # Calculate adaptive pseudo-counts
        if self.pseudo_count_method == 'adaptive':
            self._calculate_adaptive_pseudocounts(data)
        elif self.pseudo_count_method == 'gmean':
            self._calculate_gmean_pseudocounts(data)
        else:
            self.pseudo_counts_ = np.full(self.n_features_, self.min_pseudo_count)
            
        self._fitted = True
        logger.info(f"Fitted zero-inflation preprocessor. Structural zeros: {self.structural_zeros_.sum()}/{self.n_features_}")
        
    def transform(self, data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Transform data handling zero-inflation.
        
        Returns:
            Transformed data and feature names
        """
        if not self._fitted:
            raise ValueError("Preprocessor must be fitted before transform")
            
        # Create output array
        n_samples = data.shape[0]
        transformed_features = []
        feature_names = []
        
        # Process each feature type
        for i in range(self.n_features_):
            if self.structural_zeros_[i]:
                # Feature is structurally zero - create presence/absence
                logger.debug(f"Feature {self.feature_names_[i]} is structurally zero")
                presence = (data[:, i] > 0).astype(float)
                presence[np.isnan(data[:, i])] = np.nan
                transformed_features.append(presence)
                feature_names.append(f'{self.feature_names_[i]}_presence')
            else:
                # Apply transformations
                transformed = self._transform_feature(data[:, i], i)
                transformed_features.append(transformed)
                feature_names.append(f'{self.feature_names_[i]}_transformed')
                
                # Add zero indicator for hurdle model
                zero_indicator = (data[:, i] == 0).astype(float)
                zero_indicator[np.isnan(data[:, i])] = np.nan
                transformed_features.append(zero_indicator)
                feature_names.append(f'{self.feature_names_[i]}_is_zero')
        
        # Apply compositional transformation if requested
        if self.compositional_transform and not all(self.structural_zeros_):
            comp_features, comp_names = self._apply_compositional_transform(data)
            transformed_features.extend(comp_features)
            feature_names.extend(comp_names)
        
        # Stack all features
        result = np.column_stack(transformed_features)
        
        logger.info(f"Transformed {data.shape[1]} features to {result.shape[1]} features")
        return result, feature_names
    
    def fit_transform(self, data: np.ndarray, feature_names: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        """Fit and transform in one step."""
        self.fit(data, feature_names)
        return self.transform(data)
    
    def _transform_feature(self, x: np.ndarray, feature_idx: int) -> np.ndarray:
        """Transform a single feature handling zeros."""
        # Get valid (non-NaN) mask
        valid_mask = ~np.isnan(x)
        
        # Initialize output
        result = np.full_like(x, np.nan)
        
        if not np.any(valid_mask):
            return result
            
        # Add pseudo-count to non-zero values
        x_valid = x[valid_mask].copy()
        pseudo = self.pseudo_counts_[feature_idx]
        
        # For zero values, use half the pseudo-count
        zero_mask = x_valid == 0
        x_valid[zero_mask] = pseudo / 2
        x_valid[~zero_mask] += pseudo
        
        # Apply variance-stabilizing transformation
        # Use log for count data
        transformed = np.log1p(x_valid)
        
        # Additional Box-Cox if highly skewed
        if stats.skew(transformed) > 2:
            try:
                transformed, _ = stats.boxcox(x_valid + 1)
            except:
                pass  # Keep log transform if Box-Cox fails
                
        result[valid_mask] = transformed
        return result
    
    def _calculate_adaptive_pseudocounts(self, data: np.ndarray):
        """Calculate adaptive pseudo-counts based on data characteristics."""
        self.pseudo_counts_ = np.zeros(self.n_features_)
        
        for i in range(self.n_features_):
            valid_mask = ~np.isnan(data[:, i])
            if not np.any(valid_mask):
                self.pseudo_counts_[i] = self.min_pseudo_count
                continue
                
            values = data[valid_mask, i]
            non_zero_values = values[values > 0]
            
            if len(non_zero_values) > 0:
                # Use fraction of minimum non-zero value
                min_nonzero = np.min(non_zero_values)
                self.pseudo_counts_[i] = max(min_nonzero * 0.01, self.min_pseudo_count)
            else:
                self.pseudo_counts_[i] = self.min_pseudo_count
                
    def _calculate_gmean_pseudocounts(self, data: np.ndarray):
        """Calculate pseudo-counts based on geometric mean."""
        self.pseudo_counts_ = np.zeros(self.n_features_)
        
        for i in range(self.n_features_):
            valid_mask = ~np.isnan(data[:, i])
            if not np.any(valid_mask):
                self.pseudo_counts_[i] = self.min_pseudo_count
                continue
                
            values = data[valid_mask, i]
            non_zero_values = values[values > 0]
            
            if len(non_zero_values) > 0:
                # Geometric mean of non-zero values
                gmean = stats.gmean(non_zero_values)
                self.pseudo_counts_[i] = max(gmean * 0.001, self.min_pseudo_count)
            else:
                self.pseudo_counts_[i] = self.min_pseudo_count
    
    def _apply_compositional_transform(self, data: np.ndarray) -> Tuple[List[np.ndarray], List[str]]:
        """Apply compositional data transformation."""
        # Only use non-structural features
        comp_indices = np.where(~self.structural_zeros_)[0]
        
        if len(comp_indices) < 2:
            return [], []
            
        # Get compositional subset
        comp_data = data[:, comp_indices].copy()
        
        # Add pseudo-counts
        for i, idx in enumerate(comp_indices):
            valid_mask = ~np.isnan(comp_data[:, i])
            comp_data[valid_mask, i] += self.pseudo_counts_[idx]
        
        # Apply CLR transform
        if self.compositional_transform == 'clr':
            transformed, names = self._clr_transform(comp_data, comp_indices)
        elif self.compositional_transform == 'ilr':
            transformed, names = self._ilr_transform(comp_data, comp_indices)
        elif self.compositional_transform == 'hurdle':
            transformed, names = self._hurdle_transform(data, comp_indices)
        elif self.compositional_transform == 'alr':
            transformed, names = self._alr_transform(comp_data, comp_indices)
        # Simple transformations
        elif self.compositional_transform == 'log1p':
            transformed, names = self._log1p_transform(data, comp_indices)
        elif self.compositional_transform == 'sqrt':
            transformed, names = self._sqrt_transform(data, comp_indices)
        elif self.compositional_transform == 'arcsinh':
            transformed, names = self._arcsinh_transform(data, comp_indices)
        elif self.compositional_transform == 'binary':
            transformed, names = self._binary_transform(data, comp_indices)
        else:
            raise ValueError(f"Unknown transform: {self.compositional_transform}")
            
        return transformed, names
    
    def _clr_transform(self, data: np.ndarray, indices: np.ndarray) -> Tuple[List[np.ndarray], List[str]]:
        """Centered log-ratio transformation."""
        n_samples = data.shape[0]
        features = []
        names = []
        
        # Calculate geometric mean for each sample
        gmeans = np.zeros(n_samples)
        for i in range(n_samples):
            valid_mask = ~np.isnan(data[i, :])
            if np.any(valid_mask):
                gmeans[i] = stats.gmean(data[i, valid_mask])
            else:
                gmeans[i] = np.nan
        
        # CLR transform each component
        for j, idx in enumerate(indices):
            clr = np.full(n_samples, np.nan)
            for i in range(n_samples):
                if not np.isnan(data[i, j]) and not np.isnan(gmeans[i]) and gmeans[i] > 0:
                    clr[i] = np.log(data[i, j] / gmeans[i])
            
            features.append(clr)
            names.append(f'{self.feature_names_[idx]}_clr')
            
        return features, names
    
    def _ilr_transform(self, data: np.ndarray, indices: np.ndarray) -> Tuple[List[np.ndarray], List[str]]:
        """Isometric log-ratio transformation."""
        n_comp = len(indices)
        n_samples = data.shape[0]
        
        if n_comp < 2:
            return [], []
        
        # Create orthonormal basis (Helmert matrix)
        basis = np.zeros((n_comp, n_comp - 1))
        for j in range(n_comp - 1):
            basis[:j+1, j] = 1.0 / np.sqrt((j+1) * (j+2))
            basis[j+1, j] = -(j+1) / np.sqrt((j+1) * (j+2))
        
        # Apply ILR transform
        features = []
        names = []
        
        for j in range(n_comp - 1):
            ilr_j = np.full(n_samples, np.nan)
            
            for i in range(n_samples):
                valid_mask = ~np.isnan(data[i, :])
                if np.all(valid_mask):
                    log_data = np.log(data[i, :])
                    ilr_j[i] = np.dot(basis[:, j], log_data)
            
            features.append(ilr_j)
            names.append(f'ilr_component_{j+1}')
            
        return features, names
    
    def _alr_transform(self, data: np.ndarray, indices: np.ndarray) -> Tuple[List[np.ndarray], List[str]]:
        """Additive log-ratio transformation."""
        # Use last component as reference
        ref_idx = -1
        features = []
        names = []
        
        for j in range(len(indices) - 1):
            alr = np.log(data[:, j] / data[:, ref_idx])
            features.append(alr)
            names.append(f'{self.feature_names_[indices[j]]}_alr')
            
        return features, names
    
    def _hurdle_transform(self, data: np.ndarray, indices: np.ndarray) -> Tuple[List[np.ndarray], List[str]]:
        """Hurdle model transformation - two-stage approach for zero-inflated data.
        
        Stage 1: Binary presence/absence (0/1)
        Stage 2: Log-transformed abundance conditional on presence
        
        This separates structural zeros from true abundance patterns.
        """
        features = []
        names = []
        
        for idx in indices:
            feature_data = data[:, idx]
            
            # Stage 1: Binary presence/absence
            presence = (~np.isnan(feature_data) & (feature_data > 0)).astype(float)
            features.append(presence)
            names.append(f'{self.feature_names_[idx]}_presence')
            
            # Stage 2: Log abundance given presence
            # Use log1p to handle cases where values are exactly 1
            abundance = np.full_like(feature_data, np.nan, dtype=float)
            present_mask = presence == 1
            if np.any(present_mask):
                abundance[present_mask] = np.log1p(feature_data[present_mask])
            
            features.append(abundance)
            names.append(f'{self.feature_names_[idx]}_log_abundance')
            
        return features, names
    
    def _log1p_transform(self, data: np.ndarray, indices: np.ndarray) -> Tuple[List[np.ndarray], List[str]]:
        """Simple log(x+1) transformation."""
        features = []
        names = []
        
        for idx in indices:
            # Ensure non-negative values
            feature_data = np.maximum(data[:, idx], 0)
            transformed = np.log1p(feature_data)
            features.append(transformed)
            names.append(f'{self.feature_names_[idx]}_log1p')
            
        return features, names
    
    def _sqrt_transform(self, data: np.ndarray, indices: np.ndarray) -> Tuple[List[np.ndarray], List[str]]:
        """Square root transformation."""
        features = []
        names = []
        
        for idx in indices:
            # Ensure non-negative values
            feature_data = np.maximum(data[:, idx], 0)
            transformed = np.sqrt(feature_data)
            features.append(transformed)
            names.append(f'{self.feature_names_[idx]}_sqrt')
            
        return features, names
    
    def _arcsinh_transform(self, data: np.ndarray, indices: np.ndarray) -> Tuple[List[np.ndarray], List[str]]:
        """Inverse hyperbolic sine transformation (works with negative values)."""
        features = []
        names = []
        
        for idx in indices:
            transformed = np.arcsinh(data[:, idx])
            features.append(transformed)
            names.append(f'{self.feature_names_[idx]}_arcsinh')
            
        return features, names
    
    def _binary_transform(self, data: np.ndarray, indices: np.ndarray) -> Tuple[List[np.ndarray], List[str]]:
        """Convert to binary presence/absence."""
        features = []
        names = []
        
        for idx in indices:
            binary = (data[:, idx] > 0).astype(float)
            features.append(binary)
            names.append(f'{self.feature_names_[idx]}_binary')
            
        return features, names
    
    def get_transform_summary(self) -> pd.DataFrame:
        """Get summary of transformation applied to each feature."""
        if not self._fitted:
            raise ValueError("Preprocessor must be fitted first")
            
        summary = pd.DataFrame({
            'feature': self.feature_names_,
            'zero_proportion': self.zero_proportions_,
            'structural_zero': self.structural_zeros_,
            'pseudo_count': self.pseudo_counts_,
            'transform_type': ['presence' if sz else 'hurdle' 
                               for sz in self.structural_zeros_]
        })
        
        return summary