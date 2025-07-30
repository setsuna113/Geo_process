"""Spatial-aware KNN imputation for biodiversity data."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, Union
from sklearn.impute import KNNImputer as SklearnKNNImputer
from sklearn.preprocessing import StandardScaler
import logging

from ....core.registry import imputation_strategy
from ....abstractions.types.ml_types import ImputationStrategy, ImputationResult

logger = logging.getLogger(__name__)


@imputation_strategy(
    strategy_name="knn",
    handles_spatial=True,
    description="K-Nearest Neighbors imputation with spatial awareness"
)
class SpatialKNNImputer:
    """
    KNN imputation that considers spatial proximity.
    
    Uses geographic distance as part of the neighbor selection process,
    giving preference to spatially close observations when imputing values.
    """
    
    def __init__(self, 
                 n_neighbors: int = 5,
                 spatial_weight: float = 0.5,
                 use_spatial_distance: bool = True,
                 missing_values: Union[float, str] = np.nan,
                 **kwargs):
        """
        Initialize spatial KNN imputer.
        
        Args:
            n_neighbors: Number of neighbors to use
            spatial_weight: Weight for spatial distance (0-1)
            use_spatial_distance: Whether to use spatial distance
            missing_values: Placeholder for missing values
            **kwargs: Additional arguments for sklearn KNNImputer
        """
        self.n_neighbors = n_neighbors
        self.spatial_weight = spatial_weight
        self.use_spatial_distance = use_spatial_distance
        self.missing_values = missing_values
        
        # Feature scaler for distance computation
        self.feature_scaler = StandardScaler()
        self.spatial_scaler = StandardScaler()
        
        # Sklearn imputer for the actual imputation
        self.imputer = SklearnKNNImputer(
            n_neighbors=n_neighbors,
            missing_values=missing_values,
            **kwargs
        )
        
        # Tracking
        self.is_fitted = False
        self.feature_names = None
        self.spatial_columns = None
        self.imputation_result = None
    
    def fit(self, 
            X: pd.DataFrame,
            spatial_columns: Optional[Tuple[str, str]] = ('latitude', 'longitude')) -> 'SpatialKNNImputer':
        """
        Fit the imputer on training data.
        
        Args:
            X: DataFrame with features to impute
            spatial_columns: Names of latitude/longitude columns
            
        Returns:
            Self for method chaining
        """
        self.feature_names = list(X.columns)
        self.spatial_columns = spatial_columns
        
        # Check for spatial columns
        if self.use_spatial_distance and spatial_columns:
            if not all(col in X.columns for col in spatial_columns):
                logger.warning(f"Spatial columns {spatial_columns} not found. "
                             f"Falling back to standard KNN imputation.")
                self.use_spatial_distance = False
        
        # Prepare data
        if self.use_spatial_distance and spatial_columns:
            # Separate spatial and feature data
            spatial_data = X[list(spatial_columns)]
            feature_data = X.drop(columns=list(spatial_columns))
            
            # Fit scalers
            self.spatial_scaler.fit(spatial_data)
            if len(feature_data.columns) > 0:
                self.feature_scaler.fit(feature_data)
            
            # Create combined scaled data for imputation
            spatial_scaled = self.spatial_scaler.transform(spatial_data)
            
            if len(feature_data.columns) > 0:
                feature_scaled = self.feature_scaler.transform(feature_data)
                # Weight spatial features more heavily
                spatial_weighted = spatial_scaled * self.spatial_weight
                feature_weighted = feature_scaled * (1 - self.spatial_weight)
                combined_data = np.hstack([feature_weighted, spatial_weighted])
            else:
                combined_data = spatial_scaled
            
            # Fit imputer on combined data
            self.imputer.fit(combined_data)
        else:
            # Standard KNN without spatial weighting
            self.imputer.fit(X)
        
        self.is_fitted = True
        
        # Log fitting information
        missing_counts = X.isna().sum()
        logger.info(f"Fitted SpatialKNNImputer with {self.n_neighbors} neighbors")
        logger.info(f"Missing values per column: {missing_counts[missing_counts > 0].to_dict()}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in X.
        
        Args:
            X: DataFrame with missing values
            
        Returns:
            DataFrame with imputed values
        """
        if not self.is_fitted:
            raise ValueError("Imputer must be fitted before transform")
        
        # Store original index and columns
        original_index = X.index
        original_columns = X.columns
        
        # Track what was imputed
        missing_mask = X.isna()
        missing_counts_before = missing_mask.sum()
        
        # Prepare data
        if self.use_spatial_distance and self.spatial_columns:
            # Separate spatial and feature data
            spatial_data = X[list(self.spatial_columns)]
            feature_data = X.drop(columns=list(self.spatial_columns))
            
            # Scale data
            spatial_scaled = self.spatial_scaler.transform(spatial_data)
            
            if len(feature_data.columns) > 0:
                feature_scaled = self.feature_scaler.transform(feature_data)
                # Weight spatial features
                spatial_weighted = spatial_scaled * self.spatial_weight
                feature_weighted = feature_scaled * (1 - self.spatial_weight)
                combined_data = np.hstack([feature_weighted, spatial_weighted])
                
                # Impute
                imputed_combined = self.imputer.transform(combined_data)
                
                # Separate and inverse transform
                n_features = feature_scaled.shape[1]
                imputed_features = imputed_combined[:, :n_features] / (1 - self.spatial_weight)
                imputed_features = self.feature_scaler.inverse_transform(imputed_features)
                
                # Reconstruct DataFrame
                result_df = pd.DataFrame(
                    imputed_features,
                    index=original_index,
                    columns=feature_data.columns
                )
                
                # Add back spatial columns (don't impute these)
                for col in self.spatial_columns:
                    result_df[col] = X[col]
                
                # Reorder columns to match original
                result_df = result_df[original_columns]
            else:
                # Only spatial data
                imputed_data = self.imputer.transform(spatial_scaled)
                result_df = pd.DataFrame(
                    imputed_data,
                    index=original_index,
                    columns=original_columns
                )
        else:
            # Standard KNN imputation
            imputed_data = self.imputer.transform(X)
            result_df = pd.DataFrame(
                imputed_data,
                index=original_index,
                columns=original_columns
            )
        
        # Calculate imputation statistics
        missing_counts_after = result_df.isna().sum()
        
        # Create imputation result
        self.imputation_result = ImputationResult(
            strategy=ImputationStrategy.KNN,
            imputed_columns=list(missing_counts_before[missing_counts_before > 0].index),
            imputation_values={},  # KNN doesn't use fixed values
            missing_before={col: float(count/len(X)) for col, count in missing_counts_before.items()},
            missing_after={col: float(count/len(result_df)) for col, count in missing_counts_after.items()}
        )
        
        # Log imputation summary
        n_imputed = (missing_counts_before - missing_counts_after).sum()
        logger.info(f"Imputed {n_imputed} missing values using KNN")
        
        return result_df
    
    def fit_transform(self, 
                     X: pd.DataFrame,
                     spatial_columns: Optional[Tuple[str, str]] = ('latitude', 'longitude')) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X: DataFrame with missing values
            spatial_columns: Names of latitude/longitude columns
            
        Returns:
            DataFrame with imputed values
        """
        return self.fit(X, spatial_columns).transform(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get importance of features for imputation.
        
        For KNN, this could be based on how often each feature
        had non-missing values in the neighbors used for imputation.
        
        Returns:
            Feature importance scores (if available)
        """
        # KNN doesn't directly provide feature importance
        # Could be extended to track which features were most useful
        return None
    
    def get_params(self) -> Dict[str, Any]:
        """Get imputer parameters."""
        return {
            'n_neighbors': self.n_neighbors,
            'spatial_weight': self.spatial_weight,
            'use_spatial_distance': self.use_spatial_distance,
            'missing_values': self.missing_values,
            'is_fitted': self.is_fitted
        }
    
    def set_params(self, **params) -> 'SpatialKNNImputer':
        """Set imputer parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Update sklearn imputer if needed
        if 'n_neighbors' in params:
            self.imputer.n_neighbors = params['n_neighbors']
        
        return self