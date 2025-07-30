"""Feature builder for spatial and geographic features."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Set, Optional, Union
import logging

from ....base.feature_builder import BaseFeatureBuilder
from ....core.registry import feature_builder

logger = logging.getLogger(__name__)


@feature_builder(
    "spatial", 
    required_columns={'latitude', 'longitude'},
    description="Builds spatial and geographic features"
)
class SpatialFeatureBuilder(BaseFeatureBuilder):
    """
    Feature builder for spatial features.
    
    Creates features from geographic coordinates including:
    - Distance to equator
    - Polynomial latitude features
    - Distance to tropics
    - Spatial binning
    - Grid cell area adjustments
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize spatial feature builder."""
        super().__init__(config)
        
        # Get feature engineering config
        self.feature_config = config.get('machine_learning', {}).get(
            'feature_engineering', {}
        ).get('spatial_features', {}) if config else {}
        
        self.polynomial_degree = self.feature_config.get('polynomial_degree', 2)
        self.include_interactions = self.feature_config.get('include_interactions', True)
        self.binning_strategy = self.feature_config.get('binning_strategy', 'quantile')
        self.n_bins = self.feature_config.get('n_bins', 10)
        
        # Tropical boundaries
        self.tropic_of_cancer = 23.43645  # degrees N
        self.tropic_of_capricorn = -23.43645  # degrees S
        
        # Define feature names
        self._feature_names = self._define_feature_names()
        
    def _define_feature_names(self) -> List[str]:
        """Define all feature names that will be created."""
        features = [
            'latitude',
            'longitude',
            'abs_latitude',
            'distance_to_equator',
            'distance_to_nearest_tropic',
            'is_tropical',
            'hemisphere',  # 1 for North, -1 for South
            'grid_cell_area_weight'
        ]
        
        # Polynomial features
        for degree in range(2, self.polynomial_degree + 1):
            features.append(f'latitude_power_{degree}')
            features.append(f'abs_latitude_power_{degree}')
        
        # Interactions
        if self.include_interactions:
            features.extend([
                'lat_lon_interaction',
                'lat_squared_lon',
                'lat_lon_squared'
            ])
        
        # Binned features
        features.extend([
            'latitude_bin',
            'longitude_bin',
            'spatial_bin'  # Combined lat/lon bin
        ])
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names that will be created."""
        return self._feature_names.copy()
    
    def get_required_columns(self) -> Set[str]:
        """Get set of required input columns."""
        return {'latitude', 'longitude'}
    
    def _fit(self, data: pd.DataFrame, **kwargs) -> None:
        """
        Fit the feature builder on training data.
        
        Learn binning boundaries from training data.
        """
        # Compute latitude bins
        if self.binning_strategy == 'quantile':
            self._fitted_params['lat_bins'] = pd.qcut(
                data['latitude'], 
                q=self.n_bins, 
                duplicates='drop', 
                retbins=True
            )[1]
        else:  # uniform
            self._fitted_params['lat_bins'] = pd.cut(
                data['latitude'], 
                bins=self.n_bins, 
                retbins=True
            )[1]
        
        # Compute longitude bins
        if self.binning_strategy == 'quantile':
            self._fitted_params['lon_bins'] = pd.qcut(
                data['longitude'], 
                q=self.n_bins, 
                duplicates='drop', 
                retbins=True
            )[1]
        else:  # uniform
            self._fitted_params['lon_bins'] = pd.cut(
                data['longitude'], 
                bins=self.n_bins, 
                retbins=True
            )[1]
        
        # Store data bounds
        self._fitted_params['lat_min'] = data['latitude'].min()
        self._fitted_params['lat_max'] = data['latitude'].max()
        self._fitted_params['lon_min'] = data['longitude'].min()
        self._fitted_params['lon_max'] = data['longitude'].max()
        
        logger.info(f"Fitted spatial features on {len(data)} samples")
        logger.info(f"Latitude range: [{self._fitted_params['lat_min']:.2f}, "
                   f"{self._fitted_params['lat_max']:.2f}]")
        logger.info(f"Longitude range: [{self._fitted_params['lon_min']:.2f}, "
                   f"{self._fitted_params['lon_max']:.2f}]")
    
    def _transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform data to create spatial features.
        
        Args:
            data: Input DataFrame with latitude/longitude columns
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame(index=data.index)
        
        # Basic coordinates
        features['latitude'] = data['latitude']
        features['longitude'] = data['longitude']
        features['abs_latitude'] = np.abs(data['latitude'])
        
        # Distance features
        features['distance_to_equator'] = np.abs(data['latitude'])
        features['distance_to_nearest_tropic'] = self._compute_distance_to_tropics(data['latitude'])
        
        # Tropical indicator
        features['is_tropical'] = (
            (data['latitude'] >= self.tropic_of_capricorn) & 
            (data['latitude'] <= self.tropic_of_cancer)
        ).astype(int)
        
        # Hemisphere indicator
        features['hemisphere'] = np.sign(data['latitude'])
        features.loc[features['hemisphere'] == 0, 'hemisphere'] = 1  # Equator counts as North
        
        # Grid cell area weight (cosine of latitude)
        features['grid_cell_area_weight'] = np.cos(np.radians(data['latitude']))
        
        # Polynomial features
        for degree in range(2, self.polynomial_degree + 1):
            features[f'latitude_power_{degree}'] = data['latitude'] ** degree
            features[f'abs_latitude_power_{degree}'] = features['abs_latitude'] ** degree
        
        # Interaction features
        if self.include_interactions:
            features['lat_lon_interaction'] = data['latitude'] * data['longitude']
            features['lat_squared_lon'] = (data['latitude'] ** 2) * data['longitude']
            features['lat_lon_squared'] = data['latitude'] * (data['longitude'] ** 2)
        
        # Binned features
        if self.is_fitted and 'lat_bins' in self._fitted_params:
            features['latitude_bin'] = pd.cut(
                data['latitude'], 
                bins=self._fitted_params['lat_bins'],
                labels=False,
                include_lowest=True
            ).fillna(0).astype(int)
            
            features['longitude_bin'] = pd.cut(
                data['longitude'], 
                bins=self._fitted_params['lon_bins'],
                labels=False,
                include_lowest=True
            ).fillna(0).astype(int)
            
            # Combined spatial bin
            features['spatial_bin'] = (
                features['latitude_bin'] * self.n_bins + features['longitude_bin']
            )
        else:
            # If not fitted, use simple binning
            features['latitude_bin'] = pd.cut(
                data['latitude'], 
                bins=self.n_bins,
                labels=False
            ).fillna(0).astype(int)
            
            features['longitude_bin'] = pd.cut(
                data['longitude'], 
                bins=self.n_bins,
                labels=False
            ).fillna(0).astype(int)
            
            features['spatial_bin'] = (
                features['latitude_bin'] * self.n_bins + features['longitude_bin']
            )
        
        # Update metadata
        self._update_spatial_metadata(features)
        
        return features
    
    def _compute_distance_to_tropics(self, latitude: pd.Series) -> pd.Series:
        """Compute distance to nearest tropic line."""
        dist_to_cancer = np.abs(latitude - self.tropic_of_cancer)
        dist_to_capricorn = np.abs(latitude - self.tropic_of_capricorn)
        
        return np.minimum(dist_to_cancer, dist_to_capricorn)
    
    def _update_spatial_metadata(self, features: pd.DataFrame) -> None:
        """Update feature metadata with spatial-specific information."""
        metadata_updates = {
            'latitude': {
                'description': 'Geographic latitude',
                'category': 'spatial',
                'transformation': 'identity'
            },
            'longitude': {
                'description': 'Geographic longitude',
                'category': 'spatial',
                'transformation': 'identity'
            },
            'abs_latitude': {
                'description': 'Absolute value of latitude',
                'category': 'spatial',
                'transformation': 'absolute'
            },
            'distance_to_equator': {
                'description': 'Distance to equator in degrees',
                'category': 'spatial',
                'transformation': 'distance'
            },
            'distance_to_nearest_tropic': {
                'description': 'Distance to nearest tropic line',
                'category': 'spatial',
                'transformation': 'distance'
            },
            'is_tropical': {
                'description': 'Binary indicator for tropical region',
                'category': 'spatial',
                'transformation': 'indicator'
            },
            'hemisphere': {
                'description': 'Hemisphere indicator (1=North, -1=South)',
                'category': 'spatial',
                'transformation': 'indicator'
            },
            'grid_cell_area_weight': {
                'description': 'Area weight based on latitude (cosine)',
                'category': 'spatial',
                'transformation': 'weight'
            },
            'lat_lon_interaction': {
                'description': 'Latitude × Longitude interaction',
                'category': 'spatial',
                'transformation': 'interaction'
            },
            'latitude_bin': {
                'description': 'Binned latitude category',
                'category': 'spatial',
                'transformation': 'binning'
            },
            'longitude_bin': {
                'description': 'Binned longitude category',
                'category': 'spatial',
                'transformation': 'binning'
            },
            'spatial_bin': {
                'description': 'Combined spatial bin (lat × lon)',
                'category': 'spatial',
                'transformation': 'binning'
            }
        }
        
        # Add polynomial feature metadata
        for degree in range(2, self.polynomial_degree + 1):
            metadata_updates[f'latitude_power_{degree}'] = {
                'description': f'Latitude raised to power {degree}',
                'category': 'spatial',
                'transformation': 'polynomial'
            }
            metadata_updates[f'abs_latitude_power_{degree}'] = {
                'description': f'Absolute latitude raised to power {degree}',
                'category': 'spatial',
                'transformation': 'polynomial'
            }
        
        # Update base metadata
        self._update_feature_metadata(features)
        
        # Add spatial-specific metadata
        for feature, updates in metadata_updates.items():
            if feature in self.feature_metadata:
                self.feature_metadata[feature].update(updates)
    
    def get_feature_importance_prior(self) -> Optional[Dict[str, float]]:
        """
        Get prior importance for spatial features based on domain knowledge.
        
        Returns:
            Dictionary of feature importance priors
        """
        # Domain knowledge: latitude is typically most important
        importance = {
            'abs_latitude': 0.15,
            'latitude': 0.12,
            'distance_to_equator': 0.10,
            'latitude_power_2': 0.10,
            'abs_latitude_power_2': 0.08,
            'is_tropical': 0.08,
            'distance_to_nearest_tropic': 0.07,
            'grid_cell_area_weight': 0.06,
            'hemisphere': 0.05,
            'longitude': 0.05,
            'lat_lon_interaction': 0.04,
            'latitude_bin': 0.03,
            'longitude_bin': 0.03,
            'spatial_bin': 0.04
        }
        
        # Add higher degree polynomials with lower importance
        for degree in range(3, self.polynomial_degree + 1):
            importance[f'latitude_power_{degree}'] = 0.02
            importance[f'abs_latitude_power_{degree}'] = 0.02
        
        # Only return features that are actually being created
        return {k: v for k, v in importance.items() if k in self._feature_names}