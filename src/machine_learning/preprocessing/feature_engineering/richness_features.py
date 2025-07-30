"""Feature builder for biodiversity richness-based features."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Set, Optional, Union
import logging
from scipy import stats

from ....base.feature_builder import BaseFeatureBuilder
from ....core.registry import feature_builder

logger = logging.getLogger(__name__)


@feature_builder(
    "richness",
    required_columns={'plants_richness', 'terrestrial_richness'},
    description="Builds features from biodiversity richness data"
)
class RichnessFeatureBuilder(BaseFeatureBuilder):
    """
    Feature builder for richness-based features.
    
    Creates features from species richness data including:
    - Total richness
    - Richness ratios
    - Log-transformed features
    - Diversity metrics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize richness feature builder."""
        super().__init__(config)
        
        # Get feature engineering config
        self.feature_config = config.get('machine_learning', {}).get(
            'feature_engineering', {}
        ).get('richness_features', {}) if config else {}
        
        self.compute_ratios = self.feature_config.get('compute_ratios', True)
        self.log_transform = self.feature_config.get('log_transform', True)
        self.diversity_metrics = self.feature_config.get('diversity_metrics', ['shannon', 'simpson'])
        
        # Define feature names
        self._feature_names = self._define_feature_names()
        
    def _define_feature_names(self) -> List[str]:
        """Define all feature names that will be created."""
        features = [
            'total_richness',
            'plants_richness',
            'terrestrial_richness'
        ]
        
        if self.compute_ratios:
            features.extend([
                'plant_terrestrial_ratio',
                'terrestrial_plant_ratio',
                'plant_proportion',
                'terrestrial_proportion'
            ])
        
        if self.log_transform:
            features.extend([
                'log_total_richness',
                'log_plants_richness',
                'log_terrestrial_richness'
            ])
        
        if 'shannon' in self.diversity_metrics:
            features.append('shannon_diversity')
        
        if 'simpson' in self.diversity_metrics:
            features.append('simpson_diversity')
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names that will be created."""
        return self._feature_names.copy()
    
    def get_required_columns(self) -> Set[str]:
        """Get set of required input columns."""
        return {'plants_richness', 'terrestrial_richness'}
    
    def _fit(self, data: pd.DataFrame, **kwargs) -> None:
        """
        Fit the feature builder on training data.
        
        For richness features, we don't need to learn parameters,
        but we could compute statistics for normalization if needed.
        """
        # Compute basic statistics for logging
        self._fitted_params['plants_mean'] = data['plants_richness'].mean()
        self._fitted_params['plants_std'] = data['plants_richness'].std()
        self._fitted_params['terrestrial_mean'] = data['terrestrial_richness'].mean()
        self._fitted_params['terrestrial_std'] = data['terrestrial_richness'].std()
        
        logger.info(f"Fitted richness features on {len(data)} samples")
        logger.info(f"Plants richness: mean={self._fitted_params['plants_mean']:.2f}, "
                   f"std={self._fitted_params['plants_std']:.2f}")
        logger.info(f"Terrestrial richness: mean={self._fitted_params['terrestrial_mean']:.2f}, "
                   f"std={self._fitted_params['terrestrial_std']:.2f}")
    
    def _transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform data to create richness features.
        
        Args:
            data: Input DataFrame with richness columns
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame(index=data.index)
        
        # Basic richness features
        features['plants_richness'] = data['plants_richness'].fillna(0)
        features['terrestrial_richness'] = data['terrestrial_richness'].fillna(0)
        features['total_richness'] = features['plants_richness'] + features['terrestrial_richness']
        
        # Ratios and proportions
        if self.compute_ratios:
            # Avoid division by zero
            eps = 1e-8
            
            # Ratios
            features['plant_terrestrial_ratio'] = (
                features['plants_richness'] / (features['terrestrial_richness'] + eps)
            )
            features['terrestrial_plant_ratio'] = (
                features['terrestrial_richness'] / (features['plants_richness'] + eps)
            )
            
            # Proportions
            total_safe = features['total_richness'] + eps
            features['plant_proportion'] = features['plants_richness'] / total_safe
            features['terrestrial_proportion'] = features['terrestrial_richness'] / total_safe
            
            # Clip extreme ratios
            features['plant_terrestrial_ratio'] = features['plant_terrestrial_ratio'].clip(0, 100)
            features['terrestrial_plant_ratio'] = features['terrestrial_plant_ratio'].clip(0, 100)
        
        # Log transformations
        if self.log_transform:
            # Log(x + 1) to handle zeros
            features['log_total_richness'] = np.log1p(features['total_richness'])
            features['log_plants_richness'] = np.log1p(features['plants_richness'])
            features['log_terrestrial_richness'] = np.log1p(features['terrestrial_richness'])
        
        # Diversity metrics
        if 'shannon' in self.diversity_metrics:
            features['shannon_diversity'] = self._compute_shannon_diversity(
                features['plants_richness'], 
                features['terrestrial_richness']
            )
        
        if 'simpson' in self.diversity_metrics:
            features['simpson_diversity'] = self._compute_simpson_diversity(
                features['plants_richness'], 
                features['terrestrial_richness']
            )
        
        # Update metadata
        self._update_richness_metadata(features)
        
        return features
    
    def _compute_shannon_diversity(self, plants: pd.Series, terrestrial: pd.Series) -> pd.Series:
        """
        Compute Shannon diversity index.
        
        H = -sum(p_i * log(p_i))
        """
        total = plants + terrestrial
        
        # Avoid log(0)
        mask = total > 0
        diversity = pd.Series(0.0, index=plants.index)
        
        if mask.any():
            # Proportions
            p_plants = plants[mask] / total[mask]
            p_terrestrial = terrestrial[mask] / total[mask]
            
            # Shannon index
            h_plants = np.where(p_plants > 0, -p_plants * np.log(p_plants), 0)
            h_terrestrial = np.where(p_terrestrial > 0, -p_terrestrial * np.log(p_terrestrial), 0)
            
            diversity.loc[mask] = h_plants + h_terrestrial
        
        return diversity
    
    def _compute_simpson_diversity(self, plants: pd.Series, terrestrial: pd.Series) -> pd.Series:
        """
        Compute Simpson diversity index.
        
        D = 1 - sum(p_i^2)
        """
        total = plants + terrestrial
        
        # Avoid division by zero
        mask = total > 0
        diversity = pd.Series(0.0, index=plants.index)
        
        if mask.any():
            # Proportions
            p_plants = plants[mask] / total[mask]
            p_terrestrial = terrestrial[mask] / total[mask]
            
            # Simpson index
            diversity.loc[mask] = 1 - (p_plants**2 + p_terrestrial**2)
        
        return diversity
    
    def _update_richness_metadata(self, features: pd.DataFrame) -> None:
        """Update feature metadata with richness-specific information."""
        metadata_updates = {
            'total_richness': {
                'description': 'Total species richness (plants + terrestrial)',
                'category': 'richness',
                'transformation': 'sum'
            },
            'plants_richness': {
                'description': 'Plant species richness',
                'category': 'richness',
                'transformation': 'identity'
            },
            'terrestrial_richness': {
                'description': 'Terrestrial species richness',
                'category': 'richness',
                'transformation': 'identity'
            },
            'plant_terrestrial_ratio': {
                'description': 'Ratio of plant to terrestrial richness',
                'category': 'richness',
                'transformation': 'ratio'
            },
            'terrestrial_plant_ratio': {
                'description': 'Ratio of terrestrial to plant richness',
                'category': 'richness',
                'transformation': 'ratio'
            },
            'plant_proportion': {
                'description': 'Proportion of total richness from plants',
                'category': 'richness',
                'transformation': 'proportion'
            },
            'terrestrial_proportion': {
                'description': 'Proportion of total richness from terrestrial species',
                'category': 'richness',
                'transformation': 'proportion'
            },
            'log_total_richness': {
                'description': 'Log-transformed total richness',
                'category': 'richness',
                'transformation': 'log'
            },
            'log_plants_richness': {
                'description': 'Log-transformed plant richness',
                'category': 'richness',
                'transformation': 'log'
            },
            'log_terrestrial_richness': {
                'description': 'Log-transformed terrestrial richness',
                'category': 'richness',
                'transformation': 'log'
            },
            'shannon_diversity': {
                'description': 'Shannon diversity index',
                'category': 'richness',
                'transformation': 'diversity_index'
            },
            'simpson_diversity': {
                'description': 'Simpson diversity index',
                'category': 'richness',
                'transformation': 'diversity_index'
            }
        }
        
        # Update base metadata
        self._update_feature_metadata(features)
        
        # Add richness-specific metadata
        for feature, updates in metadata_updates.items():
            if feature in self.feature_metadata:
                self.feature_metadata[feature].update(updates)
    
    def get_feature_importance_prior(self) -> Optional[Dict[str, float]]:
        """
        Get prior importance for richness features based on domain knowledge.
        
        Returns:
            Dictionary of feature importance priors
        """
        # Domain knowledge suggests log-transformed features often work better
        importance = {
            'log_total_richness': 0.15,
            'log_plants_richness': 0.12,
            'log_terrestrial_richness': 0.12,
            'total_richness': 0.10,
            'plants_richness': 0.08,
            'terrestrial_richness': 0.08,
            'plant_terrestrial_ratio': 0.08,
            'terrestrial_plant_ratio': 0.07,
            'plant_proportion': 0.05,
            'terrestrial_proportion': 0.05,
            'shannon_diversity': 0.05,
            'simpson_diversity': 0.05
        }
        
        # Only return features that are actually being created
        return {k: v for k, v in importance.items() if k in self._feature_names}