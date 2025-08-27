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
    required_columns={'plants_richness', 'animal_richness'},
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
            'animal_richness'
        ]
        
        if self.compute_ratios:
            features.extend([
                'plant_animal_ratio',
                'animal_plant_ratio',
                'plant_proportion',
                'animal_proportion'
            ])
        
        if self.log_transform:
            features.extend([
                'log_total_richness',
                'log_plants_richness',
                'log_animal_richness'
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
        return {'plants_richness', 'animal_richness'}
    
    def _fit(self, data: pd.DataFrame, **kwargs) -> None:
        """
        Fit the feature builder on training data.
        
        For richness features, we don't need to learn parameters,
        but we could compute statistics for normalization if needed.
        """
        # Compute basic statistics for logging
        self._fitted_params['plants_mean'] = data['plants_richness'].mean()
        self._fitted_params['plants_std'] = data['plants_richness'].std()
        self._fitted_params['animal_mean'] = data['animal_richness'].mean()
        self._fitted_params['animal_std'] = data['animal_richness'].std()
        
        logger.info(f"Fitted richness features on {len(data)} samples")
        logger.info(f"Plants richness: mean={self._fitted_params['plants_mean']:.2f}, "
                   f"std={self._fitted_params['plants_std']:.2f}")
        logger.info(f"Terrestrial richness: mean={self._fitted_params['animal_mean']:.2f}, "
                   f"std={self._fitted_params['animal_std']:.2f}")
    
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
        features['animal_richness'] = data['animal_richness'].fillna(0)
        features['total_richness'] = features['plants_richness'] + features['animal_richness']
        
        # Ratios and proportions
        if self.compute_ratios:
            # Avoid division by zero
            eps = 1e-8
            
            # Ratios
            features['plant_animal_ratio'] = (
                features['plants_richness'] / (features['animal_richness'] + eps)
            )
            features['animal_plant_ratio'] = (
                features['animal_richness'] / (features['plants_richness'] + eps)
            )
            
            # Proportions
            total_safe = features['total_richness'] + eps
            features['plant_proportion'] = features['plants_richness'] / total_safe
            features['animal_proportion'] = features['animal_richness'] / total_safe
            
            # Clip extreme ratios
            features['plant_animal_ratio'] = features['plant_animal_ratio'].clip(0, 100)
            features['animal_plant_ratio'] = features['animal_plant_ratio'].clip(0, 100)
        
        # Log transformations
        if self.log_transform:
            # Log(x + 1) to handle zeros
            features['log_total_richness'] = np.log1p(features['total_richness'])
            features['log_plants_richness'] = np.log1p(features['plants_richness'])
            features['log_animal_richness'] = np.log1p(features['animal_richness'])
        
        # Diversity metrics
        if 'shannon' in self.diversity_metrics:
            features['shannon_diversity'] = self._compute_shannon_diversity(
                features['plants_richness'], 
                features['animal_richness']
            )
        
        if 'simpson' in self.diversity_metrics:
            features['simpson_diversity'] = self._compute_simpson_diversity(
                features['plants_richness'], 
                features['animal_richness']
            )
        
        # Update metadata
        self._update_richness_metadata(features)
        
        return features
    
    def _compute_shannon_diversity(self, plants: pd.Series, animal: pd.Series) -> pd.Series:
        """
        Compute Shannon diversity index.
        
        H = -sum(p_i * log(p_i))
        """
        total = plants + animal
        
        # Avoid log(0)
        mask = total > 0
        diversity = pd.Series(0.0, index=plants.index)
        
        if mask.any():
            # Proportions
            p_plants = plants[mask] / total[mask]
            p_animal = animal[mask] / total[mask]
            
            # Shannon index
            h_plants = np.where(p_plants > 0, -p_plants * np.log(p_plants), 0)
            h_animal = np.where(p_animal > 0, -p_animal * np.log(p_animal), 0)
            
            diversity.loc[mask] = h_plants + h_animal
        
        return diversity
    
    def _compute_simpson_diversity(self, plants: pd.Series, animal: pd.Series) -> pd.Series:
        """
        Compute Simpson diversity index.
        
        D = 1 - sum(p_i^2)
        """
        total = plants + animal
        
        # Avoid division by zero
        mask = total > 0
        diversity = pd.Series(0.0, index=plants.index)
        
        if mask.any():
            # Proportions
            p_plants = plants[mask] / total[mask]
            p_animal = animal[mask] / total[mask]
            
            # Simpson index
            diversity.loc[mask] = 1 - (p_plants**2 + p_animal**2)
        
        return diversity
    
    def _update_richness_metadata(self, features: pd.DataFrame) -> None:
        """Update feature metadata with richness-specific information."""
        metadata_updates = {
            'total_richness': {
                'description': 'Total species richness (plants + animal)',
                'category': 'richness',
                'transformation': 'sum'
            },
            'plants_richness': {
                'description': 'Plant species richness',
                'category': 'richness',
                'transformation': 'identity'
            },
            'animal_richness': {
                'description': 'Terrestrial species richness',
                'category': 'richness',
                'transformation': 'identity'
            },
            'plant_animal_ratio': {
                'description': 'Ratio of plant to animal richness',
                'category': 'richness',
                'transformation': 'ratio'
            },
            'animal_plant_ratio': {
                'description': 'Ratio of animal to plant richness',
                'category': 'richness',
                'transformation': 'ratio'
            },
            'plant_proportion': {
                'description': 'Proportion of total richness from plants',
                'category': 'richness',
                'transformation': 'proportion'
            },
            'animal_proportion': {
                'description': 'Proportion of total richness from animal species',
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
            'log_animal_richness': {
                'description': 'Log-transformed animal richness',
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
            'log_animal_richness': 0.12,
            'total_richness': 0.10,
            'plants_richness': 0.08,
            'animal_richness': 0.08,
            'plant_animal_ratio': 0.08,
            'animal_plant_ratio': 0.07,
            'plant_proportion': 0.05,
            'animal_proportion': 0.05,
            'shannon_diversity': 0.05,
            'simpson_diversity': 0.05
        }
        
        # Only return features that are actually being created
        return {k: v for k, v in importance.items() if k in self._feature_names}