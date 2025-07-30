"""Feature builder for ecological and environmental features."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Set, Optional, Union
import logging

from ....base.feature_builder import BaseFeatureBuilder
from ....core.registry import feature_builder

logger = logging.getLogger(__name__)


@feature_builder(
    "ecological",
    required_columns=set(),  # No required columns - flexible based on available data
    description="Builds ecological and environmental features (extensible for future data)"
)
class EcologicalFeatureBuilder(BaseFeatureBuilder):
    """
    Feature builder for ecological/environmental features.
    
    Currently creates placeholder features and derived metrics from available data.
    Designed to be extensible when climate, NDVI, elevation, or other environmental
    data becomes available.
    
    Future data sources can include:
    - Climate variables (temperature, precipitation, seasonality)
    - NDVI/vegetation indices
    - Elevation and terrain features
    - Soil characteristics
    - Land use/land cover
    - Distance to water bodies
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ecological feature builder."""
        super().__init__(config)
        
        # Get feature engineering config
        self.feature_config = config.get('machine_learning', {}).get(
            'feature_engineering', {}
        ).get('ecological_features', {}) if config else {}
        
        # Feature flags for future data
        self.climate_interactions = self.feature_config.get('climate_interactions', True)
        self.seasonality_metrics = self.feature_config.get('seasonality_metrics', True)
        self.placeholder_ndvi = self.feature_config.get('placeholder_ndvi', False)
        
        # Track available data sources
        self.available_climate_vars = []
        self.available_vegetation_indices = []
        self.available_terrain_vars = []
        
        # Define feature names based on available data
        self._feature_names = []
        self._optional_columns = set()
        
    def get_feature_names(self) -> List[str]:
        """Get list of feature names that will be created."""
        # Dynamically determine based on available data
        return self._feature_names.copy()
    
    def get_required_columns(self) -> Set[str]:
        """Get set of required input columns."""
        # No strictly required columns - we adapt to available data
        return set()
    
    def _detect_available_data(self, data: pd.DataFrame) -> None:
        """Detect what ecological data is available in the input."""
        self._optional_columns.clear()
        self.available_climate_vars.clear()
        self.available_vegetation_indices.clear()
        self.available_terrain_vars.clear()
        
        # Check for climate variables
        climate_patterns = {
            'temperature': ['temp', 'temperature', 'bio_1', 'mean_temp'],
            'precipitation': ['precip', 'precipitation', 'bio_12', 'annual_precip'],
            'seasonality': ['bio_4', 'bio_15', 'temp_seasonality', 'precip_seasonality']
        }
        
        for var_type, patterns in climate_patterns.items():
            for pattern in patterns:
                matching_cols = [col for col in data.columns if pattern in col.lower()]
                if matching_cols:
                    self.available_climate_vars.extend(matching_cols)
                    self._optional_columns.update(matching_cols)
                    logger.info(f"Found {var_type} variables: {matching_cols}")
        
        # Check for vegetation indices
        vegetation_patterns = ['ndvi', 'evi', 'lai', 'vegetation', 'greenness']
        for pattern in vegetation_patterns:
            matching_cols = [col for col in data.columns if pattern in col.lower()]
            if matching_cols:
                self.available_vegetation_indices.extend(matching_cols)
                self._optional_columns.update(matching_cols)
                logger.info(f"Found vegetation indices: {matching_cols}")
        
        # Check for terrain variables
        terrain_patterns = ['elevation', 'slope', 'aspect', 'terrain', 'altitude']
        for pattern in terrain_patterns:
            matching_cols = [col for col in data.columns if pattern in col.lower()]
            if matching_cols:
                self.available_terrain_vars.extend(matching_cols)
                self._optional_columns.update(matching_cols)
                logger.info(f"Found terrain variables: {matching_cols}")
    
    def _fit(self, data: pd.DataFrame, **kwargs) -> None:
        """
        Fit the feature builder on training data.
        
        Detects available ecological data and prepares transformations.
        """
        # Detect what data is available
        self._detect_available_data(data)
        
        # Update feature names based on available data
        self._feature_names = self._define_feature_names()
        
        # Store statistics for any available ecological variables
        for col in self._optional_columns:
            if col in data.columns:
                self._fitted_params[f'{col}_mean'] = data[col].mean()
                self._fitted_params[f'{col}_std'] = data[col].std()
                self._fitted_params[f'{col}_min'] = data[col].min()
                self._fitted_params[f'{col}_max'] = data[col].max()
        
        # Log what we found
        logger.info(f"Fitted ecological features with {len(self._optional_columns)} "
                   f"environmental variables")
        if not self._optional_columns:
            logger.warning("No ecological/environmental data found. Creating placeholder "
                          "features for future extension.")
    
    def _define_feature_names(self) -> List[str]:
        """Define feature names based on available data."""
        features = []
        
        # If we have climate data
        if self.available_climate_vars:
            # Original variables
            features.extend(self.available_climate_vars)
            
            # Interactions if we have both temp and precip
            if self.climate_interactions:
                temp_vars = [v for v in self.available_climate_vars if 'temp' in v.lower()]
                precip_vars = [v for v in self.available_climate_vars if 'precip' in v.lower()]
                
                for temp in temp_vars:
                    for precip in precip_vars:
                        features.append(f'{temp}_x_{precip}_interaction')
        
        # If we have vegetation indices
        if self.available_vegetation_indices:
            features.extend(self.available_vegetation_indices)
            
            # Seasonality metrics if multiple time periods
            if self.seasonality_metrics and len(self.available_vegetation_indices) > 1:
                features.append('vegetation_seasonality')
        
        # If we have terrain data
        if self.available_terrain_vars:
            features.extend(self.available_terrain_vars)
            
            # Terrain complexity metrics
            if 'slope' in self.available_terrain_vars and 'aspect' in self.available_terrain_vars:
                features.append('terrain_complexity')
        
        # Placeholder features if no ecological data available
        if not features:
            features = [
                'ecological_placeholder_1',  # Placeholder for future climate data
                'ecological_placeholder_2',  # Placeholder for future vegetation data
                'ecological_placeholder_3'   # Placeholder for future terrain data
            ]
            
            if self.placeholder_ndvi:
                features.append('ndvi_placeholder')
        
        return features
    
    def _transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform data to create ecological features.
        
        Args:
            data: Input DataFrame
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame(index=data.index)
        
        # Process available ecological variables
        if self._optional_columns:
            # Include original variables
            for col in self._optional_columns:
                if col in data.columns:
                    features[col] = data[col]
            
            # Create interactions if applicable
            if self.climate_interactions and len(self.available_climate_vars) > 1:
                temp_vars = [v for v in self.available_climate_vars if v in data.columns and 'temp' in v.lower()]
                precip_vars = [v for v in self.available_climate_vars if v in data.columns and 'precip' in v.lower()]
                
                for temp in temp_vars:
                    for precip in precip_vars:
                        interaction_name = f'{temp}_x_{precip}_interaction'
                        features[interaction_name] = data[temp] * data[precip]
            
            # Vegetation seasonality if multiple indices
            if self.seasonality_metrics and len(self.available_vegetation_indices) > 1:
                veg_cols = [v for v in self.available_vegetation_indices if v in data.columns]
                if len(veg_cols) > 1:
                    veg_data = data[veg_cols]
                    features['vegetation_seasonality'] = veg_data.std(axis=1) / (veg_data.mean(axis=1) + 1e-8)
            
            # Terrain complexity
            if 'slope' in data.columns and 'aspect' in data.columns:
                # Simple terrain ruggedness index
                features['terrain_complexity'] = np.sqrt(
                    data['slope']**2 + (data['aspect'] / 180)**2
                )
        
        else:
            # Create placeholder features
            # These will be replaced when real ecological data is available
            features['ecological_placeholder_1'] = 0.0  # Future: mean temperature
            features['ecological_placeholder_2'] = 0.0  # Future: annual precipitation
            features['ecological_placeholder_3'] = 0.0  # Future: elevation
            
            if self.placeholder_ndvi:
                features['ndvi_placeholder'] = 0.0  # Future: NDVI
            
            # Add some noise to avoid constant features
            if 'latitude' in data.columns:
                # Create pseudo-environmental gradient based on latitude
                features['ecological_placeholder_1'] += 0.1 * np.sin(np.radians(data['latitude']))
                features['ecological_placeholder_2'] += 0.1 * np.cos(np.radians(data['latitude']))
        
        # Update metadata
        self._update_ecological_metadata(features)
        
        return features
    
    def _update_ecological_metadata(self, features: pd.DataFrame) -> None:
        """Update feature metadata with ecological-specific information."""
        # Update base metadata first
        self._update_feature_metadata(features)
        
        # Add ecological-specific metadata
        for feature in features.columns:
            if feature in self.feature_metadata:
                if 'placeholder' in feature:
                    self.feature_metadata[feature].update({
                        'description': f'Placeholder for future ecological data',
                        'category': 'ecological',
                        'transformation': 'placeholder',
                        'is_placeholder': True
                    })
                elif 'interaction' in feature:
                    self.feature_metadata[feature].update({
                        'description': f'Climate variable interaction: {feature}',
                        'category': 'ecological',
                        'transformation': 'interaction'
                    })
                elif 'seasonality' in feature:
                    self.feature_metadata[feature].update({
                        'description': 'Seasonality metric',
                        'category': 'ecological',
                        'transformation': 'seasonality'
                    })
                elif 'complexity' in feature:
                    self.feature_metadata[feature].update({
                        'description': 'Terrain complexity index',
                        'category': 'ecological',
                        'transformation': 'derived'
                    })
                else:
                    self.feature_metadata[feature].update({
                        'category': 'ecological',
                        'transformation': 'identity'
                    })
    
    def add_external_data_source(self, 
                                data_type: str,
                                columns: List[str],
                                metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register an external data source for future use.
        
        This method allows the feature builder to be extended with new data sources
        without modifying the core implementation.
        
        Args:
            data_type: Type of data ('climate', 'vegetation', 'terrain', etc.)
            columns: List of column names from this data source
            metadata: Optional metadata about the data source
        """
        if data_type == 'climate':
            self.available_climate_vars.extend(columns)
        elif data_type == 'vegetation':
            self.available_vegetation_indices.extend(columns)
        elif data_type == 'terrain':
            self.available_terrain_vars.extend(columns)
        
        self._optional_columns.update(columns)
        
        # Update feature names
        self._feature_names = self._define_feature_names()
        
        logger.info(f"Added {data_type} data source with columns: {columns}")
    
    def get_feature_importance_prior(self) -> Optional[Dict[str, float]]:
        """
        Get prior importance for ecological features.
        
        Returns:
            Dictionary of feature importance priors
        """
        if not self._feature_names:
            return None
        
        importance = {}
        
        # If we have real ecological data
        if self._optional_columns:
            base_importance = 0.8 / len(self._feature_names) if self._feature_names else 0.1
            
            for feature in self._feature_names:
                if 'interaction' in feature:
                    importance[feature] = base_importance * 1.2  # Interactions often important
                elif 'seasonality' in feature:
                    importance[feature] = base_importance * 1.1
                else:
                    importance[feature] = base_importance
        else:
            # Placeholder features have low importance
            for feature in self._feature_names:
                importance[feature] = 0.01
        
        return importance