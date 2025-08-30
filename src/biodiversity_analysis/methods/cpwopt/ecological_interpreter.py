from src.abstractions import get_logger
"""Ecological interpretation helpers for CP-WOPT results.

This module provides tools to interpret CP-WOPT components in ecological terms,
including biodiversity pattern identification, biome naming, and feature analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = get_logger(__name__)


@dataclass
class EcologicalPattern:
    """Represents an identified ecological pattern."""
    component_id: int
    pattern_type: str  # 'hotspot', 'gradient', 'endemic', 'transitional'
    dominant_features: List[str]
    geographic_extent: Dict[str, float]  # lat/lon bounds
    confidence: float
    description: str


class EcologicalInterpreter:
    """Interprets CP-WOPT components in ecological terms."""
    
    def __init__(self, feature_names: List[str]):
        """Initialize interpreter.
        
        Args:
            feature_names: Names of biodiversity features
        """
        self.feature_names = feature_names
        
        # Define ecological pattern templates
        self.pattern_templates = {
            'tropical_hotspot': {
                'lat_range': (-30, 30),
                'features': ['plants_richness', 'animal_richness'],
                'description': 'Tropical biodiversity hotspot'
            },
            'temperate_forest': {
                'lat_range': (30, 60),
                'features': ['plants_richness', 'ecm_fungi_richness'],
                'description': 'Temperate forest ecosystem'
            },
            'polar_specialist': {
                'lat_range': (60, 90),
                'features': ['animal_richness'],
                'description': 'Polar-adapted fauna'
            },
            'fungal_dominated': {
                'lat_range': (-90, 90),
                'features': ['am_fungi_richness', 'ecm_fungi_richness'],
                'description': 'Fungal diversity center'
            },
            'coastal_transition': {
                'lat_range': (-90, 90),
                'features': ['plants_richness', 'animal_richness'],
                'description': 'Coastal biodiversity zone'
            }
        }
    
    def interpret_components(self, factors: List[np.ndarray],
                           component_weights: Optional[np.ndarray] = None,
                           tensor_shape: Optional[Tuple[int, ...]] = None) -> Dict[int, EcologicalPattern]:
        """Interpret all components in ecological terms.
        
        Args:
            factors: List of factor matrices [latitude, longitude, features]
            component_weights: Optional weights for each component
            tensor_shape: Shape of original tensor
            
        Returns:
            Dictionary mapping component ID to ecological interpretation
        """
        interpretations = {}
        
        lat_factor = factors[0]
        lon_factor = factors[1]
        feat_factor = factors[2]
        
        n_components = lat_factor.shape[1]
        
        for comp in range(n_components):
            # Skip weak components
            if component_weights is not None and comp < len(component_weights):
                weight = component_weights[comp] / (np.max(component_weights) + 1e-10)
                if weight < 0.1:  # Skip components with <10% relative weight
                    continue
            
            # Analyze component
            pattern = self._analyze_component(
                comp, lat_factor, lon_factor, feat_factor, tensor_shape
            )
            
            if pattern is not None:
                interpretations[comp] = pattern
                logger.info(f"Component {comp}: {pattern.description}")
        
        return interpretations
    
    def _analyze_component(self, comp_id: int,
                         lat_factor: np.ndarray,
                         lon_factor: np.ndarray,
                         feat_factor: np.ndarray,
                         tensor_shape: Optional[Tuple[int, ...]] = None) -> Optional[EcologicalPattern]:
        """Analyze a single component.
        
        Args:
            comp_id: Component index
            lat_factor: Latitude factor matrix
            lon_factor: Longitude factor matrix
            feat_factor: Feature factor matrix
            tensor_shape: Original tensor shape
            
        Returns:
            EcologicalPattern or None if no clear pattern
        """
        # Get factor loadings for this component
        lat_loading = lat_factor[:, comp_id]
        lon_loading = lon_factor[:, comp_id]
        feat_loading = feat_factor[:, comp_id]
        
        # Identify dominant features
        dominant_features = self._identify_dominant_features(feat_loading)
        
        # Identify geographic extent
        geographic_extent = self._identify_geographic_extent(
            lat_loading, lon_loading, tensor_shape
        )
        
        # Classify pattern type
        pattern_type = self._classify_pattern_type(
            lat_loading, lon_loading, feat_loading, geographic_extent
        )
        
        # Generate description
        description = self._generate_description(
            pattern_type, dominant_features, geographic_extent
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(lat_loading, lon_loading, feat_loading)
        
        return EcologicalPattern(
            component_id=comp_id,
            pattern_type=pattern_type,
            dominant_features=dominant_features,
            geographic_extent=geographic_extent,
            confidence=confidence,
            description=description
        )
    
    def _identify_dominant_features(self, feat_loading: np.ndarray,
                                  threshold: float = 0.3) -> List[str]:
        """Identify features with strong loadings.
        
        Args:
            feat_loading: Feature loadings
            threshold: Threshold for dominance
            
        Returns:
            List of dominant feature names
        """
        # Normalize loadings
        abs_loading = np.abs(feat_loading)
        if np.max(abs_loading) > 0:
            norm_loading = abs_loading / (np.max(abs_loading) + 1e-10)
        else:
            return []
        
        # Find dominant features
        dominant_idx = np.where(norm_loading > threshold)[0]
        dominant_features = [self.feature_names[i] for i in dominant_idx]
        
        # Sort by loading strength
        sorted_idx = np.argsort(abs_loading[dominant_idx])[::-1]
        dominant_features = [dominant_features[i] for i in sorted_idx]
        
        return dominant_features
    
    def _identify_geographic_extent(self, lat_loading: np.ndarray,
                                  lon_loading: np.ndarray,
                                  tensor_shape: Optional[Tuple[int, ...]] = None) -> Dict[str, float]:
        """Identify geographic extent of pattern.
        
        Args:
            lat_loading: Latitude loadings
            lon_loading: Longitude loadings
            tensor_shape: Tensor dimensions
            
        Returns:
            Dictionary with lat/lon bounds and center
        """
        # Find significant loadings (top 20%)
        lat_threshold = np.percentile(np.abs(lat_loading), 80)
        lon_threshold = np.percentile(np.abs(lon_loading), 80)
        
        significant_lats = np.where(np.abs(lat_loading) > lat_threshold)[0]
        significant_lons = np.where(np.abs(lon_loading) > lon_threshold)[0]
        
        # Convert indices to coordinates
        if tensor_shape:
            n_lat, n_lon = tensor_shape[:2]
            lat_coords = -90 + (significant_lats * 180 / n_lat)
            lon_coords = -180 + (significant_lons * 360 / n_lon)
        else:
            # Assume global coverage
            lat_coords = -90 + (significant_lats * 180 / len(lat_loading))
            lon_coords = -180 + (significant_lons * 360 / len(lon_loading))
        
        if len(lat_coords) > 0 and len(lon_coords) > 0:
            extent = {
                'lat_min': float(np.min(lat_coords)),
                'lat_max': float(np.max(lat_coords)),
                'lat_center': float(np.mean(lat_coords)),
                'lon_min': float(np.min(lon_coords)),
                'lon_max': float(np.max(lon_coords)),
                'lon_center': float(np.mean(lon_coords))
            }
        else:
            # Global pattern
            extent = {
                'lat_min': -90.0,
                'lat_max': 90.0,
                'lat_center': 0.0,
                'lon_min': -180.0,
                'lon_max': 180.0,
                'lon_center': 0.0
            }
        
        return extent
    
    def _classify_pattern_type(self, lat_loading: np.ndarray,
                              lon_loading: np.ndarray,
                              feat_loading: np.ndarray,
                              geographic_extent: Dict[str, float]) -> str:
        """Classify the type of ecological pattern.
        
        Args:
            lat_loading: Latitude loadings
            lon_loading: Longitude loadings  
            feat_loading: Feature loadings
            geographic_extent: Geographic bounds
            
        Returns:
            Pattern type string
        """
        # Calculate pattern metrics
        lat_range = geographic_extent['lat_max'] - geographic_extent['lat_min']
        lon_range = geographic_extent['lon_max'] - geographic_extent['lon_min']
        
        # Check for latitudinal gradient
        lat_correlation = np.corrcoef(np.arange(len(lat_loading)), lat_loading)[0, 1]
        
        # Check concentration
        lat_concentration = np.std(lat_loading) / (np.mean(np.abs(lat_loading)) + 1e-10)
        lon_concentration = np.std(lon_loading) / (np.mean(np.abs(lon_loading)) + 1e-10)
        
        # Classify pattern
        if lat_range < 30 and lon_range < 60:
            return 'hotspot'  # Localized biodiversity hotspot
        elif abs(lat_correlation) > 0.7:
            return 'gradient'  # Latitudinal gradient
        elif lat_range < 30:
            return 'endemic'  # Latitude-restricted (endemic zone)
        elif lat_concentration > 2 or lon_concentration > 2:
            return 'transitional'  # Transition zone between biomes
        else:
            return 'widespread'  # Widespread pattern
    
    def _generate_description(self, pattern_type: str,
                            dominant_features: List[str],
                            geographic_extent: Dict[str, float]) -> str:
        """Generate human-readable description of pattern.
        
        Args:
            pattern_type: Type of pattern
            dominant_features: List of dominant features
            geographic_extent: Geographic bounds
            
        Returns:
            Description string
        """
        # Get latitude zone
        lat_center = geographic_extent['lat_center']
        if abs(lat_center) < 23.5:
            lat_zone = "tropical"
        elif abs(lat_center) < 66.5:
            lat_zone = "temperate"
        else:
            lat_zone = "polar"
        
        # Build feature description
        if len(dominant_features) == 0:
            feat_desc = "mixed biodiversity"
        elif len(dominant_features) == 1:
            feat_desc = dominant_features[0].replace('_', ' ')
        else:
            feat_desc = " and ".join([f.replace('_', ' ') for f in dominant_features[:2]])
        
        # Generate description based on pattern type
        if pattern_type == 'hotspot':
            desc = f"{lat_zone.capitalize()} biodiversity hotspot with high {feat_desc}"
        elif pattern_type == 'gradient':
            desc = f"Latitudinal gradient in {feat_desc}"
        elif pattern_type == 'endemic':
            desc = f"{lat_zone.capitalize()} endemic zone for {feat_desc}"
        elif pattern_type == 'transitional':
            desc = f"Transition zone between biomes, characterized by {feat_desc}"
        else:
            desc = f"Widespread pattern of {feat_desc}"
        
        return desc
    
    def _calculate_confidence(self, lat_loading: np.ndarray,
                            lon_loading: np.ndarray,
                            feat_loading: np.ndarray) -> float:
        """Calculate confidence in pattern interpretation.
        
        Args:
            lat_loading: Latitude loadings
            lon_loading: Longitude loadings
            feat_loading: Feature loadings
            
        Returns:
            Confidence score (0-1)
        """
        # Factor 1: Spatial coherence (how concentrated the pattern is)
        lat_coherence = np.max(np.abs(lat_loading)) / (np.mean(np.abs(lat_loading)) + 1e-10)
        lon_coherence = np.max(np.abs(lon_loading)) / (np.mean(np.abs(lon_loading)) + 1e-10)
        spatial_coherence = min(1.0, (lat_coherence + lon_coherence) / 10)
        
        # Factor 2: Feature clarity (how distinct the feature loadings are)
        if len(feat_loading) > 1:
            sorted_features = np.sort(np.abs(feat_loading))[::-1]
            feature_clarity = (sorted_features[0] - sorted_features[1]) / (sorted_features[0] + 1e-10)
        else:
            feature_clarity = 1.0
        
        # Factor 3: Loading strength
        loading_strength = min(1.0, np.max(np.abs(feat_loading)))
        
        # Combine factors
        confidence = (spatial_coherence + feature_clarity + loading_strength) / 3
        
        return float(np.clip(confidence, 0, 1))
    
    def generate_biome_names(self, interpretations: Dict[int, EcologicalPattern]) -> Dict[int, str]:
        """Generate meaningful names for biomes based on patterns.
        
        Args:
            interpretations: Component interpretations
            
        Returns:
            Dictionary mapping component ID to biome name
        """
        biome_names = {}
        
        for comp_id, pattern in interpretations.items():
            # Start with pattern type
            if pattern.pattern_type == 'hotspot':
                base_name = "Biodiversity Hotspot"
            elif pattern.pattern_type == 'gradient':
                base_name = "Gradient Zone"
            elif pattern.pattern_type == 'endemic':
                base_name = "Endemic Region"
            elif pattern.pattern_type == 'transitional':
                base_name = "Transition Zone"
            else:
                base_name = "Biome"
            
            # Add geographic qualifier
            lat_center = pattern.geographic_extent['lat_center']
            if abs(lat_center) < 23.5:
                geo_qualifier = "Tropical"
            elif abs(lat_center) < 35:
                geo_qualifier = "Subtropical"
            elif abs(lat_center) < 55:
                geo_qualifier = "Temperate"
            elif abs(lat_center) < 66.5:
                geo_qualifier = "Boreal"
            else:
                geo_qualifier = "Polar"
            
            # Add feature qualifier
            if pattern.dominant_features:
                if 'plants_richness' in pattern.dominant_features:
                    feat_qualifier = "Floristic"
                elif any('fungi' in f for f in pattern.dominant_features):
                    feat_qualifier = "Mycological"
                elif 'animal_richness' in pattern.dominant_features:
                    feat_qualifier = "Faunal"
                else:
                    feat_qualifier = ""
            else:
                feat_qualifier = ""
            
            # Combine parts
            parts = [geo_qualifier, feat_qualifier, base_name]
            biome_name = " ".join(p for p in parts if p)
            
            biome_names[comp_id] = biome_name
        
        return biome_names
    
    def create_interpretation_report(self, interpretations: Dict[int, EcologicalPattern],
                                   biome_names: Dict[int, str]) -> pd.DataFrame:
        """Create a summary report of ecological interpretations.
        
        Args:
            interpretations: Component interpretations
            biome_names: Biome names
            
        Returns:
            DataFrame with interpretation summary
        """
        records = []
        
        for comp_id, pattern in interpretations.items():
            record = {
                'Component': comp_id + 1,  # 1-indexed for display
                'Biome Name': biome_names.get(comp_id, f'Biome {comp_id + 1}'),
                'Pattern Type': pattern.pattern_type.replace('_', ' ').title(),
                'Primary Features': ', '.join(pattern.dominant_features[:2]),
                'Latitude Range': f"{pattern.geographic_extent['lat_min']:.0f}° to {pattern.geographic_extent['lat_max']:.0f}°",
                'Longitude Range': f"{pattern.geographic_extent['lon_min']:.0f}° to {pattern.geographic_extent['lon_max']:.0f}°",
                'Confidence': f"{pattern.confidence:.0%}",
                'Description': pattern.description
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Sort by confidence
        df = df.sort_values('Component')
        
        return df