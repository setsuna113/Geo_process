"""Complete preprocessing pipeline for biodiversity clustering.

Combines all preprocessing steps into a unified pipeline following
the recommendations from KMEANS_UPGRADE_RECOMMENDATIONS.md.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.preprocessing import RobustScaler

from .spatial_features import SpatialFeatureEngineering
from .zero_inflation import ZeroInflationPreprocessor
from .diversity_features import BiodiversityFeatureEngineering

logger = logging.getLogger(__name__)


class BiodiversityPreprocessingPipeline:
    """Complete preprocessing pipeline for biodiversity data."""
    
    def __init__(self,
                 # Zero-inflation parameters
                 structural_zero_threshold: float = 0.95,
                 pseudo_count_method: str = 'adaptive',
                 compositional_transform: str = 'clr',
                 
                 # Spatial parameters
                 spatial_neighbors: int = 8,
                 spatial_scales: List[float] = None,
                 n_moran_eigenvectors: int = 20,
                 use_pmem: bool = False,
                 pmem_dwf: str = 'exponential',
                 
                 # Diversity parameters
                 calculate_interactions: bool = True,
                 calculate_diversity_indices: bool = True,
                 environmental_features: List[str] = None,
                 
                 # Standardization
                 standardize: bool = True,
                 standardization_method: str = 'robust'):
        """Initialize preprocessing pipeline.
        
        Args:
            structural_zero_threshold: Threshold for structural zeros
            pseudo_count_method: Method for pseudo-counts
            compositional_transform: Type of compositional transform ('clr', 'ilr', 'alr', 'hurdle')
            spatial_neighbors: Number of spatial neighbors
            spatial_scales: Scales for multi-scale features
            n_moran_eigenvectors: Number of Moran eigenvectors
            use_pmem: Use predictive MEM instead of standard MEM
            pmem_dwf: Distance-weighting function for pMEM
            calculate_interactions: Whether to calculate interaction features
            calculate_diversity_indices: Whether to calculate diversity indices as features
            environmental_features: Names of environmental features
            standardize: Whether to standardize features
            standardization_method: 'robust' or 'standard'
        """
        # Initialize components
        self.zero_preprocessor = ZeroInflationPreprocessor(
            structural_zero_threshold=structural_zero_threshold,
            pseudo_count_method=pseudo_count_method,
            compositional_transform=compositional_transform
        )
        
        self.spatial_engineer = SpatialFeatureEngineering(
            n_neighbors=spatial_neighbors,
            scales=spatial_scales,
            use_pmem=use_pmem,
            pmem_dwf=pmem_dwf
        )
        
        self.diversity_engineer = BiodiversityFeatureEngineering(
            calculate_interactions=calculate_interactions,
            environmental_features=environmental_features
        )
        
        self.n_moran_eigenvectors = n_moran_eigenvectors
        self.calculate_diversity_indices = calculate_diversity_indices
        self.standardize = standardize
        self.standardization_method = standardization_method
        self.scaler = None
        
        self._fitted = False
        
    def fit(self,
            species_data: np.ndarray,
            coordinates: np.ndarray,
            species_names: List[str] = None,
            environmental_data: np.ndarray = None,
            trait_data: np.ndarray = None,
            phylo_distances: np.ndarray = None) -> 'BiodiversityPreprocessingPipeline':
        """Fit preprocessing pipeline.
        
        Args:
            species_data: Species abundance matrix (n_samples, n_species)
            coordinates: Lat/lon coordinates (n_samples, 2)
            species_names: Names of species
            environmental_data: Environmental variables
            trait_data: Species functional traits
            phylo_distances: Phylogenetic distance matrix
            
        Returns:
            Self
        """
        logger.info("Fitting biodiversity preprocessing pipeline...")
        
        # Store original data info
        self.n_samples_ = species_data.shape[0]
        self.n_species_ = species_data.shape[1]
        self.species_names_ = species_names or [f'species_{i}' for i in range(self.n_species_)]
        
        # Fit zero-inflation preprocessor
        logger.info("Step 1: Fitting zero-inflation preprocessor...")
        self.zero_preprocessor.fit(species_data, self.species_names_)
        
        # Calculate diversity features to get feature names
        logger.info("Step 2: Calculating diversity features...")
        div_features, div_names = self.diversity_engineer.fit_transform(
            species_data, coordinates, species_names,
            environmental_data, trait_data, phylo_distances
        )
        
        # Calculate spatial features on diversity features
        logger.info("Step 3: Calculating spatial features...")
        spatial_features, spatial_names = self.spatial_engineer.fit_transform(
            div_features, coordinates, div_names
        )
        
        # Fit scaler if needed
        if self.standardize:
            logger.info("Step 4: Fitting scaler...")
            if self.standardization_method == 'robust':
                self.scaler = RobustScaler()
            else:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
            
            # Fit on all features
            zero_features, _ = self.zero_preprocessor.transform(species_data)
            all_features = np.hstack([zero_features, div_features, spatial_features])
            self.scaler.fit(all_features)
        
        self._fitted = True
        logger.info("Pipeline fitting complete!")
        
        return self
    
    def transform(self,
                  species_data: np.ndarray,
                  coordinates: np.ndarray,
                  environmental_data: np.ndarray = None,
                  trait_data: np.ndarray = None,
                  phylo_distances: np.ndarray = None) -> Tuple[np.ndarray, List[str], Dict]:
        """Transform data through preprocessing pipeline.
        
        Returns:
            Tuple of (transformed_data, feature_names, metadata_dict)
        """
        if not self._fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        all_features = []
        all_names = []
        metadata = {}
        
        # Step 1: Zero-inflation preprocessing
        logger.info("Step 1/5: Handling zero-inflation...")
        zero_features, zero_names = self.zero_preprocessor.transform(species_data)
        all_features.append(zero_features)
        all_names.extend(zero_names)
        metadata['n_zero_features'] = zero_features.shape[1]
        
        # Step 2: Diversity features (conditional)
        if self.calculate_diversity_indices:
            logger.info("Step 2/5: Creating diversity features...")
            div_features, div_names = self.diversity_engineer.fit_transform(
                species_data, coordinates, self.species_names_,
                environmental_data, trait_data, phylo_distances
            )
            all_features.append(div_features)
            all_names.extend(div_names)
            metadata['n_diversity_features'] = div_features.shape[1]
        else:
            logger.info("Step 2/5: Skipping diversity features (circular reasoning prevention)")
            metadata['n_diversity_features'] = 0
        
        # Step 3: Spatial features on zero-inflated data
        logger.info("Step 3/5: Creating spatial features on species data...")
        spatial_zero_features, spatial_zero_names = self.spatial_engineer.fit_transform(
            zero_features, coordinates, zero_names
        )
        all_features.append(spatial_zero_features)
        all_names.extend(spatial_zero_names)
        
        # Step 4: Spatial features on diversity indices (conditional)
        if self.calculate_diversity_indices:
            logger.info("Step 4/5: Creating spatial features on diversity indices...")
            spatial_div_features, spatial_div_names = self.spatial_engineer.fit_transform(
                div_features, coordinates, div_names
            )
            all_features.append(spatial_div_features)
            all_names.extend(spatial_div_names)
            metadata['n_spatial_features'] = spatial_zero_features.shape[1] + spatial_div_features.shape[1]
        else:
            logger.info("Step 4/5: Skipping spatial features on diversity indices")
            metadata['n_spatial_features'] = spatial_zero_features.shape[1]
        
        # Combine all features
        combined = np.hstack(all_features)
        
        # Step 5: Standardization
        if self.standardize:
            logger.info("Step 5/5: Standardizing features...")
            combined = self.scaler.transform(combined)
        
        # Calculate global Moran's I for quality check
        logger.info("Calculating spatial autocorrelation metrics...")
        morans_i = self.spatial_engineer.calculate_global_morans(
            combined[:, :min(10, combined.shape[1])],  # First 10 features
            coordinates,
            all_names[:min(10, combined.shape[1])]
        )
        metadata['global_morans_i'] = morans_i
        
        # Summary statistics
        metadata['n_total_features'] = combined.shape[1]
        metadata['n_samples'] = combined.shape[0]
        if self.calculate_diversity_indices:
            metadata['feature_groups'] = {
                'zero_inflation': (0, metadata['n_zero_features']),
                'diversity': (metadata['n_zero_features'], 
                             metadata['n_zero_features'] + metadata['n_diversity_features']),
                'spatial': (metadata['n_zero_features'] + metadata['n_diversity_features'],
                           metadata['n_total_features'])
            }
        else:
            metadata['feature_groups'] = {
                'zero_inflation': (0, metadata['n_zero_features']),
                'spatial': (metadata['n_zero_features'], metadata['n_total_features'])
            }
        
        logger.info(f"Preprocessing complete: {combined.shape[0]} samples, {combined.shape[1]} features")
        
        return combined, all_names, metadata
    
    def fit_transform(self,
                      species_data: np.ndarray,
                      coordinates: np.ndarray,
                      species_names: List[str] = None,
                      environmental_data: np.ndarray = None,
                      trait_data: np.ndarray = None,
                      phylo_distances: np.ndarray = None) -> Tuple[np.ndarray, List[str], Dict]:
        """Fit and transform in one step."""
        self.fit(species_data, coordinates, species_names,
                 environmental_data, trait_data, phylo_distances)
        return self.transform(species_data, coordinates,
                            environmental_data, trait_data, phylo_distances)
    
    def get_preprocessing_report(self) -> str:
        """Generate detailed preprocessing report."""
        if not self._fitted:
            return "Pipeline not fitted yet"
        
        report = []
        report.append("# Biodiversity Preprocessing Pipeline Report\n")
        
        # Zero-inflation summary
        report.append("## Zero-Inflation Preprocessing\n")
        zero_summary = self.zero_preprocessor.get_transform_summary()
        report.append(zero_summary.to_string())
        report.append("\n")
        
        # Feature counts
        report.append("\n## Feature Engineering Summary\n")
        report.append(f"- Original species: {self.n_species_}")
        report.append(f"- Samples: {self.n_samples_}")
        
        return "\n".join(report)
    
    def save_feature_metadata(self, filepath: str):
        """Save feature names and metadata to file."""
        # This would save feature names and groupings for later use
        pass