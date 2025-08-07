"""Shared preprocessing module for all biodiversity analysis methods.

Simple, unified preprocessing that avoids redundancy and works for SOM, 
CP-WOPT, K-means, and other methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


class SharedPreprocessor:
    """Simple shared preprocessor for all analysis methods.
    
    This replaces method-specific preprocessing to avoid redundancy.
    Each method can customize via method_name parameter.
    """
    
    def __init__(self, 
                 method_name: str = None,
                 normalize: bool = True,
                 normalization_method: str = 'auto',
                 handle_missing: bool = True,
                 missing_strategy: str = 'zero',
                 remove_constant_features: bool = False,
                 variance_threshold: float = 1e-10):
        """Initialize shared preprocessor.
        
        Args:
            method_name: Name of analysis method ('som', 'cpwopt', 'kmeans', etc.)
            normalize: Whether to normalize/standardize data
            normalization_method: 'auto' (method-specific), 'minmax', 'standard', 'robust', or 'none'
            handle_missing: Whether to handle missing values
            missing_strategy: 'zero', 'mean', 'median', or 'keep'
            remove_constant_features: Whether to remove constant features
            variance_threshold: Minimum variance to keep features
        """
        self.method_name = method_name
        self.normalize = normalize
        self.normalization_method = normalization_method
        self.handle_missing = handle_missing
        self.missing_strategy = missing_strategy
        self.remove_constant_features = remove_constant_features
        self.variance_threshold = variance_threshold
        self.scaler = None
        self._fitted = False
        
        # Auto-configure based on method
        if normalization_method == 'auto':
            self._auto_configure()
    
    def _auto_configure(self):
        """Auto-configure preprocessing based on analysis method."""
        if self.method_name == 'som':
            # SOM prefers minmax normalization
            self.normalization_method = 'minmax'
            self.missing_strategy = 'zero'  # SOM handles sparse data well
        elif self.method_name == 'cpwopt':
            # CP-WOPT needs non-negative values
            self.normalization_method = 'minmax'
            self.missing_strategy = 'zero'
        elif self.method_name == 'kmeans':
            # K-means prefers standardization
            self.normalization_method = 'standard'
            self.missing_strategy = 'mean'
        else:
            # Default settings
            self.normalization_method = 'robust'
            self.missing_strategy = 'zero'
    
    def fit(self, data: Union[np.ndarray, pd.DataFrame]) -> 'SharedPreprocessor':
        """Fit the preprocessor on training data.
        
        Args:
            data: Input data (samples x features)
            
        Returns:
            Self for method chaining
        """
        # Convert to numpy if needed
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        # Handle missing values first
        if self.handle_missing and self.missing_strategy != 'keep':
            data = self._handle_missing_values(data)
        
        # Fit scaler if needed
        if self.normalize and self.normalization_method != 'none':
            if self.normalization_method == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.normalization_method == 'standard':
                self.scaler = StandardScaler()
            elif self.normalization_method == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown normalization method: {self.normalization_method}")
            
            # Fit scaler
            self.scaler.fit(data)
            logger.info(f"Fitted {self.normalization_method} scaler")
        
        self._fitted = True
        return self
    
    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform data using fitted preprocessing.
        
        Args:
            data: Input data (samples x features)
            
        Returns:
            Preprocessed data as numpy array
        """
        # Convert to numpy if needed
        if isinstance(data, pd.DataFrame):
            column_names = data.columns.tolist()
            data = data.values
        else:
            column_names = None
        
        # Handle missing values
        if self.handle_missing and self.missing_strategy != 'keep':
            data = self._handle_missing_values(data)
        
        # Remove constant features if requested
        if self.remove_constant_features:
            data = self._remove_constant_features(data)
        
        # Apply normalization
        if self.normalize and self.normalization_method != 'none':
            if not self._fitted:
                raise ValueError("Preprocessor must be fitted before transform")
            data = self.scaler.transform(data)
            logger.debug(f"Applied {self.normalization_method} normalization")
        
        return data
    
    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit and transform in one step.
        
        Args:
            data: Input data (samples x features)
            
        Returns:
            Preprocessed data as numpy array
        """
        self.fit(data)
        return self.transform(data)
    
    def _handle_missing_values(self, data: np.ndarray) -> np.ndarray:
        """Handle missing values according to strategy."""
        if self.missing_strategy == 'zero':
            # For biodiversity data, 0 is meaningful (no species)
            return np.nan_to_num(data, nan=0.0)
        elif self.missing_strategy == 'mean':
            # Simple mean imputation
            col_means = np.nanmean(data, axis=0)
            for i in range(data.shape[1]):
                data[np.isnan(data[:, i]), i] = col_means[i]
            return data
        elif self.missing_strategy == 'median':
            # Simple median imputation
            col_medians = np.nanmedian(data, axis=0)
            for i in range(data.shape[1]):
                data[np.isnan(data[:, i]), i] = col_medians[i]
            return data
        elif self.missing_strategy == 'keep':
            # Keep NaN values as is
            return data
        else:
            raise ValueError(f"Unknown missing value strategy: {self.missing_strategy}")
    
    def _remove_constant_features(self, data: np.ndarray) -> np.ndarray:
        """Remove features with constant or near-constant values."""
        # Calculate variance
        variances = np.var(data, axis=0)
        
        # Keep features with variance above threshold
        mask = variances > self.variance_threshold
        
        n_removed = (~mask).sum()
        if n_removed > 0:
            logger.info(f"Removed {n_removed} constant features")
            data = data[:, mask]
        
        return data


# Keep the complex pipeline for backward compatibility but deprecate it
class BiodiversityPreprocessingPipeline:
    """Complete preprocessing pipeline for biodiversity data."""
    
    def __init__(self,
                 # Missing value handling
                 missing_value_strategy: str = 'keep',
                 missing_threshold: float = 0.5,
                 
                 # Zero-inflation parameters
                 structural_zero_threshold: float = 0.95,
                 pseudo_count_method: str = 'adaptive',
                 compositional_transform: str = 'clr',
                 
                 # Feature filtering
                 remove_constant_features: bool = True,
                 variance_threshold: float = 1e-10,
                 
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
            missing_value_strategy: How to handle missing values:
                - 'keep': Keep NaN values (default)
                - 'drop_samples': Remove samples with any NaN
                - 'drop_features': Remove features with >missing_threshold NaN
                - 'mean': Impute with mean
                - 'median': Impute with median
                - 'zero': Replace with zero
            missing_threshold: Max proportion of missing values for drop_features
            structural_zero_threshold: Threshold for structural zeros
            pseudo_count_method: Method for pseudo-counts
            compositional_transform: Type of compositional transform ('clr', 'ilr', 'alr', 'hurdle')
            remove_constant_features: Whether to remove constant features
            variance_threshold: Minimum variance to keep features
            spatial_neighbors: Number of spatial neighbors
            spatial_scales: Scales for multi-scale features
            n_moran_eigenvectors: Number of Moran eigenvectors
            use_pmem: Use predictive MEM instead of standard MEM
            pmem_dwf: Distance-weighting function for pMEM
            calculate_interactions: Whether to calculate interaction features
            calculate_diversity_indices: Whether to calculate diversity indices as features
            environmental_features: Names of environmental features
            standardize: Whether to standardize features
            standardization_method: 'robust', 'standard', or 'minmax'
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
        
        self.missing_value_strategy = missing_value_strategy
        self.missing_threshold = missing_threshold
        self.remove_constant_features = remove_constant_features
        self.variance_threshold = variance_threshold
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
            elif self.standardization_method == 'standard':
                self.scaler = StandardScaler()
            elif self.standardization_method == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown standardization method: {self.standardization_method}")
            
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
        
        # Step 0: Handle missing values
        species_data, coordinates = self._handle_missing_values(
            species_data, coordinates, self.species_names_
        )
        
        # Step 1: Zero-inflation preprocessing
        logger.info("Step 1/6: Handling zero-inflation...")
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
        
        # Step 5: Remove constant features
        logger.info("Step 5/6: Removing constant features...")
        combined, all_names = self._remove_constant_features(combined, all_names)
        
        # Step 6: Standardization
        if self.standardize:
            logger.info("Step 6/6: Standardizing features...")
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
    
    def _handle_missing_values(self, 
                              species_data: np.ndarray,
                              coordinates: np.ndarray,
                              species_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Handle missing values according to strategy."""
        logger.info(f"Handling missing values with strategy: {self.missing_value_strategy}")
        
        if self.missing_value_strategy == 'keep':
            return species_data, coordinates
            
        elif self.missing_value_strategy == 'drop_samples':
            # Remove samples with any missing values
            mask = ~np.any(np.isnan(species_data), axis=1)
            logger.info(f"Dropping {(~mask).sum()} samples with missing values")
            return species_data[mask], coordinates[mask]
            
        elif self.missing_value_strategy == 'drop_features':
            # Remove features with too many missing values
            missing_prop = np.isnan(species_data).mean(axis=0)
            mask = missing_prop <= self.missing_threshold
            n_dropped = (~mask).sum()
            if n_dropped > 0:
                logger.info(f"Dropping {n_dropped} features with >{self.missing_threshold:.1%} missing")
                self.species_names_ = [n for i, n in enumerate(species_names) if mask[i]]
            return species_data[:, mask], coordinates
            
        elif self.missing_value_strategy in ['mean', 'median']:
            # Impute missing values
            imputer = SimpleImputer(strategy=self.missing_value_strategy)
            species_data_imputed = imputer.fit_transform(species_data)
            logger.info(f"Imputed missing values with {self.missing_value_strategy}")
            return species_data_imputed, coordinates
            
        elif self.missing_value_strategy == 'zero':
            # Replace missing with zero
            species_data_filled = np.nan_to_num(species_data, nan=0.0)
            logger.info("Replaced missing values with zero")
            return species_data_filled, coordinates
            
        else:
            raise ValueError(f"Unknown missing value strategy: {self.missing_value_strategy}")
    
    def _remove_constant_features(self,
                                 features: np.ndarray,
                                 feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Remove features with constant or near-constant values."""
        if not self.remove_constant_features:
            return features, feature_names
            
        # Calculate variance ignoring NaN
        variances = np.nanvar(features, axis=0)
        
        # Keep features with variance above threshold
        mask = variances > self.variance_threshold
        
        n_removed = (~mask).sum()
        if n_removed > 0:
            logger.info(f"Removed {n_removed} constant features (variance < {self.variance_threshold})")
            
        return features[:, mask], [name for i, name in enumerate(feature_names) if mask[i]]