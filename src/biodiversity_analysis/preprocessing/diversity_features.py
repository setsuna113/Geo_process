"""Biodiversity-specific feature engineering.

This module creates advanced biodiversity features including:
- Diversity indices (Shannon, Simpson, etc.)
- Functional diversity metrics
- Phylogenetic diversity (when available)
- Multi-scale diversity gradients
- Interaction features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import logging

logger = logging.getLogger(__name__)


class BiodiversityFeatureEngineering:
    """Generate biodiversity-specific features."""
    
    def __init__(self,
                 calculate_interactions: bool = True,
                 environmental_features: List[str] = None,
                 gradient_scales: List[float] = None):
        """Initialize biodiversity feature engineering.
        
        Args:
            calculate_interactions: Whether to create diversity × environment interactions
            environmental_features: Names of environmental features for interactions
            gradient_scales: Scales (km) for diversity gradient calculation
        """
        self.calculate_interactions = calculate_interactions
        self.environmental_features = environmental_features or []
        self.gradient_scales = gradient_scales or [10, 25, 50]
        
    def fit_transform(self,
                      species_data: np.ndarray,
                      coordinates: np.ndarray,
                      species_names: List[str] = None,
                      environmental_data: np.ndarray = None,
                      trait_data: np.ndarray = None,
                      phylo_distances: np.ndarray = None) -> Tuple[np.ndarray, List[str]]:
        """Generate all biodiversity features.
        
        Args:
            species_data: Species abundance/presence matrix (n_samples, n_species)
            coordinates: Lat/lon coordinates (n_samples, 2)
            species_names: Names of species
            environmental_data: Environmental variables (n_samples, n_env_features)
            trait_data: Species functional traits (n_species, n_traits)
            phylo_distances: Phylogenetic distance matrix (n_species, n_species)
            
        Returns:
            Feature matrix and feature names
        """
        features = []
        feature_names = []
        
        # Alpha diversity indices
        logger.info("Calculating alpha diversity indices...")
        alpha_features, alpha_names = self._calculate_alpha_diversity(species_data, species_names)
        features.append(alpha_features)
        feature_names.extend(alpha_names)
        
        # Beta diversity features
        logger.info("Calculating beta diversity features...")
        beta_features, beta_names = self._calculate_beta_diversity(species_data, coordinates)
        features.append(beta_features)
        feature_names.extend(beta_names)
        
        # Functional diversity
        if trait_data is not None:
            logger.info("Calculating functional diversity...")
            func_features, func_names = self._calculate_functional_diversity(
                species_data, trait_data, species_names
            )
            features.append(func_features)
            feature_names.extend(func_names)
        
        # Phylogenetic diversity
        if phylo_distances is not None:
            logger.info("Calculating phylogenetic diversity...")
            phylo_features, phylo_names = self._calculate_phylogenetic_diversity(
                species_data, phylo_distances, species_names
            )
            features.append(phylo_features)
            feature_names.extend(phylo_names)
        
        # Diversity gradients
        logger.info("Calculating diversity gradients...")
        grad_features, grad_names = self._calculate_diversity_gradients(
            alpha_features[:, 0], coordinates  # Use Shannon diversity
        )
        features.append(grad_features)
        feature_names.extend(grad_names)
        
        # Interaction features
        if self.calculate_interactions and environmental_data is not None:
            logger.info("Calculating interaction features...")
            inter_features, inter_names = self._calculate_interactions(
                alpha_features, environmental_data, alpha_names
            )
            features.append(inter_features)
            feature_names.extend(inter_names)
        
        # Combine all features
        combined = np.hstack(features)
        logger.info(f"Created {combined.shape[1]} biodiversity features")
        
        return combined, feature_names
    
    def _calculate_alpha_diversity(self, 
                                   species_data: np.ndarray,
                                   species_names: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        """Calculate alpha diversity indices."""
        n_samples = species_data.shape[0]
        
        # Initialize indices
        shannon = np.zeros(n_samples)
        simpson = np.zeros(n_samples)
        richness = np.zeros(n_samples)
        evenness = np.zeros(n_samples)
        margalef = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Get abundances for this sample
            abundances = species_data[i, :]
            valid_mask = ~np.isnan(abundances)
            
            if not np.any(valid_mask):
                shannon[i] = np.nan
                simpson[i] = np.nan
                richness[i] = np.nan
                evenness[i] = np.nan
                margalef[i] = np.nan
                continue
            
            # Use valid abundances
            abund = abundances[valid_mask]
            abund = abund[abund > 0]  # Remove zeros
            
            if len(abund) == 0:
                shannon[i] = 0
                simpson[i] = 0
                richness[i] = 0
                evenness[i] = 0
                margalef[i] = 0
                continue
            
            # Species richness
            S = len(abund)
            richness[i] = S
            
            # Total abundance
            N = np.sum(abund)
            
            # Relative abundances
            p = abund / N
            
            # Shannon diversity
            shannon[i] = -np.sum(p * np.log(p + 1e-10))
            
            # Simpson diversity
            simpson[i] = 1 - np.sum(p ** 2)
            
            # Pielou's evenness
            if S > 1:
                evenness[i] = shannon[i] / np.log(S)
            else:
                evenness[i] = 1.0
            
            # Margalef's richness
            if N > 1:
                margalef[i] = (S - 1) / np.log(N)
            else:
                margalef[i] = 0
        
        # Stack features
        features = np.column_stack([shannon, simpson, richness, evenness, margalef])
        names = ['shannon_diversity', 'simpson_diversity', 'species_richness', 
                 'pielou_evenness', 'margalef_richness']
        
        return features, names
    
    def _calculate_beta_diversity(self,
                                  species_data: np.ndarray,
                                  coordinates: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Calculate beta diversity features for each sample."""
        from .spatial_features import SpatialFeatureEngineering
        
        n_samples = species_data.shape[0]
        
        # Build spatial weights for neighbors
        spatial_eng = SpatialFeatureEngineering(n_neighbors=10)
        W = spatial_eng._build_spatial_weights(coordinates)
        
        # Calculate beta diversity metrics
        mean_dissim = np.zeros(n_samples)
        turnover = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Get neighbors
            neighbors = W[i].nonzero()[1]
            
            if len(neighbors) == 0:
                mean_dissim[i] = np.nan
                turnover[i] = np.nan
                continue
            
            # Calculate dissimilarity with each neighbor
            dissims = []
            turnovers = []
            
            for j in neighbors:
                # Bray-Curtis dissimilarity
                valid_mask = ~(np.isnan(species_data[i, :]) | np.isnan(species_data[j, :]))
                
                if np.any(valid_mask):
                    a = species_data[i, valid_mask]
                    b = species_data[j, valid_mask]
                    
                    # Bray-Curtis
                    bc = np.sum(np.abs(a - b)) / (np.sum(a) + np.sum(b) + 1e-10)
                    dissims.append(bc)
                    
                    # Species turnover (Jaccard on presence/absence)
                    a_bin = a > 0
                    b_bin = b > 0
                    shared = np.sum(a_bin & b_bin)
                    total = np.sum(a_bin | b_bin)
                    
                    if total > 0:
                        jaccard = 1 - (shared / total)
                        turnovers.append(jaccard)
            
            if dissims:
                mean_dissim[i] = np.mean(dissims)
                turnover[i] = np.mean(turnovers) if turnovers else np.nan
            else:
                mean_dissim[i] = np.nan
                turnover[i] = np.nan
        
        features = np.column_stack([mean_dissim, turnover])
        names = ['mean_beta_diversity', 'species_turnover']
        
        return features, names
    
    def _calculate_functional_diversity(self,
                                        species_data: np.ndarray,
                                        trait_data: np.ndarray,
                                        species_names: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        """Calculate functional diversity metrics."""
        n_samples = species_data.shape[0]
        
        # Calculate trait distance matrix
        trait_dist = squareform(pdist(trait_data, metric='euclidean'))
        
        # Initialize metrics
        fric = np.zeros(n_samples)  # Functional richness
        feve = np.zeros(n_samples)  # Functional evenness
        fdiv = np.zeros(n_samples)  # Functional divergence
        
        for i in range(n_samples):
            # Get present species
            present = species_data[i, :] > 0
            valid = ~np.isnan(species_data[i, :])
            present_species = present & valid
            
            if np.sum(present_species) < 2:
                fric[i] = 0
                feve[i] = 0
                fdiv[i] = 0
                continue
            
            # Get abundances and traits for present species
            present_idx = np.where(present_species)[0]
            abundances = species_data[i, present_idx]
            present_traits = trait_data[present_idx, :]
            
            # Functional richness (convex hull volume proxy)
            # Simplified: use trait range
            trait_ranges = np.ptp(present_traits, axis=0)
            fric[i] = np.prod(trait_ranges + 1)  # Add 1 to avoid zero
            
            # Functional evenness (regularity of abundance distribution)
            # Based on minimum spanning tree
            n_species = len(present_idx)
            if n_species > 2:
                # Simplified: use coefficient of variation
                rel_abund = abundances / np.sum(abundances)
                feve[i] = 1 - (np.std(rel_abund) / (np.mean(rel_abund) + 1e-10))
            else:
                feve[i] = 1.0
            
            # Functional divergence (divergence in trait values)
            # Simplified: weighted variance from centroid
            weights = abundances / np.sum(abundances)
            centroid = np.average(present_traits, weights=weights, axis=0)
            distances = np.sqrt(np.sum((present_traits - centroid) ** 2, axis=1))
            fdiv[i] = np.average(distances, weights=weights)
        
        features = np.column_stack([fric, feve, fdiv])
        names = ['functional_richness', 'functional_evenness', 'functional_divergence']
        
        return features, names
    
    def _calculate_phylogenetic_diversity(self,
                                          species_data: np.ndarray,
                                          phylo_distances: np.ndarray,
                                          species_names: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        """Calculate phylogenetic diversity metrics."""
        n_samples = species_data.shape[0]
        
        # Initialize metrics
        pd = np.zeros(n_samples)    # Phylogenetic diversity
        mpd = np.zeros(n_samples)   # Mean pairwise distance
        mntd = np.zeros(n_samples)  # Mean nearest taxon distance
        
        for i in range(n_samples):
            # Get present species
            present = species_data[i, :] > 0
            valid = ~np.isnan(species_data[i, :])
            present_species = present & valid
            
            n_present = np.sum(present_species)
            if n_present < 2:
                pd[i] = 0
                mpd[i] = 0
                mntd[i] = 0
                continue
            
            # Get phylogenetic distances for present species
            present_idx = np.where(present_species)[0]
            sub_phylo = phylo_distances[np.ix_(present_idx, present_idx)]
            
            # Faith's PD (sum of branch lengths)
            # Approximated as sum of unique distances
            pd[i] = np.sum(np.unique(sub_phylo)) / 2
            
            # Mean pairwise distance
            triu_idx = np.triu_indices_from(sub_phylo, k=1)
            if len(triu_idx[0]) > 0:
                mpd[i] = np.mean(sub_phylo[triu_idx])
            
            # Mean nearest taxon distance
            np.fill_diagonal(sub_phylo, np.inf)
            min_dists = np.min(sub_phylo, axis=1)
            mntd[i] = np.mean(min_dists[min_dists < np.inf])
        
        features = np.column_stack([pd, mpd, mntd])
        names = ['phylogenetic_diversity', 'mean_pairwise_distance', 'mean_nearest_taxon_distance']
        
        return features, names
    
    def _calculate_diversity_gradients(self,
                                       diversity: np.ndarray,
                                       coordinates: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Calculate spatial gradients of diversity."""
        from scipy.spatial import cKDTree
        
        # Convert to km
        lat, lon = coordinates[:, 0], coordinates[:, 1]
        R = 6371  # Earth radius
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        x = R * lon_rad * np.cos(np.mean(lat_rad))
        y = R * lat_rad
        coords_km = np.column_stack([x, y])
        
        tree = cKDTree(coords_km)
        
        features = []
        names = []
        
        for scale in self.gradient_scales:
            # Find neighbors within scale
            neighbors = tree.query_ball_tree(tree, r=scale)
            
            # Calculate gradient magnitude
            gradient = np.zeros(len(diversity))
            
            for i, neighs in enumerate(neighbors):
                if len(neighs) > 1 and not np.isnan(diversity[i]):
                    # Calculate differences with neighbors
                    diffs = []
                    dists = []
                    
                    for j in neighs:
                        if j != i and not np.isnan(diversity[j]):
                            diff = diversity[j] - diversity[i]
                            dist = np.linalg.norm(coords_km[i] - coords_km[j])
                            if dist > 0:
                                diffs.append(diff / dist)
                    
                    if diffs:
                        gradient[i] = np.sqrt(np.mean(np.array(diffs) ** 2))
                    else:
                        gradient[i] = 0
                else:
                    gradient[i] = np.nan
            
            features.append(gradient)
            names.append(f'diversity_gradient_{scale}km')
        
        return np.column_stack(features), names
    
    def _calculate_interactions(self,
                                diversity_features: np.ndarray,
                                environmental_data: np.ndarray,
                                diversity_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Calculate diversity × environment interaction features."""
        features = []
        names = []
        
        # Use first few diversity indices for interactions
        n_div_features = min(3, diversity_features.shape[1])
        
        for i in range(n_div_features):
            div_feat = diversity_features[:, i]
            div_name = diversity_names[i]
            
            for j, env_name in enumerate(self.environmental_features[:3]):  # Limit interactions
                if j < environmental_data.shape[1]:
                    env_feat = environmental_data[:, j]
                    
                    # Multiplicative interaction
                    interaction = div_feat * env_feat
                    features.append(interaction)
                    names.append(f'{div_name}_x_{env_name}')
        
        if features:
            return np.column_stack(features), names
        else:
            return np.empty((diversity_features.shape[0], 0)), []