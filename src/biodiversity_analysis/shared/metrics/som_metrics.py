"""Biodiversity-specific metrics for SOM evaluation."""

import numpy as np
from typing import Dict, Tuple, List, Optional
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import logging

logger = logging.getLogger(__name__)


class SOMMetrics:
    """Biodiversity-specific metrics for SOM evaluation.
    
    Provides metrics that assess how well the SOM preserves
    ecological patterns and relationships in the data.
    """
    
    @staticmethod
    def calculate_species_coherence(som_weights: np.ndarray, 
                                  species_data: np.ndarray,
                                  bmu_indices: np.ndarray,
                                  species_names: Optional[List[str]] = None) -> float:
        """Calculate how well the SOM preserves species co-occurrence patterns.
        
        This metric measures whether species that co-occur in the original data
        are also grouped together in the SOM.
        
        Args:
            som_weights: SOM weight matrix (flattened to 2D)
            species_data: Original species data (n_samples, n_species)
            bmu_indices: BMU index for each sample
            species_names: Optional species names for detailed analysis
            
        Returns:
            Species coherence score (0 to 1, higher is better)
        """
        n_samples, n_species = species_data.shape
        
        # Calculate species co-occurrence in original data
        species_binary = (species_data > 0).astype(float)
        cooccurrence_original = np.dot(species_binary.T, species_binary) / n_samples
        
        # Calculate species co-occurrence in SOM clusters
        n_neurons = som_weights.shape[0]
        cooccurrence_som = np.zeros((n_species, n_species))
        
        for neuron_idx in range(n_neurons):
            # Get samples assigned to this neuron
            sample_mask = bmu_indices == neuron_idx
            if np.sum(sample_mask) > 0:
                cluster_species = species_binary[sample_mask]
                cluster_cooc = np.dot(cluster_species.T, cluster_species) / np.sum(sample_mask)
                cooccurrence_som += cluster_cooc * np.sum(sample_mask) / n_samples
        
        # Compare co-occurrence patterns
        # Flatten upper triangular matrices (excluding diagonal)
        triu_indices = np.triu_indices(n_species, k=1)
        original_flat = cooccurrence_original[triu_indices]
        som_flat = cooccurrence_som[triu_indices]
        
        # Calculate correlation
        if len(original_flat) > 0 and np.std(original_flat) > 0 and np.std(som_flat) > 0:
            coherence, _ = pearsonr(original_flat, som_flat)
            coherence = max(0, coherence)  # Ensure non-negative
        else:
            coherence = 0.0
        
        logger.info(f"Species coherence: {coherence:.3f}")
        return coherence
    
    @staticmethod
    def calculate_beta_diversity_preservation(original_data: np.ndarray,
                                            som_clusters: np.ndarray,
                                            distance_metric: str = 'bray_curtis') -> float:
        """Measure how well beta diversity is preserved in SOM clustering.
        
        Beta diversity represents the variation in species composition between sites.
        This metric assesses whether the SOM preserves these compositional differences.
        
        Args:
            original_data: Original species abundance data (n_samples, n_species)
            som_clusters: Cluster assignments from SOM (n_samples,)
            distance_metric: Distance metric for beta diversity ('bray_curtis', 'jaccard')
            
        Returns:
            Beta diversity preservation score (0 to 1, higher is better)
        """
        n_samples = original_data.shape[0]
        
        # Calculate pairwise dissimilarities in original data
        if distance_metric == 'bray_curtis':
            original_distances = pdist(original_data, metric='braycurtis')
        elif distance_metric == 'jaccard':
            binary_data = (original_data > 0).astype(float)
            original_distances = pdist(binary_data, metric='jaccard')
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
        
        # Calculate average within-cluster and between-cluster distances
        distance_matrix = squareform(original_distances)
        unique_clusters = np.unique(som_clusters)
        
        within_cluster_distances = []
        between_cluster_distances = []
        
        for cluster_id in unique_clusters:
            cluster_mask = som_clusters == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            # Within-cluster distances
            if len(cluster_indices) > 1:
                within_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
                within_cluster_distances.extend(within_distances[np.triu_indices_from(within_distances, k=1)])
            
            # Between-cluster distances
            other_indices = np.where(~cluster_mask)[0]
            if len(other_indices) > 0:
                between_distances = distance_matrix[np.ix_(cluster_indices, other_indices)]
                between_cluster_distances.extend(between_distances.flatten())
        
        # Calculate preservation score
        if within_cluster_distances and between_cluster_distances:
            avg_within = np.mean(within_cluster_distances)
            avg_between = np.mean(between_cluster_distances)
            
            # Score based on separation (between-cluster distance should be larger)
            preservation = (avg_between - avg_within) / (avg_between + avg_within + 1e-8)
            preservation = max(0, min(1, preservation))  # Clamp to [0, 1]
        else:
            preservation = 0.0
        
        logger.info(f"Beta diversity preservation: {preservation:.3f}")
        return preservation
    
    @staticmethod
    def calculate_environmental_gradient_detection(som_weights: np.ndarray,
                                                 environmental_vars: np.ndarray,
                                                 var_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Assess how well SOM captures environmental gradients.
        
        Environmental gradients often drive species distributions. This metric
        evaluates whether the SOM organization reflects these gradients.
        
        Args:
            som_weights: SOM weights reshaped to grid (n_rows, n_cols, n_features)
            environmental_vars: Environmental variables for each neuron position
            var_names: Names of environmental variables
            
        Returns:
            Dictionary mapping variable names to gradient detection scores
        """
        n_rows, n_cols, n_features = som_weights.shape
        
        if environmental_vars.shape[0] != n_rows * n_cols:
            raise ValueError("Environmental variables must match number of SOM neurons")
        
        n_env_vars = environmental_vars.shape[1]
        if var_names is None:
            var_names = [f"Env_{i}" for i in range(n_env_vars)]
        
        gradient_scores = {}
        
        # Create grid coordinates
        grid_x, grid_y = np.meshgrid(range(n_cols), range(n_rows))
        grid_coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        
        for i, var_name in enumerate(var_names):
            env_values = environmental_vars[:, i]
            
            # Calculate spatial autocorrelation of environmental variable on SOM
            # Using Moran's I as a measure of gradient structure
            moran_i = SOMMetrics._calculate_morans_i(env_values, grid_coords)
            
            # Also calculate gradient directionality
            gradient_strength = SOMMetrics._calculate_gradient_strength(
                env_values.reshape(n_rows, n_cols)
            )
            
            # Combine metrics
            gradient_score = (moran_i + gradient_strength) / 2
            gradient_scores[var_name] = max(0, min(1, gradient_score))
        
        logger.info(f"Environmental gradient detection scores: {gradient_scores}")
        return gradient_scores
    
    @staticmethod
    def calculate_rare_species_representation(species_data: np.ndarray,
                                            bmu_indices: np.ndarray,
                                            prevalence_threshold: float = 0.1) -> float:
        """Assess how well rare species are represented in the SOM.
        
        Rare species are important for biodiversity but can be overlooked
        by clustering methods. This metric checks their representation.
        
        Args:
            species_data: Original species data (n_samples, n_species)
            bmu_indices: BMU assignments for each sample
            prevalence_threshold: Threshold for considering a species rare
            
        Returns:
            Rare species representation score (0 to 1, higher is better)
        """
        n_samples, n_species = species_data.shape
        
        # Identify rare species
        species_prevalence = np.mean(species_data > 0, axis=0)
        rare_species_mask = species_prevalence < prevalence_threshold
        n_rare_species = np.sum(rare_species_mask)
        
        if n_rare_species == 0:
            logger.warning("No rare species found below prevalence threshold")
            return 1.0
        
        # Check representation in SOM clusters
        unique_clusters = np.unique(bmu_indices)
        rare_species_represented = np.zeros(n_rare_species)
        
        for cluster_id in unique_clusters:
            cluster_samples = species_data[bmu_indices == cluster_id]
            if len(cluster_samples) > 0:
                # Check which rare species are present in this cluster
                cluster_rare_presence = np.any(cluster_samples[:, rare_species_mask] > 0, axis=0)
                rare_species_represented = np.logical_or(rare_species_represented, cluster_rare_presence)
        
        # Calculate representation score
        representation_score = np.mean(rare_species_represented)
        
        logger.info(f"Rare species representation: {representation_score:.3f} "
                   f"({np.sum(rare_species_represented)}/{n_rare_species} species)")
        return representation_score
    
    @staticmethod
    def calculate_spatial_autocorrelation_preserved(original_coords: np.ndarray,
                                                  original_data: np.ndarray,
                                                  bmu_indices: np.ndarray) -> float:
        """Measure how well spatial autocorrelation is preserved.
        
        Many biodiversity patterns show spatial autocorrelation. This metric
        checks if spatially close samples remain close in the SOM.
        
        Args:
            original_coords: Spatial coordinates (n_samples, 2)
            original_data: Original feature data (n_samples, n_features)
            bmu_indices: BMU assignments
            
        Returns:
            Spatial autocorrelation preservation score
        """
        n_samples = original_coords.shape[0]
        
        # Calculate spatial distances
        spatial_distances = pdist(original_coords, metric='euclidean')
        spatial_dist_matrix = squareform(spatial_distances)
        
        # Calculate SOM distances (based on BMU positions)
        bmu_positions = np.array([divmod(idx, int(np.sqrt(np.max(bmu_indices) + 1))) 
                                 for idx in bmu_indices])
        som_distances = pdist(bmu_positions, metric='euclidean')
        
        # Calculate correlation between spatial and SOM distances
        if len(spatial_distances) > 0 and np.std(spatial_distances) > 0 and np.std(som_distances) > 0:
            correlation, _ = spearmanr(spatial_distances, som_distances)
            preservation_score = 1 - abs(correlation)  # Convert to preservation score
        else:
            preservation_score = 0.0
        
        logger.info(f"Spatial autocorrelation preservation: {preservation_score:.3f}")
        return preservation_score
    
    @staticmethod
    def _calculate_morans_i(values: np.ndarray, coords: np.ndarray) -> float:
        """Calculate Moran's I for spatial autocorrelation.
        
        Args:
            values: Values at each location
            coords: Spatial coordinates
            
        Returns:
            Moran's I statistic
        """
        n = len(values)
        if n < 3:
            return 0.0
        
        # Create spatial weights matrix (inverse distance)
        distances = pdist(coords, metric='euclidean')
        dist_matrix = squareform(distances)
        
        # Avoid division by zero
        dist_matrix[dist_matrix == 0] = np.inf
        weights = 1.0 / dist_matrix
        np.fill_diagonal(weights, 0)
        
        # Calculate Moran's I
        values_centered = values - np.mean(values)
        numerator = np.sum(weights * np.outer(values_centered, values_centered))
        denominator = np.sum(values_centered ** 2)
        
        if denominator > 0:
            morans_i = (n / np.sum(weights)) * (numerator / denominator)
            # Normalize to [0, 1]
            morans_i = (morans_i + 1) / 2
        else:
            morans_i = 0.5
        
        return morans_i
    
    @staticmethod
    def _calculate_gradient_strength(grid_values: np.ndarray) -> float:
        """Calculate strength of gradient in 2D grid.
        
        Args:
            grid_values: Values arranged in 2D grid
            
        Returns:
            Gradient strength score
        """
        # Calculate gradients in x and y directions
        grad_y, grad_x = np.gradient(grid_values)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize by value range
        value_range = np.ptp(grid_values) + 1e-8
        normalized_gradient = gradient_magnitude / value_range
        
        # Average gradient strength
        avg_gradient = np.mean(normalized_gradient)
        
        # Convert to score (higher gradient = better organization)
        return min(1.0, avg_gradient * 2)  # Scale to [0, 1]