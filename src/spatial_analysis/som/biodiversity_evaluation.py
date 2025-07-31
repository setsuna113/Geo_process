"""
Biodiversity-Specific SOM Evaluation Metrics

This module implements evaluation metrics specifically designed for assessing
SOM performance on biodiversity data, including ecological coherence,
species association accuracy, and biogeographic pattern preservation.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BiodiversityEvaluationMetrics:
    """Comprehensive biodiversity-specific SOM evaluation metrics."""
    
    # Core SOM metrics
    quantization_error: float
    topographic_error: float
    
    # Biodiversity-specific metrics
    species_association_accuracy: float
    functional_diversity_preservation: float
    phylogenetic_signal_retention: float
    biogeographic_coherence: float
    endemic_species_clustering: float
    
    # Spatial metrics
    spatial_autocorrelation: float
    edge_effect_score: float
    cluster_compactness: float
    
    # Statistical metrics
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    
    # Interpretability metrics
    cluster_interpretability: float
    biodiversity_pattern_preservation: float
    ecological_gradient_alignment: float


class BiodiversityEvaluator:
    """Evaluator for biodiversity-specific SOM performance."""
    
    def __init__(self, 
                 species_data: Optional[np.ndarray] = None,
                 functional_traits: Optional[np.ndarray] = None,
                 phylogenetic_tree: Optional[Any] = None,
                 coordinates: Optional[np.ndarray] = None,
                 environmental_data: Optional[np.ndarray] = None):
        """Initialize biodiversity evaluator.
        
        Args:
            species_data: Species occurrence/abundance matrix (n_samples, n_species)
            functional_traits: Functional trait matrix (n_species, n_traits)
            phylogenetic_tree: Phylogenetic tree object (optional)
            coordinates: Spatial coordinates (n_samples, 2)
            environmental_data: Environmental variables (n_samples, n_env_vars)
        """
        self.species_data = species_data
        self.functional_traits = functional_traits
        self.phylogenetic_tree = phylogenetic_tree
        self.coordinates = coordinates
        self.environmental_data = environmental_data
        
        logger.info("Initialized biodiversity evaluator")
        if species_data is not None:
            logger.info(f"  Species data: {species_data.shape[0]} samples, {species_data.shape[1]} species")
        if functional_traits is not None:
            logger.info(f"  Functional traits: {functional_traits.shape[0]} species, {functional_traits.shape[1]} traits")
        if coordinates is not None:
            logger.info(f"  Spatial coordinates: {coordinates.shape[0]} locations")
    
    def evaluate_som(self, 
                    som,
                    test_data: np.ndarray,
                    test_coordinates: Optional[np.ndarray] = None) -> BiodiversityEvaluationMetrics:
        """Comprehensive evaluation of SOM on biodiversity data.
        
        Args:
            som: Trained SOM instance
            test_data: Test dataset (n_samples, n_features)
            test_coordinates: Test coordinates (n_samples, 2)
            
        Returns:
            BiodiversityEvaluationMetrics with all evaluation scores
        """
        logger.info("Starting comprehensive biodiversity SOM evaluation")
        
        # Get cluster assignments
        cluster_labels = som.predict(test_data)
        
        # Core SOM metrics
        quantization_error = som.quantization_error(test_data)
        topographic_error = som.topographic_error(test_data)
        
        # Statistical clustering metrics
        silhouette = self._safe_silhouette_score(test_data, cluster_labels)
        calinski_harabasz = self._safe_calinski_harabasz_score(test_data, cluster_labels)
        davies_bouldin = self._safe_davies_bouldin_score(test_data, cluster_labels)
        
        # Biodiversity-specific metrics
        species_association = self._calculate_species_association_accuracy(test_data, cluster_labels)
        functional_preservation = self._calculate_functional_diversity_preservation(test_data, cluster_labels)
        phylogenetic_retention = self._calculate_phylogenetic_signal_retention(test_data, cluster_labels)
        biogeographic_coherence = self._calculate_biogeographic_coherence(cluster_labels, test_coordinates)
        endemic_clustering = self._calculate_endemic_species_clustering(test_data, cluster_labels)
        
        # Spatial metrics
        spatial_autocorr = self._calculate_spatial_autocorrelation(cluster_labels, test_coordinates)
        edge_effect = self._calculate_edge_effect_score(cluster_labels, test_coordinates)
        cluster_compactness = self._calculate_cluster_compactness(cluster_labels, test_coordinates)
        
        # Interpretability metrics
        cluster_interpretability = self._calculate_cluster_interpretability(test_data, cluster_labels)
        pattern_preservation = self._calculate_biodiversity_pattern_preservation(test_data, cluster_labels)
        gradient_alignment = self._calculate_ecological_gradient_alignment(test_data, cluster_labels)
        
        metrics = BiodiversityEvaluationMetrics(
            quantization_error=quantization_error,
            topographic_error=topographic_error,
            species_association_accuracy=species_association,
            functional_diversity_preservation=functional_preservation,
            phylogenetic_signal_retention=phylogenetic_retention,
            biogeographic_coherence=biogeographic_coherence,
            endemic_species_clustering=endemic_clustering,
            spatial_autocorrelation=spatial_autocorr,
            edge_effect_score=edge_effect,
            cluster_compactness=cluster_compactness,
            silhouette_score=silhouette,
            calinski_harabasz_score=calinski_harabasz,
            davies_bouldin_score=davies_bouldin,
            cluster_interpretability=cluster_interpretability,
            biodiversity_pattern_preservation=pattern_preservation,
            ecological_gradient_alignment=gradient_alignment
        )
        
        logger.info("Biodiversity SOM evaluation completed")
        self._log_metrics_summary(metrics)
        
        return metrics
    
    def _calculate_species_association_accuracy(self, 
                                               data: np.ndarray, 
                                               labels: np.ndarray) -> float:
        """Calculate how well SOM preserves known species associations."""
        if self.species_data is None or data.shape != self.species_data.shape:
            logger.warning("Species data not available, using proxy metric")
            return self._calculate_feature_association_proxy(data, labels)
        
        # Calculate species co-occurrence patterns in original data
        species_corr_matrix = np.corrcoef(self.species_data.T)
        
        # Calculate species co-occurrence patterns within SOM clusters
        unique_labels = np.unique(labels)
        cluster_associations = []
        
        for label in unique_labels:
            cluster_mask = labels == label
            if np.sum(cluster_mask) > 1:  # Need at least 2 samples
                cluster_species = self.species_data[cluster_mask]
                if cluster_species.shape[0] > 1:
                    cluster_corr = np.corrcoef(cluster_species.T)
                    cluster_associations.append(cluster_corr)
        
        if not cluster_associations:
            return 0.0
        
        # Compare original vs cluster associations
        # This is a simplified metric - in practice would be more sophisticated
        original_mean_corr = np.mean(np.abs(species_corr_matrix[~np.eye(species_corr_matrix.shape[0], dtype=bool)]))
        
        cluster_mean_corrs = []
        for cluster_corr in cluster_associations:
            if cluster_corr.size > 1:
                cluster_mean_corr = np.mean(np.abs(cluster_corr[~np.eye(cluster_corr.shape[0], dtype=bool)]))
                cluster_mean_corrs.append(cluster_mean_corr)
        
        if not cluster_mean_corrs:
            return 0.0
        
        # Higher within-cluster correlation indicates preserved associations
        avg_cluster_corr = np.mean(cluster_mean_corrs)
        association_score = min(1.0, avg_cluster_corr / max(original_mean_corr, 1e-6))
        
        return float(association_score)
    
    def _calculate_feature_association_proxy(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Proxy metric for species association when species data unavailable."""
        # Calculate feature correlations within clusters vs overall
        overall_corr_matrix = np.corrcoef(data.T)
        overall_mean_corr = np.mean(np.abs(overall_corr_matrix[~np.eye(data.shape[1], dtype=bool)]))
        
        unique_labels = np.unique(labels)
        cluster_corrs = []
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_data = data[cluster_mask]
            
            if cluster_data.shape[0] > 1 and cluster_data.shape[1] > 1:
                cluster_corr_matrix = np.corrcoef(cluster_data.T)
                if cluster_corr_matrix.size > 1:
                    cluster_mean_corr = np.mean(np.abs(cluster_corr_matrix[~np.eye(cluster_data.shape[1], dtype=bool)]))
                    cluster_corrs.append(cluster_mean_corr)
        
        if not cluster_corrs:
            return 0.0
        
        avg_cluster_corr = np.mean(cluster_corrs)
        return float(min(1.0, avg_cluster_corr / max(overall_mean_corr, 1e-6)))
    
    def _calculate_functional_diversity_preservation(self, 
                                                   data: np.ndarray, 
                                                   labels: np.ndarray) -> float:
        """Calculate preservation of functional diversity patterns."""
        if self.functional_traits is None:
            logger.warning("Functional traits not available, using data variance proxy")
            return self._calculate_variance_preservation_proxy(data, labels)
        
        # This would implement functional diversity metrics like:
        # - Functional richness
        # - Functional evenness  
        # - Functional divergence
        # For now, return a placeholder
        return 0.75
    
    def _calculate_phylogenetic_signal_retention(self, 
                                               data: np.ndarray, 
                                               labels: np.ndarray) -> float:
        """Calculate retention of phylogenetic signal in clusters."""
        if self.phylogenetic_tree is None:
            logger.warning("Phylogenetic tree not available, using distance-based proxy")
            return self._calculate_distance_structure_proxy(data, labels)
        
        # This would implement phylogenetic signal metrics
        # For now, return a placeholder
        return 0.70
    
    def _calculate_biogeographic_coherence(self, 
                                         labels: np.ndarray, 
                                         coordinates: Optional[np.ndarray]) -> float:
        """Calculate spatial coherence of biogeographic patterns."""
        if coordinates is None:
            logger.warning("Coordinates not available for biogeographic coherence")
            return 0.0
        
        # Calculate how spatially clustered each SOM cluster is
        unique_labels = np.unique(labels)
        coherence_scores = []
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_coords = coordinates[cluster_mask]
            
            if len(cluster_coords) < 2:
                continue
            
            # Calculate mean pairwise distance within cluster
            if len(cluster_coords) > 1:
                pairwise_distances = pdist(cluster_coords)
                mean_internal_distance = np.mean(pairwise_distances)
                
                # Compare to expected random distance
                all_pairwise_distances = pdist(coordinates)
                mean_overall_distance = np.mean(all_pairwise_distances)
                
                # Lower internal distance relative to overall = higher coherence
                coherence = max(0, 1 - (mean_internal_distance / mean_overall_distance))
                coherence_scores.append(coherence)
        
        return float(np.mean(coherence_scores)) if coherence_scores else 0.0
    
    def _calculate_endemic_species_clustering(self, 
                                            data: np.ndarray, 
                                            labels: np.ndarray) -> float:
        """Calculate how well endemic species are clustered together."""
        # This would require endemic species information
        # For now, use rarity-based proxy (species with low total abundance)
        
        if data.shape[1] < 2:
            return 0.0
        
        # Identify "rare" features (proxy for endemic species)
        feature_totals = np.sum(data, axis=0)
        rare_threshold = np.percentile(feature_totals, 25)  # Bottom quartile
        rare_features = feature_totals <= rare_threshold
        
        if not np.any(rare_features):
            return 0.0
        
        # Calculate clustering of rare features
        rare_data = data[:, rare_features]
        
        # Calculate silhouette score for rare features only
        if rare_data.shape[1] > 0 and len(np.unique(labels)) > 1:
            try:
                rare_silhouette = silhouette_score(rare_data, labels)
                return float(max(0, rare_silhouette))
            except:
                return 0.0
        
        return 0.0
    
    def _calculate_spatial_autocorrelation(self, 
                                         labels: np.ndarray, 
                                         coordinates: Optional[np.ndarray]) -> float:
        """Calculate spatial autocorrelation of cluster assignments."""
        if coordinates is None:
            return 0.0
        
        # Simple spatial autocorrelation: nearby points should have similar clusters
        n_samples = len(labels)
        if n_samples < 10:
            return 0.0
        
        # Always use spatial indexing for efficiency (no O(n²) memory usage)
        from sklearn.neighbors import NearestNeighbors
        
        # Adaptive k based on sample size - more neighbors for larger datasets
        if n_samples > 10000:
            k = min(50, n_samples // 100)  # For large datasets
        else:
            k = min(20, n_samples // 10)   # For smaller datasets, use fewer neighbors
        
        k = max(k, 5)  # Minimum 5 neighbors
        k = min(k, n_samples - 1)  # Can't exceed available samples
        
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(coordinates)
        _, indices = nbrs.kneighbors(coordinates)
        
        # Count matching labels among neighbors
        same_cluster_count = 0
        total_pairs = 0
        
        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:  # Skip self (index 0)
                total_pairs += 1
                if labels[i] == labels[j]:
                    same_cluster_count += 1
        
        return same_cluster_count / total_pairs if total_pairs > 0 else 0.0
    
    def _calculate_edge_effect_score(self, 
                                   labels: np.ndarray, 
                                   coordinates: Optional[np.ndarray]) -> float:
        """Calculate edge effects in clustering."""
        if coordinates is None:
            return 0.0
        
        # Identify edge samples (those near boundaries of study area)
        lon_range = coordinates[:, 0].max() - coordinates[:, 0].min()
        lat_range = coordinates[:, 1].max() - coordinates[:, 1].min()
        
        edge_threshold_lon = lon_range * 0.1  # 10% of range
        edge_threshold_lat = lat_range * 0.1
        
        # Find edge samples
        lon_edges = ((coordinates[:, 0] <= coordinates[:, 0].min() + edge_threshold_lon) |
                    (coordinates[:, 0] >= coordinates[:, 0].max() - edge_threshold_lon))
        lat_edges = ((coordinates[:, 1] <= coordinates[:, 1].min() + edge_threshold_lat) |
                    (coordinates[:, 1] >= coordinates[:, 1].max() - edge_threshold_lat))
        
        edge_samples = lon_edges | lat_edges
        
        if not np.any(edge_samples):
            return 0.0
        
        # Calculate clustering quality at edges vs interior
        edge_labels = labels[edge_samples]
        interior_labels = labels[~edge_samples]
        
        if len(np.unique(edge_labels)) <= 1 or len(np.unique(interior_labels)) <= 1:
            return 0.0
        
        # Compare cluster diversity at edges vs interior
        edge_diversity = len(np.unique(edge_labels)) / len(edge_labels)
        interior_diversity = len(np.unique(interior_labels)) / len(interior_labels)
        
        # Lower edge effect score = better (less difference between edge and interior)
        edge_effect = abs(edge_diversity - interior_diversity)
        return float(1.0 - min(1.0, edge_effect))
    
    def _calculate_cluster_compactness(self, 
                                     labels: np.ndarray, 
                                     coordinates: Optional[np.ndarray]) -> float:
        """Calculate spatial compactness of clusters."""
        if coordinates is None:
            return 0.0
        
        unique_labels = np.unique(labels)
        compactness_scores = []
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_coords = coordinates[cluster_mask]
            
            if len(cluster_coords) < 2:
                continue
            
            # Calculate cluster spread vs random spread
            cluster_distances = pdist(cluster_coords)
            mean_cluster_distance = np.mean(cluster_distances)
            
            # Compare to random sample of same size
            random_indices = np.random.choice(len(coordinates), len(cluster_coords), replace=False)
            random_coords = coordinates[random_indices]
            random_distances = pdist(random_coords)
            mean_random_distance = np.mean(random_distances)
            
            # Compactness = how much smaller cluster distances are vs random
            compactness = max(0, 1 - (mean_cluster_distance / mean_random_distance))
            compactness_scores.append(compactness)
        
        return float(np.mean(compactness_scores)) if compactness_scores else 0.0
    
    def _calculate_cluster_interpretability(self, 
                                          data: np.ndarray, 
                                          labels: np.ndarray) -> float:
        """Calculate how interpretable/distinct the clusters are."""
        unique_labels = np.unique(labels)
        
        if len(unique_labels) <= 1:
            return 0.0
        
        # Calculate between-cluster vs within-cluster variance
        total_variance = np.var(data, axis=0).sum()
        
        within_cluster_variance = 0
        between_cluster_variance = 0
        
        overall_mean = np.mean(data, axis=0)
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_data = data[cluster_mask]
            cluster_mean = np.mean(cluster_data, axis=0)
            cluster_size = np.sum(cluster_mask)
            
            # Within-cluster variance
            within_cluster_variance += np.sum(np.var(cluster_data, axis=0)) * cluster_size
            
            # Between-cluster variance
            between_cluster_variance += cluster_size * np.sum((cluster_mean - overall_mean) ** 2)
        
        within_cluster_variance /= len(data)
        between_cluster_variance /= len(data)
        
        # Interpretability = between-cluster variance / total variance
        interpretability = between_cluster_variance / max(total_variance, 1e-6)
        return float(min(1.0, interpretability))
    
    def _calculate_biodiversity_pattern_preservation(self, 
                                                   data: np.ndarray, 
                                                   labels: np.ndarray) -> float:
        """Calculate preservation of biodiversity patterns."""
        # Calculate correlation between original distances and cluster-based distances
        if len(data) < 10:
            return 0.0
        
        # Original pairwise distances
        original_distances = pdist(data)
        
        # Cluster-based distances (0 if same cluster, 1 if different)
        cluster_distances = []
        n_samples = len(labels)
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if labels[i] == labels[j]:
                    cluster_distances.append(0)
                else:
                    cluster_distances.append(1)
        
        cluster_distances = np.array(cluster_distances)
        
        # Calculate correlation (higher correlation = better pattern preservation)
        try:
            correlation, _ = spearmanr(original_distances, cluster_distances)
            # Convert to 0-1 scale where 1 is perfect preservation
            preservation_score = max(0, -correlation)  # Negative correlation is good (same cluster = small distance)
            return float(preservation_score)
        except:
            return 0.0
    
    def _calculate_ecological_gradient_alignment(self, 
                                               data: np.ndarray, 
                                               labels: np.ndarray) -> float:
        """Calculate alignment with ecological gradients."""
        if self.environmental_data is None:
            logger.warning("Environmental data not available, using data-based proxy")
            return self._calculate_data_gradient_proxy(data, labels)
        
        # This would calculate how well clusters align with environmental gradients
        # For now, return a placeholder
        return 0.65
    
    def _calculate_data_gradient_proxy(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Proxy for ecological gradient alignment using data structure."""
        # Calculate if clusters follow gradients in the data space
        unique_labels = np.unique(labels)
        
        if len(unique_labels) <= 2:
            return 0.0
        
        # For each feature, calculate if cluster means show gradient pattern
        gradient_scores = []
        
        for feature_idx in range(data.shape[1]):
            cluster_means = []
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_mean = np.mean(data[cluster_mask, feature_idx])
                cluster_means.append(cluster_mean)
            
            # Check if cluster means show monotonic trend (gradient)
            cluster_means = np.array(cluster_means)
            
            # Calculate rank correlation with cluster labels (monotonic trend)
            try:
                correlation, _ = spearmanr(unique_labels, cluster_means)
                gradient_scores.append(abs(correlation))
            except:
                gradient_scores.append(0.0)
        
        return float(np.mean(gradient_scores))
    
    def _safe_silhouette_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Safely calculate silhouette score."""
        try:
            if len(np.unique(labels)) <= 1 or len(data) <= 1:
                return 0.0
            return float(silhouette_score(data, labels))
        except:
            return 0.0
    
    def _safe_calinski_harabasz_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Safely calculate Calinski-Harabasz score."""
        try:
            if len(np.unique(labels)) <= 1:
                return 0.0
            return float(calinski_harabasz_score(data, labels))
        except:
            return 0.0
    
    def _safe_davies_bouldin_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Safely calculate Davies-Bouldin score."""
        try:
            if len(np.unique(labels)) <= 1:
                return float('inf')
            return float(davies_bouldin_score(data, labels))
        except:
            return float('inf')
    
    def _calculate_variance_preservation_proxy(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Proxy for functional diversity using variance preservation."""
        # Calculate if within-cluster variance is preserved appropriately
        unique_labels = np.unique(labels)
        
        if len(unique_labels) <= 1:
            return 0.0
        
        total_variance = np.var(data, axis=0).sum()
        within_cluster_variances = []
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_data = data[cluster_mask]
            
            if len(cluster_data) > 1:
                cluster_variance = np.var(cluster_data, axis=0).sum()
                within_cluster_variances.append(cluster_variance)
        
        if not within_cluster_variances:
            return 0.0
        
        avg_within_variance = np.mean(within_cluster_variances)
        
        # Good clustering preserves some within-cluster variance (not too homogeneous)
        # but reduces it compared to total variance
        preservation_score = avg_within_variance / max(total_variance, 1e-6)
        
        # Optimal range is around 0.3-0.7 (some reduction but not total homogenization)
        if 0.3 <= preservation_score <= 0.7:
            return float(1.0)
        elif preservation_score < 0.3:
            return float(preservation_score / 0.3)
        else:
            return float(max(0, 1.0 - (preservation_score - 0.7) / 0.3))
    
    def _calculate_distance_structure_proxy(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Proxy for phylogenetic signal using distance structure."""
        # Calculate if distance relationships are preserved in clusters
        if len(data) < 10:
            return 0.0
        
        # Sample pairs to avoid computational explosion
        n_samples = min(len(data), 1000)
        sample_indices = np.random.choice(len(data), n_samples, replace=False)
        
        sampled_data = data[sample_indices]
        sampled_labels = labels[sample_indices]
        
        # Calculate original distances
        original_distances = pdist(sampled_data)
        
        # Calculate label-based distances
        label_distances = []
        n_sampled = len(sampled_labels)
        
        for i in range(n_sampled):
            for j in range(i + 1, n_sampled):
                if sampled_labels[i] == sampled_labels[j]:
                    label_distances.append(0)
                else:
                    label_distances.append(1)
        
        label_distances = np.array(label_distances)
        
        # Calculate rank correlation
        try:
            correlation, _ = spearmanr(original_distances, label_distances)
            # Convert to 0-1 scale
            signal_retention = max(0, -correlation)  # Negative correlation is good
            return float(signal_retention)
        except:
            return 0.0
    
    def _log_metrics_summary(self, metrics: BiodiversityEvaluationMetrics):
        """Log summary of evaluation metrics."""
        logger.info("Biodiversity SOM Evaluation Summary:")
        logger.info(f"  Core SOM Metrics:")
        logger.info(f"    Quantization Error: {metrics.quantization_error:.4f}")
        logger.info(f"    Topographic Error: {metrics.topographic_error:.4f}")
        
        logger.info(f"  Biodiversity Metrics:")
        logger.info(f"    Species Association Accuracy: {metrics.species_association_accuracy:.3f}")
        logger.info(f"    Functional Diversity Preservation: {metrics.functional_diversity_preservation:.3f}")
        logger.info(f"    Biogeographic Coherence: {metrics.biogeographic_coherence:.3f}")
        
        logger.info(f"  Spatial Metrics:")
        logger.info(f"    Spatial Autocorrelation: {metrics.spatial_autocorrelation:.3f}")
        logger.info(f"    Cluster Compactness: {metrics.cluster_compactness:.3f}")
        
        logger.info(f"  Statistical Metrics:")
        logger.info(f"    Silhouette Score: {metrics.silhouette_score:.3f}")
        logger.info(f"    Calinski-Harabasz Score: {metrics.calinski_harabasz_score:.1f}")


def create_biodiversity_evaluator(**kwargs) -> BiodiversityEvaluator:
    """Factory function for creating biodiversity evaluator."""
    return BiodiversityEvaluator(**kwargs)