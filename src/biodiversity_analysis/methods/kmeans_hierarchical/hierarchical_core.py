"""Two-stage K-means → Hierarchical clustering core implementation.

Clean implementation with proper abstraction and no redundancy.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import logging

logger = logging.getLogger(__name__)


class TwoStageClusterer:
    """Two-stage clustering: K-means pre-clustering → Hierarchical clustering.
    
    This approach enables hierarchical clustering on large datasets by first
    reducing data to cluster centroids via k-means.
    """
    
    def __init__(self, 
                 n_pre_clusters: int = 750,
                 linkage_method: str = 'ward',
                 final_k: Optional[int] = None,
                 batch_size: int = 10000,
                 random_state: int = 42):
        """Initialize two-stage clusterer.
        
        Args:
            n_pre_clusters: Number of k-means clusters (stage 1)
            linkage_method: Hierarchical clustering linkage method
            final_k: Number of final clusters (if None, will be determined)
            batch_size: Batch size for MiniBatchKMeans
            random_state: Random seed for reproducibility
        """
        self.n_pre_clusters = n_pre_clusters
        self.linkage_method = linkage_method
        self.final_k = final_k
        self.batch_size = batch_size
        self.random_state = random_state
        
        # Components
        self.kmeans = None
        self.linkage_matrix = None
        self.final_labels_ = None
        self.cluster_centers_ = None
        
        # Metadata
        self.n_samples_ = None
        self.n_features_ = None
        self.variance_preserved_ = None
        
    def fit(self, X: np.ndarray) -> 'TwoStageClusterer':
        """Fit two-stage clustering.
        
        Args:
            X: Data array of shape (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        self.n_samples_, self.n_features_ = X.shape
        logger.info(f"Fitting two-stage clustering on {self.n_samples_:,} samples")
        
        # Stage 1: K-means pre-clustering
        self._fit_kmeans(X)
        
        # Stage 2: Hierarchical clustering on centroids
        self._fit_hierarchical()
        
        # Stage 3: Map back to original samples
        self._map_labels(X)
        
        return self
        
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels.
        
        Args:
            X: Data array of shape (n_samples, n_features)
            
        Returns:
            Cluster labels for each sample
        """
        self.fit(X)
        return self.final_labels_
        
    def _fit_kmeans(self, X: np.ndarray) -> None:
        """Stage 1: Fit k-means clustering."""
        logger.info(f"Stage 1: K-means with k={self.n_pre_clusters}")
        
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_pre_clusters,
            batch_size=self.batch_size,
            init='k-means++',
            n_init=3,
            random_state=self.random_state,
            verbose=0
        )
        
        self.kmeans.fit(X)
        
        # Calculate variance preserved
        self._calculate_variance_preserved(X)
        logger.info(f"K-means complete. Variance preserved: {self.variance_preserved_:.2%}")
        
    def _fit_hierarchical(self) -> None:
        """Stage 2: Hierarchical clustering on k-means centroids."""
        logger.info(f"Stage 2: Hierarchical clustering with {self.linkage_method} linkage")
        
        centroids = self.kmeans.cluster_centers_
        self.linkage_matrix = linkage(centroids, method=self.linkage_method)
        
        # Determine final k if not specified
        if self.final_k is None:
            self.final_k = self._determine_optimal_k(centroids)
            logger.info(f"Determined optimal k={self.final_k}")
        
        # Get cluster assignments for centroids
        self.centroid_labels_ = fcluster(
            self.linkage_matrix, 
            self.final_k, 
            criterion='maxclust'
        ) - 1  # Convert to 0-indexed
        
    def _map_labels(self, X: np.ndarray) -> None:
        """Stage 3: Map hierarchical labels back to original samples."""
        logger.info("Stage 3: Mapping labels to original samples")
        
        # Get k-means labels for all samples
        kmeans_labels = self.kmeans.labels_
        
        # Map through hierarchical labels
        self.final_labels_ = self.centroid_labels_[kmeans_labels]
        
        # Calculate final cluster centers
        self._calculate_cluster_centers(X)
        
    def _calculate_variance_preserved(self, X: np.ndarray) -> None:
        """Calculate variance preserved by k-means."""
        # Reconstruct data from centroids
        reconstructed = self.kmeans.cluster_centers_[self.kmeans.labels_]
        
        # Calculate variance
        total_var = np.var(X)
        residual_var = np.var(X - reconstructed)
        
        self.variance_preserved_ = 1 - (residual_var / total_var)
        
    def _determine_optimal_k(self, centroids: np.ndarray) -> int:
        """Determine optimal number of final clusters.
        
        Simple implementation using silhouette score.
        Can be extended with bootstrap stability.
        """
        k_range = range(4, min(9, len(centroids) // 10))
        scores = []
        
        for k in k_range:
            labels = fcluster(self.linkage_matrix, k, criterion='maxclust')
            score = silhouette_score(centroids, labels)
            scores.append(score)
            
        optimal_k = k_range[np.argmax(scores)]
        return optimal_k
        
    def _calculate_cluster_centers(self, X: np.ndarray) -> None:
        """Calculate final cluster centers."""
        n_clusters = len(np.unique(self.final_labels_))
        self.cluster_centers_ = np.zeros((n_clusters, self.n_features_))
        
        for k in range(n_clusters):
            mask = self.final_labels_ == k
            if mask.any():
                self.cluster_centers_[k] = X[mask].mean(axis=0)


class VariancePreservingKMeans:
    """K-means with automatic k selection based on variance preservation.
    
    Determines the minimum k that preserves a specified fraction of variance.
    """
    
    def __init__(self, 
                 variance_threshold: float = 0.95,
                 k_range: Optional[Tuple[int, int]] = None,
                 batch_size: int = 10000):
        """Initialize variance-preserving k-means.
        
        Args:
            variance_threshold: Minimum variance to preserve (0-1)
            k_range: Range of k values to test (min, max)
            batch_size: Batch size for MiniBatchKMeans
        """
        self.variance_threshold = variance_threshold
        self.k_range = k_range or (100, 2000)
        self.batch_size = batch_size
        self.optimal_k_ = None
        self.variance_curve_ = {}
        
    def determine_k(self, X: np.ndarray, sample_size: int = 100000) -> int:
        """Determine optimal k for variance preservation.
        
        Args:
            X: Data array
            sample_size: Sample size for testing (for efficiency)
            
        Returns:
            Optimal number of clusters
        """
        # Sample data if too large
        if len(X) > sample_size:
            idx = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X
            
        # Test k values
        k_values = self._get_k_values()
        
        for k in k_values:
            logger.debug(f"Testing k={k}")
            
            kmeans = MiniBatchKMeans(
                n_clusters=k,
                batch_size=self.batch_size,
                n_init=1,  # Single run for speed
                random_state=42
            )
            
            kmeans.fit(X_sample)
            
            # Calculate variance preserved
            reconstructed = kmeans.cluster_centers_[kmeans.labels_]
            total_var = np.var(X_sample)
            residual_var = np.var(X_sample - reconstructed)
            variance_preserved = 1 - (residual_var / total_var)
            
            self.variance_curve_[k] = variance_preserved
            logger.debug(f"k={k}: {variance_preserved:.3f} variance preserved")
            
            if variance_preserved >= self.variance_threshold:
                self.optimal_k_ = k
                logger.info(f"Optimal k={k} preserves {variance_preserved:.3f} variance")
                return k
                
        # If threshold not met, use maximum k
        self.optimal_k_ = k_values[-1]
        logger.warning(f"Variance threshold not met. Using k={self.optimal_k_}")
        return self.optimal_k_
        
    def _get_k_values(self) -> list:
        """Get k values to test."""
        # Exponential spacing for efficiency
        min_k, max_k = self.k_range
        k_values = []
        k = min_k
        while k <= max_k:
            k_values.append(k)
            k = int(k * 1.5)  # Exponential growth
        
        # Ensure max_k is included
        if k_values[-1] < max_k:
            k_values.append(max_k)
            
        return k_values


class BootstrapStability:
    """Bootstrap stability assessment for cluster validation.
    
    Tests clustering stability through resampling.
    """
    
    def __init__(self, n_bootstrap: int = 30, sample_ratio: float = 0.9):
        """Initialize bootstrap stability tester.
        
        Args:
            n_bootstrap: Number of bootstrap iterations
            sample_ratio: Fraction of data to sample each iteration
        """
        self.n_bootstrap = n_bootstrap
        self.sample_ratio = sample_ratio
        self.stability_scores_ = {}
        
    def evaluate(self, 
                 centroids: np.ndarray,
                 linkage_matrix: np.ndarray,
                 k_range: Tuple[int, int] = (4, 8)) -> int:
        """Evaluate stability for different k values.
        
        Args:
            centroids: Cluster centroids from k-means
            linkage_matrix: Linkage matrix from hierarchical clustering
            k_range: Range of k values to test
            
        Returns:
            Optimal k with highest stability
        """
        k_values = range(k_range[0], k_range[1] + 1)
        
        for k in k_values:
            logger.debug(f"Testing stability for k={k}")
            
            # Original clustering
            orig_labels = fcluster(linkage_matrix, k, criterion='maxclust')
            
            # Bootstrap iterations
            jaccard_scores = []
            
            for i in range(self.n_bootstrap):
                # Resample centroids
                n_samples = int(len(centroids) * self.sample_ratio)
                idx = np.random.choice(len(centroids), n_samples, replace=False)
                boot_centroids = centroids[idx]
                
                # Re-cluster
                boot_linkage = linkage(boot_centroids, method='ward')
                boot_labels = fcluster(boot_linkage, k, criterion='maxclust')
                
                # Calculate Jaccard similarity
                jaccard = self._jaccard_similarity(
                    orig_labels[idx], 
                    boot_labels
                )
                jaccard_scores.append(jaccard)
                
            stability = np.mean(jaccard_scores)
            self.stability_scores_[k] = stability
            logger.debug(f"k={k}: stability={stability:.3f}")
            
        # Return k with highest stability
        optimal_k = max(self.stability_scores_, key=self.stability_scores_.get)
        logger.info(f"Optimal k={optimal_k} with stability={self.stability_scores_[optimal_k]:.3f}")
        
        return optimal_k
        
    def _jaccard_similarity(self, labels1: np.ndarray, labels2: np.ndarray) -> float:
        """Calculate Jaccard similarity between two clusterings."""
        n = len(labels1)
        
        # Count pairs in same cluster
        same_cluster1 = np.zeros((n, n), dtype=bool)
        same_cluster2 = np.zeros((n, n), dtype=bool)
        
        for i in range(n):
            same_cluster1[i] = (labels1 == labels1[i])
            same_cluster2[i] = (labels2 == labels2[i])
            
        # Jaccard = intersection / union
        intersection = np.sum(same_cluster1 & same_cluster2)
        union = np.sum(same_cluster1 | same_cluster2)
        
        return intersection / union if union > 0 else 0.0


def calculate_clustering_metrics(X: np.ndarray, 
                                labels: np.ndarray,
                                sample_size: int = 50000) -> Dict[str, float]:
    """Calculate clustering quality metrics.
    
    Args:
        X: Data array
        labels: Cluster labels
        sample_size: Max samples for expensive metrics
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {}
    
    # Sample for expensive metrics if needed
    if len(X) > sample_size:
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[idx]
        labels_sample = labels[idx]
    else:
        X_sample = X
        labels_sample = labels
        
    # Silhouette score (expensive)
    if len(np.unique(labels_sample)) > 1:
        metrics['silhouette'] = float(silhouette_score(X_sample, labels_sample))
    
    # Davies-Bouldin score (moderate cost)
    metrics['davies_bouldin'] = float(davies_bouldin_score(X, labels))
    
    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    metrics['n_clusters'] = len(unique)
    metrics['min_cluster_size'] = int(counts.min())
    metrics['max_cluster_size'] = int(counts.max())
    metrics['median_cluster_size'] = float(np.median(counts))
    
    return metrics