"""Two-stage K-means → Hierarchical clustering core implementation.

Clean implementation with proper abstraction and no redundancy.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from scipy.cluster.hierarchy import linkage, fcluster, cophenet, is_valid_linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


def calculate_variance_preserved(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculate variance preserved after reconstruction.
    
    Args:
        original: Original data array
        reconstructed: Reconstructed data array
        
    Returns:
        Fraction of variance preserved (0-1)
    """
    # Calculate variance properly for multi-dimensional data
    # Sum variance across all dimensions
    total_var = np.var(original, axis=0).sum()
    residual_var = np.var(original - reconstructed, axis=0).sum()
    
    return 1 - (residual_var / total_var) if total_var > 0 else 0.0


class TwoStageClusterer:
    """Two-stage clustering: K-means pre-clustering → Hierarchical clustering.
    
    This approach enables hierarchical clustering on large datasets by first
    reducing data to cluster centroids via k-means.
    """
    
    # Constants for cluster filtering
    MIN_CLUSTER_SIZE = 2  # Minimum samples per cluster
    MIN_CLUSTER_FRACTION = 1 / 10000  # Minimum fraction of total samples (0.01%)
    
    def __init__(self, 
                 n_pre_clusters: int = 750,
                 linkage_method: str = 'ward',
                 distance_metric: str = 'euclidean',
                 final_k: Optional[int] = None,
                 batch_size: int = 10000,
                 random_state: int = 42):
        """Initialize two-stage clusterer.
        
        Args:
            n_pre_clusters: Number of k-means clusters (stage 1)
            linkage_method: Hierarchical clustering linkage method
            distance_metric: Distance metric for hierarchical clustering
            final_k: Number of final clusters (if None, will be determined)
            batch_size: Batch size for MiniBatchKMeans
            random_state: Random seed for reproducibility
        """
        self.n_pre_clusters = n_pre_clusters
        self.linkage_method = linkage_method
        self.distance_metric = distance_metric
        self.final_k = final_k
        self.batch_size = batch_size
        self.random_state = random_state
        
        # Components
        self.kmeans = None
        self.linkage_matrix = None
        self.final_labels_ = None
        self.cluster_centers_ = None
        self.filtered_centroids_ = None  # Store filtered centroids for bootstrap
        
        # Metadata
        self.n_samples_ = None
        self.n_features_ = None
        self.variance_preserved_ = None
        self.cophenetic_correlation_ = None
        self.scaler = None  # For data standardization
        
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
        
        # Standardize data for better clustering
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_pre_clusters,
            batch_size=self.batch_size,
            init='k-means++',
            n_init='auto',  # Will use 1 with k-means++ for memory efficiency
            random_state=self.random_state,
            verbose=0
        )
        
        self.kmeans.fit(X_scaled)
        
        # Calculate variance preserved
        self._calculate_variance_preserved(X)
        logger.info(f"K-means complete. Variance preserved: {self.variance_preserved_:.2%}")
        
    def _fit_hierarchical(self) -> None:
        """Stage 2: Hierarchical clustering on k-means centroids."""
        logger.info(f"Stage 2: Hierarchical clustering with {self.linkage_method} linkage")
        
        centroids = self.kmeans.cluster_centers_
        
        # Filter out tiny clusters if any (can cause issues)
        cluster_sizes = np.bincount(self.kmeans.labels_)
        min_size = max(self.MIN_CLUSTER_SIZE, int(self.n_samples_ * self.MIN_CLUSTER_FRACTION))
        valid_clusters = cluster_sizes >= min_size
        
        # Create mapping for filtered clusters
        self._cluster_mapping = None
        if not all(valid_clusters):
            logger.warning(f"Filtering out {(~valid_clusters).sum()} tiny clusters")
            centroids = centroids[valid_clusters]
            
            # Create mapping from original cluster IDs to new IDs
            self._cluster_mapping = np.full(len(valid_clusters), -1, dtype=int)
            new_id = 0
            for old_id, is_valid in enumerate(valid_clusters):
                if is_valid:
                    self._cluster_mapping[old_id] = new_id
                    new_id += 1
        
        # Store filtered centroids for bootstrap
        self.filtered_centroids_ = centroids
        
        # Use configured distance metric (with validation for ward)
        if self.linkage_method == 'ward':
            if self.distance_metric != 'euclidean':
                logger.warning(f"Ward linkage requires euclidean distance, ignoring {self.distance_metric}")
            metric = 'euclidean'
        else:
            metric = self.distance_metric
            
        self.linkage_matrix = linkage(centroids, method=self.linkage_method, metric=metric)
        
        # Validate linkage matrix
        if not is_valid_linkage(self.linkage_matrix, warning=True):
            logger.warning("Linkage matrix validation failed")
        
        # Calculate cophenetic correlation for quality assessment
        c, coph_dists = cophenet(self.linkage_matrix, pdist(centroids, metric=metric))
        self.cophenetic_correlation_ = c
        logger.info(f"Cophenetic correlation: {c:.3f} (>0.7 is good)")
        
        # Determine final k if not specified
        if self.final_k is None:
            self.final_k = self._determine_optimal_k(centroids)
            logger.info(f"Determined optimal k={self.final_k}")
        
        # Get cluster assignments for centroids
        # Note: fcluster is 1-indexed, we convert to 0-indexed
        self.centroid_labels_ = fcluster(
            self.linkage_matrix, 
            self.final_k, 
            criterion='maxclust'
        ) - 1  # Convert to 0-indexed
        
        # Verify we got the expected number of clusters
        # This is a known scipy bug where fcluster may return fewer clusters
        actual_k = len(np.unique(self.centroid_labels_))
        if actual_k != self.final_k:
            logger.warning(f"Requested {self.final_k} clusters but fcluster returned {actual_k}")
            logger.warning("This is a known scipy bug with fcluster. Adjusting to actual count.")
            self.final_k = actual_k  # Update to actual
        
    def _map_labels(self, X: np.ndarray) -> None:
        """Stage 3: Map hierarchical labels back to original samples."""
        logger.info("Stage 3: Mapping labels to original samples")
        
        # Get k-means labels for all samples
        kmeans_labels = self.kmeans.labels_
        
        # If we filtered clusters, map k-means labels through the mapping
        if self._cluster_mapping is not None:
            # Map original cluster IDs to new filtered IDs
            mapped_kmeans_labels = self._cluster_mapping[kmeans_labels]
            
            # Handle samples in filtered clusters (marked as -1)
            valid_mask = mapped_kmeans_labels >= 0
            
            # Initialize final labels with -1 for filtered samples
            self.final_labels_ = np.full(len(kmeans_labels), -1, dtype=int)
            
            # Map valid samples through hierarchical labels
            self.final_labels_[valid_mask] = self.centroid_labels_[mapped_kmeans_labels[valid_mask]]
        else:
            # No filtering, direct mapping
            self.final_labels_ = self.centroid_labels_[kmeans_labels]
        
        # Calculate final cluster centers
        self._calculate_cluster_centers(X)
        
    def _calculate_variance_preserved(self, X: np.ndarray) -> None:
        """Calculate variance preserved by k-means."""
        # Reconstruct data from centroids (in original scale)
        X_scaled = self.scaler.transform(X)
        reconstructed_scaled = self.kmeans.cluster_centers_[self.kmeans.labels_]
        reconstructed = self.scaler.inverse_transform(reconstructed_scaled)
        
        # Use shared variance calculation method
        self.variance_preserved_ = calculate_variance_preserved(X, reconstructed)
        
    def _determine_optimal_k(self, centroids: np.ndarray) -> int:
        """Determine optimal number of final clusters.
        
        Simple implementation using silhouette score.
        Can be extended with bootstrap stability.
        """
        # More flexible k range based on number of centroids
        max_k = min(20, max(9, len(centroids) // 20))  # Allow up to 20 clusters
        k_range = range(4, max_k)
        scores = []
        max_retries = 3  # Prevent infinite loops
        
        for k in k_range:
            retry_count = 0
            while retry_count < max_retries:
                labels = fcluster(self.linkage_matrix, k, criterion='maxclust')
                # Verify we got expected clusters (handle scipy bug)
                actual_k = len(np.unique(labels))
                if actual_k < k:
                    logger.debug(f"fcluster returned {actual_k} clusters instead of {k} (attempt {retry_count + 1}/{max_retries})")
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.warning(f"Skipping k={k} after {max_retries} attempts")
                        break
                else:
                    score = silhouette_score(centroids, labels)
                    scores.append(score)
                    break
            
        if not scores:
            raise ValueError(
                "Could not determine optimal k: no valid clustering found. "
                "This may indicate data issues or too restrictive parameters."
            )
            
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


class ElbowMethodKMeans:
    """K-means with automatic k selection using elbow method.
    
    Finds optimal k where WCSS improvement becomes marginal.
    """
    
    def __init__(self, 
                 k_range: Optional[Tuple[int, int]] = None,
                 batch_size: int = 10000,
                 elbow_threshold: float = 0.01,
                 save_elbow_plot: bool = True):
        """Initialize elbow method k-means.
        
        Args:
            k_range: Range of k values to test (min, max)
            batch_size: Batch size for MiniBatchKMeans
            elbow_threshold: Relative improvement threshold (default 1%)
            save_elbow_plot: Whether to save elbow curve plot
        """
        self.k_range = k_range or (100, 2000)
        self.batch_size = batch_size
        self.elbow_threshold = elbow_threshold
        self.save_elbow_plot = save_elbow_plot
        self.optimal_k_ = None
        self.wcss_curve_ = {}
        self.variance_curve_ = {}
        
    def determine_k(self, X: np.ndarray, sample_size: int = 100000) -> int:
        """Determine optimal k using elbow method.
        
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
            
        # Get evenly spaced k values for elbow method
        k_values = self._get_k_values()
        logger.info(f"Testing k values for elbow method: {k_values}")
        
        # Test ALL k values (don't stop early!)
        for k in k_values:
            logger.debug(f"Testing k={k}")
            
            kmeans = MiniBatchKMeans(
                n_clusters=k,
                batch_size=self.batch_size,
                n_init='auto',
                random_state=42
            )
            
            kmeans.fit(X_sample)
            
            # Calculate WCSS and variance
            wcss = kmeans.inertia_
            reconstructed = kmeans.cluster_centers_[kmeans.labels_]
            variance_preserved = calculate_variance_preserved(X_sample, reconstructed)
            
            self.wcss_curve_[k] = wcss
            self.variance_curve_[k] = variance_preserved
            logger.info(f"k={k}: WCSS={wcss:.2f}, variance={variance_preserved:.3f}")
        
        # Find elbow point
        self.optimal_k_ = self._find_elbow_point()
        logger.info(f"Elbow method selected k={self.optimal_k_} (variance={self.variance_curve_[self.optimal_k_]:.3f})")
        
        # Save elbow plot if requested
        if self.save_elbow_plot:
            self._save_elbow_plot()
        
        return self.optimal_k_
        
    def _get_k_values(self) -> list:
        """Get evenly spaced k values for elbow method."""
        min_k, max_k = self.k_range
        
        # Determine step size based on range
        if max_k - min_k > 5000:
            step = 1000  # Test every 1000 for large ranges
        elif max_k - min_k > 1000:
            step = 500   # Test every 500 for medium ranges
        elif max_k - min_k > 500:
            step = 100   # Test every 100 for smaller ranges
        else:
            step = 50    # Test every 50 for small ranges
            
        k_values = list(range(min_k, max_k + 1, step))
        
        # Ensure max_k is included
        if k_values[-1] < max_k:
            k_values.append(max_k)
            
        return k_values
    
    def _find_elbow_point(self) -> int:
        """Find the elbow point using kneedle algorithm (maximum curvature)."""
        k_values = sorted(self.wcss_curve_.keys())
        wcss_values = [self.wcss_curve_[k] for k in k_values]
        
        if len(k_values) < 3:
            return k_values[-1]
        
        # Normalize both axes to [0, 1] for fair comparison
        k_norm = np.array(k_values, dtype=float)
        k_norm = (k_norm - k_norm.min()) / (k_norm.max() - k_norm.min())
        
        wcss_norm = np.array(wcss_values, dtype=float)
        wcss_norm = (wcss_norm - wcss_norm.min()) / (wcss_norm.max() - wcss_norm.min())
        
        # Method 1: Maximum distance from line connecting first and last points
        # This is the "kneedle" algorithm
        first_point = np.array([k_norm[0], wcss_norm[0]])
        last_point = np.array([k_norm[-1], wcss_norm[-1]])
        line_vec = last_point - first_point
        line_length = np.linalg.norm(line_vec)
        
        if line_length > 0:
            max_distance = 0
            elbow_idx = 0
            
            for i in range(1, len(k_norm) - 1):
                point = np.array([k_norm[i], wcss_norm[i]])
                # Calculate perpendicular distance to line
                distance = np.abs(np.cross(line_vec, first_point - point)) / line_length
                
                if distance > max_distance:
                    max_distance = distance
                    elbow_idx = i
            
            logger.info(f"Kneedle algorithm found elbow at k={k_values[elbow_idx]} (distance={max_distance:.3f})")
            
            # Method 2: Also calculate rate of change as secondary criterion
            improvements = []
            for i in range(1, len(wcss_values)):
                if wcss_values[i-1] > 0:
                    relative_improvement = (wcss_values[i-1] - wcss_values[i]) / wcss_values[i-1]
                    improvements.append((k_values[i], relative_improvement))
            
            # Log the improvements for transparency
            for k, imp in improvements[:5]:  # Show first 5
                logger.debug(f"k={k}: {imp:.1%} improvement")
            
            # If kneedle gives very low k, check if we have sufficient variance
            if self.variance_curve_[k_values[elbow_idx]] < 0.99:
                # Find smallest k with 99% variance
                for k in k_values:
                    if self.variance_curve_[k] >= 0.99:
                        logger.info(f"Adjusting to k={k} for 99% variance (kneedle suggested {k_values[elbow_idx]})")
                        return k
            
            return k_values[elbow_idx]
        
        # Fallback: use variance criterion
        for k in k_values:
            if self.variance_curve_[k] >= 0.99:
                logger.info(f"Using k={k} with 99% variance (line detection failed)")
                return k
        
        return k_values[-1]
    
    def _save_elbow_plot(self) -> None:
        """Save elbow curve plot for k selection visualization."""
        try:
            import matplotlib.pyplot as plt
            
            k_values = sorted(self.wcss_curve_.keys())
            wcss_values = [self.wcss_curve_[k] for k in k_values]
            variance_values = [self.variance_curve_[k] for k in k_values]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot WCSS
            ax1.plot(k_values, wcss_values, 'b-o', markersize=6)
            ax1.axvline(x=self.optimal_k_, color='r', linestyle='--', 
                       label=f'Selected k={self.optimal_k_}')
            ax1.set_xlabel('Number of Clusters (k)')
            ax1.set_ylabel('Within-Cluster Sum of Squares')
            ax1.set_title('Elbow Method: WCSS vs k')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot Variance Preserved
            ax2.plot(k_values, variance_values, 'g-o', markersize=6)
            ax2.axvline(x=self.optimal_k_, color='r', linestyle='--',
                       label=f'Selected k={self.optimal_k_}')
            ax2.axhline(y=0.99, color='orange', linestyle=':', 
                       label='99% variance threshold')
            ax2.set_xlabel('Number of Clusters (k)')
            ax2.set_ylabel('Variance Preserved')
            ax2.set_title('Variance Preservation vs k')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.suptitle('K-means Pre-cluster Selection Analysis', fontsize=14)
            plt.tight_layout()
            
            # Save to outputs directory
            import os
            output_dir = 'outputs/analysis_results/kmeans_hierarchical/production_kmeans_hierarchical'
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, 'elbow_curve.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved elbow curve plot to {plot_path}")
            
        except Exception as e:
            logger.warning(f"Could not save elbow plot: {e}")


class BootstrapStability:
    """Bootstrap stability assessment for cluster validation.
    
    Tests clustering stability through resampling.
    """
    
    def __init__(self, n_bootstrap: int = 30, sample_ratio: float = 0.9, linkage_method: str = 'ward'):
        """Initialize bootstrap stability tester.
        
        Args:
            n_bootstrap: Number of bootstrap iterations
            sample_ratio: Fraction of data to sample each iteration
            linkage_method: Linkage method to use for bootstrap clustering
        """
        self.n_bootstrap = n_bootstrap
        self.sample_ratio = sample_ratio
        self.linkage_method = linkage_method
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
                
                # Re-cluster using same method as main clustering
                boot_linkage = linkage(boot_centroids, method=self.linkage_method)
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
        """Calculate Jaccard similarity between two clusterings.
        
        Memory-efficient implementation using pairwise comparisons.
        """
        n = len(labels1)
        
        # More memory-efficient approach for large datasets
        if n > 1000:  # Use sampling for large datasets
            sample_size = min(1000, n)
            idx = np.random.choice(n, sample_size, replace=False)
            labels1_sample = labels1[idx]
            labels2_sample = labels2[idx]
            
            # Count pairs in same cluster
            intersection = 0
            union = 0
            
            for i in range(len(idx)):
                for j in range(i+1, len(idx)):
                    same1 = labels1_sample[i] == labels1_sample[j]
                    same2 = labels2_sample[i] == labels2_sample[j]
                    if same1 or same2:
                        union += 1
                        if same1 and same2:
                            intersection += 1
        else:
            # Original approach for small datasets
            same_cluster1 = np.zeros((n, n), dtype=bool)
            same_cluster2 = np.zeros((n, n), dtype=bool)
            
            for i in range(n):
                same_cluster1[i] = (labels1 == labels1[i])
                same_cluster2[i] = (labels2 == labels2[i])
                
            intersection = np.sum(same_cluster1 & same_cluster2)
            union = np.sum(same_cluster1 | same_cluster2)
        
        return intersection / union if union > 0 else 0.0


def calculate_clustering_metrics(X: np.ndarray, 
                                labels: np.ndarray,
                                sample_size: int = 50000,
                                method_name: str = 'kmeans_hierarchical') -> Dict[str, float]:
    """Calculate clustering quality metrics.
    
    Args:
        X: Data array
        labels: Cluster labels
        sample_size: Max samples for expensive metrics
        method_name: Name of clustering method for tracking
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {'analysis_method': method_name}
    
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