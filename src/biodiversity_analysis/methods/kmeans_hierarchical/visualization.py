"""Visualization utilities for k-means clustering results."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class KMeansVisualizer:
    """Visualization tools for k-means clustering results."""
    
    @staticmethod
    def plot_cluster_profiles(features: np.ndarray, labels: np.ndarray, 
                            feature_names: Optional[list] = None,
                            figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Plot feature profiles for each cluster.
        
        Args:
            features: Feature data
            labels: Cluster assignments
            feature_names: Names of features
            figsize: Figure size
            
        Returns:
            fig: Matplotlib figure
        """
        n_clusters = len(np.unique(labels[labels >= 0]))
        n_features = features.shape[1]
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(n_features)]
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
        
        for i, (ax, fname) in enumerate(zip(axes, feature_names)):
            cluster_values = []
            cluster_labels = []
            
            for k in range(n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    values = features[mask, i]
                    valid_values = values[~np.isnan(values)]
                    if len(valid_values) > 0:
                        cluster_values.append(valid_values)
                        cluster_labels.append(f'Cluster {k}')
            
            if cluster_values:
                ax.boxplot(cluster_values, labels=cluster_labels)
                ax.set_ylabel(fname)
                ax.set_xlabel('Cluster')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
        
        fig.suptitle('Feature Distributions by Cluster', fontsize=14)
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_geographic_clusters(coordinates: np.ndarray, labels: np.ndarray,
                               figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
        """Plot geographic distribution of clusters.
        
        Args:
            coordinates: Geographic coordinates [lat, lon]
            labels: Cluster assignments
            figsize: Figure size
            
        Returns:
            fig: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create colormap
        n_clusters = len(np.unique(labels[labels >= 0]))
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
        
        # Plot each cluster
        for k in range(n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                ax.scatter(coordinates[mask, 1], coordinates[mask, 0],
                          c=[colors[k]], label=f'Cluster {k}',
                          s=10, alpha=0.6)
        
        # Plot outliers if any
        outlier_mask = labels == -1
        if outlier_mask.sum() > 0:
            ax.scatter(coordinates[outlier_mask, 1], coordinates[outlier_mask, 0],
                      c='gray', label='Outliers', s=5, alpha=0.3)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Geographic Distribution of Clusters')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_cluster_quality(quality_metrics: Dict[str, Any],
                           figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """Plot cluster quality metrics.
        
        Args:
            quality_metrics: Dictionary of quality metrics
            figsize: Figure size
            
        Returns:
            fig: Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot silhouette scores by cluster if available
        if 'silhouette_by_cluster' in quality_metrics:
            ax = axes[0]
            scores = quality_metrics['silhouette_by_cluster']
            clusters = list(scores.keys())
            values = list(scores.values())
            
            ax.bar(clusters, values)
            ax.axhline(y=quality_metrics.get('mean_silhouette', 0), 
                      color='r', linestyle='--', label='Mean')
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Silhouette Score')
            ax.set_title('Silhouette Scores by Cluster')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot cluster sizes
        if 'cluster_sizes' in quality_metrics:
            ax = axes[1]
            sizes = quality_metrics['cluster_sizes']
            clusters = list(sizes.keys())
            values = list(sizes.values())
            
            ax.bar(clusters, values)
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Number of Samples')
            ax.set_title('Cluster Sizes')
            ax.grid(True, alpha=0.3)
            
            # Add percentage labels
            total = sum(values)
            for i, (c, v) in enumerate(zip(clusters, values)):
                ax.text(i, v + max(values)*0.01, f'{100*v/total:.1f}%', 
                       ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_elbow_curve(k_values: list, inertias: list,
                        figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """Plot elbow curve for optimal k selection.
        
        Args:
            k_values: List of k values tested
            inertias: Corresponding inertia values
            figsize: Figure size
            
        Returns:
            fig: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Curve for Optimal k')
        ax.grid(True, alpha=0.3)
        
        # Try to identify elbow point
        if len(k_values) > 2:
            # Calculate rate of change
            deltas = np.diff(inertias)
            delta_deltas = np.diff(deltas)
            
            # Find point with maximum curvature
            if len(delta_deltas) > 0:
                elbow_idx = np.argmax(delta_deltas) + 2
                if elbow_idx < len(k_values):
                    ax.axvline(x=k_values[elbow_idx], color='r', 
                             linestyle='--', label=f'Elbow at k={k_values[elbow_idx]}')
                    ax.legend()
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_missing_data_patterns(features: np.ndarray, labels: np.ndarray,
                                 figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """Plot missing data patterns by cluster.
        
        Args:
            features: Feature data
            labels: Cluster assignments
            figsize: Figure size
            
        Returns:
            fig: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        n_clusters = len(np.unique(labels[labels >= 0]))
        n_features = features.shape[1]
        
        # Calculate missing data percentage by cluster and feature
        missing_patterns = np.zeros((n_clusters, n_features))
        
        for k in range(n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                cluster_data = features[mask]
                missing_patterns[k] = np.isnan(cluster_data).mean(axis=0) * 100
        
        # Create heatmap
        sns.heatmap(missing_patterns, 
                   xticklabels=[f'Feature {i}' for i in range(n_features)],
                   yticklabels=[f'Cluster {i}' for i in range(n_clusters)],
                   cmap='YlOrRd', 
                   annot=True, 
                   fmt='.1f',
                   cbar_kws={'label': 'Missing %'},
                   ax=ax)
        
        ax.set_title('Missing Data Patterns by Cluster')
        plt.tight_layout()
        
        return fig