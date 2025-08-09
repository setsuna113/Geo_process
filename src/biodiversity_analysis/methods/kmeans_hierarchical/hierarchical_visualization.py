"""Visualization module for two-stage hierarchical clustering.

Extends base visualizations with hierarchical-specific plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from typing import Optional, Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)


class HierarchicalVisualizer:
    """Extended visualizer for two-stage clustering with heavy visualization focus."""
    
    def plot_dendrogram(self, linkage_matrix: np.ndarray, optimal_k: int,
                       truncate_level: int = 30) -> plt.Figure:
        """Generate hierarchical dendrogram.
        
        Args:
            linkage_matrix: Scipy linkage matrix
            optimal_k: Number of clusters (for cut line)
            truncate_level: Number of levels to show
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Calculate cut height for optimal_k
        from scipy.cluster.hierarchy import fcluster
        cut_height = self._get_cut_height(linkage_matrix, optimal_k)
        
        # Plot dendrogram
        dendrogram(
            linkage_matrix,
            truncate_mode='lastp',
            p=truncate_level,
            orientation='top',
            leaf_font_size=10,
            show_contracted=True,
            ax=ax
        )
        
        # Add cut line
        ax.axhline(y=cut_height, color='red', linestyle='--', 
                  label=f'Cut for {optimal_k} clusters', linewidth=2)
        
        ax.set_title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
        ax.set_xlabel('Cluster Index or (Cluster Size)', fontsize=12)
        ax.set_ylabel('Distance (Ward Linkage)', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_geographic_clusters(self, coordinates: np.ndarray, labels: np.ndarray,
                                sample_size: int = 100000) -> go.Figure:
        """Create interactive geographic cluster map.
        
        Args:
            coordinates: Array of (lat, lon) coordinates
            labels: Cluster labels
            sample_size: Maximum points to plot
            
        Returns:
            Plotly figure
        """
        # Sample if needed
        if len(coordinates) > sample_size:
            idx = np.random.choice(len(coordinates), sample_size, replace=False)
            coords_sample = coordinates[idx]
            labels_sample = labels[idx]
        else:
            coords_sample = coordinates
            labels_sample = labels
        
        # Create dataframe
        df = pd.DataFrame({
            'latitude': coords_sample[:, 0],
            'longitude': coords_sample[:, 1],
            'cluster': labels_sample.astype(str)
        })
        
        # Filter out noise points
        df = df[df['cluster'] != '-1']
        
        # Create map
        fig = px.scatter_geo(
            df,
            lat='latitude',
            lon='longitude',
            color='cluster',
            title=f'Global Biodiversity Clusters (n={len(df):,} samples)',
            color_discrete_sequence=px.colors.qualitative.Set3,
            projection='natural earth',
            hover_data={'latitude': ':.2f', 'longitude': ':.2f'}
        )
        
        fig.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth'
            ),
            height=600,
            margin=dict(t=50, b=0, l=0, r=0)
        )
        
        return fig
    
    def plot_cluster_densities(self, coordinates: np.ndarray, labels: np.ndarray) -> plt.Figure:
        """Create density heatmaps for each cluster.
        
        Args:
            coordinates: Array of (lat, lon) coordinates
            labels: Cluster labels
            
        Returns:
            Matplotlib figure with subplots
        """
        unique_labels = np.unique(labels[labels >= 0])
        n_clusters = len(unique_labels)
        
        # Create subplot grid
        n_cols = min(3, n_clusters)
        n_rows = (n_clusters + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        
        if n_clusters == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else axes
        
        for i, cluster_id in enumerate(unique_labels):
            ax = axes[i] if n_clusters > 1 else axes[0]
            
            # Get cluster coordinates
            mask = labels == cluster_id
            cluster_coords = coordinates[mask]
            
            # Subsample if too many points
            if len(cluster_coords) > 10000:
                idx = np.random.choice(len(cluster_coords), 10000, replace=False)
                cluster_coords = cluster_coords[idx]
            
            # Create hexbin plot
            if len(cluster_coords) > 0:
                hb = ax.hexbin(
                    cluster_coords[:, 1],  # longitude
                    cluster_coords[:, 0],  # latitude
                    gridsize=30,
                    cmap='YlOrRd',
                    mincnt=1
                )
                ax.set_title(f'Cluster {cluster_id} (n={mask.sum():,})', fontweight='bold')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.set_xlim(-180, 180)
                ax.set_ylim(-90, 90)
                plt.colorbar(hb, ax=ax, label='Density')
        
        # Hide unused subplots
        for i in range(n_clusters, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Geographic Distribution by Cluster', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_3d_clusters(self, features: np.ndarray, labels: np.ndarray,
                        feature_names: Optional[List[str]] = None,
                        sample_size: int = 50000) -> go.Figure:
        """Create interactive 3D cluster visualization.
        
        Args:
            features: First 3 features for visualization
            labels: Cluster labels
            feature_names: Names of features
            sample_size: Maximum points to plot
            
        Returns:
            Plotly 3D scatter plot
        """
        # Sample if needed
        if len(features) > sample_size:
            idx = np.random.choice(len(features), sample_size, replace=False)
            features_sample = features[idx]
            labels_sample = labels[idx]
        else:
            features_sample = features
            labels_sample = labels
        
        # Default feature names
        if feature_names is None:
            feature_names = ['Feature 1', 'Feature 2', 'Feature 3']
        
        # Create 3D scatter
        fig = go.Figure(data=[go.Scatter3d(
            x=features_sample[:, 0],
            y=features_sample[:, 1],
            z=features_sample[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=labels_sample,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Cluster"),
                opacity=0.6
            ),
            text=[f'Cluster {l}' for l in labels_sample],
            hovertemplate='<b>%{text}</b><br>' +
                         f'{feature_names[0]}: %{{x:.2f}}<br>' +
                         f'{feature_names[1]}: %{{y:.2f}}<br>' +
                         f'{feature_names[2]}: %{{z:.2f}}<br>' +
                         '<extra></extra>'
        )])
        
        fig.update_layout(
            title=f'3D Biodiversity Clusters (n={len(features_sample):,})',
            scene=dict(
                xaxis_title=feature_names[0],
                yaxis_title=feature_names[1],
                zaxis_title=feature_names[2],
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=700,
            margin=dict(t=50, b=0, l=0, r=0)
        )
        
        return fig
    
    def plot_validation_metrics(self, metrics: Dict[str, Any],
                               variance_preserved: Optional[float] = None) -> plt.Figure:
        """Plot comprehensive validation metrics.
        
        Args:
            metrics: Dictionary of metric values
            variance_preserved: Variance preserved by k-means
            
        Returns:
            Matplotlib figure with metric plots
        """
        fig = plt.figure(figsize=(15, 10))
        
        # Create subplots
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Silhouette score
        ax1 = fig.add_subplot(gs[0, 0])
        if 'silhouette' in metrics:
            ax1.bar(['Silhouette'], [metrics['silhouette']], color='steelblue')
            ax1.set_ylim(0, 1)
            ax1.set_ylabel('Score')
            ax1.set_title('Silhouette Score', fontweight='bold')
            ax1.text(0, metrics['silhouette'] + 0.02, f"{metrics['silhouette']:.3f}", 
                    ha='center', fontweight='bold')
        
        # 2. Davies-Bouldin score
        ax2 = fig.add_subplot(gs[0, 1])
        if 'davies_bouldin' in metrics:
            ax2.bar(['Davies-Bouldin'], [metrics['davies_bouldin']], color='coral')
            ax2.set_ylabel('Score (lower is better)')
            ax2.set_title('Davies-Bouldin Score', fontweight='bold')
            ax2.text(0, metrics['davies_bouldin'] + 0.02, 
                    f"{metrics['davies_bouldin']:.3f}", ha='center', fontweight='bold')
        
        # 3. Variance preserved
        ax3 = fig.add_subplot(gs[0, 2])
        if variance_preserved:
            ax3.bar(['Variance'], [variance_preserved], color='green')
            ax3.set_ylim(0, 1)
            ax3.set_ylabel('Fraction')
            ax3.set_title('Variance Preserved', fontweight='bold')
            ax3.text(0, variance_preserved + 0.02, f"{variance_preserved:.3f}", 
                    ha='center', fontweight='bold')
        
        # 4. Cluster sizes
        ax4 = fig.add_subplot(gs[1, :])
        if 'n_clusters' in metrics:
            sizes = []
            labels = []
            for i in range(metrics['n_clusters']):
                if f'cluster_{i}_size' in metrics:
                    sizes.append(metrics[f'cluster_{i}_size'])
                    labels.append(f'C{i}')
                elif 'min_cluster_size' in metrics:  # Use summary stats if available
                    break
            
            if sizes:
                ax4.bar(labels, sizes, color='mediumpurple')
                ax4.set_xlabel('Cluster')
                ax4.set_ylabel('Number of Points')
                ax4.set_title('Cluster Size Distribution', fontweight='bold')
                ax4.grid(True, alpha=0.3, axis='y')
            elif 'min_cluster_size' in metrics:
                # Show summary statistics
                stats_data = [
                    metrics.get('min_cluster_size', 0),
                    metrics.get('median_cluster_size', 0),
                    metrics.get('max_cluster_size', 0)
                ]
                ax4.bar(['Min', 'Median', 'Max'], stats_data, color=['red', 'yellow', 'green'])
                ax4.set_ylabel('Cluster Size')
                ax4.set_title('Cluster Size Statistics', fontweight='bold')
                ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Clustering Validation Metrics', fontsize=16, fontweight='bold')
        return fig
    
    def plot_cluster_profiles(self, features: np.ndarray, labels: np.ndarray,
                            feature_names: List[str]) -> plt.Figure:
        """Plot average feature profiles for each cluster.
        
        Args:
            features: Feature array
            labels: Cluster labels
            feature_names: Names of features
            
        Returns:
            Matplotlib figure with cluster profiles
        """
        n_clusters = len(np.unique(labels))
        n_features = features.shape[1]
        
        # Calculate cluster centers
        centers = np.zeros((n_clusters, n_features))
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                centers[k] = features[mask].mean(axis=0)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 6))
        
        im = ax.imshow(centers.T, aspect='auto', cmap='RdBu_r')
        
        # Set ticks and labels
        ax.set_xticks(range(n_clusters))
        ax.set_xticklabels([f'C{i}' for i in range(n_clusters)])
        ax.set_yticks(range(n_features))
        ax.set_yticklabels(feature_names[:n_features] if feature_names else 
                          [f'F{i}' for i in range(n_features)])
        
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Feature')
        ax.set_title('Average Feature Values by Cluster')
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        
        return fig
    
    def plot_parallel_coordinates(self, features: np.ndarray, labels: np.ndarray,
                                 feature_names: List[str], 
                                 sample_size: int = 10000) -> go.Figure:
        """Create parallel coordinates plot.
        
        Args:
            features: Feature array
            labels: Cluster labels
            feature_names: Names of features
            sample_size: Maximum points to plot
            
        Returns:
            Plotly parallel coordinates figure
        """
        # Sample and normalize
        if len(features) > sample_size:
            idx = np.random.choice(len(features), sample_size, replace=False)
            features_sample = features[idx]
            labels_sample = labels[idx]
        else:
            features_sample = features
            labels_sample = labels
        
        # Normalize features to 0-1
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        features_norm = scaler.fit_transform(features_sample)
        
        # Create dataframe
        df = pd.DataFrame(features_norm, columns=feature_names)
        df['cluster'] = labels_sample
        
        # Filter out noise
        df = df[df['cluster'] >= 0]
        
        # Create parallel coordinates
        fig = px.parallel_coordinates(
            df,
            color='cluster',
            dimensions=feature_names,
            color_continuous_scale=px.colors.diverging.Spectral,
            title=f'Parallel Coordinates View (n={len(df):,})'
        )
        
        fig.update_layout(
            height=600,
            margin=dict(t=80, b=50)
        )
        
        return fig
    
    def _get_cut_height(self, linkage_matrix: np.ndarray, n_clusters: int) -> float:
        """Calculate dendrogram cut height for n_clusters.
        
        Args:
            linkage_matrix: Scipy linkage matrix
            n_clusters: Desired number of clusters
            
        Returns:
            Height to cut dendrogram
        """
        # Get merge heights
        heights = linkage_matrix[:, 2]
        
        # Number of merges to get n_clusters
        n_merges = len(linkage_matrix) - n_clusters + 1
        
        if n_merges >= 0 and n_merges < len(heights):
            # Return height just above the merge that creates n_clusters
            return (heights[n_merges] + heights[n_merges + 1]) / 2 if n_merges + 1 < len(heights) else heights[n_merges] * 1.1
        else:
            return heights[-1] * 0.7  # Default to 70% of max height