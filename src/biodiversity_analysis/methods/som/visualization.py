"""Visualization tools for biodiversity SOM analysis."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Tuple, List
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import logging

logger = logging.getLogger(__name__)


class SOMVisualizer:
    """Visualization tools for biodiversity SOM analysis.
    
    Provides various visualization methods for SOM results including:
    - U-matrix (unified distance matrix)
    - Component planes
    - Hit maps
    - Biodiversity-specific visualizations
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 10), 
                 style: str = 'whitegrid'):
        """Initialize visualizer.
        
        Args:
            figsize: Default figure size
            style: Seaborn style
        """
        self.figsize = figsize
        sns.set_style(style)
    
    def plot_u_matrix(self, weights: np.ndarray, 
                     save_path: Optional[str] = None,
                     cmap: str = 'viridis',
                     interpolation: str = 'bilinear') -> plt.Figure:
        """Plot unified distance matrix showing neuron distances.
        
        The U-matrix visualizes the distance between neighboring neurons,
        helping to identify cluster boundaries.
        
        Args:
            weights: SOM weights of shape (n_rows, n_cols, n_features)
            save_path: Optional path to save figure
            cmap: Colormap name
            interpolation: Interpolation method
            
        Returns:
            Matplotlib figure
        """
        n_rows, n_cols, n_features = weights.shape
        u_matrix = np.zeros((n_rows, n_cols))
        
        # Calculate distances to neighbors
        for i in range(n_rows):
            for j in range(n_cols):
                distances = []
                
                # Check all 8 neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        
                        ni, nj = i + di, j + dj
                        if 0 <= ni < n_rows and 0 <= nj < n_cols:
                            dist = np.linalg.norm(weights[i, j] - weights[ni, nj])
                            distances.append(dist)
                
                # Average distance to neighbors
                u_matrix[i, j] = np.mean(distances) if distances else 0
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.imshow(u_matrix, cmap=cmap, interpolation=interpolation)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Average distance to neighbors', rotation=270, labelpad=20)
        
        # Labels
        ax.set_title('SOM U-Matrix', fontsize=16, fontweight='bold')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        
        # Add grid
        ax.set_xticks(np.arange(n_cols))
        ax.set_yticks(np.arange(n_rows))
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"U-matrix saved to {save_path}")
        
        return fig
    
    def plot_component_planes(self, weights: np.ndarray,
                            feature_names: List[str],
                            n_cols: int = 4,
                            save_path: Optional[str] = None,
                            cmap: str = 'coolwarm') -> plt.Figure:
        """Plot individual feature planes showing feature values across SOM.
        
        Args:
            weights: SOM weights of shape (n_rows, n_cols, n_features)
            feature_names: Names of features
            n_cols: Number of columns in subplot grid
            save_path: Optional path to save figure
            cmap: Colormap name
            
        Returns:
            Matplotlib figure
        """
        n_rows_som, n_cols_som, n_features = weights.shape
        
        # Calculate subplot grid dimensions
        n_rows_plot = int(np.ceil(n_features / n_cols))
        
        # Create figure
        fig, axes = plt.subplots(n_rows_plot, n_cols, 
                               figsize=(n_cols * 3, n_rows_plot * 3))
        
        # Flatten axes for easier iteration
        if n_rows_plot == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot each component
        for idx, feature_name in enumerate(feature_names):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            # Get feature values
            feature_plane = weights[:, :, idx]
            
            # Plot
            im = ax.imshow(feature_plane, cmap=cmap, aspect='auto')
            ax.set_title(feature_name, fontsize=10)
            ax.set_xlabel('Column', fontsize=8)
            ax.set_ylabel('Row', fontsize=8)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=8)
        
        # Hide empty subplots
        for idx in range(n_features, n_rows_plot * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.suptitle('SOM Component Planes', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Component planes saved to {save_path}")
        
        return fig
    
    def plot_hit_map(self, cluster_assignments: np.ndarray,
                    grid_shape: Tuple[int, int],
                    save_path: Optional[str] = None,
                    cmap: str = 'YlOrRd') -> plt.Figure:
        """Plot hit map showing sample density across SOM.
        
        Args:
            cluster_assignments: Array of (row, col) assignments
            grid_shape: Shape of SOM grid
            save_path: Optional path to save figure
            cmap: Colormap name
            
        Returns:
            Matplotlib figure
        """
        n_rows, n_cols = grid_shape
        hit_map = np.zeros((n_rows, n_cols))
        
        # Count hits for each neuron
        for row, col in cluster_assignments:
            if 0 <= row < n_rows and 0 <= col < n_cols:
                hit_map[int(row), int(col)] += 1
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.imshow(hit_map, cmap=cmap, interpolation='nearest')
        
        # Add text annotations
        for i in range(n_rows):
            for j in range(n_cols):
                count = int(hit_map[i, j])
                if count > 0:
                    ax.text(j, i, str(count), ha='center', va='center',
                           color='white' if hit_map[i, j] > hit_map.max() / 2 else 'black')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Number of samples', rotation=270, labelpad=20)
        
        # Labels
        ax.set_title('SOM Hit Map', fontsize=16, fontweight='bold')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        
        # Grid
        ax.set_xticks(np.arange(n_cols))
        ax.set_yticks(np.arange(n_rows))
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Hit map saved to {save_path}")
        
        return fig
    
    def plot_biodiversity_patterns(self, 
                                 weights: np.ndarray,
                                 cluster_assignments: np.ndarray,
                                 species_data: np.ndarray,
                                 coordinates: Optional[np.ndarray] = None,
                                 species_names: Optional[List[str]] = None,
                                 save_path: Optional[str] = None) -> Dict[str, plt.Figure]:
        """Create biodiversity-specific visualizations.
        
        Args:
            weights: SOM weights
            cluster_assignments: Cluster assignments for each sample
            species_data: Original species abundance data
            coordinates: Optional spatial coordinates
            species_names: Optional species names
            save_path: Optional base path for saving figures
            
        Returns:
            Dictionary of figures
        """
        figures = {}
        
        # 1. Species richness per cluster
        fig_richness = self._plot_species_richness_map(
            weights, cluster_assignments, species_data
        )
        figures['species_richness'] = fig_richness
        
        # 2. Diversity indices per cluster
        fig_diversity = self._plot_diversity_indices_map(
            weights, cluster_assignments, species_data
        )
        figures['diversity_indices'] = fig_diversity
        
        # 3. Spatial distribution of clusters (if coordinates provided)
        if coordinates is not None:
            fig_spatial = self._plot_spatial_clusters(
                cluster_assignments, coordinates, weights.shape[:2]
            )
            figures['spatial_clusters'] = fig_spatial
        
        # 4. Species composition heatmap
        if species_names is not None and len(species_names) <= 50:
            fig_composition = self._plot_species_composition_heatmap(
                weights, cluster_assignments, species_data, species_names
            )
            figures['species_composition'] = fig_composition
        
        # Save figures if requested
        if save_path:
            for name, fig in figures.items():
                fig_path = f"{save_path}_{name}.png"
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                logger.info(f"{name} figure saved to {fig_path}")
        
        return figures
    
    def _plot_species_richness_map(self, weights: np.ndarray,
                                  cluster_assignments: np.ndarray,
                                  species_data: np.ndarray) -> plt.Figure:
        """Plot average species richness per SOM neuron."""
        n_rows, n_cols = weights.shape[:2]
        richness_map = np.zeros((n_rows, n_cols))
        count_map = np.zeros((n_rows, n_cols))
        
        # Calculate richness for each sample
        species_richness = np.sum(species_data > 0, axis=1)
        
        # Aggregate by cluster
        for idx, (row, col) in enumerate(cluster_assignments):
            if 0 <= row < n_rows and 0 <= col < n_cols:
                richness_map[int(row), int(col)] += species_richness[idx]
                count_map[int(row), int(col)] += 1
        
        # Average richness
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_richness = np.divide(richness_map, count_map)
            avg_richness[count_map == 0] = 0
        
        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.imshow(avg_richness, cmap='YlGnBu', interpolation='bilinear')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Average Species Richness', rotation=270, labelpad=20)
        
        # Labels
        ax.set_title('Species Richness across SOM', fontsize=16, fontweight='bold')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        
        plt.tight_layout()
        return fig
    
    def _plot_diversity_indices_map(self, weights: np.ndarray,
                                  cluster_assignments: np.ndarray,
                                  species_data: np.ndarray) -> plt.Figure:
        """Plot Shannon diversity per SOM neuron."""
        n_rows, n_cols = weights.shape[:2]
        
        # Calculate Shannon diversity for each sample
        shannon_diversity = self._calculate_shannon_diversity(species_data)
        
        # Create diversity map
        diversity_map = np.zeros((n_rows, n_cols))
        count_map = np.zeros((n_rows, n_cols))
        
        for idx, (row, col) in enumerate(cluster_assignments):
            if 0 <= row < n_rows and 0 <= col < n_cols:
                diversity_map[int(row), int(col)] += shannon_diversity[idx]
                count_map[int(row), int(col)] += 1
        
        # Average diversity
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_diversity = np.divide(diversity_map, count_map)
            avg_diversity[count_map == 0] = 0
        
        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)
        im = ax.imshow(avg_diversity, cmap='viridis', interpolation='bilinear')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Average Shannon Diversity', rotation=270, labelpad=20)
        
        # Labels
        ax.set_title('Shannon Diversity across SOM', fontsize=16, fontweight='bold')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        
        plt.tight_layout()
        return fig
    
    def _plot_spatial_clusters(self, cluster_assignments: np.ndarray,
                             coordinates: np.ndarray,
                             grid_shape: Tuple[int, int]) -> plt.Figure:
        """Plot spatial distribution of SOM clusters."""
        # Create color map for clusters
        n_clusters = grid_shape[0] * grid_shape[1]
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
        
        # Convert cluster assignments to linear indices
        cluster_indices = cluster_assignments[:, 0] * grid_shape[1] + cluster_assignments[:, 1]
        
        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1],
                           c=cluster_indices, cmap='tab20',
                           alpha=0.6, s=10)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('SOM Cluster', rotation=270, labelpad=20)
        
        # Labels
        ax.set_title('Spatial Distribution of SOM Clusters', fontsize=16, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        plt.tight_layout()
        return fig
    
    def _plot_species_composition_heatmap(self, weights: np.ndarray,
                                        cluster_assignments: np.ndarray,
                                        species_data: np.ndarray,
                                        species_names: List[str]) -> plt.Figure:
        """Plot heatmap of species composition per cluster."""
        n_rows, n_cols = weights.shape[:2]
        n_clusters = n_rows * n_cols
        n_species = len(species_names)
        
        # Calculate average abundance per cluster
        cluster_composition = np.zeros((n_clusters, n_species))
        cluster_counts = np.zeros(n_clusters)
        
        for idx, (row, col) in enumerate(cluster_assignments):
            cluster_idx = int(row) * n_cols + int(col)
            if 0 <= cluster_idx < n_clusters:
                cluster_composition[cluster_idx] += species_data[idx]
                cluster_counts[cluster_idx] += 1
        
        # Average composition
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_composition = cluster_composition / cluster_counts[:, np.newaxis]
            avg_composition[cluster_counts == 0] = 0
        
        # Select top species by variance
        species_variance = np.var(avg_composition, axis=0)
        top_species_idx = np.argsort(species_variance)[-30:]  # Top 30 species
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(avg_composition[:, top_species_idx].T,
                   xticklabels=[f"({i//n_cols},{i%n_cols})" for i in range(n_clusters)],
                   yticklabels=[species_names[i] for i in top_species_idx],
                   cmap='YlOrRd', cbar_kws={'label': 'Average Abundance'},
                   ax=ax)
        
        ax.set_title('Species Composition across SOM Clusters', fontsize=16, fontweight='bold')
        ax.set_xlabel('SOM Cluster (row, col)')
        ax.set_ylabel('Species')
        
        plt.tight_layout()
        return fig
    
    def _calculate_shannon_diversity(self, abundance_data: np.ndarray) -> np.ndarray:
        """Calculate Shannon diversity index for each sample."""
        # Normalize to proportions
        row_sums = abundance_data.sum(axis=1, keepdims=True)
        proportions = np.divide(abundance_data, row_sums, 
                              out=np.zeros_like(abundance_data), 
                              where=row_sums != 0)
        
        # Shannon diversity: -sum(p * log(p))
        with np.errstate(divide='ignore', invalid='ignore'):
            log_prop = np.log(proportions)
            log_prop[proportions == 0] = 0
        
        shannon = -np.sum(proportions * log_prop, axis=1)
        return shannon
    
    def plot_training_history(self, training_result,
                            save_path: Optional[str] = None) -> plt.Figure:
        """Plot training history with quantization and topographic errors.
        
        Args:
            training_result: SOMTrainingResult object
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        epochs = range(len(training_result.quantization_errors))
        
        # Quantization error
        ax1.plot(epochs, training_result.quantization_errors, 
                'b-', linewidth=2, label='Quantization Error')
        ax1.set_ylabel('Quantization Error', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Topographic error
        ax2.plot(epochs, training_result.topographic_errors, 
                'r-', linewidth=2, label='Topographic Error')
        ax2.set_ylabel('Topographic Error', fontsize=12)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Mark convergence
        if training_result.convergence_epoch is not None:
            for ax in [ax1, ax2]:
                ax.axvline(x=training_result.convergence_epoch, 
                         color='green', linestyle='--', alpha=0.7,
                         label=f'Convergence (epoch {training_result.convergence_epoch})')
        
        plt.suptitle('SOM Training History', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history saved to {save_path}")
        
        return fig