# src/spatial_analysis/som/som_visualizer.py
"""Visualization tools for SOM analysis results."""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import statistics
import matplotlib.pyplot as plt


from matplotlib.colors import ListedColormap


from src.spatial_analysis.base_analyzer import AnalysisResult

logger = logging.getLogger(__name__)

class SOMVisualizer:
    """Create visualizations for SOM analysis results."""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        self.figsize = figsize
        self.cmap = plt.colormaps.get_cmap("viridis")
        
    def plot_cluster_map(self, result: AnalysisResult, 
                        save_path: Optional[str] = None,
                        show_empty: bool = True) -> Any:
        """
        Plot spatial distribution of SOM clusters.
        
        Args:
            result: SOM analysis result
            save_path: Optional path to save figure
            show_empty: Whether to show empty grid cells
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Get spatial output
        spatial_data = result.spatial_output
        
        # Create discrete colormap
        n_clusters = result.statistics.get('n_clusters', 0) if result.statistics is not None else 0
        colors = plt.colormaps.get_cmap("tab20")(np.linspace(0, 1, max(1, n_clusters)))
        cmap = ListedColormap(colors)
        
        # Plot
        if spatial_data is not None:
            data_to_plot = spatial_data.values if hasattr(spatial_data, "values") else spatial_data
        else:
            data_to_plot = np.zeros((1, 1))
        im = ax.imshow(data_to_plot, cmap=cmap, aspect="auto")
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Cluster ID')
        cbar.set_ticks(range(n_clusters))
        
        # Set labels
        ax.set_title('SOM Cluster Distribution', fontsize=16, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_umatrix(self, result: AnalysisResult,
                    save_path: Optional[str] = None) -> Any:
        """
        Plot U-matrix (distance map) showing cluster boundaries.
        
        Args:
            result: SOM analysis result
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        distance_map = result.additional_outputs['distance_map'] if result.additional_outputs else np.zeros((1, 1))
        
        # Plot U-matrix
        im = ax.imshow(distance_map.T, cmap='bone_r', aspect='auto')
        plt.colorbar(im, ax=ax, label='Distance')
        
        # Add activation overlay
        activation = result.additional_outputs['activation_map'].T if result.additional_outputs else np.zeros((1, 1))
        # Normalize activation for overlay
        activation_norm = activation / activation.max()
        
        # Create overlay showing node usage
        for i in range(activation.shape[0]):
            for j in range(activation.shape[1]):
                if activation[i, j] > 0:
                    circle = plt.Circle((j, i), 
                                      radius=0.3 * activation_norm[i, j],
                                      color='red', alpha=0.5)
                    ax.add_patch(circle)
        
        ax.set_title('SOM U-Matrix with Node Activation', fontsize=16, fontweight='bold')
        ax.set_xlabel('SOM Grid X')
        ax.set_ylabel('SOM Grid Y')
        
        # Set ticks
        ax.set_xticks(range(distance_map.shape[0]))
        ax.set_yticks(range(distance_map.shape[1]))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_component_planes(self, result: AnalysisResult,
                            save_path: Optional[str] = None) -> Any:
        """
        Plot component planes showing feature weights.
        
        Args:
            result: SOM analysis result
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        component_planes = result.additional_outputs.get('component_planes', {}) if result.additional_outputs else {}
        n_components = len(component_planes)
        
        if n_components == 0:
            # Return empty figure if no component planes available
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No component planes available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return fig
        
        # Create subplot grid
        ncols = min(3, n_components)
        nrows = (n_components + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        if n_components == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot each component
        for idx, (name, plane) in enumerate(component_planes.items()):
            ax = axes[idx]
            
            im = ax.imshow(plane.T, cmap="RdBu_r", aspect="auto")
            plt.colorbar(im, ax=ax, label='Weight')
            
            ax.set_title(f'{name} Component', fontsize=12, fontweight='bold')
            ax.set_xlabel('Grid X')
            ax.set_ylabel('Grid Y')
            ax.grid(True, alpha=0.3)
        
        # Hide extra subplots
        for idx in range(n_components, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('SOM Component Planes', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_cluster_profiles(self, result: AnalysisResult,
                            normalize: bool = True,
                            save_path: Optional[str] = None) -> Any:
        """
        Plot average profiles for each cluster.
        
        Args:
            result: SOM analysis result
            normalize: Whether to normalize profiles for comparison
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        cluster_stats = result.statistics.get('cluster_statistics', {}) if result.statistics is not None else {}
        n_clusters = len(cluster_stats)
        
        if n_clusters == 0:
            # Return empty figure if no cluster statistics available
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No cluster statistics available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return fig
        
        # Extract means for each cluster
        clusters = sorted(cluster_stats.keys())
        band_names = result.metadata.input_bands
        
        # Create data matrix
        data = []
        for cluster in clusters:
            means = cluster_stats[cluster]['mean']
            data.append(means)
        
        data = np.array(data)
        
        if normalize:
            # Normalize to [0, 1] for each feature
            data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-8)
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=self.figsize, subplot_kw=dict(projection='polar'))
        
        # Set up angles
        angles = np.linspace(0, 2 * np.pi, len(band_names), endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])  # Complete the circle
        
        # Plot each cluster
        colors = plt.colormaps.get_cmap("tab20")(np.linspace(0, 1, max(1, n_clusters)))
        
        for idx, (cluster, color) in enumerate(zip(clusters, colors)):
            values = data[idx]
            values = np.concatenate([values, [values[0]]])  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster}',
                   color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        # Customize plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(band_names)
        ax.set_ylim(0, 1 if normalize else None)
        ax.set_title('Cluster Biodiversity Profiles', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_summary_figure(self, result: AnalysisResult,
                            save_path: Optional[str] = None) -> Any:
        """Create a comprehensive summary figure."""
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Cluster map (large)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        spatial_data = result.spatial_output
        n_clusters = result.statistics.get('n_clusters', 0) if result.statistics is not None else 0
        colors = plt.colormaps.get_cmap("tab20")(np.linspace(0, 1, max(1, n_clusters)))
        cmap = ListedColormap(colors)
        im1 = ax1.imshow(spatial_data.values if spatial_data is not None and hasattr(spatial_data, "values") else (spatial_data if spatial_data is not None else np.array([])), cmap=cmap, aspect='auto')
        ax1.set_title('Spatial Cluster Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        plt.colorbar(im1, ax=ax1, label='Cluster ID')
        
        # 2. U-matrix
        ax2 = fig.add_subplot(gs[0, 2])
        distance_map = result.additional_outputs.get('distance_map', np.zeros((1, 1))) if result.additional_outputs else np.zeros((1, 1))
        _ = ax2.imshow(distance_map.T, cmap='bone_r', aspect='auto')  # Intentionally unused
        ax2.set_title('U-Matrix', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Grid X')
        ax2.set_ylabel('Grid Y')
        
        # 3. Cluster sizes
        ax3 = fig.add_subplot(gs[1, 2])
        cluster_stats = result.statistics.get('cluster_statistics', {}) if result.statistics is not None else {}
        if cluster_stats:
            sizes = [stats['count'] for stats in cluster_stats.values()]
            clusters = list(cluster_stats.keys())
            _ = ax3.bar(clusters, sizes, color=colors[:len(clusters)])  # Intentionally unused
        ax3.set_title('Cluster Sizes', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Cluster ID')
        ax3.set_ylabel('Number of Pixels')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Quality metrics
        ax4 = fig.add_subplot(gs[2, 0])
        metrics_text = (
            f"Quantization Error: {result.statistics.get('quantization_error', 0) if result.statistics is not None else 0:.4f}\n"
            f"Topographic Error: {result.statistics.get('topographic_error', 0) if result.statistics is not None else 0:.4f}\n"
            f"Empty Neurons: {result.statistics.get('empty_neurons', 0) if result.statistics is not None else 0}\n"
            f"Cluster Balance: {result.statistics.get('cluster_balance', 0) if result.statistics is not None else 0:.4f}"
        )
        ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes,
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray'))
        ax4.set_title('Quality Metrics', fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        # 5. Top clusters info
        ax5 = fig.add_subplot(gs[2, 1:])
        # Get top 5 clusters by size
        if cluster_stats:
            sorted_clusters = sorted(cluster_stats.items(), 
                                   key=lambda x: x[1]['count'], 
                                   reverse=True)[:5]
        else:
            sorted_clusters = []
        
        table_data = []
        for cluster_id, stats in sorted_clusters:
            means = stats['mean']
            mean_str = ', '.join([f'{m:.1f}' for m in means])
            table_data.append([
                f"Cluster {cluster_id}",
                f"{stats['count']:,}",
                f"{stats['percentage']:.1f}%",
                mean_str
            ])
        
        table = ax5.table(cellText=table_data,
                         colLabels=['Cluster', 'Size', 'Percentage', 'Mean Values'],
                         cellLoc='left',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax5.set_title('Top 5 Clusters by Size', fontsize=12, fontweight='bold')
        ax5.axis('off')
        
        plt.suptitle('SOM Analysis Summary', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig