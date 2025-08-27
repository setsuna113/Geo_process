"""
Optimized visualization for biome analysis.

Linus: "Optimization that matters is optimization you can measure."
This handles large biome datasets efficiently.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def create_biome_heatmap(
    data: Dict[str, Any],
    output_path: str,
    title: str = "Biodiversity by Biome",
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """Create heatmap of biodiversity by biome.
    
    Args:
        data: Dictionary with biome statistics
        output_path: Where to save
        title: Plot title
        figsize: Figure size
    """
    # Extract biome data
    biome_stats = data.get('biome_stats', {})
    
    if not biome_stats:
        logger.warning("No biome statistics to visualize")
        return
    
    # Create matrix for heatmap
    biomes = list(biome_stats.keys())
    metrics = ['mean', 'median', 'std', 'coverage']
    species = []
    
    # Find all species
    for biome_data in biome_stats.values():
        species.extend(biome_data.keys())
    species = sorted(set(species))
    
    # Build data matrix
    n_biomes = len(biomes)
    n_species = len(species)
    n_metrics = len(metrics)
    
    # Create subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        matrix = np.zeros((n_biomes, n_species))
        
        for i, biome in enumerate(biomes):
            for j, sp in enumerate(species):
                if sp in biome_stats[biome]:
                    value = biome_stats[biome][sp].get(metric, 0)
                    matrix[i, j] = value
        
        # Plot heatmap
        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd')
        
        # Labels
        ax.set_xticks(range(n_species))
        ax.set_xticklabels(species, rotation=45, ha='right')
        ax.set_yticks(range(n_biomes))
        ax.set_yticklabels(biomes)
        ax.set_title(f'{metric.title()}', fontweight='bold')
        
        # Colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved biome heatmap to {output_path}")


def create_biome_summary_plot(
    biome_stats: Dict[str, Dict[str, Any]],
    output_path: str,
    title: str = "Biome Biodiversity Summary"
) -> None:
    """Create summary bar plot of biodiversity by biome.
    
    Simple bar chart. No fancy stuff.
    """
    if not biome_stats:
        logger.warning("No biome data to plot")
        return
    
    # Calculate total biodiversity per biome
    biomes = []
    totals = []
    
    for biome, stats in biome_stats.items():
        biomes.append(biome)
        # Sum mean values across species
        total = sum(s.get('mean', 0) for s in stats.values())
        totals.append(total)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(biomes)))
    bars = ax.bar(range(len(biomes)), totals, color=colors)
    
    # Labels
    ax.set_xticks(range(len(biomes)))
    ax.set_xticklabels(biomes, rotation=45, ha='right')
    ax.set_ylabel('Total Biodiversity (sum of means)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar, value in zip(bars, totals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.0f}',
               ha='center', va='bottom', fontsize=9)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved biome summary to {output_path}")