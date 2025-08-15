"""
Biome-based statistical analysis - Linus style.

"Good code has no special cases"
ONE data type, ONE path, ZERO bullshit.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union
import logging

from src.biodiversity_analysis.statistical.data_utils import save_results
from src.biodiversity_analysis.statistical.holdridge_classifier import (
    classify_holdridge, get_altitudinal_belt, holdridge_to_wwf
)

logger = logging.getLogger(__name__)


def analyze_by_biome(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    chunk_size: int = 100000,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze biodiversity data grouped by biome.
    
    ONE input type: DataFrame. That's it.
    
    Args:
        df: DataFrame with biodiversity and climate data
        output_dir: Output directory
        chunk_size: Max rows to process (for sampling)
        
    Returns:
        Biome-grouped statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample if needed for performance
    if chunk_size and len(df) > chunk_size:
        df = df.sample(n=chunk_size, random_state=42)
        logger.info(f"Sampled {chunk_size:,} points for biome analysis")
    
    # Check required columns - NO try-except bullshit
    required_climate = ['bio01', 'bio12']
    if not all(col in df.columns for col in required_climate):
        raise ValueError(f"Missing required climate columns: {required_climate}")
    
    # Get biodiversity columns - whatever we have
    biodiv_cols = []
    for col in ['animal_richness', 'plant_richness', 'am_fungi_richness', 'ecm_fungi_richness']:
        if col in df.columns:
            biodiv_cols.append(col)
    
    if not biodiv_cols:
        raise ValueError("No biodiversity columns found in DataFrame")
    
    logger.info(f"Analyzing {len(biodiv_cols)} biodiversity columns across biomes")
    
    # Classify biomes using Holdridge
    temp_celsius = df['bio01'].values / 10.0  # Convert from °C*10 to °C
    precip_mm = df['bio12'].values
    
    # Use elevation if available, otherwise assume sea level
    elevation_m = df['elevation'].values if 'elevation' in df.columns else np.zeros(len(df))
    latitude = df['y'].values if 'y' in df.columns else None
    
    df['biome'] = classify_holdridge(temp_celsius, precip_mm, elevation_m, latitude)
    
    # Add altitudinal belt if we have both elevation and coordinates
    if 'elevation' in df.columns and 'y' in df.columns:
        df['altitude_belt'] = get_altitudinal_belt(elevation_m, df['y'].values)
    
    # Group by biome and calculate statistics
    results = {
        'overall': {},
        'by_biome': {},
        'by_altitude': {},
        'biome_altitude_matrix': {}
    }
    
    # Overall statistics
    for col in biodiv_cols:
        valid_data = df[col].dropna()
        if len(valid_data) > 0:
            results['overall'][col] = {
                'mean': float(valid_data.mean()),
                'std': float(valid_data.std()),
                'min': float(valid_data.min()),
                'max': float(valid_data.max()),
                'count': int(len(valid_data))
            }
    
    # By biome
    for biome, group in df.groupby('biome'):
        if len(group) > 0:
            biome_stats = {'count': int(len(group))}
            for col in biodiv_cols:
                valid_data = group[col].dropna()
                if len(valid_data) > 0:
                    biome_stats[col] = {
                        'mean': float(valid_data.mean()),
                        'std': float(valid_data.std()),
                        'count': int(len(valid_data))
                    }
            results['by_biome'][biome] = biome_stats
    
    # By altitude belt (if available)
    if 'altitude_belt' in df.columns:
        for belt, group in df.groupby('altitude_belt'):
            if len(group) > 0:
                belt_stats = {'count': int(len(group))}
                for col in biodiv_cols:
                    valid_data = group[col].dropna()
                    if len(valid_data) > 0:
                        belt_stats[col] = {
                            'mean': float(valid_data.mean()),
                            'std': float(valid_data.std()),
                            'count': int(len(valid_data))
                        }
                results['by_altitude'][belt] = belt_stats
    
    # Biome-altitude matrix (if both available)
    if 'altitude_belt' in df.columns:
        for biome in df['biome'].unique():
            biome_data = df[df['biome'] == biome]
            matrix = {}
            for belt in biome_data['altitude_belt'].unique():
                belt_data = biome_data[biome_data['altitude_belt'] == belt]
                if len(belt_data) > 0:
                    matrix[belt] = {
                        'count': int(len(belt_data)),
                        'mean_richness': {}
                    }
                    for col in biodiv_cols:
                        valid = belt_data[col].dropna()
                        if len(valid) > 0:
                            matrix[belt]['mean_richness'][col] = float(valid.mean())
            if matrix:
                results['biome_altitude_matrix'][biome] = matrix
    
    # Add WWF biome mapping for reference
    results['biome_mapping'] = {}
    for biome in df['biome'].unique():
        results['biome_mapping'][biome] = holdridge_to_wwf(biome)
    
    # Summary statistics
    results['summary'] = {
        'n_samples': int(len(df)),
        'n_biomes': int(df['biome'].nunique()),
        'dominant_biome': df['biome'].value_counts().index[0] if len(df) > 0 else None,
        'biodiversity_columns': biodiv_cols
    }
    
    # Save results
    save_results(results, output_dir / 'biome_analysis.json')
    
    # Generate plots if requested
    if kwargs.get('plot', True):
        _plot_biome_analysis(df, biodiv_cols, output_dir)
    
    return results


def _plot_biome_analysis(df: pd.DataFrame, biodiv_cols: list, output_dir: Path):
    """Generate biome analysis plots."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Biodiversity by biome boxplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, col in enumerate(biodiv_cols[:4]):  # Max 4 plots
        if i < len(axes):
            # Order biomes by median richness
            biome_order = df.groupby('biome')[col].median().sort_values(ascending=False).index
            
            # Create boxplot
            bp = axes[i].boxplot(
                [df[df['biome'] == b][col].dropna() for b in biome_order],
                labels=biome_order,
                patch_artist=True
            )
            
            # Color boxes
            colors = plt.cm.viridis(np.linspace(0, 1, len(biome_order)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            axes[i].set_xlabel('Biome')
            axes[i].set_ylabel(f'{col.replace("_", " ").title()}')
            axes[i].set_title(f'{col.replace("_", " ").title()} by Biome')
            axes[i].tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for i in range(len(biodiv_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'biome_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Biome distribution pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    biome_counts = df['biome'].value_counts()
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        biome_counts.values,
        labels=biome_counts.index,
        autopct='%1.1f%%',
        startangle=90
    )
    
    # Enhance text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')
    
    ax.set_title('Biome Distribution in Dataset')
    plt.savefig(output_dir / 'biome_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Correlation heatmap within each biome
    if len(biodiv_cols) > 1:
        n_biomes = df['biome'].nunique()
        fig, axes = plt.subplots(
            int(np.ceil(n_biomes / 3)), 3,
            figsize=(15, 5 * int(np.ceil(n_biomes / 3)))
        )
        axes = axes.ravel() if n_biomes > 3 else [axes]
        
        for i, (biome, group) in enumerate(df.groupby('biome')):
            if i < len(axes):
                # Calculate correlation matrix
                corr = group[biodiv_cols].corr(method='spearman')
                
                # Plot heatmap
                sns.heatmap(
                    corr,
                    annot=True,
                    fmt='.2f',
                    cmap='coolwarm',
                    center=0,
                    ax=axes[i],
                    vmin=-1,
                    vmax=1
                )
                axes[i].set_title(f'{biome}')
        
        # Hide unused subplots
        for i in range(n_biomes, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'biome_correlations.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Saved biome analysis plots to {output_dir}")