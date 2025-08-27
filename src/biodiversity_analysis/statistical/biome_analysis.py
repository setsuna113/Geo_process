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
from src.biodiversity_analysis.statistical.holdridge_correct import (
    classify_holdridge_correct as classify_holdridge
)
from src.biodiversity_analysis.statistical.holdridge_proper import (
    get_altitudinal_belt, 
    holdridge_to_wwf
)
from src.biodiversity_analysis.statistical.data_schema import DataSchema
from src.biodiversity_analysis.statistical.result_types import MethodResult
from src.biodiversity_analysis.statistical.optimized_visualization import (
    precompute_boxplot_stats, plot_from_stats
)

logger = logging.getLogger(__name__)


def analyze_by_biome(
    df: Any,
    output_dir: Path,
    **kwargs
) -> MethodResult:
    """
    Analyze biodiversity data grouped by biome.
    
    NO SAMPLING. Use what pipeline gives us.
    
    Args:
        df: DataFrame with biodiversity and climate data
        output_dir: Output directory
        
    Returns:
        Biome-grouped statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert BiodiversityData to DataFrame if needed
    if not isinstance(df, pd.DataFrame):
        # Handle BiodiversityData object
        if hasattr(df, 'features') and hasattr(df, 'coordinates'):
            df_dict = {
                'x': df.coordinates[:, 0],
                'y': df.coordinates[:, 1]
            }
            # Add features with standardized names (already standardized in loader)
            for i, name in enumerate(df.feature_names):
                df_dict[name] = df.features[:, i]
            df = pd.DataFrame(df_dict)
            logger.info(f"Converted BiodiversityData to DataFrame with {len(df)} rows")
        else:
            raise TypeError(f"biome_analysis expects DataFrame or BiodiversityData, got {type(df).__name__}")
    
    # Validate and complete schema ONCE
    df = DataSchema.validate(df, 'biome_analysis')
    
    # Now we TRUST the schema - no more checks
    biodiv_cols = DataSchema.get_biodiversity_columns(df)
    
    logger.info(f"Analyzing {len(df):,} points for biome classification")
    logger.info(f"Biodiversity columns: {biodiv_cols}")
    
    # Classify biomes - check if optional columns exist
    temp_celsius = df['bio01'].values / 10.0  # Convert from °C*10 to °C
    precip_mm = df['bio12'].values
    
    # Use elevation if available, otherwise use 0 (sea level)
    elevation_m = df['elevation'].values if 'elevation' in df.columns else np.zeros(len(df))
    
    # Require latitude for proper biome classification
    if 'y' not in df.columns:
        raise ValueError("Biome analysis requires latitude ('y' column)")
    latitude = df['y'].values
    
    df['biome'] = classify_holdridge(temp_celsius, precip_mm, elevation_m)
    df['altitude_belt'] = get_altitudinal_belt(elevation_m, latitude)
    
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
    
    # By altitude belt - we KNOW this column exists
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
    
    # Biome-altitude matrix
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
    
    return MethodResult(
        success=True,
        method='biome_analysis',
        output_dir=output_dir,
        metrics=results
    )


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
            # Pre-compute statistics for MILLIONS of points efficiently
            logger.info(f"Computing boxplot statistics for {col} with {len(df):,} points")
            
            # Group by biome and collect data
            biome_groups = df.groupby('biome')[col]
            data_groups = []
            labels = []
            
            for biome, group in biome_groups:
                values = group.dropna().values
                if len(values) > 0:
                    data_groups.append(values)
                    labels.append(biome)
            
            # Pre-compute statistics (fast for any data size)
            stats = precompute_boxplot_stats(data_groups, labels)
            
            # Order by median
            ordered_stats = dict(sorted(stats.items(), 
                                      key=lambda x: x[1]['median'] if x[1] else -np.inf,
                                      reverse=True))
            
            # Plot from pre-computed statistics (instant!)
            plot_from_stats(axes[i], ordered_stats, patch_artist=True)
            
            # Color boxes based on ordered stats (skip coloring for now)
            # colors = plt.cm.viridis(np.linspace(0, 1, len(ordered_stats)))
            
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