"""
Global map visualization - Linus style.

284行 -> 90行
"Bad programmers worry about the code. Good programmers worry about data structures."
"""

import numpy as np
from pathlib import Path
from typing import Dict, Union, Any
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.biodiversity_analysis.statistical.data_utils import save_results, save_figure
from src.biodiversity_analysis.shared.data.raster_loader import RasterLoader
import logging

logger = logging.getLogger(__name__)


def _create_standalone_biome_chart(df, feature_col, name, output_dir):
    """Create standalone detailed biome distribution chart."""
    import pandas as pd
    import seaborn as sns
    
    # Use detailed Holdridge biomes for comprehensive view
    biome_col = 'biome_detailed' if 'biome_detailed' in df.columns else 'biome'
    
    # Prepare data for detailed boxplot
    biome_data = df.groupby(biome_col)[feature_col].apply(list).to_dict()
    
    # Remove biomes with too few samples
    biome_data = {k: v for k, v in biome_data.items() 
                  if len([x for x in v if pd.notna(x)]) >= 5}
    
    # Sort biomes by median richness
    biome_medians = {k: np.median([x for x in v if pd.notna(x)]) 
                     for k, v in biome_data.items()}
    sorted_biomes = sorted(biome_medians.keys(), key=lambda x: biome_medians[x])
    
    # Create figure for detailed biome analysis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                   gridspec_kw={'height_ratios': [1, 2]})
    
    # Top panel: Biome sample sizes
    biome_counts = {b: len([x for x in biome_data[b] if pd.notna(x)]) 
                   for b in sorted_biomes}
    
    ax1.bar(range(len(sorted_biomes)), list(biome_counts.values()), 
           color='steelblue', alpha=0.7)
    ax1.set_xticks([])
    ax1.set_ylabel('Sample Size (n)', fontsize=10)
    ax1.set_title(f'Sample Distribution Across {len(sorted_biomes)} Holdridge Life Zones', 
                 fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add text labels for large biomes
    for i, (biome, count) in enumerate(biome_counts.items()):
        if count > max(biome_counts.values()) * 0.1:  # Label if >10% of max
            ax1.text(i, count, f'{count}', ha='center', va='bottom', fontsize=8)
    
    # Bottom panel: Boxplots
    box_data = [biome_data[b] for b in sorted_biomes]
    
    # Transform ECM if needed
    if 'ecm' in feature_col.lower() and df[feature_col].max() > 100:
        box_data = [[np.sqrt(x) if pd.notna(x) else np.nan for x in b] 
                    for b in box_data]
    
    bp = ax2.boxplot(box_data, labels=sorted_biomes, patch_artist=True,
                     whis=1.5, showfliers=True,
                     flierprops={'marker': 'o', 'markersize': 1, 
                                'markerfacecolor': 'gray', 'alpha': 0.3})
    
    # Color boxes by temperature zone
    colors = []
    for biome in sorted_biomes:
        if 'Tropical' in biome or 'Subtropical' in biome:
            colors.append('coral')
        elif 'Temperate' in biome:
            colors.append('lightgreen')
        elif 'Boreal' in biome:
            colors.append('lightblue')
        elif 'Polar' in biome or 'Subpolar' in biome:
            colors.append('lavender')
        else:
            colors.append('lightgray')
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Formatting
    ax2.set_xlabel('Holdridge Life Zone', fontsize=11)
    
    if 'fungi' in feature_col.lower():
        if 'ecm' in feature_col.lower() and df[feature_col].max() > 100:
            ylabel = 'Predicted Richness Index (√transformed)'
        else:
            ylabel = 'Predicted Richness Index'
    else:
        ylabel = 'Species Count'
    
    ax2.set_ylabel(ylabel, fontsize=11)
    ax2.tick_params(axis='x', rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Title
    title_map = {
        'animals': 'Animal Species Richness',
        'plants': 'Plant Species Richness',
        'am_fungi': 'AM Fungi Richness',
        'ecm_fungi': 'ECM Fungi Richness'
    }
    
    plt.suptitle(f'{title_map.get(name, name)} Distribution Across All Biomes',
                fontsize=14, y=0.98)
    
    plt.tight_layout()
    save_figure(fig, output_dir / f'biome_distribution_{name}.png', dpi=150)
    plt.close(fig)


def _create_nature_style_figure(arr, df, feature_col, name, stats_info, output_dir):
    """Create Nature-style figure with map, density curve, and SIMPLIFIED biome boxplots."""
    import pandas as pd
    import seaborn as sns
    
    # Create figure with complex layout
    fig = plt.figure(figsize=(16, 12))
    
    # Layout: Top = map, Middle = density, Bottom = simplified WWF biome boxplots
    gs = GridSpec(3, 1, figure=fig, height_ratios=[2, 0.5, 1.5], hspace=0.3)
    
    # 1. GLOBAL MAP (top)
    ax_map = fig.add_subplot(gs[0, :])
    masked = np.ma.masked_invalid(arr)
    vmin, vmax = np.percentile(masked.compressed(), [2, 98])
    
    im = ax_map.imshow(masked, cmap='viridis', aspect='equal',
                      extent=[-180, 180, -90, 90],
                      vmin=vmin, vmax=vmax)
    
    title_map = {
        'animals': 'Global Terrestrial Animal Species Richness',
        'plants': 'Global Plant Species Richness',
        'am_fungi': 'Global AM Fungi Richness', 
        'ecm_fungi': 'Global ECM Fungi Richness'
    }
    ax_map.set_title(title_map.get(name, name), fontsize=16, fontweight='bold')
    ax_map.set_xlabel('Longitude')
    ax_map.set_ylabel('Latitude')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.02)
    cbar.set_label('Species Richness', rotation=270, labelpad=20)
    
    # 2. DENSITY CURVE (middle)
    ax_density = fig.add_subplot(gs[1, :])
    
    # Get valid data
    valid_data = df[feature_col].dropna()
    
    # Transform ECM data if needed (square root for visualization like Nature paper)
    if 'ecm' in feature_col.lower() and valid_data.max() > 100:
        valid_data = np.sqrt(valid_data)
        ax_density.set_xlabel('Species Richness (√transformed)', fontsize=11)
    else:
        ax_density.set_xlabel('Species Richness', fontsize=11)
    
    # Plot density
    valid_data.plot(kind='density', ax=ax_density, color='steelblue', linewidth=2)
    ax_density.fill_between(ax_density.get_lines()[0].get_xdata(),
                           ax_density.get_lines()[0].get_ydata(),
                           alpha=0.3, color='steelblue')
    ax_density.set_ylabel('Density', fontsize=11)
    ax_density.set_title('Global Distribution of Richness Estimates', fontsize=12)
    ax_density.grid(True, alpha=0.3)
    
    # 3. BIOME BOXPLOTS (bottom)
    ax_box = fig.add_subplot(gs[2, :])
    
    # Prepare data for boxplot
    biome_data = df.groupby('biome')[feature_col].apply(list).to_dict()
    
    # Remove biomes with too few samples (like Nature paper)
    biome_data = {k: v for k, v in biome_data.items() 
                  if len([x for x in v if pd.notna(x)]) >= 10}
    
    # Sort biomes by median richness
    biome_medians = {k: np.median([x for x in v if pd.notna(x)]) 
                     for k, v in biome_data.items()}
    sorted_biomes = sorted(biome_medians.keys(), key=lambda x: biome_medians[x])
    
    # Create boxplot data
    box_data = [biome_data[b] for b in sorted_biomes]
    
    # Transform ECM if needed
    if 'ecm' in feature_col.lower() and df[feature_col].max() > 100:
        box_data = [[np.sqrt(x) if pd.notna(x) else np.nan for x in b] 
                    for b in box_data]
    
    # Create boxplot WITH outliers (Nature standard: 1.5 × IQR)
    bp = ax_box.boxplot(box_data, labels=sorted_biomes, patch_artist=True,
                        whis=1.5,  # 1.5 × IQR for whiskers (Nature standard)
                        showfliers=True,  # Show outliers
                        flierprops={'marker': 'o', 'markersize': 2, 
                                   'markerfacecolor': 'gray', 'alpha': 0.5})
    
    # Style boxplots
    for patch in bp['boxes']:
        patch.set_facecolor('lightsteelblue')
        patch.set_alpha(0.7)
    
    # Formatting with correct labels
    ax_box.set_xlabel('Biome (Simplified WWF Categories)', fontsize=11)
    
    # Different labels for different data types
    if 'fungi' in feature_col.lower():
        if 'ecm' in feature_col.lower() and df[feature_col].max() > 100:
            ylabel = 'Predicted Richness Index (√transformed)'
        else:
            ylabel = 'Predicted Richness Index'
    else:
        ylabel = 'Species Count'
    
    ax_box.set_ylabel(ylabel, fontsize=11)
    ax_box.set_title('Richness Patterns Across Terrestrial Biomes', fontsize=12)
    ax_box.tick_params(axis='x', rotation=45)
    plt.setp(ax_box.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
    ax_box.grid(True, alpha=0.3, axis='y')
    
    # Add sample sizes
    for i, biome in enumerate(sorted_biomes):
        n = len([x for x in biome_data[biome] if pd.notna(x)])
        ax_box.text(i+1, ax_box.get_ylim()[0], f'n={n}', 
                   ha='center', va='top', fontsize=8, color='gray')
    
    # Add data description note
    fig.text(0.02, 0.02, 
            'Note: Animal/plant values are species counts. Fungi values are predicted richness indices.\n' + 
            'Boxplots show median (center), quartiles (box), and 1.5×IQR whiskers with outliers.',
            fontsize=8, style='italic', wrap=True)
    
    plt.suptitle(f'{title_map.get(name, name)} - Biome Analysis', 
                fontsize=18, y=0.98)
    
    plt.tight_layout()
    save_figure(fig, output_dir / f'nature_style_{name}.png', dpi=300)
    plt.close(fig)


def analyze_global_maps(
    data_source: Dict[str, str],
    output_dir: Union[str, Path],
    **kwargs
) -> Dict[str, Any]:
    """
    Create global map visualizations.
    
    Args:
        data_source: Dict of dataset names to raster paths
        output_dir: Output directory
        
    Returns:
        Map statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load rasters WITH DOWNSAMPLING for visualization
    import rasterio
    data = {}
    stats = {}
    
    # Target resolution for visualization (max 1000x500 pixels)
    max_viz_width = 1000
    max_viz_height = 500
    
    for name, path in data_source.items():
        if path is None:
            continue
        
        # Open without loading
        with rasterio.open(path) as src:
            # Calculate downsampling factor
            downsample = max(1, max(src.width // max_viz_width, 
                                   src.height // max_viz_height))
            
            # Read downsampled for visualization
            arr = src.read(1, out_shape=(src.height // downsample, 
                                        src.width // downsample))
            data[name] = arr
            
            # For stats, use windowed reading (more accurate than downsampled)
            n = 0
            value_sum = 0.0
            value_min = float('inf')
            value_max = float('-inf')
            
            # Read in chunks for statistics
            chunk_size = 1000
            for row in range(0, src.height, chunk_size):
                window = ((row, min(row + chunk_size, src.height)), (0, src.width))
                chunk = src.read(1, window=window)
                valid = ~np.isnan(chunk) & (chunk != src.nodata if src.nodata else True)
                
                if valid.sum() > 0:
                    n += valid.sum()
                    value_sum += chunk[valid].sum()
                    value_min = min(value_min, chunk[valid].min())
                    value_max = max(value_max, chunk[valid].max())
            
            if n > 0:
                stats[name] = {
                    'min': float(value_min),
                    'max': float(value_max),
                    'mean': float(value_sum / n),
                    'coverage': float(100 * n / (src.height * src.width)),
                    'original_shape': src.shape,
                    'display_shape': arr.shape,
                    'downsample_factor': downsample
                }
    
    if not data:
        return {'error': 'No data loaded'}
    
    # Create INDIVIDUAL high-quality maps for each dataset
    for name, arr in data.items():
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Mask invalid values
        masked = np.ma.masked_invalid(arr)
        
        # Use robust colormap with percentile clipping
        vmin, vmax = np.percentile(masked.compressed(), [2, 98])
        
        # Better titles
        title_map = {
            'animals': 'Global Terrestrial Animal Species Richness',
            'plants': 'Global Plant Species Richness', 
            'am_fungi': 'Global Arbuscular Mycorrhizal (AM) Fungi Richness',
            'ecm_fungi': 'Global Ectomycorrhizal (ECM) Fungi Richness',
            'am_fungi_richness': 'Global AM Fungi Richness',
            'ecm_fungi_richness': 'Global ECM Fungi Richness'
        }
        title = title_map.get(name, name.replace('_', ' ').title())
        
        # Plot with better styling
        im = ax.imshow(masked, cmap='viridis', aspect='equal', 
                      extent=[-180, 180, -90, 90],
                      vmin=vmin, vmax=vmax)
        
        # Formatting
        ax.set_title(f'{title}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add statistics to title
        ax.text(0.02, 0.98, f'Min: {stats[name]["min"]:.0f}\nMax: {stats[name]["max"]:.0f}\nMean: {stats[name]["mean"]:.0f}',
                transform=ax.transAxes, fontsize=10, va='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Colorbar with label
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Species Richness', rotation=270, labelpad=20, fontsize=12)
        if stats[name]['max'] > 1000:
            cbar.formatter.set_powerlimits((0, 0))
        
        plt.tight_layout()
        save_figure(fig, output_dir / f'global_map_{name}.png', dpi=150)
        plt.close(fig)
    
    # Create Nature-style combined figure with map + density + biome boxplots
    # IF we have parquet data with biome information
    parquet_path = Path('biodiversity_climate_production_1x_corrected.parquet')
    
    if parquet_path.exists():
        try:
            import pandas as pd
            from src.biodiversity_analysis.statistical.holdridge_classifier import classify_holdridge
            
            # Load parquet for biome analysis
            # Sample stratified by location for representative data
            print(f"Loading biome data from {parquet_path}...")
            
            df = pd.read_parquet(parquet_path, 
                                columns=['x', 'y', 'bio01', 'bio12', 'elevation',
                                        'animal_richness', 'plant_richness', 
                                        'am_fungi_richness', 'ecm_fungi_richness'])
            
            # Stratified sampling by grid cells for representation
            # Create grid cells (10x10 degree cells)
            df['grid_x'] = (df['x'] // 10).astype(int)
            df['grid_y'] = (df['y'] // 10).astype(int)
            
            # Sample up to 100 points per grid cell
            sampled = df.groupby(['grid_x', 'grid_y']).apply(
                lambda x: x.sample(min(100, len(x)), random_state=42)
            ).reset_index(drop=True)
            
            df = sampled.drop(['grid_x', 'grid_y'], axis=1)
            print(f"Using {len(df):,} representative samples from global grid")
            
            # Note: For fungi, these are predicted richness indices (not species counts)
            # For animals/plants, these are actual species counts
            
            # Classify biomes
            # IMPORTANT: bio01 is temperature × 10 (WorldClim standard)
            from src.biodiversity_analysis.statistical.holdridge_classifier import holdridge_to_wwf
            
            holdridge_biomes = classify_holdridge(
                df['bio01'].values / 10.0,  # Convert to °C
                df['bio12'].values,         # Already in mm
                df['elevation'].values      # Already in meters
            )
            
            # Use simplified WWF biomes for visualization
            df['biome'] = holdridge_to_wwf(holdridge_biomes)
            df['biome_detailed'] = holdridge_biomes  # Keep detailed for reference
            
            # Create Nature-style figure for each dataset
            for name, arr in data.items():
                # Map feature names
                feature_map = {
                    'animals': 'animal_richness',
                    'plants': 'plant_richness',
                    'am_fungi': 'am_fungi_richness',
                    'ecm_fungi': 'ecm_fungi_richness'
                }
                
                if name in feature_map and feature_map[name] in df.columns:
                    # Create Nature-style figure with simplified WWF biomes
                    _create_nature_style_figure(
                        arr, df, feature_map[name], name, 
                        stats[name], output_dir
                    )
                    
                    # Also create standalone detailed biome distribution chart
                    _create_standalone_biome_chart(
                        df, feature_map[name], name, output_dir
                    )
                    
        except Exception as e:
            logger.warning(f"Could not create Nature-style figures: {e}")
    
    # Save results
    results = {
        'datasets': list(data.keys()),
        'statistics': stats
    }
    save_results(results, output_dir / 'global_map_results.json')
    
    return results