#!/usr/bin/env python3
"""
Plot LMG (Lindeman-Merenda-Gold) Feature Importance Results
Generates 4 plots from JSON results:
1. Fungi response feature importance (bar chart)
2. Plant response feature importance (bar chart)
3. Animal response feature importance (bar chart)
4. Combined stacked bar chart showing all three models with R² decomposition
"""

import json
from src.abstractions import load_json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys
from datetime import datetime
import glob

# Set Seaborn style for beautiful plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# Use a carefully designed 10-color palette for maximum distinction
# Based on research in color perception and accessibility
# Combining ColorBrewer's Set1, Dark2, and custom colors for optimal distinction

# Define consistent color scheme with maximally distinct colors
PREDICTOR_COLORS = {
    # Spatial - Purple (distinct from all others)
    'spatial': '#9467BD',           # Medium Purple
    
    # Earth/Land predictors - Brown/Orange tones
    'topographic': '#8C564B',       # Saddle Brown (distinct brown)
    'soil': '#D2691E',              # Chocolate (lighter brown/orange)
    
    # Water/Climate predictors - Blue tones  
    'precipitation': '#1F77B4',      # Strong Blue
    
    # Temperature predictors - Red/Pink tones
    'annual_temp': '#E41A1C',       # Bright Red
    'seasonal_temp': '#FF69B4',     # Hot Pink (very distinct from red)
    
    # Energy - Orange (distinct from browns)
    'pet': '#FF7F0E',               # Vivid Orange
    
    # Biodiversity - Green/Teal/Yellow for the three life forms
    'plant_biodiv': '#2CA02C',      # Forest Green
    'animal_biodiv': '#17BECF',     # Cyan/Teal (distinct from greens and blues)
    'fungi_biodiv': '#BCBD22'       # Yellow-Green (distinct from pure green)
}

# Pretty names for display
PREDICTOR_NAMES = {
    'spatial': 'Spatial (MESF)',
    'topographic': 'Topographic',
    'soil': 'Soil',
    'precipitation': 'Precipitation',
    'annual_temp': 'Annual Temperature',
    'seasonal_temp': 'Seasonal Temperature',
    'pet': 'PET',
    'plant_biodiv': 'Plant Biodiversity',
    'animal_biodiv': 'Animal Biodiversity',
    'fungi_biodiv': 'Fungi Biodiversity'
}

def load_lmg_results(response_type):
    """Load the most recent LMG results for a given response type."""
    # Define search patterns for different response types
    if response_type == 'fungi':
        pattern = '/scratch/yl998/lmg_results/lmg_importance_fixed_theta_*.json'
    elif response_type == 'plant':
        pattern = '/scratch/yl998/lmg_results/lmg_importance_plant_*.json'
    elif response_type == 'animal':
        pattern = '/scratch/yl998/lmg_results/lmg_importance_animal_*.json'
    else:
        raise ValueError(f"Unknown response type: {response_type}")
    
    # Find all matching files
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No LMG results found for {response_type} response. Pattern: {pattern}")
    
    # Get the most recent file (by timestamp in filename)
    latest_file = sorted(files)[-1]
    print(f"Loading {response_type} results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def plot_individual_importance(data, response_type, output_dir):
    """Create individual bar chart for one model's feature importance using Seaborn."""
    
    # Extract importance values and sort by importance
    importance = data['importance']
    sorted_predictors = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    # Prepare data for plotting as DataFrame (Seaborn works better with DataFrames)
    plot_data = pd.DataFrame([
        {
            'Predictor': PREDICTOR_NAMES.get(pred, pred),
            'Importance': val * 100,  # Convert to percentage
            'predictor_key': pred  # Keep original key for color mapping
        }
        for pred, val in sorted_predictors
    ])
    
    # Get colors in the right order
    colors = [PREDICTOR_COLORS.get(pred, sns.color_palette()[0]) for pred in plot_data['predictor_key']]
    
    # Create figure with Seaborn styling
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # Create horizontal barplot with Seaborn
    sns.barplot(data=plot_data, y='Predictor', x='Importance', 
                palette=colors, orient='h', ax=ax,
                edgecolor='black', linewidth=0.8)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(plot_data.iterrows()):
        ax.text(row['Importance'] + 0.5, i, f"{row['Importance']:.1f}%",
                va='center', fontsize=10, fontweight='bold')
    
    # Customize plot with better typography
    ax.set_xlabel('Relative Importance (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('')  # Remove y-label as predictor names are self-explanatory
    ax.set_title(f'LMG Feature Importance - {response_type.capitalize()} Response\n' + 
                 f'Total R² = {data["total_r2"]:.3f}', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(0, plot_data['Importance'].max() * 1.12)
    
    # Style the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Add subtle grid
    ax.grid(True, axis='x', alpha=0.2, linestyle='--')
    
    # Add metadata text with better styling
    n_samples = data['metadata']['n_samples']
    n_models = data['metadata']['n_models']
    ax.text(0.98, 0.02, f'n = {n_samples:,} samples | {n_models:,} models fitted',
            transform=ax.transAxes, ha='right', fontsize=9, 
            alpha=0.6, style='italic')
    
    plt.tight_layout()
    
    # Save figure with high quality
    output_path = output_dir / f'lmg_importance_{response_type}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

def plot_combined_stacked_bars(fungi_data, plant_data, animal_data, output_dir):
    """Create combined stacked bar chart showing R² decomposition for all models with Seaborn styling."""
    
    # Set figure style
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Model names and their data
    models = ['Fungi', 'Plant', 'Animal']
    model_data = [fungi_data, plant_data, animal_data]
    
    # Get all unique predictors across models
    all_predictors = set()
    for data in model_data:
        all_predictors.update(data['importance'].keys())
    
    # Sort predictors for consistent ordering (by average importance)
    predictor_avg_importance = {}
    for predictor in all_predictors:
        avg_imp = np.mean([d['importance'].get(predictor, 0) for d in model_data])
        predictor_avg_importance[predictor] = avg_imp
    
    sorted_predictors = sorted(all_predictors, key=lambda x: predictor_avg_importance[x])
    
    # Prepare data for stacked bars
    bar_width = 0.65
    x_positions = np.arange(len(models))
    
    # Create stacked bars with beautiful colors
    bottom = np.zeros(len(models))
    
    # Store bars for legend
    bar_handles = []
    bar_labels = []
    
    for predictor in sorted_predictors:
        values = []
        for data in model_data:
            # Get importance value or 0 if predictor not in model
            val = data['importance'].get(predictor, 0) * 100
            values.append(val)
        
        # Create bars with enhanced styling
        bars = ax.bar(x_positions, values, bar_width, bottom=bottom,
                     color=PREDICTOR_COLORS.get(predictor, sns.color_palette()[0]),
                     edgecolor='white', linewidth=1.5,
                     label=PREDICTOR_NAMES.get(predictor, predictor),
                     alpha=0.95)
        
        # Only add to legend if predictor has non-zero contribution
        if any(v > 0.5 for v in values):  # Only show in legend if > 0.5%
            bar_handles.append(bars[0])
            bar_labels.append(PREDICTOR_NAMES.get(predictor, predictor))
        
        # Update bottom for next stack
        bottom += values
    
    # Add total R² values on top of bars with better styling
    for i, (model, data) in enumerate(zip(models, model_data)):
        total_r2 = data['total_r2']
        ax.text(i, total_r2 * 100 + 1.5, f'R² = {total_r2:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=13,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='gray', alpha=0.8))
    
    # Add horizontal line at 100% for reference
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    
    # Customize plot with better typography
    ax.set_ylabel('Cumulative R² (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Response Variable', fontsize=14, fontweight='bold')
    ax.set_title('LMG Feature Importance Comparison\nR² Decomposition Across Models', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{m}\n(R²={d["total_r2"]:.3f})' 
                        for m, d in zip(models, model_data)], fontsize=12)
    ax.set_ylim(0, 105)
    
    # Style the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    
    # Add legend with better styling
    legend = ax.legend(bar_handles[::-1], bar_labels[::-1],  # Reverse order for better reading
                      loc='upper left', bbox_to_anchor=(1.02, 1),
                      ncol=1, frameon=True, fancybox=True, shadow=True,
                      title='Predictors', title_fontsize=12, fontsize=11)
    legend.get_frame().set_alpha(0.95)
    
    # Add subtle grid
    ax.grid(True, axis='y', alpha=0.2, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add sample size information with better styling
    sample_text = "Sample size: 100,000 rows | Fixed θ = 18.83 | 3,628 orderings sampled"
    ax.text(0.5, -0.1, sample_text, transform=ax.transAxes,
            ha='center', fontsize=10, alpha=0.7, style='italic')
    
    plt.tight_layout()
    
    # Save figure with high quality
    output_path = output_dir / 'lmg_importance_combined.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Reset style for next plots
    sns.set_style("whitegrid")

def create_importance_table(fungi_data, plant_data, animal_data, output_dir):
    """Create a beautiful heatmap comparison of importance values using Seaborn."""
    
    # Get all predictors
    all_predictors = set()
    for data in [fungi_data, plant_data, animal_data]:
        all_predictors.update(data['importance'].keys())
    
    # Create DataFrame for heatmap
    predictor_names = []
    fungi_values = []
    plant_values = []
    animal_values = []
    
    for predictor in sorted(all_predictors):
        predictor_names.append(PREDICTOR_NAMES.get(predictor, predictor))
        fungi_values.append(fungi_data['importance'].get(predictor, 0) * 100)
        plant_values.append(plant_data['importance'].get(predictor, 0) * 100)
        animal_values.append(animal_data['importance'].get(predictor, 0) * 100)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Fungi': fungi_values,
        'Plant': plant_values,
        'Animal': animal_values
    }, index=predictor_names)
    
    # Sort by average importance
    df['Average'] = df.mean(axis=1)
    df = df.sort_values('Average', ascending=False)
    df = df.drop('Average', axis=1)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), 
                                   gridspec_kw={'width_ratios': [1, 1.2]})
    
    # First subplot: Heatmap with better color scheme
    # Use 'RdYlBu_r' for a diverging palette or 'viridis' for perceptually uniform
    sns.heatmap(df, annot=True, fmt='.1f', cmap='RdBu_r', 
                cbar_kws={'label': 'Importance (%)'}, 
                linewidths=1, linecolor='white',
                ax=ax1, vmin=0, vmax=20, center=10)
    ax1.set_title('Feature Importance Heatmap', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Response Variable', fontsize=12, fontweight='bold')
    ax1.set_ylabel('')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
    
    # Second subplot: Grouped bar chart
    df_plot = df.reset_index()
    df_melted = pd.melt(df_plot, id_vars='index', var_name='Model', value_name='Importance')
    
    # Create grouped barplot with distinct colors for each response model
    model_palette = {
        'Fungi': '#8B4513',  # Brown for fungi (soil dweller)
        'Plant': '#228B22',  # Green for plants
        'Animal': '#4169E1'  # Royal blue for animals
    }
    sns.barplot(data=df_melted, x='index', y='Importance', hue='Model',
                palette=model_palette, ax=ax2, edgecolor='black', linewidth=0.8)
    ax2.set_title('Feature Importance Comparison', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('')
    ax2.set_ylabel('Importance (%)', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.legend(title='Response', fontsize=10, title_fontsize=11)
    ax2.grid(True, axis='y', alpha=0.2, linestyle='--')
    
    # Add R² annotations
    r2_text = f"R² values:\nFungi: {fungi_data['total_r2']:.3f}\nPlant: {plant_data['total_r2']:.3f}\nAnimal: {animal_data['total_r2']:.3f}"
    ax2.text(0.98, 0.98, r2_text, transform=ax2.transAxes,
             ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                      edgecolor='gray', alpha=0.9))
    
    # Overall title
    fig.suptitle('LMG Feature Importance Analysis - Comprehensive View', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figure with high quality
    output_path = output_dir / 'lmg_importance_table.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    """Main function to generate all plots."""
    
    # Create output directory
    output_dir = Path('/scratch/yl998/lmg_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("LMG Feature Importance Visualization")
    print("="*60)
    
    try:
        # Load all results
        print("\nLoading LMG results...")
        fungi_data = load_lmg_results('fungi')
        plant_data = load_lmg_results('plant')
        animal_data = load_lmg_results('animal')
        
        print("\nGenerating plots...")
        
        # Create individual plots
        plot_individual_importance(fungi_data, 'fungi', output_dir)
        plot_individual_importance(plant_data, 'plant', output_dir)
        plot_individual_importance(animal_data, 'animal', output_dir)
        
        # Create combined plot
        plot_combined_stacked_bars(fungi_data, plant_data, animal_data, output_dir)
        
        # Create comparison table
        create_importance_table(fungi_data, plant_data, animal_data, output_dir)
        
        print("\n" + "="*60)
        print("SUCCESS: All plots generated successfully!")
        print(f"Output directory: {output_dir}")
        print("Generated files:")
        print("  - lmg_importance_fungi.png")
        print("  - lmg_importance_plant.png")
        print("  - lmg_importance_animal.png")
        print("  - lmg_importance_combined.png")
        print("  - lmg_importance_table.png")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please ensure all LMG analyses have been completed.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()