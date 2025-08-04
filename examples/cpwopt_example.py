#!/usr/bin/env python
"""
Example script demonstrating CP-WOPT tensor factorization for biodiversity analysis.

This example shows how to:
1. Generate synthetic biodiversity data
2. Run CP-WOPT analysis
3. Interpret the results
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import CP-WOPT components
from src.biodiversity_analysis.methods.cpwopt.tests.test_data_generator import (
    SyntheticBiodiversityGenerator, SyntheticDataConfig
)
from src.biodiversity_analysis.preprocessing.tensor_preprocessing import TensorPreprocessor
from src.biodiversity_analysis.methods.cpwopt.core import CPWOPT


def generate_example_data():
    """Generate synthetic biodiversity data for demonstration."""
    print("Generating synthetic biodiversity data...")
    
    # Configure data generation
    config = SyntheticDataConfig(
        n_samples=5000,        # Number of observations
        n_species=30,          # Number of species
        lat_range=(-30, 30),   # Tropical to subtropical
        lon_range=(-80, -40),  # Eastern South America
        rank=4,                # True number of patterns
        sparsity=0.85,         # 85% sparse (realistic)
        noise_level=0.1,       # Moderate noise
        seed=42                # For reproducibility
    )
    
    # Generate data
    generator = SyntheticBiodiversityGenerator(config)
    df = generator.generate_dataframe()
    
    print(f"Generated {len(df)} observations with {config.n_species} species")
    print(f"Geographic extent: Lat {config.lat_range}, Lon {config.lon_range}")
    
    return df


def preprocess_data(df):
    """Preprocess biodiversity data into tensor format."""
    print("\nPreprocessing data into tensor format...")
    
    # Configure preprocessing
    preprocessor = TensorPreprocessor({
        'lat_bins': 20,
        'lon_bins': 20,
        'aggregation': 'sum',      # Sum observations in each cell
        'transform': 'log1p',      # Log transform for count data
        'lat_range': (-30, 30),
        'lon_range': (-80, -40)
    })
    
    # Convert to tensor
    tensor_data = preprocessor.dataframe_to_tensor(df)
    
    print(f"Tensor shape: {tensor_data.shape}")
    print(f"Sparsity: {(tensor_data.tensor == 0).sum() / tensor_data.tensor.size:.1%}")
    print(f"Features: {len(tensor_data.feature_names)} species")
    
    return tensor_data


def run_cpwopt_analysis(tensor_data):
    """Run CP-WOPT tensor factorization."""
    print("\nRunning CP-WOPT analysis...")
    
    # Configure CP-WOPT
    model = CPWOPT(
        rank=5,                    # Number of components to extract
        max_iterations=100,        # Maximum iterations
        convergence_threshold=1e-4,# Convergence criterion
        initialization='random',   # Initialization method
        random_state=42,          # For reproducibility
        verbose=True              # Show progress
    )
    
    # Fit model
    model.fit(tensor_data)
    
    # Get results
    print(f"\nAnalysis completed in {model.n_iterations_} iterations")
    print(f"Final fit: {model.metrics_.fit:.3f}")
    print(f"Relative error: {model.metrics_.relative_error:.3f}")
    
    return model


def interpret_results(model, tensor_data):
    """Interpret and visualize CP-WOPT results."""
    print("\nInterpreting results...")
    
    # Get factor matrices
    factors = model.get_component_matrices()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('CP-WOPT Analysis Results', fontsize=16)
    
    # 1. Component importance
    ax = axes[0, 0]
    importance = model.metrics_.component_weights
    ax.bar(range(len(importance)), importance)
    ax.set_xlabel('Component')
    ax.set_ylabel('Importance')
    ax.set_title('Component Importance')
    
    # 2. Spatial pattern of first component
    ax = axes[0, 1]
    lat_factor = factors['latitude'][:, 0]
    lon_factor = factors['longitude'][:, 0]
    spatial_pattern = np.outer(lat_factor, lon_factor)
    im = ax.imshow(spatial_pattern, cmap='YlOrRd', aspect='auto')
    ax.set_title('Component 1: Spatial Pattern')
    ax.set_xlabel('Longitude bin')
    ax.set_ylabel('Latitude bin')
    plt.colorbar(im, ax=ax)
    
    # 3. Top species for first component
    ax = axes[0, 2]
    species_loadings = factors['features'][:, 0]
    top_species_idx = np.argsort(np.abs(species_loadings))[-5:]
    top_species = [tensor_data.feature_names[i] for i in top_species_idx]
    top_loadings = species_loadings[top_species_idx]
    
    y_pos = np.arange(len(top_species))
    ax.barh(y_pos, top_loadings)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_species)
    ax.set_xlabel('Loading')
    ax.set_title('Component 1: Top Species')
    
    # 4. Convergence history
    ax = axes[1, 0]
    ax.plot(model.convergence_history_)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Reconstruction Error')
    ax.set_title('Convergence History')
    ax.set_yscale('log')
    
    # 5. Reconstruction quality heatmap
    ax = axes[1, 1]
    reconstructed = model.predict()
    # Average across species for visualization
    original_richness = np.sum(tensor_data.tensor, axis=2)
    reconstructed_richness = np.sum(reconstructed, axis=2)
    
    # Only show where we have data
    mask = np.sum(tensor_data.weights, axis=2) > 0
    quality = np.ones_like(original_richness) * np.nan
    quality[mask] = 1 - np.abs(original_richness[mask] - reconstructed_richness[mask]) / (original_richness[mask] + 1e-10)
    
    im = ax.imshow(quality, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_title('Reconstruction Quality')
    ax.set_xlabel('Longitude bin')
    ax.set_ylabel('Latitude bin')
    plt.colorbar(im, ax=ax, label='Quality (0-1)')
    
    # 6. Component correlations
    ax = axes[1, 2]
    # Calculate correlation between spatial patterns of components
    n_components = model.rank
    correlations = np.zeros((n_components, n_components))
    
    for i in range(n_components):
        for j in range(n_components):
            spatial_i = np.outer(factors['latitude'][:, i], factors['longitude'][:, i]).flatten()
            spatial_j = np.outer(factors['latitude'][:, j], factors['longitude'][:, j]).flatten()
            correlations[i, j] = np.corrcoef(spatial_i, spatial_j)[0, 1]
    
    im = ax.imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title('Component Correlations')
    ax.set_xlabel('Component')
    ax.set_ylabel('Component')
    ax.set_xticks(range(n_components))
    ax.set_yticks(range(n_components))
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    return fig


def print_component_interpretation(model, tensor_data):
    """Print ecological interpretation of components."""
    print("\n" + "="*60)
    print("ECOLOGICAL INTERPRETATION OF COMPONENTS")
    print("="*60)
    
    factors = model.get_component_matrices()
    n_components = model.rank
    
    for comp in range(min(n_components, 3)):  # Show first 3 components
        print(f"\nComponent {comp + 1}:")
        print("-" * 40)
        
        # Spatial characterization
        lat_factor = factors['latitude'][:, comp]
        lon_factor = factors['longitude'][:, comp]
        
        # Find peak latitude
        peak_lat_bin = np.argmax(np.abs(lat_factor))
        lat_range = tensor_data.metadata['lat_range']
        peak_lat = lat_range[0] + peak_lat_bin * (lat_range[1] - lat_range[0]) / len(lat_factor)
        
        # Find peak longitude  
        peak_lon_bin = np.argmax(np.abs(lon_factor))
        lon_range = tensor_data.metadata['lon_range']
        peak_lon = lon_range[0] + peak_lon_bin * (lon_range[1] - lon_range[0]) / len(lon_factor)
        
        print(f"  Spatial center: {peak_lat:.1f}°N, {peak_lon:.1f}°W")
        
        # Characterize spatial pattern
        lat_spread = np.std(lat_factor)
        lon_spread = np.std(lon_factor)
        
        if lat_spread > lon_spread * 1.5:
            pattern = "Latitudinal gradient"
        elif lon_spread > lat_spread * 1.5:
            pattern = "Longitudinal gradient"
        else:
            pattern = "Regional hotspot"
        
        print(f"  Spatial pattern: {pattern}")
        
        # Top species
        species_loadings = factors['features'][:, comp]
        top_species_idx = np.argsort(np.abs(species_loadings))[-5:]
        
        print("  Top associated species:")
        for idx in reversed(top_species_idx):
            species_name = tensor_data.feature_names[idx]
            loading = species_loadings[idx]
            print(f"    - {species_name}: {loading:.3f}")
        
        # Importance
        importance = model.metrics_.component_weights[comp]
        print(f"  Relative importance: {importance:.1%}")


def save_results(model, output_dir='outputs/cpwopt_example'):
    """Save analysis results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving results to {output_path}...")
    
    # Save factor matrices
    factors = model.get_component_matrices()
    for mode_name, factor_matrix in factors.items():
        np.save(output_path / f'{mode_name}_factors.npy', factor_matrix)
    
    # Save metrics
    metrics = {
        'fit': float(model.metrics_.fit),
        'relative_error': float(model.metrics_.relative_error),
        'iterations': int(model.n_iterations_),
        'rank': int(model.rank),
        'component_weights': model.metrics_.component_weights.tolist()
    }
    
    import json
    with open(output_path / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Results saved successfully!")


def main():
    """Run the complete example."""
    print("CP-WOPT Biodiversity Analysis Example")
    print("=====================================\n")
    
    # Generate data
    df = generate_example_data()
    
    # Preprocess
    tensor_data = preprocess_data(df)
    
    # Run analysis
    model = run_cpwopt_analysis(tensor_data)
    
    # Interpret results
    fig = interpret_results(model, tensor_data)
    
    # Print interpretation
    print_component_interpretation(model, tensor_data)
    
    # Save results
    save_results(model)
    
    # Show plots
    plt.savefig('outputs/cpwopt_example/analysis_summary.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nExample completed successfully!")


if __name__ == '__main__':
    main()