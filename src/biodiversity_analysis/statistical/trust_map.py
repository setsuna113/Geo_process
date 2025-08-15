"""
Trust map analysis - Pure functions, no classes.

"Object-oriented programming is a hoax."
Classes are just namespaces with extra complexity.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
import logging

from .data_utils import save_results, save_figure

logger = logging.getLogger(__name__)


def analyze_trust_maps(
    data_source: Dict[str, np.ndarray],
    output_dir: Path,
    downsample: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze uncertainty in biodiversity data.
    
    No class needed. Just functions doing their job.
    
    Args:
        data_source: Dict of dataset names to arrays
        output_dir: Output directory
        downsample: Downsampling factor for visualization
        
    Returns:
        Trust analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Process each dataset
    for name, data_array in data_source.items():
        if data_array is None:
            continue
            
        # Skip CV maps themselves
        if '_cv' in name:
            continue
            
        logger.info(f"Processing trust map for {name}")
        
        # Look for corresponding CV map
        cv_name = f"{name}_cv"
        cv_array = data_source.get(cv_name)
        
        if cv_array is not None:
            # Compute trust metrics
            trust_metrics = _compute_trust_metrics(data_array, cv_array)
            
            # Generate visualizations
            if downsample > 1:
                data_viz = data_array[::downsample, ::downsample]
                cv_viz = cv_array[::downsample, ::downsample] if cv_array is not None else None
            else:
                data_viz = data_array
                cv_viz = cv_array
            
            # Create plots
            _plot_trust_analysis(
                data_viz, cv_viz, name, 
                output_dir, trust_metrics
            )
            
            results[name] = trust_metrics
        else:
            # No uncertainty data available
            basic_metrics = _compute_basic_metrics(data_array)
            results[name] = basic_metrics
            
            # Simple visualization
            if downsample > 1:
                data_viz = data_array[::downsample, ::downsample]
            else:
                data_viz = data_array
                
            _plot_basic_map(data_viz, name, output_dir, basic_metrics)
    
    # Combined analysis if we have multiple datasets
    if len(results) > 1:
        combined = _analyze_combined_trust(data_source, downsample)
        results['combined'] = combined
        
        # Plot combined
        _plot_combined_trust(combined, output_dir)
    
    # Save results
    save_results(results, output_dir / 'trust_metrics.json')
    
    return results


def _compute_trust_metrics(data: np.ndarray, cv: np.ndarray) -> Dict[str, Any]:
    """Compute trust metrics from data and CV."""
    # Remove invalid pixels
    valid_mask = ~np.isnan(data) & ~np.isnan(cv) & (data > 0)
    
    if not valid_mask.any():
        return {'error': 'No valid data'}
    
    data_valid = data[valid_mask]
    cv_valid = cv[valid_mask]
    
    # Compute metrics
    metrics = {
        'mean_cv': float(np.mean(cv_valid)),
        'median_cv': float(np.median(cv_valid)),
        'std_cv': float(np.std(cv_valid)),
        'high_uncertainty_fraction': float(np.mean(cv_valid > 0.5)),
        'very_high_uncertainty_fraction': float(np.mean(cv_valid > 1.0)),
        'data_coverage': float(valid_mask.mean()),
        'trust_score': float(1.0 / (1.0 + np.median(cv_valid))),  # Simple trust score
        'n_pixels': int(valid_mask.sum())
    }
    
    # Spatial correlation of uncertainty
    if cv_valid.shape[0] > 100:
        # Sample for efficiency
        sample_idx = np.random.choice(len(cv_valid), min(1000, len(cv_valid)), replace=False)
        metrics['uncertainty_data_correlation'] = float(
            np.corrcoef(data_valid[sample_idx], cv_valid[sample_idx])[0, 1]
        )
    
    return metrics


def _compute_basic_metrics(data: np.ndarray) -> Dict[str, Any]:
    """Compute basic metrics when no CV available."""
    valid_mask = ~np.isnan(data)
    
    if not valid_mask.any():
        return {'error': 'No valid data'}
    
    data_valid = data[valid_mask]
    
    return {
        'mean': float(np.mean(data_valid)),
        'median': float(np.median(data_valid)),
        'std': float(np.std(data_valid)),
        'min': float(np.min(data_valid)),
        'max': float(np.max(data_valid)),
        'data_coverage': float(valid_mask.mean()),
        'n_pixels': int(valid_mask.sum()),
        'note': 'No uncertainty data available'
    }


def _analyze_combined_trust(
    data_source: Dict[str, np.ndarray], 
    downsample: int
) -> Dict[str, Any]:
    """Analyze combined trust across all datasets."""
    combined_trust = []
    
    for name, data_array in data_source.items():
        if '_cv' in name or data_array is None:
            continue
            
        cv_name = f"{name}_cv"
        cv_array = data_source.get(cv_name)
        
        if cv_array is not None:
            # Compute average trust
            valid_mask = ~np.isnan(cv_array) & (cv_array > 0)
            if valid_mask.any():
                trust = 1.0 / (1.0 + cv_array[valid_mask].mean())
                combined_trust.append({
                    'dataset': name,
                    'trust_score': float(trust),
                    'mean_cv': float(cv_array[valid_mask].mean())
                })
    
    if not combined_trust:
        return {'error': 'No uncertainty data available'}
    
    # Sort by trust score
    combined_trust.sort(key=lambda x: x['trust_score'], reverse=True)
    
    return {
        'rankings': combined_trust,
        'average_trust': float(np.mean([x['trust_score'] for x in combined_trust])),
        'most_trusted': combined_trust[0]['dataset'],
        'least_trusted': combined_trust[-1]['dataset']
    }


def _plot_trust_analysis(
    data: np.ndarray,
    cv: np.ndarray,
    name: str,
    output_dir: Path,
    metrics: Dict[str, Any]
) -> None:
    """Plot trust analysis visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Data
    im1 = axes[0].imshow(data, cmap='viridis', aspect='auto')
    axes[0].set_title(f'{name} - Data')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # CV/Uncertainty
    im2 = axes[1].imshow(cv, cmap='hot', aspect='auto')
    axes[1].set_title(f'{name} - Coefficient of Variation')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    
    # Trust map (inverse CV)
    trust = 1.0 / (1.0 + cv)
    im3 = axes[2].imshow(trust, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[2].set_title(f'{name} - Trust Score')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], shrink=0.8)
    
    # Add metrics text
    fig.text(0.5, 0.02, 
             f"Trust Score: {metrics.get('trust_score', 0):.2f} | "
             f"Mean CV: {metrics.get('mean_cv', 0):.2f} | "
             f"High Uncertainty: {metrics.get('high_uncertainty_fraction', 0)*100:.1f}%",
             ha='center', fontsize=10)
    
    plt.tight_layout()
    save_figure(fig, output_dir / f'trust_{name}.png')
    plt.close(fig)


def _plot_basic_map(
    data: np.ndarray,
    name: str,
    output_dir: Path,
    metrics: Dict[str, Any]
) -> None:
    """Plot basic map when no uncertainty available."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(data, cmap='viridis', aspect='auto')
    ax.set_title(f'{name} - No uncertainty data available')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Add metrics
    fig.text(0.5, 0.02,
             f"Mean: {metrics.get('mean', 0):.2f} | "
             f"Std: {metrics.get('std', 0):.2f} | "
             f"Coverage: {metrics.get('data_coverage', 0)*100:.1f}%",
             ha='center', fontsize=10)
    
    plt.tight_layout()
    save_figure(fig, output_dir / f'map_{name}.png')
    plt.close(fig)


def _plot_combined_trust(combined: Dict[str, Any], output_dir: Path) -> None:
    """Plot combined trust analysis."""
    if 'rankings' not in combined:
        return
    
    rankings = combined['rankings']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = [r['dataset'] for r in rankings]
    trust_scores = [r['trust_score'] for r in rankings]
    
    bars = ax.barh(datasets, trust_scores, color='green', alpha=0.7)
    
    # Color code by trust level
    for i, (bar, score) in enumerate(zip(bars, trust_scores)):
        if score < 0.5:
            bar.set_color('red')
        elif score < 0.7:
            bar.set_color('orange')
        else:
            bar.set_color('green')
    
    ax.set_xlabel('Trust Score')
    ax.set_title('Dataset Trust Rankings')
    ax.set_xlim(0, 1)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, trust_scores)):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.2f}', va='center')
    
    plt.tight_layout()
    save_figure(fig, output_dir / 'combined_trust_rankings.png')
    plt.close(fig)


# For backward compatibility
def TrustMap(downsample: int = 10):
    """
    Fake class for backward compatibility.
    Will be removed in v3.0.
    """
    logger.warning("TrustMap class is deprecated. Use analyze_trust_maps() directly.")
    
    class _FakeTrustMap:
        def __init__(self, downsample=10):
            self.downsample = downsample
            
        def run_complete_analysis(self, data_source, output_dir):
            return analyze_trust_maps(data_source, output_dir, self.downsample)
    
    return _FakeTrustMap(downsample)