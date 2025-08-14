"""
Correlation analysis - Linus style.

"If you need more than 3 levels of indentation, you're screwed"
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from src.biodiversity_analysis.statistical.data_utils import load_data, save_results, save_figure
from src.biodiversity_analysis.shared.data.parquet_loader import ParquetLoader
from src.biodiversity_analysis.shared.data.raster_loader import RasterLoader

logger = logging.getLogger(__name__)


def analyze_correlation(
    data_source: Union[str, Dict[str, str], pd.DataFrame],
    output_dir: Union[str, Path],
    method: str = 'all',
    columns: Optional[List[str]] = None,
    sample_size: Optional[int] = None,
    plot: bool = True
) -> Dict[str, Any]:
    """
    Perform correlation analysis. No class needed.
    
    Args:
        data_source: Data source (parquet path, raster dict, or DataFrame)
        output_dir: Where to save results
        method: 'pearson', 'spearman', 'kendall', or 'all'
        columns: Specific columns to analyze
        sample_size: Random sample size for large datasets
        plot: Whether to generate plots
        
    Returns:
        Dictionary of correlation results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    parquet_loader = ParquetLoader()
    raster_loader = RasterLoader()
    data = load_data(data_source, parquet_loader, raster_loader)
    
    # Convert to DataFrame if needed
    if isinstance(data, dict):
        # Dict of rasters - flatten to DataFrame
        data = pd.DataFrame({name: arr.flatten() for name, arr in data.items()})
    elif isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    
    # Filter columns if specified
    if columns:
        data = data[columns]
    
    # Sample if needed
    if sample_size and len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=42)
        logger.info(f"Sampled {sample_size} points from {len(data)} total")
    
    # Get numeric columns only
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Compute correlations
    results = {}
    
    if method in ['pearson', 'all']:
        results['pearson'] = numeric_data.corr(method='pearson').to_dict()
    
    if method in ['spearman', 'all']:
        results['spearman'] = numeric_data.corr(method='spearman').to_dict()
    
    if method in ['kendall', 'all']:
        results['kendall'] = numeric_data.corr(method='kendall').to_dict()
    
    # Add basic statistics
    results['statistics'] = {
        'n_samples': len(data),
        'n_features': len(numeric_data.columns),
        'features': list(numeric_data.columns)
    }
    
    # Generate plot if requested
    if plot and len(numeric_data.columns) > 1:
        fig = _plot_correlation_matrix(
            numeric_data.corr(method='pearson' if method == 'all' else method),
            title=f"Correlation Matrix ({method})"
        )
        save_figure(fig, output_dir / 'correlation_matrix.png')
        plt.close(fig)
    
    # Save results
    save_results(results, output_dir / 'correlation_results.json')
    
    return results


def _plot_correlation_matrix(corr_matrix: pd.DataFrame, title: str) -> plt.Figure:
    """Create correlation matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Set labels
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr_matrix.columns)
    
    # Add values
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            value = corr_matrix.iloc[i, j]
            text = ax.text(j, i, f'{value:.2f}',
                          ha='center', va='center',
                          color='white' if abs(value) > 0.5 else 'black')
    
    # Add colorbar and title
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    
    return fig