"""
Correlation analysis with chunking support - Linus style.

"Solve real problems, not imaginary ones."
Your data is big? Fine, we'll handle it.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from src.biodiversity_analysis.statistical.data_utils import save_results, save_figure
from src.biodiversity_analysis.shared.data.raster_loader import RasterLoader

logger = logging.getLogger(__name__)


def analyze_correlation(
    data_source: Union[str, Dict[str, str], pd.DataFrame],
    output_dir: Union[str, Path],
    method: str = 'all',
    columns: Optional[List[str]] = None,
    sample_size: Optional[int] = None,
    chunk_size: int = 1000000,  # Process 1M pixels at a time
    plot: bool = True
) -> Dict[str, Any]:
    """
    Correlation analysis with memory-efficient chunking.
    
    Args:
        data_source: Data source
        output_dir: Output directory
        method: Correlation method
        columns: Columns to analyze
        sample_size: Sample size (takes precedence over chunking)
        chunk_size: Pixels per chunk for memory efficiency
        plot: Generate plots
        
    Returns:
        Correlation results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # If it's already a DataFrame, use original logic
    if isinstance(data_source, pd.DataFrame):
        return _analyze_dataframe(data_source, output_dir, method, columns, sample_size, plot)
    
    # If it's a parquet file, also use original logic (already efficient)
    if isinstance(data_source, str) and data_source.endswith('.parquet'):
        from src.biodiversity_analysis.shared.data.parquet_loader import ParquetLoader
        loader = ParquetLoader()
        biodiv_data = loader.load(data_source)
        df = pd.DataFrame(biodiv_data.features)
        return _analyze_dataframe(df, output_dir, method, columns, sample_size, plot)
    
    # For raster dict - use chunked processing
    if isinstance(data_source, dict):
        logger.info(f"Using chunked processing with {chunk_size:,} pixels per chunk")
        return _analyze_rasters_chunked(data_source, output_dir, method, 
                                       chunk_size, sample_size, plot)
    
    raise ValueError(f"Unsupported data source: {type(data_source)}")


def _analyze_rasters_chunked(
    raster_paths: Dict[str, str],
    output_dir: Path,
    method: str,
    chunk_size: int,
    sample_size: Optional[int],
    plot: bool
) -> Dict[str, Any]:
    """
    Analyze rasters using chunked processing to avoid memory explosion.
    
    Instead of loading all rasters and flattening (memory bomb),
    we process in chunks and accumulate statistics.
    """
    raster_loader = RasterLoader()
    
    # Load rasters but don't flatten yet
    rasters = {}
    shapes = []
    for name, path_or_array in raster_paths.items():
        if path_or_array is None:
            continue
        # Handle both paths and arrays
        if isinstance(path_or_array, str):
            data = raster_loader.load_single(path_or_array)
        elif isinstance(path_or_array, np.ndarray):
            data = path_or_array
        else:
            continue
        rasters[name] = data
        shapes.append(data.shape)
    
    if not rasters:
        return {'error': 'No valid rasters loaded'}
    
    # Check all rasters have same shape
    if len(set(shapes)) > 1:
        logger.warning(f"Rasters have different shapes: {shapes}")
    
    # Calculate total pixels
    total_pixels = shapes[0][0] * shapes[0][1]
    
    # If sampling requested and smaller than total, use random sampling
    if sample_size and sample_size < total_pixels:
        return _analyze_with_sampling(rasters, output_dir, method, sample_size, plot)
    
    # Otherwise use chunked correlation computation
    logger.info(f"Processing {total_pixels:,} pixels in chunks of {chunk_size:,}")
    
    # For correlation, we need running sums for online algorithm
    n = 0
    sums = {name: 0.0 for name in rasters}
    sums2 = {name: 0.0 for name in rasters}
    cross_products = {}
    
    for i in rasters:
        for j in rasters:
            if i < j:
                cross_products[f"{i}_{j}"] = 0.0
    
    # Process in chunks
    height, width = shapes[0]
    for row_start in range(0, height, chunk_size // width):
        row_end = min(row_start + chunk_size // width, height)
        
        # Extract chunk from each raster
        chunk_data = {}
        for name, raster in rasters.items():
            chunk = raster[row_start:row_end].flatten()
            valid_mask = ~np.isnan(chunk)
            chunk_data[name] = chunk[valid_mask]
        
        # Find common valid pixels
        min_len = min(len(arr) for arr in chunk_data.values())
        if min_len == 0:
            continue
        
        # Truncate to common length
        for name in chunk_data:
            chunk_data[name] = chunk_data[name][:min_len]
        
        # Update statistics
        n += min_len
        for name, values in chunk_data.items():
            sums[name] += values.sum()
            sums2[name] += (values ** 2).sum()
        
        # Update cross products
        for i in rasters:
            for j in rasters:
                if i < j:
                    key = f"{i}_{j}"
                    cross_products[key] += (chunk_data[i] * chunk_data[j]).sum()
    
    # Calculate correlations from accumulated statistics
    correlations = {}
    for i in rasters:
        correlations[i] = {}
        for j in rasters:
            if i == j:
                correlations[i][j] = 1.0
            else:
                key = f"{min(i,j)}_{max(i,j)}"
                if key in cross_products and n > 0:
                    # Pearson correlation from sums
                    mean_i = sums[i] / n
                    mean_j = sums[j] / n
                    var_i = (sums2[i] / n) - mean_i ** 2
                    var_j = (sums2[j] / n) - mean_j ** 2
                    cov = (cross_products[key] / n) - mean_i * mean_j
                    
                    if var_i > 0 and var_j > 0:
                        r = cov / (np.sqrt(var_i) * np.sqrt(var_j))
                        correlations[i][j] = float(np.clip(r, -1, 1))
                    else:
                        correlations[i][j] = 0.0
    
    results = {
        'pearson': correlations,
        'statistics': {
            'n_samples': n,
            'n_features': len(rasters),
            'features': list(rasters.keys()),
            'method': 'chunked_processing',
            'chunk_size': chunk_size
        }
    }
    
    # Generate plot if requested
    if plot and len(rasters) > 1:
        corr_df = pd.DataFrame(correlations)
        fig = _plot_correlation_matrix(corr_df, f"Correlation Matrix (chunked)")
        save_figure(fig, output_dir / 'correlation_matrix.png')
        plt.close(fig)
    
    save_results(results, output_dir / 'correlation_results.json')
    return results


def _analyze_with_sampling(rasters: Dict, output_dir: Path, method: str, 
                          sample_size: int, plot: bool) -> Dict[str, Any]:
    """Use random sampling instead of chunks."""
    # Generate random indices
    height, width = next(iter(rasters.values())).shape
    total_pixels = height * width
    
    np.random.seed(42)
    sample_indices = np.random.choice(total_pixels, size=sample_size, replace=False)
    
    # Extract samples
    sampled_data = {}
    for name, raster in rasters.items():
        flat = raster.flatten()
        sampled_data[name] = flat[sample_indices]
    
    # Create DataFrame and use standard analysis
    df = pd.DataFrame(sampled_data)
    return _analyze_dataframe(df, output_dir, method, None, None, plot)


def _analyze_dataframe(df: pd.DataFrame, output_dir: Path, method: str,
                       columns: Optional[List[str]], sample_size: Optional[int],
                       plot: bool) -> Dict[str, Any]:
    """Standard DataFrame analysis (original logic)."""
    if columns:
        df = df[columns]
    
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        logger.info(f"Sampled {sample_size} points")
    
    numeric_data = df.select_dtypes(include=[np.number])
    
    results = {}
    if method in ['pearson', 'all']:
        results['pearson'] = numeric_data.corr(method='pearson').to_dict()
    if method in ['spearman', 'all']:
        results['spearman'] = numeric_data.corr(method='spearman').to_dict()
    if method in ['kendall', 'all']:
        results['kendall'] = numeric_data.corr(method='kendall').to_dict()
    
    results['statistics'] = {
        'n_samples': len(df),
        'n_features': len(numeric_data.columns),
        'features': list(numeric_data.columns)
    }
    
    if plot and len(numeric_data.columns) > 1:
        fig = _plot_correlation_matrix(
            numeric_data.corr(method='pearson' if method == 'all' else method),
            f"Correlation Matrix ({method})"
        )
        save_figure(fig, output_dir / 'correlation_matrix.png')
        plt.close(fig)
    
    save_results(results, output_dir / 'correlation_results.json')
    return results


def _plot_correlation_matrix(corr_matrix: pd.DataFrame, title: str) -> plt.Figure:
    """Create correlation matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
    
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr_matrix.columns)
    
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            value = corr_matrix.iloc[i, j]
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                   color='white' if abs(value) > 0.5 else 'black')
    
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    return fig