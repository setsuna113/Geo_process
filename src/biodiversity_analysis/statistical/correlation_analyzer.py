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
    Analyze rasters using CORRECT chunked processing with Welford's algorithm.
    
    The old version was mathematically wrong. This is the right way.
    """
    from src.biodiversity_analysis.statistical.welford import WelfordAccumulator
    
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
    
    # If sampling requested and smaller than total, use spatial sampling (not random!)
    if sample_size and sample_size < total_pixels:
        return _analyze_with_spatial_sampling(rasters, output_dir, method, sample_size, plot)
    
    # Use Welford's algorithm for numerically stable incremental computation
    logger.info(f"Processing {total_pixels:,} pixels with Welford's algorithm")
    
    names = list(rasters.keys())
    n_features = len(names)
    accumulator = WelfordAccumulator(n_features)
    
    # Process in chunks
    height, width = shapes[0]
    pixels_processed = 0
    
    for start_idx in range(0, total_pixels, chunk_size):
        end_idx = min(start_idx + chunk_size, total_pixels)
        chunk_pixels = end_idx - start_idx
        
        # Extract chunk for each feature
        chunk_data = np.full((chunk_pixels, n_features), np.nan)
        for i, name in enumerate(names):
            flat = rasters[name].flatten()
            chunk_data[:, i] = flat[start_idx:end_idx]
        
        # Find rows with all valid data
        valid_mask = ~np.any(np.isnan(chunk_data), axis=1)
        valid_data = chunk_data[valid_mask]
        
        if len(valid_data) > 0:
            accumulator.update(valid_data)
            pixels_processed += len(valid_data)
    
    # Get final statistics
    stats = accumulator.get_statistics()
    
    # Format results
    correlations = {}
    corr_matrix = stats['correlation']
    for i, name_i in enumerate(names):
        correlations[name_i] = {}
        for j, name_j in enumerate(names):
            correlations[name_i][name_j] = float(corr_matrix[i, j])
    
    results = {
        'pearson': correlations,
        'statistics': {
            'n_samples': stats['n'],
            'n_features': n_features,
            'features': names,
            'means': {name: float(stats['mean'][i]) for i, name in enumerate(names)},
            'stds': {name: float(stats['std'][i]) for i, name in enumerate(names)},
            'method': 'welford_incremental',
            'chunk_size': chunk_size
        }
    }
    
    # Generate plot if requested
    if plot and len(rasters) > 1:
        corr_df = pd.DataFrame(correlations)
        fig = _plot_correlation_matrix(corr_df, f"Correlation Matrix (Welford)")
        save_figure(fig, output_dir / 'correlation_matrix.png')
        plt.close(fig)
    
    save_results(results, output_dir / 'correlation_results.json')
    return results


def _analyze_with_spatial_sampling(rasters: Dict, output_dir: Path, method: str, 
                                  sample_size: int, plot: bool) -> Dict[str, Any]:
    """
    Use SPATIAL sampling, not random garbage.
    
    Systematic grid sampling preserves spatial structure.
    """
    height, width = next(iter(rasters.values())).shape
    total_pixels = height * width
    
    # Calculate grid step for systematic sampling
    step = int(np.sqrt(total_pixels / sample_size))
    if step < 1:
        step = 1
    
    # Create systematic grid indices
    row_indices = np.arange(0, height, step)
    col_indices = np.arange(0, width, step)
    
    # Extract samples at grid points
    sampled_data = {}
    for name, raster in rasters.items():
        samples = []
        for r in row_indices:
            for c in col_indices:
                samples.append(raster[r, c])
        sampled_data[name] = np.array(samples)
    
    # Remove rows with any NaN
    df = pd.DataFrame(sampled_data)
    df = df.dropna()
    
    logger.info(f"Spatial sampling: {len(df)} valid samples from {step}×{step} grid")
    
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