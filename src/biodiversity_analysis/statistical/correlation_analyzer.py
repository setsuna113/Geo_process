"""
Correlation analysis for biodiversity data - Linus style.

"Get your statistics right or get out."
No bullshit correlations that assume normality when your data is skewed to hell.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import pdist, squareform

from src.biodiversity_analysis.statistical.data_utils import save_results, save_figure
from src.biodiversity_analysis.shared.data.raster_loader import RasterLoader

logger = logging.getLogger(__name__)


def analyze_correlation(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    method: str = 'spearman',
    columns: Optional[List[str]] = None,
    sample_size: Optional[int] = None,
    spatial_step: int = 10,
    plot: bool = True,
    **kwargs  # Ignore extra config from adapter
) -> Dict[str, Any]:
    """
    PROPER correlation analysis - DataFrame ONLY.
    
    No more type guessing bullshit. I want a DataFrame, period.
    
    Args:
        df: DataFrame with features to correlate
        output_dir: Where to save results
        method: 'spearman', 'dcor', 'mi', 'rda', or 'all'
        columns: Specific columns to analyze
        sample_size: Random sample size (overrides spatial_step)
        spatial_step: Spatial grid step for systematic sampling
        plot: Generate visualizations
        
    Returns:
        Dictionary with correlation results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # One path, no bullshit
    return _analyze_dataframe(
        df, output_dir, method, columns,
        sample_size, spatial_step, plot
    )


# DELETED: _analyze_rasters_chunked - Data loading happens in orchestrator, not here
    
    # Load rasters
    rasters = {}
    shapes = []
    am_fungi = None
    ecm_fungi = None
    
    # First pass: load all data and identify fungi
    temp_rasters = {}
    for name, path_or_array in raster_paths.items():
        if path_or_array is None:
            continue
        if isinstance(path_or_array, str):
            data = raster_loader.load_single(path_or_array)
        elif isinstance(path_or_array, np.ndarray):
            data = path_or_array
        else:
            continue
            
        # Clean NoData values (typically -3.4e38 or similar)
        data = np.where(data < -1e10, np.nan, data)
        data = np.where(data > 1e10, np.nan, data)
            
        # Store AM and ECM separately for combining
        if 'am_fungi' in name.lower():
            am_fungi = data
        elif 'ecm_fungi' in name.lower():
            ecm_fungi = data
        else:
            temp_rasters[name] = data
    
    # Find the base resolution (smallest dimensions)
    base_shapes = [r.shape for r in temp_rasters.values()]
    if base_shapes:
        base_height = min(s[0] for s in base_shapes)
        base_width = min(s[1] for s in base_shapes)
        logger.info(f"Base resolution: {base_height} x {base_width}")
    
    # Process fungi with resolution matching
    if am_fungi is not None or ecm_fungi is not None:
        # Check if fungi has higher resolution (usually 2x)
        fungi_height = am_fungi.shape[0] if am_fungi is not None else ecm_fungi.shape[0]
        fungi_width = am_fungi.shape[1] if am_fungi is not None else ecm_fungi.shape[1]
        
        # Calculate downsampling factor
        height_ratio = fungi_height / base_height if base_shapes else 1
        width_ratio = fungi_width / base_width if base_shapes else 1
        
        if height_ratio > 1.5 or width_ratio > 1.5:
            # Fungi is higher resolution, need to downsample
            downsample_h = int(round(height_ratio))
            downsample_w = int(round(width_ratio))
            logger.info(f"Downsampling fungi from {fungi_height}x{fungi_width} by {downsample_h}x{downsample_w}")
            
            if am_fungi is not None:
                am_fungi = am_fungi[::downsample_h, ::downsample_w][:base_height, :base_width]
            if ecm_fungi is not None:
                ecm_fungi = ecm_fungi[::downsample_h, ::downsample_w][:base_height, :base_width]
    
    # Add non-fungi rasters
    for name, data in temp_rasters.items():
        rasters[name] = data[:base_height, :base_width] if base_shapes else data
        shapes.append(rasters[name].shape)
    
    # Combine AM and ECM into total fungi if both present
    if am_fungi is not None and ecm_fungi is not None:
        # Now they should have matching dimensions
        min_h = min(am_fungi.shape[0], ecm_fungi.shape[0])
        min_w = min(am_fungi.shape[1], ecm_fungi.shape[1])
        
        # Use nansum to handle NaN properly
        total_fungi = np.nansum([am_fungi[:min_h, :min_w], ecm_fungi[:min_h, :min_w]], axis=0)
        rasters['fungi'] = total_fungi
        shapes.append(total_fungi.shape)
        logger.info("Combined AM and ECM fungi into total fungi")
    elif am_fungi is not None:
        rasters['fungi'] = am_fungi
        shapes.append(am_fungi.shape)
    elif ecm_fungi is not None:
        rasters['fungi'] = ecm_fungi
        shapes.append(ecm_fungi.shape)
    
    if not rasters:
        return {'error': 'No valid rasters loaded'}
    
    # Check shapes match
    if len(set(shapes)) > 1:
        logger.warning(f"Rasters have different shapes: {shapes}")
    
    height, width = shapes[0]
    total_pixels = height * width
    
    # Decide on sampling strategy
    if sample_size and sample_size < total_pixels:
        # Use explicit sample size
        effective_step = int(np.sqrt(total_pixels / sample_size))
        logger.info(f"Using sample_size={sample_size}, step={effective_step}")
    elif spatial_step > 1:
        # Use spatial step for systematic sampling
        effective_step = spatial_step
        actual_samples = (height // effective_step) * (width // effective_step)
        logger.info(f"Spatial sampling: step={effective_step}, ~{actual_samples:,} samples")
    else:
        # Process all data
        effective_step = 1
        logger.info(f"Processing all {total_pixels:,} pixels")
    
    # Extract sampled or full data
    if effective_step > 1:
        # Systematic spatial sampling
        sampled_data = _spatial_sample_rasters(rasters, effective_step)
    else:
        # Full data - flatten and stack
        sampled_data = {}
        for name, raster in rasters.items():
            sampled_data[name] = raster.flatten()
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(sampled_data)
    
    # Clean data: remove NaN and infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()  # Remove NaN rows
    
    # Check for extreme values and clip if needed
    for col in df.columns:
        q99 = df[col].quantile(0.99)
        q01 = df[col].quantile(0.01)
        if q99 > 1e10 or q01 < -1e10:
            logger.warning(f"Clipping extreme values in {col}")
            df[col] = df[col].clip(lower=q01*1.1, upper=q99*1.1)
    
    logger.info(f"Analyzing {len(df):,} valid samples")
    
    # Run requested analyses
    results = {}
    
    if method in ['spearman', 'all']:
        results['spearman'] = _compute_spearman(df)
    
    if method in ['dcor', 'all']:
        results['distance_correlation'] = _compute_dcor(df)
    
    if method in ['mi', 'all']:
        results['mutual_information'] = _compute_mutual_info(df)
    
    # Add partial correlation if climate data is available
    if method in ['partial', 'all']:
        # Check if climate variables exist
        climate_vars = ['bio01', 'bio04', 'bio12', 'bio15', 'dem']
        available_climate = [c for c in climate_vars if c in df.columns]
        
        if available_climate and len(df.columns) > len(available_climate) + 1:
            results['partial_correlation'] = _compute_partial_correlations(df, available_climate)
        else:
            logger.info("Partial correlation skipped - need climate variables to control for")
    
    # Skip RDA for 3-variable analysis - it's not appropriate
    if method == 'rda':
        logger.info("RDA skipped - not appropriate for 3-variable correlation analysis")
    
    # Add metadata
    results['statistics'] = {
        'n_samples': len(df),
        'n_features': len(df.columns),
        'features': list(df.columns),
        'sampling_step': effective_step if effective_step > 1 else None,
        'total_pixels': total_pixels
    }
    
    # Generate plots
    if plot and len(df.columns) > 1:
        _generate_plots(df, results, output_dir)
    
    # Save results
    save_results(results, output_dir / 'correlation_results.json')
    
    return results


def _spatial_sample_rasters(rasters: Dict[str, np.ndarray], step: int) -> Dict[str, np.ndarray]:
    """
    SPATIAL sampling that preserves structure.
    Not random garbage that destroys spatial patterns.
    """
    # Find minimum dimensions across all rasters (for alignment)
    min_height = min(r.shape[0] for r in rasters.values())
    min_width = min(r.shape[1] for r in rasters.values())
    
    # Create systematic grid based on minimum dimensions
    row_indices = np.arange(0, min_height, step)
    col_indices = np.arange(0, min_width, step)
    
    sampled = {}
    for name, raster in rasters.items():
        samples = []
        for r in row_indices:
            for c in col_indices:
                # Only sample if within this raster's bounds
                if r < raster.shape[0] and c < raster.shape[1]:
                    samples.append(raster[r, c])
                else:
                    samples.append(np.nan)
        sampled[name] = np.array(samples)
    
    return sampled


def _analyze_dataframe(
    df: pd.DataFrame,
    output_dir: Path,
    method: str,
    columns: Optional[List[str]],
    sample_size: Optional[int],
    spatial_step: int,
    plot: bool
) -> Dict[str, Any]:
    """Analyze pre-loaded DataFrame."""
    
    if columns:
        df = df[columns]
    
    # Sample if needed
    if sample_size and len(df) > sample_size:
        # For DataFrames, we don't have spatial structure, so random is OK
        df = df.sample(n=sample_size, random_state=42)
        logger.info(f"Random sampled {sample_size} points from DataFrame")
    
    numeric_data = df.select_dtypes(include=[np.number])
    
    results = {}
    
    if method in ['spearman', 'all']:
        results['spearman'] = _compute_spearman(numeric_data)
    
    if method in ['dcor', 'all']:
        results['distance_correlation'] = _compute_dcor(numeric_data)
    
    if method in ['mi', 'all']:
        results['mutual_information'] = _compute_mutual_info(numeric_data)
    
    if method in ['partial', 'all']:
        # Check if climate variables exist
        climate_vars = ['bio01', 'bio04', 'bio12', 'bio15', 'dem', 'elevation']
        available_climate = [c for c in climate_vars if c in numeric_data.columns]
        
        if available_climate and len(numeric_data.columns) > len(available_climate) + 1:
            results['partial_correlation'] = _compute_partial_correlations(numeric_data, available_climate)
        else:
            logger.info("Partial correlation skipped - need climate variables to control for")
    
    if method in ['rda', 'all'] and len(numeric_data.columns) > 2:
        results['rda'] = _compute_rda(numeric_data)
    
    results['statistics'] = {
        'n_samples': len(df),
        'n_features': len(numeric_data.columns),
        'features': list(numeric_data.columns)
    }
    
    if plot and len(numeric_data.columns) > 1:
        _generate_plots(numeric_data, results, output_dir)
    
    save_results(results, output_dir / 'correlation_results.json')
    
    return results


def _compute_spearman(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Spearman rank correlation - the MINIMUM acceptable method.
    At least it doesn't assume normality.
    """
    corr_matrix = df.corr(method='spearman')
    
    # Compute p-values
    n = len(df)
    cols = df.columns
    p_values = np.zeros((len(cols), len(cols)))
    
    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):
            if i != j:
                r, p = stats.spearmanr(df[col1], df[col2])
                p_values[i, j] = p
    
    return {
        'correlation_matrix': corr_matrix.to_dict(),
        'p_values': pd.DataFrame(p_values, index=cols, columns=cols).to_dict(),
        'method': 'spearman_rank'
    }


def _compute_dcor(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Distance correlation - NOW we're talking.
    Detects ANY dependency, not just monotonic.
    """
    try:
        import dcor
    except ImportError:
        logger.warning("dcor not installed. Install with: pip install dcor")
        return {'error': 'dcor package not installed'}
    
    # dcor can overflow with large datasets, subsample if needed
    max_dcor_samples = 5000
    if len(df) > max_dcor_samples:
        logger.info(f"Subsampling to {max_dcor_samples} for distance correlation (overflow prevention)")
        df_dcor = df.sample(n=max_dcor_samples, random_state=42)
    else:
        df_dcor = df
    
    cols = df_dcor.columns
    n_cols = len(cols)
    dcor_matrix = np.zeros((n_cols, n_cols))
    
    # Compute pairwise distance correlations
    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):
            if i == j:
                dcor_matrix[i, j] = 1.0
            else:
                try:
                    # Use dcor's efficient implementation
                    dcor_val = dcor.distance_correlation(
                        df_dcor[col1].values, 
                        df_dcor[col2].values
                    )
                    dcor_matrix[i, j] = dcor_val
                except (OverflowError, ValueError) as e:
                    logger.warning(f"dcor failed for {col1}-{col2}: {e}")
                    dcor_matrix[i, j] = np.nan
    
    return {
        'correlation_matrix': pd.DataFrame(
            dcor_matrix, index=cols, columns=cols
        ).to_dict(),
        'method': 'distance_correlation',
        'interpretation': 'Values range [0,1]. 0 means independent, 1 means perfectly dependent'
    }


def _compute_mutual_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Mutual information - information theory FTW.
    How much does knowing one variable reduce uncertainty about another?
    """
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.preprocessing import KBinsDiscretizer
    
    # Clean data first - MI can't handle NaN
    df_clean = df.dropna()
    if len(df_clean) < 100:
        logger.warning(f"Only {len(df_clean)} samples after removing NaN for MI")
        return {'error': 'Insufficient data after removing NaN'}
    
    cols = df_clean.columns
    n_cols = len(cols)
    mi_matrix = np.zeros((n_cols, n_cols))
    
    # Discretize for MI computation (required for continuous data)
    discretizer = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='quantile')
    df_discrete = pd.DataFrame(
        discretizer.fit_transform(df_clean),
        columns=cols
    )
    
    # Compute pairwise MI
    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):
            if i == j:
                # MI of variable with itself is its entropy
                # For normalized MI, set diagonal to 1
                mi_matrix[i, j] = 1.0
            else:
                mi = mutual_info_regression(
                    df_discrete[[col1]], 
                    df_discrete[col2],
                    discrete_features=True,
                    random_state=42
                )[0]
                # Normalize by min entropy for comparability
                h1 = mutual_info_regression(
                    df_discrete[[col1]], df_discrete[col1],
                    discrete_features=True, random_state=42
                )[0]
                h2 = mutual_info_regression(
                    df_discrete[[col2]], df_discrete[col2],
                    discrete_features=True, random_state=42
                )[0]
                mi_matrix[i, j] = mi / min(h1, h2) if min(h1, h2) > 0 else 0
    
    return {
        'mutual_info_matrix': pd.DataFrame(
            mi_matrix, index=cols, columns=cols
        ).to_dict(),
        'method': 'normalized_mutual_information',
        'interpretation': 'Values range [0,1]. Higher means more information shared'
    }


def _compute_partial_correlations(df: pd.DataFrame, control_vars: List[str]) -> Dict[str, Any]:
    """
    Compute partial correlations controlling for confounding variables.
    
    THIS reveals the TRUE relationships after removing confounders.
    """
    from src.biodiversity_analysis.statistical.partial_correlation import compute_partial_correlation
    
    # Get biodiversity variables (non-climate)
    biodiv_vars = [col for col in df.columns if col not in control_vars]
    
    # Compute partial correlation matrix
    n_vars = len(biodiv_vars)
    partial_matrix = np.zeros((n_vars, n_vars))
    p_matrix = np.zeros((n_vars, n_vars))
    
    for i, var1 in enumerate(biodiv_vars):
        for j, var2 in enumerate(biodiv_vars):
            if i == j:
                partial_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
            elif i < j:  # Upper triangle only
                result = compute_partial_correlation(
                    df, var1, var2, control_vars, method='spearman'
                )
                partial_matrix[i, j] = result['r']
                partial_matrix[j, i] = result['r']  # Symmetric
                p_matrix[i, j] = result['p']
                p_matrix[j, i] = result['p']
    
    # Also compute simple correlations for comparison
    simple_matrix = df[biodiv_vars].corr(method='spearman')
    
    # Calculate effect of controlling
    delta_matrix = partial_matrix - simple_matrix.values
    
    return {
        'partial_correlation_matrix': pd.DataFrame(
            partial_matrix, index=biodiv_vars, columns=biodiv_vars
        ).to_dict(),
        'simple_correlation_matrix': simple_matrix.to_dict(),
        'delta_matrix': pd.DataFrame(
            delta_matrix, index=biodiv_vars, columns=biodiv_vars
        ).to_dict(),
        'p_values': pd.DataFrame(
            p_matrix, index=biodiv_vars, columns=biodiv_vars
        ).to_dict(),
        'control_variables': control_vars,
        'method': 'partial_spearman',
        'interpretation': 'Partial correlation removes the linear effects of control variables'
    }


def _compute_rda(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Redundancy Analysis - multivariate relationships.
    Only on sampled data because this is expensive.
    """
    from sklearn.cross_decomposition import CCA
    from sklearn.preprocessing import StandardScaler
    
    # Clean data first - RDA can't handle NaN
    df_clean = df.dropna()
    if len(df_clean) < 100:
        logger.warning(f"Only {len(df_clean)} samples after removing NaN for RDA")
        return {'error': 'Insufficient data after removing NaN'}
    
    if len(df_clean) > 10000:
        logger.info("RDA on large dataset - using subsample")
        df_clean = df_clean.sample(n=10000, random_state=42)
    
    # Standardize data
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_clean),
        columns=df_clean.columns
    )
    
    # Split into groups for CCA (example: first half vs second half of variables)
    mid = len(df_scaled.columns) // 2
    if mid < 1:
        return {'error': 'Need at least 2 variables for RDA'}
    
    X = df_scaled.iloc[:, :mid]
    Y = df_scaled.iloc[:, mid:]
    
    # Fit CCA
    n_components = min(2, min(X.shape[1], Y.shape[1]))
    cca = CCA(n_components=n_components)
    cca.fit(X, Y)
    
    # Get canonical correlations
    X_c, Y_c = cca.transform(X, Y)
    canonical_corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] 
                       for i in range(n_components)]
    
    return {
        'canonical_correlations': canonical_corrs,
        'n_components': n_components,
        'x_vars': list(X.columns),
        'y_vars': list(Y.columns),
        'method': 'redundancy_analysis',
        'interpretation': 'Canonical correlations show strength of multivariate relationships'
    }


def _generate_plots(df: pd.DataFrame, results: Dict[str, Any], output_dir: Path):
    """Generate PROPER visualization for correlation methods."""
    
    # Count valid methods (exclude errors and statistics)
    valid_methods = [k for k in results.keys() 
                    if k != 'statistics' and 'error' not in results.get(k, {})]
    n_methods = len(valid_methods)
    
    if n_methods == 0:
        logger.warning("No valid correlation results to plot")
        return
    
    # Create figure with better layout
    if n_methods == 1:
        fig, axes = plt.subplots(1, 1, figsize=(6, 5))
    elif n_methods == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    elif n_methods == 3:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    elif n_methods == 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    else:
        cols = min(3, n_methods)
        rows = (n_methods + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    
    if n_methods == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_methods > 3 else axes
    
    plot_idx = 0
    
    # Spearman correlation heatmap
    if 'spearman' in valid_methods:
        ax = axes[plot_idx]
        corr_df = pd.DataFrame(results['spearman']['correlation_matrix'])
        im = ax.imshow(corr_df.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_title('Spearman Rank Correlation\n(Monotonic relationships)', fontsize=11, fontweight='bold')
        _add_labels_and_values(ax, corr_df, im, fmt='.2f')
        plot_idx += 1
    
    # Distance correlation heatmap
    if 'distance_correlation' in valid_methods:
        ax = axes[plot_idx]
        dcor_df = pd.DataFrame(results['distance_correlation']['correlation_matrix'])
        im = ax.imshow(dcor_df.values, cmap='viridis', vmin=0, vmax=1, aspect='auto')
        ax.set_title('Distance Correlation\n(ANY dependency)', fontsize=11, fontweight='bold')
        _add_labels_and_values(ax, dcor_df, im, fmt='.2f')
        plot_idx += 1
    
    # Mutual information heatmap
    if 'mutual_information' in valid_methods:
        ax = axes[plot_idx]
        mi_df = pd.DataFrame(results['mutual_information']['mutual_info_matrix'])
        im = ax.imshow(mi_df.values, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
        ax.set_title('Mutual Information\n(Shared information)', fontsize=11, fontweight='bold')
        _add_labels_and_values(ax, mi_df, im, fmt='.2f')
        plot_idx += 1
    
    # Partial correlation heatmap
    if 'partial_correlation' in valid_methods:
        ax = axes[plot_idx]
        partial_df = pd.DataFrame(results['partial_correlation']['partial_correlation_matrix'])
        im = ax.imshow(partial_df.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        control_vars = ', '.join(results['partial_correlation']['control_variables'][:3])
        if len(results['partial_correlation']['control_variables']) > 3:
            control_vars += '...'
        ax.set_title(f'Partial Correlation\n(Controlling for {control_vars})', 
                    fontsize=11, fontweight='bold')
        _add_labels_and_values(ax, partial_df, im, fmt='.2f')
        plot_idx += 1
    
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Biodiversity Correlation Analysis - Multiple Methods', 
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, output_dir / 'correlation_matrices.png')
    plt.close(fig)
    
    # Generate comparison plot
    _generate_comparison_plot(results, output_dir)


def _add_labels_and_values(ax, df, im, fmt='.2f'):
    """Add labels and values to heatmap."""
    ax.set_xticks(range(len(df.columns)))
    ax.set_yticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha='right')
    ax.set_yticklabels(df.columns)
    
    # Add text annotations
    for i in range(len(df.columns)):
        for j in range(len(df.columns)):
            value = df.iloc[i, j]
            text_color = 'white' if abs(value - 0.5) > 0.3 else 'black'
            ax.text(j, i, format(value, fmt), ha='center', va='center',
                   color=text_color, fontsize=9)
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _generate_comparison_plot(results: Dict[str, Any], output_dir: Path):
    """Generate comparison plot showing all methods side by side."""
    
    valid_methods = [k for k in results.keys() 
                    if k != 'statistics' and 'error' not in results.get(k, {})]
    
    if len(valid_methods) < 2:
        return  # Need at least 2 methods to compare
    
    # Extract key correlations for comparison
    features = list(results.get('statistics', {}).get('features', []))
    if len(features) < 2:
        return
    
    # Create comparison data
    comparison_data = []
    for i, feat1 in enumerate(features):
        for j, feat2 in enumerate(features):
            if i >= j:  # Skip diagonal and lower triangle
                continue
            row = {'Pair': f'{feat1}-{feat2}'}
            
            # Add values from each method
            if 'spearman' in valid_methods:
                row['Spearman'] = results['spearman']['correlation_matrix'][feat1][feat2]
            if 'distance_correlation' in valid_methods:
                row['Distance'] = results['distance_correlation']['correlation_matrix'][feat1][feat2]
            if 'mutual_information' in valid_methods:
                row['MI'] = results['mutual_information']['mutual_info_matrix'][feat1][feat2]
            if 'partial_correlation' in valid_methods:
                # Partial correlation matrix may only have biodiv variables
                partial_matrix = results['partial_correlation'].get('partial_correlation_matrix', {})
                if feat1 in partial_matrix and feat2 in partial_matrix[feat1]:
                    row['Partial'] = partial_matrix[feat1][feat2]
            
            comparison_data.append(row)
    
    if not comparison_data:
        return
    
    # Create bar plot comparison
    comp_df = pd.DataFrame(comparison_data)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(comp_df))
    width = 0.25
    offset = 0
    
    colors = {'Spearman': 'steelblue', 'Distance': 'forestgreen', 'MI': 'coral', 'Partial': 'purple'}
    
    for method in ['Spearman', 'Distance', 'MI', 'Partial']:
        if method in comp_df.columns:
            ax.bar(x + offset, comp_df[method], width, label=method, 
                  color=colors.get(method, 'gray'), alpha=0.8)
            offset += width
    
    ax.set_xlabel('Variable Pairs', fontsize=11)
    ax.set_ylabel('Correlation/Dependency Strength', fontsize=11)
    ax.set_title('Comparison of Correlation Methods', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(comp_df['Pair'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, output_dir / 'correlation_comparison.png')
    plt.close(fig)


# Backward compatibility wrapper
def chunked_correlation(rasters: Dict[str, np.ndarray], 
                        chunk_size: int = 100000) -> Dict[str, Any]:
    """
    Legacy wrapper - use analyze_correlation instead.
    """
    logger.warning("chunked_correlation is deprecated. Use analyze_correlation.")
    return analyze_correlation(
        data_source=rasters,
        output_dir=Path.cwd() / 'output',
        method='spearman',
        chunk_size=chunk_size
    )