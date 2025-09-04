from src.abstractions import get_logger
"""
Distribution transformation analysis - Linus style.

"If you can't measure it, you can't improve it. Test your damn distributions."
Test distributions, apply transforms, measure improvement. No bullshit.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json
import warnings

from src.biodiversity_analysis.statistical.data_utils import save_results, save_figure

logger = get_logger(__name__)


def run_distribution_tests(data: np.ndarray) -> Dict[str, Any]:
    """
    Run comprehensive distribution tests on a single array.
    
    LINUS: "One function, multiple tests. No code duplication."
    
    Args:
        data: 1D array of values (NaNs are removed)
        
    Returns:
        Dict with test statistics and p-values
    """
    # Remove NaNs
    clean_data = data[~np.isnan(data)]
    
    if len(clean_data) < 3:
        return {
            'error': 'Insufficient data',
            'n_valid': len(clean_data)
        }
    
    results = {
        'n_samples': len(clean_data),
        'mean': np.mean(clean_data),
        'std': np.std(clean_data),
        'skewness': stats.skew(clean_data),
        'kurtosis': stats.kurtosis(clean_data),
        'min': np.min(clean_data),
        'max': np.max(clean_data)
    }
    
    # Normality tests
    if len(clean_data) >= 3:
        # Shapiro-Wilk test (best for small samples)
        if len(clean_data) <= 5000:
            sw_stat, sw_pval = stats.shapiro(clean_data)
            results['shapiro_wilk'] = {'statistic': sw_stat, 'p_value': sw_pval}
        
        # D'Agostino-Pearson test (needs n >= 8)
        if len(clean_data) >= 8:
            k2_stat, k2_pval = stats.normaltest(clean_data)
            results['dagostino_pearson'] = {'statistic': k2_stat, 'p_value': k2_pval}
        
        # Anderson-Darling test
        ad_result = stats.anderson(clean_data, dist='norm')
        results['anderson_darling'] = {
            'statistic': ad_result.statistic,
            'critical_values': dict(zip(['15%', '10%', '5%', '2.5%', '1%'], 
                                       ad_result.critical_values)),
            'reject_normal': ad_result.statistic > ad_result.critical_values[2]  # 5% level
        }
        
        # Kolmogorov-Smirnov test for normality
        # Standardize data first
        standardized = (clean_data - np.mean(clean_data)) / np.std(clean_data)
        ks_stat, ks_pval = stats.kstest(standardized, 'norm')
        results['ks_normal'] = {'statistic': ks_stat, 'p_value': ks_pval}
        
        # Jarque-Bera test (based on skewness and kurtosis)
        jb_stat, jb_pval = stats.jarque_bera(clean_data)
        results['jarque_bera'] = {'statistic': jb_stat, 'p_value': jb_pval}
    
    return results


def apply_transform(data: np.ndarray, transform: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply transformation to data.
    
    Args:
        data: Original data
        transform: 'raw', 'sqrt', 'log', 'boxcox'
        
    Returns:
        Transformed data and transform parameters
    """
    clean_data = data[~np.isnan(data)]
    transformed = np.full_like(data, np.nan)
    valid_mask = ~np.isnan(data)
    params = {}
    
    if transform == 'raw':
        transformed[valid_mask] = data[valid_mask]
        params = {'type': 'raw'}
        
    elif transform == 'sqrt':
        # Handle negative values
        min_val = np.min(clean_data)
        if min_val < 0:
            # Shift to make positive
            shift = abs(min_val) + 1
            transformed[valid_mask] = np.sqrt(data[valid_mask] + shift)
            params = {'type': 'sqrt', 'shift': shift}
        else:
            transformed[valid_mask] = np.sqrt(data[valid_mask])
            params = {'type': 'sqrt', 'shift': 0}
            
    elif transform == 'log':
        # Log1p to handle zeros
        min_val = np.min(clean_data)
        if min_val < 0:
            # Shift to make positive
            shift = abs(min_val) + 1
            transformed[valid_mask] = np.log1p(data[valid_mask] + shift)
            params = {'type': 'log1p', 'shift': shift}
        else:
            transformed[valid_mask] = np.log1p(data[valid_mask])
            params = {'type': 'log1p', 'shift': 0}
            
    elif transform == 'boxcox':
        # Box-Cox requires strictly positive data
        min_val = np.min(clean_data)
        if min_val <= 0:
            shift = abs(min_val) + 1
            shifted_data = clean_data + shift
        else:
            shift = 0
            shifted_data = clean_data
            
        # Find optimal lambda
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            transformed_clean, lambda_val = stats.boxcox(shifted_data)
        
        # Apply to full data
        if shift > 0:
            transformed[valid_mask] = stats.boxcox(data[valid_mask] + shift, lmbda=lambda_val)
        else:
            transformed[valid_mask] = stats.boxcox(data[valid_mask], lmbda=lambda_val)
            
        params = {'type': 'boxcox', 'lambda': lambda_val, 'shift': shift}
    
    return transformed, params


def analyze_transform(
    data: np.ndarray, 
    transform: str, 
    feature_name: str, 
    output_dir: Path
) -> Dict[str, Any]:
    """
    Apply transformation, test distribution, and create visualization.
    
    LINUS: "Do it all in one place. Data locality matters."
    
    Args:
        data: Original feature data
        transform: Transformation type
        feature_name: Name of the feature
        output_dir: Where to save plots
        
    Returns:
        Test results and transformation parameters
    """
    # Apply transformation
    transformed_data, transform_params = apply_transform(data, transform)
    
    # Run distribution tests
    test_results = run_distribution_tests(transformed_data)
    
    # Create multi-panel visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    clean_data = transformed_data[~np.isnan(transformed_data)]
    
    if len(clean_data) > 0:
        # Panel 1: Histogram with fitted normal
        ax1 = axes[0]
        n, bins, patches = ax1.hist(clean_data, bins=30, density=True, 
                                    alpha=0.7, color='blue', edgecolor='black')
        
        # Fit and plot normal distribution
        mu, sigma = np.mean(clean_data), np.std(clean_data)
        x = np.linspace(clean_data.min(), clean_data.max(), 100)
        ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, 
                label=f'Normal fit\nμ={mu:.2f}, σ={sigma:.2f}')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.set_title(f'{feature_name} - {transform.upper()} Transform')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Q-Q plot
        ax2 = axes[1]
        stats.probplot(clean_data, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normal)')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Test statistics
        ax3 = axes[2]
        ax3.axis('off')
        
        # Format test results as text
        text_lines = [
            f"{'='*30}",
            f"DISTRIBUTION TESTS",
            f"{'='*30}",
            f"N samples: {test_results.get('n_samples', 'N/A')}",
            f"Skewness: {test_results.get('skewness', 'N/A'):.3f}",
            f"Kurtosis: {test_results.get('kurtosis', 'N/A'):.3f}",
            f"",
            f"NORMALITY TESTS (p-values):",
            f"{'-'*25}",
        ]
        
        # Add test results
        if 'shapiro_wilk' in test_results:
            p_val = test_results['shapiro_wilk']['p_value']
            sig = "✓" if p_val > 0.05 else "✗"
            text_lines.append(f"Shapiro-Wilk: {p_val:.4f} {sig}")
            
        if 'dagostino_pearson' in test_results:
            p_val = test_results['dagostino_pearson']['p_value']
            sig = "✓" if p_val > 0.05 else "✗"
            text_lines.append(f"D'Agostino: {p_val:.4f} {sig}")
            
        if 'jarque_bera' in test_results:
            p_val = test_results['jarque_bera']['p_value']
            sig = "✓" if p_val > 0.05 else "✗"
            text_lines.append(f"Jarque-Bera: {p_val:.4f} {sig}")
            
        if 'ks_normal' in test_results:
            p_val = test_results['ks_normal']['p_value']
            sig = "✓" if p_val > 0.05 else "✗"
            text_lines.append(f"KS Test: {p_val:.4f} {sig}")
            
        if 'anderson_darling' in test_results:
            reject = test_results['anderson_darling']['reject_normal']
            sig = "✗" if reject else "✓"
            stat = test_results['anderson_darling']['statistic']
            text_lines.append(f"Anderson-Darling: {stat:.3f} {sig}")
        
        # Add transformation parameters
        text_lines.extend([
            f"",
            f"TRANSFORM PARAMS:",
            f"{'-'*25}",
        ])
        
        if transform == 'boxcox' and 'lambda' in transform_params:
            text_lines.append(f"Lambda: {transform_params['lambda']:.3f}")
        if 'shift' in transform_params and transform_params['shift'] > 0:
            text_lines.append(f"Shift: {transform_params['shift']:.3f}")
        
        # Display text
        text = '\n'.join(text_lines)
        ax3.text(0.1, 0.95, text, transform=ax3.transAxes, 
                fontsize=10, verticalalignment='top',
                fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save figure
    plot_path = output_dir / f"{feature_name}_{transform}.png"
    save_figure(fig, plot_path)
    plt.close()
    
    # Return results
    return {
        'feature': feature_name,
        'transform': transform,
        'transform_params': transform_params,
        'test_results': test_results,
        'plot_path': str(plot_path)
    }


def analyze_distribution_transformation(
    data: Any,
    output_dir: Path,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze distributions of features before and after transformations.
    
    LINUS: "Show me the distribution. Then fix it. Then show me again."
    
    Args:
        data: PyArrow Table or DataFrame with features
        output_dir: Output directory for results
        **kwargs: Config including:
            - columns: List of columns to analyze (default: all numeric)
            - transforms: List of transforms to apply (default: all)
            - parallel: Use parallel processing (default: False)
            
    Returns:
        Dict with all test results and paths to plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert data to DataFrame
    if isinstance(data, pa.Table):
        df = data.to_pandas()
        logger.info(f"Converted Arrow Table to DataFrame: {len(df):,} rows")
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError(f"Expected Arrow Table or DataFrame, got {type(data).__name__}")
    
    # Get config
    columns = kwargs.get('columns', None)
    transforms = kwargs.get('transforms', ['raw', 'sqrt', 'log', 'boxcox'])
    
    # Select columns to analyze
    if columns:
        selected_cols = columns
    else:
        # Use all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Filter out coordinate columns
        selected_cols = [c for c in numeric_cols 
                        if c not in ['x', 'y', 'lon', 'lat', 'longitude', 'latitude']]
    
    logger.info(f"Analyzing {len(selected_cols)} columns with {len(transforms)} transforms")
    
    # Store all results
    all_results = {}
    summary_stats = {}
    
    # Process each feature
    for col in selected_cols:
        logger.info(f"Processing feature: {col}")
        
        feature_data = df[col].values
        feature_results = {}
        
        # Apply each transformation
        for transform in transforms:
            try:
                result = analyze_transform(
                    feature_data, 
                    transform, 
                    col, 
                    output_dir
                )
                feature_results[transform] = result
                
                # Track which transform gives best normality
                if 'shapiro_wilk' in result['test_results']:
                    p_val = result['test_results']['shapiro_wilk']['p_value']
                    if col not in summary_stats or p_val > summary_stats[col]['best_p']:
                        summary_stats[col] = {
                            'best_transform': transform,
                            'best_p': p_val,
                            'original_skew': feature_results['raw']['test_results']['skewness']
                        }
                        
            except Exception as e:
                logger.error(f"Failed {transform} transform on {col}: {e}")
                feature_results[transform] = {'error': str(e)}
        
        all_results[col] = feature_results
    
    # Save comprehensive results
    results_path = output_dir / 'distribution_analysis.json'
    save_results({
        'features': all_results,
        'summary': summary_stats,
        'config': {
            'columns': selected_cols,
            'transforms': transforms,
            'n_samples': len(df)
        }
    }, results_path)
    
    # Create summary visualization
    _create_summary_plot(summary_stats, output_dir)
    
    return {
        'success': True,
        'method': 'distribution_transformation',
        'features_analyzed': len(selected_cols),
        'transforms_applied': transforms,
        'results_path': str(results_path),
        'summary': summary_stats
    }


def _create_summary_plot(summary_stats: Dict[str, Any], output_dir: Path) -> None:
    """Create a summary plot showing best transforms for each feature."""
    if not summary_stats:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    features = list(summary_stats.keys())
    best_transforms = [summary_stats[f]['best_transform'] for f in features]
    best_p_values = [summary_stats[f]['best_p'] for f in features]
    original_skews = [abs(summary_stats[f]['original_skew']) for f in features]
    
    # Plot 1: Best transform for each feature
    ax1 = axes[0]
    transform_types = list(set(best_transforms))
    colors = plt.cm.Set3(np.linspace(0, 1, len(transform_types)))
    color_map = dict(zip(transform_types, colors))
    
    x_pos = np.arange(len(features))
    bars = ax1.bar(x_pos, best_p_values, 
                   color=[color_map[t] for t in best_transforms])
    ax1.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='p=0.05')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(features, rotation=45, ha='right')
    ax1.set_ylabel('Best p-value (Shapiro-Wilk)')
    ax1.set_title('Best Transformation for Normality (per feature)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add transform labels
    for i, (bar, transform) in enumerate(zip(bars, best_transforms)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                transform, ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Original skewness
    ax2 = axes[1]
    bars2 = ax2.bar(x_pos, original_skews, color='steelblue')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(features, rotation=45, ha='right')
    ax2.set_ylabel('|Skewness| (Original Data)')
    ax2.set_title('Original Data Skewness')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, output_dir / 'summary_transformations.png')
    plt.close()