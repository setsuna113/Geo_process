"""Bayesian DML visualization for posterior distributions and credible intervals.

Publication-standard visualizations for Bayesian DML analysis following
modern Bayesian visualization standards (bayesplot, tidybayes conventions).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
import seaborn as sns
from scipy import stats
from typing import Dict, Any, Optional, List, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Set publication-quality style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def visualize_bayesian_dml(
    bayesian_results: Dict[str, Any],
    figsize: Tuple[int, int] = (16, 12),
    credible_mass: float = 0.89,  # Use 89% as recommended for stability
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create comprehensive Bayesian DML visualization.
    
    Creates a 6-panel figure following modern Bayesian visualization standards:
    1. Posterior distribution with HDI
    2. Prior vs Posterior comparison
    3. Trace plots (if MCMC samples available)
    4. Forest plot with credible intervals
    5. Posterior predictive check
    6. Bayesian model comparison (if multiple models)
    
    Args:
        bayesian_results: Results from BayesianDML analysis
        figsize: Figure size
        credible_mass: Credible interval mass (default 89% for stability)
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: Posterior distribution with HDI
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_posterior_distribution(ax1, bayesian_results, credible_mass)
    
    # Panel 2: Prior vs Posterior
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_prior_posterior(ax2, bayesian_results)
    
    # Panel 3: Trace plots or Ridge plot
    ax3 = fig.add_subplot(gs[1, 0])
    if 'mcmc_samples' in bayesian_results:
        _plot_trace(ax3, bayesian_results)
    else:
        _plot_ridge(ax3, bayesian_results)
    
    # Panel 4: Forest plot with credible intervals
    ax4 = fig.add_subplot(gs[1, 1])
    _plot_bayesian_forest(ax4, bayesian_results, credible_mass)
    
    # Panel 5: Posterior predictive check or violin plot
    ax5 = fig.add_subplot(gs[2, 0])
    _plot_posterior_predictive(ax5, bayesian_results)
    
    # Panel 6: Model comparison or summary
    ax6 = fig.add_subplot(gs[2, 1])
    _plot_model_comparison(ax6, bayesian_results)
    
    # Main title
    fig.suptitle('Bayesian DML Analysis: Posterior Inference',
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Bayesian DML visualization saved to {save_path}")
    
    return fig


def _plot_posterior_distribution(ax: plt.Axes, results: Dict[str, Any], credible_mass: float):
    """Plot posterior distribution with HDI (Highest Density Interval)."""
    
    # Extract posterior samples or parameters
    if 'posterior_samples' in results:
        samples = np.array(results['posterior_samples'])
    elif 'posterior_mean' in results and 'posterior_std' in results:
        # Generate samples from posterior parameters
        mean = results['posterior_mean']
        std = results['posterior_std']
        samples = np.random.normal(mean, std, 10000)
    else:
        ax.text(0.5, 0.5, 'No posterior data available', ha='center', va='center')
        ax.set_title('Posterior Distribution')
        return
    
    # Calculate HDI (Highest Density Interval)
    hdi_lower, hdi_upper = _calculate_hdi(samples, credible_mass)
    
    # Plot density
    density = stats.gaussian_kde(samples)
    x_range = np.linspace(samples.min() - 0.5 * samples.std(),
                         samples.max() + 0.5 * samples.std(), 1000)
    y_density = density(x_range)
    
    # Main density plot
    ax.plot(x_range, y_density, 'b-', linewidth=2, label='Posterior')
    ax.fill_between(x_range, 0, y_density, alpha=0.3, color='blue')
    
    # Shade HDI region
    hdi_mask = (x_range >= hdi_lower) & (x_range <= hdi_upper)
    ax.fill_between(x_range[hdi_mask], 0, y_density[hdi_mask],
                    alpha=0.5, color='darkblue',
                    label=f'{credible_mass*100:.0f}% HDI')
    
    # Add vertical lines for key statistics
    posterior_mean = np.mean(samples)
    posterior_median = np.median(samples)
    posterior_mode = x_range[np.argmax(y_density)]
    
    ax.axvline(posterior_mean, color='red', linestyle='-', alpha=0.7,
              linewidth=2, label=f'Mean: {posterior_mean:.3f}')
    ax.axvline(posterior_median, color='green', linestyle='--', alpha=0.7,
              linewidth=2, label=f'Median: {posterior_median:.3f}')
    ax.axvline(posterior_mode, color='orange', linestyle=':', alpha=0.7,
              linewidth=2, label=f'Mode: {posterior_mode:.3f}')
    
    # Add HDI annotation
    ax.annotate(f'HDI: [{hdi_lower:.3f}, {hdi_upper:.3f}]',
               xy=(hdi_lower, 0), xytext=(hdi_lower, max(y_density) * 0.5),
               arrowprops=dict(arrowstyle='->', color='darkblue'),
               fontsize=10, fontweight='bold')
    
    # Styling
    ax.set_xlabel('Parameter Value', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Posterior Distribution with {credible_mass*100:.0f}% HDI',
                fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)


def _plot_prior_posterior(ax: plt.Axes, results: Dict[str, Any]):
    """Plot prior vs posterior comparison."""
    
    # Extract prior and posterior
    if 'prior_samples' in results:
        prior_samples = np.array(results['prior_samples'])
    elif 'prior_mean' in results and 'prior_std' in results:
        prior_samples = np.random.normal(results['prior_mean'], 
                                        results['prior_std'], 10000)
    else:
        prior_samples = None
    
    if 'posterior_samples' in results:
        posterior_samples = np.array(results['posterior_samples'])
    elif 'posterior_mean' in results and 'posterior_std' in results:
        posterior_samples = np.random.normal(results['posterior_mean'],
                                           results['posterior_std'], 10000)
    else:
        posterior_samples = None
    
    if prior_samples is None or posterior_samples is None:
        ax.text(0.5, 0.5, 'Prior/Posterior comparison not available',
               ha='center', va='center')
        ax.set_title('Prior vs Posterior')
        return
    
    # Calculate densities
    x_min = min(prior_samples.min(), posterior_samples.min()) - 0.5
    x_max = max(prior_samples.max(), posterior_samples.max()) + 0.5
    x_range = np.linspace(x_min, x_max, 1000)
    
    prior_density = stats.gaussian_kde(prior_samples)(x_range)
    posterior_density = stats.gaussian_kde(posterior_samples)(x_range)
    
    # Plot densities
    ax.plot(x_range, prior_density, 'gray', linewidth=2, 
           linestyle='--', label='Prior', alpha=0.7)
    ax.fill_between(x_range, 0, prior_density, alpha=0.2, color='gray')
    
    ax.plot(x_range, posterior_density, 'blue', linewidth=2.5,
           label='Posterior')
    ax.fill_between(x_range, 0, posterior_density, alpha=0.3, color='blue')
    
    # Add Bayes Factor if available
    if 'bayes_factor' in results:
        bf = results['bayes_factor']
        bf_text = f'Bayes Factor: {bf:.2f}'
        if bf > 10:
            bf_interpretation = 'Strong evidence'
        elif bf > 3:
            bf_interpretation = 'Moderate evidence'
        else:
            bf_interpretation = 'Weak evidence'
        
        ax.text(0.98, 0.98, f'{bf_text}\n({bf_interpretation})',
               transform=ax.transAxes, ha='right', va='top',
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Add vertical line at zero
    ax.axvline(0, color='black', linestyle='-', alpha=0.3)
    
    # Calculate Bayesian learning (KL divergence)
    kl_div = stats.entropy(posterior_density + 1e-10, prior_density + 1e-10)
    ax.text(0.02, 0.98, f'Information gain: {kl_div:.3f} nats',
           transform=ax.transAxes, fontsize=9, va='top',
           style='italic', alpha=0.7)
    
    # Styling
    ax.set_xlabel('Parameter Value', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Prior vs Posterior Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)


def _plot_trace(ax: plt.Axes, results: Dict[str, Any]):
    """Plot MCMC trace plots for convergence diagnostics."""
    
    if 'mcmc_samples' not in results:
        _plot_ridge(ax, results)  # Fallback to ridge plot
        return
    
    samples = results['mcmc_samples']
    
    # If multiple chains, plot them in different colors
    if isinstance(samples, list) and len(samples) > 0:
        if isinstance(samples[0], (list, np.ndarray)):
            # Multiple chains
            n_chains = len(samples)
            colors = plt.cm.Set1(np.linspace(0, 1, n_chains))
            
            for i, chain in enumerate(samples):
                ax.plot(chain, alpha=0.7, linewidth=0.5, color=colors[i],
                       label=f'Chain {i+1}')
        else:
            # Single chain
            ax.plot(samples, alpha=0.8, linewidth=0.5, color='blue')
    
    # Add running mean
    if len(samples) > 0:
        chain = samples[0] if isinstance(samples[0], (list, np.ndarray)) else samples
        window = min(100, len(chain) // 10)
        running_mean = pd.Series(chain).rolling(window=window).mean()
        ax.plot(running_mean, 'r-', linewidth=2, label='Running mean')
    
    # Add convergence diagnostics if available
    if 'rhat' in results:
        rhat = results['rhat']
        color = 'green' if rhat < 1.1 else 'red'
        ax.text(0.98, 0.98, f'R̂ = {rhat:.3f}',
               transform=ax.transAxes, ha='right', va='top',
               fontsize=10, fontweight='bold', color=color,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Styling
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Parameter Value', fontsize=11)
    ax.set_title('MCMC Trace Plot', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)


def _plot_ridge(ax: plt.Axes, results: Dict[str, Any]):
    """Plot ridge plot for multiple parameters."""
    
    # Extract multiple parameters if available
    params = {}
    for key in results:
        if 'posterior' in key or 'samples' in key:
            if isinstance(results[key], (list, np.ndarray)):
                param_name = key.replace('posterior_', '').replace('_samples', '')
                params[param_name] = results[key]
    
    if not params:
        ax.text(0.5, 0.5, 'No parameter distributions available',
               ha='center', va='center')
        ax.set_title('Parameter Distributions')
        return
    
    # Create ridge plot
    n_params = len(params)
    y_positions = np.arange(n_params)
    colors = plt.cm.viridis(np.linspace(0.3, 0.8, n_params))
    
    for i, (name, samples) in enumerate(params.items()):
        samples = np.array(samples)
        
        # Calculate density
        density = stats.gaussian_kde(samples)
        x_range = np.linspace(samples.min() - samples.std(),
                            samples.max() + samples.std(), 200)
        y_density = density(x_range)
        
        # Scale and shift density for ridge effect
        y_density_scaled = y_density / y_density.max() * 0.8
        y_density_shifted = y_density_scaled + i
        
        # Plot density
        ax.fill_between(x_range, i, y_density_shifted,
                       alpha=0.6, color=colors[i])
        ax.plot(x_range, y_density_shifted, color=colors[i], linewidth=2)
        
        # Add median line
        median = np.median(samples)
        ax.plot([median, median], [i, i + 0.8], 'k-', linewidth=2, alpha=0.5)
        
        # Add parameter name
        ax.text(-0.1, i + 0.4, name, ha='right', va='center',
               fontsize=10, fontweight='bold')
    
    # Add zero line
    ax.axvline(0, color='black', linestyle='--', alpha=0.3)
    
    # Styling
    ax.set_xlabel('Parameter Value', fontsize=11)
    ax.set_ylabel('Parameters', fontsize=11)
    ax.set_title('Ridge Plot of Parameter Distributions', fontsize=12, fontweight='bold')
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')


def _plot_bayesian_forest(ax: plt.Axes, results: Dict[str, Any], credible_mass: float):
    """Create forest plot with credible intervals."""
    
    # Extract parameters
    params = {}
    for key in results:
        if 'posterior' in key:
            if isinstance(results[key], (list, np.ndarray)):
                param_name = key.replace('posterior_', '').replace('_samples', '')
                params[param_name] = results[key]
    
    if not params:
        ax.text(0.5, 0.5, 'No parameters for forest plot',
               ha='center', va='center')
        ax.set_title('Bayesian Forest Plot')
        return
    
    names = list(params.keys())
    y_pos = np.arange(len(names))
    
    # Calculate statistics for each parameter
    means = []
    medians = []
    ci_lowers = []
    ci_uppers = []
    
    for name in names:
        samples = np.array(params[name])
        means.append(np.mean(samples))
        medians.append(np.median(samples))
        
        # Calculate HDI
        hdi_lower, hdi_upper = _calculate_hdi(samples, credible_mass)
        ci_lowers.append(hdi_lower)
        ci_uppers.append(hdi_upper)
    
    # Plot intervals
    for i, (mean, median, ci_l, ci_u) in enumerate(zip(means, medians, 
                                                        ci_lowers, ci_uppers)):
        # Thick line for 50% interval
        samples = np.array(params[names[i]])
        q25, q75 = np.percentile(samples, [25, 75])
        ax.plot([q25, q75], [i, i], 'b-', linewidth=4, alpha=0.7)
        
        # Thin line for full credible interval
        ax.plot([ci_l, ci_u], [i, i], 'b-', linewidth=1.5)
        
        # Point for median
        ax.plot(median, i, 'wo', markersize=8, markeredgecolor='blue',
               markeredgewidth=2)
        
        # Add text with values
        ax.text(ci_u + 0.01, i, f'{median:.3f} [{ci_l:.3f}, {ci_u:.3f}]',
               va='center', fontsize=9)
    
    # Add zero line
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    
    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Parameter Value', fontsize=11)
    ax.set_title(f'Forest Plot with {credible_mass*100:.0f}% Credible Intervals',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    ax.text(0.98, 0.02, 'Thick: 50% CI\nThin: Full CI\nDot: Median',
           transform=ax.transAxes, ha='right', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def _plot_posterior_predictive(ax: plt.Axes, results: Dict[str, Any]):
    """Plot posterior predictive check."""
    
    if 'posterior_predictive' in results:
        # Posterior predictive check
        observed = results.get('observed_data', [])
        predicted = results['posterior_predictive']
        
        if observed and predicted:
            # Create violin plot comparing observed vs predicted
            ax.violinplot([observed], positions=[0], showmeans=True,
                         showmedians=True, widths=0.7)
            ax.violinplot([predicted], positions=[1], showmeans=True,
                         showmedians=True, widths=0.7)
            
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Observed', 'Predicted'])
            ax.set_ylabel('Values', fontsize=11)
            ax.set_title('Posterior Predictive Check', fontsize=12, fontweight='bold')
            
            # Add p-value if available
            if 'ppp_value' in results:
                ppp = results['ppp_value']
                color = 'green' if 0.1 < ppp < 0.9 else 'red'
                ax.text(0.98, 0.98, f'PPP = {ppp:.3f}',
                       transform=ax.transAxes, ha='right', va='top',
                       fontsize=10, fontweight='bold', color=color)
        else:
            ax.text(0.5, 0.5, 'Posterior predictive data not available',
                   ha='center', va='center', fontsize=12, alpha=0.7)
            ax.set_title('Posterior Predictive Check', fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Posterior predictive check not performed',
               ha='center', va='center', fontsize=12, alpha=0.7)
        ax.set_title('Posterior Predictive Check', fontsize=12, fontweight='bold')
    
    ax.grid(True, alpha=0.3)


def _plot_model_comparison(ax: plt.Axes, results: Dict[str, Any]):
    """Plot model comparison or summary statistics."""
    
    if 'model_comparison' in results:
        # Model comparison metrics
        comparison = results['model_comparison']
        
        models = list(comparison.keys())
        
        # Extract metrics
        waic = [comparison[m].get('waic', 0) for m in models]
        loo = [comparison[m].get('loo', 0) for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, waic, width, label='WAIC', 
                      color='blue', alpha=0.7)
        bars2 = ax.bar(x + width/2, loo, width, label='LOO',
                      color='red', alpha=0.7)
        
        ax.set_xlabel('Models', fontsize=11)
        ax.set_ylabel('Information Criterion', fontsize=11)
        ax.set_title('Model Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        
        # Highlight best model
        best_idx = np.argmin(waic)
        ax.text(best_idx, waic[best_idx], '★', ha='center', va='bottom',
               fontsize=20, color='gold')
    else:
        # Summary statistics
        ax.axis('off')
        
        summary_text = """Bayesian DML Summary
========================

"""
        
        # Add key statistics
        if 'posterior_mean' in results:
            summary_text += f"Posterior Mean: {results['posterior_mean']:.4f}\n"
        if 'posterior_std' in results:
            summary_text += f"Posterior Std: {results['posterior_std']:.4f}\n"
        if 'hdi_lower' in results and 'hdi_upper' in results:
            summary_text += f"89% HDI: [{results['hdi_lower']:.4f}, {results['hdi_upper']:.4f}]\n"
        if 'bayes_factor' in results:
            summary_text += f"Bayes Factor: {results['bayes_factor']:.2f}\n"
        if 'rhat' in results:
            summary_text += f"R-hat: {results['rhat']:.3f}\n"
        if 'ess' in results:
            summary_text += f"ESS: {results['ess']:.0f}\n"
        
        # Add interpretation
        if 'interpretation' in results:
            summary_text += f"\nInterpretation:\n{results['interpretation']}"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=10, va='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        
        ax.set_title('Summary Statistics', fontsize=12, fontweight='bold')
    
    ax.grid(True, alpha=0.3)


def _calculate_hdi(samples: np.ndarray, credible_mass: float = 0.89) -> Tuple[float, float]:
    """Calculate Highest Density Interval (HDI).
    
    Args:
        samples: Posterior samples
        credible_mass: Probability mass for the interval (default 89%)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    sorted_samples = np.sort(samples)
    n = len(sorted_samples)
    
    # Number of samples to include
    n_samples = int(np.ceil(credible_mass * n))
    
    # Find the narrowest interval
    n_intervals = n - n_samples
    interval_widths = sorted_samples[n_samples:] - sorted_samples[:n_intervals]
    
    # Get the narrowest interval
    min_idx = np.argmin(interval_widths)
    hdi_lower = sorted_samples[min_idx]
    hdi_upper = sorted_samples[min_idx + n_samples]
    
    return hdi_lower, hdi_upper


def create_bayesian_comparison(
    model_results: Dict[str, Dict[str, Any]],
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create comparison visualization for multiple Bayesian models.
    
    Args:
        model_results: Dictionary of model names to results
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        matplotlib Figure
    """
    n_models = len(model_results)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Panel 1: Posterior distributions comparison
    ax1 = axes[0, 0]
    colors = plt.cm.Set1(np.linspace(0, 1, n_models))
    
    for i, (name, results) in enumerate(model_results.items()):
        if 'posterior_samples' in results:
            samples = results['posterior_samples']
        elif 'posterior_mean' in results and 'posterior_std' in results:
            samples = np.random.normal(results['posterior_mean'],
                                     results['posterior_std'], 10000)
        else:
            continue
        
        density = stats.gaussian_kde(samples)
        x_range = np.linspace(samples.min() - samples.std(),
                            samples.max() + samples.std(), 1000)
        ax1.plot(x_range, density(x_range), color=colors[i],
                linewidth=2, label=name)
        ax1.fill_between(x_range, 0, density(x_range),
                        alpha=0.2, color=colors[i])
    
    ax1.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Parameter Value', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Posterior Distributions', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Credible intervals comparison
    ax2 = axes[0, 1]
    names = []
    means = []
    ci_lowers = []
    ci_uppers = []
    
    for name, results in model_results.items():
        names.append(name)
        
        if 'posterior_samples' in results:
            samples = results['posterior_samples']
            means.append(np.mean(samples))
            hdi_l, hdi_u = _calculate_hdi(samples, 0.89)
            ci_lowers.append(hdi_l)
            ci_uppers.append(hdi_u)
        else:
            means.append(results.get('posterior_mean', 0))
            ci_lowers.append(results.get('hdi_lower', 0))
            ci_uppers.append(results.get('hdi_upper', 0))
    
    y_pos = np.arange(len(names))
    
    for i, (mean, ci_l, ci_u) in enumerate(zip(means, ci_lowers, ci_uppers)):
        ax2.plot([ci_l, ci_u], [i, i], 'b-', linewidth=2)
        ax2.plot(mean, i, 'ro', markersize=8)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names)
    ax2.set_xlabel('Parameter Value', fontsize=11)
    ax2.set_title('89% Credible Intervals', fontsize=12, fontweight='bold')
    ax2.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Panel 3: Bayes Factors or Model weights
    ax3 = axes[1, 0]
    
    bayes_factors = []
    model_names = []
    
    for name, results in model_results.items():
        model_names.append(name)
        bayes_factors.append(results.get('bayes_factor', 1))
    
    bars = ax3.bar(range(len(model_names)), bayes_factors,
                  color='steelblue', alpha=0.7, edgecolor='black')
    
    # Color by strength of evidence
    for i, (bar, bf) in enumerate(zip(bars, bayes_factors)):
        if bf > 10:
            bar.set_color('green')
        elif bf > 3:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    ax3.set_xticks(range(len(model_names)))
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    ax3.set_ylabel('Bayes Factor', fontsize=11)
    ax3.set_title('Model Evidence (Bayes Factors)', fontsize=12, fontweight='bold')
    ax3.axhline(1, color='black', linestyle='--', alpha=0.5)
    ax3.axhline(3, color='orange', linestyle=':', alpha=0.5)
    ax3.axhline(10, color='green', linestyle=':', alpha=0.5)
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary table
    table_data = []
    for name, results in model_results.items():
        row = [
            name,
            f"{results.get('posterior_mean', 0):.3f}",
            f"{results.get('posterior_std', 0):.3f}",
            f"{results.get('bayes_factor', 1):.2f}"
        ]
        table_data.append(row)
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Model', 'Mean', 'Std', 'BF'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    ax4.set_title('Model Summary', fontsize=12, fontweight='bold')
    
    plt.suptitle('Bayesian Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Bayesian comparison saved to {save_path}")
    
    return fig