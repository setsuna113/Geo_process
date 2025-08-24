"""Residual Diagnostics for GLM Models.

Provides residual diagnostic visualizations WITH integrated statistical tests.
Shows both visual patterns AND quantitative test results on the same plots.
Note: Model comparison plots are handled by visualization/model_comparison_plots.py
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from scipy import stats

logger = logging.getLogger(__name__)


class ResidualDiagnostics:
    """Create residual diagnostic plots for GLM models.
    
    This class generates residual-based diagnostic visualizations
    for assessing model fit quality and assumptions.
    """
    
    def __init__(self):
        """Initialize diagnostic plots generator."""
        self.fig = None
        self.axes = None
    def create_residual_diagnostics(self,
                                   residuals: Dict[str, np.ndarray],
                                   fitted_values: Dict[str, np.ndarray],
                                   diagnostics_results: Optional[Dict[str, Any]] = None,
                                   output_dir: Optional[Path] = None,
                                   filename: str = "residual_diagnostics.png") -> Optional[Path]:
        """Create residual diagnostic plots WITH integrated statistical tests.
        
        Args:
            residuals: Dictionary of residuals for each model (preferably deviance residuals)
            fitted_values: Dictionary of fitted values for each model
            diagnostics_results: Optional pre-computed diagnostic results from DHARMaDiagnostics
            output_dir: Directory to save plot
            filename: Name of output file
            
        Returns:
            Path to saved plot or None
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 11))
            
            model_names = list(residuals.keys())
            colors = ['coral', 'steelblue', 'green', 'purple'][:len(model_names)]
            
            for i, (name, color) in enumerate(zip(model_names, colors)):
                if name in residuals and name in fitted_values:
                    resid = residuals[name]
                    fitted = fitted_values[name]
                    
                    # Get pre-computed results if available
                    residual_analysis = None
                    if diagnostics_results and 'residual_analysis' in diagnostics_results:
                        residual_analysis = diagnostics_results['residual_analysis']
                    
                    # 1. Residuals vs Fitted WITH heteroscedasticity test
                    ax = axes[0, 0]
                    ax.scatter(fitted, resid, alpha=0.5, s=1, color=color, label=name.capitalize())
                    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    
                    # Add LOESS smoothing line to detect patterns
                    from scipy.signal import savgol_filter
                    if len(fitted) > 50:
                        sorted_idx = np.argsort(fitted)
                        try:
                            smoothed = savgol_filter(resid[sorted_idx], 
                                                    window_length=min(51, len(fitted)//4*2+1), 
                                                    polyorder=3)
                            ax.plot(fitted[sorted_idx], smoothed, color='red', alpha=0.7, 
                                   linewidth=2, label='Smooth trend')
                        except:
                            pass
                    
                    # Test for heteroscedasticity - compute if not provided
                    corr, p_val = stats.spearmanr(fitted, resid**2)
                    
                    if p_val < 0.05:
                        ax.set_title(f'Residuals vs Fitted (Heteroscedastic: p={p_val:.3f})', 
                                   color='darkred', fontweight='bold')
                    else:
                        ax.set_title(f'Residuals vs Fitted (Homoscedastic: p={p_val:.3f})', 
                                   color='darkgreen')
                    
                    ax.set_xlabel('Fitted Values')
                    ax.set_ylabel('Residuals')
                    ax.legend()
                    
                    # 2. Q-Q plot WITH normality test
                    ax = axes[0, 1]
                    stats.probplot(resid, dist="norm", plot=ax)
                    
                    # Use pre-computed Jarque-Bera if available, else compute
                    if residual_analysis and 'tests' in residual_analysis and 'jarque_bera' in residual_analysis['tests']:
                        jb_stat = residual_analysis['tests']['jarque_bera']['statistic']
                        jb_pvalue = residual_analysis['tests']['jarque_bera']['p_value']
                    else:
                        jb_stat, jb_pvalue = stats.jarque_bera(resid)
                    
                    # Color-code title based on test result
                    if jb_pvalue < 0.05:
                        ax.set_title(f'Q-Q Plot - NOT NORMAL (JB p={jb_pvalue:.3f})', 
                                   color='darkred', fontweight='bold')
                    else:
                        ax.set_title(f'Q-Q Plot - Normal (JB p={jb_pvalue:.3f})', 
                                   color='darkgreen')
                    
                    # Add test statistics - include Shapiro-Wilk and Durbin-Watson if available
                    test_text = f'JB stat: {jb_stat:.2f}'
                    
                    # Add Shapiro-Wilk if available
                    if residual_analysis and 'tests' in residual_analysis and 'shapiro_wilk' in residual_analysis['tests']:
                        sw_p = residual_analysis['tests']['shapiro_wilk']['p_value']
                        test_text += f'\nSW p-val: {sw_p:.3f}'
                    
                    # Note: Durbin-Watson removed - not appropriate for GLMs
                    
                    ax.text(0.05, 0.95, test_text, 
                           transform=ax.transAxes, fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    
                    # 3. Scale-Location WITH variance trend test
                    ax = axes[1, 0]
                    standardized_resid = resid / np.std(resid)
                    sqrt_abs_resid = np.sqrt(np.abs(standardized_resid))
                    ax.scatter(fitted, sqrt_abs_resid, alpha=0.5, s=1, color=color)
                    
                    # Add smooth line to detect variance trends
                    if len(fitted) > 50:
                        sorted_idx = np.argsort(fitted)
                        try:
                            smoothed = savgol_filter(sqrt_abs_resid[sorted_idx], 
                                                    window_length=min(51, len(fitted)//4*2+1), 
                                                    polyorder=3)
                            ax.plot(fitted[sorted_idx], smoothed, color='red', alpha=0.7, 
                                   linewidth=2, label='Trend')
                        except:
                            pass
                    
                    # Test for trend in variance (correlation test)
                    corr_var, p_var = stats.spearmanr(fitted, sqrt_abs_resid)
                    
                    if abs(corr_var) > 0.1 and p_var < 0.05:
                        ax.set_title(f'Scale-Location (Variance trend: ρ={corr_var:.2f}, p={p_var:.3f})', 
                                   color='darkred', fontweight='bold')
                    else:
                        ax.set_title(f'Scale-Location (Constant variance: p={p_var:.3f})', 
                                   color='darkgreen')
                    
                    ax.set_xlabel('Fitted Values')
                    ax.set_ylabel('√|Standardized Residuals|')
                    
                    # 4. Residual histogram WITH statistics
                    ax = axes[1, 1]
                    n, bins, patches = ax.hist(resid, bins=50, alpha=0.5, color=color, 
                                              density=True, label=name.capitalize())
                    
                    # Overlay normal distribution
                    mu, sigma = np.mean(resid), np.std(resid)
                    x = np.linspace(resid.min(), resid.max(), 100)
                    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, 
                           label='Normal fit', alpha=0.7)
                    
                    # Use pre-computed statistics if available, else compute
                    if residual_analysis and 'statistics' in residual_analysis:
                        skewness = residual_analysis['statistics']['skewness']
                        kurtosis = residual_analysis['statistics']['kurtosis']
                    else:
                        skewness = stats.skew(resid)
                        kurtosis = stats.kurtosis(resid)
                    
                    if residual_analysis and 'outliers' in residual_analysis:
                        outliers = residual_analysis['outliers']['n_outliers']
                        outlier_pct = residual_analysis['outliers']['percentage']
                    else:
                        outliers = np.sum(np.abs(resid) > 3)
                        outlier_pct = (outliers / len(resid)) * 100
                    
                    # Color-code based on skewness
                    if abs(skewness) > 1:
                        skew_color = 'darkred'
                    elif abs(skewness) > 0.5:
                        skew_color = 'darkorange'
                    else:
                        skew_color = 'darkgreen'
                    
                    ax.set_title('Residual Distribution', fontweight='bold')
                    ax.set_xlabel('Residuals')
                    ax.set_ylabel('Density')
                    
                    # Add statistics text box - include issues if available
                    stats_text = f'Skewness: {skewness:.2f}\n'
                    stats_text += f'Kurtosis: {kurtosis:.2f}\n'
                    stats_text += f'Outliers: {outliers} ({outlier_pct:.1f}%)'
                    
                    # Add any issues from residual analysis
                    if residual_analysis and 'issues' in residual_analysis and len(residual_analysis['issues']) > 0:
                        stats_text += '\n' + '-'*15
                        for issue in residual_analysis['issues'][:2]:  # Show max 2 issues
                            stats_text += f'\n⚠ {issue[:25]}'  # Truncate long issues
                    
                    ax.text(0.65, 0.95, stats_text, transform=ax.transAxes,
                           fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    
                    ax.legend()
            
            plt.suptitle('Residual Diagnostics', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if output_dir:
                output_dir = Path(output_dir)
                save_path = output_dir / filename
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Residual diagnostics saved to {save_path}")
                return save_path
            else:
                return fig
                
        except Exception as e:
            logger.error(f"Failed to create residual diagnostics: {e}")
            return None
    
    def plot_deviance_df(self, 
                        models: Dict[str, Dict[str, Any]],
                        figsize: tuple = (10, 6),
                        save_path: Optional[Path] = None) -> Union[plt.Figure, None]:
        """Create deviance-df plot for model selection diagnostics.
        
        This plot helps visualize the trade-off between model complexity (df) 
        and goodness of fit (deviance) for model selection. Lower deviance 
        indicates better fit, while higher df indicates simpler models.
        
        Args:
            models: Dictionary of models with 'deviance' and 'df' (or 'residual_df')
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure or None if error
        """
        try:
            fig, ax = plt.subplots(figsize=figsize)
            
            # Extract model information
            model_names = []
            deviances = []
            dfs = []
            
            for name, model_info in models.items():
                # Get deviance
                dev = model_info.get('deviance', model_info.get('residual_deviance', None))
                # Get degrees of freedom
                df = model_info.get('df', model_info.get('residual_df', 
                                   model_info.get('degrees_of_freedom', None)))
                
                if dev is not None and df is not None:
                    model_names.append(name)
                    deviances.append(dev)
                    dfs.append(df)
            
            if not model_names:
                logger.warning("No valid models with deviance and df found")
                return None
            
            # Create color map
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names)))
            
            # Plot points
            for i, (name, dev, df) in enumerate(zip(model_names, deviances, dfs)):
                ax.scatter(df, dev, s=200, c=[colors[i]], alpha=0.7,
                          edgecolors='black', linewidth=2, label=name)
                
                # Add model name annotation
                ax.annotate(name, (df, dev), xytext=(5, 5), 
                           textcoords='offset points', fontsize=9)
            
            # Connect points if we have exactly 2 models (nested comparison)
            if len(model_names) == 2:
                ax.plot(dfs, deviances, 'k--', alpha=0.3, linewidth=1)
                
                # Add arrow showing direction of improvement
                if deviances[1] < deviances[0]:  # Lower deviance is better
                    ax.annotate('', xy=(dfs[1], deviances[1]), 
                               xytext=(dfs[0], deviances[0]),
                               arrowprops=dict(arrowstyle='->', lw=2, 
                                             color='green', alpha=0.5))
                    
                    # Add improvement text
                    mid_x = (dfs[0] + dfs[1]) / 2
                    mid_y = (deviances[0] + deviances[1]) / 2
                    reduction = deviances[0] - deviances[1]
                    pct_reduction = (reduction / deviances[0] * 100) if deviances[0] > 0 else 0
                    
                    ax.text(mid_x, mid_y, 
                           f'Δ Dev = {reduction:.2f}\n({pct_reduction:.1f}%)',
                           ha='center', va='bottom', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            
            # Add ideal region (lower left corner)
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            # Shade the "good" region (low deviance, high df = simple and good fit)
            good_x = [xlim[1] * 0.6, xlim[1], xlim[1], xlim[1] * 0.6]
            good_y = [ylim[0], ylim[0], ylim[1] * 0.4, ylim[1] * 0.4]
            ax.fill(good_x, good_y, color='green', alpha=0.05, zorder=0)
            ax.text(xlim[1] * 0.8, ylim[1] * 0.2, 'Ideal:\nSimple +\nGood Fit',
                   ha='center', va='center', fontsize=9, style='italic', alpha=0.5)
            
            # Styling
            ax.set_xlabel('Degrees of Freedom (Residual)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Deviance', fontsize=12, fontweight='bold')
            ax.set_title('Deviance-df Plot: Model Selection Diagnostic', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='upper left', framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add interpretation guide
            interpretation = (
                "Lower deviance → Better fit\n"
                "Higher df → Simpler model\n"
                "Goal: Balance simplicity & fit"
            )
            ax.text(0.02, 0.98, interpretation, transform=ax.transAxes,
                   fontsize=9, va='top', style='italic',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
            
            # Invert x-axis if needed (so simpler models are on the right)
            if len(dfs) > 1 and dfs[0] < dfs[1]:
                ax.invert_xaxis()
            
            plt.tight_layout()
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Deviance-df plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create deviance-df plot: {e}")
            return None
    
    def plot_model_diagnostics(self,
                              diagnostics_results: Dict[str, Any],
                              output_dir: Optional[Path] = None,
                              filename: str = "model_diagnostics.png") -> Optional[Path]:
        """Create comprehensive model diagnostic plots.
        
        Visualizes:
        - GVIF multicollinearity values
        - Sample size adequacy
        - Overdispersion metrics
        - Coefficient diagnostics
        
        Args:
            diagnostics_results: Results from DHARMaDiagnostics.run_diagnostics()
            output_dir: Directory to save plot
            filename: Name of output file
            
        Returns:
            Path to saved plot or None
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 11))
            
            # 1. GVIF Multicollinearity Plot
            ax = axes[0, 0]
            if 'multicollinearity' in diagnostics_results:
                multi = diagnostics_results['multicollinearity']
                # Check for GVIF in the multicollinearity dict
                if 'gvif' in multi:
                    gvif_dict = multi['gvif']
                    if isinstance(gvif_dict, dict) and len(gvif_dict) > 0:
                        vars = list(gvif_dict.keys())
                        values = list(gvif_dict.values())
                        sqrt_values = [np.sqrt(v) for v in values]
                        
                        colors = ['red' if v > 2 else 'orange' if v > 1.5 else 'green' 
                                 for v in sqrt_values]
                        
                        y_pos = np.arange(len(vars))
                        bars = ax.barh(y_pos, sqrt_values, color=colors, alpha=0.7)
                        
                        # Add threshold lines
                        ax.axvline(x=1.5, color='orange', linestyle='--', alpha=0.5, label='Low concern')
                        ax.axvline(x=2, color='red', linestyle='--', alpha=0.5, label='High concern')
                        
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(vars)
                        ax.set_xlabel('√GVIF (Adjusted GVIF)')
                        ax.set_title('Multicollinearity Check (GVIF)', fontweight='bold')
                        ax.legend()
                        
                        # Add interpretation
                        interp = multi.get('interpretation', '')
                        ax.text(0.98, 0.02, interp, transform=ax.transAxes,
                               ha='right', fontsize=9, style='italic')
                    else:
                        ax.text(0.5, 0.5, 'GVIF not available', ha='center', va='center')
                else:
                    ax.text(0.5, 0.5, 'GVIF not available', ha='center', va='center')
            else:
                ax.text(0.5, 0.5, 'Multicollinearity check not performed', ha='center', va='center')
            ax.set_title('Multicollinearity (GVIF)', fontweight='bold')
            
            # 2. Sample Size Adequacy
            ax = axes[0, 1]
            if 'sample_size' in diagnostics_results:
                ss = diagnostics_results['sample_size']
                n_obs = ss.get('n_obs', 0)
                n_params = ss.get('n_parameters', 0)
                obs_per_param = ss.get('obs_per_parameter', 0)
                adequate = ss.get('adequate', False)
                
                # Create bar chart
                categories = ['Actual\nObs/Param', 'Minimum\nRequired']
                values = [obs_per_param, 10]  # 10 is the minimum
                colors = ['green' if adequate else 'red', 'gray']
                
                bars = ax.bar(categories, values, color=colors, alpha=0.7)
                
                # Add value labels
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
                
                # Add threshold line
                ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Min required')
                ax.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='Recommended')
                
                ax.set_ylabel('Observations per Parameter')
                title_color = 'green' if adequate else 'red'
                ax.set_title(f'Sample Size Adequacy: {n_obs} obs, {n_params} params', 
                           color=title_color, fontweight='bold')
                ax.legend()
                
                # Add message
                msg = ss.get('message', '')
                ax.text(0.5, 0.02, msg, transform=ax.transAxes,
                       ha='center', fontsize=9, style='italic')
            else:
                ax.text(0.5, 0.5, 'Sample size check not performed', ha='center', va='center')
                ax.set_title('Sample Size Adequacy', fontweight='bold')
            
            # 3. Overdispersion & Theta
            ax = axes[1, 0]
            if 'overdispersion' in diagnostics_results:
                od = diagnostics_results['overdispersion']
                disp_ratio = od.get('dispersion_ratio', 1)
                theta = diagnostics_results.get('theta', None)
                
                # Create metrics dict
                metrics = {'Dispersion\nRatio': disp_ratio}
                if theta:
                    metrics['Theta (θ)'] = theta
                metrics['Target\n(Ideal)'] = 1.0
                
                x_pos = np.arange(len(metrics))
                values = list(metrics.values())
                
                # Color based on interpretation
                colors = []
                for i, (k, v) in enumerate(metrics.items()):
                    if 'Target' in k:
                        colors.append('green')
                    elif 'Dispersion' in k:
                        if v < 0.5 or v > 5:
                            colors.append('red')
                        elif v < 1.2:
                            colors.append('green')
                        else:
                            colors.append('orange')
                    elif 'Theta' in k:
                        if v < 0.1:
                            colors.append('red')
                        elif v > 100:
                            colors.append('orange')
                        else:
                            colors.append('steelblue')
                    else:
                        colors.append('gray')
                
                bars = ax.bar(x_pos, values, color=colors, alpha=0.7)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(metrics.keys())
                ax.set_ylabel('Value')
                
                # Add value labels
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{val:.2f}', ha='center', va='bottom')
                
                ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Ideal')
                ax.set_title('Overdispersion Diagnostics', fontweight='bold')
                ax.legend()
                
                # Add interpretation
                interp = od.get('interpretation', '')
                ax.text(0.5, 0.02, interp, transform=ax.transAxes,
                       ha='center', fontsize=9, style='italic')
            else:
                ax.text(0.5, 0.5, 'Overdispersion check not performed', ha='center', va='center')
                ax.set_title('Overdispersion Diagnostics', fontweight='bold')
            
            # 4. Coefficient Issues Summary
            ax = axes[1, 1]
            if 'coefficients' in diagnostics_results:
                coef_check = diagnostics_results['coefficients']
                issues = coef_check.get('issues', [])
                
                if issues:
                    # Create text summary of issues
                    y_pos = 0.9
                    ax.text(0.1, y_pos, 'Coefficient Issues Detected:', 
                           fontweight='bold', fontsize=12, transform=ax.transAxes)
                    
                    for i, issue in enumerate(issues[:8]):  # Show max 8 issues
                        y_pos -= 0.1
                        # Color based on issue type
                        if 'separation' in issue.lower():
                            color = 'darkred'
                        elif 'perfect' in issue.lower():
                            color = 'red'
                        elif 'extreme' in issue.lower():
                            color = 'darkorange'
                        else:
                            color = 'black'
                        
                        ax.text(0.1, y_pos, f'• {issue}', 
                               color=color, fontsize=10, transform=ax.transAxes)
                    
                    if len(issues) > 8:
                        y_pos -= 0.1
                        ax.text(0.1, y_pos, f'... and {len(issues)-8} more', 
                               style='italic', fontsize=9, transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, '✓ No coefficient issues detected', 
                           ha='center', va='center', color='green', fontsize=14, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'Coefficient check not performed', ha='center', va='center')
            
            ax.set_title('Coefficient Diagnostics', fontweight='bold')
            ax.axis('off')
            
            plt.suptitle('Model Diagnostics', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if output_dir:
                output_dir = Path(output_dir)
                save_path = output_dir / filename
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Model diagnostics saved to {save_path}")
                return save_path
            else:
                return fig
                
        except Exception as e:
            logger.error(f"Failed to create model diagnostics: {e}")
            return None
    
    def plot_coefficient_estimates(self,
                                  model_results: Dict[str, Any],
                                  output_dir: Optional[Path] = None,
                                  filename: str = "coefficient_estimates.png") -> Optional[Path]:
        """Create coefficient plot with confidence intervals.
        
        Shows coefficient estimates with 95% CI for statistical inference.
        This is the traditional statistical view of model parameters.
        
        Args:
            model_results: Model results with coefficients, std_errors, coef_names
            output_dir: Directory to save plot
            filename: Name of output file
            
        Returns:
            Path to saved plot or Figure object
        """
        try:
            # Extract coefficient information
            coefs = model_results.get('coefficients', [])
            std_errors = model_results.get('std_errors', [])
            coef_names = model_results.get('coef_names', 
                                         [f"coef_{i}" for i in range(len(coefs))])
            
            if len(coefs) == 0 or len(std_errors) == 0:
                logger.warning("No coefficients or standard errors available")
                return None
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, max(6, len(coefs) * 0.5)))
            
            # Calculate z-scores and p-values if available
            z_scores = np.array(coefs) / np.array(std_errors)
            p_values = model_results.get('coef_p_values', 
                                        2 * (1 - stats.norm.cdf(np.abs(z_scores))))
            
            # Prepare data for plotting
            y_pos = np.arange(len(coefs))
            
            # Color code by significance
            colors = []
            for i, p in enumerate(p_values):
                if p < 0.001:
                    colors.append('darkgreen')  # Highly significant
                elif p < 0.01:
                    colors.append('green')       # Very significant
                elif p < 0.05:
                    colors.append('orange')      # Significant
                else:
                    colors.append('gray')        # Not significant
            
            # Plot coefficients with error bars
            for i, (coef, stderr, color, name) in enumerate(zip(coefs, std_errors, colors, coef_names)):
                ax.errorbar(coef, y_pos[i], xerr=1.96*stderr, 
                          fmt='o', capsize=5, color=color, markersize=8,
                          label=f'p<0.001' if i == 0 and colors[0] == 'darkgreen' else '')
            
            # Add vertical line at zero
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
            
            # Customize plot
            ax.set_yticks(y_pos)
            ax.set_yticklabels(coef_names)
            ax.set_xlabel('Coefficient Value (95% CI)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Variables', fontsize=12, fontweight='bold')
            ax.set_title('Coefficient Estimates with 95% Confidence Intervals', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add significance legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='darkgreen', label='p < 0.001'),
                Patch(facecolor='green', label='p < 0.01'),
                Patch(facecolor='orange', label='p < 0.05'),
                Patch(facecolor='gray', label='p ≥ 0.05')
            ]
            ax.legend(handles=legend_elements, loc='best', title='Significance')
            
            # Add text annotations for exact values
            for i, (coef, stderr, p) in enumerate(zip(coefs, std_errors, p_values)):
                # Add coefficient value
                x_offset = max(abs(min(coefs)), abs(max(coefs))) * 0.02
                ax.text(coef + x_offset, y_pos[i], 
                       f'{coef:.3f}', 
                       va='center', fontsize=9)
                
                # Add significance stars
                if p < 0.001:
                    stars = '***'
                elif p < 0.01:
                    stars = '**'
                elif p < 0.05:
                    stars = '*'
                else:
                    stars = ''
                
                if stars:
                    ax.text(coef + 1.96*stderr + x_offset, y_pos[i], 
                           stars, va='center', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            if output_dir:
                output_dir = Path(output_dir)
                save_path = output_dir / filename
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Coefficient estimates plot saved to {save_path}")
                return save_path
            else:
                return fig
                
        except Exception as e:
            logger.error(f"Failed to create coefficient plot: {e}")
            return None
    
    def plot_gvif_only(self,
                       model_results: Dict[str, Any],
                       output_dir: Optional[Path] = None,
                       filename: str = "gvif_diagnostics.png") -> Optional[Path]:
        """Create GVIF-only diagnostic plot for multicollinearity assessment.
        
        This is a simplified version that only shows GVIF values, which is
        the most important diagnostic for GLM models.
        
        Args:
            model_results: Model results containing GVIF values
            output_dir: Directory to save plot
            filename: Name of output file
            
        Returns:
            Path to saved plot or None
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Extract GVIF data
            gvif_data = model_results.get('gvif', {})
            
            # Handle different GVIF data formats
            if isinstance(gvif_data, list):
                # If GVIF is a list, map to coefficient names (excluding intercept)
                coef_names = model_results.get('coef_names', [])
                if len(gvif_data) == len(coef_names) - 1:  # Exclude intercept
                    gvif_dict = {}
                    for name, value in zip(coef_names[1:], gvif_data):
                        gvif_dict[name] = value
                    gvif_data = gvif_dict
                else:
                    # Try to match by position
                    gvif_dict = {}
                    for i, value in enumerate(gvif_data):
                        name = coef_names[i+1] if i+1 < len(coef_names) else f"var_{i+1}"
                        gvif_dict[name] = value
                    gvif_data = gvif_dict
            
            if not gvif_data or not isinstance(gvif_data, dict):
                ax.text(0.5, 0.5, 'GVIF values not available', 
                       ha='center', va='center', fontsize=14)
                ax.set_title('Multicollinearity Check (GVIF)', fontsize=16, fontweight='bold')
                ax.axis('off')
            else:
                # Extract variables and GVIF values
                vars = list(gvif_data.keys())
                gvif_values = list(gvif_data.values())
                
                # Calculate adjusted GVIF (square root for single df predictors)
                adjusted_gvif = [np.sqrt(v) for v in gvif_values]
                
                # Color coding based on severity
                colors = []
                for v in adjusted_gvif:
                    if v > 2:
                        colors.append('#d62728')  # Red - problematic
                    elif v > 1.5:
                        colors.append('#ff7f0e')  # Orange - moderate concern
                    else:
                        colors.append('#2ca02c')  # Green - no concern
                
                # Create horizontal bar chart
                y_pos = np.arange(len(vars))
                bars = ax.barh(y_pos, adjusted_gvif, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
                
                # Add threshold lines
                ax.axvline(x=1.5, color='#ff7f0e', linestyle='--', alpha=0.5, 
                          label='Low concern (√GVIF = 1.5)', linewidth=2)
                ax.axvline(x=2.0, color='#d62728', linestyle='--', alpha=0.5, 
                          label='High concern (√GVIF = 2.0)', linewidth=2)
                
                # Customize axes
                ax.set_yticks(y_pos)
                ax.set_yticklabels(vars, fontsize=11)
                ax.set_xlabel('√GVIF (Adjusted GVIF)', fontsize=12, fontweight='bold')
                ax.set_title('Generalized Variance Inflation Factor (GVIF) - Multicollinearity Check', 
                           fontsize=14, fontweight='bold')
                
                # Add value labels on bars
                for i, (bar, val, orig_val) in enumerate(zip(bars, adjusted_gvif, gvif_values)):
                    label_x = bar.get_width() + 0.02
                    ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                           f'{val:.2f}\n(GVIF={orig_val:.2f})', 
                           va='center', fontsize=9)
                
                # Add legend
                ax.legend(loc='upper right', framealpha=0.9)
                
                # Add interpretation text
                max_adjusted = max(adjusted_gvif)
                # Remove interpretation text per user request - just show the GVIF values
                
                # Add grid for better readability
                ax.grid(True, alpha=0.3, axis='x', linestyle='--')
                
                # Add sample size info if available
                n_obs = model_results.get('n_obs', 0)
                n_params = len(model_results.get('coefficients', []))
                if n_obs > 0:
                    sample_info = f"Sample size: {n_obs:,} obs, {n_params} params"
                    obs_per_param = n_obs / n_params if n_params > 0 else 0
                    sample_info += f"\n{obs_per_param:.0f} obs/param"
                    
                    ax.text(0.02, 0.98, sample_info, transform=ax.transAxes,
                           va='top', fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
            
            plt.tight_layout()
            
            if output_dir:
                output_dir = Path(output_dir)
                save_path = output_dir / filename
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"GVIF diagnostics saved to {save_path}")
                return save_path
            else:
                return fig
                
        except Exception as e:
            logger.error(f"Failed to create GVIF diagnostics: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_all_diagnostics(self,
                               model_results: Dict[str, Any],
                               output_dir: Path,
                               model_name: str = "model") -> List[Path]:
        """Create all diagnostic plots for a model.
        
        Simple wrapper that generates all available diagnostic visualizations.
        
        Args:
            model_results: Model results from Julia containing residuals, coefficients, etc.
            output_dir: Directory to save plots
            model_name: Name prefix for files
            
        Returns:
            List of paths to saved plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        
        # 1. Residual diagnostics (if residuals available)
        if 'residuals' in model_results or 'deviance_residuals' in model_results:
            residuals = {model_name: model_results.get('deviance_residuals', 
                                                       model_results.get('residuals', []))}
            fitted = {model_name: model_results.get('fitted_values', [])}
            
            if len(residuals[model_name]) > 0 and len(fitted[model_name]) > 0:
                path = self.create_residual_diagnostics(
                    residuals=residuals,
                    fitted_values=fitted,
                    output_dir=output_dir,
                    filename=f"{model_name}_residual_diagnostics.png"
                )
                if path:
                    paths.append(path)
        
        # 2. Coefficient plot (if coefficients available)
        if 'coefficients' in model_results and 'std_errors' in model_results:
            path = self.plot_coefficient_estimates(
                model_results=model_results,
                output_dir=output_dir,
                filename=f"{model_name}_coefficients.png"
            )
            if path:
                paths.append(path)
        
        # 3. Deviance-df plot (if comparing multiple models)
        # This requires multiple models, so skip for single model
        
        # 4. Model diagnostics (if diagnostic results available)
        if 'gvif' in model_results or 'overdispersion' in model_results:
            # Convert GVIF list to dict if needed
            gvif_data = model_results.get('gvif', {})
            if isinstance(gvif_data, list) and 'coef_names' in model_results:
                # Map GVIF values to names (excluding intercept)
                names = model_results['coef_names']
                if len(gvif_data) == len(names) - 1:
                    gvif_dict = {}
                    for name, value in zip(names[1:], gvif_data):
                        gvif_dict[name] = value
                    gvif_data = gvif_dict
            
            diagnostics_data = {
                'multicollinearity': {'gvif': gvif_data},
                'overdispersion': {
                    'dispersion_ratio': model_results.get('dispersion', 1.0),
                    'interpretation': 'Check dispersion ratio'
                },
                'theta': model_results.get('theta'),
                'sample_size': {
                    'n_obs': model_results.get('n_obs', 0),
                    'n_parameters': len(model_results.get('coefficients', [])),
                    'obs_per_parameter': model_results.get('n_obs', 0) / max(1, len(model_results.get('coefficients', [])))
                },
                'coefficients': {
                    'issues': []  # No issues to report for now
                }
            }
            
            path = self.plot_model_diagnostics(
                diagnostics_results=diagnostics_data,
                output_dir=output_dir,
                filename=f"{model_name}_model_diagnostics.png"
            )
            if path:
                paths.append(path)
        
        logger.info(f"Generated {len(paths)} diagnostic plots for {model_name}")
        return paths