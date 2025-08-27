"""
Visualization Module for EconML Diagnostics

Provides comprehensive plotting functions for all diagnostic results
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


class DiagnosticsVisualizer:
    """
    Create visualizations for EconML diagnostic results
    """
    
    def __init__(self, results: Dict[str, Any], output_dir: Optional[str] = None):
        """
        Initialize visualizer with diagnostic results
        
        Parameters
        ----------
        results : dict
            Diagnostic results from EconMLDiagnostics
        output_dir : str, optional
            Directory to save plots
        """
        self.results = results
        self.output_dir = output_dir
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def plot_all(self, save: bool = True, show: bool = False) -> Dict[str, plt.Figure]:
        """
        Generate all available plots
        
        Returns
        -------
        dict of figure names to Figure objects
        """
        figures = {}
        
        # 1. Sensitivity Analysis Plots
        if 'sensitivity' in self.results:
            try:
                fig = self.plot_sensitivity_analysis()
                figures['sensitivity'] = fig
            except Exception as e:
                print(f"Could not plot sensitivity: {e}")
        
        # 2. Nuisance Model Performance
        if 'nuisance_scores' in self.results:
            try:
                fig = self.plot_nuisance_scores()
                figures['nuisance'] = fig
            except Exception as e:
                print(f"Could not plot nuisance scores: {e}")
        
        # 3. Treatment Effects
        if 'coefficients' in self.results:
            try:
                fig = self.plot_treatment_effects()
                figures['effects'] = fig
            except Exception as e:
                print(f"Could not plot treatment effects: {e}")
        
        # 4. Feature Importance
        if 'feature_importance' in self.results:
            try:
                fig = self.plot_feature_importance()
                figures['importance'] = fig
            except Exception as e:
                print(f"Could not plot feature importance: {e}")
        
        # Save if requested
        if save and self.output_dir:
            import os
            os.makedirs(self.output_dir, exist_ok=True)
            for name, fig in figures.items():
                fig.savefig(f"{self.output_dir}/{name}.png", dpi=300, bbox_inches='tight')
                print(f"Saved {name}.png")
        
        if show:
            plt.show()
        
        return figures
    
    def plot_sensitivity_analysis(self) -> plt.Figure:
        """
        Create comprehensive sensitivity analysis visualization
        
        Includes:
        - Robustness values bar chart
        - Sensitivity intervals under different confounding levels
        - Confidence interval evolution
        """
        sens = self.results.get('sensitivity', {})
        
        # Check if multi-treatment
        treatments = []
        for key, value in sens.items():
            if isinstance(value, dict) and 'treatment' in value:
                treatments.append(value)
        
        if not treatments:
            # Single treatment or fallback
            return self._plot_single_sensitivity(sens)
        
        # Multi-treatment sensitivity plot
        n_treatments = len(treatments)
        fig, axes = plt.subplots(n_treatments, 2, figsize=(12, 5 * n_treatments))
        
        # Handle single treatment case
        if n_treatments == 1:
            axes = axes.reshape(1, -1)
        
        for i, treatment in enumerate(treatments):
            treatment_name = treatment.get('treatment', f'Treatment {i+1}')
            
            # Plot 1: Robustness Value (left column)
            ax1 = axes[i, 0]
            if 'robustness' in treatment and treatment['robustness'].get('value'):
                rv = treatment['robustness']['value']
                color = 'green' if rv > 0.5 else 'orange' if rv > 0.1 else 'red'
                
                ax1.barh(['RV'], [rv], color=color, alpha=0.7, height=0.5)
                ax1.set_xlim(0, 1)
                ax1.axvline(x=0.1, color='orange', linestyle='--', alpha=0.5, label='Moderate')
                ax1.axvline(x=0.5, color='green', linestyle='--', alpha=0.5, label='Robust')
                ax1.set_xlabel('R² of Confounder Needed to Invalidate')
                ax1.set_title(f'{treatment_name}: Robustness Value = {rv:.3f}')
                if i == 0:  # Only show legend on first plot
                    ax1.legend(loc='upper right', fontsize=9)
            
            # Plot 2: Sensitivity Intervals (right column)
            ax2 = axes[i, 1]
            if 'sensitivity_intervals' in treatment:
                
                cf_levels = []
                lower_bounds = []
                upper_bounds = []
                significant = []
                
                for cf_key, interval in treatment['sensitivity_intervals'].items():
                    if isinstance(interval, dict) and 'lower' in interval:
                        cf_level = interval.get('confounding_level', float(cf_key.split('_')[1]))
                        cf_levels.append(cf_level)
                        lower_bounds.append(interval['lower'])
                        upper_bounds.append(interval['upper'])
                        significant.append(interval.get('significant', False))
                
                if cf_levels:
                    # Plot confidence intervals
                    x = np.arange(len(cf_levels))
                    centers = [(l + u) / 2 for l, u in zip(lower_bounds, upper_bounds)]
                    errors = [[c - l for c, l in zip(centers, lower_bounds)],
                             [u - c for u, c in zip(upper_bounds, centers)]]
                    
                    colors = ['green' if sig else 'red' for sig in significant]
                    
                    # Plot error bars
                    for j, (xi, c, lb, ub, col) in enumerate(zip(x, centers, lower_bounds, upper_bounds, colors)):
                        ax2.errorbar(xi, c, yerr=[[c - lb], [ub - c]], fmt='o', 
                                   color=col, capsize=5, capthick=2, markersize=10,
                                   alpha=0.7, elinewidth=2)
                    
                    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    ax2.set_xticks(x)
                    ax2.set_xticklabels([f'{cf:.0%}' for cf in cf_levels])
                    ax2.set_xlabel('Confounding Level (R²)')
                    ax2.set_ylabel('Treatment Effect')
                    ax2.set_title(f'Sensitivity to Confounding')
                    ax2.grid(True, alpha=0.3)
                    
                    # Add legend
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor='green', alpha=0.7, label='Still Significant'),
                        Patch(facecolor='red', alpha=0.7, label='Not Significant')
                    ]
                    ax2.legend(handles=legend_elements, loc='upper right')
        
        plt.suptitle('Sensitivity Analysis: Robustness to Unobserved Confounding', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def _plot_single_sensitivity(self, sens: Dict) -> plt.Figure:
        """Plot sensitivity for single treatment or fallback"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot robustness value if available
        if 'robustness' in sens and sens['robustness'].get('value'):
            rv = sens['robustness']['value']
            color = 'green' if rv > 0.5 else 'orange' if rv > 0.1 else 'red'
            
            ax1.barh(['Robustness\nValue'], [rv], color=color, alpha=0.7)
            ax1.set_xlim(0, 1)
            ax1.axvline(x=0.1, color='orange', linestyle='--', alpha=0.5)
            ax1.axvline(x=0.5, color='green', linestyle='--', alpha=0.5)
            ax1.set_xlabel('R² of Confounder Needed')
            ax1.set_title(f'Robustness Value: {rv:.3f}')
        
        # Plot treatments if fallback
        if 'treatments' in sens:
            treatments = sens['treatments']
            names = list(treatments.keys())
            rvs = [t.get('robustness_value', 0) for t in treatments.values()]
            
            # Convert '>1.0' to 1.0
            rvs = [1.0 if rv == '>1.0' else rv for rv in rvs]
            
            colors = ['green' if rv > 0.5 else 'orange' if rv > 0.1 else 'red' 
                     for rv in rvs]
            
            ax2.barh(names, rvs, color=colors, alpha=0.7)
            ax2.set_xlim(0, 1.1)
            ax2.axvline(x=0.1, color='orange', linestyle='--', alpha=0.5, label='Moderate')
            ax2.axvline(x=0.5, color='green', linestyle='--', alpha=0.5, label='Robust')
            ax2.set_xlabel('Robustness Value')
            ax2.set_title('Treatment Robustness')
            ax2.legend()
        
        plt.suptitle('Sensitivity Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_nuisance_scores(self) -> plt.Figure:
        """
        Plot nuisance model performance (R² scores)
        """
        nuisance = self.results.get('nuisance_scores', {})
        
        if not nuisance or 'error' in nuisance:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(0.5, 0.5, 'No nuisance scores available', 
                   ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Check for models structure
        models = nuisance.get('models', {})
        
        # Plot Y model scores
        y_model = models.get('model_Y (fungi richness)', {})
        if y_model and 'fold_scores' in y_model:
            scores = y_model['fold_scores']
            mean_score = y_model.get('mean_r2', np.mean(scores))
            std_score = y_model.get('std_r2', np.std(scores))
            
            axes[0].hist(scores, bins=min(len(scores), 10), alpha=0.7, color='blue', edgecolor='black')
            axes[0].axvline(mean_score, color='red', linestyle='--', 
                          label=f'Mean: {mean_score:.4f}')
            axes[0].set_xlabel('R² Score')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title(f'Outcome Model (Y) Performance\nR²={mean_score:.4f} (±{std_score:.4f})')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_xlim(min(scores) - 0.01, max(scores) + 0.01)
        
        # Plot T model scores
        t_model = models.get('model_T (biodiversity)', {})
        if t_model and 'fold_scores' in t_model:
            scores = t_model['fold_scores']
            mean_score = t_model.get('mean_r2', np.mean(scores))
            std_score = t_model.get('std_r2', np.std(scores))
            
            axes[1].hist(scores, bins=min(len(scores), 10), alpha=0.7, color='green', edgecolor='black')
            axes[1].axvline(mean_score, color='red', linestyle='--', 
                          label=f'Mean: {mean_score:.4f}')
            axes[1].set_xlabel('R² Score')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title(f'Treatment Model (T) Performance\nR²={mean_score:.4f} (±{std_score:.4f})')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_xlim(min(scores) - 0.001, max(scores) + 0.001)
        
        # Add interpretation
        fig.text(0.5, 0.02, 
                'Higher R² indicates confounders explain more variation (good for DML)',
                ha='center', fontsize=10, style='italic')
        
        plt.suptitle('Nuisance Model Performance (Cross-Validation)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_treatment_effects(self) -> plt.Figure:
        """
        Plot treatment effects with confidence intervals
        """
        coef = self.results.get('coefficients', {})
        
        if not coef or not coef.get('has_coefficients', True):
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(0.5, 0.5, 'No coefficients available\n(Non-parametric model)', 
                   ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Get coefficients
        intercepts = coef.get('intercept', [])
        if not isinstance(intercepts, list):
            intercepts = [intercepts]
        
        # Get confidence intervals - try intercept CI first
        ci_lower = coef.get('intercept_ci_lower', coef.get('coef_ci_lower', []))
        ci_upper = coef.get('intercept_ci_upper', coef.get('coef_ci_upper', []))
        
        # Flatten if nested
        if ci_lower and isinstance(ci_lower[0], list):
            ci_lower = [item for sublist in ci_lower for item in sublist] if any(ci_lower) else []
        if ci_upper and isinstance(ci_upper[0], list):
            ci_upper = [item for sublist in ci_upper for item in sublist] if any(ci_upper) else []
        
        if not isinstance(ci_lower, list):
            ci_lower = [ci_lower]
        if not isinstance(ci_upper, list):
            ci_upper = [ci_upper]
        
        # Default treatment names
        treatment_names = ['Plants\nRichness', 'Vertebrates\nRichness', 'Plants ×\nVertebrates']
        treatment_names = treatment_names[:len(intercepts)]
        
        # Create bar plot with error bars
        x = np.arange(len(treatment_names))
        
        # Determine colors based on significance
        colors = []
        for i in range(len(intercepts)):
            if i < len(ci_lower) and i < len(ci_upper):
                # Significant if CI doesn't include 0
                if ci_lower[i] > 0 or ci_upper[i] < 0:
                    colors.append('green')
                else:
                    colors.append('gray')
            else:
                colors.append('blue')
        
        # Plot bars
        bars = ax.bar(x, intercepts, color=colors, alpha=0.7, edgecolor='black')
        
        # Add error bars if available
        if ci_lower and ci_upper and len(ci_lower) == len(intercepts):
            errors = [[intercepts[i] - ci_lower[i] for i in range(len(intercepts))],
                     [ci_upper[i] - intercepts[i] for i in range(len(intercepts))]]
            ax.errorbar(x, intercepts, yerr=errors, fmt='none', 
                       color='black', capsize=5, capthick=2)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, intercepts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.6f}' if abs(val) < 0.01 else f'{val:.4f}',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=10)
        
        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels(treatment_names)
        ax.set_ylabel('Effect Size')
        ax.set_title('Treatment Effects on Fungi Richness\n(with 95% Confidence Intervals)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Significant (p<0.05)'),
            Patch(facecolor='gray', alpha=0.7, label='Not Significant')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        return fig
    
    def plot_feature_importance(self) -> plt.Figure:
        """
        Plot feature importance or coefficient magnitudes
        """
        importance = self.results.get('feature_importance', {})
        
        if not importance or 'error' in importance:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(0.5, 0.5, 'Feature importance not available', 
                   ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        # Check if we have treatment effects to show instead
        if 'magnitudes' in importance and importance.get('shape') == [3, 0]:
            # LinearDML case - show treatment effects as "importance"
            coef = self.results.get('coefficients', {})
            if coef and 'intercept' in coef:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                
                intercepts = coef['intercept']
                if not isinstance(intercepts, list):
                    intercepts = [intercepts]
                
                treatment_names = ['Plants Richness', 'Vertebrates Richness', 'Plants × Vertebrates']
                treatment_names = treatment_names[:len(intercepts)]
                
                # Sort by absolute magnitude
                sorted_pairs = sorted(zip(treatment_names, intercepts), 
                                    key=lambda x: abs(x[1]), reverse=True)
                sorted_names, sorted_mags = zip(*sorted_pairs)
                
                # Create horizontal bar plot
                y = np.arange(len(sorted_names))
                colors = ['green' if m > 0 else 'red' for m in sorted_mags]
                
                bars = ax.barh(y, sorted_mags, color=colors, alpha=0.7)
                
                # Add value labels
                for bar, val in zip(bars, sorted_mags):
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2,
                           f'{val:.6f}' if abs(val) < 0.01 else f'{val:.4f}',
                           ha='left' if width > 0 else 'right',
                           va='center', fontsize=10)
                
                ax.set_yticks(y)
                ax.set_yticklabels(sorted_names)
                ax.set_xlabel('Treatment Effect Size')
                ax.set_title('Treatment Effect Magnitudes\n(Causal Effects on Fungi Richness)')
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                ax.grid(True, alpha=0.3, axis='x')
                
                # Add note
                fig.text(0.5, 0.02, 
                        'Note: LinearDML estimates treatment effects only, not feature importance',
                        ha='center', fontsize=10, style='italic')
                
                plt.tight_layout()
                return fig
        
        # For linear models, plot coefficient magnitudes
        if 'coefficient_magnitudes' in importance:
            mags = importance['coefficient_magnitudes']
            
            if isinstance(mags, (list, np.ndarray)) and len(mags) > 0:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                
                # Default feature names
                feature_names = [f'Feature {i+1}' for i in range(len(mags))]
                
                # Sort by magnitude
                sorted_idx = np.argsort(np.abs(mags))[::-1][:20]  # Top 20
                sorted_mags = [mags[i] for i in sorted_idx]
                sorted_names = [feature_names[i] for i in sorted_idx]
                
                # Create horizontal bar plot
                y = np.arange(len(sorted_names))
                colors = ['green' if m > 0 else 'red' for m in sorted_mags]
                
                ax.barh(y, sorted_mags, color=colors, alpha=0.7)
                ax.set_yticks(y)
                ax.set_yticklabels(sorted_names)
                ax.set_xlabel('Coefficient Magnitude')
                ax.set_title('Feature Importance (Top 20)')
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
                plt.tight_layout()
                return fig
        
        # Fallback - explain why no feature importance
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.6, 'Feature Importance Not Available', 
               ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(0.5, 0.4, 'LinearDML estimates treatment effects only.\n' +
                         'Confounders (W) are handled by nuisance models\n' +
                         '(LightGBM) which have their own feature importance\n' +
                         'but are not exposed in the final causal model.', 
               ha='center', va='center', fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        return fig
    
    def create_summary_figure(self) -> plt.Figure:
        """
        Create a comprehensive summary figure with key metrics
        """
        fig = plt.figure(figsize=(15, 10))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Model Score (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        score = self.results.get('score', {})
        if score and 'final_mse' in score:
            mse = score['final_mse']
            ax1.text(0.5, 0.5, f'Final MSE\n{mse:.4f}', 
                    ha='center', va='center', fontsize=14, fontweight='bold')
            ax1.set_title('Model Performance')
        else:
            ax1.text(0.5, 0.5, 'No score\navailable', ha='center', va='center')
        ax1.axis('off')
        
        # 2. Nuisance R² (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        nuisance = self.results.get('nuisance_scores', {})
        if nuisance and 'Y_score_mean' in nuisance:
            y_r2 = nuisance['Y_score_mean']
            t_r2 = nuisance.get('T_score_mean', 0)
            ax2.bar(['Outcome (Y)', 'Treatment (T)'], [y_r2, t_r2], 
                   color=['blue', 'green'], alpha=0.7)
            ax2.set_ylim(0, 1)
            ax2.set_ylabel('R²')
            ax2.set_title('Nuisance Models')
        else:
            ax2.text(0.5, 0.5, 'No nuisance\nscores', ha='center', va='center')
            ax2.axis('off')
        
        # 3. Sensitivity Summary (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        sens = self.results.get('sensitivity', {})
        
        # Extract robustness values
        rvs = []
        names = []
        
        # Check for multi-treatment
        for key, value in sens.items():
            if isinstance(value, dict) and 'robustness' in value:
                if value['robustness'] and value['robustness'].get('value'):
                    rvs.append(value['robustness']['value'])
                    names.append(value.get('treatment', key))
        
        if rvs:
            colors = ['green' if rv > 0.5 else 'orange' if rv > 0.1 else 'red' 
                     for rv in rvs]
            ax3.barh(names[:3], rvs[:3], color=colors, alpha=0.7)
            ax3.set_xlim(0, 1)
            ax3.set_xlabel('Robustness Value')
            ax3.set_title('Sensitivity Analysis')
        else:
            ax3.text(0.5, 0.5, 'No sensitivity\nanalysis', ha='center', va='center')
            ax3.axis('off')
        
        # 4. Treatment Effects (bottom, spanning 2 columns)
        ax4 = fig.add_subplot(gs[1:, :2])
        coef = self.results.get('coefficients', {})
        if coef and 'intercept' in coef:
            intercepts = coef['intercept']
            if not isinstance(intercepts, list):
                intercepts = [intercepts]
            
            treatment_names = ['Plants', 'Vertebrates', 'Interaction'][:len(intercepts)]
            x = np.arange(len(treatment_names))
            
            # Get CIs - try intercept CI first
            ci_lower = coef.get('intercept_ci_lower', coef.get('coef_ci_lower', []))
            ci_upper = coef.get('intercept_ci_upper', coef.get('coef_ci_upper', []))
            
            # Flatten if nested
            if ci_lower and isinstance(ci_lower[0], list):
                ci_lower = [item for sublist in ci_lower for item in sublist] if any(ci_lower) else []
            if ci_upper and isinstance(ci_upper[0], list):
                ci_upper = [item for sublist in ci_upper for item in sublist] if any(ci_upper) else []
            
            if not isinstance(ci_lower, list):
                ci_lower = [ci_lower]
            if not isinstance(ci_upper, list):
                ci_upper = [ci_upper]
            
            # Plot
            bars = ax4.bar(x, intercepts, alpha=0.7, edgecolor='black')
            
            if ci_lower and ci_upper and len(ci_lower) == len(intercepts):
                errors = [[intercepts[i] - ci_lower[i] for i in range(len(intercepts))],
                         [ci_upper[i] - intercepts[i] for i in range(len(intercepts))]]
                ax4.errorbar(x, intercepts, yerr=errors, fmt='none', 
                           color='black', capsize=5)
            
            ax4.set_xticks(x)
            ax4.set_xticklabels(treatment_names)
            ax4.set_ylabel('Effect Size')
            ax4.set_title('Treatment Effects with 95% CI')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Key Statistics (bottom right)
        ax5 = fig.add_subplot(gs[1:, 2])
        ax5.axis('off')
        
        # Collect key stats
        stats_text = "KEY STATISTICS\n" + "="*20 + "\n\n"
        
        # Sample size
        model_info = self.results.get('model_info', {})
        if model_info:
            if 'cv_folds' in model_info:
                stats_text += f"CV Folds: {model_info['cv_folds']}\n"
            if 'mc_iters' in model_info:
                stats_text += f"MC Iterations: {model_info['mc_iters']}\n"
        
        # Add sensitivity summary
        if rvs:
            min_rv = min(rvs)
            if min_rv > 0.5:
                stats_text += "\n✓ ROBUST to confounding"
            elif min_rv > 0.1:
                stats_text += "\n⚠ MODERATELY ROBUST"
            else:
                stats_text += "\n✗ SENSITIVE to confounding"
        
        ax5.text(0.1, 0.9, stats_text, ha='left', va='top', 
                fontsize=11, transform=ax5.transAxes)
        
        plt.suptitle('EconML Diagnostics Summary', fontsize=16, fontweight='bold')
        
        return fig