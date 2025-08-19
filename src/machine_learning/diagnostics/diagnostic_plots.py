"""Diagnostic Plots for GLM Models.

Generic diagnostic visualization module that can be used by any method.
Creates diagnostic plots comparing model statistics (theta, deviance, etc.)
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


class DiagnosticPlots:
    """Create diagnostic plots for GLM model comparisons.
    
    This class generates diagnostic visualizations for any set of models,
    typically comparing null vs full models or different model specifications.
    """
    
    def __init__(self):
        """Initialize diagnostic plots generator."""
        self.fig = None
        self.axes = None
        
    def create_diagnostic_plots(self,
                               models: Dict[str, Dict[str, Any]],
                               output_dir: Optional[Path] = None,
                               filename: str = "diagnostics.png",
                               title: str = "Model Diagnostics") -> Optional[Path]:
        """Create diagnostic plots for model comparison.
        
        Args:
            models: Dictionary of models to compare, e.g.:
                   {'null': {...model_data...}, 'full': {...model_data...}}
                   Each model should have: theta, deviance, df_residual, aic, n_obs, n_parameters
            output_dir: Directory to save plot (if None, returns figure)
            filename: Name of output file
            title: Main title for the plot
            
        Returns:
            Path to saved plot or None
        """
        try:
            # Validate we have at least one model
            if not models:
                logger.warning("No models provided for diagnostic plots")
                return None
            
            # Setup figure
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            self.fig = fig
            self.axes = axes
            
            # Extract model names and sort (null first if present)
            model_names = list(models.keys())
            if 'null' in model_names and 'full' in model_names:
                # Ensure null comes first for consistent ordering
                model_names = ['null', 'full'] + [m for m in model_names if m not in ['null', 'full']]
            
            # Prepare data for plotting
            thetas = []
            deviances = []
            mean_deviances = []
            model_labels = []
            
            for name in model_names:
                if name in models:
                    model = models[name]
                    thetas.append(model.get('theta', 0))
                    deviances.append(model.get('deviance', 0))
                    
                    # Calculate mean deviance
                    dev = model.get('deviance', 0)
                    df = model.get('df_residual', 1)
                    mean_deviances.append(dev / df if df > 0 else 0)
                    
                    # Create display label
                    if name == 'null':
                        model_labels.append('Null')
                    elif name == 'full':
                        model_labels.append('Full')
                    else:
                        model_labels.append(name.capitalize())
            
            # Define colors for consistency
            colors = ['coral', 'steelblue', 'green', 'purple'][:len(model_names)]
            
            # 1. Theta (Overdispersion) Comparison
            ax = axes[0, 0]
            bars = ax.bar(model_labels, thetas, color=colors, alpha=0.7)
            ax.set_ylabel('Theta (θ)', fontsize=11)
            ax.set_title('Overdispersion Parameter', fontsize=12, fontweight='bold')
            
            # Add value labels and interpretation
            for bar, theta in zip(bars, thetas):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{theta:.2f}', ha='center', va='bottom', fontsize=10)
            
            # Add interpretation text if we have 2 models
            if len(thetas) == 2:
                if thetas[1] > thetas[0] * 1.2:
                    interp = "Full model shows less overdispersion"
                elif thetas[1] < thetas[0] * 0.8:
                    interp = "Full model shows more overdispersion"
                else:
                    interp = "Similar overdispersion in both models"
                ax.text(0.5, -0.15, interp, ha='center', va='top', 
                       transform=ax.transAxes, fontsize=9, style='italic')
            
            # 2. Deviance Comparison
            ax = axes[0, 1]
            bars = ax.bar(model_labels, deviances, color=colors, alpha=0.7)
            ax.set_ylabel('Deviance', fontsize=11)
            ax.set_title('Model Deviance', fontsize=12, fontweight='bold')
            
            for bar, dev in zip(bars, deviances):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{dev:.0f}', ha='center', va='bottom', fontsize=10)
            
            # Show improvement if comparing null vs full
            if len(deviances) == 2 and 'null' in models and 'full' in models:
                improvement = deviances[0] - deviances[1]
                if improvement > 0:
                    ax.text(0.5, -0.15, f'Improvement: {improvement:.0f} ({improvement/deviances[0]*100:.1f}%)',
                           ha='center', va='top', transform=ax.transAxes, fontsize=9, style='italic')
            
            # 3. Mean Deviance (Deviance/df)
            ax = axes[1, 0]
            bars = ax.bar(model_labels, mean_deviances, color=colors, alpha=0.7)
            ax.set_ylabel('Mean Deviance', fontsize=11)
            ax.set_title('Deviance per Degree of Freedom', fontsize=12, fontweight='bold')
            ax.axhline(y=1, color='black', linestyle='--', alpha=0.3, label='Ideal (=1)')
            
            for bar, md in zip(bars, mean_deviances):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{md:.3f}', ha='center', va='bottom', fontsize=10)
            
            ax.legend()
            
            # 4. Summary Statistics
            ax = axes[1, 1]
            ax.axis('off')
            
            summary_text = "Model Diagnostics Summary\n" + "="*30 + "\n\n"
            
            # Add summary for each model
            for name, label in zip(model_names, model_labels):
                if name in models:
                    model = models[name]
                    summary_text += f"{label} Model"
                    if name == 'null':
                        summary_text += " (baseline)"
                    elif name == 'full':
                        summary_text += " (with predictors)"
                    summary_text += ":\n"
                    
                    summary_text += f"  • Sample size: {model.get('n_obs', 0):,}\n"
                    summary_text += f"  • Parameters: {model.get('n_parameters', 0)}\n"
                    summary_text += f"  • AIC: {model.get('aic', 0):.2f}\n"
                    summary_text += f"  • BIC: {model.get('bic', 0):.2f}\n"
                    summary_text += f"  • Theta: {model.get('theta', 0):.3f}\n"
                    summary_text += f"  • Deviance: {model.get('deviance', 0):.0f}\n\n"
            
            # Add goodness of fit assessment
            fit_assessments = []
            for md, label in zip(mean_deviances, model_labels):
                if md < 1.2 and md > 0.8:
                    fit_assessments.append(f"{label}: ✓ Good fit")
                elif md > 1.5:
                    fit_assessments.append(f"{label}: ⚠ Potential underfitting")
                elif md < 0.5:
                    fit_assessments.append(f"{label}: ⚠ Potential overfitting")
                else:
                    fit_assessments.append(f"{label}: ○ Acceptable fit")
            
            summary_text += "Fit Assessment:\n"
            for assessment in fit_assessments:
                summary_text += f"  {assessment}\n"
            
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.5))
            
            plt.suptitle(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save or return
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / filename
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Diagnostic plots saved to {save_path}")
                return save_path
            else:
                return fig
            
        except Exception as e:
            logger.error(f"Failed to create diagnostic plots: {e}")
            if self.fig:
                plt.close(self.fig)
            return None
    
    def create_residual_diagnostics(self,
                                   residuals: Dict[str, np.ndarray],
                                   fitted_values: Dict[str, np.ndarray],
                                   output_dir: Optional[Path] = None,
                                   filename: str = "residual_diagnostics.png") -> Optional[Path]:
        """Create residual diagnostic plots.
        
        Args:
            residuals: Dictionary of residuals for each model
            fitted_values: Dictionary of fitted values for each model
            output_dir: Directory to save plot
            filename: Name of output file
            
        Returns:
            Path to saved plot or None
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            model_names = list(residuals.keys())
            colors = ['coral', 'steelblue', 'green', 'purple'][:len(model_names)]
            
            for i, (name, color) in enumerate(zip(model_names, colors)):
                if name in residuals and name in fitted_values:
                    resid = residuals[name]
                    fitted = fitted_values[name]
                    
                    # 1. Residuals vs Fitted
                    ax = axes[0, 0]
                    ax.scatter(fitted, resid, alpha=0.5, s=1, color=color, label=name.capitalize())
                    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    ax.set_xlabel('Fitted Values')
                    ax.set_ylabel('Residuals')
                    ax.set_title('Residuals vs Fitted')
                    ax.legend()
                    
                    # 2. Q-Q plot
                    ax = axes[0, 1]
                    from scipy import stats
                    stats.probplot(resid, dist="norm", plot=ax)
                    ax.set_title(f'Q-Q Plot ({name.capitalize()})')
                    
                    # 3. Scale-Location
                    ax = axes[1, 0]
                    standardized_resid = resid / np.std(resid)
                    ax.scatter(fitted, np.sqrt(np.abs(standardized_resid)), 
                              alpha=0.5, s=1, color=color)
                    ax.set_xlabel('Fitted Values')
                    ax.set_ylabel('√|Standardized Residuals|')
                    ax.set_title('Scale-Location')
                    
                    # 4. Residual histogram
                    ax = axes[1, 1]
                    ax.hist(resid, bins=50, alpha=0.5, color=color, label=name.capitalize())
                    ax.set_xlabel('Residuals')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Residual Distribution')
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