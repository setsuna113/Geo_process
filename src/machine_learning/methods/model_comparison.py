"""Model Comparison Visualization Module.

Generic model comparison module that can be used by any method to compare
models using information criteria (AIC, BIC), weights, and evidence ratios.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from scipy.stats import chi2

logger = logging.getLogger(__name__)


class ModelComparison:
    """Create comprehensive model comparison visualizations.
    
    This class generates model comparison plots that can be used by any method
    to compare nested or non-nested models using various criteria.
    """
    
    def __init__(self):
        """Initialize model comparison visualizer."""
        self.fig = None
        
    def create_comparison_plot(self,
                               models: Dict[str, Dict[str, Any]],
                               test_results: Optional[Dict[str, Any]] = None,
                               output_dir: Optional[Path] = None,
                               filename: str = "model_comparison.png",
                               title: str = "Model Comparison") -> Optional[Path]:
        """Create comprehensive model comparison visualization.
        
        Args:
            models: Dictionary of models to compare with their metrics
                   Each model should have: aic, bic, n_parameters, theta (optional)
                   Example: {'null': {...}, 'full': {...}}
            test_results: Optional dictionary with test results (LRT, Score, etc.)
                         Example: {'lrt': {...}, 'score': {...}}
            output_dir: Directory to save plot
            filename: Name of output file
            title: Main title for the plot
            
        Returns:
            Path to saved plot or None
        """
        try:
            # Calculate derived metrics
            comparison_metrics = self._calculate_comparison_metrics(models)
            
            # Create figure with subplots
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            self.fig = fig
            
            # 1. Information Criteria Values (top left)
            self._plot_information_criteria(fig.add_subplot(gs[0, 0]), models, comparison_metrics)
            
            # 2. Delta Information Criteria (top middle)
            self._plot_delta_criteria(fig.add_subplot(gs[0, 1]), comparison_metrics)
            
            # 3. Model Weights (top right)
            self._plot_model_weights(fig.add_subplot(gs[0, 2]), comparison_metrics)
            
            # 4. Evidence Ratio (bottom left)
            self._plot_evidence_ratio(fig.add_subplot(gs[1, 0]), comparison_metrics)
            
            # 5. Summary Table (bottom middle and right)
            self._create_summary_table(fig.add_subplot(gs[1, 1:]), models, comparison_metrics, test_results)
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save or return
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / filename
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Model comparison plot saved to {save_path}")
                return save_path
            else:
                return fig
                
        except Exception as e:
            logger.error(f"Failed to create model comparison: {e}")
            if self.fig:
                plt.close(self.fig)
            return None
    
    def _calculate_comparison_metrics(self, models: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comparison metrics from models.
        
        Args:
            models: Dictionary of models with their metrics
            
        Returns:
            Dictionary with calculated comparison metrics
        """
        metrics = {}
        
        # Get model names
        model_names = list(models.keys())
        
        # Extract AIC and BIC values
        aic_values = {name: models[name].get('aic', np.inf) for name in model_names}
        bic_values = {name: models[name].get('bic', np.inf) for name in model_names}
        
        # Find best models
        best_aic_model = min(aic_values, key=aic_values.get)
        best_bic_model = min(bic_values, key=bic_values.get)
        
        # Calculate deltas
        metrics['delta_aic'] = {name: aic_values[name] - aic_values[best_aic_model] 
                                for name in model_names}
        metrics['delta_bic'] = {name: bic_values[name] - bic_values[best_bic_model] 
                                for name in model_names}
        
        # Calculate AIC weights
        aic_array = np.array(list(aic_values.values()))
        delta_aic_array = aic_array - np.min(aic_array)
        aic_rel_likelihood = np.exp(-0.5 * delta_aic_array)
        aic_weights = aic_rel_likelihood / np.sum(aic_rel_likelihood)
        metrics['aic_weights'] = {name: weight for name, weight in zip(model_names, aic_weights)}
        
        # Calculate BIC weights
        bic_array = np.array(list(bic_values.values()))
        delta_bic_array = bic_array - np.min(bic_array)
        bic_rel_likelihood = np.exp(-0.5 * delta_bic_array)
        bic_weights = bic_rel_likelihood / np.sum(bic_rel_likelihood)
        metrics['bic_weights'] = {name: weight for name, weight in zip(model_names, bic_weights)}
        
        # Calculate evidence ratios (for first two models if they exist)
        if len(model_names) >= 2:
            model1, model2 = model_names[0], model_names[1]
            metrics['evidence_ratio'] = metrics['aic_weights'][model1] / metrics['aic_weights'][model2] \
                                       if metrics['aic_weights'][model2] > 0 else np.inf
        
        metrics['best_aic_model'] = best_aic_model
        metrics['best_bic_model'] = best_bic_model
        metrics['aic_values'] = aic_values
        metrics['bic_values'] = bic_values
        
        return metrics
    
    def _plot_information_criteria(self, ax, models: Dict, metrics: Dict):
        """Plot information criteria values."""
        model_names = list(models.keys())
        
        # Prepare data
        criteria = ['AIC', 'BIC']
        x = np.arange(len(criteria))
        width = 0.35
        
        # Create bars for each model
        colors = ['coral', 'steelblue', 'green', 'purple'][:len(model_names)]
        
        for i, (name, color) in enumerate(zip(model_names, colors)):
            values = [metrics['aic_values'][name], metrics['bic_values'][name]]
            offset = width * (i - len(model_names)/2 + 0.5)
            bars = ax.bar(x + offset, values, width, label=name.capitalize(), 
                          color=color, alpha=0.8)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}', ha='center', va='bottom', fontsize=9)
        
        # Add indicators for best models
        y_min, y_max = ax.get_ylim()
        for i, criterion in enumerate(criteria):
            best_model = metrics[f'best_{criterion.lower()}_model']
            ax.text(i, y_max * 0.95, f'✓ {best_model.capitalize()}', 
                   ha='center', fontsize=10, fontweight='bold', color='green')
        
        ax.set_ylabel('Information Criterion Value', fontsize=11)
        ax.set_title('Model Selection: Information Criteria', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(criteria)
        ax.legend(loc='upper left')
    
    def _plot_delta_criteria(self, ax, metrics: Dict):
        """Plot delta information criteria."""
        model_names = list(metrics['delta_aic'].keys())
        
        # Prepare data
        x = np.arange(len(model_names))
        width = 0.35
        
        delta_aic = [metrics['delta_aic'][name] for name in model_names]
        delta_bic = [metrics['delta_bic'][name] for name in model_names]
        
        # Create bars
        bars1 = ax.bar(x - width/2, delta_aic, width, label='ΔAIC', color='steelblue', alpha=0.7)
        bars2 = ax.bar(x + width/2, delta_bic, width, label='ΔBIC', color='coral', alpha=0.7)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                val = bar.get_height()
                y_pos = val if val > 0 else 0
                va = 'bottom' if val > 0 else 'top'
                ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                       f'{val:.1f}', ha='center', va=va, fontsize=9)
        
        ax.set_ylabel('Δ Information Criterion', fontsize=11)
        ax.set_title('Model Improvement\n(Δ = 0 is best model)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([name.capitalize() for name in model_names])
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(y=2, color='gray', linestyle='--', alpha=0.5, label='Δ=2 (weak)')
        ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='Δ=10 (strong)')
        ax.legend()
    
    def _plot_model_weights(self, ax, metrics: Dict):
        """Plot model weights."""
        model_names = list(metrics['aic_weights'].keys())
        
        x = np.arange(2)  # AIC and BIC
        width = 0.6
        colors = ['coral', 'steelblue', 'green', 'purple'][:len(model_names)]
        
        # Stack the weights
        bottom_aic = 0
        bottom_bic = 0
        
        for i, (name, color) in enumerate(zip(model_names, colors)):
            aic_weight = metrics['aic_weights'][name]
            bic_weight = metrics['bic_weights'][name]
            
            ax.bar(0, aic_weight, width, bottom=bottom_aic, 
                  label=name.capitalize() if i == 0 else "", color=color, alpha=0.8)
            ax.bar(1, bic_weight, width, bottom=bottom_bic, 
                  label="" if i > 0 else None, color=color, alpha=0.8)
            
            # Add percentage labels
            if aic_weight > 0.01:
                ax.text(0, bottom_aic + aic_weight/2, f'{aic_weight*100:.1f}%',
                       ha='center', va='center', fontsize=9, color='white', fontweight='bold')
            if bic_weight > 0.01:
                ax.text(1, bottom_bic + bic_weight/2, f'{bic_weight*100:.1f}%',
                       ha='center', va='center', fontsize=9, color='white', fontweight='bold')
            
            bottom_aic += aic_weight
            bottom_bic += bic_weight
        
        ax.set_ylabel('Model Weight (Probability)', fontsize=11)
        ax.set_title('Model Weights\n(Probability of being best model)', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['AIC Weights', 'BIC Weights'])
        ax.set_ylim([0, 1])
        ax.legend(loc='upper right')
    
    def _plot_evidence_ratio(self, ax, metrics: Dict):
        """Plot evidence ratio."""
        if 'evidence_ratio' not in metrics:
            ax.text(0.5, 0.5, 'Evidence ratio not available\n(need 2+ models)',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        
        evidence_ratio = metrics['evidence_ratio']
        
        # Calculate log evidence ratio
        if evidence_ratio != np.inf and evidence_ratio > 0:
            log_ratio = np.log10(evidence_ratio) if evidence_ratio >= 1 else -np.log10(1/evidence_ratio)
        else:
            log_ratio = 10 if evidence_ratio == np.inf else -10
        
        # Create horizontal bar
        color = 'green' if log_ratio > 0 else 'red'
        ax.barh([0], [log_ratio], color=color, alpha=0.7, height=0.4)
        
        ax.set_xlim([-10, 10])
        ax.set_xlabel('Log₁₀(Evidence Ratio)', fontsize=11)
        
        # Get model names for title
        model_names = list(metrics['aic_weights'].keys())
        if len(model_names) >= 2:
            ax.set_title(f'Evidence Ratio: {evidence_ratio:.0f}:1\n({model_names[0].capitalize()} vs {model_names[1].capitalize()})',
                        fontsize=12, fontweight='bold')
        
        ax.set_yticks([])
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        
        # Add reference lines
        for x_val, label in [(1, '10:1'), (2, '100:1'), (3, '1000:1')]:
            ax.axvline(x=x_val, color='gray', linestyle='--', alpha=0.3)
            ax.axvline(x=-x_val, color='gray', linestyle='--', alpha=0.3)
            ax.text(x_val, 0.5, label, ha='center', fontsize=8, color='gray')
            ax.text(-x_val, 0.5, f'1:{10**x_val}', ha='center', fontsize=8, color='gray')
        
        # Add interpretation
        if abs(log_ratio) >= 3:
            strength = "Decisive"
        elif abs(log_ratio) >= 2:
            strength = "Very strong"
        elif abs(log_ratio) >= 1:
            strength = "Strong"
        elif abs(log_ratio) >= 0.5:
            strength = "Moderate"
        else:
            strength = "Weak"
        
        winner = model_names[0] if log_ratio > 0 else model_names[1]
        ax.text(0.5, -0.3, f'{strength} evidence for {winner.capitalize()} model',
               transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold')
    
    def _create_summary_table(self, ax, models: Dict, metrics: Dict, test_results: Optional[Dict]):
        """Create summary table."""
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        headers = ['Metric'] + [name.capitalize() for name in models.keys()] + ['Interpretation']
        table_data = []
        
        # Add rows for each metric
        model_names = list(models.keys())
        
        # Number of parameters
        row = ['N Parameters']
        for name in model_names:
            row.append(f"{models[name].get('n_parameters', 'N/A')}")
        if len(model_names) == 2:
            diff = models[model_names[1]].get('n_parameters', 0) - models[model_names[0]].get('n_parameters', 0)
            row.append(f"Difference: {diff}")
        else:
            row.append("")
        table_data.append(row)
        
        # Theta if available
        if any('theta' in models[name] for name in model_names):
            row = ['Theta (θ)']
            for name in model_names:
                row.append(f"{models[name].get('theta', 'N/A'):.2f}" if 'theta' in models[name] else 'N/A')
            row.append('Higher θ = less overdispersion')
            table_data.append(row)
        
        # AIC
        row = ['AIC']
        for name in model_names:
            row.append(f"{metrics['aic_values'][name]:.0f}")
        best = metrics['best_aic_model']
        row.append(f"{best.capitalize()} is best")
        table_data.append(row)
        
        # BIC
        row = ['BIC']
        for name in model_names:
            row.append(f"{metrics['bic_values'][name]:.0f}")
        best = metrics['best_bic_model']
        row.append(f"{best.capitalize()} is best")
        table_data.append(row)
        
        # Weights
        row = ['AIC Weight']
        for name in model_names:
            row.append(f"{metrics['aic_weights'][name]:.1%}")
        row.append('Model probability')
        table_data.append(row)
        
        # Add test results if provided
        if test_results:
            if 'lrt' in test_results or 'likelihood_ratio_test' in test_results:
                lrt = test_results.get('lrt', test_results.get('likelihood_ratio_test', {}))
                row = ['LR Test']
                row.append('-')  # First model column
                if len(model_names) > 1:
                    row.append(f"χ²={lrt.get('lr_statistic', lrt.get('statistic', 0)):.0f}")
                    for _ in range(len(model_names) - 2):
                        row.append('')
                p_val = lrt.get('p_value', 1)
                row.append(f"p<0.001" if p_val < 0.001 else f"p={p_val:.4f}")
                table_data.append(row)
            
            if 'score' in test_results or 'score_test' in test_results:
                score = test_results.get('score', test_results.get('score_test', {}))
                row = ['Score Test']
                row.append('-')
                if len(model_names) > 1:
                    row.append(f"χ²={score.get('statistic', 0):.0f}")
                    for _ in range(len(model_names) - 2):
                        row.append('')
                p_val = score.get('p_value', 1)
                row.append(f"p<0.001" if p_val < 0.001 else f"p={p_val:.4f}")
                table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        colWidths=[0.15] * (len(model_names) + 1) + [0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight best model rows
        best_aic_idx = model_names.index(metrics['best_aic_model']) + 1
        best_bic_idx = model_names.index(metrics['best_bic_model']) + 1
        
        # AIC row
        for j in range(len(headers)):
            table[(3, j)].set_facecolor('#E8F5E9' if j == best_aic_idx else 'white')
        
        # BIC row
        for j in range(len(headers)):
            table[(4, j)].set_facecolor('#E8F5E9' if j == best_bic_idx else 'white')
        
        ax.set_title('Model Comparison Summary', fontsize=14, fontweight='bold', pad=20)