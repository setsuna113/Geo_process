"""Model comparison using information criteria (AIC, BIC)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)

# Set style for publication-quality plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


class ModelComparison:
    """Model comparison using information criteria.
    
    Implements AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion)
    for model selection and comparison.
    """
    
    @staticmethod
    def compare_models(
        models: Union[List[Any], Dict[str, Any]],
        model_names: Optional[List[str]] = None,
        criteria: List[str] = ['aic', 'bic']
    ) -> Dict[str, Any]:
        """Compare multiple models using information criteria.
        
        Args:
            models: List or dict of model objects
            model_names: Optional names for models (if models is a list)
            criteria: List of criteria to use ('aic', 'bic', 'deviance')
            
        Returns:
            Dictionary with comparison results and best models
        """
        # Handle dict input
        if isinstance(models, dict):
            model_names = list(models.keys())
            models = list(models.values())
        elif model_names is None:
            model_names = [f"Model_{i}" for i in range(len(models))]
        
        if len(models) < 2:
            raise ValueError("Need at least 2 models for comparison")
        
        if len(model_names) != len(models):
            raise ValueError("Number of model names must match number of models")
        
        # Initialize results
        results = {
            'n_models': len(models),
            'model_names': model_names,
            'criteria_used': criteria,
            'model_stats': {},
            'rankings': {},
            'best_models': {}
        }
        
        # Collect statistics for each model
        for name, model in zip(model_names, models):
            model_stat = {}
            
            # Get AIC
            if 'aic' in criteria:
                if hasattr(model, 'get_aic'):
                    model_stat['aic'] = model.get_aic()
                elif hasattr(model, 'aic'):
                    model_stat['aic'] = model.aic
                else:
                    logger.warning(f"Model {name} doesn't have AIC")
            
            # Get BIC
            if 'bic' in criteria:
                if hasattr(model, 'get_bic'):
                    model_stat['bic'] = model.get_bic()
                elif hasattr(model, 'bic'):
                    model_stat['bic'] = model.bic
                else:
                    logger.warning(f"Model {name} doesn't have BIC")
            
            # Get deviance
            if 'deviance' in criteria:
                if hasattr(model, 'get_deviance'):
                    model_stat['deviance'] = model.get_deviance()
                elif hasattr(model, 'deviance'):
                    model_stat['deviance'] = model.deviance
                else:
                    logger.warning(f"Model {name} doesn't have deviance")
            
            # Get number of parameters if available
            if hasattr(model, 'get_num_parameters'):
                model_stat['n_parameters'] = model.get_num_parameters()
            elif hasattr(model, 'n_parameters'):
                model_stat['n_parameters'] = model.n_parameters
            
            results['model_stats'][name] = model_stat
        
        # Rank models and compute differences for each criterion
        for criterion in criteria:
            values = []
            valid_models = []
            
            for name in model_names:
                if criterion in results['model_stats'][name]:
                    values.append(results['model_stats'][name][criterion])
                    valid_models.append(name)
            
            if len(values) > 0:
                values = np.array(values)
                
                # Lower is better for AIC, BIC, deviance
                sorted_idx = np.argsort(values)
                
                ranking = {
                    'ranked_models': [valid_models[i] for i in sorted_idx],
                    'values': [float(values[i]) for i in sorted_idx],
                    'best_model': valid_models[sorted_idx[0]],
                    'best_value': float(values[sorted_idx[0]])
                }
                
                # Compute differences from best model
                if len(values) > 1:
                    best_val = values[sorted_idx[0]]
                    differences = values - best_val
                    ranking['delta_from_best'] = {
                        model: float(diff) 
                        for model, diff in zip(valid_models, differences)
                    }
                    
                    # Interpret differences for AIC/BIC
                    if criterion in ['aic', 'bic']:
                        ranking['interpretation'] = ModelComparison._interpret_ic_differences(
                            ranking['delta_from_best']
                        )
                
                results['rankings'][criterion] = ranking
                results['best_models'][criterion] = ranking['best_model']
        
        # Add consensus best model if multiple criteria agree
        if len(results['best_models']) > 1:
            best_counts = {}
            for best_model in results['best_models'].values():
                best_counts[best_model] = best_counts.get(best_model, 0) + 1
            
            max_count = max(best_counts.values())
            consensus_models = [m for m, c in best_counts.items() if c == max_count]
            
            if len(consensus_models) == 1:
                results['consensus_best'] = consensus_models[0]
            else:
                results['consensus_best'] = None
                results['consensus_note'] = f"No clear consensus: {consensus_models} tied"
        
        return results
    
    @staticmethod
    def _interpret_ic_differences(delta_dict: Dict[str, float]) -> Dict[str, str]:
        """Interpret AIC/BIC differences according to Burnham & Anderson guidelines.
        
        Delta AIC/BIC interpretation:
        - 0-2: Substantial support (models are competitive)
        - 2-4: Some support
        - 4-7: Considerably less support
        - 7-10: Very little support
        - >10: Essentially no support
        
        Args:
            delta_dict: Dictionary of model names to delta values
            
        Returns:
            Dictionary with interpretation for each model
        """
        interpretations = {}
        
        for model, delta in delta_dict.items():
            if delta < 2:
                interpretations[model] = "Strong support (competitive model)"
            elif delta < 4:
                interpretations[model] = "Some support"
            elif delta < 7:
                interpretations[model] = "Considerably less support"
            elif delta < 10:
                interpretations[model] = "Very little support"
            else:
                interpretations[model] = "Essentially no support"
        
        return interpretations
    
    @staticmethod
    def compute_aic_weights(
        models: Union[List[Any], Dict[str, Any]],
        model_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compute Akaike weights for model averaging.
        
        Akaike weights represent the probability that each model is the best model
        given the data and the set of candidate models.
        
        Args:
            models: List or dict of model objects
            model_names: Optional names for models
            
        Returns:
            Dictionary with AIC weights and related statistics
        """
        # Get AIC values
        comparison = ModelComparison.compare_models(
            models, model_names, criteria=['aic']
        )
        
        if 'aic' not in comparison['rankings']:
            raise ValueError("Models must have AIC values for weight computation")
        
        aic_ranking = comparison['rankings']['aic']
        model_names = aic_ranking['ranked_models']
        aic_values = np.array(aic_ranking['values'])
        
        # Compute AIC differences from minimum
        min_aic = np.min(aic_values)
        delta_aic = aic_values - min_aic
        
        # Compute relative likelihoods
        relative_likelihood = np.exp(-0.5 * delta_aic)
        
        # Compute Akaike weights
        weights = relative_likelihood / np.sum(relative_likelihood)
        
        # Evidence ratios relative to best model
        evidence_ratios = weights[0] / weights
        
        results = {
            'model_names': model_names,
            'aic_values': aic_values.tolist(),
            'delta_aic': delta_aic.tolist(),
            'akaike_weights': weights.tolist(),
            'relative_likelihood': relative_likelihood.tolist(),
            'evidence_ratios': evidence_ratios.tolist(),
            'weight_rankings': sorted(
                zip(model_names, weights),
                key=lambda x: x[1],
                reverse=True
            )
        }
        
        # Add interpretation
        weight_dict = dict(zip(model_names, weights))
        results['interpretation'] = {}
        for model, weight in weight_dict.items():
            if weight > 0.5:
                results['interpretation'][model] = f"Strong evidence (weight={weight:.3f})"
            elif weight > 0.25:
                results['interpretation'][model] = f"Moderate evidence (weight={weight:.3f})"
            elif weight > 0.1:
                results['interpretation'][model] = f"Some evidence (weight={weight:.3f})"
            else:
                results['interpretation'][model] = f"Little evidence (weight={weight:.3f})"
        
        return results
    
    @staticmethod
    def compare_nested_models(
        null_model: Any,
        full_model: Any
    ) -> Dict[str, Any]:
        """Compare nested models using multiple criteria.
        
        Args:
            null_model: Simpler (nested) model
            full_model: More complex model
            
        Returns:
            Dictionary with comparison results
        """
        results = {
            'comparison_type': 'nested_models'
        }
        
        # Get model statistics
        for model, name in [(null_model, 'null'), (full_model, 'full')]:
            model_stats = {}
            
            if hasattr(model, 'get_aic'):
                model_stats['aic'] = model.get_aic()
            if hasattr(model, 'get_bic'):
                model_stats['bic'] = model.get_bic()
            if hasattr(model, 'get_deviance'):
                model_stats['deviance'] = model.get_deviance()
            if hasattr(model, 'get_num_parameters'):
                model_stats['n_parameters'] = model.get_num_parameters()
            
            results[f'{name}_model'] = model_stats
        
        # Compute differences
        differences = {}
        for metric in ['aic', 'bic', 'deviance']:
            if metric in results['null_model'] and metric in results['full_model']:
                differences[f'delta_{metric}'] = (
                    results['null_model'][metric] - results['full_model'][metric]
                )
        
        results['differences'] = differences
        
        # Determine which model is preferred
        preferences = {}
        
        # For AIC and BIC, lower is better
        if 'delta_aic' in differences:
            if differences['delta_aic'] > 2:
                preferences['aic'] = 'full_model'
            elif differences['delta_aic'] < -2:
                preferences['aic'] = 'null_model'
            else:
                preferences['aic'] = 'no_clear_preference'
        
        if 'delta_bic' in differences:
            if differences['delta_bic'] > 2:
                preferences['bic'] = 'full_model'
            elif differences['delta_bic'] < -2:
                preferences['bic'] = 'null_model'
            else:
                preferences['bic'] = 'no_clear_preference'
        
        results['preferences'] = preferences
        
        # Add interpretation
        if preferences:
            if all(p == 'full_model' for p in preferences.values()):
                results['conclusion'] = "Full model is preferred by all criteria"
            elif all(p == 'null_model' for p in preferences.values()):
                results['conclusion'] = "Null model is preferred by all criteria (simpler is better)"
            else:
                results['conclusion'] = "Mixed evidence - criteria disagree on best model"
        
        return results
    
    @staticmethod
    def generate_publication_report(
        comparison_results: Dict[str, Any],
        study_description: str = "ecological model comparison",
        hypothesis: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate publication-ready report for model comparison.
        
        Following ecology journal standards for information criteria reporting.
        
        Args:
            comparison_results: Results from compare_models or compute_aic_weights
            study_description: Brief description of the study
            hypothesis: Optional hypothesis being tested
            
        Returns:
            Dictionary with formatted text, tables, and latex code
        """
        # Check if this is from compute_aic_weights
        if 'akaike_weights' in comparison_results:
            return ModelComparison._report_aic_weights(comparison_results, study_description)
        
        # Standard model comparison report
        model_stats = comparison_results.get('model_stats', {})
        rankings = comparison_results.get('rankings', {})
        best_models = comparison_results.get('best_models', {})
        
        # Create model comparison table
        rows = []
        for model_name, stats in model_stats.items():
            row = {'Model': model_name}
            
            # Add number of parameters
            if 'n_parameters' in stats:
                row['K'] = stats['n_parameters']
            
            # Add AIC if available
            if 'aic' in stats:
                row['AIC'] = f"{stats['aic']:.1f}"
                if 'aic' in rankings:
                    delta_aic = rankings['aic']['delta_from_best'].get(model_name, 0)
                    row['ΔAIC'] = f"{delta_aic:.1f}"
            
            # Add BIC if available
            if 'bic' in stats:
                row['BIC'] = f"{stats['bic']:.1f}"
                if 'bic' in rankings:
                    delta_bic = rankings['bic']['delta_from_best'].get(model_name, 0)
                    row['ΔBIC'] = f"{delta_bic:.1f}"
            
            # Add deviance if available
            if 'deviance' in stats:
                row['Deviance'] = f"{stats['deviance']:.1f}"
            
            rows.append(row)
        
        # Sort by AIC if available
        if 'aic' in rankings:
            model_order = rankings['aic']['ranked_models']
            rows = sorted(rows, key=lambda x: model_order.index(x['Model']))
        
        results_table = pd.DataFrame(rows)
        
        # Generate main text
        main_text = f"Model selection was performed for {study_description}. "
        
        if 'aic' in best_models:
            best_aic = best_models['aic']
            main_text += f"Based on AIC, {best_aic} was selected as the best model. "
            
            # Report models within 2 AIC units
            if 'aic' in rankings:
                competitive_models = []
                for model, delta in rankings['aic']['delta_from_best'].items():
                    if delta < 2 and model != best_aic:
                        competitive_models.append(model)
                
                if competitive_models:
                    main_text += f"Models {', '.join(competitive_models)} were also competitive (ΔAIC < 2). "
        
        if 'bic' in best_models:
            best_bic = best_models['bic']
            if best_bic != best_models.get('aic'):
                main_text += f"BIC selected {best_bic}, suggesting a preference for parsimony. "
        
        if 'consensus_best' in comparison_results:
            if comparison_results['consensus_best']:
                main_text += f"All criteria agreed on {comparison_results['consensus_best']} as the best model."
            else:
                main_text += comparison_results.get('consensus_note', '')
        
        # Methods text
        methods_text = (
            "Model selection was conducted using information-theoretic approaches. "
            "Models were compared using the Akaike Information Criterion (AIC) and "
            "Bayesian Information Criterion (BIC). Delta values (Δ) represent the "
            "difference from the best model. Following Burnham & Anderson (2002), "
            "models with ΔAIC < 2 were considered to have substantial support, "
            "ΔAIC 4-7 considerably less support, and ΔAIC > 10 essentially no support. "
            "Lower values of AIC and BIC indicate better model fit after accounting "
            "for model complexity."
        )
        
        # Supplementary information
        supplementary_info = {
            'interpretation_guidelines': {
                'ΔAIC 0-2': 'Substantial support (competitive model)',
                'ΔAIC 2-4': 'Some support',
                'ΔAIC 4-7': 'Considerably less support',
                'ΔAIC 7-10': 'Very little support',
                'ΔAIC > 10': 'Essentially no support'
            }
        }
        
        return {
            'main_text': main_text,
            'methods_text': methods_text,
            'results_table': results_table,
            'supplementary_info': supplementary_info,
            'latex_table': ModelComparison._generate_latex_table(results_table),
            'best_models': best_models
        }
    
    @staticmethod
    def _report_aic_weights(weights_results: Dict[str, Any], 
                            study_description: str) -> Dict[str, Any]:
        """Generate report specifically for AIC weights analysis."""
        model_names = weights_results['model_names']
        aic_values = weights_results['aic_values']
        delta_aic = weights_results['delta_aic']
        weights = weights_results['akaike_weights']
        evidence_ratios = weights_results['evidence_ratios']
        
        # Create detailed table
        results_table = pd.DataFrame({
            'Model': model_names,
            'AIC': [f"{v:.1f}" for v in aic_values],
            'ΔAIC': [f"{d:.1f}" for d in delta_aic],
            'Weight': [f"{w:.3f}" for w in weights],
            'Evidence Ratio': [f"{er:.1f}" if er < 100 else f">100" 
                              for er in evidence_ratios]
        })
        
        # Calculate cumulative weights
        cumulative_weights = np.cumsum(weights)
        results_table['Cumulative Weight'] = [f"{cw:.3f}" for cw in cumulative_weights]
        
        # Main text
        best_model = model_names[0]
        best_weight = weights[0]
        
        main_text = (
            f"Model selection for {study_description} was performed using "
            f"Akaike weights. The {best_model} had the highest weight "
            f"(w = {best_weight:.3f}), representing a {best_weight:.1%} probability "
            f"of being the best model given the candidate set. "
        )
        
        # Report confidence set (95% cumulative weight)
        confidence_set = []
        for i, cw in enumerate(cumulative_weights):
            confidence_set.append(model_names[i])
            if cw >= 0.95:
                main_text += (
                    f"The 95% confidence set includes {len(confidence_set)} models: "
                    f"{', '.join(confidence_set)}. "
                )
                break
        
        # Evidence ratios interpretation
        if len(model_names) > 1 and evidence_ratios[1] > 2.7:
            main_text += (
                f"The evidence ratio of {evidence_ratios[1]:.1f} indicates "
                f"the best model is {evidence_ratios[1]:.1f} times more likely "
                f"than the second-best model."
            )
        
        return {
            'main_text': main_text,
            'results_table': results_table,
            'latex_table': ModelComparison._generate_latex_table(results_table),
            'best_model': best_model,
            'confidence_set': confidence_set
        }
    
    @staticmethod
    def visualize_model_comparison(
        comparison_results: Dict[str, Any],
        figsize: Tuple[int, int] = (15, 5)
    ) -> plt.Figure:
        """Create publication-quality visualization of model comparison.
        
        Args:
            comparison_results: Results from compare_models or compute_aic_weights
            figsize: Figure size
            
        Returns:
            matplotlib figure suitable for publication
        """
        # Check what type of results we have
        if 'akaike_weights' in comparison_results:
            return ModelComparison._visualize_aic_weights(comparison_results, figsize)
        
        # Standard model comparison visualization
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        model_stats = comparison_results.get('model_stats', {})
        rankings = comparison_results.get('rankings', {})
        
        # Panel 1: AIC comparison
        ax1 = axes[0]
        if 'aic' in rankings:
            models = rankings['aic']['ranked_models']
            values = rankings['aic']['values']
            deltas = [rankings['aic']['delta_from_best'][m] for m in models]
            
            # Create bar plot
            x_pos = np.arange(len(models))
            bars = ax1.bar(x_pos, deltas, color=['green' if d < 2 else 'yellow' if d < 4 
                                                 else 'orange' if d < 7 else 'red' 
                                                 for d in deltas], alpha=0.7)
            
            # Add value labels
            for bar, val in zip(bars, deltas):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9)
            
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(models, rotation=45, ha='right')
            ax1.set_ylabel('ΔAIC', fontsize=11)
            ax1.set_title('AIC Differences from Best Model', fontsize=12, fontweight='bold')
            ax1.axhline(2, color='gray', linestyle='--', alpha=0.5, label='Substantial support')
            ax1.axhline(7, color='gray', linestyle=':', alpha=0.5, label='Little support')
            ax1.legend(loc='upper right', fontsize=9)
            ax1.grid(True, alpha=0.3, axis='y')
        
        # Panel 2: BIC comparison
        ax2 = axes[1]
        if 'bic' in rankings:
            models = rankings['bic']['ranked_models']
            values = rankings['bic']['values']
            deltas = [rankings['bic']['delta_from_best'][m] for m in models]
            
            x_pos = np.arange(len(models))
            bars = ax2.bar(x_pos, deltas, color=['green' if d < 2 else 'yellow' if d < 4 
                                                 else 'orange' if d < 7 else 'red' 
                                                 for d in deltas], alpha=0.7)
            
            for bar, val in zip(bars, deltas):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9)
            
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(models, rotation=45, ha='right')
            ax2.set_ylabel('ΔBIC', fontsize=11)
            ax2.set_title('BIC Differences from Best Model', fontsize=12, fontweight='bold')
            ax2.axhline(2, color='gray', linestyle='--', alpha=0.5)
            ax2.axhline(7, color='gray', linestyle=':', alpha=0.5)
            ax2.grid(True, alpha=0.3, axis='y')
        
        # Panel 3: Model complexity vs fit
        ax3 = axes[2]
        model_names = list(model_stats.keys())
        n_params = []
        aic_vals = []
        
        for name in model_names:
            if 'n_parameters' in model_stats[name] and 'aic' in model_stats[name]:
                n_params.append(model_stats[name]['n_parameters'])
                aic_vals.append(model_stats[name]['aic'])
        
        if n_params and aic_vals:
            # Color by model rank
            if 'aic' in rankings:
                colors = ['green' if name == rankings['aic']['best_model'] else 'blue' 
                         for name in model_names]
            else:
                colors = 'blue'
            
            ax3.scatter(n_params, aic_vals, s=100, c=colors, alpha=0.7, edgecolors='black')
            
            # Add labels
            for i, name in enumerate(model_names[:len(n_params)]):
                ax3.annotate(name, (n_params[i], aic_vals[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
            
            ax3.set_xlabel('Number of Parameters (K)', fontsize=11)
            ax3.set_ylabel('AIC', fontsize=11)
            ax3.set_title('Model Complexity vs Fit', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
        
        plt.suptitle('Model Comparison Results', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def _visualize_aic_weights(
        weights_results: Dict[str, Any],
        figsize: Tuple[int, int]
    ) -> plt.Figure:
        """Visualize AIC weights analysis."""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        model_names = weights_results['model_names']
        weights = weights_results['akaike_weights']
        delta_aic = weights_results['delta_aic']
        evidence_ratios = weights_results['evidence_ratios']
        
        # Panel 1: AIC weights bar plot
        ax1 = axes[0]
        x_pos = np.arange(len(model_names))
        bars = ax1.bar(x_pos, weights, color='steelblue', alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar, weight in zip(bars, weights):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{weight:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.set_ylabel('Akaike Weight', fontsize=11)
        ax1.set_title('Model Weights', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, min(1, max(weights) * 1.2))
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Panel 2: Delta AIC on number line
        ax2 = axes[1]
        y_pos = 0
        
        # Draw number line
        max_delta = max(delta_aic) if max(delta_aic) > 10 else 10
        ax2.plot([0, max_delta], [y_pos, y_pos], 'k-', linewidth=2)
        
        # Add reference lines
        for threshold, label, color in [(2, 'Substantial', 'green'),
                                        (4, 'Some', 'yellow'),
                                        (7, 'Little', 'orange'),
                                        (10, 'Very little', 'red')]:
            if threshold <= max_delta:
                ax2.axvline(threshold, color=color, linestyle='--', alpha=0.3)
                ax2.text(threshold, 0.5, label, rotation=90, 
                        va='bottom', ha='right', fontsize=8, color=color)
        
        # Plot models on number line
        for i, (name, delta) in enumerate(zip(model_names, delta_aic)):
            color = 'green' if delta < 2 else 'yellow' if delta < 4 else 'orange' if delta < 7 else 'red'
            ax2.plot(delta, y_pos, 'o', markersize=10, color=color, 
                    markeredgecolor='black', markeredgewidth=1)
            
            # Add model label
            ax2.text(delta, -0.15 - (i % 2) * 0.15, name,
                    ha='center', fontsize=8, rotation=45)
        
        ax2.set_xlim(-0.5, max_delta + 0.5)
        ax2.set_ylim(-0.5, 0.8)
        ax2.set_xlabel('ΔAIC', fontsize=11)
        ax2.set_yticks([])
        ax2.set_title('Model Support on ΔAIC Scale', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Panel 3: Evidence ratios
        ax3 = axes[2]
        
        # Only show evidence ratios > 1 (compared to best model)
        er_to_plot = evidence_ratios[1:] if len(evidence_ratios) > 1 else []
        models_to_plot = model_names[1:] if len(model_names) > 1 else []
        
        if er_to_plot:
            x_pos = np.arange(len(models_to_plot))
            bars = ax3.barh(x_pos, er_to_plot, color='coral', alpha=0.7)
            
            # Add value labels
            for bar, er in zip(bars, er_to_plot):
                width = bar.get_width()
                label = f'{er:.1f}' if er < 100 else '>100'
                ax3.text(width, bar.get_y() + bar.get_height()/2.,
                        label, ha='left', va='center', fontsize=9)
            
            ax3.set_yticks(x_pos)
            ax3.set_yticklabels(models_to_plot)
            ax3.set_xlabel('Evidence Ratio vs Best Model', fontsize=11)
            ax3.set_title('Relative Evidence', fontsize=12, fontweight='bold')
            ax3.axvline(1, color='black', linestyle='-', linewidth=1)
            ax3.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('AIC-based Model Selection', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def _generate_latex_table(df: pd.DataFrame) -> str:
        """Generate LaTeX table code for publication.
        
        Args:
            df: DataFrame with results
            
        Returns:
            LaTeX table code suitable for ecology journals
        """
        # Determine number of columns
        n_cols = len(df.columns)
        col_align = 'l' + 'c' * (n_cols - 1)
        
        latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Model comparison using information criteria}}
\\label{{tab:model_comparison}}
\\begin{{tabular}}{{{col_align}}}
\\toprule
"""
        
        # Add headers
        headers = ' & '.join(df.columns) + ' \\\\'
        latex += headers + '\n\\midrule\n'
        
        # Add data rows
        for _, row in df.iterrows():
            row_str = ' & '.join(str(val) for val in row.values) + ' \\\\'
            latex += row_str + '\n'
        
        # Add footer
        latex += """\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\\small
\\item Note: Models ranked by information criteria. $\\Delta$ values show difference from best model.
\\item AIC weights represent probability each model is best given the candidate set.
\\item Following Burnham \\& Anderson (2002), $\\Delta$AIC < 2 indicates substantial support.
\\end{tablenotes}
\\end{table}"""
        
        return latex