"""Model comparison using information criteria (AIC, BIC)."""

import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


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