"""Marginal effects and variable importance for GLM models.

Implements interpretation methods from regression_analysis_decisions.md lines 536-576.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
import h2o

logger = logging.getLogger(__name__)


class MarginalEffects:
    """Calculate marginal effects for GLM models.
    
    Marginal effects show how much the outcome changes for a unit change in predictor.
    """
    
    def __init__(self, model: Any):
        """Initialize with fitted model.
        
        Args:
            model: Fitted GLM model (H2O or similar)
        """
        self.model = model
        
    def compute_marginal_effects(self, 
                                data: Union[h2o.H2OFrame, pd.DataFrame],
                                variables: Optional[List[str]] = None,
                                at_values: Optional[Dict[str, List[float]]] = None) -> Dict[str, Any]:
        """Compute marginal effects for specified variables.
        
        Equivalent to margins() in R. Shows how outcome changes with predictors.
        
        Args:
            data: Data to compute effects on
            variables: Variables to compute effects for (default: all)
            at_values: Values to compute effects at (e.g., {'bio4_std': [-1, 0, 1]})
            
        Returns:
            Dictionary with marginal effects for each variable
        """
        if variables is None:
            # Get all predictor variables
            if hasattr(self.model, 'get_feature_names'):
                variables = self.model.get_feature_names()
            else:
                variables = list(data.columns)
                if 'fungi_richness' in variables:
                    variables.remove('fungi_richness')
        
        marginal_effects = {}
        
        for var in variables:
            if at_values and var in at_values:
                # Compute at specified values
                effects_at = []
                for value in at_values[var]:
                    effect = self._compute_effect_at_value(data, var, value)
                    effects_at.append({
                        'value': value,
                        'marginal_effect': effect['effect'],
                        'se': effect.get('se', None)
                    })
                marginal_effects[var] = effects_at
            else:
                # Compute at mean
                effect = self._compute_average_marginal_effect(data, var)
                marginal_effects[var] = effect
        
        return {
            'marginal_effects': marginal_effects,
            'interpretation': self._interpret_effects(marginal_effects)
        }
    
    def _compute_average_marginal_effect(self, data, variable: str) -> Dict[str, float]:
        """Compute average marginal effect for a variable.
        
        For Negative Binomial GLM with log link:
        - Marginal effect = ∂E[Y|X]/∂X_j = β_j * E[Y|X]
        - At mean: ME = β_j * mean(predicted values)
        """
        # Get coefficient
        if hasattr(self.model, 'get_coefficient'):
            coef_info = self.model.get_coefficient(variable)
            coef = coef_info.get('coefficient', 0)
            se = coef_info.get('std_error', None)
        elif hasattr(self.model, 'coef'):
            # For H2O models
            coef = self.model.coef().get(variable, 0)
            se = self.model.std_coef_names().get(variable, None) if hasattr(self.model, 'std_coef_names') else None
        else:
            coef = 0
            se = None
        
        # For NB GLM with log link, compute proper marginal effect
        if hasattr(self.model, 'predict'):
            # Get predictions on response scale (counts)
            predictions = self.model.predict(data)
            if hasattr(predictions, 'as_data_frame'):
                predictions = predictions.as_data_frame().values.flatten()
            mean_prediction = np.mean(predictions)
            
            # Marginal effect = β * mean(Y_hat)
            marginal_effect = coef * mean_prediction
            
            # Standard error via delta method: SE(ME) = |mean(Y_hat)| * SE(β)
            se_marginal = abs(mean_prediction) * se if se is not None else None
        else:
            # Fallback to multiplicative interpretation
            marginal_effect = np.exp(coef) - 1  # Percentage change
            se_marginal = se
        
        return {
            'effect': marginal_effect,
            'se': se_marginal,
            'coefficient': coef,
            'multiplicative_effect': np.exp(coef),
            'interpretation': f"1-unit increase in {variable} → {marginal_effect:.3f} change in fungi count (multiplicative: {np.exp(coef):.3f}x)"
        }
    
    def _compute_effect_at_value(self, data, variable: str, value: float) -> Dict[str, float]:
        """Compute marginal effect at specific value of variable.
        
        For NB GLM: ME(x) = β * E[Y|X=x]
        """
        # Create data with variable set to specific value
        data_at_value = data.copy()
        if hasattr(data_at_value, 'as_data_frame'):
            data_at_value = data_at_value.as_data_frame()
        data_at_value[variable] = value
        
        # Get coefficient
        if hasattr(self.model, 'coef'):
            coef = self.model.coef().get(variable, 0)
            se = None  # Would need more info for SE at specific value
        else:
            coef = 0
            se = None
        
        # Get prediction at this value
        if hasattr(self.model, 'predict'):
            predictions = self.model.predict(data_at_value)
            if hasattr(predictions, 'as_data_frame'):
                predictions = predictions.as_data_frame().values.flatten()
            mean_prediction = np.mean(predictions)
            
            # Marginal effect at this value
            effect = coef * mean_prediction
        else:
            effect = coef
        
        return {
            'effect': effect,
            'se': se,
            'at_value': value
        }
    
    def _interpret_effects(self, effects: Dict) -> str:
        """Create interpretation of marginal effects."""
        interpretations = []
        for var, effect in effects.items():
            if isinstance(effect, dict):
                interpretations.append(effect.get('interpretation', ''))
            elif isinstance(effect, list):
                # Multiple values
                interpretations.append(f"{var} has context-dependent effects")
        return "; ".join(interpretations)


class VariableImportance:
    """Calculate variable importance for GLM models.
    
    Implements hierarchical partitioning approach for variance decomposition.
    """
    
    def __init__(self, model: Any):
        """Initialize with fitted model.
        
        Args:
            model: Fitted GLM model
        """
        self.model = model
        
    def compute_importance(self, method: str = 'deviance') -> Dict[str, Any]:
        """Compute variable importance.
        
        Args:
            method: Method to use ('deviance', 'coefficients', 'wald')
            
        Returns:
            Dictionary with importance scores
        """
        if method == 'deviance':
            return self._deviance_based_importance()
        elif method == 'coefficients':
            return self._coefficient_based_importance()
        elif method == 'wald':
            return self._wald_based_importance()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _deviance_based_importance(self) -> Dict[str, Any]:
        """Compute importance based on deviance explained.
        
        This approximates hier.part() from R by looking at deviance reduction.
        """
        importance = {}
        
        # Get model deviance info
        if hasattr(self.model, 'null_deviance'):
            null_deviance = self.model.null_deviance()
            residual_deviance = self.model.residual_deviance()
            
            # Total deviance explained
            total_explained = null_deviance - residual_deviance
            
            # Get coefficient contributions (simplified)
            if hasattr(self.model, 'get_all_coefficients'):
                coeffs = self.model.get_all_coefficients()
                
                # Normalize by sum of absolute values
                coef_sum = sum(abs(v) for v in coeffs.values() if v != 0)
                
                for var, coef in coeffs.items():
                    if var != 'Intercept':
                        # Approximate importance
                        importance[var] = {
                            'deviance_explained': abs(coef) / coef_sum * total_explained,
                            'percent': abs(coef) / coef_sum * 100
                        }
        
        return {
            'importance': importance,
            'method': 'deviance_based',
            'top_variables': self._get_top_variables(importance, n=5)
        }
    
    def _coefficient_based_importance(self) -> Dict[str, Any]:
        """Compute importance based on standardized coefficients."""
        importance = {}
        
        if hasattr(self.model, 'get_all_coefficients'):
            coeffs = self.model.get_all_coefficients()
            std_errors = self.model.get_standard_errors() if hasattr(self.model, 'get_standard_errors') else {}
            
            # Calculate t-statistics as importance measure
            for var, coef in coeffs.items():
                if var != 'Intercept':
                    se = std_errors.get(var, 1)
                    t_stat = abs(coef / se) if se > 0 else 0
                    importance[var] = {
                        'coefficient': coef,
                        't_statistic': t_stat,
                        'importance_score': t_stat ** 2  # Similar to Wald statistic
                    }
        
        # Normalize importance scores
        total_importance = sum(v['importance_score'] for v in importance.values())
        for var in importance:
            importance[var]['percent'] = importance[var]['importance_score'] / total_importance * 100
        
        return {
            'importance': importance,
            'method': 'coefficient_based',
            'top_variables': self._get_top_variables(importance, n=5)
        }
    
    def _wald_based_importance(self) -> Dict[str, Any]:
        """Compute importance based on Wald statistics."""
        # Similar to coefficient-based but uses Wald chi-square
        return self._coefficient_based_importance()  # Simplified for now
    
    def _get_top_variables(self, importance: Dict, n: int = 5) -> List[Tuple[str, float]]:
        """Get top n most important variables."""
        if not importance:
            return []
        
        # Sort by importance score or percent
        sorted_vars = sorted(
            importance.items(),
            key=lambda x: x[1].get('percent', x[1].get('importance_score', 0)),
            reverse=True
        )
        
        return [(var, info.get('percent', 0)) for var, info in sorted_vars[:n]]
    
    def create_importance_table(self) -> pd.DataFrame:
        """Create a formatted importance table.
        
        Returns:
            DataFrame with variable importance rankings
        """
        importance = self.compute_importance(method='coefficients')
        
        data = []
        for var, info in importance['importance'].items():
            data.append({
                'Variable': var,
                'Coefficient': info.get('coefficient', 0),
                't-statistic': info.get('t_statistic', 0),
                'Importance (%)': info.get('percent', 0)
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Importance (%)', ascending=False)
        df['Rank'] = range(1, len(df) + 1)
        
        return df[['Rank', 'Variable', 'Coefficient', 't-statistic', 'Importance (%)']]