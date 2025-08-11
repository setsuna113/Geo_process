"""Likelihood Ratio Test implementation for GLM models."""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class LikelihoodRatioTest:
    """Likelihood Ratio Test for nested model comparison.
    
    The LRT compares nested models by:
    LR = -2 * (log L_null - log L_full) ~ chi2(df)
    where df = difference in number of parameters
    """
    
    @staticmethod
    def test_nested_models(
        null_model: Any,
        full_model: Any,
        df_difference: Optional[int] = None
    ) -> Dict[str, Any]:
        """Perform likelihood ratio test between nested models.
        
        Args:
            null_model: Restricted (null) model
            full_model: Unrestricted (full) model
            df_difference: Difference in degrees of freedom (if not provided, will try to infer)
            
        Returns:
            Dictionary with test statistics and p-value
        """
        # Get log-likelihoods or deviances
        if hasattr(null_model, 'get_log_likelihood') and hasattr(full_model, 'get_log_likelihood'):
            null_ll = null_model.get_log_likelihood()
            full_ll = full_model.get_log_likelihood()
            lr_statistic = 2 * (full_ll - null_ll)
        elif hasattr(null_model, 'get_deviance') and hasattr(full_model, 'get_deviance'):
            # Deviance = -2 * log-likelihood (up to constant)
            null_deviance = null_model.get_deviance()
            full_deviance = full_model.get_deviance()
            lr_statistic = null_deviance - full_deviance
        else:
            raise ValueError("Models must have either get_log_likelihood() or get_deviance() methods")
        
        # Get degrees of freedom difference
        if df_difference is None:
            if hasattr(null_model, 'get_num_parameters') and hasattr(full_model, 'get_num_parameters'):
                null_params = null_model.get_num_parameters()
                full_params = full_model.get_num_parameters()
                df_difference = full_params - null_params
            elif hasattr(null_model, 'get_degrees_of_freedom') and hasattr(full_model, 'get_degrees_of_freedom'):
                null_df = null_model.get_degrees_of_freedom()
                full_df = full_model.get_degrees_of_freedom()
                df_difference = null_df - full_df  # Note: df decreases as params increase
            else:
                raise ValueError("Cannot determine degrees of freedom difference. Please provide df_difference parameter.")
        
        if df_difference <= 0:
            raise ValueError(f"Degrees of freedom difference must be positive (got {df_difference}). Ensure full model has more parameters than null model.")
        
        # Compute p-value from chi-squared distribution
        p_value = 1 - stats.chi2.cdf(lr_statistic, df=df_difference)
        
        # Get additional model information if available
        result = {
            'lr_statistic': lr_statistic,
            'degrees_of_freedom': df_difference,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'test_type': 'likelihood_ratio'
        }
        
        # Add model fit statistics if available
        if hasattr(null_model, 'get_aic'):
            result['null_model_aic'] = null_model.get_aic()
        if hasattr(full_model, 'get_aic'):
            result['full_model_aic'] = full_model.get_aic()
        if hasattr(null_model, 'get_bic'):
            result['null_model_bic'] = null_model.get_bic()
        if hasattr(full_model, 'get_bic'):
            result['full_model_bic'] = full_model.get_bic()
        if hasattr(null_model, 'get_deviance'):
            result['null_model_deviance'] = null_model.get_deviance()
        if hasattr(full_model, 'get_deviance'):
            result['full_model_deviance'] = full_model.get_deviance()
        
        return result
    
    @staticmethod
    def test_coefficient_subset(
        full_model: Any,
        null_model: Any,
        coefficient_names: Optional[Union[str, list]] = None
    ) -> Dict[str, Any]:
        """Test significance of a subset of coefficients.
        
        This is a convenience wrapper for testing whether specific coefficients
        significantly improve model fit.
        
        Args:
            full_model: Model with all coefficients
            null_model: Model without the tested coefficients
            coefficient_names: Names of coefficients being tested (for documentation)
            
        Returns:
            Dictionary with test results
        """
        result = LikelihoodRatioTest.test_nested_models(null_model, full_model)
        
        if coefficient_names:
            if isinstance(coefficient_names, str):
                coefficient_names = [coefficient_names]
            result['tested_coefficients'] = coefficient_names
            
            # Try to get coefficient values from full model
            if hasattr(full_model, 'get_coefficient'):
                coef_values = {}
                for name in coefficient_names:
                    coef_info = full_model.get_coefficient(name)
                    if coef_info:
                        coef_values[name] = {
                            'estimate': coef_info.get('coefficient'),
                            'std_error': coef_info.get('std_error'),
                            'p_value': coef_info.get('p_value')
                        }
                if coef_values:
                    result['coefficient_estimates'] = coef_values
        
        return result
    
    @staticmethod
    def test_deviance_difference(
        null_deviance: float,
        full_deviance: float,
        df_difference: int
    ) -> Dict[str, Any]:
        """Perform LRT using deviance values directly.
        
        Useful when you have deviance values but not model objects.
        
        Args:
            null_deviance: Deviance of null model
            full_deviance: Deviance of full model
            df_difference: Difference in degrees of freedom
            
        Returns:
            Dictionary with test statistics and p-value
        """
        if df_difference <= 0:
            raise ValueError(f"Degrees of freedom difference must be positive (got {df_difference})")
        
        lr_statistic = null_deviance - full_deviance
        
        if lr_statistic < 0:
            logger.warning(f"Negative LR statistic ({lr_statistic}). Full model may have higher deviance than null model.")
        
        # Compute p-value from chi-squared distribution
        p_value = 1 - stats.chi2.cdf(max(0, lr_statistic), df=df_difference)
        
        return {
            'null_deviance': null_deviance,
            'full_deviance': full_deviance,
            'lr_statistic': lr_statistic,
            'degrees_of_freedom': df_difference,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'test_type': 'likelihood_ratio_deviance'
        }
    
    @staticmethod
    def sequential_test(
        models: list,
        model_names: Optional[list] = None
    ) -> Dict[str, Any]:
        """Perform sequential likelihood ratio tests on a series of nested models.
        
        Args:
            models: List of models, each nested within the next
            model_names: Optional names for the models
            
        Returns:
            Dictionary with sequential test results
        """
        if len(models) < 2:
            raise ValueError("Need at least 2 models for comparison")
        
        if model_names and len(model_names) != len(models):
            raise ValueError("Number of model names must match number of models")
        
        if not model_names:
            model_names = [f"Model_{i}" for i in range(len(models))]
        
        results = {
            'n_models': len(models),
            'model_names': model_names,
            'comparisons': [],
            'test_type': 'sequential_likelihood_ratio'
        }
        
        # Perform pairwise comparisons
        for i in range(len(models) - 1):
            comparison = {
                'null_model': model_names[i],
                'full_model': model_names[i + 1],
            }
            
            try:
                test_result = LikelihoodRatioTest.test_nested_models(
                    null_model=models[i],
                    full_model=models[i + 1]
                )
                comparison.update(test_result)
            except Exception as e:
                logger.warning(f"Failed to compare {model_names[i]} vs {model_names[i+1]}: {e}")
                comparison['error'] = str(e)
            
            results['comparisons'].append(comparison)
        
        # Overall test (first vs last)
        if len(models) > 2:
            try:
                overall_test = LikelihoodRatioTest.test_nested_models(
                    null_model=models[0],
                    full_model=models[-1]
                )
                results['overall_test'] = {
                    'null_model': model_names[0],
                    'full_model': model_names[-1],
                    **overall_test
                }
            except Exception as e:
                logger.warning(f"Failed to perform overall test: {e}")
                results['overall_test'] = {'error': str(e)}
        
        return results