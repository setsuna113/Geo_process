"""Simple Python-based emmeans computation without R dependency."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class SimpleEMMeans:
    """Compute estimated marginal means without R emmeans package."""
    
    def __init__(self, 
                 coefficients: np.ndarray,
                 vcov_matrix: np.ndarray,
                 coefficient_names: List[str],
                 data_sample: pd.DataFrame):
        """
        Initialize simple emmeans calculator.
        
        Args:
            coefficients: Model coefficients
            vcov_matrix: Variance-covariance matrix
            coefficient_names: Names of coefficients
            data_sample: Representative data sample
        """
        self.coefficients = coefficients
        self.vcov_matrix = vcov_matrix
        self.coefficient_names = coefficient_names
        self.data_sample = data_sample
        
        # Create coefficient lookup
        self.coef_dict = dict(zip(coefficient_names, coefficients))
        
    def compute_2x2_emmeans(self,
                           factor1: str = 'temperate',
                           factor2: str = 'plant_high') -> Dict[str, any]:
        """
        Compute EMMs for 2x2 factorial design.
        
        Args:
            factor1: First factor (temperate)
            factor2: Second factor (plant_high)
            
        Returns:
            Dictionary with EMMs and contrasts
        """
        logger.info(f"Computing simple EMMs for {factor1} × {factor2}")
        
        # Get means of continuous predictors
        continuous_cols = [col for col in self.data_sample.columns 
                          if col not in [factor1, factor2, 'fungi_richness'] 
                          and self.data_sample[col].dtype in ['float64', 'float32']]
        
        continuous_means = self.data_sample[continuous_cols].mean()
        
        # Compute EMMs for each cell of 2x2 design
        emmeans_results = {}
        
        for f1_val in [0, 1]:
            for f2_val in [0, 1]:
                cell_name = f"{factor1}_{f1_val}_{factor2}_{f2_val}"
                
                # Create prediction point at means with specific factor levels
                pred_point = continuous_means.to_dict()
                pred_point[factor1] = f1_val
                pred_point[factor2] = f2_val
                
                # Add interaction if it exists
                interaction_name = f"{factor1}_x_{factor2}"
                if interaction_name in self.coefficient_names:
                    pred_point[interaction_name] = f1_val * f2_val
                
                # Compute linear predictor
                linear_pred = self.coef_dict.get('Intercept', 0)
                
                for coef_name, coef_val in self.coef_dict.items():
                    if coef_name != 'Intercept':
                        # Handle different coefficient naming patterns
                        if coef_name in pred_point:
                            linear_pred += coef_val * pred_point[coef_name]
                        elif coef_name.replace('_', '.') in pred_point:
                            # Handle R-style naming (bio04_std vs bio04.std)
                            col_name = coef_name.replace('_', '.')
                            linear_pred += coef_val * pred_point[col_name]
                
                # For Poisson GLM, transform from log scale
                emmean_response = np.exp(linear_pred)
                
                emmeans_results[cell_name] = {
                    'emmean_link': linear_pred,
                    'emmean_response': emmean_response,
                    factor1: f1_val,
                    factor2: f2_val
                }
        
        # Compute key contrasts
        contrasts = {}
        
        # Main effect of plant diversity
        plant_effect_low_temp = (
            emmeans_results[f'{factor1}_0_{factor2}_1']['emmean_response'] - 
            emmeans_results[f'{factor1}_0_{factor2}_0']['emmean_response']
        )
        
        plant_effect_high_temp = (
            emmeans_results[f'{factor1}_1_{factor2}_1']['emmean_response'] - 
            emmeans_results[f'{factor1}_1_{factor2}_0']['emmean_response']
        )
        
        # Interaction: difference of differences
        interaction_effect = plant_effect_high_temp - plant_effect_low_temp
        
        contrasts['plant_effect_low_temp'] = plant_effect_low_temp
        contrasts['plant_effect_high_temp'] = plant_effect_high_temp
        contrasts['interaction'] = interaction_effect
        
        # Test H2 hypothesis: plant effect stronger in low temperate regions
        h2_support = plant_effect_low_temp > plant_effect_high_temp
        
        results = {
            'emmeans': emmeans_results,
            'contrasts': contrasts,
            'h2_hypothesis_supported': h2_support,
            'interpretation': self._interpret_results(contrasts)
        }
        
        logger.info(f"EMMs computed successfully")
        logger.info(f"Plant effect in low temperate: {plant_effect_low_temp:.4f}")
        logger.info(f"Plant effect in high temperate: {plant_effect_high_temp:.4f}")
        logger.info(f"Interaction effect: {interaction_effect:.4f}")
        logger.info(f"H2 hypothesis supported: {h2_support}")
        
        return results
    
    def _interpret_results(self, contrasts: Dict) -> str:
        """Generate interpretation of results."""
        
        if contrasts['interaction'] < 0:
            # Negative interaction supports H2
            interpretation = (
                "The negative interaction suggests that the positive effect of "
                "plant diversity on fungi is stronger in low-seasonality (temperate) "
                "regions, supporting the temperate mismatch hypothesis (H2)."
            )
        else:
            interpretation = (
                "The positive interaction suggests that the effect of plant diversity "
                "on fungi does not follow the expected temperate mismatch pattern."
            )
        
        return interpretation