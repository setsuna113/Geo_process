"""
Core Estimated Marginal Means (emmeans) computation for GLM models.

This module provides the basic functionality to compute emmeans (predicted values)
for each group in a factorial design, holding covariates at their mean values.
"""

import logging
import numpy as np
import pandas as pd
import h2o
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)


class EMMeansCore:
    """Core emmeans computation for GLM models.
    
    Computes estimated marginal means (predicted values) for each group
    in a factorial design while holding covariates at their mean values.
    """
    
    def __init__(self, model: Any, data: Union[pd.DataFrame, h2o.H2OFrame]):
        """Initialize emmeans calculator.
        
        Args:
            model: Fitted GLM model (H2O or similar)
            data: Data used for fitting (with categorical variables)
        """
        self.model = model
        self.data = data
        self.emmeans_results = None
        
    def compute_emmeans_2x2(
        self, 
        factor1: str = 'plant_high',
        factor2: str = 'temperate',
        covariates: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compute estimated marginal means for 2x2 factorial design.
        
        This is the core emmeans computation that produces 4 predicted values
        for the combinations of two binary factors.
        
        Args:
            factor1: First categorical factor (e.g., 'plant_high')
            factor2: Second categorical factor (e.g., 'temperate')
            covariates: List of covariates to hold at mean values
            
        Returns:
            Dictionary with emmeans for all 4 groups:
            {
                'group_0_0': {'emmean': value, 'factor1': 0, 'factor2': 0},
                'group_0_1': {'emmean': value, 'factor1': 0, 'factor2': 1},
                'group_1_0': {'emmean': value, 'factor1': 1, 'factor2': 0},
                'group_1_1': {'emmean': value, 'factor1': 1, 'factor2': 1}
            }
        """
        logger.info(f"Computing emmeans for {factor1} × {factor2}")
        
        # Define the 4 groups in 2x2 design
        groups = [
            {factor1: 0, factor2: 0, 'label': f'group_0_0'},
            {factor1: 0, factor2: 1, 'label': f'group_0_1'},
            {factor1: 1, factor2: 0, 'label': f'group_1_0'},
            {factor1: 1, factor2: 1, 'label': f'group_1_1'}
        ]
        
        # Compute mean values for covariates
        covariate_means = self._compute_covariate_means(covariates)
        
        # Compute emmeans for each group
        emmeans = {}
        for group in groups:
            # Create prediction dataset for this group
            pred_data = self._create_prediction_data(
                group, factor1, factor2, covariate_means
            )
            
            # Get prediction (emmean)
            if isinstance(self.data, h2o.H2OFrame):
                pred_frame = h2o.H2OFrame(pred_data)
                
                # Add interaction term if it exists in the model
                if f'{factor2}_x_{factor1}' in self.data.columns:
                    pred_frame[f'{factor2}_x_{factor1}'] = (
                        pred_frame[factor1] * pred_frame[factor2]
                    )
                elif f'{factor1}_x_{factor2}' in self.data.columns:
                    pred_frame[f'{factor1}_x_{factor2}'] = (
                        pred_frame[factor1] * pred_frame[factor2]
                    )
                
                prediction = self.model.predict(pred_frame)
                emmean = prediction[0, 0] if hasattr(prediction, '__getitem__') else prediction
            else:  # pandas DataFrame
                pred_df = pd.DataFrame([pred_data])
                prediction = self.model.predict(pred_df)
                emmean = prediction[0]
            
            # Store result
            emmeans[group['label']] = {
                'emmean': float(emmean),
                factor1: group[factor1],
                factor2: group[factor2]
            }
        
        self.emmeans_results = emmeans
        
        # Log results
        logger.info("Estimated marginal means:")
        for label, values in emmeans.items():
            logger.info(f"  {label}: {values['emmean']:.3f}")
        
        return emmeans
    
    def _compute_covariate_means(self, covariates: Optional[List[str]]) -> Dict[str, float]:
        """Compute mean values for covariates.
        
        Args:
            covariates: List of covariate names
            
        Returns:
            Dictionary of covariate means
        """
        covariate_means = {}
        
        if covariates:
            for cov in covariates:
                if cov in self.data.columns:
                    if isinstance(self.data, h2o.H2OFrame):
                        mean_val = self.data[cov].mean()
                        if isinstance(mean_val, list):
                            mean_val = mean_val[0]
                    else:
                        mean_val = self.data[cov].mean()
                    covariate_means[cov] = mean_val
        
        # Also compute means for standard covariates
        standard_covariates = ['bio1_std', 'bio4_std', 'bio12_std', 'bio15_std', 
                               'elevation_std', 'elevation_std_sq']
        
        for cov in standard_covariates:
            if cov in self.data.columns and cov not in covariate_means:
                if isinstance(self.data, h2o.H2OFrame):
                    mean_val = self.data[cov].mean()
                    if isinstance(mean_val, list):
                        mean_val = mean_val[0]
                else:
                    mean_val = self.data[cov].mean()
                covariate_means[cov] = mean_val
        
        # Add MESF eigenvectors
        for col in self.data.columns:
            if col.startswith('ME') and col not in covariate_means:
                if isinstance(self.data, h2o.H2OFrame):
                    mean_val = self.data[col].mean()
                    if isinstance(mean_val, list):
                        mean_val = mean_val[0]
                else:
                    mean_val = self.data[col].mean()
                covariate_means[col] = mean_val
        
        return covariate_means
    
    def _create_prediction_data(
        self, 
        group: Dict[str, Any],
        factor1: str,
        factor2: str,
        covariate_means: Dict[str, float]
    ) -> Dict[str, Any]:
        """Create prediction dataset for a specific group.
        
        Args:
            group: Group specification with factor levels
            factor1: First factor name
            factor2: Second factor name
            covariate_means: Dictionary of covariate means
            
        Returns:
            Dictionary ready for prediction
        """
        pred_dict = {
            factor1: [group[factor1]], 
            factor2: [group[factor2]]
        }
        
        # Add all covariate means
        for cov, mean_val in covariate_means.items():
            pred_dict[cov] = [mean_val]
        
        return pred_dict
    
    def get_emmeans_array(self) -> np.ndarray:
        """Get emmeans as a numpy array in standard order.
        
        Returns:
            Array of 4 emmeans in order: [00, 01, 10, 11]
        """
        if self.emmeans_results is None:
            raise ValueError("Must compute emmeans first")
        
        return np.array([
            self.emmeans_results['group_0_0']['emmean'],
            self.emmeans_results['group_0_1']['emmean'],
            self.emmeans_results['group_1_0']['emmean'],
            self.emmeans_results['group_1_1']['emmean']
        ])
    
    def get_emmeans_table(self) -> pd.DataFrame:
        """Convert emmeans results to a formatted DataFrame.
        
        Returns:
            DataFrame with emmeans results
        """
        if self.emmeans_results is None:
            raise ValueError("Must compute emmeans first")
        
        rows = []
        for label, values in self.emmeans_results.items():
            row = {'group': label}
            row.update(values)
            rows.append(row)
        
        return pd.DataFrame(rows)