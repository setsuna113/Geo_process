"""Method 3: Double Machine Learning (REFACTORED).

Tests causal effects using DML methodology.
This refactored version assumes data is already preprocessed by the pipeline.
"""

import logging
import numpy as np
import h2o
from typing import Dict, Any, Optional, Union, Tuple, List
from pathlib import Path

from src.base.ml import BaseRegressionMethod
from src.abstractions.types.ml_types import ModelType, ModelMetadata
from ...models.gbm.h2o_gbm import H2OGBM
from ...config_loader import MLRegressionConfig
from ...preprocessing import (
    add_plants_bio4_interaction,
    create_temperate_categories,
    create_biodiversity_categories,
    add_categorical_interactions
)

logger = logging.getLogger(__name__)


class Method3DML(BaseRegressionMethod):
    """Refactored Method 3 - Double Machine Learning for causal inference.
    
    Tests three hypotheses:
    - H0: No effect (independence)
    - H1: Biodiversity begets biodiversity
    - H2: Temperate mismatch
    
    Expected preprocessed features:
    - Climate variables: bio1_std, bio4_std, bio12_std, bio15_std
    - Elevation: elevation_std, elevation_std_sq
    - Biodiversity: plants_richness_log, vertebrates_richness_log
    - MESF: ME1, ME2, ... (spatial eigenvectors)
    - Target: fungi_richness
    """
    
    def __init__(self, config: MLRegressionConfig) -> None:
        """Initialize Method 3 DML.
        
        Args:
            config: ML regression configuration
        """
        super().__init__(config)
        self.dml_config = self.config.get_dml_config()
        
        # DML parameters
        self.n_folds = self.dml_config.get('n_folds', 5)
        self.n_rep = self.dml_config.get('n_rep', 1)
        
        # Storage
        self.data_h2o = None
        self.results = {}
        self.models = {}
    
    def add_dml_features(self, h2o_data: h2o.H2OFrame) -> h2o.H2OFrame:
        """Add Method3-specific features for DML.
        
        Args:
            h2o_data: Preprocessed data from pipeline
            
        Returns:
            Data with DML-specific features added
        """
        logger.info("Adding Method3-specific DML features")
        
        # H1: Continuous interaction (plants × bio4)
        h2o_data = add_plants_bio4_interaction(
            h2o_data,
            'plants_richness_log',
            'bio4_std',
            'plants_bio4_interaction'
        )
        
        # H2: Categorical interaction (temperate × high plant diversity)
        # Create temperate category
        h2o_data = create_temperate_categories(
            h2o_data,
            bio1_col='bio1',
            output_col='temperate',
            method='median'
        )
        
        # Create plant diversity category
        h2o_data = create_biodiversity_categories(
            h2o_data,
            richness_col='plants_richness',
            output_col='plant_high',
            method='median'
        )
        
        # Create categorical interaction
        h2o_data = add_categorical_interactions(
            h2o_data,
            cat1='temperate',
            cat2='plant_high',
            output_col='temperate_x_plant_high'
        )
        
        logger.info("DML features created")
        return h2o_data
    
    def _get_treatment_confounders_outcome(
        self, 
        hypothesis: str,
        h2o_data: h2o.H2OFrame
    ) -> Tuple[str, List[str], str]:
        """Get treatment, confounders, and outcome for each hypothesis.
        
        Args:
            hypothesis: 'h0', 'h1', or 'h2'
            h2o_data: Data with features
            
        Returns:
            Tuple of (treatment_col, confounder_cols, outcome_col)
        """
        # Get MESF columns
        me_cols = [col for col in h2o_data.columns if col.startswith('ME')]
        
        # Base confounders (climate + spatial)
        base_confounders = [
            'bio1_std', 'bio4_std', 'bio12_std', 'bio15_std',
            'elevation_std', 'elevation_std_sq',
            'vertebrates_richness_log'
        ] + me_cols
        
        if hypothesis == 'h0':
            # H0: No interaction
            treatment = 'plants_richness_log'
            confounders = base_confounders
            
        elif hypothesis == 'h1':
            # H1: Continuous interaction
            treatment = 'plants_bio4_interaction'
            confounders = base_confounders + ['plants_richness_log']
            
        elif hypothesis == 'h2':
            # H2: Categorical interaction
            treatment = 'temperate_x_plant_high'
            confounders = base_confounders + ['temperate', 'plant_high']
            
        else:
            raise ValueError(f"Unknown hypothesis: {hypothesis}")
        
        outcome = 'fungi_richness'
        return treatment, confounders, outcome
    
    def _fit_nuisance_models(
        self,
        h2o_data: h2o.H2OFrame,
        treatment: str,
        confounders: List[str],
        outcome: str,
        fold: int
    ) -> Tuple[Any, Any]:
        """Fit nuisance models for DML.
        
        Args:
            h2o_data: Training data
            treatment: Treatment variable
            confounders: Confounder variables
            outcome: Outcome variable
            fold: Fold number for naming
            
        Returns:
            Tuple of (g_model, m_model)
        """
        # Model for E[Y|X] (outcome model)
        logger.debug(f"Fitting g-model (outcome) for fold {fold}")
        g_model = H2OGBM(self.config.config)
        g_features = confounders + [treatment]
        g_data = h2o_data[g_features + [outcome]]
        g_model.fit(g_data, y=outcome)
        
        # Model for E[D|X] (treatment model)
        logger.debug(f"Fitting m-model (treatment) for fold {fold}")
        m_model = H2OGBM(self.config.config)
        m_data = h2o_data[confounders + [treatment]]
        m_model.fit(m_data, y=treatment)
        
        return g_model, m_model
    
    def _compute_dml_estimate(
        self,
        h2o_data: h2o.H2OFrame,
        treatment: str,
        confounders: List[str],
        outcome: str
    ) -> Dict[str, float]:
        """Compute DML estimate using cross-fitting.
        
        Args:
            h2o_data: Full dataset
            treatment: Treatment variable
            confounders: Confounder variables
            outcome: Outcome variable
            
        Returns:
            Dictionary with DML estimates
        """
        n = h2o_data.nrows
        
        # Create fold column
        h2o_data['fold'] = h2o_data.runif(seed=42)
        h2o_data['fold'] = (h2o_data['fold'] * self.n_folds).floor()
        
        # Storage for residuals
        residuals_y = []
        residuals_d = []
        
        # Cross-fitting
        for k in range(self.n_folds):
            logger.debug(f"Processing fold {k+1}/{self.n_folds}")
            
            # Split data
            train = h2o_data[h2o_data['fold'] != k]
            test = h2o_data[h2o_data['fold'] == k]
            
            # Fit nuisance models on training
            g_model, m_model = self._fit_nuisance_models(
                train, treatment, confounders, outcome, k
            )
            
            # Predict on test
            g_pred = g_model.predict(test)
            m_pred = m_model.predict(test)
            
            # Compute residuals
            y_resid = test[outcome] - g_pred
            d_resid = test[treatment] - m_pred
            
            # Convert to numpy if needed
            if hasattr(y_resid, 'as_data_frame'):
                y_resid = y_resid.as_data_frame().values.flatten()
                d_resid = d_resid.as_data_frame().values.flatten()
            
            residuals_y.extend(y_resid)
            residuals_d.extend(d_resid)
        
        # Convert to arrays
        residuals_y = np.array(residuals_y)
        residuals_d = np.array(residuals_d)
        
        # DML estimator (simple version)
        theta_dml = np.mean(residuals_y * residuals_d) / np.mean(residuals_d ** 2)
        
        # Standard error
        n = len(residuals_y)
        score = residuals_y - theta_dml * residuals_d
        variance = np.mean(score ** 2) / (np.mean(residuals_d ** 2) ** 2)
        se_dml = np.sqrt(variance / n)
        
        # Confidence interval
        ci_lower = theta_dml - 1.96 * se_dml
        ci_upper = theta_dml + 1.96 * se_dml
        
        # P-value (two-sided)
        z_stat = theta_dml / se_dml if se_dml > 0 else 0
        from scipy import stats
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return {
            'causal_effect': theta_dml,
            'std_error': se_dml,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'z_statistic': z_stat,
            'p_value': p_value,
            'n_obs': n
        }
    
    def test_hypothesis(
        self, 
        hypothesis: str,
        h2o_data: Optional[h2o.H2OFrame] = None
    ) -> Dict[str, Any]:
        """Test a specific hypothesis using DML.
        
        Args:
            hypothesis: 'h0', 'h1', or 'h2'
            h2o_data: Preprocessed data (uses self.data_h2o if None)
            
        Returns:
            DML results for the hypothesis
        """
        if h2o_data is None:
            h2o_data = self.data_h2o
        
        if h2o_data is None:
            raise ValueError("Must provide data or call with preprocessed data")
        
        logger.info(f"Testing hypothesis {hypothesis.upper()} using DML")
        
        # Add DML-specific features if not already present
        if 'plants_bio4_interaction' not in h2o_data.columns:
            h2o_data = self.add_dml_features(h2o_data)
            self.data_h2o = h2o_data
        
        # Get treatment, confounders, outcome
        treatment, confounders, outcome = self._get_treatment_confounders_outcome(
            hypothesis, h2o_data
        )
        
        logger.info(f"Treatment: {treatment}")
        logger.info(f"Confounders: {len(confounders)} variables")
        logger.info(f"Outcome: {outcome}")
        
        # Compute DML estimate
        results = self._compute_dml_estimate(
            h2o_data, treatment, confounders, outcome
        )
        
        # Add interpretation
        results['significant'] = results['p_value'] < 0.05
        results['interpretation'] = self._interpret_results(
            hypothesis, 
            results['causal_effect'],
            results['p_value'],
            results['ci_lower'],
            results['ci_upper']
        )
        
        # Store results
        self.results[hypothesis] = results
        
        logger.info(f"{hypothesis.upper()} Results:")
        logger.info(f"  Causal effect: {results['causal_effect']:.4f}")
        logger.info(f"  P-value: {results['p_value']:.4f}")
        logger.info(f"  Significant: {results['significant']}")
        
        return results
    
    def test_all_hypotheses(
        self,
        h2o_data: Optional[h2o.H2OFrame] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Test all three hypotheses.
        
        Args:
            h2o_data: Preprocessed data
            
        Returns:
            Dictionary with results for all hypotheses
        """
        results = {}
        
        for hypothesis in ['h0', 'h1', 'h2']:
            results[hypothesis] = self.test_hypothesis(hypothesis, h2o_data)
        
        return results
    
    def _interpret_results(
        self,
        hypothesis: str,
        effect: float,
        p_value: float,
        ci_lower: float,
        ci_upper: float
    ) -> str:
        """Interpret DML results.
        
        Args:
            hypothesis: Hypothesis being tested
            effect: Causal effect estimate
            p_value: P-value
            ci_lower: Lower CI bound
            ci_upper: Upper CI bound
            
        Returns:
            Interpretation string
        """
        if p_value < 0.05:
            if hypothesis == 'h0':
                if effect > 0:
                    return f"Significant positive effect (β={effect:.4f}): Plant diversity increases fungi richness"
                else:
                    return f"Significant negative effect (β={effect:.4f}): Plant diversity decreases fungi richness"
                    
            elif hypothesis == 'h1':
                if effect > 0:
                    return f"Significant positive interaction (β={effect:.4f}): Biodiversity begets biodiversity effect is stronger in seasonal climates"
                else:
                    return f"Significant negative interaction (β={effect:.4f}): Biodiversity begets biodiversity effect is weaker in seasonal climates"
                    
            elif hypothesis == 'h2':
                if effect > 0:
                    return f"Significant interaction (β={effect:.4f}): Plant-fungi relationship is stronger in temperate regions with high plant diversity"
                else:
                    return f"Significant interaction (β={effect:.4f}): Plant-fungi relationship is weaker in temperate regions with high plant diversity"
        else:
            return f"No significant effect detected (CI includes 0: [{ci_lower:.4f}, {ci_upper:.4f}])"
    
    def run_robustness_checks(self) -> Dict[str, Any]:
        """Run robustness checks for DML estimates.
        
        Returns:
            Dictionary with robustness check results
        """
        if not self.results:
            raise ValueError("Must run hypothesis tests first")
        
        robustness = {}
        
        # Check consistency across hypotheses
        if 'h0' in self.results and 'h1' in self.results:
            # H1 should show interaction if H0 shows main effect
            h0_sig = self.results['h0']['significant']
            h1_sig = self.results['h1']['significant']
            robustness['consistency_h0_h1'] = {
                'consistent': (h0_sig and h1_sig) or (not h0_sig and not h1_sig),
                'h0_significant': h0_sig,
                'h1_significant': h1_sig
            }
        
        return robustness
    
    def fit_model(self, data: h2o.H2OFrame) -> Dict[str, Any]:
        """Fit the model - implements abstract method from base class.
        
        Args:
            data: Preprocessed H2O Frame
            
        Returns:
            Model fitting results
        """
        return self.test_all_hypotheses(data)
    
    def load_data(self, *args, **kwargs):
        """REMOVED - Data loading is now handled by pipeline."""
        raise NotImplementedError(
            "Data loading has been moved to the pipeline. "
            "Use pipeline.load_and_preprocess_data() instead."
        )
    
    def prepare_data(self, *args, **kwargs):
        """REMOVED - Data preparation is now handled by pipeline."""
        raise NotImplementedError(
            "Data preparation has been moved to the pipeline. "
            "Common preprocessing is done in pipeline.load_and_preprocess_data()."
        )
    
    def predict(self, data: h2o.H2OFrame, **kwargs) -> np.ndarray:
        """Make predictions using fitted model.
        
        Note: DML doesn't directly predict, it estimates causal effects.
        This returns the treatment effect for compatibility.
        
        Args:
            data: Input data for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Treatment effects
        """
        if not self.results:
            raise ValueError("Must run hypothesis tests first")
        
        # Return the main effect (H0) as "predictions"
        if 'h0' in self.results:
            effect = self.results['h0']['causal_effect']
            return np.full(data.nrows, effect)
        else:
            raise ValueError("No H0 results available")