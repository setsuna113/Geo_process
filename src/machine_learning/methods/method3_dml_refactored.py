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
from ...evaluation.dml import DMLParameterCalculator
from ...evaluation.dml_comparison import DMLModelComparison
from ...hypothesis import (
    DMLClassicalTest,
    DMLEffectSize,
    DMLBayesian
)

logger = logging.getLogger(__name__)


class Method3DML(BaseRegressionMethod):
    """Method 3 - Double Machine Learning for causal inference.
    
    Tests H0 vs H1 (biodiversity begets biodiversity) using DML.
    Note: H2 (temperate mismatch) requires heterogeneous treatment effects,
    which standard DML cannot test. Use Method 2 GLM for H2.
    
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
    
    
    
    
    def _compute_dml_estimate(
        self,
        h2o_data: h2o.H2OFrame,
        treatments: List[str],  # Changed to handle multiple treatments
        confounders: List[str],
        outcome: str
    ) -> Dict[str, Any]:
        """Compute DML estimate using cross-fitting.
        
        Args:
            h2o_data: Full dataset
            treatments: Treatment variables (plants, vertebrates)
            confounders: Confounder variables
            outcome: Outcome variable
            
        Returns:
            Dictionary with raw DML outputs for hypothesis testing
        """
        n = h2o_data.nrows
        
        # Create fold column
        h2o_data['fold'] = h2o_data.runif(seed=42)
        h2o_data['fold'] = (h2o_data['fold'] * self.n_folds).floor()
        
        # Storage for residuals
        residuals_y = []
        residuals_treatments = {t: [] for t in treatments}
        
        # Cross-fitting
        for k in range(self.n_folds):
            logger.debug(f"Processing fold {k+1}/{self.n_folds}")
            
            # Split data
            train = h2o_data[h2o_data['fold'] != k]
            test = h2o_data[h2o_data['fold'] == k]
            
            # Fit outcome nuisance model
            g_model = H2OGBM(self.config.config)
            g_features = confounders + treatments
            g_data = train[g_features + [outcome]]
            g_model.fit(g_data, y=outcome)
            
            # Fit treatment nuisance models
            m_models = {}
            for treatment in treatments:
                m_model = H2OGBM(self.config.config)
                m_data = train[confounders + [treatment]]
                m_model.fit(m_data, y=treatment)
                m_models[treatment] = m_model
            
            # Predict on test
            g_pred = g_model.predict(test)
            
            # Compute residuals
            y_resid = test[outcome] - g_pred
            if hasattr(y_resid, 'as_data_frame'):
                y_resid = y_resid.as_data_frame().values.flatten()
            residuals_y.extend(y_resid)
            
            # Treatment residuals
            for treatment in treatments:
                m_pred = m_models[treatment].predict(test)
                d_resid = test[treatment] - m_pred
                if hasattr(d_resid, 'as_data_frame'):
                    d_resid = d_resid.as_data_frame().values.flatten()
                residuals_treatments[treatment].extend(d_resid)
        
        # Convert to arrays
        residuals_y = np.array(residuals_y)
        residuals_a1 = np.array(residuals_treatments[treatments[0]])
        residuals_a2 = np.array(residuals_treatments[treatments[1]]) if len(treatments) > 1 else None
        
        # Simple OLS on residuals for causal effects
        theta1 = np.dot(residuals_a1, residuals_y) / np.dot(residuals_a1, residuals_a1)
        theta2 = None
        if residuals_a2 is not None:
            theta2 = np.dot(residuals_a2, residuals_y) / np.dot(residuals_a2, residuals_a2)
        
        # Return raw outputs for hypothesis testing
        return {
            'residuals_Y': residuals_y,
            'residuals_A1': residuals_a1,
            'residuals_A2': residuals_a2,
            'theta1': theta1,
            'theta2': theta2,
            'n_obs': n
        }
    
    def test_biodiversity_hypothesis(
        self,
        h2o_data: Optional[h2o.H2OFrame] = None,
        hypothesis_tests: List[str] = ['classical', 'effect_size', 'bayesian']
    ) -> Dict[str, Any]:
        """Test biodiversity begets biodiversity hypothesis using DML.
        
        Note: DML can only test H0 vs H1 (causal effects).
        H2 (temperate mismatch) requires heterogeneous treatment effects or GLM.
        
        Args:
            h2o_data: Preprocessed data
            hypothesis_tests: Which tests to run
            
        Returns:
            Comprehensive hypothesis testing results
        """
        if h2o_data is None:
            h2o_data = self.data_h2o
        
        if h2o_data is None:
            raise ValueError("Must provide data")
        
        logger.info("Testing H0 vs H1 using DML")
        
        # Get MESF columns
        me_cols = [col for col in h2o_data.columns if col.startswith('ME')]
        
        # Define treatments and confounders for H0/H1 test
        treatments = ['plants_richness_log', 'vertebrates_richness_log']
        confounders = [
            'bio1_std', 'bio4_std', 'bio12_std', 'bio15_std',
            'elevation_std', 'elevation_std_sq'
        ] + me_cols
        outcome = 'fungi_richness'
        
        logger.info(f"Treatments: {treatments}")
        logger.info(f"Confounders: {len(confounders)} variables")
        logger.info(f"Outcome: {outcome}")
        
        # Store data for model comparison
        self.data_h2o = h2o_data
        
        # Compute raw DML outputs
        dml_outputs = self._compute_dml_estimate(
            h2o_data, treatments, confounders, outcome
        )
        
        # Calculate parameters for hypothesis testing
        params = DMLParameterCalculator.compute_all_parameters(
            residuals_Y=dml_outputs['residuals_Y'],
            residuals_A1=dml_outputs['residuals_A1'],
            residuals_A2=dml_outputs['residuals_A2'],
            theta1=dml_outputs['theta1'],
            theta2=dml_outputs['theta2'],
            robust_se=True
        )
        
        # Run selected hypothesis tests
        results = {
            'raw_outputs': dml_outputs,
            'parameters': params,
            'hypothesis_tests': {}
        }
        
        if 'classical' in hypothesis_tests:
            results['hypothesis_tests']['classical'] = DMLClassicalTest.test_biodiversity_hypothesis(
                theta_plants=params['theta1']['estimate'],
                se_plants=params['theta1']['standard_error'],
                theta_vertebrates=params['theta2']['estimate'],
                se_vertebrates=params['theta2']['standard_error']
            )
        
        if 'effect_size' in hypothesis_tests:
            results['hypothesis_tests']['effect_size'] = DMLEffectSize.evaluate_biodiversity_effects(
                theta_plants=params['theta1']['estimate'],
                se_plants=params['theta1']['standard_error'],
                theta_vertebrates=params['theta2']['estimate'],
                se_vertebrates=params['theta2']['standard_error']
            )
        
        if 'bayesian' in hypothesis_tests:
            results['hypothesis_tests']['bayesian'] = DMLBayesian.test_biodiversity_hypothesis_bayesian(
                theta_plants=params['theta1']['estimate'],
                se_plants=params['theta1']['standard_error'],
                theta_vertebrates=params['theta2']['estimate'],
                se_vertebrates=params['theta2']['standard_error']
            )
        
        # Store results
        self.results = results
        
        # Log summary
        logger.info("DML Results Summary:")
        logger.info(f"  θ_plants: {params['theta1']['estimate']:.4f} (SE: {params['theta1']['standard_error']:.4f})")
        logger.info(f"  θ_vertebrates: {params['theta2']['estimate']:.4f} (SE: {params['theta2']['standard_error']:.4f})")
        logger.info(f"  P-values: plants={params['theta1']['p_value']:.4f}, vertebrates={params['theta2']['p_value']:.4f}")
        
        return results
    
    def compare_models_with_without_biodiversity(
        self,
        h2o_data: Optional[h2o.H2OFrame] = None
    ) -> Dict[str, Any]:
        """Compare DML models with and without biodiversity predictors.
        
        Args:
            h2o_data: Preprocessed data
            
        Returns:
            Model comparison results
        """
        if h2o_data is None:
            h2o_data = self.data_h2o
        
        if h2o_data is None:
            raise ValueError("Must provide data")
        
        logger.info("Comparing models with/without biodiversity")
        
        # Get MESF columns
        me_cols = [col for col in h2o_data.columns if col.startswith('ME')]
        
        # Confounders only (no biodiversity)
        confounders = [
            'bio1_std', 'bio4_std', 'bio12_std', 'bio15_std',
            'elevation_std', 'elevation_std_sq'
        ] + me_cols
        outcome = 'fungi_richness'
        
        # For null model, fit only outcome nuisance model
        logger.info("Fitting model without biodiversity predictors")
        
        # Create fold column
        h2o_data['fold'] = h2o_data.runif(seed=42)
        h2o_data['fold'] = (h2o_data['fold'] * self.n_folds).floor()
        
        residuals_y_null = []
        
        for k in range(self.n_folds):
            train = h2o_data[h2o_data['fold'] != k]
            test = h2o_data[h2o_data['fold'] == k]
            
            # Fit outcome model without treatments
            g_model = H2OGBM(self.config.config)
            g_data = train[confounders + [outcome]]
            g_model.fit(g_data, y=outcome)
            
            # Predict and compute residuals
            g_pred = g_model.predict(test)
            y_resid = test[outcome] - g_pred
            if hasattr(y_resid, 'as_data_frame'):
                y_resid = y_resid.as_data_frame().values.flatten()
            residuals_y_null.extend(y_resid)
        
        dml_without = {'residuals_Y': np.array(residuals_y_null)}
        
        # Model with biodiversity
        logger.info("Fitting model with biodiversity predictors")
        treatments = ['plants_richness_log', 'vertebrates_richness_log']
        dml_with = self._compute_dml_estimate(
            h2o_data, treatments, confounders, outcome
        )
        
        # Compare models
        comparison = DMLModelComparison.compare_models(
            residuals_with_biodiv=dml_with['residuals_Y'],
            residuals_without_biodiv=dml_without['residuals_Y'],
            residuals_A1=dml_with['residuals_A1'],
            residuals_A2=dml_with['residuals_A2'],
            Y_observed=h2o_data[outcome].as_data_frame().values.flatten()
        )
        
        return comparison
    
    
    
    def fit_model(self, data: h2o.H2OFrame) -> Dict[str, Any]:
        """Fit the model - implements abstract method from base class.
        
        Args:
            data: Preprocessed H2O Frame
            
        Returns:
            Model fitting results with comprehensive hypothesis testing
        """
        self.data_h2o = data
        
        # Test H0 vs H1 with all hypothesis testing approaches
        results = self.test_biodiversity_hypothesis(
            data, 
            hypothesis_tests=['classical', 'effect_size', 'bayesian']
        )
        
        # Add model comparison if configured
        if self.config.get('include_model_comparison', True):
            results['model_comparison'] = self.compare_models_with_without_biodiversity(data)
        
        return results
    
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
        
        Note: DML estimates causal effects, not predictions.
        Returns average treatment effects for compatibility.
        
        Args:
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            Average treatment effects
        """
        if not self.results:
            raise ValueError("Must run hypothesis tests first")
        
        # Return average treatment effect
        if 'parameters' in self.results:
            theta1 = self.results['parameters']['theta1']['estimate']
            return np.full(data.nrows, theta1)
        else:
            raise ValueError("No treatment effects estimated")