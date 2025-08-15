"""Method 3: Double Machine Learning (REFACTORED).

Tests causal effects using DML methodology.
This refactored version assumes data is already preprocessed by the pipeline.
"""

import logging
import gc
import numpy as np
import pandas as pd
import h2o
from typing import Dict, Any, Optional, Union, Tuple, List
from pathlib import Path

# No base class - each method stands alone
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from ..config_loader import MLRegressionConfig
from ..evaluation.dml import DMLParameterCalculator
from ..evaluation.dml_comparison import DMLModelComparison
from ..hypothesis import (
    DMLClassicalTest,
    DMLEffectSize,
    DMLBayesian
)
from ..utils import H2OManager

logger = logging.getLogger(__name__)


class Method3DML:
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
        self.config = config
        # Get config dict once
        config_dict = config.config if hasattr(config, 'config') else {}
        self.dml_config = config_dict.get('dml', {})
        
        # DML parameters
        self.n_folds = self.dml_config.get('n_folds', 5)
        self.n_rep = self.dml_config.get('n_rep', 1)
        
        # Initialize H2O safely
        self.h2o = H2OManager.get_or_init(
            max_mem_size=self.config.get('h2o_memory', '32G'),
            nthreads=self.config.get('h2o_threads', -1)
        )
        
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
        try:
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
                
                # Fit outcome nuisance model - Direct H2O
                g_model = H2OGradientBoostingEstimator(
                    ntrees=100,
                    max_depth=5,
                    learn_rate=0.1,
                    nfolds=0  # No CV for DML nuisance
                )
                g_features = confounders + treatments
                g_model.train(x=g_features, y=outcome, training_frame=train)
                
                # Fit treatment nuisance models
                m_models = {}
                for treatment in treatments:
                    # Direct H2O GBM for treatment model
                    m_model = H2OGradientBoostingEstimator(
                        ntrees=100,
                        max_depth=5,
                        learn_rate=0.1,
                        nfolds=0
                    )
                    m_model.train(x=confounders, y=treatment, training_frame=train)
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
                
                # Clean up fold models to free memory
                del g_model, m_models
                gc.collect()
            
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
        
        finally:
            # Explicit cleanup of temporary arrays
            if 'residuals_y' in locals():
                del residuals_y
            if 'residuals_treatments' in locals():
                del residuals_treatments
            gc.collect()
            logger.debug("Memory cleanup completed")
    
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
            'bio01_std', 'bio04_std', 'bio12_std', 'bio15_std',
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
            # Get pure classical test result
            classical_result = DMLClassicalTest.test_biodiversity_hypothesis(
                theta_plants=params['theta1']['estimate'],
                se_plants=params['theta1']['standard_error'],
                theta_vertebrates=params['theta2']['estimate'],
                se_vertebrates=params['theta2']['standard_error']
            )
            
            # Add practical significance interpretation for large samples
            from src.machine_learning.hypothesis.practical_significance import PracticalSignificance
            n_obs = h2o_data.nrows
            
            # Enhance plants effect with practical significance
            if 'plants' in classical_result:
                classical_result['plants'] = PracticalSignificance.interpret_dml_with_practical_significance(
                    dml_result=classical_result['plants'],
                    treatment_effect=params['theta1']['estimate'],
                    standard_error=params['theta1']['standard_error'],
                    n_obs=n_obs,
                    sesoi=0.1,  # 0.1 log-unit change as meaningful causal effect
                    effect_type="Plants ATE"
                )
            
            # Enhance vertebrates effect with practical significance  
            if 'vertebrates' in classical_result:
                classical_result['vertebrates'] = PracticalSignificance.interpret_dml_with_practical_significance(
                    dml_result=classical_result['vertebrates'],
                    treatment_effect=params['theta2']['estimate'],
                    standard_error=params['theta2']['standard_error'],
                    n_obs=n_obs,
                    sesoi=0.1,  # 0.1 log-unit change as meaningful causal effect
                    effect_type="Vertebrates ATE"
                )
            
            results['hypothesis_tests']['classical'] = classical_result
        
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
    
    def test_all_hypotheses(self, data: Union[pd.DataFrame, h2o.H2OFrame]) -> Dict[str, Any]:
        """Test all relevant hypotheses for Method 3.
        
        Method 3 only tests H0-H1 (biodiversity effect) using DML.
        H2 (temperate mismatch) requires heterogeneous treatment effects
        which standard DML cannot test.
        
        Args:
            data: Preprocessed data
            
        Returns:
            Dictionary with hypothesis test results
        """
        # For Method 3, we only test H0-H1
        logger.info("Testing H0-H1 hypothesis using DML...")
        
        results = self.test_biodiversity_hypothesis(
            h2o_data=data,
            hypothesis_tests=['classical', 'effect_size']
        )
        
        # Format results for consistency with other methods
        hypothesis_results = {
            'h0': {
                'causal_effect': 0,
                'p_value': 1.0,
                'interpretation': 'No causal effect (null hypothesis)'
            },
            'h1': {
                'causal_effect': results['parameters']['theta1']['estimate'],
                'p_value': results['parameters']['theta1']['p_value'],
                'interpretation': 'Biodiversity has causal effect on fungi richness' 
                                if results['parameters']['theta1']['p_value'] < 0.05
                                else 'No significant causal effect detected'
            }
        }
        
        # Add detailed DML results
        hypothesis_results['dml_details'] = results
        
        return hypothesis_results
    
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
            'bio01_std', 'bio04_std', 'bio12_std', 'bio15_std',
            'elevation_std', 'elevation_std_sq'
        ] + me_cols
        outcome = 'fungi_richness'
        
        # For null model, fit only outcome nuisance model
        logger.info("Fitting model without biodiversity predictors")
        
        try:
            # Create fold column
            h2o_data['fold'] = h2o_data.runif(seed=42)
            h2o_data['fold'] = (h2o_data['fold'] * self.n_folds).floor()
            
            residuals_y_null = []
            
            for k in range(self.n_folds):
                train = h2o_data[h2o_data['fold'] != k]
                test = h2o_data[h2o_data['fold'] == k]
                
                # Fit outcome model without treatments - Direct H2O
                g_model = H2OGradientBoostingEstimator(
                    ntrees=100,
                    max_depth=5,
                    learn_rate=0.1,
                    nfolds=0
                )
                g_model.train(x=confounders, y=outcome, training_frame=train)
                
                # Predict and compute residuals
                g_pred = g_model.predict(test)
                y_resid = test[outcome] - g_pred
                if hasattr(y_resid, 'as_data_frame'):
                    y_resid = y_resid.as_data_frame().values.flatten()
                residuals_y_null.extend(y_resid)
                
                # Clean up fold model
                del g_model
                gc.collect()
            
            dml_without = {'residuals_Y': np.array(residuals_y_null)}
        
        finally:
            # Explicit cleanup
            if 'residuals_y_null' in locals():
                del residuals_y_null
            gc.collect()
        
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
    
    def generate_report(self, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive report for Method 3 DML causal inference.
        
        Coordinates reporting from DML hypothesis tests (Classical, Effect Size, Bayesian).
        
        Args:
            output_dir: Optional directory to save report files
            
        Returns:
            Dictionary with all report components
        """
        if not self.results:
            logger.warning("No DML results available. Run fit_model first.")
            return {}
        
        report = {
            'method': 'Method 3: DML Causal Inference',
            'hypotheses': {
                'H0': 'No causal effect of biodiversity on fungi richness',
                'H1': 'Biodiversity has causal effect on fungi richness'
            },
            'causal_parameters': {},
            'tests': {},
            'visualizations': []
        }
        
        # Extract causal parameter estimates
        if 'parameters' in self.results:
            params = self.results['parameters']
            report['causal_parameters'] = {
                'plants_effect': {
                    'theta': params.get('theta1', {}).get('estimate'),
                    'se': params.get('theta1', {}).get('std_error'),
                    'interpretation': f"1% increase in plant richness causes {params.get('theta1', {}).get('estimate', 0):.3f}% change in fungi"
                },
                'vertebrates_effect': {
                    'theta': params.get('theta2', {}).get('estimate'),
                    'se': params.get('theta2', {}).get('std_error'),
                    'interpretation': f"1% increase in vertebrate richness causes {params.get('theta2', {}).get('estimate', 0):.3f}% change in fungi"
                }
            }
        
        # Generate reports from each hypothesis testing approach
        if 'hypothesis_tests' in self.results:
            tests = self.results['hypothesis_tests']
            
            # Classical testing report
            if 'classical' in tests:
                classical_report = DMLClassicalTest.generate_publication_report(
                    tests['classical'],
                    method_number=3,
                    model_type="Double Machine Learning"
                )
                report['tests']['classical'] = classical_report
                
                # Generate visualization if output directory provided
                if output_dir:
                    try:
                        fig = DMLClassicalTest.visualize_causal_effects(
                            tests['classical'],
                            title="Method 3: DML Causal Effects"
                        )
                        viz_path = output_dir / "method3_classical_test.png"
                        fig.savefig(viz_path, dpi=300, bbox_inches='tight')
                        report['visualizations'].append(str(viz_path))
                        logger.info(f"DML classical test visualization saved to {viz_path}")
                    except Exception as e:
                        logger.warning(f"Failed to create DML classical visualization: {e}")
            
            # Effect size evaluation report
            if 'effect_size' in tests:
                effect_report = DMLEffectSize.generate_publication_report(
                    tests['effect_size'],
                    method_number=3
                )
                report['tests']['effect_size'] = effect_report
                
                # Generate visualization if output directory provided
                if output_dir:
                    try:
                        fig = DMLEffectSize.visualize_effect_sizes(
                            tests['effect_size'],
                            title="Method 3: DML Effect Sizes"
                        )
                        viz_path = output_dir / "method3_effect_sizes.png"
                        fig.savefig(viz_path, dpi=300, bbox_inches='tight')
                        report['visualizations'].append(str(viz_path))
                        logger.info(f"DML effect size visualization saved to {viz_path}")
                    except Exception as e:
                        logger.warning(f"Failed to create DML effect size visualization: {e}")
            
            # Bayesian interpretation report
            if 'bayesian' in tests:
                bayesian_report = DMLBayesian.generate_publication_report(
                    tests['bayesian'],
                    method_number=3
                )
                report['tests']['bayesian'] = bayesian_report
                
                # Generate visualization if output directory provided
                if output_dir:
                    try:
                        fig = DMLBayesian.visualize_posterior_distributions(
                            tests['bayesian'],
                            title="Method 3: DML Bayesian Analysis"
                        )
                        viz_path = output_dir / "method3_bayesian.png"
                        fig.savefig(viz_path, dpi=300, bbox_inches='tight')
                        report['visualizations'].append(str(viz_path))
                        logger.info(f"DML Bayesian visualization saved to {viz_path}")
                    except Exception as e:
                        logger.warning(f"Failed to create DML Bayesian visualization: {e}")
        
        # Model comparison results if available
        if 'model_comparison' in self.results:
            report['model_comparison'] = self.results['model_comparison']
        
        # Overall conclusion
        if 'hypothesis_tests' in self.results:
            tests = self.results['hypothesis_tests']
            # Synthesize conclusions from different approaches
            classical_support = tests.get('classical', {}).get('h1_supported', False)
            effect_meaningful = tests.get('effect_size', {}).get('both_meaningful', False)
            bayesian_support = tests.get('bayesian', {}).get('joint_probabilities', {}).get('both_positive', 0) > 0.95
            
            if classical_support and effect_meaningful and bayesian_support:
                overall_conclusion = "Strong causal evidence for biodiversity begets biodiversity"
            elif classical_support or bayesian_support:
                overall_conclusion = "Moderate causal evidence for biodiversity effects"
            else:
                overall_conclusion = "Weak or no causal evidence for biodiversity effects"
            
            report['conclusion'] = {
                'overall': overall_conclusion,
                'classical_support': classical_support,
                'effect_meaningful': effect_meaningful,
                'bayesian_support': bayesian_support
            }
        
        # Save report summary if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save JSON summary
            import json
            summary_path = output_dir / "method3_report_summary.json"
            with open(summary_path, 'w') as f:
                # Convert non-serializable objects
                report_serializable = self._make_serializable(report)
                json.dump(report_serializable, f, indent=2)
            logger.info(f"Method 3 report summary saved to {summary_path}")
        
        return report
    
    def _make_serializable(self, obj):
        """Convert pandas/numpy objects to serializable format."""
        import pandas as pd
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj