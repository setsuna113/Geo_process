"""Method 2: H2 Hypothesis Testing with Both Continuous and Categorical Models (REFACTORED).

Tests H2: Climate-diversity interaction differs in temperate vs non-temperate regions.
- Model 3a: Continuous interaction (bio1 × plants_richness)
- Model 3b: Categorical interaction (temperate × plant_high)
This refactored version assumes data is already preprocessed by the pipeline.
"""

import logging
import h2o
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from src.base.ml import BaseRegressionMethod
from src.abstractions.types.ml_types import ModelType, ModelMetadata
from ..models.glm.h2o_negative_binomial import H2ONegativeBinomialGLM
from ..config_loader import MLRegressionConfig
from ..evaluation.emmeans_core import REMmeansWrapper, SimpleEMmeansApproximation
from ..utils.vcov_extractor import extract_vcov_from_h2o
from ..hypothesis import (
    WaldTest, LikelihoodRatioTest, ScoreTest,
    ContrastAnalysis, SimpleSlopesAnalysis
)
from ..evaluation import GLMComparison
from ..preprocessing import (
    create_temperate_categories,
    create_biodiversity_categories,
    add_categorical_interactions
)

logger = logging.getLogger(__name__)


class Method2H2Testing(BaseRegressionMethod):
    """Method 2: Tests H2 hypothesis using both continuous and categorical interactions.
    
    Model 3a: Continuous interaction (bio1 × plants_richness)
    Model 3b: Categorical interaction (temperate × plant_high)
    
    Expected preprocessed features:
    - Climate variables: bio1_std, bio4_std, bio12_std, bio15_std
    - Elevation: elevation_std, elevation_std_sq
    - Biodiversity: plants_richness_log, vertebrates_richness_log
    - MESF: ME1, ME2, ... (spatial eigenvectors)
    - Target: fungi_richness
    """
    
    def __init__(self, config: MLRegressionConfig) -> None:
        """Initialize Method 2.
        
        Args:
            config: ML regression configuration
        """
        super().__init__(config)
        self.glm_config = self.config.get_glm_config()
        
        self.model_3a_continuous = None  # Model 3a: Continuous interaction
        self.model_3b_categorical = None  # Model 3b: Categorical interaction
        self.data_h2o = None
        self.emmeans_results = None
        self.contrast_results = None
    
    def add_categorical_features(self, h2o_data: h2o.H2OFrame) -> h2o.H2OFrame:
        """Add Method2-specific categorical features.
        
        Args:
            h2o_data: Preprocessed data from pipeline
            
        Returns:
            Data with categorical features added
        """
        logger.info("Adding Method2-specific categorical features")
        
        # Create temperate category (bio4 > 75th percentile)
        # High temperature seasonality = temperate regions
        h2o_data = create_temperate_categories(
            h2o_data,
            bio4_col='bio04_std',  # Use bio4 (temperature seasonality) not bio1!
            output_col='temperate',
            method='quantile',
            quantile=0.75  # Top 25% as per regression_analysis_decisions.md
        )
        
        # Create plant diversity category (plants > 75th percentile)
        h2o_data = create_biodiversity_categories(
            h2o_data,
            richness_col='plants_richness',
            output_col='plant_high',
            method='quantile',
            quantile=0.75  # Top 25% as per regression_analysis_decisions.md
        )
        
        # Create interaction term
        h2o_data = add_categorical_interactions(
            h2o_data,
            cat1='temperate',
            cat2='plant_high',
            output_col='temperate_x_plant_high'
        )
        
        # Log categorical levels
        temperate_counts = h2o_data['temperate'].table()
        plant_high_counts = h2o_data['plant_high'].table()
        
        logger.info(f"Temperate regions: {temperate_counts}")
        logger.info(f"High plant diversity: {plant_high_counts}")
        
        return h2o_data
    
    def fit_models(self, h2o_data: h2o.H2OFrame) -> Dict[str, Any]:
        """Fit both continuous and categorical interaction models for H2 testing.
        
        Args:
            h2o_data: Preprocessed data from pipeline
            
        Returns:
            Model fitting results with hypothesis tests
        """
        logger.info("Fitting Method 2 models for H2 hypothesis testing")
        
        # Store original data
        self.data_h2o = h2o_data
        
        # Get MESF columns
        me_cols = [col for col in h2o_data.columns if col.startswith('ME')]
        logger.info(f"Using {len(me_cols)} MESF eigenvectors")
        
        # Initialize GLM class
        GLMClass = H2ONegativeBinomialGLM
        
        # ========== Model 3a: Continuous Interaction ==========
        logger.info("Fitting Model 3a: Continuous interaction (bio1 × plants_richness)")
        
        # Create continuous interaction term
        h2o_data['bio01_x_plants'] = h2o_data['bio01_std'] * h2o_data['plants_richness_log']
        
        # Features for Model 3a
        feature_cols_3a = [
            # Climate variables
            'bio01_std', 'bio04_std', 'bio12_std', 'bio15_std',
            'elevation_std', 'elevation_std_sq',
            # Biodiversity
            'plants_richness_log', 'vertebrates_richness_log',
            # Continuous interaction
            'bio01_x_plants'  # Fixed: was 'bio1_x_plants', must match line 136
        ] + me_cols
        
        model_3a_data = h2o_data[feature_cols_3a + ['fungi_richness']]
        self.model_3a_continuous = GLMClass(self.config.config)
        self.model_3a_continuous.fit(model_3a_data, y='fungi_richness')
        
        # ========== Model 3b: Categorical Interaction ==========
        logger.info("Fitting Model 3b: Categorical interaction (temperate × plant_high)")
        
        # Add categorical features
        h2o_data_cat = self.add_categorical_features(h2o_data)
        
        # Features for Model 3b
        feature_cols_3b = [
            # Climate variables
            'bio01_std', 'bio04_std', 'bio12_std', 'bio15_std',
            'elevation_std', 'elevation_std_sq',
            # Biodiversity
            'plants_richness_log', 'vertebrates_richness_log',
            # Categorical variables and interaction
            'temperate', 'plant_high', 'temperate_x_plant_high'
        ] + me_cols
        
        model_3b_data = h2o_data_cat[feature_cols_3b + ['fungi_richness']]
        self.model_3b_categorical = GLMClass(self.config.config)
        self.model_3b_categorical.fit(model_3b_data, y='fungi_richness')
        
        # ========== Hypothesis Testing ==========
        results = {}
        
        # === Model 3a: Continuous Interaction Testing ===
        # Multiple tests are appropriate for continuous interaction
        
        # 1a. Wald test for continuous interaction coefficient
        continuous_wald = WaldTest.test_coefficients_from_model(
            model=self.model_3a_continuous,
            coefficient_names='bio1_x_plants',
            null_values=0.0
        )
        
        # 1b. Score test for continuous interaction (if possible)
        try:
            # Try to use score test if model provides necessary components
            continuous_score = ScoreTest.test_interaction(
                null_model=self.model_3a_continuous,  # This would need a model without interaction
                var1='bio01_std',
                var2='plants_richness_log',
                data=h2o_data
            )
        except:
            # Fallback to approximate score test
            continuous_score = {'note': 'Score test not available for H2O models'}
        
        # 1c. Model comparison (AIC/BIC) for continuous model
        results['model_3a_continuous'] = {
            'aic': self.model_3a_continuous.get_aic(),
            'bic': self.model_3a_continuous.get_bic(),
            'deviance': self.model_3a_continuous.get_deviance(),
            'wald_test': continuous_wald,
            'score_test': continuous_score,
            'h2_supported': continuous_wald.get('significant_at_0.05', False),
            'interpretation': 'Tests continuous bio1 × plants interaction'
        }
        
        # === Model 3b: Categorical Interaction Testing ===
        # For categorical, emmeans is primary; Wald is secondary
        
        # 2a. Wald test for categorical interaction (secondary test)
        categorical_wald = WaldTest.test_coefficients_from_model(
            model=self.model_3b_categorical,
            coefficient_names='temperate_x_plant_high',
            null_values=0.0
        )
        
        # 2b. Note about emmeans being the primary test for 3b
        results['model_3b_categorical'] = {
            'aic': self.model_3b_categorical.get_aic(),
            'bic': self.model_3b_categorical.get_bic(),
            'deviance': self.model_3b_categorical.get_deviance(),
            'wald_test': categorical_wald,
            'primary_test': 'emmeans contrasts (see compute_emmeans and test_h2_contrast methods)',
            'h2_supported': categorical_wald.get('significant_at_0.05', False),
            'interpretation': 'Tests categorical temperate × plant_high interaction',
            'note': 'emmeans provides more interpretable test of H2 hypothesis'
        }
        
        # 3. Compare Model 3a vs 3b (for model selection only, not hypothesis testing)
        model_comparison = GLMComparison.compare_models(
            models={'model_3a': self.model_3a_continuous, 'model_3b': self.model_3b_categorical},
            criteria=['aic', 'bic']
        )
        results['model_comparison'] = {
            **model_comparison,
            'note': 'Model comparison for selection purposes only; H2 testing uses interaction coefficients'
        }
        
        # Overall H2 conclusion
        h2_supported = (
            results['model_3a_continuous']['h2_supported'] or
            results['model_3b_categorical']['h2_supported']
        )
        
        results['h2_conclusion'] = {
            'h2_supported': h2_supported,
            'best_model': model_comparison['best_models'].get('aic'),
            'interpretation': 'Temperate mismatch effect detected (H2)' if h2_supported else 'No temperate mismatch effect'
        }
        
        logger.info("Model fitting and hypothesis testing complete")
        return results
    
    def fit_model(self, h2o_data: h2o.H2OFrame) -> Dict[str, Any]:
        """Wrapper for compatibility - calls fit_models."""
        return self.fit_models(h2o_data)
    
    def compute_emmeans(self) -> Dict[str, Any]:
        """Compute estimated marginal means using proper emmeans approach.
        
        Uses the EMMeans class for proper contrast testing as specified
        in regression_analysis_decisions.md.
        
        Returns:
            Dictionary with emmeans results
        """
        if self.model_3b_categorical is None:
            raise ValueError("Must fit models first")
        
        logger.info("Computing estimated marginal means with full vcov matrix")
        
        # Extract variance-covariance matrix from H2O model
        logger.info("Extracting variance-covariance matrix from H2O model")
        coefficients, vcov_matrix, coef_names = extract_vcov_from_h2o(
            self.model_3b_categorical.model,
            self.data_h2o,
            method='bootstrap',
            n_bootstrap=100
        )
        
        # Store vcov for later use in hypothesis tests
        self.vcov_matrix = vcov_matrix
        self.coefficients = coefficients
        self.coef_names = coef_names
        
        # Use SimpleEMmeansApproximation for now (R integration can be added later)
        # This computes emmeans at the mean of covariates
        # Convert H2O frame to pandas for emmeans calculation
        data_pd = self.data_h2o.as_data_frame()
        
        self.emmeans_calculator = SimpleEMmeansApproximation(
            coefficients=coefficients,
            coefficient_names=coef_names,
            data=data_pd,
            formula="fungi_richness ~ temperate * plant_high + bio01_std + bio12_std + bio15_std + elevation_std + elevation_std_sq + ME*"
        )
        
        # Compute emmeans for 2x2 design
        # Get covariate list (climate variables and MESF)
        covariates = ['bio1_std', 'bio12_std', 'bio15_std', 
                      'elevation_std', 'elevation_std_sq']
        covariates += [col for col in self.data_h2o.columns if col.startswith('ME')]
        
        emmeans_results = self.emmeans_calculator.compute_emmeans_2x2(
            factor1='plant_high',
            factor2='temperate',
            covariates=covariates
        )
        
        self.emmeans_results = emmeans_results
        return emmeans_results
    
    def test_h2_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive H2 hypothesis testing using contrast and simple slopes analysis.
        
        H2: High-plant temperate regions have fewer fungi than high-plant non-temperate.
        
        Uses two main testing approaches:
        1. Contrast Analysis (Test 6): Tests specific H2 pattern
        2. Simple Slopes Analysis (Test 7): Tests temperate effect within plant levels
        
        Returns:
            Dictionary with comprehensive test results
        """
        if self.model_3b_categorical is None:
            raise ValueError("Must fit models first")
        
        # Ensure emmeans are computed
        if not hasattr(self, 'emmeans_calculator') or self.emmeans_calculator is None:
            self.compute_emmeans()
        
        results = {}
        
        # TEST 6: Contrast Analysis on Fitted Values
        logger.info("Performing Test 6: Contrast Analysis")
        # Pass vcov_matrix for accurate standard errors
        contrast_analyzer = ContrastAnalysis(
            self.emmeans_calculator,
            vcov_matrix=self.vcov_matrix if hasattr(self, 'vcov_matrix') else None
        )
        
        # Primary H2 contrast
        h2_contrast = contrast_analyzer.test_h2_contrast(
            contrast_vector=[0, -1, 0, 1]
        )
        results['contrast_analysis'] = {
            'h2_test': h2_contrast,
            'test_name': 'Contrast Analysis on Fitted Values (Test 6)'
        }
        
        # Additional contrasts for completeness
        additional_contrasts = {
            'main_effect_plant': [-1, 1, -1, 1],
            'main_effect_temperate': [-1, -1, 1, 1]
        }
        additional_results = contrast_analyzer.test_custom_contrasts(additional_contrasts)
        results['contrast_analysis']['additional_contrasts'] = additional_results
        
        # TEST 7: Simple Slopes Analysis
        logger.info("Performing Test 7: Simple Slopes Analysis")
        # Pass vcov_matrix for accurate standard errors
        slopes_analyzer = SimpleSlopesAnalysis(
            self.emmeans_calculator,
            vcov_matrix=self.vcov_matrix if hasattr(self, 'vcov_matrix') else None
        )
        
        # H2-specific simple slopes test
        h2_slopes = slopes_analyzer.test_simple_slopes_h2()
        results['simple_slopes_analysis'] = {
            'h2_test': h2_slopes,
            'test_name': 'Simple Slopes Analysis (Test 7)'
        }
        
        # All simple effects for completeness
        all_slopes = slopes_analyzer.test_all_simple_effects()
        results['simple_slopes_analysis']['all_effects'] = all_slopes
        
        # 5. Get interaction coefficient from model (for reference)
        interaction_coef = self.model_3b_categorical.get_coefficient('temperate_x_plant_high')
        results['model_coefficient'] = {
            'coefficient': interaction_coef['coefficient'],
            'p_value': interaction_coef.get('p_value'),
            'note': 'Raw interaction coefficient - emmeans contrasts provide better interpretation'
        }
        
        # Get interaction coefficient from model (for reference only)
        interaction_coef = self.model_3b_categorical.get_coefficient('temperate_x_plant_high')
        results['model_coefficient'] = {
            'coefficient': interaction_coef['coefficient'],
            'p_value': interaction_coef.get('p_value'),
            'note': 'Raw interaction coefficient - Tests 6 & 7 provide better H2 evaluation'
        }
        
        # Overall H2 conclusion based on both tests
        h2_supported_contrast = results['contrast_analysis']['h2_test'].get('h2_supported', False)
        h2_supported_slopes = results['simple_slopes_analysis']['h2_test'].get('h2_supported', False)
        
        h2_supported = h2_supported_contrast or h2_supported_slopes
        
        results['overall_conclusion'] = {
            'h2_supported': h2_supported,
            'consensus': self._evaluate_consensus_new(results),
            'interpretation': self._generate_h2_interpretation_new(results)
        }
        
        self.contrast_results = results
        return results
    
    def test_h2_contrast(self) -> Dict[str, Any]:
        """Wrapper for backward compatibility - calls comprehensive test."""
        return self.test_h2_comprehensive()
    
    def _evaluate_consensus_new(self, results: Dict[str, Any]) -> str:
        """Evaluate consensus between Test 6 and Test 7.
        
        Args:
            results: Dictionary with test results
            
        Returns:
            Consensus evaluation string
        """
        contrast_h2 = results['contrast_analysis']['h2_test'].get('h2_supported', False)
        slopes_h2 = results['simple_slopes_analysis']['h2_test'].get('h2_supported', False)
        
        if contrast_h2 and slopes_h2:
            return "Strong support: Both tests confirm H2"
        elif contrast_h2 or slopes_h2:
            return "Moderate support: One test confirms H2"
        else:
            return "No support: Neither test confirms H2"
    
    def _generate_h2_interpretation_new(self, results: Dict[str, Any]) -> str:
        """Generate interpretation based on Test 6 and Test 7 results.
        
        Args:
            results: Dictionary with test results
            
        Returns:
            Interpretation string
        """
        contrast_test = results['contrast_analysis']['h2_test']
        slopes_test = results['simple_slopes_analysis']['h2_test']
        
        if contrast_test.get('h2_supported'):
            estimate = contrast_test.get('contrast_estimate', 0)
            p_value = contrast_test.get('p_value', 1)
            return (
                f"H2 SUPPORTED: Contrast analysis shows high-plant temperate regions have "
                f"{abs(estimate):.2f} fewer fungi than non-temperate (p={p_value:.4f})"
            )
        elif slopes_test.get('h2_supported'):
            slope = slopes_test['slope_temperate_in_high_plant']['estimate']
            p_value = slopes_test['slope_temperate_in_high_plant']['p_value']
            return (
                f"H2 SUPPORTED: Simple slopes shows temperate effect of {abs(slope):.2f} "
                f"fewer fungi in high-plant regions (p={p_value:.4f})"
            )
        else:
            return "H2 NOT SUPPORTED: No evidence for temperate mismatch pattern"
    
    def _evaluate_consensus(self, results: Dict[str, Any]) -> str:
        """Evaluate consensus across different H2 tests.
        
        Args:
            results: Dictionary with all test results
            
        Returns:
            Consensus evaluation string
        """
        supporting_tests = []
        contradicting_tests = []
        
        # Check primary contrast
        if results['primary_contrast'].get('h2_supported'):
            supporting_tests.append('primary_contrast')
        elif results['primary_contrast'].get('p_value', 1) < 0.05:
            contradicting_tests.append('primary_contrast')
        
        # Check simple effects
        if results['simple_effects'].get('h2_supported'):
            supporting_tests.append('simple_effects')
        elif results['simple_effects'].get('p_value', 1) < 0.05:
            contradicting_tests.append('simple_effects')
        
        # Check interaction decomposition
        if results['interaction_decomposition'].get('h2_specific', {}).get('h2_supported'):
            supporting_tests.append('interaction_decomposition')
        
        # Evaluate consensus
        if len(supporting_tests) >= 2:
            return f"Strong support for H2 ({len(supporting_tests)} tests confirm)"
        elif len(supporting_tests) == 1:
            return f"Moderate support for H2 (1 test confirms)"
        elif len(contradicting_tests) >= 1:
            return "Evidence against H2"
        else:
            return "No significant evidence for H2"
    
    def _generate_h2_interpretation(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive interpretation of H2 results.
        
        Args:
            results: Dictionary with all test results
            
        Returns:
            Interpretation string
        """
        primary = results['primary_contrast']
        simple = results['simple_effects']
        
        if primary.get('h2_supported'):
            estimate = primary.get('contrast_estimate', 0)
            p_value = primary.get('p_value', 1)
            return (
                f"H2 SUPPORTED: High-plant temperate regions have {abs(estimate):.2f} "
                f"fewer fungi species than high-plant non-temperate regions (p={p_value:.4f}). "
                "This confirms the temperate mismatch hypothesis where biodiversity "
                "relationships decouple in temperate climates."
            )
        elif simple.get('h2_supported'):
            diff = simple.get('difference', 0)
            p_value = simple.get('p_value', 1)
            return (
                f"H2 SUPPORTED (simple effects): Within high-plant regions, "
                f"temperate areas have {abs(diff):.2f} fewer fungi (p={p_value:.4f}), "
                "supporting the temperate mismatch pattern."
            )
        else:
            return (
                "H2 NOT SUPPORTED: No significant evidence for temperate mismatch. "
                "The relationship between plant and fungi diversity appears consistent "
                "across temperate and non-temperate regions."
            )
    
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
        
        Args:
            data: Input data for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Predicted values
        """
        if self.model_3b_categorical is None:
            raise ValueError("Must fit models first")
        
        # Use categorical model for prediction by default
        return self.model_3b_categorical.predict(data)
    
    def generate_report(self, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive report for Method 2 H2 testing.
        
        Coordinates reporting from ContrastAnalysis and SimpleSlopesAnalysis.
        
        Args:
            output_dir: Optional directory to save report files
            
        Returns:
            Dictionary with all report components
        """
        if not self.contrast_results:
            logger.warning("No H2 test results available. Run test_h2_comprehensive first.")
            return {}
        
        report = {
            'method': 'Method 2: H2 Hypothesis Testing',
            'hypothesis': 'H2: Temperate mismatch - high-plant temperate regions have fewer fungi',
            'models': {},
            'tests': {},
            'visualizations': []
        }
        
        # Model summaries
        if self.model_3a_continuous:
            report['models']['continuous'] = {
                'type': 'Model 3a: Continuous interaction (bio1 × plants)',
                'aic': self.model_3a_continuous.get_aic() if hasattr(self.model_3a_continuous, 'get_aic') else None,
                'bic': self.model_3a_continuous.get_bic() if hasattr(self.model_3a_continuous, 'get_bic') else None
            }
        
        if self.model_3b_categorical:
            report['models']['categorical'] = {
                'type': 'Model 3b: Categorical interaction (temperate × plant_high)',
                'aic': self.model_3b_categorical.get_aic() if hasattr(self.model_3b_categorical, 'get_aic') else None,
                'bic': self.model_3b_categorical.get_bic() if hasattr(self.model_3b_categorical, 'get_bic') else None
            }
        
        # Generate reports from hypothesis tests
        if hasattr(self, 'emmeans_calculator') and hasattr(self, 'vcov_matrix'):
            # Contrast Analysis report
            contrast_analyzer = ContrastAnalysis(
                self.emmeans_calculator,
                vcov_matrix=self.vcov_matrix
            )
            
            if 'contrast_analysis' in self.contrast_results:
                h2_test = self.contrast_results['contrast_analysis']['h2_test']
                contrast_report = contrast_analyzer.generate_publication_report(
                    h2_test,
                    method_number=2,
                    hypothesis_description="H2: Temperate mismatch hypothesis"
                )
                report['tests']['contrast_analysis'] = contrast_report
                
                # Generate visualization if output directory provided
                if output_dir:
                    try:
                        fig = contrast_analyzer.visualize_contrast_results(
                            h2_test,
                            self.emmeans_results if hasattr(self, 'emmeans_results') else None
                        )
                        viz_path = output_dir / "method2_contrast_analysis.png"
                        fig.savefig(viz_path, dpi=300, bbox_inches='tight')
                        report['visualizations'].append(str(viz_path))
                        logger.info(f"Contrast analysis visualization saved to {viz_path}")
                    except Exception as e:
                        logger.warning(f"Failed to create contrast visualization: {e}")
            
            # Simple Slopes Analysis report
            slopes_analyzer = SimpleSlopesAnalysis(
                self.emmeans_calculator,
                vcov_matrix=self.vcov_matrix
            )
            
            if 'simple_slopes_analysis' in self.contrast_results:
                slopes_test = self.contrast_results['simple_slopes_analysis']['h2_test']
                slopes_report = slopes_analyzer.generate_publication_report(
                    slopes_test,
                    method_number=2
                )
                report['tests']['simple_slopes'] = slopes_report
                
                # Generate visualization if output directory provided
                if output_dir:
                    try:
                        fig = slopes_analyzer.visualize_simple_slopes(
                            slopes_test,
                            self.emmeans_results if hasattr(self, 'emmeans_results') else None
                        )
                        viz_path = output_dir / "method2_simple_slopes.png"
                        fig.savefig(viz_path, dpi=300, bbox_inches='tight')
                        report['visualizations'].append(str(viz_path))
                        logger.info(f"Simple slopes visualization saved to {viz_path}")
                    except Exception as e:
                        logger.warning(f"Failed to create slopes visualization: {e}")
        
        # Overall conclusion
        report['conclusion'] = self.contrast_results.get('overall_conclusion', {})
        
        # Save report summary if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save JSON summary
            import json
            summary_path = output_dir / "method2_report_summary.json"
            with open(summary_path, 'w') as f:
                # Convert non-serializable objects
                report_serializable = self._make_serializable(report)
                json.dump(report_serializable, f, indent=2)
            logger.info(f"Method 2 report summary saved to {summary_path}")
        
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