"""Method 1: H0-H1 Testing with GLM (REFACTORED).

Tests:
- H0: Fungi richness is independent of plant/vertebrate richness
- H1: Biodiversity begets biodiversity (positive correlation exists)

Compares Model 1 (climate only) vs Model 2 (climate + biodiversity).
This refactored version assumes data is already preprocessed by the pipeline.
"""

import logging
import numpy as np
import h2o
from typing import Dict, Any, Optional, Union, Tuple, List
from pathlib import Path

from src.base.ml import BaseRegressionMethod
from src.abstractions.types.ml_types import ModelType, ModelMetadata
from ..models.glm.h2o_negative_binomial import H2ONegativeBinomialGLM
from ..config_loader import MLRegressionConfig
from ..hypothesis import WaldTest, LikelihoodRatioTest, ScoreTest
from ..evaluation.glm_comparison import ModelComparison

logger = logging.getLogger(__name__)


class Method1H0H1GLM(BaseRegressionMethod):
    """Method 1: Tests H0 vs H1 using nested GLM models.
    
    Model 1 (H0): Climate + spatial only
    Model 2 (H1): Climate + spatial + biodiversity
    
    Expected preprocessed features:
    - Climate variables: bio1_std, bio4_std, bio12_std, bio15_std
    - Elevation: elevation_std, elevation_std_sq
    - Biodiversity: plants_richness_log, vertebrates_richness_log
    - MESF: ME1, ME2, ... (spatial eigenvectors)
    - Target: fungi_richness
    """
    
    def __init__(self, config: MLRegressionConfig) -> None:
        """Initialize Method 1.
        
        Args:
            config: ML regression configuration
        """
        super().__init__(config)
        self.glm_config = self.config.get_glm_config()
        
        self.model_null = None  # Model 1: Climate only
        self.model_full = None  # Model 2: Climate + biodiversity
        self.data_h2o = None
    
    def fit_models(self, h2o_data: h2o.H2OFrame) -> Dict[str, Any]:
        """Fit the nested models using preprocessed data.
        
        Args:
            h2o_data: Preprocessed H2O DataFrame from pipeline
            
        Returns:
            Dictionary with fitted models and results
        """
        logger.info("Fitting Method 1 models with preprocessed data")
        
        # Store data
        self.data_h2o = h2o_data
        
        # Check overdispersion
        fungi_mean = h2o_data['fungi_richness'].mean()
        fungi_var = h2o_data['fungi_richness'].var()
        if isinstance(fungi_mean, list):
            fungi_mean = fungi_mean[0]
        if isinstance(fungi_var, list):
            fungi_var = fungi_var[0]
        dispersion_ratio = fungi_var / fungi_mean
        logger.info(f"Dispersion ratio: {dispersion_ratio:.2f}")
        
        # Use H2O GLM
        GLMClass = H2ONegativeBinomialGLM
        logger.info("Using H2O GLM for massive parallelization")
        
        # Get MESF columns
        me_cols = [col for col in h2o_data.columns if col.startswith('ME')]
        logger.info(f"Using {len(me_cols)} MESF eigenvectors")
        
        # Model 1: Climate only (null model)
        logger.info("Fitting Model 1: Climate only (null)")
        # Use actual column names from data (bio01, not bio1)
        feature_cols_null = [
            'bio01_std', 'bio04_std', 'bio12_std', 'bio15_std', 
            'elevation_std', 'elevation_std_sq'
        ] + me_cols
        
        null_data = h2o_data[feature_cols_null + ['fungi_richness']]
        self.model_null = GLMClass(self.config.config)
        self.model_null.fit(null_data, y='fungi_richness')
        
        # Model 2: Climate + biodiversity (full model)
        logger.info("Fitting Model 2: Climate + biodiversity (full)")
        biotic_cols = ['plants_richness_log', 'vertebrates_richness_log']
        
        feature_cols_full = feature_cols_null + biotic_cols
        full_data = h2o_data[feature_cols_full + ['fungi_richness']]
        
        self.model_full = GLMClass(self.config.config)
        self.model_full.fit(full_data, y='fungi_richness')
        
        # Use hypothesis testing classes for proper statistical tests
        results = {}
        
        # 1. Likelihood Ratio Test for H0 vs H1
        lrt_results = LikelihoodRatioTest.test_nested_models(
            null_model=self.model_null,
            full_model=self.model_full,
            df_difference=len(biotic_cols)  # Number of biodiversity predictors
        )
        results['likelihood_ratio_test'] = lrt_results
        
        # 2. Score Test (approximate from deviance if H2O doesn't provide score)
        # Using the approximate version since H2O may not expose score/information directly
        if hasattr(self.model_null, 'get_null_deviance') and hasattr(self.model_null, 'get_deviance'):
            score_results = ScoreTest.test_from_model_statistics(
                deviance_null=self.model_null.get_null_deviance(),
                deviance_residual=self.model_null.get_deviance(),
                df_null=self.model_null.get_null_degrees_of_freedom(),
                df_residual=self.model_null.get_residual_degrees_of_freedom(),
                n_obs=len(null_data)
            )
        else:
            # Fallback: approximate from deviance difference
            score_results = ScoreTest.test_from_model_statistics(
                deviance_null=self.model_null.get_deviance(),
                deviance_residual=self.model_full.get_deviance(),
                df_null=len(feature_cols_null),
                df_residual=len(feature_cols_full),
                n_obs=len(null_data)
            )
        results['score_test'] = score_results
        
        # 3. Model comparison using AIC/BIC
        model_comparison = GLMComparison.compare_models(
            models={'null_model': self.model_null, 'full_model': self.model_full},
            criteria=['aic', 'bic', 'deviance']
        )
        results['model_comparison'] = model_comparison
        
        # 4. Wald test for biodiversity coefficients
        wald_results = {}
        for coef_name in biotic_cols:
            wald_test = WaldTest.test_coefficients_from_model(
                model=self.model_full,
                coefficient_names=coef_name,
                null_values=0.0
            )
            wald_results[coef_name] = wald_test
        results['wald_tests'] = wald_results
        
        # Overall H0 vs H1 conclusion
        h1_supported = (
            lrt_results['significant_at_0.05'] or
            model_comparison['best_models'].get('aic') == 'full_model' or
            any(w.get('significant_at_0.05', False) for w in wald_results.values())
        )
        
        results['h0_h1_conclusion'] = {
            'h1_supported': h1_supported,
            'interpretation': 'Biodiversity begets biodiversity (H1)' if h1_supported else 'No biodiversity effect (H0)'
        }
        
        logger.info("Model fitting complete")
        return results
    
    def get_biodiversity_coefficients(self) -> Dict[str, Any]:
        """Get biodiversity coefficients for H1 hypothesis test.
        
        Returns:
            Dictionary with coefficient info for plants and vertebrates
        """
        if self.model_full is None:
            raise ValueError("Must fit models first")
        
        # Get coefficients for biodiversity predictors
        plants_coef = self.model_full.get_coefficient('plants_richness_log')
        vertebrates_coef = self.model_full.get_coefficient('vertebrates_richness_log')
        
        # H1 is supported if either biodiversity predictor is significant
        supports_h1 = (
            (plants_coef.get('p_value') is not None and plants_coef['p_value'] < 0.05) or
            (vertebrates_coef.get('p_value') is not None and vertebrates_coef['p_value'] < 0.05)
        )
        
        return {
            'plants': {
                'coefficient': plants_coef['coefficient'],
                'std_error': plants_coef.get('std_error'),
                'z_value': plants_coef.get('z_value'),
                'p_value': plants_coef.get('p_value')
            },
            'vertebrates': {
                'coefficient': vertebrates_coef['coefficient'],
                'std_error': vertebrates_coef.get('std_error'),
                'z_value': vertebrates_coef.get('z_value'),
                'p_value': vertebrates_coef.get('p_value')
            },
            'supports_h1': supports_h1,
            'interpretation': 'Biodiversity begets biodiversity' if supports_h1 else 'No biodiversity effect'
        }
    
    def fit_model(self, data: h2o.H2OFrame) -> Dict[str, Any]:
        """Fit the model - implements abstract method from base class.
        
        Args:
            data: Preprocessed H2O Frame
            
        Returns:
            Model fitting results
        """
        return self.fit_models(data)
    
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
        if self.model_full is None:
            raise ValueError("Must fit models first")
        
        return self.model_full.predict(data)
    
    def generate_report(self, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive report for Method 1 H0-H1 testing.
        
        Coordinates reporting from hypothesis tests (Wald, LRT, Score).
        
        Args:
            output_dir: Optional directory to save report files
            
        Returns:
            Dictionary with all report components
        """
        if not self.hypothesis_results:
            logger.warning("No hypothesis test results available. Run test_h0_h1 first.")
            return {}
        
        report = {
            'method': 'Method 1: H0-H1 Hypothesis Testing',
            'hypotheses': {
                'H0': 'No effect of biodiversity on fungi richness',
                'H1': 'Biodiversity begets biodiversity - positive effects'
            },
            'models': {},
            'tests': {},
            'visualizations': []
        }
        
        # Model summaries
        if self.model_full:
            report['models']['full'] = {
                'type': 'Model 1a: Full GLM with biodiversity predictors',
                'formula': self.hypothesis_results.get('model_formula', ''),
                'aic': self.model_full.get_aic() if hasattr(self.model_full, 'get_aic') else None,
                'bic': self.model_full.get_bic() if hasattr(self.model_full, 'get_bic') else None
            }
        
        if self.model_null:
            report['models']['null'] = {
                'type': 'Model 1b: Null model without biodiversity',
                'aic': self.model_null.get_aic() if hasattr(self.model_null, 'get_aic') else None,
                'bic': self.model_null.get_bic() if hasattr(self.model_null, 'get_bic') else None
            }
        
        # Generate report from Score Test if available
        if 'score_test' in self.hypothesis_results:
            score_result = self.hypothesis_results['score_test']
            score_report = ScoreTest.generate_publication_report(
                score_result,
                test_name="Score Test",
                hypothesis="biodiversity effects",
                variable_names=['plants_richness_log', 'vertebrates_richness_log']
            )
            report['tests']['score_test'] = score_report
            
            # Generate visualization if output directory provided
            if output_dir:
                try:
                    fig = ScoreTest.visualize_score_test(
                        score_result,
                        variable_names=['plants_richness_log', 'vertebrates_richness_log']
                    )
                    viz_path = output_dir / "method1_score_test.png"
                    fig.savefig(viz_path, dpi=300, bbox_inches='tight')
                    report['visualizations'].append(str(viz_path))
                    logger.info(f"Score test visualization saved to {viz_path}")
                except Exception as e:
                    logger.warning(f"Failed to create score test visualization: {e}")
        
        # LRT results with full report
        if 'lrt' in self.hypothesis_results:
            lrt_result = self.hypothesis_results['lrt']
            lrt_report = LikelihoodRatioTest.generate_publication_report(
                lrt_result,
                test_name="Likelihood Ratio Test",
                hypothesis="biodiversity improves model fit",
                null_model_name="Climate-only model",
                full_model_name="Climate + Biodiversity model"
            )
            report['tests']['lrt'] = lrt_report
            
            # Generate LRT visualization if output directory provided
            if output_dir:
                try:
                    fig = LikelihoodRatioTest.visualize_lrt(
                        lrt_result,
                        null_model_name="Null",
                        full_model_name="Full"
                    )
                    viz_path = output_dir / "method1_lrt_test.png"
                    fig.savefig(viz_path, dpi=300, bbox_inches='tight')
                    report['visualizations'].append(str(viz_path))
                    logger.info(f"LRT visualization saved to {viz_path}")
                except Exception as e:
                    logger.warning(f"Failed to create LRT visualization: {e}")
        
        # Wald test results with visualization
        if 'plants_effect' in self.hypothesis_results or 'vertebrates_effect' in self.hypothesis_results:
            # Combine Wald test results
            wald_results = {
                'coefficients': {},
                'std_errors': {},
                'p_values': {}
            }
            
            if 'plants_effect' in self.hypothesis_results:
                wald_results['coefficients']['plants_richness_log'] = \
                    self.hypothesis_results['plants_effect'].get('coefficient')
                wald_results['std_errors']['plants_richness_log'] = \
                    self.hypothesis_results['plants_effect'].get('std_error')
                wald_results['plants_richness_log_p_value'] = \
                    self.hypothesis_results['plants_effect'].get('p_value')
            
            if 'vertebrates_effect' in self.hypothesis_results:
                wald_results['coefficients']['vertebrates_richness_log'] = \
                    self.hypothesis_results['vertebrates_effect'].get('coefficient')
                wald_results['std_errors']['vertebrates_richness_log'] = \
                    self.hypothesis_results['vertebrates_effect'].get('std_error')
                wald_results['vertebrates_richness_log_p_value'] = \
                    self.hypothesis_results['vertebrates_effect'].get('p_value')
            
            # Generate Wald test report
            wald_report = WaldTest.generate_publication_report(
                wald_results,
                test_name="Wald Test",
                hypothesis="individual biodiversity effects",
                variable_names=['plants_richness_log', 'vertebrates_richness_log']
            )
            report['tests']['wald_test'] = wald_report
            
            # Generate visualization if output directory provided
            if output_dir:
                try:
                    fig = WaldTest.visualize_wald_test(
                        wald_results,
                        variable_names=['plants_richness_log', 'vertebrates_richness_log'],
                        title="Biodiversity Effects (Wald Test)"
                    )
                    viz_path = output_dir / "method1_wald_test.png"
                    fig.savefig(viz_path, dpi=300, bbox_inches='tight')
                    report['visualizations'].append(str(viz_path))
                    logger.info(f"Wald test visualization saved to {viz_path}")
                except Exception as e:
                    logger.warning(f"Failed to create Wald test visualization: {e}")
        
        # Model comparison using GLMComparison
        if self.model_null and self.model_full:
            try:
                # Compare models using information criteria
                comparison_results = ModelComparison.compare_models(
                    {'Null (Climate-only)': self.model_null, 
                     'Full (Climate + Biodiversity)': self.model_full},
                    criteria=['aic', 'bic']
                )
                report['model_comparison'] = comparison_results
                
                # Generate comparison visualization if output directory provided
                if output_dir:
                    try:
                        fig = ModelComparison.visualize_model_comparison(comparison_results)
                        viz_path = output_dir / "method1_model_comparison.png"
                        fig.savefig(viz_path, dpi=300, bbox_inches='tight')
                        report['visualizations'].append(str(viz_path))
                        logger.info(f"Model comparison visualization saved to {viz_path}")
                    except Exception as e:
                        logger.warning(f"Failed to create model comparison visualization: {e}")
                
                # Generate publication report for model comparison
                comparison_report = ModelComparison.generate_publication_report(
                    comparison_results,
                    study_description="biodiversity effects on fungi richness",
                    hypothesis="Biodiversity begets biodiversity"
                )
                report['model_comparison_report'] = comparison_report
            except Exception as e:
                logger.warning(f"Model comparison failed: {e}")
        
        # Overall conclusion
        report['conclusion'] = {
            'supports_h1': self.hypothesis_results.get('supports_h1', False),
            'interpretation': self.hypothesis_results.get('interpretation', '')
        }
        
        # Save report summary if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save JSON summary
            import json
            summary_path = output_dir / "method1_report_summary.json"
            with open(summary_path, 'w') as f:
                # Convert non-serializable objects
                report_serializable = self._make_serializable(report)
                json.dump(report_serializable, f, indent=2)
            logger.info(f"Method 1 report summary saved to {summary_path}")
        
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