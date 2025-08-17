"""Method 2: H2 Hypothesis Testing with Both Continuous and Categorical Models (Julia-based).

Tests H2: Climate-diversity interaction differs in temperate vs non-temperate regions.
- Model 3a: Continuous interaction (bio4 × plants_richness) - seasonality gradient
- Model 3b: Categorical interaction (temperate × plant_high) - binary temperate regions
This refactored version works with data stored in Julia.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from scipy import stats

# No base class - each method stands alone
from ..config_loader import MLRegressionConfig
from ..utils import JuliaManager
from ..evaluation.emmeans_julia import JuliaEMmeans
from ..hypothesis import (
    WaldTest, LikelihoodRatioTest, ScoreTest,
    ContrastAnalysis, SimpleSlopesAnalysis
)
from ..hypothesis.wald_test import WaldTest  # Ensure we have the enhanced WaldTest
from ..hypothesis.h0_definitions import H0Definition
from ..hypothesis.tost_wrapper import TOSTWrapper

logger = logging.getLogger(__name__)


class Method2H2Testing:
    """Method 2: Tests H2 hypothesis using both continuous and categorical interactions.
    
    Model 3a: Continuous interaction (bio4 × plants_richness) - seasonality gradient
    Model 3b: Categorical interaction (temperate × plant_high) - binary categories
    
    Data remains in Julia throughout the analysis.
    """
    
    def __init__(self, config: MLRegressionConfig) -> None:
        """Initialize Method 2.
        
        Args:
            config: ML regression configuration
        """
        self.config = config
        # Get config dict once, not config.config nonsense
        config_dict = config.config if hasattr(config, 'config') else {}
        self.glm_config = config_dict.get('models', {}).get('glm', {})
        
        self.julia = JuliaManager.get_or_init()
        self.model_3a_continuous = None  # Model 3a: Continuous interaction
        self.model_3b_categorical = None  # Model 3b: Categorical interaction
        self.data_metadata = None  # Metadata about data in Julia
        self.emmeans_results = None
        self.contrast_results = None
    
    def fit_models(self, data_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Fit both continuous and categorical interaction models for H2 testing.
        
        Args:
            data_metadata: Metadata about data loaded in Julia
            
        Returns:
            Model fitting results with hypothesis tests
        """
        logger.info("Fitting Method 2 models for H2 hypothesis testing")
        
        # Store metadata
        self.data_metadata = data_metadata
        
        # Get data summary from Julia
        data_summary = self.julia.get_data_summary()
        
        # Check if categorical features have been added
        if not data_summary.get('has_categorical', False):
            logger.info("Adding categorical features in Julia for Method 2")
            cat_result = self.julia.add_categorical_features()
            logger.info(f"Added features: {cat_result['features']}")
            # Update data summary
            data_summary = self.julia.get_data_summary()
        
        # Get MESF columns from Julia data
        mesf_cols = data_summary.get('mesf_columns', [])
        logger.info(f"Using {len(mesf_cols)} MESF eigenvectors")
        
        # Get all columns
        columns = data_summary['columns']
        
        # ========== Model 3a: Continuous Interaction ==========
        logger.info("Fitting Model 3a: Continuous interaction (bio4 × plants_richness)")
        
        # Features for Model 3a (continuous interaction already added in Julia)
        climate_cols = ['bio01_std', 'bio04_std', 'bio12_std', 'bio15_std']
        elevation_cols = ['elevation_std', 'elevation_std_sq']
        biodiv_cols = ['plants_richness_log', 'vertebrates_richness_log']
        interaction_3a = ['bio04_std_x_plants_log']  # Added by Julia
        
        feature_cols_3a = climate_cols + elevation_cols + biodiv_cols + interaction_3a + mesf_cols
        
        # Create formula for Julia GLM
        formula_3a = f"fungi_richness ~ {' + '.join(feature_cols_3a)}"
        
        # Fit model in Julia
        self.model_3a_continuous = self.julia.fit_glm(formula=formula_3a)
        logger.info(f"Model 3a fitted: AIC={self.model_3a_continuous['aic']:.2f}")
        
        # ========== Model 3b: Categorical Interaction ==========
        logger.info("Fitting Model 3b: Categorical interaction (temperate × plant_high)")
        
        # Features for Model 3b
        # NOTE: Exclude bio04_std to avoid collinearity with 'temperate'
        climate_cols_3b = ['bio01_std', 'bio12_std', 'bio15_std']
        categorical_cols = ['temperate', 'plant_high', 'temperate_x_plant_high']
        
        feature_cols_3b = climate_cols_3b + elevation_cols + biodiv_cols + categorical_cols + mesf_cols
        
        # Create formula for Julia GLM
        formula_3b = f"fungi_richness ~ {' + '.join(feature_cols_3b)}"
        
        # Fit model in Julia
        self.model_3b_categorical = self.julia.fit_glm(formula=formula_3b)
        logger.info(f"Model 3b fitted: AIC={self.model_3b_categorical['aic']:.2f}")
        
        # ========== Hypothesis Testing ==========
        results = {}
        
        # Test Model 3a: Continuous interaction
        results['model_3a_continuous'] = self._test_continuous_interaction(
            self.model_3a_continuous,
            interaction_term='bio04_std_x_plants_log',
            feature_cols_3a=feature_cols_3a
        )
        
        # Test Model 3b: Categorical interaction
        results['model_3b_categorical'] = self._test_categorical_interaction(
            self.model_3b_categorical,
            feature_cols_3b=feature_cols_3b
        )
        
        # Model comparison
        results['model_comparison'] = self._compare_models()
        
        return results
    
    def _test_continuous_interaction(
        self, 
        model_results: Dict[str, Any],
        interaction_term: str,
        feature_cols_3a: List[str]
    ) -> Dict[str, Any]:
        """Test continuous interaction for H2.
        
        Args:
            model_results: Julia GLM results
            interaction_term: Name of interaction coefficient
            feature_cols_3a: List of feature names
            
        Returns:
            Test results for continuous interaction
        """
        results = {}
        
        # Get coefficients and vcov from Julia results
        coef_values = model_results['coefficients']
        coef_names = ['(Intercept)'] + feature_cols_3a
        vcov_matrix = model_results['vcov']
        std_errors = model_results['std_errors']
        
        # Find interaction coefficient
        if interaction_term in feature_cols_3a:
            idx = feature_cols_3a.index(interaction_term) + 1  # +1 for intercept
            interaction_coef = coef_values[idx]
            interaction_se = std_errors[idx]
            
            # Wald test for interaction
            z_stat = interaction_coef / interaction_se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            results['wald_test'] = {
                'coefficient': interaction_coef,
                'std_error': interaction_se,
                'z_statistic': z_stat,
                'p_value': p_value,
                'significant_at_0.05': p_value < 0.05,
                'interpretation': 'Biodiversity effect varies with temperature seasonality' if p_value < 0.05 
                                else 'No evidence of seasonality-dependent biodiversity effect'
            }
            
            # Effect size interpretation
            results['effect_size'] = {
                'coefficient': interaction_coef,
                'interpretation': self._interpret_interaction_effect(interaction_coef)
            }
        else:
            results['wald_test'] = {'error': f'Interaction term {interaction_term} not found'}
        
        # Model statistics
        results['model_stats'] = {
            'aic': model_results['aic'],
            'bic': model_results['bic'],
            'deviance': model_results['deviance'],
            'df_residual': model_results['df_residual'],
            'n_obs': model_results['n_obs']
        }
        
        return results
    
    def _test_categorical_interaction(
        self,
        model_results: Dict[str, Any],
        feature_cols_3b: List[str]
    ) -> Dict[str, Any]:
        """Test categorical interaction for H2.
        
        Args:
            model_results: Julia GLM results
            feature_cols_3b: List of feature names
            
        Returns:
            Test results for categorical interaction
        """
        results = {}
        
        # Get coefficients from Julia results
        coef_values = model_results['coefficients']
        coef_names = ['(Intercept)'] + feature_cols_3b
        std_errors = model_results['std_errors']
        
        # Find interaction coefficient
        interaction_term = 'temperate_x_plant_high'
        if interaction_term in feature_cols_3b:
            idx = feature_cols_3b.index(interaction_term) + 1  # +1 for intercept
            interaction_coef = coef_values[idx]
            interaction_se = std_errors[idx]
            
            # Wald test for interaction
            z_stat = interaction_coef / interaction_se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            results['interaction_test'] = {
                'coefficient': interaction_coef,
                'std_error': interaction_se,
                'z_statistic': z_stat,
                'p_value': p_value,
                'significant_at_0.05': p_value < 0.05,
                'interpretation': 'Plant diversity effect differs between temperate/non-temperate' if p_value < 0.05
                                else 'Plant diversity effect consistent across regions'
            }
        
        # Model statistics
        results['model_stats'] = {
            'aic': model_results['aic'],
            'bic': model_results['bic'],
            'deviance': model_results['deviance'],
            'df_residual': model_results['df_residual'],
            'n_obs': model_results['n_obs']
        }
        
        return results
    
    def _compare_models(self) -> Dict[str, Any]:
        """Compare Model 3a and 3b.
        
        Returns:
            Model comparison results
        """
        aic_3a = self.model_3a_continuous['aic']
        aic_3b = self.model_3b_categorical['aic']
        bic_3a = self.model_3a_continuous['bic']
        bic_3b = self.model_3b_categorical['bic']
        
        return {
            'best_model_aic': '3a_continuous' if aic_3a < aic_3b else '3b_categorical',
            'best_model_bic': '3a_continuous' if bic_3a < bic_3b else '3b_categorical',
            'aic_difference': abs(aic_3a - aic_3b),
            'bic_difference': abs(bic_3a - bic_3b),
            'model_3a': {
                'aic': aic_3a,
                'bic': bic_3a
            },
            'model_3b': {
                'aic': aic_3b,
                'bic': bic_3b
            }
        }
    
    def _interpret_interaction_effect(self, coef: float) -> str:
        """Interpret the magnitude of interaction effect.
        
        Args:
            coef: Interaction coefficient
            
        Returns:
            String interpretation
        """
        if abs(coef) < 0.01:
            return "Negligible interaction"
        elif abs(coef) < 0.05:
            return "Weak interaction"
        elif abs(coef) < 0.1:
            return "Moderate interaction"
        else:
            return "Strong interaction"
    
    def compute_emmeans(self) -> Dict[str, Any]:
        """Compute estimated marginal means for categorical model.
        
        Returns:
            EMmeans results from Julia model
        """
        if self.model_3b_categorical is None:
            return {'error': 'Must fit models first'}
        
        # Get feature names for model 3b
        data_summary = self.julia.get_data_summary()
        columns = data_summary['columns']
        mesf_cols = data_summary.get('mesf_columns', [])
        
        climate_cols_3b = ['bio01_std', 'bio12_std', 'bio15_std']
        elevation_cols = ['elevation_std', 'elevation_std_sq']
        biodiv_cols = ['plants_richness_log', 'vertebrates_richness_log']
        categorical_cols = ['temperate', 'plant_high', 'temperate_x_plant_high']
        
        feature_cols_3b = climate_cols_3b + elevation_cols + biodiv_cols + categorical_cols + mesf_cols
        
        # Create EMmeans calculator
        emmeans_calc = JuliaEMmeans(self.model_3b_categorical, feature_cols_3b)
        
        # Compute EMmeans for the 2x2 design
        emmeans_df = emmeans_calc.compute_emmeans(
            factors={'temperate': [0, 1], 'plant_high': [0, 1]},
            type='response'  # Back-transform to response scale
        )
        
        # Compute contrasts
        contrasts_df = emmeans_calc.compute_contrasts(emmeans_df, contrast_type='interaction')
        
        self.emmeans_results = {
            'emmeans': emmeans_df.to_dict('records'),
            'contrasts': contrasts_df.to_dict('records')
        }
        
        logger.info(f"Computed EMmeans for {len(emmeans_df)} factor combinations")
        
        return self.emmeans_results
    
    def test_h2_contrast(self) -> Dict[str, Any]:
        """Test the H2 contrast.
        
        Returns:
            H2 test results
        """
        # Use the categorical model results
        if self.model_3b_categorical is None:
            return {'error': 'Must fit models first'}
        
        # Get interaction coefficient
        coef_names = self.model_3b_categorical.get('coef_names', [])
        coef_values = self.model_3b_categorical['coefficients']
        std_errors = self.model_3b_categorical['std_errors']
        
        # Find interaction term
        interaction_idx = None
        for i, name in enumerate(coef_names):
            if 'temperate_x_plant_high' in name:
                interaction_idx = i
                break
        
        if interaction_idx is not None:
            interaction_coef = coef_values[interaction_idx]
            interaction_se = std_errors[interaction_idx]
            
            # Test statistic
            z_stat = interaction_coef / interaction_se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            return {
                'h2_supported': p_value < 0.05 and interaction_coef < 0,
                'contrast_estimate': interaction_coef,
                'std_error': interaction_se,
                'z_statistic': z_stat,
                'p_value': p_value,
                'interpretation': self._interpret_h2_result(interaction_coef, p_value)
            }
        
        return {'error': 'Interaction term not found'}
    
    def _interpret_h2_result(self, coef: float, p_value: float) -> str:
        """Interpret H2 test result.
        
        Args:
            coef: Interaction coefficient
            p_value: P-value
            
        Returns:
            String interpretation
        """
        if p_value >= 0.05:
            return "No significant interaction - H2 not supported"
        elif coef < 0:
            return "Negative interaction - Plant diversity effect weaker in temperate regions (H2 supported)"
        else:
            return "Positive interaction - Plant diversity effect stronger in temperate regions (H2 not supported)"
    
    def fit_model(self, data_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper for compatibility - calls fit_models."""
        return self.fit_models(data_metadata)
    
    def predict(self, **kwargs) -> np.ndarray:
        """Make predictions using fitted model.
        
        Note: Prediction happens in Julia on already loaded data.
        
        Args:
            **kwargs: Additional prediction parameters
            
        Returns:
            Predicted values (placeholder)
        """
        if self.model_3b_categorical is None:
            raise ValueError("Must fit models first")
        
        logger.warning("Prediction on new data not yet implemented for Julia models")
        return self.model_3b_categorical['coefficients']
    
    def generate_report(self, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive report for Method 2.
        
        Args:
            output_dir: Optional directory to save report files
            
        Returns:
            Dictionary with all report components
        """
        report = {
            'method': 'Method 2: H2 Testing (Temperate Mismatch)',
            'models': {
                '3a_continuous': self.model_3a_continuous,
                '3b_categorical': self.model_3b_categorical
            },
            'hypothesis': 'H2: Biodiversity effect differs in temperate vs non-temperate regions',
            'emmeans': self.emmeans_results,
            'contrast': self.contrast_results
        }
        
        # Save if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            import json
            report_file = output_dir / 'method2_h2_report.json'
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_arrays(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_arrays(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_arrays(v) for v in obj]
                return obj
            
            with open(report_file, 'w') as f:
                json.dump(convert_arrays(report), f, indent=2)
            
            logger.info(f"Method 2 report saved to {report_file}")
        
        return report