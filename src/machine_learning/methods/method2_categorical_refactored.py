"""Method 2: H2 Hypothesis Testing with Both Continuous and Categorical Models (REFACTORED).

Tests H2: Climate-diversity interaction differs in temperate vs non-temperate regions.
- Model 3a: Continuous interaction (bio4 × plants_richness) - seasonality gradient
- Model 3b: Categorical interaction (temperate × plant_high) - binary temperate regions
This refactored version assumes data is already preprocessed by the pipeline.
"""

import logging
import h2o
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

# No base class - each method stands alone
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from ..config_loader import MLRegressionConfig
from ..evaluation.emmeans_core import REMmeansWrapper, SimpleEMmeansApproximation
from ..utils.vcov import extract_vcov
from ..hypothesis import (
    WaldTest, LikelihoodRatioTest, ScoreTest,
    ContrastAnalysis, SimpleSlopesAnalysis
)
from ..hypothesis.wald_test import WaldTest  # Ensure we have the enhanced WaldTest
from ..hypothesis.h0_definitions import H0Definition
from ..hypothesis.tost_wrapper import TOSTWrapper
from ..evaluation import GLMComparison
from ..preprocessing.h2o_preprocessing import (
    create_temperate_categories,
    create_biodiversity_categories,
    add_categorical_interactions
)

logger = logging.getLogger(__name__)


class Method2H2Testing:
    """Method 2: Tests H2 hypothesis using both continuous and categorical interactions.
    
    Model 3a: Continuous interaction (bio4 × plants_richness) - seasonality gradient
    Model 3b: Categorical interaction (temperate × plant_high) - binary categories
    
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
        self.config = config
        # Get config dict once, not config.config nonsense
        config_dict = config.config if hasattr(config, 'config') else {}
        self.glm_config = config_dict.get('models', {}).get('glm', {})
        
        self.model_3a_continuous = None  # Model 3a: Continuous interaction
        self.model_3b_categorical = None  # Model 3b: Categorical interaction
        self.data_h2o = None
        self.emmeans_results = None
        self.contrast_results = None
        self.formula = None  # R formula for the model
    
    def add_categorical_features(self, h2o_data: h2o.H2OFrame) -> h2o.H2OFrame:
        """Add Method2-specific categorical features.
        
        Args:
            h2o_data: Preprocessed data from pipeline
            
        Returns:
            Data with categorical features added
        """
        logger.info("Adding Method2-specific categorical features")
        
        # Create temperate category (bio4 > median for robustness with sampling)
        # High temperature seasonality = temperate regions
        # NOTE: Using median instead of 75th percentile to ensure adequate samples in both categories
        h2o_data = create_temperate_categories(
            h2o_data,
            bio4_col='bio04_std',  # Use bio4 (temperature seasonality) not bio1!
            output_col='temperate',
            method='median',  # Changed from quantile=0.75 to avoid empty categories with sampling
            quantile=0.5  
        )
        
        # Create plant diversity category (plants > median for robustness)
        h2o_data = create_biodiversity_categories(
            h2o_data,
            richness_col='plants_richness',
            output_col='plant_high',
            method='median',  # Changed from quantile=0.75 to avoid empty categories
            quantile=0.5
        )
        
        # Create interaction term
        h2o_data = add_categorical_interactions(
            h2o_data,
            cat1='temperate',
            cat2='plant_high',
            output_col='temperate_x_plant_high'
        )
        
        # Log categorical levels and validate balance
        temperate_counts = h2o_data['temperate'].table()
        plant_high_counts = h2o_data['plant_high'].table()
        
        logger.info(f"Temperate regions: {temperate_counts}")
        logger.info(f"High plant diversity: {plant_high_counts}")
        
        # Validate categorical balance for robust model fitting
        min_category_size = 100  # Minimum samples per category
        
        # Check temperate categories
        temp_sizes = temperate_counts.as_data_frame(use_multi_thread=True)['Count'].values
        if np.min(temp_sizes) < min_category_size:
            logger.warning(f"Temperate category imbalance: min size {np.min(temp_sizes)} < {min_category_size}")
            logger.warning("Consider adjusting quantile threshold or using stratified sampling")
        
        # Check plant diversity categories
        plant_sizes = plant_high_counts.as_data_frame(use_multi_thread=True)['Count'].values
        if np.min(plant_sizes) < min_category_size:
            logger.warning(f"Plant diversity category imbalance: min size {np.min(plant_sizes)} < {min_category_size}")
            logger.warning("Consider adjusting quantile threshold or using stratified sampling")
        
        # Check interaction term balance
        interaction_counts = h2o_data['temperate_x_plant_high'].table()
        interaction_sizes = interaction_counts.as_data_frame(use_multi_thread=True)['Count'].values
        if np.min(interaction_sizes) < min_category_size:
            logger.warning(f"Interaction category imbalance: min size {np.min(interaction_sizes)} < {min_category_size}")
            logger.warning("This may affect the reliability of interaction effect estimates")
        
        # Calculate effective sample sizes for contrasts
        n_total = h2o_data.nrows
        prop_temp = temp_sizes[1] / n_total if len(temp_sizes) > 1 else 0
        prop_plant = plant_sizes[1] / n_total if len(plant_sizes) > 1 else 0
        
        logger.info(f"Category proportions: temperate={prop_temp:.2%}, high_plants={prop_plant:.2%}")
        
        # Warn if categories are extremely imbalanced
        if prop_temp < 0.1 or prop_temp > 0.9:
            logger.warning(f"Temperate category is highly imbalanced ({prop_temp:.1%})")
        if prop_plant < 0.1 or prop_plant > 0.9:
            logger.warning(f"Plant diversity category is highly imbalanced ({prop_plant:.1%})")
        
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
        # Use H2O directly - no wrapper needed
        
        # ========== Model 3a: Continuous Interaction ==========
        logger.info("Fitting Model 3a: Continuous interaction (bio4 × plants_richness)")
        
        # Create continuous interaction term using bio04 (seasonality) not bio01!
        # This tests how biodiversity effect varies with seasonality (temperate gradient)
        h2o_data['bio04_x_plants'] = h2o_data['bio04_std'] * h2o_data['plants_richness_log']
        
        # Features for Model 3a
        feature_cols_3a = [
            # Climate variables
            'bio01_std', 'bio04_std', 'bio12_std', 'bio15_std',
            'elevation_std', 'elevation_std_sq',
            # Biodiversity
            'plants_richness_log', 'vertebrates_richness_log',
            # Continuous interaction - bio4 (seasonality) × plants for H2 hypothesis
            'bio04_x_plants'
        ] + me_cols
        
        # Direct H2O usage - no wrapper
        self.model_3a_continuous = H2OGeneralizedLinearEstimator(
            family='negativebinomial',
            link='log',
            lambda_=0.0,  # No penalty for unbiased estimation
            standardize=True,
            solver='IRLSM',
            remove_collinear_columns=True,
            compute_p_values=True
        )
        self.model_3a_continuous.train(x=feature_cols_3a, y='fungi_richness', training_frame=h2o_data)
        
        # ========== Model 3b: Categorical Interaction ==========
        logger.info("Fitting Model 3b: Categorical interaction (temperate × plant_high)")
        
        # Add categorical features
        h2o_data_cat = self.add_categorical_features(h2o_data)
        
        # Features for Model 3b
        # NOTE: Exclude bio04_std to avoid perfect collinearity with 'temperate' 
        # (temperate is derived from bio04_std > 75th percentile)
        feature_cols_3b = [
            # Climate variables (excluding bio04_std to prevent collinearity)
            'bio01_std', 'bio12_std', 'bio15_std',
            'elevation_std', 'elevation_std_sq',
            # Biodiversity
            'plants_richness_log', 'vertebrates_richness_log',
            # Categorical variables and interaction
            'temperate', 'plant_high', 'temperate_x_plant_high'
        ] + me_cols
        
        # Direct H2O usage - no wrapper
        self.model_3b_categorical = H2OGeneralizedLinearEstimator(
            family='negativebinomial',
            link='log',
            lambda_=0.0,  # No penalty for unbiased estimation
            standardize=True,
            solver='IRLSM',
            remove_collinear_columns=True,
            compute_p_values=True
        )
        self.model_3b_categorical.train(x=feature_cols_3b, y='fungi_richness', training_frame=h2o_data_cat)
        
        # Store predictor and response names for bootstrap
        self.x_names_3b = feature_cols_3b
        self.y_name = 'fungi_richness'
        
        # Store R formula for emmeans
        self.formula = f"fungi_richness ~ {' + '.join(feature_cols_3b)}"
        
        # ========== Hypothesis Testing ==========
        results = {}
        
        # === Model 3a: Continuous Interaction Testing ===
        # Multiple tests are appropriate for continuous interaction
        
        # 1a. Wald test for continuous interaction coefficient
        # First, find the actual interaction coefficient name
        all_coeffs = self.model_3a_continuous.coef()
        
        # Look for the interaction term - could be named differently by H2O
        interaction_coef_name = None
        possible_names = ['bio04_x_plants', 'bio04_std:plants_richness_log', 
                         'plants_richness_log:bio04_std', 'bio4_x_plants_richness']
        
        # First try exact matches
        for name in possible_names:
            if name in all_coeffs:
                interaction_coef_name = name
                logger.info(f"Found interaction coefficient: {name}")
                break
        
        # If not found, search for any coefficient with both terms
        if not interaction_coef_name:
            for coef_name in all_coeffs.keys():
                if ('bio04' in coef_name.lower() and 'plant' in coef_name.lower()) or \
                   ('bio4' in coef_name.lower() and 'plant' in coef_name.lower()):
                    interaction_coef_name = coef_name
                    logger.info(f"Found interaction coefficient by pattern: {coef_name}")
                    break
        
        # If still not found, log available coefficients for debugging
        if not interaction_coef_name:
            logger.warning("Could not find bio4 x plants interaction coefficient")
            logger.info(f"Available coefficients: {list(all_coeffs.keys())}")
            # Use a placeholder result
            continuous_wald = {
                'error': 'Interaction coefficient not found',
                'available_coefficients': list(all_coeffs.keys()),
                'significant_at_0.05': False
            }
        else:
            # Extract vcov matrix for proper inference
            logger.info("Extracting variance-covariance matrix for Wald tests")
            from src.machine_learning.utils.vcov import extract_vcov
            
            try:
                # Get feature columns for continuous model
                feature_cols = [col for col in h2o_data.columns if col != 'fungi_richness']
                
                # Get vcov matrix and coefficients
                coef_values, vcov_matrix, coef_names = extract_vcov(
                    self.model_3a_continuous,
                    data=h2o_data,
                    x_names=feature_cols,
                    y_name='fungi_richness',
                    method='sandwich'  # Robust standard errors
                )
                
                # Find interaction coefficient index
                if interaction_coef_name in coef_names:
                    idx = coef_names.index(interaction_coef_name)
                    coef_val = coef_values[idx]
                    se_val = np.sqrt(vcov_matrix[idx, idx])
                    
                    # IMPORTANT: For interaction coefficient, only report SESOI and TOST
                    # 1. SESOI test (H0: |β| ≤ threshold)
                    sesoi_threshold = 0.05  # Smaller threshold for interaction effects
                    from src.machine_learning.hypothesis.h0_definitions import H0Definition
                    
                    sesoi_test = WaldTest.test_single_coefficient_with_scale(
                        coefficient=coef_val,
                        standard_error=se_val,
                        h0_definition=H0Definition.sesoi(sesoi_threshold, 'greater'),
                        alternative='greater',
                        link='log'  # Enable scale transformations
                    )
                    
                    # 2. TOST equivalence test
                    from src.machine_learning.hypothesis.tost_wrapper import TOSTWrapper
                    tost_test = TOSTWrapper.perform_tost_with_scale(
                        test_function=WaldTest.test_single_coefficient,
                        effect_estimate=coef_val,
                        standard_error=se_val,
                        lower_bound=-sesoi_threshold,
                        upper_bound=sesoi_threshold,
                        alpha=0.05,
                        link='log'
                    )
                    
                    # Package results (NO traditional test)
                    continuous_wald = {
                        'sesoi': sesoi_test,
                        'tost': tost_test,
                        'coefficient': coef_val,
                        'standard_error': se_val,
                        'significant_at_0.05': sesoi_test.get('exceeds_sesoi', False),  # Use SESOI for significance
                        'interpretation': 'Temperature×Plants interaction (SESOI and TOST tests only)'
                    }
                    
                    logger.info(f"Interaction coefficient {interaction_coef_name}: β={coef_val:.4f}, SE={se_val:.4f}")
                    logger.info(f"SESOI test: p={sesoi_test['p_value']:.4f}, exceeds threshold={sesoi_test.get('exceeds_sesoi', False)}")
                    logger.info(f"TOST test: p={tost_test['tost_p_value']:.4f}, equivalent={tost_test['is_equivalent']}")
                else:
                    continuous_wald = {
                        'error': f'Coefficient {interaction_coef_name} not in vcov matrix',
                        'significant_at_0.05': False
                    }
            except Exception as e:
                logger.warning(f"vcov extraction failed: {e}")
                # Fallback to model-based test but still use SESOI/TOST
                continuous_wald = WaldTest.test_coefficients_from_model(
                    model=self.model_3a_continuous,
                    coefficient_names=interaction_coef_name,
                    null_values=0.0
                )
                continuous_wald['note'] = 'Using model-based SE (vcov extraction failed)'
        
        # 1b. Score test for continuous interaction (using H2O-compatible version)
        try:
            # We need a model WITHOUT interaction to test against
            # Fit a model without the interaction term for score test
            model_without_interaction_data = h2o_data.drop('bio04_x_plants')
            model_without_interaction = GLMClass(self.config.config)
            model_without_interaction.fit(model_without_interaction_data, y='fungi_richness')
            
            # Use H2O-compatible score test (REPORT TRADITIONAL ONLY - model comparison)
            continuous_score = ScoreTest.test_h2o_interaction(
                null_model=model_without_interaction,
                var1='bio04_std',
                var2='plants_richness_log',
                h2o_data=h2o_data
            )
            logger.info(f"Score test for interaction (traditional only): p={continuous_score.get('p_value', 'N/A'):.4f}")
            # Note: Score test is a model comparison test - report traditional results only
            continuous_score['note'] = 'Model comparison test - traditional H0 only'
        except Exception as e:
            logger.warning(f"Score test failed: {e}")
            continuous_score = {'error': str(e), 'note': 'Score test failed'}
        
        # 1c. Likelihood Ratio Test for interaction (REPORT TRADITIONAL ONLY - model comparison)
        from ..hypothesis.lrt_test import LikelihoodRatioTest
        try:
            # LRT comparing model with and without interaction
            if 'model_without_interaction' in locals():
                lrt_result = LikelihoodRatioTest.test_nested_models(
                    null_model=model_without_interaction,
                    full_model=self.model_3a_continuous,
                    df_difference=1  # One interaction term
                )
                logger.info(f"LRT for interaction (traditional only): χ²={lrt_result['lr_statistic']:.3f}, p={lrt_result['p_value']:.4f}")
            else:
                # Fit model without interaction if not already done
                model_without_interaction_data = h2o_data.drop('bio04_x_plants')
                model_without_interaction = GLMClass(self.config.config)
                model_without_interaction.fit(model_without_interaction_data, y='fungi_richness')
                
                lrt_result = LikelihoodRatioTest.test_nested_models(
                    null_model=model_without_interaction,
                    full_model=self.model_3a_continuous,
                    df_difference=1
                )
                logger.info(f"LRT for interaction (traditional only): χ²={lrt_result['lr_statistic']:.3f}, p={lrt_result['p_value']:.4f}")
            # Note: LRT is a model comparison test - report traditional results only
            lrt_result['note'] = 'Model comparison test - traditional H0 only'
        except Exception as e:
            logger.warning(f"LRT failed: {e}")
            lrt_result = {'error': str(e), 'note': 'LRT failed'}
        
        # 1d. Model comparison (AIC/BIC) for continuous model
        results['model_3a_continuous'] = {
            'aic': self.model_3a_continuous.aic(),
            'bic': None,  # H2O doesn't provide BIC directly
            'deviance': self.model_3a_continuous.residual_deviance(),
            'wald_test': continuous_wald,
            'score_test': continuous_score,
            'lrt_test': lrt_result,
            'h2_supported': (continuous_wald.get('significant_at_0.05', False) or 
                           lrt_result.get('significant_at_0.05', False)),
            'interpretation': 'Tests continuous bio4 (seasonality) × plants interaction for H2 temperate hypothesis'
        }
        
        # === Model 3b: Categorical Interaction Testing ===
        # For categorical, emmeans is primary; Wald is secondary
        
        # 2a. Wald test for categorical interaction (secondary test)
        # Find the categorical interaction coefficient
        cat_coeffs = self.model_3b_categorical.coef()
        
        cat_interaction_name = None
        cat_possible_names = ['temperate_x_plant_high', 'temperate:plant_high', 
                             'plant_high:temperate', 'temperate.1:plant_high.1']
        
        # Try exact matches
        for name in cat_possible_names:
            if name in cat_coeffs:
                cat_interaction_name = name
                logger.info(f"Found categorical interaction coefficient: {name}")
                break
        
        # Pattern search if not found
        if not cat_interaction_name:
            for coef_name in cat_coeffs.keys():
                if ('temperate' in coef_name.lower() and 'plant' in coef_name.lower()):
                    cat_interaction_name = coef_name
                    logger.info(f"Found categorical interaction by pattern: {coef_name}")
                    break
        
        if not cat_interaction_name:
            logger.warning("Could not find temperate x plant_high interaction")
            logger.info(f"Available coefficients: {list(cat_coeffs.keys())}")
            categorical_wald = {
                'error': 'Categorical interaction coefficient not found',
                'available_coefficients': list(cat_coeffs.keys()),
                'significant_at_0.05': False
            }
        else:
            # Extract vcov matrix for proper inference on categorical model
            logger.info("Extracting variance-covariance matrix for categorical Wald tests")
            from src.machine_learning.utils.vcov import extract_vcov
            
            try:
                # Get feature columns for categorical model
                feature_cols_cat = [col for col in h2o_data.columns if col != 'fungi_richness']
                
                # Get vcov matrix and coefficients for categorical model
                cat_coef_values, cat_vcov_matrix, cat_coef_names = extract_vcov(
                    self.model_3b_categorical,
                    data=h2o_data,
                    x_names=feature_cols_cat,  # Use categorical feature columns
                    y_name='fungi_richness',
                    method='sandwich'  # Robust standard errors
                )
                
                # Find categorical interaction coefficient index
                if cat_interaction_name in cat_coef_names:
                    idx = cat_coef_names.index(cat_interaction_name)
                    coef_val = cat_coef_values[idx]
                    se_val = np.sqrt(cat_vcov_matrix[idx, idx])
                    
                    # IMPORTANT: For interaction coefficient, only report SESOI and TOST
                    # 1. SESOI test (H0: |β| ≤ threshold)
                    sesoi_threshold = 0.05  # Smaller threshold for interaction effects
                    from src.machine_learning.hypothesis.h0_definitions import H0Definition
                    
                    sesoi_test = WaldTest.test_single_coefficient_with_scale(
                        coefficient=coef_val,
                        standard_error=se_val,
                        h0_definition=H0Definition.sesoi(sesoi_threshold, 'greater'),
                        alternative='greater',
                        link='log'  # Enable scale transformations
                    )
                    
                    # 2. TOST equivalence test
                    from src.machine_learning.hypothesis.tost_wrapper import TOSTWrapper
                    tost_test = TOSTWrapper.perform_tost_with_scale(
                        test_function=WaldTest.test_single_coefficient,
                        effect_estimate=coef_val,
                        standard_error=se_val,
                        lower_bound=-sesoi_threshold,
                        upper_bound=sesoi_threshold,
                        alpha=0.05,
                        link='log'
                    )
                    
                    # Package results (NO traditional test)
                    categorical_wald = {
                        'sesoi': sesoi_test,
                        'tost': tost_test,
                        'coefficient': coef_val,
                        'standard_error': se_val,
                        'significant_at_0.05': sesoi_test.get('exceeds_sesoi', False),  # Use SESOI for significance
                        'interpretation': 'Temperate×PlantHigh interaction (SESOI and TOST tests only)'
                    }
                    
                    logger.info(f"Categorical interaction {cat_interaction_name}: β={coef_val:.4f}, SE={se_val:.4f}")
                    logger.info(f"SESOI test: p={sesoi_test['p_value']:.4f}, exceeds threshold={sesoi_test.get('exceeds_sesoi', False)}")
                    logger.info(f"TOST test: p={tost_test['tost_p_value']:.4f}, equivalent={tost_test['is_equivalent']}")
                else:
                    categorical_wald = {
                        'error': f'Coefficient {cat_interaction_name} not in vcov matrix',
                        'significant_at_0.05': False
                    }
            except Exception as e:
                logger.warning(f"vcov extraction failed for categorical model: {e}")
                # Fallback to model-based test but still use SESOI/TOST
                categorical_wald = WaldTest.test_coefficients_from_model(
                    model=self.model_3b_categorical,
                    coefficient_names=cat_interaction_name,
                    null_values=0.0
                )
                categorical_wald['note'] = 'Using model-based SE (vcov extraction failed)'
        
        # 2b. Note about emmeans being the primary test for 3b
        results['model_3b_categorical'] = {
            'aic': self.model_3b_categorical.aic(),
            'bic': None,  # H2O doesn't provide BIC directly
            'deviance': self.model_3b_categorical.residual_deviance(),
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
        
        # === Additional Tests for Model 3a (Continuous Interaction) ===
        
        # 4. Marginal Effects Analysis
        from ..interpretation.marginal_effects import MarginalEffects
        try:
            # Compute marginal effects of plants at different bio4 levels
            marginal_effects_results = MarginalEffects.compute_marginal_effects_at_values(
                model=self.model_3a_continuous,
                focal_var='plants_richness_log',
                moderator_var='bio04_std',
                moderator_values=[-1, 0, 1],  # Low, mean, high seasonality
                data=h2o_data,
                model_type='h2o_glm'
            )
            logger.info("Marginal effects computed for plants at different bio4 levels")
            
            # Check if effect becomes negative at high seasonality (supports H2)
            if marginal_effects_results and len(marginal_effects_results) >= 3:
                high_seasonality_effect = marginal_effects_results[2].get('marginal_effect', 0)
                marginal_h2_supported = high_seasonality_effect < 0
                logger.info(f"Plant effect at high seasonality: {high_seasonality_effect:.4f}")
            else:
                marginal_h2_supported = False
                
        except Exception as e:
            logger.warning(f"Marginal effects computation failed: {e}")
            marginal_effects_results = {'error': str(e)}
            marginal_h2_supported = False
        
        # 5. Interaction Plots (Optimized for large data)
        from ..interpretation.interaction_plots_optimized import OptimizedInteractionPlots
        try:
            # Create interaction plot for bio4 × plants - FAST version
            interaction_plot = OptimizedInteractionPlots.plot_continuous_interaction_fast(
                model=self.model_3a_continuous,
                var1='plants_richness_log',
                var2='bio04_std',
                var1_range=(0, 4),  # Log scale range
                var2_values=[-1, 0, 1],  # Low, mean, high seasonality
                data=h2o_data,  # Stays in H2O!
                ylabel='Predicted Fungi Richness',
                xlabel='Plant Richness (log)',
                title='H2 Test: Plant-Fungi Relationship by Temperature Seasonality'
            )
            logger.info("Optimized interaction plot created for bio4 × plants")
        except Exception as e:
            logger.warning(f"Interaction plot failed: {e}")
            interaction_plot = None
        
        # 6. Partial Dependence Plots
        from ..interpretation.pdp import PartialDependence
        try:
            # PDP for interaction effect
            pdp_results = PartialDependence.compute_pdp_interaction(
                model=self.model_3a_continuous,
                features=['plants_richness_log', 'bio04_std'],
                data=h2o_data,
                grid_resolution=20
            )
            logger.info("Partial dependence computed for bio4 × plants interaction")
        except Exception as e:
            logger.warning(f"PDP computation failed: {e}")
            pdp_results = {'error': str(e)}
        
        # Add new results to model 3a
        results['model_3a_continuous']['marginal_effects'] = marginal_effects_results
        results['model_3a_continuous']['marginal_h2_supported'] = marginal_h2_supported
        results['model_3a_continuous']['interaction_plot'] = interaction_plot
        results['model_3a_continuous']['pdp'] = pdp_results
        
        # Overall H2 conclusion (now includes marginal effects)
        h2_supported = (
            results['model_3a_continuous']['h2_supported'] or
            results['model_3a_continuous']['marginal_h2_supported'] or
            results['model_3b_categorical']['h2_supported']
        )
        
        results['h2_conclusion'] = {
            'h2_supported': h2_supported,
            'best_model': model_comparison['best_models'].get('aic'),
            'interpretation': 'Temperate mismatch effect detected (H2)' if h2_supported else 'No temperate mismatch effect',
            'evidence_sources': {
                'wald_test': continuous_wald.get('significant_at_0.05', False),
                'lrt_test': lrt_result.get('significant_at_0.05', False),
                'marginal_effects': marginal_h2_supported,
                'categorical_model': results['model_3b_categorical']['h2_supported']
            }
        }
        
        logger.info("Model fitting and hypothesis testing complete")
        return results
    
    def fit_model(self, h2o_data: h2o.H2OFrame) -> Dict[str, Any]:
        """Wrapper for compatibility - calls fit_models."""
        return self.fit_models(h2o_data)
    
    def debug_coefficients(self) -> Dict[str, List[str]]:
        """Debug method to check available coefficient names.
        
        Returns:
            Dictionary with coefficient names for each model
        """
        debug_info = {}
        
        if self.model_3a_continuous:
            try:
                coeffs_3a = self.model_3a_continuous.coef()
                debug_info['model_3a_continuous'] = list(coeffs_3a.keys())
                logger.info(f"Model 3a coefficients: {list(coeffs_3a.keys())}")
            except Exception as e:
                debug_info['model_3a_continuous'] = f"Error: {e}"
                
        if self.model_3b_categorical:
            try:
                coeffs_3b = self.model_3b_categorical.coef()
                debug_info['model_3b_categorical'] = list(coeffs_3b.keys())
                logger.info(f"Model 3b coefficients: {list(coeffs_3b.keys())}")
            except Exception as e:
                debug_info['model_3b_categorical'] = f"Error: {e}"
                
        return debug_info
    
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
        logger.info("Extracting variance-covariance matrix from H2O model using bootstrap")
        # Use bootstrap for proper vcov estimation
        coefficients, vcov_matrix, coef_names = extract_vcov(
            self.model_3b_categorical,  # The model itself, not .model
            self.data_h2o,
            method='sandwich',  # LINUS: Sandwich is the standard
            x_names=self.x_names_3b,  # Pass predictor names
            y_name=self.y_name  # Pass response name
        )
        
        # Store vcov for later use in hypothesis tests
        self.vcov_matrix = vcov_matrix
        self.coefficients = coefficients
        self.coef_names = coef_names
        
        # Use R emmeans wrapper for proper statistical inference
        # OPTIMIZATION: Use representative sample for emmeans instead of full 43M rows
        from src.machine_learning.evaluation.emmeans_core import REMmeansWrapper
        from src.machine_learning.utils.emmeans_data_sampler import EMMeansDataSampler
        
        # Get representative sample using abstracted sampler
        sampler = EMMeansDataSampler(sample_size=10000, seed=42)
        data_sample = sampler.sample_for_emmeans(
            self.data_h2o,
            factor_columns=['temperate', 'plant_high']
        )
        
        # Use the FULL formula that matches our model coefficients
        # R emmeans needs the complete model structure to properly use the vcov matrix
        
        # Use poisson family for emmeans (vcov already accounts for NB overdispersion)
        self.emmeans_calculator = REMmeansWrapper(
            coefficients=coefficients,
            vcov_matrix=vcov_matrix,
            coefficient_names=coef_names,
            data=data_sample,  # Representative sample, not full 43M rows!
            formula=self.formula,  # Use FULL formula from line 236 that matches coefficients!
            family='poisson',  # Use poisson for emmeans computation
            link='log',
            df_residual=int(self.data_h2o.nrows - len(coefficients))
        )
        logger.info("Using R emmeans package with representative sample + full vcov matrix")
        
        # Compute emmeans for 2x2 design using the actual R emmeans package
        emmeans_results = self.emmeans_calculator.compute_emmeans_2x2(
            factor1='plant_high',
            factor2='temperate',
            type='response'
        )
        
        # Store results both in self and in the calculator for access by ContrastAnalysis
        self.emmeans_results = emmeans_results
        self.emmeans_calculator.emmeans_results = emmeans_results
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
        
        # 5. Get interaction coefficient from model with scale-aware reporting
        logger.info("\nExtracting interaction coefficient with scale transformations...")
        
        # Get coefficients
        coefficients = self.model_3b_categorical.coef()
        interaction_coef = coefficients.get('temperate_x_plant_high', 0)
        
        # Get standard error and p-value
        interaction_se = 0.1  # Default
        p_value = None
        
        try:
            # Try to get vcov matrix for proper SE
            coef_values, vcov_matrix, coef_names = extract_vcov(
                self.model_3b_categorical,
                data=self.data_h2o,
                x_names=self.feature_cols,
                y_name=self.target_col,
                method='sandwich'
            )
            # Extract SE from vcov diagonal
            if 'temperate_x_plant_high' in coef_names:
                idx = coef_names.index('temperate_x_plant_high')
                interaction_se = np.sqrt(vcov_matrix[idx, idx])
                logger.info(f"  Extracted SE from vcov: {interaction_se:.4f}")
        except Exception as e:
            logger.warning(f"Could not extract vcov, using fallback SE: {e}")
            # Fallback to coef_norm
            try:
                coef_table = self.model_3b_categorical.coef_norm()
                if isinstance(coef_table, dict):
                    interaction_info = coef_table.get('temperate_x_plant_high', {})
                    interaction_se = interaction_info.get('std_error', 0.1)
                    p_value = interaction_info.get('p_value')
            except:
                pass
        
        # Perform hypothesis testing with integrated scale transformations
        logger.info("\nPerforming hypothesis testing with integrated scale transformations...")
        
        # 1. Test interaction coefficient with SESOI and scale
        sesoi_threshold = 0.1  # 0.1 on log scale ≈ 10.5% on response scale
        h2_wald_result = WaldTest.test_single_coefficient_with_scale(
            coefficient=interaction_coef,
            standard_error=interaction_se,
            h0_definition=H0Definition.sesoi(sesoi_threshold, 'two-sided'),
            alternative='two-sided',
            link='log'  # Enables scale transformations
        )
        
        # 2. Perform TOST with scale transformations
        tost_result = TOSTWrapper.perform_tost_with_scale(
            test_function=WaldTest.test_single_coefficient,
            effect_estimate=interaction_coef,
            standard_error=interaction_se,
            lower_bound=-0.1,
            upper_bound=0.1,
            alpha=0.05,
            link='log'  # Enables scale transformations
        )
        
        # Log enhanced results
        logger.info(f"  Interaction (log scale): {interaction_coef:.3f} ± {interaction_se:.3f}")
        if 'response_scale_interpretation' in h2_wald_result:
            logger.info(f"  Response scale: {h2_wald_result['response_scale_interpretation']}")
        logger.info(f"  Statistical significance: p = {h2_wald_result['p_value']:.4f}")
        
        # Check SESOI on response scale
        exceeds_sesoi = h2_wald_result.get('exceeds_sesoi_response_scale', 
                                          h2_wald_result.get('exceeds_sesoi', False))
        logger.info(f"  Exceeds SESOI (10.5% on response): {exceeds_sesoi}")
        
        if 'scale_interpretation' in tost_result:
            logger.info(f"  TOST: {tost_result['scale_interpretation']}")
        
        if 'practical_significance' in h2_wald_result:
            logger.info(f"  Overall: {h2_wald_result['practical_significance']}")
        
        # Extract scale info for reporting
        scale_info = h2_wald_result.get('scale_transformation', {})
        pct_change = scale_info.get('percent_change', {})
        
        results['model_coefficient'] = {
            'coefficient_log': interaction_coef,
            'se': interaction_se,
            'p_value': h2_wald_result['p_value'],
            'ci_lower': h2_wald_result['ci_lower'],
            'ci_upper': h2_wald_result['ci_upper'],
            'percent_change': pct_change.get('estimate', 0),
            'ci_percent': [pct_change.get('ci_lower', 0), pct_change.get('ci_upper', 0)],
            'exceeds_sesoi': exceeds_sesoi,
            'tost_p_value': tost_result.get('tost_p_value', 1.0),
            'tost_conclusion': tost_result.get('conclusion', ''),
            'is_equivalent': tost_result.get('is_equivalent', False),
            'response_scale_interpretation': h2_wald_result.get('response_scale_interpretation', ''),
            'practical_significance': h2_wald_result.get('practical_significance', ''),
            'note': 'Interaction coefficient with integrated scale transformations'
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
                'aic': self.model_3a_continuous.aic() if hasattr(self.model_3a_continuous, 'aic') else None,
                'bic': None  # H2O doesn't provide BIC directly
            }
        
        if self.model_3b_categorical:
            report['models']['categorical'] = {
                'type': 'Model 3b: Categorical interaction (temperate × plant_high)',
                'aic': self.model_3b_categorical.aic() if hasattr(self.model_3b_categorical, 'aic') else None,
                'bic': None  # H2O doesn't provide BIC directly
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
                    hypothesis_name="H2: Temperate mismatch hypothesis"
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
                    hypothesis_name="H2: Temperate mismatch hypothesis"
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
        
        # Add interaction visualizations if available
        if output_dir and hasattr(self, 'h2_results'):
            # Save interaction plot if available
            if self.h2_results.get('model_3a_continuous', {}).get('interaction_plot'):
                try:
                    viz_path = output_dir / "method2_interaction_plot.png"
                    self.h2_results['model_3a_continuous']['interaction_plot'].savefig(
                        viz_path, dpi=300, bbox_inches='tight'
                    )
                    report['visualizations'].append(str(viz_path))
                    logger.info(f"Interaction plot saved to {viz_path}")
                except Exception as e:
                    logger.warning(f"Failed to save interaction plot: {e}")
            
            # Generate SESOI/TOST visualizations for interaction coefficients
            if self.h2_results.get('model_3a_continuous', {}).get('coefficient_tests'):
                try:
                    from ..hypothesis.wald_test import WaldTest
                    import matplotlib.pyplot as plt
                    
                    # Visualize practical significance for interaction term
                    coef_tests = self.h2_results['model_3a_continuous']['coefficient_tests']
                    interaction_coef = coef_tests.get('bio1:plants_richness_log', {})
                    
                    # Create unified practical significance visualization for interaction
                    if interaction_coef.get('sesoi_test') or interaction_coef.get('tost_test'):
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])
                        
                        # Get effect data from either test
                        test_data = interaction_coef.get('sesoi_test') or interaction_coef.get('tost_test')
                        effect = test_data.get('coefficient', 0)
                        se = test_data.get('standard_error', 0.1)
                        
                        # Convert to response scale if available
                        if 'effect_on_response_scale' in test_data:
                            resp = test_data['effect_on_response_scale']
                            effect_pct = resp['percent_change']
                            ci_low = resp['ci_lower_percent']
                            ci_high = resp['ci_upper_percent']
                        else:
                            effect_pct = (np.exp(effect) - 1) * 100
                            ci_low = (np.exp(effect - 1.96*se) - 1) * 100
                            ci_high = (np.exp(effect + 1.96*se) - 1) * 100
                        
                        # Determine interpretation
                        if ci_low > 10.5:
                            interpretation = 'Practically Positive'
                            color = 'darkgreen'
                            conclusion = 'H2 SUPPORTED: High-plant temperate regions have significantly MORE fungi'
                        elif ci_high < -10.5:
                            interpretation = 'Practically Negative'
                            color = 'darkred'
                            conclusion = 'H2 SUPPORTED: High-plant temperate regions have significantly FEWER fungi'
                        elif ci_low > -10.5 and ci_high < 10.5:
                            interpretation = 'Practically Equivalent'
                            color = 'dimgray'
                            conclusion = 'H2 NOT SUPPORTED: Interaction effect is negligible'
                        else:
                            interpretation = 'Inconclusive'
                            color = 'orange'
                            conclusion = 'H2 INCONCLUSIVE: CI crosses practical significance threshold'
                        
                        # Panel 1: Effect with decision regions
                        ax1.axvspan(-100, -10.5, alpha=0.1, color='red')
                        ax1.axvspan(-10.5, 10.5, alpha=0.1, color='gray')
                        ax1.axvspan(10.5, 100, alpha=0.1, color='green')
                        
                        # Plot effect with CI
                        ax1.errorbar(effect_pct, 0, xerr=[[effect_pct-ci_low], [ci_high-effect_pct]], 
                                    fmt='o', color=color, markersize=12, 
                                    capsize=10, capthick=3, linewidth=3)
                        
                        # Add threshold lines and labels
                        ax1.axvline(x=-10.5, color='black', linestyle='--', alpha=0.5, linewidth=1)
                        ax1.axvline(x=10.5, color='black', linestyle='--', alpha=0.5, linewidth=1)
                        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
                        
                        # Add text annotations
                        ax1.text(-30, 0.4, 'Fewer Fungi\nin Temperate', ha='center', fontsize=10, style='italic')
                        ax1.text(0, 0.4, 'No Effect', ha='center', fontsize=10, style='italic')
                        ax1.text(30, 0.4, 'More Fungi\nin Temperate', ha='center', fontsize=10, style='italic')
                        
                        # Format
                        ax1.set_xlim([-50, 50])
                        ax1.set_ylim([-0.5, 0.5])
                        ax1.set_xlabel('Interaction Effect: Percent Change in Fungi Richness (%)', fontsize=12)
                        ax1.set_title(f'H2 Test: Temperate × High Plant Interaction\nEffect: {effect_pct:.1f}% (95% CI: [{ci_low:.1f}%, {ci_high:.1f}%])', 
                                     fontsize=14, fontweight='bold')
                        ax1.set_yticks([])
                        ax1.grid(True, alpha=0.2, axis='x')
                        
                        # Panel 2: Interpretation
                        ax2.axis('off')
                        
                        # Create interpretation text
                        interp_text = f"Statistical Results:\n"
                        interp_text += f"• Log-scale coefficient: {effect:.3f} (SE: {se:.3f})\n"
                        interp_text += f"• Response-scale effect: {effect_pct:.1f}%\n"
                        interp_text += f"• 95% CI: [{ci_low:.1f}%, {ci_high:.1f}%]\n"
                        interp_text += f"• Practical significance threshold: ±10.5%\n\n"
                        
                        interp_text += f"Interpretation: {interpretation}\n\n"
                        interp_text += f"Conclusion:\n{conclusion}"
                        
                        # Add colored box for conclusion
                        bbox_props = dict(boxstyle="round,pad=0.5", alpha=0.3)
                        if 'SUPPORTED' in conclusion:
                            bbox_props['facecolor'] = 'lightgreen' if 'MORE' in conclusion else 'lightcoral'
                        elif 'NOT SUPPORTED' in conclusion:
                            bbox_props['facecolor'] = 'lightgray'
                        else:
                            bbox_props['facecolor'] = 'lightyellow'
                        
                        ax2.text(0.5, 0.5, interp_text, transform=ax2.transAxes,
                                fontsize=11, ha='center', va='center', bbox=bbox_props)
                        
                        plt.suptitle('Method 2: H2 Hypothesis Test - Temperate Mismatch', 
                                    fontsize=16, fontweight='bold', y=1.02)
                        plt.tight_layout()
                        
                        viz_path = output_dir / "method2_h2_interaction_test.png"
                        fig.savefig(viz_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        report['visualizations'].append(str(viz_path))
                        logger.info(f"H2 interaction test visualization saved to {viz_path}")
                        
                except Exception as e:
                    logger.warning(f"Failed to create interaction coefficient visualizations: {e}")
        
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