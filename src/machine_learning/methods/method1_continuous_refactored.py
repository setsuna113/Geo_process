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
from scipy import stats
from typing import Dict, Any, Optional, Union, Tuple, List
from pathlib import Path

from ..config_loader import MLRegressionConfig
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from ..hypothesis import WaldTest, LikelihoodRatioTest, ScoreTest
from ..evaluation.glm_comparison import ModelComparison
from ..hypothesis.h0_definitions import H0Definition
from ..hypothesis.tost_wrapper import TOSTWrapper
# vcov import moved to where it's used for cleaner separation

logger = logging.getLogger(__name__)


class Method1H0H1GLM:
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
        self.config = config
        
        self.model_null = None  # Model 1: Climate only
        self.model_full = None  # Model 2: Climate + biodiversity
        self.data_h2o = None
        self.hypothesis_results = {}  # Store hypothesis test results
    
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
        
        # Check overdispersion - H2O is inconsistent, handle both cases
        fungi_mean = h2o_data['fungi_richness'].mean()
        fungi_var = h2o_data['fungi_richness'].var()
        # Handle H2O returning list or float inconsistently
        if isinstance(fungi_mean, list): fungi_mean = fungi_mean[0]
        if isinstance(fungi_var, list): fungi_var = fungi_var[0]
        dispersion_ratio = fungi_var / fungi_mean
        logger.info(f"Dispersion ratio: {dispersion_ratio:.2f}")
        
        logger.info("Fitting negative binomial GLM models with independent theta estimation")
        
        # Get MESF columns
        me_cols = [col for col in h2o_data.columns if col.startswith('ME')]
        logger.info(f"Using {len(me_cols)} MESF eigenvectors")
        
        # Define feature sets
        climate_cols = [col for col in h2o_data.columns if col.startswith('bio') and col.endswith('_std')]
        elevation_cols = [col for col in h2o_data.columns if col.startswith('elevation')]
        base_features = climate_cols + elevation_cols + me_cols  # Null model features
        biotic_cols = ['plants_richness_log', 'vertebrates_richness_log']  # Additional features
        
        # Use standard H2O GLM with negative binomial family
        from h2o.estimators.glm import H2OGeneralizedLinearEstimator
        
        # Fit null model (climate + spatial) - let it estimate its own theta
        logger.info(f"Fitting null model with {len(base_features)} features")
        self.model_null = H2OGeneralizedLinearEstimator(
            family='negativebinomial',
            link='log',
            max_iterations=1000,
            standardize=True,
            lambda_=0.0,  # No penalty for fair nested model comparison
            solver='IRLSM',
            compute_p_values=True,
            remove_collinear_columns=True,  # Handle collinearity automatically
            intercept=True,
            seed=42
        )
        self.model_null.train(
            x=base_features,
            y='fungi_richness',
            training_frame=h2o_data
        )
        
        # Fit full model (climate + spatial + biodiversity) - let it estimate its own theta
        feature_cols_full = base_features + biotic_cols
        logger.info(f"Fitting full model with {len(feature_cols_full)} features")
        self.model_full = H2OGeneralizedLinearEstimator(
            family='negativebinomial',
            link='log',
            max_iterations=1000,
            standardize=True,
            lambda_=0.0,  # No penalty for fair nested model comparison
            solver='IRLSM',
            compute_p_values=True,
            remove_collinear_columns=True,  # Handle collinearity automatically
            intercept=True,
            seed=42
        )
        self.model_full.train(
            x=feature_cols_full,
            y='fungi_richness',
            training_frame=h2o_data
        )
        
        logger.info(f"Models fitted successfully")
        logger.info(f"Null model features: {len(base_features)}")
        logger.info(f"Full model features: {len(feature_cols_full)}")
        
        # Use hypothesis testing classes for proper statistical tests
        results = {}
        
        # Extract degrees of freedom for proper reporting
        null_df = self.model_null.null_degrees_of_freedom()
        residual_df_null = self.model_null.residual_degrees_of_freedom()
        residual_df_full = self.model_full.residual_degrees_of_freedom()
        
        # Get model performance objects for proper metrics
        n_obs = h2o_data.nrows
        null_perf = self.model_null.model_performance()
        full_perf = self.model_full.model_performance()
        
        # Get proper AIC from model performance (requires calc_like=True)
        try:
            aic_null = null_perf.aic()
            aic_full = full_perf.aic()
        except Exception as e:
            logger.warning(f"AIC extraction failed: {e}")
            aic_null = None
            aic_full = None
        
        # Get log-likelihood from AIC: AIC = -2*loglik + 2*k
        # Number of parameters (including intercept)
        k_null = len(base_features) + 1  # +1 for intercept
        k_full = len(feature_cols_full) + 1
        
        # Handle case where H2O returns None or string for AIC
        try:
            # Try to convert to float if it's a string
            if isinstance(aic_null, str):
                aic_null = float(aic_null) if aic_null != 'NaN' else None
            if isinstance(aic_full, str):
                aic_full = float(aic_full) if aic_full != 'NaN' else None
        except (ValueError, TypeError):
            aic_null = None
            aic_full = None
            
        if aic_null is None or aic_full is None:
            # Use deviance as fallback
            logger.warning("AIC not available, using deviance-based approximation")
            null_deviance = self.model_null.residual_deviance()
            full_deviance = self.model_full.residual_deviance()
            # Approximate log-likelihood from deviance
            null_loglik = -null_deviance / 2
            full_loglik = -full_deviance / 2
            # Calculate AIC from approximated log-likelihood
            aic_null = -2 * null_loglik + 2 * k_null
            aic_full = -2 * full_loglik + 2 * k_full
        else:
            # Back-calculate log-likelihood from AIC
            null_loglik = -(aic_null - 2 * k_null) / 2
            full_loglik = -(aic_full - 2 * k_full) / 2
        
        # Calculate BIC manually: BIC = -2*log(L) + k*log(n)
        bic_null = -2 * null_loglik + k_null * np.log(n_obs)
        bic_full = -2 * full_loglik + k_full * np.log(n_obs)
        
        # Get deviance for reporting
        null_deviance = self.model_null.residual_deviance()
        full_deviance = self.model_full.residual_deviance()
        
        # Store model metrics
        results['model_metrics'] = {
            'null_model': {
                'aic': aic_null,  # Use our calculated AIC
                'bic': bic_null,
                'loglikelihood': null_loglik,
                'deviance': null_deviance,
                'mean_deviance': null_deviance / residual_df_null,  # Normalized deviance
                'null_df': null_df,
                'residual_df': residual_df_null,
                'n_parameters': k_null
            },
            'full_model': {
                'aic': aic_full,  # Use our calculated AIC
                'bic': bic_full,
                'loglikelihood': full_loglik,
                'deviance': full_deviance,
                'mean_deviance': full_deviance / residual_df_full,  # Normalized deviance
                'null_df': null_df,
                'residual_df': residual_df_full,
                'n_parameters': k_full
            },
            'n_observations': n_obs
        }
        
        # 1. Likelihood Ratio Test for H0 vs H1
        # Clean LRT now reports McFadden R² and interprets effect size
        lrt_results = LikelihoodRatioTest.test_nested_models(
            null_model=self.model_null,
            full_model=self.model_full,
            df_difference=len(biotic_cols)  # Number of biodiversity predictors
        )
        lrt_results['null_df'] = residual_df_null
        lrt_results['full_df'] = residual_df_full
        
        # Add context about large sample size
        if n_obs > 100000:
            lrt_results['large_sample_warning'] = (
                f"With {n_obs:,} observations, focus on McFadden R² ({lrt_results['mcfadden_r2']:.4f}) "
                f"rather than p-value ({lrt_results['p_value']:.4f}) for practical significance."
            )
        
        results['likelihood_ratio_test'] = lrt_results
        
        # 2. Score Test - Test the improvement from adding biodiversity predictors
        # For testing H0 vs H1, we compare null model to full model
        # Note: Score test should use NULL DEVIANCE of null model, not residual deviance
        try:
            # Score test: Compare improvement from null to full model
            score_results = ScoreTest.test_from_model_statistics(
                deviance_null=self.model_null.residual_deviance(),  # Null model deviance
                deviance_residual=self.model_full.residual_deviance(),  # Full model deviance
                df_null=residual_df_null,  # Null model df
                df_residual=residual_df_full,  # Full model df
                n_obs=n_obs
            )
            # Adjust interpretation for H0 vs H1 test
            score_results['interpretation'] = 'Testing if biodiversity improves over climate-only model'
        except Exception as e:
            logger.warning(f"Score test failed: {e}")
            # Create a simple fallback result using deviance difference
            deviance_improvement = null_deviance - full_deviance
            if deviance_improvement < 0:
                deviance_improvement = 0  # Can't have negative improvement
            score_results = {
                'statistic': deviance_improvement,
                'df': len(biotic_cols),  # 2 for biodiversity predictors
                'p_value': 1 - stats.chi2.cdf(deviance_improvement, df=len(biotic_cols)),
                'significant_at_0.05': deviance_improvement > stats.chi2.ppf(0.95, df=len(biotic_cols))
            }
        results['score_test'] = score_results
        
        # 3. Model comparison using AIC/BIC with our calculated BIC
        # Override model comparison to include BIC
        model_comparison = ModelComparison.compare_models(
            models={'null_model': self.model_null, 'full_model': self.model_full},
            criteria=['aic', 'deviance']  # Don't request BIC from models
        )
        # Add our calculated BIC
        model_comparison['model_stats']['null_model']['bic'] = bic_null
        model_comparison['model_stats']['full_model']['bic'] = bic_full
        # Rank by BIC
        model_comparison['rankings']['bic'] = {
            'ranked_models': ['null_model', 'full_model'] if bic_null < bic_full else ['full_model', 'null_model'],
            'values': [bic_null, bic_full] if bic_null < bic_full else [bic_full, bic_null],
            'best_model': 'null_model' if bic_null < bic_full else 'full_model',
            'best_value': min(bic_null, bic_full),
            'delta_from_best': {
                'null_model': 0.0 if bic_null < bic_full else bic_null - bic_full,
                'full_model': bic_full - bic_null if bic_null < bic_full else 0.0
            }
        }
        model_comparison['best_models']['bic'] = model_comparison['rankings']['bic']['best_model']
        results['model_comparison'] = model_comparison
        
        # 4. Wald test for biodiversity coefficients using vcov extraction
        wald_results = {}
        try:
            # Extract vcov matrix from H2O model
            logger.info("Extracting variance-covariance matrix for Wald tests")
            from src.machine_learning.utils.vcov import extract_vcov
            coef_values, vcov_matrix, coef_names = extract_vcov(
                self.model_full,
                data=h2o_data,
                x_names=feature_cols_full,
                y_name='fungi_richness',
                method='sandwich'  # LINUS: Sandwich is the standard - robust and fast
            )
            
            # Create coefficient lookup
            coef_dict = dict(zip(coef_names, coef_values))
            
            # Extract standard errors from diagonal of vcov
            std_errors = np.sqrt(np.diag(vcov_matrix))
            se_dict = dict(zip(coef_names, std_errors))
            
            # Use enhanced Wald and TOST tests with scale transformations
            logger.info("Performing Wald tests with integrated scale transformations")
            logger.info("Using vcov-derived standard errors for proper inference")
            
            # Perform tests for each biodiversity coefficient
            # IMPORTANT: For Wald tests, only report SESOI and TOST, never traditional
            for coef_name in biotic_cols:
                if coef_name in coef_dict:
                    coef = coef_dict[coef_name]
                    se = se_dict[coef_name]
                    logger.info(f"\nTesting {coef_name}: β={coef:.4f}, SE={se:.4f}")
                    
                    # 1. SESOI test with scale transformations (REPORT THIS)
                    sesoi_threshold = 0.1  # 0.1 on log scale ≈ 10.5% on response scale
                    wald_sesoi = WaldTest.test_single_coefficient_with_scale(
                        coefficient=coef,
                        standard_error=se,
                        h0_definition=H0Definition.sesoi(sesoi_threshold, 'greater'),
                        alternative='greater',
                        link='log'  # Enables scale transformations
                    )
                    
                    # 2. TOST test with scale transformations (REPORT THIS)
                    tost_result = TOSTWrapper.perform_tost_with_scale(
                        test_function=WaldTest.test_single_coefficient,
                        effect_estimate=coef,
                        standard_error=se,
                        lower_bound=-0.1,
                        upper_bound=0.1,
                        alpha=0.05,
                        link='log'  # Enables scale transformations
                    )
                    
                    # DO NOT run traditional test - only SESOI and TOST
                    logger.info(f"  SESOI Test: p={wald_sesoi['p_value']:.4f}, exceeds={wald_sesoi.get('exceeds_sesoi', False)}")
                    logger.info(f"  TOST Test: p={tost_result['tost_p_value']:.4f}, equivalent={tost_result['is_equivalent']}")
                    if 'response_scale_interpretation' in wald_sesoi:
                        logger.info(f"  Response scale: {wald_sesoi['response_scale_interpretation']}")
                    if 'practical_significance' in wald_sesoi:
                        logger.info(f"  Practical significance: {wald_sesoi['practical_significance']}")
                    
                    # Store results
                    wald_results[coef_name] = {
                        'sesoi': wald_sesoi,
                        'tost': tost_result,
                        'summary': {
                            'coefficient': coef,
                            'se': se,
                            'p_value': wald_sesoi['p_value'],
                            'ci_lower': wald_sesoi['ci_lower'],
                            'ci_upper': wald_sesoi['ci_upper'],
                            'exceeds_sesoi': wald_sesoi.get('exceeds_sesoi_response_scale', 
                                                           wald_sesoi.get('exceeds_sesoi', False)),
                            'is_equivalent': tost_result.get('is_equivalent', False),
                            'response_scale_interpretation': wald_sesoi.get('response_scale_interpretation', ''),
                            'practical_significance': wald_sesoi.get('practical_significance', '')
                        }
                    }
                else:
                    logger.warning(f"Coefficient {coef_name} not found in model")
            
            # Joint Wald test for biodiversity coefficients
            # This is a joint test, but we apply SESOI logic to it
            bio_indices = [coef_names.index(name) for name in biotic_cols if name in coef_names]
            if len(bio_indices) >= 2:
                bio_coefs = coef_values[bio_indices]
                bio_vcov = vcov_matrix[np.ix_(bio_indices, bio_indices)]
                
                # Test joint hypothesis with SESOI thresholds
                # For joint test, use null values of 0.1 for SESOI interpretation
                sesoi_nulls = np.ones(len(bio_coefs)) * 0.1  # All coefficients at SESOI threshold
                
                # Joint SESOI test: H0: all |βi| ≤ 0.1
                joint_sesoi = WaldTest.test_multiple_coefficients(
                    coefficients=bio_coefs,
                    covariance_matrix=bio_vcov,
                    null_values=sesoi_nulls  # Test against SESOI thresholds
                )
                joint_sesoi['interpretation'] = 'Joint test for practical significance (H0: all |βi| ≤ 0.1)'
                
                # Traditional joint test for comparison (but don't report in final)
                joint_traditional = WaldTest.test_multiple_coefficients(
                    coefficients=bio_coefs,
                    covariance_matrix=bio_vcov,
                    null_values=None  # Traditional H0: all βi = 0
                )
                
                wald_results['joint_biodiversity'] = {
                    'sesoi': joint_sesoi,
                    'traditional_for_reference': joint_traditional  # Keep for internal use only
                }
                
                logger.info(f"Joint SESOI test: chi2={joint_sesoi['wald_statistic']:.2f}, p={joint_sesoi['p_value']:.4f}")
                logger.info(f"Joint test interpretation: Testing if any biodiversity effect exceeds SESOI threshold")
            
            # Extract z-values for reporting from SESOI test
            for coef_name in biotic_cols:
                if coef_name in wald_results and 'sesoi' in wald_results[coef_name]:
                    wald_results[coef_name]['z_value'] = wald_results[coef_name]['sesoi']['wald_statistic']
                
        except Exception as e:
            logger.warning(f"Could not perform proper Wald tests: {e}")
            # Fallback: try to extract from H2O's internal p-values if available
            try:
                coef_table = self.model_full._model_json['output']['coefficients_table']
                names = coef_table.cell_values[0]  # First row is names
                for i, row in enumerate(coef_table.cell_values[1:], 1):  # Skip header
                    coef_name = row[0]
                    if coef_name in biotic_cols:
                        wald_results[coef_name] = {
                            'coefficient': float(row[1]) if row[1] is not None else 0.0,
                            'standard_error': float(row[2]) if len(row) > 2 and row[2] is not None else np.nan,
                            'p_value': float(row[4]) if len(row) > 4 and row[4] is not None else np.nan,
                            'significant_at_0.05': float(row[4]) < 0.05 if len(row) > 4 and row[4] is not None else False
                        }
            except Exception as e2:
                logger.warning(f"Fallback Wald extraction also failed: {e2}")
                for coef_name in biotic_cols:
                    wald_results[coef_name] = {
                        'coefficient': 'See model output',
                        'p_value': 'See model output',
                        'significant_at_0.05': 'See model output'
                    }
        results['wald_tests'] = wald_results
        
        # Overall H0 vs H1 conclusion based on new architecture
        # LRT: Check if McFadden R² shows meaningful improvement
        mcfadden_r2 = lrt_results.get('mcfadden_r2', 0)
        r2_interpretation = lrt_results.get('r2_interpretation', '')
        h1_supported_by_lrt = mcfadden_r2 >= 0.05  # At least modest improvement
        
        # Wald: Check SESOI and TOST results
        h1_supported_by_sesoi = False
        h1_supported_by_tost = False
        
        for k, w in wald_results.items():
            if 'sesoi' in k and isinstance(w, dict):
                # SESOI: Effect exceeds meaningful threshold
                if w.get('exceeds_sesoi', False):
                    h1_supported_by_sesoi = True
            elif 'tost' in k and isinstance(w, dict):
                # TOST: Effect is NOT equivalent to zero (meaningful)
                if not w.get('is_equivalent', True):  # Not equivalent means meaningful
                    h1_supported_by_tost = True
        
        h1_supported_by_aic = model_comparison['best_models'].get('aic') == 'full_model'
        h1_supported_by_bic = model_comparison['best_models'].get('bic') == 'full_model'
        
        # Overall decision: LRT McFadden R² is primary, SESOI/TOST confirm
        h1_supported = h1_supported_by_lrt and (h1_supported_by_sesoi or not h1_supported_by_tost)
        
        # Store detailed support evidence with new metrics
        results['h1_support_evidence'] = {
            'lrt_mcfadden_r2': mcfadden_r2,
            'lrt_interpretation': r2_interpretation,
            'lrt_meaningful': h1_supported_by_lrt,
            'wald_sesoi': h1_supported_by_sesoi,
            'wald_tost_not_equivalent': h1_supported_by_tost,
            'aic': h1_supported_by_aic,
            'bic': h1_supported_by_bic,
            'overall': h1_supported
        }
        
        # Enhanced conclusion with McFadden R² and SESOI/TOST
        conclusion_text = (
            f"LRT McFadden R² = {mcfadden_r2:.4f} ({r2_interpretation}). "
        )
        if h1_supported_by_sesoi:
            conclusion_text += "SESOI: Effects exceed meaningful threshold (0.1). "
        if h1_supported_by_tost:
            conclusion_text += "TOST: Effects are not equivalent to zero. "
        
        results['h0_h1_conclusion'] = {
            'h1_supported': h1_supported,
            'interpretation': 'Biodiversity begets biodiversity (H1)' if h1_supported else 'No biodiversity effect (H0)',
            'details': conclusion_text
        }
        
        # Store results for report generation with all metrics
        self.hypothesis_results = {
            'supports_h1': h1_supported,
            'interpretation': results['h0_h1_conclusion']['interpretation'],
            'lrt': lrt_results,
            'score': score_results,  # Add score test results
            'wald': wald_results,    # Add wald test results
            'model_comparison': model_comparison,
            'model_metrics': results['model_metrics'],  # Add comprehensive metrics
            'h1_support_evidence': results['h1_support_evidence'],  # Add support breakdown
            'model_formula': 'fungi_richness ~ climate + biodiversity + spatial'
        }
        
        # Get coefficients from the full model for biodiversity effects
        # H2O models use coef() and coef_with_p_values()
        if self.model_full:
            try:
                full_coeffs = self.model_full.coef()
                # H2O provides coefficient table with p-values
                coef_table = self.model_full._model_json['output']['coefficients_table']
                
                # Extract names and values from H2OTwoDimTable
                if hasattr(coef_table, 'cell_values'):
                    # H2OTwoDimTable format
                    names = [row[0] for row in coef_table.cell_values]
                    std_errors = [row[2] if len(row) > 2 else None for row in coef_table.cell_values]
                    p_values = [row[4] if len(row) > 4 else None for row in coef_table.cell_values]
                    full_std_errors = {name: se for name, se in zip(names, std_errors) if se is not None}
                    full_p_values = {name: pval for name, pval in zip(names, p_values) if pval is not None}
                else:
                    # Try dictionary format
                    full_std_errors = {name: se for name, se in 
                                     zip(coef_table.get('names', []), coef_table.get('std_error', []))}
                    full_p_values = {name: pval for name, pval in 
                                   zip(coef_table.get('names', []), coef_table.get('p_value', []))}
            except Exception as e:
                logger.warning(f"Could not extract full model coefficients: {e}")
                full_coeffs = {}
                full_std_errors = {}
                full_p_values = {}
        else:
            full_coeffs = {}
            full_std_errors = {}
            full_p_values = {}
        
        # Add biodiversity effects if available
        if 'plants_richness_log' in full_coeffs:
            self.hypothesis_results['plants_effect'] = {
                'coefficient': full_coeffs['plants_richness_log'],
                'std_error': full_std_errors.get('plants_richness_log', np.nan),
                'p_value': full_p_values.get('plants_richness_log', np.nan)
            }
        
        if 'vertebrates_richness_log' in full_coeffs:
            self.hypothesis_results['vertebrates_effect'] = {
                'coefficient': full_coeffs['vertebrates_richness_log'],
                'std_error': full_std_errors.get('vertebrates_richness_log', np.nan),
                'p_value': full_p_values.get('vertebrates_richness_log', np.nan)
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
        
        # H2O models: Extract coefficients properly
        coefficients = self.model_full.coef()
        
        # Get detailed coefficient info from H2O model
        try:
            coef_table = self.model_full._model_json['output']['coefficients_table']
            names = coef_table['names']
            coef_values = coef_table['coefficients']
            std_errors = coef_table.get('std_error', [None] * len(names))
            z_values = coef_table.get('z_value', [None] * len(names))
            p_values = coef_table.get('p_value', [None] * len(names))
            
            # Build lookup
            coef_info = {}
            for i, name in enumerate(names):
                coef_info[name] = {
                    'coefficient': coef_values[i],
                    'std_error': std_errors[i] if i < len(std_errors) else None,
                    'z_value': z_values[i] if i < len(z_values) else None,
                    'p_value': p_values[i] if i < len(p_values) else None
                }
            
            plants_coef = coef_info.get('plants_richness_log', {})
            vertebrates_coef = coef_info.get('vertebrates_richness_log', {})
            
        except Exception as e:
            logger.warning(f"Using basic coefficients: {e}")
            plants_coef = {'coefficient': coefficients.get('plants_richness_log', 0)}
            vertebrates_coef = {'coefficient': coefficients.get('vertebrates_richness_log', 0)}
        
        # H1 is supported if either biodiversity predictor is significant
        supports_h1 = False
        if plants_coef.get('p_value') is not None and vertebrates_coef.get('p_value') is not None:
            supports_h1 = (plants_coef['p_value'] < 0.05 or vertebrates_coef['p_value'] < 0.05)
        
        return {
            'plants': plants_coef,
            'vertebrates': vertebrates_coef,
            'supports_h1': supports_h1,
            'interpretation': 'Biodiversity begets biodiversity' if supports_h1 else 'No biodiversity effect'
        }
    
    def _interpret_wald_results(self, sesoi: Dict[str, Any], tost: Dict[str, Any]) -> str:
        """Interpret combined SESOI and TOST results.
        
        Args:
            sesoi: SESOI test results
            tost: TOST equivalence test results
            
        Returns:
            Interpretation string
        """
        exceeds_sesoi = sesoi.get('exceeds_sesoi', False)
        is_equivalent = tost.get('is_equivalent', False)
        coef = sesoi.get('coefficient', 0)
        
        if exceeds_sesoi:
            return f"Practically meaningful effect (β={coef:.3f} exceeds threshold)"
        elif is_equivalent:
            return f"Confirmed negligible effect (β={coef:.3f} within equivalence bounds)"
        else:
            return f"Effect below threshold but not confirmed negligible (β={coef:.3f})"
    
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
        
        Reports:
        - LRT with McFadden pseudo R² (values 0.2-0.4 indicate excellent fit)
        - Wald tests with SESOI (H0: |β| ≤ 0.1) and TOST equivalence testing
        - Model comparison with AIC/BIC
        
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
        
        # Model summaries with all metrics
        if self.model_full and 'model_metrics' in self.hypothesis_results:
            metrics = self.hypothesis_results['model_metrics']['full_model']
            report['models']['full'] = {
                'type': 'Model 1a: Full GLM with biodiversity predictors',
                'formula': self.hypothesis_results.get('model_formula', ''),
                'aic': metrics.get('aic'),
                'bic': metrics.get('bic'),
                'deviance': metrics.get('deviance'),
                'residual_df': metrics.get('residual_df'),
                'n_parameters': metrics.get('n_parameters')
            }
        
        if self.model_null and 'model_metrics' in self.hypothesis_results:
            metrics = self.hypothesis_results['model_metrics']['null_model']
            report['models']['null'] = {
                'type': 'Model 1b: Null model without biodiversity',
                'aic': metrics.get('aic'),
                'bic': metrics.get('bic'),
                'deviance': metrics.get('deviance'),
                'residual_df': metrics.get('residual_df'),
                'n_parameters': metrics.get('n_parameters')
            }
        
        # Score Test - REPORT TRADITIONAL ONLY (model comparison test)
        if 'score' in self.hypothesis_results:
            score_result = self.hypothesis_results['score']
            # Score test is a model comparison test - report traditional results only
            score_report = ScoreTest.generate_publication_report(
                score_result,
                test_name="Score Test (Traditional H0: No effect)",
                hypothesis="biodiversity effects",
                variable_names=['plants_richness_log', 'vertebrates_richness_log']
            )
            report['tests']['score_test'] = score_report
            logger.info("Score Test: Reporting traditional results only (model comparison test)")
            
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
        
        # LRT results - REPORT TRADITIONAL ONLY (model comparison test)
        if 'lrt' in self.hypothesis_results:
            lrt_result = self.hypothesis_results['lrt']
            # LRT is a model comparison test - report traditional results only
            lrt_report = LikelihoodRatioTest.generate_publication_report(
                lrt_result,
                test_name="Likelihood Ratio Test (Traditional H0: No improvement)",
                hypothesis="biodiversity improves model fit",
                model_description="Comparing climate-only model (null) with climate + biodiversity model (full)"
            )
            report['tests']['lrt'] = lrt_report
            logger.info("LRT: Reporting traditional results only (model comparison test)")
            
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
        
        # Wald test results - REPORT SESOI AND TOST ONLY (coefficient test)
        if 'wald' in self.hypothesis_results and self.hypothesis_results['wald']:
            wald_results = self.hypothesis_results['wald']
            
            # Generate Wald test report for individual coefficients
            # IMPORTANT: Only report SESOI and TOST, never traditional
            for var_name, wald_result in wald_results.items():
                if isinstance(wald_result, dict):
                    if var_name == 'joint_biodiversity':
                        # Report joint SESOI test only
                        if 'sesoi' in wald_result:
                            joint_report = {
                                'test_name': "Joint SESOI Test - All Biodiversity Predictors",
                                'hypothesis': "Joint practical significance (H0: all |βi| ≤ 0.1)",
                                'results': wald_result['sesoi']
                            }
                            report['tests']['joint_sesoi_biodiversity'] = joint_report
                            logger.info("Joint Wald: Reporting SESOI version only (coefficient test)")
                    else:
                        # Extract SESOI and TOST results for individual coefficients
                        if 'sesoi' in wald_result and 'tost' in wald_result:
                            sesoi_report = {
                                'test_name': f"SESOI Test - {var_name}",
                                'hypothesis': f"Practical significance of {var_name} (H0: |β| ≤ 0.1)",
                                'results': wald_result['sesoi']
                            }
                            tost_report = {
                                'test_name': f"TOST Equivalence Test - {var_name}",
                                'hypothesis': f"Equivalence test for {var_name} (bounds: ±0.1)",
                                'results': wald_result['tost']
                            }
                            report['tests'][f'sesoi_{var_name}'] = sesoi_report
                            report['tests'][f'tost_{var_name}'] = tost_report
                            logger.info(f"Wald {var_name}: Reporting SESOI and TOST only (coefficient test)")
            
            # Generate visualization if output directory provided
            if output_dir:
                try:
                    # Prepare data for visualization
                    viz_data = {
                        'coefficients': {},
                        'std_errors': {},
                        'p_values': {}
                    }
                    
                    for var_name, wald_result in wald_results.items():
                        if var_name != 'joint_biodiversity' and isinstance(wald_result, dict):
                            # Use SESOI results for visualization
                            if 'sesoi' in wald_result:
                                sesoi_data = wald_result['sesoi']
                                if 'coefficient' in sesoi_data:
                                    viz_data['coefficients'][var_name] = sesoi_data['coefficient']
                                if 'standard_error' in sesoi_data:
                                    viz_data['std_errors'][var_name] = sesoi_data['standard_error']
                                if 'p_value' in sesoi_data:
                                    viz_data['p_values'][var_name] = sesoi_data['p_value']
                    
                    if viz_data['coefficients']:  # Only create viz if we have data
                        fig = WaldTest.visualize_wald_test(
                            viz_data,
                            variable_names=list(viz_data['coefficients'].keys()),
                            title="Biodiversity Effects (Wald Test)"
                        )
                        viz_path = output_dir / "method1_wald_test.png"
                        fig.savefig(viz_path, dpi=300, bbox_inches='tight')
                        report['visualizations'].append(str(viz_path))
                        logger.info(f"Wald test visualization saved to {viz_path}")
                except Exception as e:
                    logger.warning(f"Failed to create Wald test visualization: {e}")
        
        # Model comparison using ModelComparison
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
        
        # Overall conclusion with new reporting standards
        evidence = self.hypothesis_results.get('h1_support_evidence', {})
        
        # Format conclusion following reporting guidelines
        conclusion_text = "Model Evaluation Summary:\n\n"
        
        # Report McFadden R² (primary metric)
        if 'lrt_mcfadden_r2' in evidence:
            r2 = evidence['lrt_mcfadden_r2']
            interpretation = evidence.get('lrt_interpretation', '')
            conclusion_text += f"1. Model Improvement (LRT):\n"
            conclusion_text += f"   McFadden pseudo-R² = {r2:.4f} ({interpretation})\n"
            if r2 >= 0.2:
                conclusion_text += "   This indicates EXCELLENT model fit (McFadden 1979 criterion: >0.2)\n"
            elif r2 >= 0.05:
                conclusion_text += "   This indicates meaningful model improvement\n"
            else:
                conclusion_text += "   This indicates negligible model improvement\n"
            conclusion_text += "\n"
        
        # Report SESOI and TOST results
        conclusion_text += "2. Practical Significance (Wald Tests):\n"
        if evidence.get('wald_sesoi', False):
            conclusion_text += "   SESOI: Biodiversity effects EXCEED meaningful threshold (|β| > 0.1)\n"
        else:
            conclusion_text += "   SESOI: Biodiversity effects do NOT exceed meaningful threshold (|β| ≤ 0.1)\n"
        
        if evidence.get('wald_tost_not_equivalent', False):
            conclusion_text += "   TOST: Effects are NOT equivalent to zero (meaningful difference)\n"
        else:
            conclusion_text += "   TOST: Effects are equivalent to zero within [-0.1, 0.1]\n"
        conclusion_text += "\n"
        
        # Report AIC/BIC
        conclusion_text += "3. Information Criteria:\n"
        if evidence.get('aic', False):
            conclusion_text += "   AIC selects full model (biodiversity improves fit)\n"
        else:
            conclusion_text += "   AIC selects null model (parsimony preferred)\n"
        if evidence.get('bic', False):
            conclusion_text += "   BIC selects full model (biodiversity improves fit)\n"
        else:
            conclusion_text += "   BIC selects null model (parsimony preferred)\n"
        conclusion_text += "\n"
        
        # Final verdict
        conclusion_text += "4. Overall Conclusion:\n   "
        if self.hypothesis_results.get('supports_h1', False):
            conclusion_text += "SUPPORT for H1: Biodiversity begets biodiversity\n"
            conclusion_text += "   Evidence: Model shows meaningful improvement with biodiversity predictors"
        else:
            conclusion_text += "SUPPORT for H0: No meaningful biodiversity effect\n"
            conclusion_text += "   Evidence: Biodiversity predictors do not meaningfully improve model"
        
        report['conclusion'] = {
            'supports_h1': self.hypothesis_results.get('supports_h1', False),
            'interpretation': self.hypothesis_results.get('interpretation', ''),
            'evidence_summary': evidence,
            'formatted_conclusion': conclusion_text,
            'model_metrics': self.hypothesis_results.get('model_metrics', {})
        }
        
        # Add coefficient analysis if available
        if 'wald' in self.hypothesis_results:
            coef_summary = {}
            for var_name, wald_result in self.hypothesis_results['wald'].items():
                if var_name != 'joint_biodiversity' and isinstance(wald_result, dict):
                    coef_summary[var_name] = {
                        'coefficient': wald_result.get('coefficient', 'N/A'),
                        'std_error': wald_result.get('standard_error', 'N/A'),
                        'z_value': wald_result.get('z_value', 'N/A'),
                        'p_value': wald_result.get('p_value', 'N/A'),
                        'significant': wald_result.get('significant_at_0.05', False)
                    }
            report['coefficient_analysis'] = coef_summary
        
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