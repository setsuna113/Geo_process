"""Method 1: H0-H1 Testing with GLM (Julia-based).

Tests:
- H0: Fungi richness is independent of plant/vertebrate richness
- H1: Biodiversity begets biodiversity (positive correlation exists)

Compares Model 1 (climate only) vs Model 2 (climate + biodiversity).
Uses Julia GLM for correct AIC calculation.
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Optional, Union, Tuple, List
from pathlib import Path

from ..config_loader import MLRegressionConfig
from ..utils import JuliaManager
from ..hypothesis.julia_hypothesis_tests import JuliaHypothesisTests
from ..evaluation.glm_comparison import ModelComparison

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
        self.julia = JuliaManager.get_or_init()
        
        self.model_null = None  # Model 1: Climate only
        self.model_full = None  # Model 2: Climate + biodiversity
        self.data = None  # pandas DataFrame
        self.hypothesis_results = {}  # Store hypothesis test results
    
    def fit_models(self, data_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Fit the nested models using preprocessed data.
        
        Args:
            data_metadata: Metadata about data loaded in Julia
            
        Returns:
            Dictionary with fitted models and results
        """
        logger.info("Fitting Method 1 models with preprocessed data")
        
        # Get data summary from Julia
        data_summary = self.julia.get_data_summary()
        
        # Data is in Julia, we work with metadata
        n_rows = data_summary['n_rows']
        columns = data_summary['columns']
        
        logger.info("Fitting negative binomial GLM models with Julia (theta estimated)")
        
        # Get MESF columns from Julia data
        mesf_cols = data_summary.get('mesf_columns', [])
        logger.info(f"Using {len(mesf_cols)} MESF eigenvectors")
        
        # Define feature sets based on known preprocessing
        climate_cols = [col for col in columns if col.startswith('bio') and col.endswith('_std')]
        elevation_cols = [col for col in columns if col.startswith('elevation')]
        base_features = climate_cols + elevation_cols + mesf_cols  # Null model features
        biotic_cols = ['plants_richness_log', 'vertebrates_richness_log']  # Additional features
        
        # Fit models using Julia GLM
        
        # Create formulas for Julia
        null_formula = f"fungi_richness ~ {' + '.join(base_features)}"
        full_formula = f"fungi_richness ~ {' + '.join(base_features + biotic_cols)}"
        
        # Fit null model (climate + spatial) - data is already in Julia
        logger.info(f"Fitting null model with {len(base_features)} features")
        self.model_null = self.julia.fit_glm(formula=null_formula)
        
        # Fit full model (climate + spatial + biodiversity) - let it estimate its own theta
        feature_cols_full = base_features + biotic_cols
        logger.info(f"Fitting full model with {len(feature_cols_full)} features")
        self.model_full = self.julia.fit_glm(formula=full_formula)
        
        logger.info(f"Models fitted successfully")
        logger.info(f"Null model features: {len(base_features)}")
        logger.info(f"Full model features: {len(feature_cols_full)}")
        
        # Use hypothesis testing classes for proper statistical tests
        results = {}
        
        # Extract degrees of freedom for proper reporting
        null_df = self.model_null['n_obs'] - 1
        residual_df_null = self.model_null['df_residual']
        residual_df_full = self.model_full['df_residual']
        
        # Get model performance from Julia results
        n_obs = self.model_null['n_obs']
        null_perf = self.model_null
        full_perf = self.model_full
        
        # Get proper AIC from Julia model results
        aic_null = self.model_null['aic']
        aic_full = self.model_full['aic']
        
        # Get log-likelihood from AIC: AIC = -2*loglik + 2*k
        # Number of parameters (including intercept)
        k_null = len(base_features) + 1  # +1 for intercept
        k_full = len(feature_cols_full) + 1
        
        # Get log-likelihood and BIC from Julia results
        null_loglik = self.model_null['loglik']
        full_loglik = self.model_full['loglik']
        bic_null = self.model_null['bic']
        bic_full = self.model_full['bic']
        
        # Get deviance for reporting
        null_deviance = self.model_null['deviance']
        full_deviance = self.model_full['deviance']
        
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
        # Use JuliaHypothesisTests for consistency
        lrt_results = JuliaHypothesisTests.likelihood_ratio_test(
            null_loglik=null_loglik,
            full_loglik=full_loglik,
            df_difference=len(biotic_cols)
        )
        # Add additional fields for backward compatibility
        lrt_results['statistic'] = lrt_results['lr_statistic']
        lrt_results['df'] = lrt_results.get('df', len(biotic_cols))
        lrt_results['deviance_difference'] = null_deviance - full_deviance
        lrt_results['null_df'] = residual_df_null
        lrt_results['full_df'] = residual_df_full
        
        # Add context about large sample size
        if n_obs > 100000:
            lrt_results['large_sample_warning'] = (
                f"With {n_obs:,} observations, focus on deviance change ({lrt_results['deviance_difference']:.2f}) "
                f"and effect sizes rather than p-value ({lrt_results['p_value']:.4f}) for practical significance."
            )
        
        results['likelihood_ratio_test'] = lrt_results
        
        # 2. Score Test - Test the improvement from adding biodiversity predictors
        # For testing H0 vs H1, we compare null model to full model
        # Note: Score test should use NULL DEVIANCE of null model, not residual deviance
        try:
            # Score test: Compare improvement from null to full model
            score_results = JuliaHypothesisTests.score_test_from_deviance(
                null_deviance=self.model_null['deviance'],  # Null model deviance
                full_deviance=self.model_full['deviance'],  # Full model deviance
                df_difference=residual_df_null - residual_df_full  # Difference in df
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
        
        # 3. Model comparison using AIC/BIC
        # Create model comparison results manually
        model_comparison = {
            'model_stats': {
                'null_model': {
                    'aic': aic_null,
                    'bic': bic_null,
                    'deviance': null_deviance
                },
                'full_model': {
                    'aic': aic_full,
                    'bic': bic_full,
                    'deviance': full_deviance
                }
            },
            'rankings': {
                'aic': {
                    'best_model': 'null_model' if aic_null < aic_full else 'full_model',
                    'delta_from_best': {
                        'null_model': 0.0 if aic_null < aic_full else aic_null - aic_full,
                        'full_model': aic_full - aic_null if aic_null < aic_full else 0.0
                    }
                },
                'bic': {
                    'best_model': 'null_model' if bic_null < bic_full else 'full_model',
                    'delta_from_best': {
                        'null_model': 0.0 if bic_null < bic_full else bic_null - bic_full,
                        'full_model': bic_full - bic_null if bic_null < bic_full else 0.0
                    }
                }
            },
            'best_models': {
                'aic': 'null_model' if aic_null < aic_full else 'full_model',
                'bic': 'null_model' if bic_null < bic_full else 'full_model'
            }
        }
        results['model_comparison'] = model_comparison
        
        # 4. Wald test for biodiversity coefficients using vcov extraction
        wald_results = {}
        try:
            # Get vcov matrix from Julia model
            logger.info("Extracting variance-covariance matrix for Wald tests")
            coef_values = self.model_full['coefficients']
            vcov_matrix = self.model_full['vcov']
            coef_names = ['(Intercept)'] + feature_cols_full
            
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
                    wald_sesoi = JuliaHypothesisTests.sesoi_test(
                        coefficient=coef,
                        standard_error=se,
                        sesoi_threshold=sesoi_threshold,
                        alternative='greater'
                    )
                    
                    # 2. TOST test with scale transformations (REPORT THIS)
                    tost_result = JuliaHypothesisTests.tost_test(
                        coefficient=coef,
                        standard_error=se,
                        lower_bound=-0.1,
                        upper_bound=0.1,
                        alpha=0.05
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
                        'sesoi_test': wald_sesoi,
                        'tost_test': tost_result,
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
                joint_sesoi = JuliaHypothesisTests.joint_wald_test(
                    coefficients=bio_coefs,
                    covariance_matrix=bio_vcov,
                    null_values=sesoi_nulls  # Test against SESOI thresholds
                )
                joint_sesoi['interpretation'] = 'Joint test for practical significance (H0: all |βi| ≤ 0.1)'
                
                # Traditional joint test for comparison (but don't report in final)
                joint_traditional = JuliaHypothesisTests.joint_wald_test(
                    coefficients=bio_coefs,
                    covariance_matrix=bio_vcov,
                    null_values=None  # Traditional H0: all βi = 0
                )
                
                wald_results['joint_biodiversity'] = {
                    'sesoi': joint_sesoi,
                    'traditional_for_reference': joint_traditional  # Keep for internal use only
                }
                
                logger.info(f"Joint SESOI test: chi2={joint_sesoi['chi2_statistic']:.2f}, p={joint_sesoi['p_value']:.4f}")
                logger.info(f"Joint test interpretation: Testing if any biodiversity effect exceeds SESOI threshold")
            
            # Extract z-values for reporting from SESOI test
            for coef_name in biotic_cols:
                if coef_name in wald_results and 'sesoi_test' in wald_results[coef_name]:
                    wald_results[coef_name]['z_value'] = wald_results[coef_name]['sesoi_test'].get('z_statistic', 0)
                
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
        # LRT: Check if deviance reduced (positive LR statistic means improvement)
        deviance_difference = lrt_results.get('deviance_difference', 0)
        deviance_interpretation = lrt_results.get('deviance_interpretation', '')
        h1_supported_by_lrt = deviance_difference > 0 and lrt_results.get('significant', False)
        
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
        
        # Overall decision: LRT deviance reduction is primary, SESOI/TOST confirm
        h1_supported = h1_supported_by_lrt and (h1_supported_by_sesoi or not h1_supported_by_tost)
        
        # Store detailed support evidence with new metrics
        results['h1_support_evidence'] = {
            'lrt_deviance_difference': deviance_difference,
            'lrt_interpretation': deviance_interpretation,
            'lrt_meaningful': h1_supported_by_lrt,
            'wald_sesoi': h1_supported_by_sesoi,
            'wald_tost_not_equivalent': h1_supported_by_tost,
            'aic': h1_supported_by_aic,
            'bic': h1_supported_by_bic,
            'overall': h1_supported
        }
        
        # Enhanced conclusion with deviance change and SESOI/TOST
        conclusion_text = (
            f"LRT deviance change = {deviance_difference:.2f} ({deviance_interpretation}). "
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
    
    def fit_model(self, data_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Fit the model - implements abstract method from base class.
        
        Args:
            data_metadata: Metadata about data loaded in Julia
            
        Returns:
            Model fitting results
        """
        return self.fit_models(data_metadata)
    
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
    
    def predict(self, **kwargs) -> np.ndarray:
        """Make predictions using fitted model.
        
        Note: Prediction happens in Julia on the already loaded data.
        
        Args:
            **kwargs: Additional prediction parameters
            
        Returns:
            Predicted values from Julia model
        """
        if self.model_full is None:
            raise ValueError("Must fit models first")
        
        # For now, return the coefficients as we can't predict without new data in Julia
        # This would need to be extended to load new data in Julia for prediction
        logger.warning("Prediction on new data not yet implemented - returning coefficients")
        return self.model_full['coefficients']
    
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
            score_report = JuliaHypothesisTests.generate_hypothesis_report(
                score_result,
                test_name="Score Test (Traditional H0: No effect)",
                hypothesis="biodiversity effects"
            )
            report['tests']['score_test'] = score_report
            logger.info("Score Test: Reporting traditional results only (model comparison test)")
            
            # Generate visualization if output directory provided
            if output_dir:
                try:
                    fig = JuliaHypothesisTests.visualize_hypothesis_test(
                        score_result,
                        test_name="Score Test",
                        save_path=output_dir / "method1_score_test.png"
                    )
                    report['visualizations'].append(str(output_dir / "method1_score_test.png"))
                    logger.info(f"Score test visualization saved to {output_dir / 'method1_score_test.png'}")
                except Exception as e:
                    logger.warning(f"Failed to create score test visualization: {e}")
        
        # LRT results - REPORT TRADITIONAL ONLY (model comparison test)
        if 'lrt' in self.hypothesis_results:
            lrt_result = self.hypothesis_results['lrt']
            # LRT is a model comparison test - report traditional results only
            lrt_report = JuliaHypothesisTests.generate_hypothesis_report(
                lrt_result,
                test_name="Likelihood Ratio Test (Traditional H0: No improvement)",
                hypothesis="biodiversity improves model fit"
            )
            report['tests']['lrt'] = lrt_report
            logger.info("LRT: Reporting traditional results only (model comparison test)")
            
            # Generate LRT visualization if output directory provided
            if output_dir:
                try:
                    fig = JuliaHypothesisTests.visualize_hypothesis_test(
                        lrt_result,
                        test_name="Likelihood Ratio Test",
                        save_path=output_dir / "method1_lrt_test.png"
                    )
                    report['visualizations'].append(str(output_dir / "method1_lrt_test.png"))
                    logger.info(f"LRT visualization saved to {output_dir / 'method1_lrt_test.png'}")
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
                        if 'sesoi_test' in wald_result:
                            sesoi_report = {
                                'test_name': f"SESOI Test - {var_name}",
                                'hypothesis': f"Practical significance of {var_name} (H0: |β| ≤ 0.1)",
                                'results': wald_result['sesoi_test']
                            }
                            report['tests'][f'sesoi_{var_name}'] = sesoi_report
                        if 'tost_test' in wald_result:
                            tost_report = {
                                'test_name': f"TOST Equivalence Test - {var_name}",
                                'hypothesis': f"Equivalence test for {var_name} (bounds: ±0.1)",
                                'results': wald_result['tost_test']
                            }
                            report['tests'][f'tost_{var_name}'] = tost_report
                            logger.info(f"Wald {var_name}: Reporting SESOI and TOST only (coefficient test)")
            
            # Generate visualization if output directory provided
            if output_dir:
                try:
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    import numpy as np
                    
                    # Create scientifically correct visualization
                    fig = plt.figure(figsize=(14, 8))
                    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])
                    
                    # Collect data for visualization
                    coef_names = []
                    coefficients = []
                    std_errors = []
                    response_scale_effects = []
                    response_scale_cis = []
                    interpretations = []
                    
                    for var_name, wald_result in wald_results.items():
                        if var_name != 'joint_biodiversity' and isinstance(wald_result, dict):
                            if 'sesoi_test' in wald_result and 'summary' in wald_result:
                                coef_names.append(var_name.replace('_log', '').replace('_', ' ').title())
                                coefficients.append(wald_result['summary']['coefficient'])
                                std_errors.append(wald_result['summary']['se'])
                                
                                # Get response scale effects if available
                                if 'effect_on_response_scale' in wald_result['sesoi_test']:
                                    resp_scale = wald_result['sesoi_test']['effect_on_response_scale']
                                    response_scale_effects.append(resp_scale['percent_change'])
                                    response_scale_cis.append((
                                        resp_scale['ci_lower_percent'],
                                        resp_scale['ci_upper_percent']
                                    ))
                                else:
                                    # Convert log scale to response scale manually
                                    coef = wald_result['summary']['coefficient']
                                    se = wald_result['summary']['se']
                                    pct_change = (np.exp(coef) - 1) * 100
                                    ci_lower = (np.exp(coef - 1.96*se) - 1) * 100
                                    ci_upper = (np.exp(coef + 1.96*se) - 1) * 100
                                    response_scale_effects.append(pct_change)
                                    response_scale_cis.append((ci_lower, ci_upper))
                                
                                # Determine interpretation based on CI
                                ci_low, ci_high = response_scale_cis[-1]
                                if ci_low > 10.5:
                                    interpretations.append('positive')
                                elif ci_high < -10.5:
                                    interpretations.append('negative')
                                elif ci_low > -10.5 and ci_high < 10.5:
                                    interpretations.append('equivalent')
                                else:
                                    interpretations.append('inconclusive')
                    
                    if coef_names:
                        # Main panel: Effect sizes with decision regions
                        ax_main = fig.add_subplot(gs[:, 0])
                        y_pos = np.arange(len(coef_names))
                        
                        # Add decision region backgrounds
                        ax_main.axvspan(-100, -10.5, alpha=0.1, color='red', label='Practically Negative')
                        ax_main.axvspan(-10.5, 10.5, alpha=0.1, color='gray', label='Practically Equivalent')
                        ax_main.axvspan(10.5, 100, alpha=0.1, color='green', label='Practically Positive')
                        
                        # Plot effects with CIs using appropriate colors
                        for i, (effect, (ci_low, ci_high), interp) in enumerate(zip(
                            response_scale_effects, response_scale_cis, interpretations
                        )):
                            # Choose color based on interpretation
                            if interp == 'positive':
                                color = 'darkgreen'
                                marker = '>'
                            elif interp == 'negative':
                                color = 'darkred'
                                marker = '<'
                            elif interp == 'equivalent':
                                color = 'dimgray'
                                marker = 'o'
                            else:  # inconclusive
                                color = 'orange'
                                marker = 'd'
                            
                            # Plot CI as error bar
                            ax_main.errorbar(effect, i, xerr=[[effect-ci_low], [ci_high-effect]], 
                                           fmt=marker, color=color, markersize=8, 
                                           capsize=5, capthick=2, linewidth=2)
                        
                        # Add threshold lines
                        ax_main.axvline(x=-10.5, color='black', linestyle='--', alpha=0.5, linewidth=1)
                        ax_main.axvline(x=10.5, color='black', linestyle='--', alpha=0.5, linewidth=1)
                        ax_main.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
                        
                        # Labels and formatting
                        ax_main.set_yticks(y_pos)
                        ax_main.set_yticklabels(coef_names)
                        ax_main.set_xlabel('Percent Change in Fungi Richness (%)', fontsize=12)
                        ax_main.set_title('Biodiversity Effects on Fungi Richness\nwith Practical Significance Threshold (±10.5%)', 
                                         fontsize=14, fontweight='bold')
                        ax_main.grid(True, alpha=0.2, axis='x')
                        ax_main.set_xlim([-50, 50])
                        
                        # Add value labels
                        for i, (effect, (ci_low, ci_high)) in enumerate(zip(response_scale_effects, response_scale_cis)):
                            ax_main.text(min(48, max(-48, effect)), i+0.3, 
                                       f'{effect:.1f}%', ha='center', va='bottom', fontsize=9)
                        
                        # Legend panel
                        ax_legend = fig.add_subplot(gs[0, 1])
                        ax_legend.axis('off')
                        
                        # Create legend content
                        legend_text = "Interpretation Guide:\n\n"
                        legend_text += "Decision Rules:\n"
                        legend_text += "• CI entirely > +10.5%:\n  → Practically positive\n\n"
                        legend_text += "• CI entirely < -10.5%:\n  → Practically negative\n\n"
                        legend_text += "• CI within ±10.5%:\n  → Practically equivalent\n\n"
                        legend_text += "• CI crosses threshold:\n  → Inconclusive\n\n"
                        legend_text += "Symbols:\n"
                        legend_text += "> Positive effect\n"
                        legend_text += "< Negative effect\n"
                        legend_text += "○ Equivalent to zero\n"
                        legend_text += "◊ Inconclusive"
                        
                        ax_legend.text(0.1, 0.95, legend_text, transform=ax_legend.transAxes,
                                     fontsize=10, verticalalignment='top')
                        
                        # Summary panel
                        ax_summary = fig.add_subplot(gs[1, 1])
                        ax_summary.axis('off')
                        
                        # Count interpretations
                        n_positive = interpretations.count('positive')
                        n_negative = interpretations.count('negative')
                        n_equivalent = interpretations.count('equivalent')
                        n_inconclusive = interpretations.count('inconclusive')
                        
                        summary_text = "Summary:\n\n"
                        summary_text += f"Total predictors: {len(coef_names)}\n\n"
                        summary_text += f"Practically positive: {n_positive}\n"
                        summary_text += f"Practically negative: {n_negative}\n"
                        summary_text += f"Practically equivalent: {n_equivalent}\n"
                        summary_text += f"Inconclusive: {n_inconclusive}\n\n"
                        
                        # Overall conclusion
                        if n_positive > 0 or n_negative > 0:
                            summary_text += "Conclusion:\n"
                            summary_text += "H1 SUPPORTED\n"
                            summary_text += "Biodiversity has\n"
                            summary_text += "practically significant\n"
                            summary_text += "effects on fungi richness"
                        else:
                            summary_text += "Conclusion:\n"
                            summary_text += "H1 NOT SUPPORTED\n"
                            summary_text += "No practically significant\n"
                            summary_text += "biodiversity effects"
                        
                        ax_summary.text(0.1, 0.95, summary_text, transform=ax_summary.transAxes,
                                      fontsize=11, verticalalignment='top', 
                                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.5))
                        
                        plt.suptitle('Method 1: Practical Significance Analysis', fontsize=16, fontweight='bold', y=1.02)
                        plt.tight_layout()
                        viz_path = output_dir / "method1_practical_significance.png"
                        fig.savefig(viz_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        report['visualizations'].append(str(viz_path))
                        logger.info(f"Practical significance visualization saved to {viz_path}")
                        
                except Exception as e:
                    logger.warning(f"Failed to create SESOI/TOST visualization: {e}")
        
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
        
        # Report deviance change (primary metric)
        if 'lrt_deviance_difference' in evidence:
            deviance_diff = evidence['lrt_deviance_difference']
            interpretation = evidence.get('lrt_interpretation', '')
            conclusion_text += f"1. Model Improvement (LRT):\n"
            conclusion_text += f"   Deviance change = {deviance_diff:.2f} ({interpretation})\n"
            if deviance_diff > 0:
                conclusion_text += f"   Full model REDUCES deviance by {abs(deviance_diff):.2f} (better fit)\n"
            elif deviance_diff < 0:
                conclusion_text += f"   Full model INCREASES deviance by {abs(deviance_diff):.2f} (worse fit)\n"
            else:
                conclusion_text += "   Models have identical deviance\n"
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