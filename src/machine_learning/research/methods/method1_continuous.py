"""Method 1: Continuous Interaction GLM.

Based on regression_analysis_decisions.md lines 296-304.
Tests H2 hypothesis using continuous interaction term plants_richness:bio4.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Union, Tuple

from ...models.glm import PoissonGLM
from ...models.glm.h2o_negative_binomial import H2ONegativeBinomialGLM
from ...spatial import compute_mesf  # Now uses vectorized implementation
from ...config_loader import MLRegressionConfig

logger = logging.getLogger(__name__)


class Method1ContinuousGLM:
    """Method 1: GLM with continuous interaction term.
    
    Tests the temperate mismatch hypothesis (H2) using continuous interaction
    between plant richness and temperature seasonality (bio4).
    """
    
    def __init__(self, config: Optional[MLRegressionConfig] = None):
        """Initialize Method 1.
        
        Args:
            config: ML regression configuration
        """
        self.config = config or MLRegressionConfig()
        self.preprocessing_config = self.config.get_preprocessing_config()
        self.glm_config = self.config.get_glm_config()
        self.spatial_config = self.config.get_spatial_config()
        
        self.model_null = None
        self.model_full = None
        self.model_interaction = None
        self.data_processed = None
        
        logger.info("Initialized Method 1: Continuous Interaction GLM")
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for Method 1 analysis.
        
        Implements preprocessing from regression_analysis_decisions.md lines 199-277.
        
        Args:
            data: Raw biodiversity data with columns:
                - fungi_richness, plants_richness, vertebrates_richness
                - bio1, bio4, bio12, bio15, elevation
                - longitude, latitude
        
        Returns:
            Processed DataFrame ready for GLM
        """
        logger.info("Preparing data for Method 1 (continuous interaction)")
        
        # Apply complete case analysis (lines 203-213)
        initial_rows = len(data)
        data_clean = data.dropna(subset=[
            'fungi_richness', 'plants_richness', 'vertebrates_richness',
            'bio1', 'bio4', 'bio12', 'bio15', 'elevation',
            'longitude', 'latitude'
        ])
        
        rows_dropped = initial_rows - len(data_clean)
        logger.info(f"Complete case analysis: dropped {rows_dropped} rows "
                   f"({rows_dropped/initial_rows*100:.2f}%)")
        
        # Standardize climate variables using 2SD method (lines 220-229)
        climate_vars = ['bio1', 'bio4', 'bio12', 'bio15', 'elevation']
        
        for var in climate_vars:
            mean_val = data_clean[var].mean()
            sd_val = data_clean[var].std()
            data_clean[f'{var}_std'] = (data_clean[var] - mean_val) / (2 * sd_val)
            logger.info(f"Standardized {var}: mean={mean_val:.2f}, 2*sd={2*sd_val:.2f}")
        
        # Add quadratic elevation term (from regression_analysis_decisions.md lines 288, 301, 314)
        data_clean['elevation_std_sq'] = data_clean['elevation_std'] ** 2
        logger.info("Created quadratic elevation term: elevation_std_sq")
        
        # Handle biotic predictors (lines 235-257)
        transformation = self.preprocessing_config['biotic_predictors']['transformation']
        
        if transformation in ['log', 'both']:
            # Log transformation
            data_clean['plants_richness_log'] = np.log(data_clean['plants_richness'] + 1)
            data_clean['vertebrates_richness_log'] = np.log(data_clean['vertebrates_richness'] + 1)
            logger.info("Created log-transformed biotic predictors")
        
        # Compute spatial eigenvectors (MESF)
        logger.info("Computing spatial eigenvectors...")
        coords = data_clean[['longitude', 'latitude']].values
        mesf_df = compute_mesf(coords, self.spatial_config)
        
        # Add MESF columns to data
        for col in mesf_df.columns:
            data_clean[col] = mesf_df[col].values
        
        logger.info(f"Added {len(mesf_df.columns)} spatial eigenvectors")
        
        self.data_processed = data_clean
        return data_clean
    
    def create_interaction_term(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create the continuous interaction term.
        
        Implements interaction from regression_analysis_decisions.md line 303.
        
        Args:
            data: Processed data
            
        Returns:
            Data with interaction term added
        """
        # Create interaction: plants_richness_log × bio4_std
        if 'plants_richness_log' in data.columns:
            data['plants_bio4_interaction'] = (
                data['plants_richness_log'] * data['bio4_std']
            )
            logger.info("Created interaction term: plants_richness_log × bio4_std")
        else:
            # Use raw counts if log not available
            data['plants_bio4_interaction'] = (
                data['plants_richness'] * data['bio4_std']
            )
            logger.info("Created interaction term: plants_richness × bio4_std")
        
        return data
    
    def fit_models(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Fit the nested models for Method 1.
        
        Fits three models as specified in regression_analysis_decisions.md lines 286-304:
        1. Null model (climate only)
        2. Full model (climate + biodiversity)
        3. Interaction model (climate + biodiversity + interaction)
        
        Args:
            data: Optional processed data (uses self.data_processed if None)
            
        Returns:
            Dictionary with fitted models and results
        """
        if data is None:
            data = self.data_processed
        
        if data is None:
            raise ValueError("Must call prepare_data() first or provide data")
        
        # Check overdispersion to determine GLM family
        dispersion_ratio = data['fungi_richness'].var() / data['fungi_richness'].mean()
        distribution = self.config.get_distribution_family(dispersion_ratio)
        
        # Select GLM class - use H2O for negative binomial (massive parallelization)
        if distribution == 'negative_binomial':
            GLMClass = H2ONegativeBinomialGLM
            logger.info("Using H2O Negative Binomial GLM for 256-core parallelization")
        else:
            GLMClass = PoissonGLM
        
        # Get ME columns
        me_cols = [col for col in data.columns if col.startswith('ME')]
        
        # Model 1: Climate only (null model) - lines 286-288
        logger.info("Fitting Model 1: Climate only (null)")
        feature_cols_null = [
            'bio1_std', 'bio4_std', 'bio12_std', 'bio15_std', 
            'elevation_std', 'elevation_std_sq'  # Added quadratic term
        ] + me_cols
        
        X_null = data[feature_cols_null]
        y = data['fungi_richness']
        
        self.model_null = GLMClass(self.config.config)
        self.model_null.fit(X_null, y)
        
        # Model 2: Add biodiversity (full model) - lines 290-293
        logger.info("Fitting Model 2: Climate + biodiversity (full)")
        
        # Use log-transformed if available, otherwise raw
        if 'plants_richness_log' in data.columns:
            biotic_cols = ['plants_richness_log', 'vertebrates_richness_log']
        else:
            biotic_cols = ['plants_richness', 'vertebrates_richness']
        
        feature_cols_full = feature_cols_null + biotic_cols
        X_full = data[feature_cols_full]
        
        self.model_full = GLMClass(self.config.config)
        self.model_full.fit(X_full, y)
        
        # Model 3: Add interaction (interaction model) - lines 299-304
        logger.info("Fitting Model 3: Climate + biodiversity + interaction")
        
        # Create interaction term
        data = self.create_interaction_term(data)
        
        feature_cols_interaction = feature_cols_full + ['plants_bio4_interaction']
        X_interaction = data[feature_cols_interaction]
        
        self.model_interaction = GLMClass(self.config.config)
        self.model_interaction.fit(X_interaction, y)
        
        # Compile results
        results = {
            'model_null': self.model_null,
            'model_full': self.model_full,
            'model_interaction': self.model_interaction,
            'distribution': distribution,
            'dispersion_ratio': dispersion_ratio,
            'n_samples': len(data),
            'n_features': {
                'null': len(feature_cols_null),
                'full': len(feature_cols_full),
                'interaction': len(feature_cols_interaction)
            }
        }
        
        # Log model comparisons
        self._log_model_comparisons()
        
        return results
    
    def _log_model_comparisons(self):
        """Log model comparison statistics."""
        # LRT: Null vs Full
        lrt_null_full = self.model_full.likelihood_ratio_test(self.model_null)
        logger.info(f"LRT (Null vs Full): χ²={lrt_null_full['lrt_statistic']:.2f}, "
                   f"p={lrt_null_full['p_value']:.4f}")
        
        # LRT: Full vs Interaction
        lrt_full_interaction = self.model_interaction.likelihood_ratio_test(self.model_full)
        logger.info(f"LRT (Full vs Interaction): χ²={lrt_full_interaction['lrt_statistic']:.2f}, "
                   f"p={lrt_full_interaction['p_value']:.4f}")
        
        # AIC comparisons
        aic_comparison = self.model_interaction.aic_comparison(self.model_full)
        logger.info(f"AIC comparison: {aic_comparison['interpretation']}")
    
    def get_interaction_coefficient(self) -> Dict[str, float]:
        """Extract and interpret the interaction coefficient.
        
        Returns:
            Dictionary with interaction coefficient and interpretation
        """
        if self.model_interaction is None:
            raise ValueError("Must fit models first")
        
        coef_df = self.model_interaction.get_coefficients()
        
        # Get interaction coefficient
        interaction_coef = coef_df.loc['plants_bio4_interaction', 'coefficient']
        interaction_pval = coef_df.loc['plants_bio4_interaction', 'p_value']
        interaction_se = coef_df.loc['plants_bio4_interaction', 'std_error']
        
        # Interpretation
        if interaction_pval < 0.05:
            if interaction_coef < 0:
                interpretation = ("Significant negative interaction: "
                                "Plant-fungi relationship weakens in temperate regions")
            else:
                interpretation = ("Significant positive interaction: "
                                "Plant-fungi relationship strengthens in temperate regions")
        else:
            interpretation = "No significant interaction between plants and seasonality"
        
        logger.info(f"Interaction coefficient: {interaction_coef:.4f} ± {interaction_se:.4f}, "
                   f"p={interaction_pval:.4f}")
        logger.info(f"Interpretation: {interpretation}")
        
        return {
            'coefficient': interaction_coef,
            'std_error': interaction_se,
            'p_value': interaction_pval,
            'exp_coefficient': np.exp(interaction_coef),
            'interpretation': interpretation,
            'supports_h2': interaction_coef < 0 and interaction_pval < 0.05
        }
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """Run DHARMa diagnostics on the interaction model.
        
        Returns:
            Dictionary with diagnostic results
        """
        from ...models.glm import DHARMaDiagnostics
        
        if self.model_interaction is None:
            raise ValueError("Must fit models first")
        
        logger.info("Running DHARMa diagnostics for interaction model")
        
        # Get data for diagnostics
        me_cols = [col for col in self.data_processed.columns if col.startswith('ME')]
        
        if 'plants_richness_log' in self.data_processed.columns:
            biotic_cols = ['plants_richness_log', 'vertebrates_richness_log']
        else:
            biotic_cols = ['plants_richness', 'vertebrates_richness']
        
        feature_cols = [
            'bio1_std', 'bio4_std', 'bio12_std', 'bio15_std', 
            'elevation_std', 'elevation_std_sq'  # Added quadratic term
        ] + me_cols + biotic_cols + ['plants_bio4_interaction']
        
        X = self.data_processed[feature_cols]
        y = self.data_processed['fungi_richness']
        
        # Run diagnostics
        diagnostics = DHARMaDiagnostics(self.model_interaction)
        results = diagnostics.run_all_tests(
            X, y,
            longitude=self.data_processed['longitude'].values,
            latitude=self.data_processed['latitude'].values
        )
        
        return results
    
    def create_visualization(self, save_path: Optional[str] = None) -> Any:
        """Create visualization for Method 1 interaction results.
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        from ...visualization import create_method1_interaction_plots
        
        if self.model_interaction is None:
            raise ValueError("Must fit models first")
        
        # Gather model results
        model_results = {
            'models': {
                'null': {
                    'aic': self.model_null.model_result.aic,
                    'bic': self.model_null.model_result.bic,
                    'llf': self.model_null.model_result.llf
                },
                'full': {
                    'aic': self.model_full.model_result.aic,
                    'bic': self.model_full.model_result.bic,
                    'llf': self.model_full.model_result.llf
                },
                'interaction': {
                    'aic': self.model_interaction.model_result.aic,
                    'bic': self.model_interaction.model_result.bic,
                    'llf': self.model_interaction.model_result.llf
                }
            },
            'interaction_term': self.get_interaction_coefficient()
        }
        
        # Create visualization
        fig = create_method1_interaction_plots(
            self.data_processed,
            model_results,
            save_path=save_path
        )
        
        logger.info("Created Method 1 visualization")
        return fig
    
    def summarize_results(self) -> Dict[str, Any]:
        """Summarize Method 1 results.
        
        Returns:
            Dictionary with complete Method 1 results
        """
        if self.model_interaction is None:
            raise ValueError("Must fit models first")
        
        interaction_results = self.get_interaction_coefficient()
        
        summary = {
            'method': 'Method 1: Continuous Interaction GLM',
            'models': {
                'null': {
                    'aic': self.model_null.model_result.aic,
                    'bic': self.model_null.model_result.bic,
                    'llf': self.model_null.model_result.llf
                },
                'full': {
                    'aic': self.model_full.model_result.aic,
                    'bic': self.model_full.model_result.bic,
                    'llf': self.model_full.model_result.llf
                },
                'interaction': {
                    'aic': self.model_interaction.model_result.aic,
                    'bic': self.model_interaction.model_result.bic,
                    'llf': self.model_interaction.model_result.llf
                }
            },
            'interaction_term': interaction_results,
            'h2_test': {
                'supported': interaction_results['supports_h2'],
                'interpretation': interaction_results['interpretation']
            }
        }
        
        return summary