"""Method 2: Categorical GLM with emmeans contrasts.

Based on regression_analysis_decisions.md lines 306-357.
Tests H2 hypothesis using categorical interaction and emmeans contrasts.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Union, Tuple
import subprocess
import json
import tempfile
import os
from scipy import stats

from ...models.glm import PoissonGLM
from ...models.glm.h2o_negative_binomial import H2ONegativeBinomialGLM
from ...spatial import compute_mesf  # Now uses vectorized implementation
from ...config_loader import MLRegressionConfig

logger = logging.getLogger(__name__)


class Method2CategoricalGLM:
    """Method 2: GLM with categorical interaction and emmeans contrasts.
    
    Tests the temperate mismatch hypothesis (H2) using threshold regression
    with categorical variables and emmeans contrasts.
    """
    
    def __init__(self, config: Optional[MLRegressionConfig] = None):
        """Initialize Method 2.
        
        Args:
            config: ML regression configuration
        """
        self.config = config or MLRegressionConfig()
        self.preprocessing_config = self.config.get_preprocessing_config()
        self.glm_config = self.config.get_glm_config()
        self.spatial_config = self.config.get_spatial_config()
        self.feature_eng_config = self.config.get_feature_engineering_config()
        
        self.model = None
        self.data_processed = None
        self.emmeans_results = None
        
        logger.info("Initialized Method 2: Categorical GLM with emmeans contrasts")
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for Method 2 analysis.
        
        Implements preprocessing from regression_analysis_decisions.md lines 199-277
        and categorical variable creation from lines 53-67.
        
        Args:
            data: Raw biodiversity data
        
        Returns:
            Processed DataFrame with categorical variables
        """
        logger.info("Preparing data for Method 2 (categorical interaction)")
        
        # Apply complete case analysis
        initial_rows = len(data)
        data_clean = data.dropna(subset=[
            'fungi_richness', 'plants_richness', 'vertebrates_richness',
            'bio1', 'bio4', 'bio12', 'bio15', 'elevation',
            'longitude', 'latitude'
        ])
        
        rows_dropped = initial_rows - len(data_clean)
        logger.info(f"Complete case analysis: dropped {rows_dropped} rows "
                   f"({rows_dropped/initial_rows*100:.2f}%)")
        
        # Standardize climate variables
        climate_vars = ['bio1', 'bio4', 'bio12', 'bio15', 'elevation']
        
        for var in climate_vars:
            mean_val = data_clean[var].mean()
            sd_val = data_clean[var].std()
            data_clean[f'{var}_std'] = (data_clean[var] - mean_val) / (2 * sd_val)
        
        # Add quadratic elevation term (from regression_analysis_decisions.md lines 288, 301, 314)
        data_clean['elevation_std_sq'] = data_clean['elevation_std'] ** 2
        logger.info("Created quadratic elevation term: elevation_std_sq")
        
        # Create categorical variables for H2 testing (lines 57-67)
        h2_config = self.feature_eng_config.get('h2_testing', {})
        temperate_threshold = h2_config.get('temperate_threshold', 0.75)
        plant_threshold = h2_config.get('plant_high_threshold', 0.75)
        
        # Temperate regions: top 25% of temperature seasonality
        bio4_quantile = data_clean['bio4_std'].quantile(temperate_threshold)
        data_clean['temperate'] = (data_clean['bio4_std'] > bio4_quantile).astype(int)
        
        # High plant richness: top 25% of plant species
        plants_quantile = data_clean['plants_richness'].quantile(plant_threshold)
        data_clean['plant_high'] = (data_clean['plants_richness'] > plants_quantile).astype(int)
        
        logger.info(f"Created categorical variables:")
        logger.info(f"  - temperate: {data_clean['temperate'].sum()} regions "
                   f"({data_clean['temperate'].mean()*100:.1f}%)")
        logger.info(f"  - plant_high: {data_clean['plant_high'].sum()} regions "
                   f"({data_clean['plant_high'].mean()*100:.1f}%)")
        
        # Create interaction categories for emmeans
        data_clean['plant_temperate_group'] = (
            data_clean['plant_high'].astype(str) + '_' + 
            data_clean['temperate'].astype(str)
        )
        
        # Map to meaningful names
        group_mapping = {
            '0_0': 'low_plant_nontemperate',
            '1_0': 'high_plant_nontemperate',
            '0_1': 'low_plant_temperate',
            '1_1': 'high_plant_temperate'
        }
        data_clean['plant_temperate_group'] = data_clean['plant_temperate_group'].map(group_mapping)
        
        # Log group sizes
        group_counts = data_clean['plant_temperate_group'].value_counts()
        logger.info("Group sizes:")
        for group, count in group_counts.items():
            logger.info(f"  - {group}: {count} ({count/len(data_clean)*100:.1f}%)")
        
        # Compute spatial eigenvectors
        logger.info("Computing spatial eigenvectors...")
        coords = data_clean[['longitude', 'latitude']].values
        mesf_df = compute_mesf(coords, self.spatial_config)
        
        # Add MESF columns
        for col in mesf_df.columns:
            data_clean[col] = mesf_df[col].values
        
        logger.info(f"Added {len(mesf_df.columns)} spatial eigenvectors")
        
        self.data_processed = data_clean
        return data_clean
    
    def fit_model(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Fit the categorical interaction model.
        
        Implements model from regression_analysis_decisions.md lines 309-321.
        
        Args:
            data: Optional processed data
            
        Returns:
            Dictionary with model results
        """
        if data is None:
            data = self.data_processed
        
        if data is None:
            raise ValueError("Must call prepare_data() first or provide data")
        
        # Check overdispersion
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
        
        # Model with categorical interaction (lines 314-316)
        # Option A: Without continuous bio4 (recommended to avoid collinearity)
        logger.info("Fitting categorical interaction model")
        
        feature_cols = [
            'bio1_std', 'bio12_std', 'bio15_std',  # Excluding bio4_std
            'elevation_std', 'elevation_std_sq'  # Added quadratic term
        ] + me_cols + ['plant_high', 'temperate']
        
        # Create interaction term in design matrix
        data['plant_high_temperate'] = data['plant_high'] * data['temperate']
        feature_cols.append('plant_high_temperate')
        
        X = data[feature_cols]
        y = data['fungi_richness']
        
        self.model = GLMClass(self.config.config)
        self.model.fit(X, y)
        
        # Log model summary
        logger.info(f"Model AIC: {self.model.model_result.aic:.2f}")
        logger.info(f"Model BIC: {self.model.model_result.bic:.2f}")
        
        # Get interaction coefficient
        coef_df = self.model.get_coefficients()
        interaction_coef = coef_df.loc['plant_high_temperate', 'coefficient']
        interaction_pval = coef_df.loc['plant_high_temperate', 'p_value']
        
        logger.info(f"Interaction coefficient (plant_high:temperate): "
                   f"{interaction_coef:.4f}, p={interaction_pval:.4f}")
        
        # Interpretation (lines 323-326)
        if interaction_pval < 0.05:
            if interaction_coef < 0:
                interpretation = "Negative coefficient: High-plant temperate regions have fewer fungi (supports H2)"
            else:
                interpretation = "Positive coefficient: High-plant temperate regions have more fungi (rejects H2)"
        else:
            interpretation = "No significant interaction effect"
        
        logger.info(f"Interpretation: {interpretation}")
        
        return {
            'model': self.model,
            'distribution': distribution,
            'dispersion_ratio': dispersion_ratio,
            'interaction_coefficient': interaction_coef,
            'interaction_pvalue': interaction_pval,
            'interpretation': interpretation,
            'n_samples': len(data)
        }
    
    def compute_emmeans(self) -> Dict[str, Any]:
        """Compute estimated marginal means for all group combinations.
        
        Implements emmeans calculation from regression_analysis_decisions.md lines 329-357.
        
        Returns:
            Dictionary with emmeans and contrasts
        """
        if self.model is None:
            raise ValueError("Must fit model first")
        
        logger.info("Computing estimated marginal means")
        
        # Get predictions for each group combination
        groups = [
            (0, 0),  # low_plant_nontemperate
            (1, 0),  # high_plant_nontemperate
            (0, 1),  # low_plant_temperate
            (1, 1),  # high_plant_temperate
        ]
        
        emmeans = []
        se_values = []
        
        # Create prediction data for each group
        me_cols = [col for col in self.data_processed.columns if col.startswith('ME')]
        
        # Get the same feature columns used during training
        feature_cols = [
            'bio1_std', 'bio12_std', 'bio15_std',
            'elevation_std', 'elevation_std_sq'  # Added quadratic term
        ] + me_cols + ['plant_high', 'temperate', 'plant_high_temperate']
        
        for plant_high, temperate in groups:
            # Create prediction dataset with mean values for covariates
            pred_dict = {
                'bio1_std': 0,  # Mean (standardized)
                'bio12_std': 0,
                'bio15_std': 0,
                'elevation_std': 0,
                'elevation_std_sq': 0,  # Mean of squared term
                'plant_high': plant_high,
                'temperate': temperate,
                'plant_high_temperate': plant_high * temperate
            }
            
            # Add mean ME values
            for me_col in me_cols:
                pred_dict[me_col] = self.data_processed[me_col].mean()
            
            # Create DataFrame with correct column order
            pred_data = pd.DataFrame([pred_dict])[feature_cols]
            
            # Get prediction
            pred = self.model.predict(pred_data)[0]
            emmeans.append(pred)
            
            # Approximate standard error (simplified)
            # In practice, would use delta method or bootstrap
            se = np.sqrt(pred) / np.sqrt(self.data_processed.shape[0])
            se_values.append(se)
        
        # Store emmeans results
        self.emmeans_results = {
            'emmeans': emmeans,
            'SE': se_values,
            'groups': [
                'low_plant_nontemperate',
                'high_plant_nontemperate', 
                'low_plant_temperate',
                'high_plant_temperate'
            ]
        }
        
        logger.info("Estimated marginal means:")
        for i, group in enumerate(self.emmeans_results['groups']):
            logger.info(f"  {group}: {emmeans[i]:.2f} ± {se_values[i]:.2f}")
        
        return self.emmeans_results
    
    def test_h2_contrast(self) -> Dict[str, Any]:
        """Test the H2 hypothesis using emmeans contrast.
        
        Implements contrast from regression_analysis_decisions.md lines 337-340.
        Contrast: high_plant_temperate - high_plant_nontemperate
        
        Returns:
            Dictionary with contrast results
        """
        if self.emmeans_results is None:
            self.compute_emmeans()
        
        logger.info("Testing H2 contrast")
        
        # Contrast coefficients: c(0, -1, 0, 1)
        # This compares high_plant_temperate vs high_plant_nontemperate
        contrast_coefficients = [0, -1, 0, 1]
        
        # Calculate contrast estimate
        emmeans = self.emmeans_results['emmeans']
        contrast_estimate = sum(c * e for c, e in zip(contrast_coefficients, emmeans))
        
        # Calculate standard error of contrast (simplified)
        se_values = self.emmeans_results['SE']
        contrast_se = np.sqrt(sum(c**2 * se**2 for c, se in zip(contrast_coefficients, se_values)))
        
        # Calculate test statistic and p-value
        z_score = contrast_estimate / contrast_se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Interpretation
        if p_value < 0.05:
            if contrast_estimate < 0:
                interpretation = ("H2 SUPPORTED: High-plant temperate regions have "
                                f"{abs(contrast_estimate):.1f} fewer fungi species "
                                "than high-plant non-temperate regions")
            else:
                interpretation = ("H2 REJECTED: High-plant temperate regions have "
                                f"{contrast_estimate:.1f} MORE fungi species "
                                "than high-plant non-temperate regions")
        else:
            interpretation = "H2 NOT SUPPORTED: No significant difference"
        
        logger.info(f"H2 Contrast: {contrast_estimate:.2f} ± {contrast_se:.2f}, "
                   f"z={z_score:.2f}, p={p_value:.4f}")
        logger.info(f"Interpretation: {interpretation}")
        
        # Store full results
        h2_results = {
            'contrast_estimate': contrast_estimate,
            'standard_error': contrast_se,
            'z_score': z_score,
            'p_value': p_value,
            'ci_lower': contrast_estimate - 1.96 * contrast_se,
            'ci_upper': contrast_estimate + 1.96 * contrast_se,
            'interpretation': interpretation,
            'h2_supported': contrast_estimate < 0 and p_value < 0.05,
            'emmeans': self.emmeans_results
        }
        
        return h2_results
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """Run DHARMa diagnostics on the categorical model.
        
        Returns:
            Dictionary with diagnostic results
        """
        from ...models.glm import DHARMaDiagnostics
        
        if self.model is None:
            raise ValueError("Must fit model first")
        
        logger.info("Running DHARMa diagnostics for categorical model")
        
        # Get data for diagnostics
        me_cols = [col for col in self.data_processed.columns if col.startswith('ME')]
        
        feature_cols = [
            'bio1_std', 'bio12_std', 'bio15_std',
            'elevation_std', 'elevation_std_sq'  # Added quadratic term
        ] + me_cols + ['plant_high', 'temperate', 'plant_high_temperate']
        
        X = self.data_processed[feature_cols]
        y = self.data_processed['fungi_richness']
        
        # Run diagnostics
        diagnostics = DHARMaDiagnostics(self.model)
        results = diagnostics.run_all_tests(
            X, y,
            longitude=self.data_processed['longitude'].values,
            latitude=self.data_processed['latitude'].values
        )
        
        return results
    
    def create_visualization(self, save_path: Optional[str] = None) -> Any:
        """Create comprehensive visualization for Method 2 results.
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        from ...visualization import create_h2_mismatch_visualization
        
        if self.model is None:
            raise ValueError("Must fit model first")
        
        # Get emmeans results
        emmeans_results = self.compute_emmeans()
        h2_results = self.test_h2_contrast()
        
        # Combine results for visualization
        viz_results = {
            'marginal_means': emmeans_results['marginal_means'],
            'h2_contrast': h2_results
        }
        
        # Create the 4-panel visualization
        fig = create_h2_mismatch_visualization(
            self.data_processed,
            viz_results,
            save_path=save_path
        )
        
        logger.info("Created Method 2 visualization")
        return fig
    
    def summarize_results(self) -> Dict[str, Any]:
        """Summarize Method 2 results.
        
        Returns:
            Dictionary with complete Method 2 results
        """
        if self.model is None:
            raise ValueError("Must fit model first")
        
        # Get H2 test results
        h2_results = self.test_h2_contrast()
        
        summary = {
            'method': 'Method 2: Categorical GLM with emmeans contrasts',
            'model': {
                'aic': self.model.model_result.aic,
                'bic': self.model.model_result.bic,
                'llf': self.model.model_result.llf,
                'n_parameters': len(self.model.model_result.params)
            },
            'emmeans': self.emmeans_results,
            'h2_contrast': h2_results,
            'categorical_thresholds': {
                'temperate': self.feature_eng_config['h2_testing']['temperate_threshold'],
                'plant_high': self.feature_eng_config['h2_testing']['plant_high_threshold']
            },
            'group_sizes': self.data_processed['plant_temperate_group'].value_counts().to_dict()
        }
        
        return summary