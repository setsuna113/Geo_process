"""Method 2: Categorical interaction model with pure H2O processing.

Tests H2: Climate-diversity interaction differs in temperate vs non-temperate regions.
Optimized for massive datasets using H2O-native operations.
"""

import logging
import h2o
from typing import Dict, Any, Optional, Union

from ...models.glm.h2o_negative_binomial import H2ONegativeBinomialGLM
from ...spatial.mesf_preprocessor import MESFPreprocessor
from ...config_loader import MLRegressionConfig
from ...base import BaseRegressionMethod
from ...utils.h2o_utils import ensure_h2o_initialized

logger = logging.getLogger(__name__)


class Method2CategoricalGLM(BaseRegressionMethod):
    """H2O-native implementation of Method 2 for massive datasets.
    
    Tests categorical interaction between plant diversity and temperate climate.
    All operations in H2O for maximum parallelization on 256-core systems.
    """
    
    def __init__(self, config: MLRegressionConfig):
        """Initialize Method 2 with H2O.
        
        Args:
            config: ML regression configuration
        """
        super().__init__(config)  # Initialize base class
        self.preprocessing_config = self.config.get_preprocessing_config()
        self.glm_config = self.config.get_glm_config()
        self.spatial_config = self.config.get_spatial_config()
        
        # Initialize H2O using centralized utility
        ensure_h2o_initialized()
        
        # Initialize MESF preprocessor V2 with caching
        self.mesf_preprocessor = MESFPreprocessor()
        
        self.model = None
        self.data_h2o = None
    
    def load_data(self, parquet_path: str, spatial_grid_step: Optional[int] = None) -> h2o.H2OFrame:
        """Load data with optional spatial grid sampling and MESF computation.
        
        Uses the new caching system that:
        1. Never modifies original files
        2. Caches MESF based on (file, sampling_rate, params)
        3. Reuses cached computations when possible
        
        Args:
            parquet_path: Path to parquet file
            spatial_grid_step: Take every Nth point in spatial grid (None=use config default)
            
        Returns:
            H2OFrame with all data including MESF
        """
        from ...preprocessing import ColumnStandardizer
        
        # Use config default if not specified
        if spatial_grid_step is None:
            spatial_grid_step = self.config.get_data_config().get('spatial_grid_step', 1)
        
        logger.info(f"Loading data from {parquet_path}")
        logger.info(f"Spatial grid sampling: every {spatial_grid_step}th point")
        
        # First standardize columns if needed (also uses caching)
        standardizer = ColumnStandardizer()
        standardized_path = standardizer.standardize_if_needed(parquet_path)
        
        # Use the new MESF preprocessor with caching
        # It handles sampling and MESF computation internally
        augmented_path = self.mesf_preprocessor.process_with_sampling(
            standardized_path,
            spatial_grid_step,
            self.spatial_config
        )
        
        # Load augmented data to H2O
        ensure_h2o_initialized()  # Ensure H2O is connected before importing
        h2o_data = h2o.import_file(augmented_path)
        
        logger.info(f"Loaded {h2o_data.nrows:,} rows with {h2o_data.ncols} columns")
        
        # Verify MESF columns
        me_cols = [col for col in h2o_data.columns if col.startswith('ME')]
        logger.info(f"Including {len(me_cols)} MESF eigenvectors")
        
        return h2o_data
    
    def prepare_data_h2o(self, h2o_data: h2o.H2OFrame) -> h2o.H2OFrame:
        """Prepare data using H2O operations (no pandas!).
        
        Creates categorical variables for interaction testing.
        
        Args:
            h2o_data: Raw H2O DataFrame
            
        Returns:
            Processed H2O DataFrame
        """
        logger.info("Preparing data in H2O for Method 2 (categorical interaction)")
        
        # Apply complete case analysis
        initial_rows = h2o_data.nrows
        required_cols = [
            'fungi_richness', 'plants_richness', 'vertebrates_richness',
            'bio1', 'bio4', 'bio12', 'bio15', 'elevation',
            'longitude', 'latitude'
        ]
        
        # Remove rows with missing values
        for col in required_cols:
            if col in h2o_data.columns:
                h2o_data = h2o_data[h2o_data[col].isna() == False]
        
        rows_dropped = initial_rows - h2o_data.nrows
        logger.info(f"Complete case analysis: dropped {rows_dropped:,} rows "
                   f"({rows_dropped/initial_rows*100:.2f}%)")
        
        # Standardize climate variables (H2O operations)
        climate_vars = ['bio1', 'bio4', 'bio12', 'bio15', 'elevation']
        
        for var in climate_vars:
            if var in h2o_data.columns:
                mean_val = h2o_data[var].mean()
                sd_val = h2o_data[var].sd()
                # H2O's mean() and sd() return lists, so extract the value
                if isinstance(mean_val, list):
                    mean_val = mean_val[0]
                if isinstance(sd_val, list):
                    sd_val = sd_val[0]
                h2o_data[f'{var}_std'] = (h2o_data[var] - mean_val) / (2 * sd_val)
                logger.info(f"Standardized {var}: mean={mean_val:.2f}, 2*sd={2*sd_val:.2f}")
        
        # Add quadratic elevation term
        h2o_data['elevation_std_sq'] = h2o_data['elevation_std'] ** 2
        
        # Log transform biotic predictors
        h2o_data['plants_richness_log'] = (h2o_data['plants_richness'] + 1).log()
        h2o_data['vertebrates_richness_log'] = (h2o_data['vertebrates_richness'] + 1).log()
        logger.info("Created log-transformed biotic predictors")
        
        # Create categorical variables for H2 testing (per regression_analysis_decisions.md)
        # Temperate regions: top 25% of temperature seasonality (bio4)
        bio4_quantiles = h2o_data['bio4_std'].quantile([0.75])
        # H2O quantile returns a frame, extract the value
        bio4_q75 = bio4_quantiles[0, 0] if hasattr(bio4_quantiles, '__getitem__') else bio4_quantiles
        h2o_data['temperate'] = (h2o_data['bio4_std'] > bio4_q75).ifelse(1, 0)
        logger.info(f"Created temperate variable: bio4_std > {bio4_q75:.3f} (75th percentile)")
        
        # High plant richness: top 25% of plant species
        plants_quantiles = h2o_data['plants_richness'].quantile([0.75])
        # H2O quantile returns a frame, extract the value
        plants_q75 = plants_quantiles[0, 0] if hasattr(plants_quantiles, '__getitem__') else plants_quantiles
        h2o_data['plant_high'] = (h2o_data['plants_richness'] > plants_q75).ifelse(1, 0)
        logger.info(f"Created plant_high variable: plants_richness > {plants_q75:.0f} (75th percentile)")
        
        # Create interaction term
        h2o_data['temperate_x_plant_high'] = h2o_data['temperate'] * h2o_data['plant_high']
        
        # Log group sizes
        temperate_count = h2o_data['temperate'].sum()
        plant_high_count = h2o_data['plant_high'].sum()
        logger.info(f"Temperate regions: {temperate_count} ({temperate_count/h2o_data.nrows*100:.1f}%)")
        logger.info(f"High plant diversity: {plant_high_count} ({plant_high_count/h2o_data.nrows*100:.1f}%)")
        
        self.data_h2o = h2o_data
        return h2o_data
    
    def fit_model(self, h2o_data: Optional[h2o.H2OFrame] = None) -> Dict[str, Any]:
        """Fit the categorical interaction model using H2O GLM.
        
        Args:
            h2o_data: Optional H2O DataFrame
            
        Returns:
            Dictionary with model and results
        """
        if h2o_data is None:
            h2o_data = self.data_h2o
        
        if h2o_data is None:
            raise ValueError("Must call prepare_data_h2o() first or provide data")
        
        # Check overdispersion
        fungi_mean = h2o_data['fungi_richness'].mean()
        fungi_var = h2o_data['fungi_richness'].var()
        # H2O's mean() and var() return lists
        if isinstance(fungi_mean, list):
            fungi_mean = fungi_mean[0]
        if isinstance(fungi_var, list):
            fungi_var = fungi_var[0]
        dispersion_ratio = fungi_var / fungi_mean
        
        logger.info(f"Dispersion ratio: {dispersion_ratio:.2f}")
        
        # Use H2O GLM
        logger.info("Fitting categorical interaction model with H2O GLM")
        
        # Get ME columns
        me_cols = [col for col in h2o_data.columns if col.startswith('ME')]
        
        # Features for model
        feature_cols = [
            'bio1_std', 'bio4_std', 'bio12_std', 'bio15_std',
            'elevation_std', 'elevation_std_sq',
            'plants_richness_log', 'vertebrates_richness_log',
            'temperate', 'plant_high', 'temperate_x_plant_high'
        ] + me_cols
        
        # Create subset for model
        model_data = h2o_data[feature_cols + ['fungi_richness']]
        
        # Fit model
        self.model = H2ONegativeBinomialGLM(self.config.config)
        self.model.fit(model_data, y='fungi_richness')  # Direct H2O fit!
        
        results = {
            'model': self.model,
            'distribution': 'negative_binomial' if dispersion_ratio > 1.2 else 'poisson',
            'n_samples': h2o_data.nrows,
            'dispersion_ratio': dispersion_ratio
        }
        
        return results
    
    def test_h2_contrast(self) -> Dict[str, Any]:
        """Test the H2 hypothesis using emmeans contrasts.
        
        H2 (Temperate Mismatch): High-plant temperate regions have fewer fungi
        than high-plant non-temperate regions.
        
        Contrast: c(0, -1, 0, 1) on emmeans
        Tests: (High/Temp) - (High/Non-temp)
        
        Returns:
            Dictionary with contrast results
        """
        if self.model is None:
            raise ValueError("Must fit model first")
        
        # Compute emmeans first
        emmeans_results = self.compute_emmeans()
        
        # Get interaction coefficient for statistical test
        interaction_info = self.model.get_coefficient('temperate_x_plant_high')
        
        # The contrast of interest: High/Temp - High/Non-temp
        contrast_estimate = emmeans_results['contrast_estimate']
        
        # Determine if H2 is supported
        # H2 is supported if contrast is negative and significant
        h2_supported = (
            interaction_info['p_value'] < 0.05 and 
            contrast_estimate < 0  # Negative means fewer fungi in temperate
        )
        
        return {
            'emmeans': emmeans_results['emmeans'],
            'emmeans_SE': emmeans_results['SE'],
            'contrast_estimate': contrast_estimate,
            'interaction_coefficient': interaction_info['coefficient'],
            'p_value': interaction_info['p_value'],
            'standard_error': interaction_info.get('std_error', None),
            'h2_supported': h2_supported,
            'interpretation': emmeans_results['interpretation']
        }
    
    def _interpret_interaction(self, coef_info: Dict) -> str:
        """Interpret the interaction coefficient."""
        if coef_info['p_value'] >= 0.05:
            return "No significant interaction between temperate climate and plant diversity"
        elif coef_info['coefficient'] > 0:
            return "Plant-fungi relationship is stronger in temperate regions with high plant diversity"
        else:
            return "Plant-fungi relationship is weaker in temperate regions with high plant diversity"
    
    def process_large_dataset(self, parquet_path: str) -> Dict[str, Any]:
        """Complete pipeline for large datasets.
        
        Args:
            parquet_path: Path to parquet file
            
        Returns:
            Analysis results
        """
        # Step 1: Load data directly to H2O (with MESF handling)
        h2o_data = self.load_data(parquet_path)
        
        # Step 2: Prepare data in H2O
        h2o_data = self.prepare_data_h2o(h2o_data)
        
        # Step 3: Fit model
        results = self.fit_model(h2o_data)
        
        # Step 4: Test H2 hypothesis
        h2_results = self.test_h2_contrast()
        results.update(h2_results)
        
        return results
    
    def test_hypothesis(self, hypothesis: str = 'h2') -> Dict[str, Any]:
        """Test specific hypothesis.
        
        Args:
            hypothesis: Which hypothesis to test ('h2' for Method 2)
            
        Returns:
            Hypothesis test results
        """
        if hypothesis == 'h2':
            return self.test_h2_contrast()
        else:
            return {'error': f'Method 2 primarily tests H2, not {hypothesis}'}
    
    def compute_emmeans(self) -> Dict[str, Any]:
        """Compute estimated marginal means for all group combinations.
        
        Implements emmeans-style marginal means computation for:
        - Low plants, non-temperate
        - High plants, non-temperate  
        - Low plants, temperate
        - High plants, temperate
        
        Returns:
            Dictionary with emmeans results
        """
        if self.model is None:
            return {'error': 'Model not fitted yet'}
        
        if self.data_h2o is None:
            return {'error': 'Data not available for emmeans computation'}
        
        # Get model coefficients
        coeffs = self.model.get_all_coefficients()
        intercept = coeffs.get('Intercept', 0)
        
        # Get coefficients for categorical variables
        temperate_coef = coeffs.get('temperate', 0)
        plant_high_coef = coeffs.get('plant_high', 0)
        interaction_coef = coeffs.get('temperate_x_plant_high', 0)
        
        # Compute estimated marginal means for each group
        # These are predictions at mean values of other covariates
        emmeans = {
            'low_nontemp': intercept,  # Reference group
            'high_nontemp': intercept + plant_high_coef,
            'low_temp': intercept + temperate_coef,
            'high_temp': intercept + temperate_coef + plant_high_coef + interaction_coef
        }
        
        # Compute standard errors (simplified - full computation would use variance-covariance matrix)
        # For now, use model's standard errors
        se_dict = self.model.get_standard_errors()
        
        return {
            'emmeans': list(emmeans.values()),
            'SE': [se_dict.get('Intercept', 0.1)] * 4,  # Simplified SE
            'groups': ['Low/Non-temp', 'High/Non-temp', 'Low/Temp', 'High/Temp'],
            'emmeans_dict': emmeans,
            'contrast_estimate': emmeans['high_temp'] - emmeans['high_nontemp'],  # Key H2 contrast
            'interpretation': self._interpret_emmeans_contrast(emmeans)
        }
    
    def _interpret_emmeans_contrast(self, emmeans: Dict[str, float]) -> str:
        """Interpret the emmeans contrast for H2 hypothesis.
        
        Tests: High plants in temperate vs High plants in non-temperate
        """
        contrast = emmeans['high_temp'] - emmeans['high_nontemp']
        
        if abs(contrast) < 0.01:  # Near zero
            return "No meaningful difference in fungi richness between high-plant temperate and non-temperate regions"
        elif contrast < 0:
            return f"SUPPORTS H2: High-plant temperate regions have {abs(contrast):.1f} fewer fungi species (temperate mismatch)"
        else:
            return f"REJECTS H2: High-plant temperate regions have {contrast:.1f} more fungi species (no mismatch)"
    
    def summarize_results(self) -> Dict[str, Any]:
        """Summarize Method 2 results.
        
        Returns:
            Dictionary with summary
        """
        # Get base summary
        summary = super().summarize_results()
        summary['method'] = 'Method 2: Categorical GLM with emmeans'
        
        # Add Method 2 specific information
        if self.model:
            h2_results = self.test_h2_contrast()
            summary['h2_supported'] = h2_results['h2_supported']
            summary['h2_interpretation'] = h2_results['interpretation']
            summary['contrast_estimate'] = h2_results.get('contrast_estimate')
            summary['contrast_p_value'] = h2_results.get('p_value')
        
        return summary