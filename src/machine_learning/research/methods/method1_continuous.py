"""Method 1: Continuous interaction model with H2O-native processing.

Optimized for large-scale data processing with H2O, avoiding pandas conversion.
Tests H2: Climate-diversity interaction differs in temperate vs non-temperate regions.
"""

import logging
import numpy as np
import h2o
from typing import Dict, Any, Optional, Union, Tuple

from ...models.glm.h2o_negative_binomial import H2ONegativeBinomialGLM
from ...spatial.mesf_preprocessor import MESFPreprocessor
from ...config_loader import MLRegressionConfig
from ...base import BaseRegressionMethod
from ...utils.h2o_utils import ensure_h2o_initialized

logger = logging.getLogger(__name__)


class Method1ContinuousGLM(BaseRegressionMethod):
    """H2O-native implementation of Method 1 for massive datasets.
    
    Key improvements:
    1. Direct parquet to H2O loading (no pandas intermediate)
    2. MESF computed separately and cached
    3. All feature engineering in H2O
    4. No memory duplication
    """
    
    def __init__(self, config: MLRegressionConfig):
        """Initialize Method 1 with H2O.
        
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
        
        self.model_null = None
        self.model_full = None
        self.model_interaction = None
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
        h2o_data = h2o.import_file(augmented_path)
        
        logger.info(f"Loaded {h2o_data.nrows:,} rows with {h2o_data.ncols} columns")
        
        # Verify MESF columns
        me_cols = [col for col in h2o_data.columns if col.startswith('ME')]
        logger.info(f"Including {len(me_cols)} MESF eigenvectors")
        
        return h2o_data
    
    def prepare_data_h2o(self, h2o_data: h2o.H2OFrame) -> h2o.H2OFrame:
        """Prepare data using H2O operations (no pandas!).
        
        Args:
            h2o_data: Raw H2O DataFrame
            
        Returns:
            Processed H2O DataFrame
        """
        logger.info("Preparing data in H2O (parallel processing)")
        
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
        
        # Standardize climate variables using 2SD method (H2O operations)
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
        logger.info("Created quadratic elevation term")
        
        # Handle biotic predictors
        transformation = self.preprocessing_config['biotic_predictors']['transformation']
        
        if transformation in ['log', 'both']:
            # Log transformation in H2O
            h2o_data['plants_richness_log'] = (h2o_data['plants_richness'] + 1).log()
            h2o_data['vertebrates_richness_log'] = (h2o_data['vertebrates_richness'] + 1).log()
            logger.info("Created log-transformed biotic predictors")
        
        self.data_h2o = h2o_data
        return h2o_data
    
    def create_interaction_term_h2o(self, h2o_data: h2o.H2OFrame) -> h2o.H2OFrame:
        """Create interaction term in H2O.
        
        Args:
            h2o_data: Processed H2O data
            
        Returns:
            H2O data with interaction term
        """
        # Create interaction: plants_richness_log × bio4_std
        if 'plants_richness_log' in h2o_data.columns:
            h2o_data['plants_bio4_interaction'] = (
                h2o_data['plants_richness_log'] * h2o_data['bio4_std']
            )
            logger.info("Created interaction term: plants_richness_log × bio4_std")
        else:
            # Use raw counts if log not available
            h2o_data['plants_bio4_interaction'] = (
                h2o_data['plants_richness'] * h2o_data['bio4_std']
            )
            logger.info("Created interaction term: plants_richness × bio4_std")
        
        return h2o_data
    
    def fit_model(self, data: Any) -> Dict[str, Any]:
        """Fit the model - implements abstract method from base class.
        
        Args:
            data: Input data (H2OFrame or None to use self.data_h2o)
            
        Returns:
            Model fitting results
        """
        return self.fit_models(data)
    
    def fit_models(self, h2o_data: Optional[h2o.H2OFrame] = None) -> Dict[str, Any]:
        """Fit the nested models using H2O GLM.
        
        Args:
            h2o_data: Optional H2O DataFrame (uses self.data_h2o if None)
            
        Returns:
            Dictionary with fitted models and results
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
        
        # Always use H2O GLM (handles both Poisson and Negative Binomial)
        GLMClass = H2ONegativeBinomialGLM
        logger.info("Using H2O GLM for massive parallelization")
        
        # Get ME columns
        me_cols = [col for col in h2o_data.columns if col.startswith('ME')]
        
        # Model 1: Climate only (null model)
        logger.info("Fitting Model 1: Climate only (null)")
        feature_cols_null = [
            'bio1_std', 'bio4_std', 'bio12_std', 'bio15_std', 
            'elevation_std', 'elevation_std_sq'
        ] + me_cols
        
        # Create H2O subset for null model
        null_data = h2o_data[feature_cols_null + ['fungi_richness']]
        
        self.model_null = GLMClass(self.config.config)
        self.model_null.fit(null_data, y='fungi_richness')  # Direct H2O fit!
        
        # Model 2: Climate + biodiversity (full model)
        logger.info("Fitting Model 2: Climate + biodiversity (full)")
        biotic_cols = ['plants_richness_log', 'vertebrates_richness_log'] \
                     if 'plants_richness_log' in h2o_data.columns \
                     else ['plants_richness', 'vertebrates_richness']
        
        feature_cols_full = feature_cols_null + biotic_cols
        full_data = h2o_data[feature_cols_full + ['fungi_richness']]
        
        self.model_full = GLMClass(self.config.config)
        self.model_full.fit(full_data, y='fungi_richness')
        
        # Model 3: With interaction
        logger.info("Fitting Model 3: With interaction")
        h2o_data = self.create_interaction_term_h2o(h2o_data)
        
        feature_cols_interaction = feature_cols_full + ['plants_bio4_interaction']
        interaction_data = h2o_data[feature_cols_interaction + ['fungi_richness']]
        
        self.model_interaction = GLMClass(self.config.config)
        self.model_interaction.fit(interaction_data, y='fungi_richness')
        
        # Compute model comparisons using GLMComparison
        from ...evaluation.statistical_tests import GLMComparison
        
        # Compare null vs full (H0 vs H1)
        comparison_h1 = GLMComparison.comprehensive_glm_comparison(
            self.model_null, self.model_full
        )
        
        # Compare full vs interaction (H1 vs H2)
        comparison_h2 = GLMComparison.comprehensive_glm_comparison(
            self.model_full, self.model_interaction
        )
        
        results = {
            'distribution': 'negative_binomial' if dispersion_ratio > 1.2 else 'poisson',
            'n_samples': h2o_data.nrows,
            'model_null': self.model_null,
            'model_full': self.model_full,
            'model_interaction': self.model_interaction,
            'dispersion_ratio': dispersion_ratio,
            'h1_test': comparison_h1,  # H0 vs H1 comparison
            'h2_test': comparison_h2   # Test for interaction significance
        }
        
        logger.info(f"H1 test (biodiversity effect): {comparison_h1['summary']}")
        # Handle case where p_value might be 'NA' string
        h2_p = comparison_h2.get('lrt', {}).get('p_value', 'NA')
        if isinstance(h2_p, (int, float)):
            logger.info(f"H2 test (interaction): LRT p={h2_p:.4f}")
        else:
            logger.info(f"H2 test (interaction): LRT p={h2_p}")
        
        return results
    
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
        
        # Step 3: Fit models
        results = self.fit_models(h2o_data)
        
        # Step 4: Get interaction coefficient
        interaction_results = self.get_interaction_coefficient()
        results.update(interaction_results)
        
        return results
    
    def get_interaction_coefficient(self) -> Dict[str, Any]:
        """Extract and test the interaction coefficient.
        
        Returns:
            Dictionary with coefficient, p-value, and H2 support determination
        """
        if self.model_interaction is None:
            raise ValueError("Must fit models first")
        
        # Get coefficient for plants_bio4_interaction
        coef_info = self.model_interaction.get_coefficient('plants_bio4_interaction')
        
        return {
            'interaction_coefficient': coef_info['coefficient'],
            'interaction_p_value': coef_info['p_value'],
            'supports_h2': coef_info['p_value'] < 0.05 and coef_info['coefficient'] != 0,
            'interpretation': self._interpret_interaction(coef_info)
        }
    
    def _interpret_interaction(self, coef_info: Dict) -> str:
        """Interpret the interaction coefficient."""
        if coef_info['p_value'] >= 0.05:
            return "No significant interaction between plant diversity and temperature seasonality"
        elif coef_info['coefficient'] > 0:
            return "Positive interaction: Plant-fungi relationship strengthens with temperature seasonality"
        else:
            return "Negative interaction: Plant-fungi relationship weakens with temperature seasonality"
    
    def test_hypothesis(self, hypothesis: str = 'h2') -> Dict[str, Any]:
        """Test specific hypothesis.
        
        Args:
            hypothesis: Which hypothesis to test (mainly 'h2' for Method 1)
            
        Returns:
            Hypothesis test results
        """
        if hypothesis == 'h2':
            return self.get_interaction_coefficient()
        else:
            return {'error': f'Method 1 primarily tests H2, not {hypothesis}'}
    
    def summarize_results(self) -> Dict[str, Any]:
        """Summarize Method 1 results.
        
        Returns:
            Dictionary with summary
        """
        # Get base summary
        summary = super().summarize_results()
        summary['method'] = 'Method 1: Continuous GLM with interaction'
        
        # Add Method 1 specific information
        if self.model_interaction:
            h2_results = self.get_interaction_coefficient()
            summary['h2_supported'] = h2_results['supports_h2']
            summary['h2_interpretation'] = h2_results['interpretation']
            summary['interaction_coefficient'] = h2_results.get('interaction_coefficient')
            summary['interaction_p_value'] = h2_results.get('interaction_p_value')
        
        # Use the proper model for diagnostics
        self.model = self.model_interaction or self.model_full or self.model_null
        
        return summary