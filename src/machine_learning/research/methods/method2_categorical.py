"""Method 2: Categorical interaction model with pure H2O processing.

Tests H2: Climate-diversity interaction differs in temperate vs non-temperate regions.
Optimized for massive datasets using H2O-native operations.
"""

import os
import logging
import h2o
from pathlib import Path
from typing import Dict, Any, Optional, Union

from ...models.glm.h2o_negative_binomial import H2ONegativeBinomialGLM
from ...spatial.mesf_preprocessor import MESFPreprocessor
from ...config_loader import MLRegressionConfig

logger = logging.getLogger(__name__)


class Method2CategoricalGLM:
    """H2O-native implementation of Method 2 for massive datasets.
    
    Tests categorical interaction between plant diversity and temperate climate.
    All operations in H2O for maximum parallelization on 256-core systems.
    """
    
    def __init__(self, config: MLRegressionConfig):
        """Initialize Method 2 with H2O.
        
        Args:
            config: ML regression configuration
        """
        self.config = config
        self.preprocessing_config = self.config.get_preprocessing_config()
        self.glm_config = self.config.get_glm_config()
        self.spatial_config = self.config.get_spatial_config()
        
        # Initialize H2O
        self._ensure_h2o_initialized()
        
        # Initialize MESF preprocessor
        self.mesf_preprocessor = MESFPreprocessor()
        
        self.model = None
        self.data_h2o = None
        
    def _ensure_h2o_initialized(self):
        """Ensure H2O cluster is running."""
        try:
            h2o.cluster()
            logger.info("Using existing H2O cluster")
        except:
            n_threads = os.cpu_count() or 256
            logger.info(f"Starting H2O cluster with {n_threads} threads")
            h2o.init(
                nthreads=n_threads,
                max_mem_size="128G",
                min_mem_size="8G",
                strict_version_check=False
            )
    
    def load_data(self, parquet_path: str) -> h2o.H2OFrame:
        """Load data directly from parquet to H2O.
        
        Automatically handles MESF computation if not present.
        
        Args:
            parquet_path: Path to parquet file
            
        Returns:
            H2OFrame with all data including MESF
        """
        logger.info(f"Loading data from {parquet_path}")
        
        # Process parquet to ensure MESF is included
        augmented_path = self.mesf_preprocessor.process_and_augment_parquet(
            parquet_path,
            self.spatial_config,
            force_recompute=False
        )
        
        # Load directly to H2O (MESF already included!)
        logger.info("Loading augmented data to H2O...")
        h2o_data = h2o.import_file(augmented_path, parse_dates=False)
        
        # Verify MESF columns
        me_cols = [col for col in h2o_data.columns if col.startswith('ME')]
        logger.info(f"Loaded {h2o_data.nrows:,} rows with {h2o_data.ncols} columns")
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
                mean_val = h2o_data[var].mean()[0, 0]
                sd_val = h2o_data[var].sd()[0, 0]
                h2o_data[f'{var}_std'] = (h2o_data[var] - mean_val) / (2 * sd_val)
                logger.info(f"Standardized {var}: mean={mean_val:.2f}, 2*sd={2*sd_val:.2f}")
        
        # Add quadratic elevation term
        h2o_data['elevation_std_sq'] = h2o_data['elevation_std'] ** 2
        
        # Log transform biotic predictors
        h2o_data['plants_richness_log'] = (h2o_data['plants_richness'] + 1).log()
        h2o_data['vertebrates_richness_log'] = (h2o_data['vertebrates_richness'] + 1).log()
        logger.info("Created log-transformed biotic predictors")
        
        # Create categorical variables for interaction
        # Temperate: bio1 < median(bio1)
        bio1_median = h2o_data['bio1'].median()[0, 0]
        h2o_data['temperate'] = (h2o_data['bio1'] < bio1_median).ifelse(1, 0)
        
        # Plant diversity: high if > median
        plants_median = h2o_data['plants_richness'].median()[0, 0]
        h2o_data['plant_high'] = (h2o_data['plants_richness'] > plants_median).ifelse(1, 0)
        
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
        fungi_mean = h2o_data['fungi_richness'].mean()[0, 0]
        fungi_var = h2o_data['fungi_richness'].var()[0, 0]
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
        """Test the H2 hypothesis using contrasts.
        
        H2: The interaction term should be significant.
        
        Returns:
            Dictionary with contrast results
        """
        if self.model is None:
            raise ValueError("Must fit model first")
        
        # Get interaction coefficient
        interaction_info = self.model.get_coefficient('temperate_x_plant_high')
        
        # Determine if H2 is supported
        h2_supported = (
            interaction_info['p_value'] < 0.05 and 
            interaction_info['coefficient'] != 0
        )
        
        return {
            'interaction_coefficient': interaction_info['coefficient'],
            'interaction_p_value': interaction_info['p_value'],
            'interaction_se': interaction_info.get('std_error', None),
            'h2_supported': h2_supported,
            'interpretation': self._interpret_interaction(interaction_info)
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