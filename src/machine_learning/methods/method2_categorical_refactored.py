"""Method 2: Categorical interaction model (REFACTORED).

Tests H2: Climate-diversity interaction differs in temperate vs non-temperate regions.
This refactored version assumes data is already preprocessed by the pipeline.
"""

import logging
import h2o
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from src.base.ml import BaseRegressionMethod
from src.abstractions.types.ml_types import ModelType, ModelMetadata
from ...models.glm.h2o_negative_binomial import H2ONegativeBinomialGLM
from ...config_loader import MLRegressionConfig
from ...preprocessing import (
    create_temperate_categories,
    create_biodiversity_categories,
    add_categorical_interactions
)

logger = logging.getLogger(__name__)


class Method2CategoricalGLM(BaseRegressionMethod):
    """Refactored Method 2 - expects preprocessed data from pipeline.
    
    Tests categorical interaction between plant diversity and temperate climate.
    
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
        super().__init__(config)
        self.glm_config = self.config.get_glm_config()
        
        self.model = None
        self.data_h2o = None
        self.emmeans_results = None
        self.contrast_results = None
    
    def add_categorical_features(self, h2o_data: h2o.H2OFrame) -> h2o.H2OFrame:
        """Add Method2-specific categorical features.
        
        Args:
            h2o_data: Preprocessed data from pipeline
            
        Returns:
            Data with categorical features added
        """
        logger.info("Adding Method2-specific categorical features")
        
        # Create temperate category (bio4 > 75th percentile)
        # High temperature seasonality = temperate regions
        h2o_data = create_temperate_categories(
            h2o_data,
            bio4_col='bio4_std',  # Use bio4 (temperature seasonality) not bio1!
            output_col='temperate',
            method='quantile',
            quantile=0.75  # Top 25% as per regression_analysis_decisions.md
        )
        
        # Create plant diversity category (plants > 75th percentile)
        h2o_data = create_biodiversity_categories(
            h2o_data,
            richness_col='plants_richness',
            output_col='plant_high',
            method='quantile',
            quantile=0.75  # Top 25% as per regression_analysis_decisions.md
        )
        
        # Create interaction term
        h2o_data = add_categorical_interactions(
            h2o_data,
            cat1='temperate',
            cat2='plant_high',
            output_col='temperate_x_plant_high'
        )
        
        # Log categorical levels
        temperate_counts = h2o_data['temperate'].table()
        plant_high_counts = h2o_data['plant_high'].table()
        
        logger.info(f"Temperate regions: {temperate_counts}")
        logger.info(f"High plant diversity: {plant_high_counts}")
        
        return h2o_data
    
    def fit_model(self, h2o_data: h2o.H2OFrame) -> Dict[str, Any]:
        """Fit categorical interaction model.
        
        Args:
            h2o_data: Preprocessed data from pipeline
            
        Returns:
            Model fitting results
        """
        logger.info("Fitting Method 2 categorical interaction model")
        
        # Store data
        self.data_h2o = h2o_data
        
        # Add Method2-specific categorical features
        h2o_data = self.add_categorical_features(h2o_data)
        
        # Get MESF columns
        me_cols = [col for col in h2o_data.columns if col.startswith('ME')]
        logger.info(f"Using {len(me_cols)} MESF eigenvectors")
        
        # Build feature list for categorical model
        feature_cols = [
            # Climate variables (standardized)
            'bio1_std', 'bio4_std', 'bio12_std', 'bio15_std',
            'elevation_std', 'elevation_std_sq',
            # Biodiversity (log-transformed)
            'plants_richness_log', 'vertebrates_richness_log',
            # Categorical variables and interaction
            'temperate', 'plant_high', 'temperate_x_plant_high'
        ] + me_cols
        
        # Create model data
        model_data = h2o_data[feature_cols + ['fungi_richness']]
        
        # Fit H2O GLM
        GLMClass = H2ONegativeBinomialGLM
        self.model = GLMClass(self.config.config)
        self.model.fit(model_data, y='fungi_richness')
        
        logger.info("Model fitting complete")
        
        # Get model statistics
        results = {
            'aic': self.model.get_aic(),
            'bic': self.model.get_bic(),
            'deviance': self.model.get_deviance(),
            'coefficients': self.model.get_coefficients()
        }
        
        return results
    
    def compute_emmeans(self) -> Dict[str, Any]:
        """Compute estimated marginal means for categorical groups.
        
        Returns:
            Dictionary with emmeans results
        """
        if self.model is None:
            raise ValueError("Must fit model first")
        
        logger.info("Computing estimated marginal means")
        
        # Get predictions for each combination
        # Create prediction dataset with all 4 combinations
        pred_data = self.data_h2o.copy()
        
        # Get means for other variables
        me_cols = [col for col in pred_data.columns if col.startswith('ME')]
        
        # Create 4 groups
        groups = [
            {'temperate': 0, 'plant_high': 0, 'label': 'non_temp_low_plant'},
            {'temperate': 0, 'plant_high': 1, 'label': 'non_temp_high_plant'},
            {'temperate': 1, 'plant_high': 0, 'label': 'temp_low_plant'},
            {'temperate': 1, 'plant_high': 1, 'label': 'temp_high_plant'}
        ]
        
        emmeans = {}
        for group in groups:
            # Set categorical values
            subset = pred_data[
                (pred_data['temperate'] == group['temperate']) &
                (pred_data['plant_high'] == group['plant_high'])
            ]
            
            if subset.nrows > 0:
                predictions = self.model.predict(subset)
                mean_pred = predictions.mean()
                if isinstance(mean_pred, list):
                    mean_pred = mean_pred[0]
                
                emmeans[group['label']] = {
                    'mean': mean_pred,
                    'n': subset.nrows
                }
        
        self.emmeans_results = emmeans
        
        logger.info("Estimated marginal means:")
        for label, values in emmeans.items():
            logger.info(f"  {label}: {values['mean']:.3f} (n={values['n']})")
        
        return emmeans
    
    def test_h2_contrast(self) -> Dict[str, Any]:
        """Test H2 hypothesis using contrasts.
        
        H2: The effect of plant diversity on fungi is different in temperate vs non-temperate.
        This tests the interaction term significance.
        
        Returns:
            Dictionary with contrast test results
        """
        if self.model is None:
            raise ValueError("Must fit model first")
        
        # Get interaction coefficient
        interaction_coef = self.model.get_coefficient('temperate_x_plant_high')
        
        # Compute contrast
        # The interaction tests if the plant effect differs by temperature
        contrast_estimate = interaction_coef['coefficient']
        contrast_se = interaction_coef['std_error']
        contrast_z = contrast_estimate / contrast_se if contrast_se > 0 else 0
        contrast_p = interaction_coef['p_value']
        
        # Determine if H2 is supported
        h2_supported = contrast_p < 0.05
        
        self.contrast_results = {
            'contrast_estimate': contrast_estimate,
            'contrast_se': contrast_se,
            'contrast_z': contrast_z,
            'p_value': contrast_p,
            'h2_supported': h2_supported,
            'interpretation': self._interpret_contrast(contrast_estimate, contrast_p)
        }
        
        logger.info(f"H2 contrast test: estimate={contrast_estimate:.4f}, p={contrast_p:.4f}")
        logger.info(f"H2 supported: {h2_supported}")
        
        return self.contrast_results
    
    def _interpret_contrast(self, estimate: float, p_value: float) -> str:
        """Interpret the contrast results.
        
        Args:
            estimate: Contrast estimate
            p_value: P-value
            
        Returns:
            Interpretation string
        """
        if p_value >= 0.05:
            return "No significant interaction between temperature and plant diversity"
        
        if estimate > 0:
            return ("Significant positive interaction: Plant-fungi relationship is "
                   "stronger in temperate regions with high plant diversity")
        else:
            return ("Significant negative interaction: Plant-fungi relationship is "
                   "weaker in temperate regions with high plant diversity")
    
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
        if self.model is None:
            raise ValueError("Must fit model first")
        
        return self.model.predict(data)