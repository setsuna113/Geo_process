"""Method 1: Continuous interaction model (REFACTORED).

Tests H2: Climate-diversity interaction differs in temperate vs non-temperate regions.
This refactored version assumes data is already preprocessed by the pipeline.
"""

import logging
import numpy as np
import h2o
from typing import Dict, Any, Optional, Union, Tuple, List
from pathlib import Path

from src.base.ml import BaseRegressionMethod
from src.abstractions.types.ml_types import ModelType, ModelMetadata
from ...models.glm.h2o_negative_binomial import H2ONegativeBinomialGLM
from ...config_loader import MLRegressionConfig
from ...preprocessing import add_plants_bio4_interaction

logger = logging.getLogger(__name__)


class Method1ContinuousGLM(BaseRegressionMethod):
    """Refactored Method 1 - expects preprocessed data from pipeline.
    
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
        super().__init__(config)
        self.glm_config = self.config.get_glm_config()
        
        self.model_null = None
        self.model_full = None
        self.model_interaction = None
        self.data_h2o = None
    
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
        
        # Check overdispersion
        fungi_mean = h2o_data['fungi_richness'].mean()
        fungi_var = h2o_data['fungi_richness'].var()
        if isinstance(fungi_mean, list):
            fungi_mean = fungi_mean[0]
        if isinstance(fungi_var, list):
            fungi_var = fungi_var[0]
        dispersion_ratio = fungi_var / fungi_mean
        logger.info(f"Dispersion ratio: {dispersion_ratio:.2f}")
        
        # Use H2O GLM
        GLMClass = H2ONegativeBinomialGLM
        logger.info("Using H2O GLM for massive parallelization")
        
        # Get MESF columns
        me_cols = [col for col in h2o_data.columns if col.startswith('ME')]
        logger.info(f"Using {len(me_cols)} MESF eigenvectors")
        
        # Model 1: Climate only (null model)
        logger.info("Fitting Model 1: Climate only (null)")
        feature_cols_null = [
            'bio1_std', 'bio4_std', 'bio12_std', 'bio15_std', 
            'elevation_std', 'elevation_std_sq'
        ] + me_cols
        
        null_data = h2o_data[feature_cols_null + ['fungi_richness']]
        self.model_null = GLMClass(self.config.config)
        self.model_null.fit(null_data, y='fungi_richness')
        
        # Model 2: Climate + biodiversity (full model)
        logger.info("Fitting Model 2: Climate + biodiversity (full)")
        biotic_cols = ['plants_richness_log', 'vertebrates_richness_log']
        
        feature_cols_full = feature_cols_null + biotic_cols
        full_data = h2o_data[feature_cols_full + ['fungi_richness']]
        
        self.model_full = GLMClass(self.config.config)
        self.model_full.fit(full_data, y='fungi_richness')
        
        # Model 3: With interaction (METHOD-SPECIFIC FEATURE)
        logger.info("Fitting Model 3: With interaction")
        
        # Add Method1-specific interaction term
        h2o_data = add_plants_bio4_interaction(
            h2o_data, 
            'plants_richness_log', 
            'bio4_std',
            'plants_bio4_interaction'
        )
        
        feature_cols_interaction = feature_cols_full + ['plants_bio4_interaction']
        interaction_data = h2o_data[feature_cols_interaction + ['fungi_richness']]
        
        self.model_interaction = GLMClass(self.config.config)
        self.model_interaction.fit(interaction_data, y='fungi_richness')
        
        # Compare models
        results = {
            'null_model': {
                'aic': self.model_null.get_aic(),
                'bic': self.model_null.get_bic(),
                'deviance': self.model_null.get_deviance()
            },
            'full_model': {
                'aic': self.model_full.get_aic(),
                'bic': self.model_full.get_bic(),
                'deviance': self.model_full.get_deviance()
            },
            'interaction_model': {
                'aic': self.model_interaction.get_aic(),
                'bic': self.model_interaction.get_bic(),
                'deviance': self.model_interaction.get_deviance()
            }
        }
        
        # Likelihood ratio tests
        lr_full_vs_null = 2 * (
            self.model_null.get_deviance() - self.model_full.get_deviance()
        )
        lr_interaction_vs_full = 2 * (
            self.model_full.get_deviance() - self.model_interaction.get_deviance()
        )
        
        results['lr_tests'] = {
            'full_vs_null': {
                'statistic': lr_full_vs_null,
                'df': len(biotic_cols),
                'significant': lr_full_vs_null > 5.99  # chi2 critical value
            },
            'interaction_vs_full': {
                'statistic': lr_interaction_vs_full,
                'df': 1,
                'significant': lr_interaction_vs_full > 3.84
            }
        }
        
        logger.info("Model fitting complete")
        return results
    
    def get_interaction_coefficient(self) -> Dict[str, Any]:
        """Get interaction coefficient for H2 hypothesis test.
        
        Returns:
            Dictionary with coefficient info
        """
        if self.model_interaction is None:
            raise ValueError("Must fit models first")
        
        # Get coefficient for plants_bio4_interaction
        coef_info = self.model_interaction.get_coefficient('plants_bio4_interaction')
        
        return {
            'interaction_coefficient': coef_info['coefficient'],
            'interaction_p_value': coef_info['p_value'],
            'supports_h2': coef_info['p_value'] < 0.05 and coef_info['coefficient'] != 0,
            'coefficient': coef_info['coefficient'],
            'p_value': coef_info['p_value']
        }
    
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
        if self.model_interaction is None:
            raise ValueError("Must fit models first")
        
        return self.model_interaction.predict(data)