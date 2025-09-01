from src.abstractions import get_logger
"""Configuration loader for ML regression analysis."""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
# No validation needed

logger = get_logger(__name__)


class MLRegressionConfig:
    """Load ML regression configuration."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration loader.
        
        Args:
            config_path: Path to ml_config.yml. If None, uses default location.
        """
        if config_path is None:
            # Config lives in ONE place: src/config/ml/ml_config.yml
            config_path = Path(__file__).parent.parent / 'config' / 'ml' / 'ml_config.yml'
            if not config_path.exists():
                raise FileNotFoundError(f"Config not found at {config_path}")
        
        self.config_path = Path(config_path)
        logger.info(f"Loading ML configuration from {self.config_path}")
        
        # Just load the damn YAML
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract regression analysis config
        self.regression_config = self.config.get('regression_analysis', {})
        
        # Log key configuration decisions
        self._log_configuration()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)
    
    def get_glm_config(self) -> Dict[str, Any]:
        """Get GLM-specific configuration.
        
        Returns:
            GLM configuration dictionary
        """
        return self.regression_config.get('models', {}).get('glm', {})
    
    def _log_configuration(self):
        """Log important configuration settings."""
        preprocessing = self.regression_config.get('preprocessing', {})
        logger.info(preprocessing.get('missing_data', {}).get(
            'log_message', 'Using default missing data strategy'))
        logger.info(preprocessing.get('standardization', {}).get(
            'log_message', 'Using default standardization'))
        logger.info(preprocessing.get('response_variable', {}).get(
            'log_message', 'Using default response transformation'))
        logger.info(preprocessing.get('biotic_predictors', {}).get(
            'log_message', 'Using default biotic transformation'))
        
        models = self.regression_config.get('models', {})
        glm_config = models.get('glm', {})
        if 'distribution' in glm_config:
            logger.info(glm_config['distribution'].get(
                'log_message', 'Using default GLM distribution'))
        lightgbm_config = models.get('lightgbm', {})
        if 'log_message' in lightgbm_config:
            logger.info(lightgbm_config['log_message'])
        
        # Simplified spatial config logging
        spatial = self.regression_config.get('spatial', {})
        if spatial:
            logger.info(f"MESF spatial analysis: {spatial.get('n_eigenvectors', 10)} eigenvectors, "
                       f"{spatial.get('distance_threshold', 100.0)}km threshold")
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self.regression_config.get('preprocessing', {})
    
    def get_lightgbm_config(self) -> Dict[str, Any]:
        """Get LightGBM configuration for DML."""
        return self.regression_config.get('models', {}).get('lightgbm', {})
    
    def get_spatial_config(self) -> Dict[str, Any]:
        """Get spatial analysis configuration."""
        return self.regression_config.get('spatial', {})
    
    def get_dml_config(self) -> Dict[str, Any]:
        """Get DML configuration."""
        return self.regression_config.get('dml', {})
    
    def get_hypothesis_config(self) -> Dict[str, Any]:
        """Get hypothesis testing configuration."""
        return self.regression_config.get('hypothesis_testing', {})
    
    def get_feature_engineering_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration."""
        return self.regression_config.get('feature_engineering', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.regression_config.get('output', {})
    
    def get_experiment_config(self, experiment_name: str) -> Dict[str, Any]:
        """Get configuration for a specific experiment.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Experiment configuration dictionary
        """
        experiments = self.regression_config.get('experiments', {})
        if experiment_name not in experiments:
            raise ValueError(
                f"Experiment '{experiment_name}' not found. "
                f"Available experiments: {list(experiments.keys())}"
            )
        
        exp_config = experiments[experiment_name].copy()
        logger.info(f"Loaded experiment '{experiment_name}': {exp_config['description']}")
        return exp_config
    
    def should_use_nystrom(self, n_samples: int) -> bool:
        """Determine if Nystrom approximation should be used.
        
        Args:
            n_samples: Number of samples in dataset
            
        Returns:
            True if Nystrom should be used, False otherwise
        """
        spatial_config = self.get_spatial_config()
        nystrom_config = spatial_config.get('nystrom', {})
        
        if not nystrom_config.get('enable_for_large_data', False):
            return False
        
        threshold = nystrom_config.get('size_threshold', 100000)
        use_nystrom = n_samples > threshold
        
        if use_nystrom:
            logger.info(f"Dataset has {n_samples:,} samples > {threshold:,}, using Nystrom approximation")
        else:
            logger.info(f"Dataset has {n_samples:,} samples <= {threshold:,}, using standard MESF")
        
        return use_nystrom
    
    def get_distribution_family(self, dispersion_ratio: float) -> str:
        """Determine GLM distribution family based on dispersion ratio.
        
        Args:
            dispersion_ratio: Variance/mean ratio
            
        Returns:
            'poisson' or 'negative_binomial'
        """
        glm_config = self.get_glm_config()
        dist_config = glm_config.get('distribution', {})
        
        lower, upper = dist_config.get('poisson_threshold', [0.8, 1.2])
        
        if lower <= dispersion_ratio <= upper:
            distribution = 'poisson'
            logger.info(f"Dispersion ratio {dispersion_ratio:.2f} in [{lower}, {upper}], using Poisson GLM")
        else:
            distribution = 'negative_binomial'
            logger.info(f"Dispersion ratio {dispersion_ratio:.2f} outside [{lower}, {upper}], using Negative Binomial GLM")
        
        # Check if override is set
        if dist_config.get('use_negative_binomial', False):
            distribution = 'negative_binomial'
            logger.info("Override: Using Negative Binomial GLM as configured")
        
        return distribution