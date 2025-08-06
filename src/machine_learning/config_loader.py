"""Configuration loader for ML regression analysis."""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from .utils.security_utils import ConfigValidator, PathValidator

logger = logging.getLogger(__name__)


class MLRegressionConfig:
    """Load and validate ML regression configuration."""
    
    def __init__(self, config_path: Optional[Path] = None, validate: bool = True):
        """Initialize configuration loader with optional validation.
        
        Args:
            config_path: Path to ml_config.yml. If None, uses default location.
            validate: Whether to validate configuration against schema.
        """
        if config_path is None:
            # Try multiple default locations
            possible_paths = [
                Path(__file__).parent.parent.parent / 'config' / 'ml' / 'ml_config.yml',
                Path('config/ml/ml_config.yml'),
                Path('/home/yl998/dev/geo/config/ml/ml_config.yml')
            ]
            for path in possible_paths:
                if path.exists():
                    config_path = path
                    break
            else:
                raise FileNotFoundError(
                    "Could not find ml_config.yml in default locations. "
                    "Please specify config_path explicitly."
                )
        
        self.config_path = Path(config_path)
        logger.info(f"Loading ML configuration from {self.config_path}")
        
        if validate:
            # Use ConfigValidator for secure loading and validation
            validator = ConfigValidator()
            try:
                self.config = validator.validate_yaml_file(self.config_path)
                logger.info("Configuration loaded and validated successfully")
            except Exception as e:
                logger.error(f"Configuration validation failed: {e}")
                # Fall back to unsafe loading with warning
                logger.warning("Falling back to unsafe YAML loading")
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                # Sanitize the config to remove dangerous patterns
                self.config = validator.sanitize_config(self.config)
        else:
            # Legacy loading without validation
            logger.warning("Loading configuration without schema validation")
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Extract regression analysis config
        self.regression_config = self.config.get('regression_analysis', {})
        
        # Log key configuration decisions
        self._log_configuration()
    
    def _log_configuration(self):
        """Log important configuration settings."""
        preprocessing = self.regression_config.get('preprocessing', {})
        logger.info(preprocessing['missing_data']['log_message'])
        logger.info(preprocessing['standardization']['log_message'])
        logger.info(preprocessing['response_variable']['log_message'])
        logger.info(preprocessing['biotic_predictors']['log_message'])
        
        models = self.regression_config.get('models', {})
        logger.info(models['glm']['distribution']['log_message'])
        logger.info(models['lightgbm']['log_message'])
        
        spatial = self.regression_config.get('spatial', {})
        logger.info(spatial['mesf']['log_message'])
        if spatial['nystrom']['enable_for_large_data']:
            logger.info(spatial['nystrom']['log_message'])
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self.regression_config.get('preprocessing', {})
    
    def get_glm_config(self) -> Dict[str, Any]:
        """Get GLM model configuration."""
        return self.regression_config.get('models', {}).get('glm', {})
    
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