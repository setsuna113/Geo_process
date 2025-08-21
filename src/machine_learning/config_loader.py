"""Configuration loader for ML regression analysis."""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from .utils.security_utils import ConfigValidator, PathValidator

logger = logging.getLogger(__name__)


class MLRegressionConfig:
    """Load and validate ML regression configuration.
    
    Configuration validation is MANDATORY by default for security.
    Only disable for testing purposes.
    """
    
    # Class-level flag to enforce validation globally
    ENFORCE_VALIDATION = True
    
    def __init__(self, config_path: Optional[Path] = None, validate: bool = True):
        """Initialize configuration loader with mandatory validation by default.
        
        Args:
            config_path: Path to ml_config.yml. If None, uses default location.
            validate: Whether to validate configuration against schema.
                     Default is True for security. Set to False only for testing.
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
        
        # Always use ConfigValidator for loading
        validator = ConfigValidator()
        
        # Override validate parameter if global enforcement is enabled
        if self.ENFORCE_VALIDATION and not validate:
            logger.warning("Global validation enforcement is ON. Overriding validate=False.")
            validate = True
        
        if validate:
            # Strict validation mode (default)
            try:
                self.config = validator.validate_yaml_file(self.config_path)
                logger.info("Configuration loaded and validated successfully")
            except Exception as e:
                logger.error(f"Configuration validation failed: {e}")
                # No fallback in strict mode - fail fast for security
                raise ValueError(
                    f"Configuration validation failed: {e}\n"
                    "Please fix the configuration file or check the schema."
                )
        else:
            # Relaxed mode for testing only - still sanitize
            logger.warning("Loading configuration in relaxed mode (testing only)")
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            # Always sanitize even in relaxed mode
            self.config = validator.sanitize_config(self.config)
            logger.warning("Configuration loaded without full validation - use only for testing")
        
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
        logger.info(preprocessing.get('missing_data', {}).get('log_message', 'Using default missing data strategy'))
        logger.info(preprocessing.get('standardization', {}).get('log_message', 'Using default standardization'))
        logger.info(preprocessing.get('response_variable', {}).get('log_message', 'Using default response transformation'))
        logger.info(preprocessing.get('biotic_predictors', {}).get('log_message', 'Using default biotic transformation'))
        
        models = self.regression_config.get('models', {})
        glm_config = models.get('glm', {})
        if 'distribution' in glm_config:
            logger.info(glm_config['distribution'].get('log_message', 'Using default GLM distribution'))
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