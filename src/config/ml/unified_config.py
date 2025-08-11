"""Unified configuration system for machine learning module.

Provides a clean, hierarchical configuration structure that's easy to understand
and maintain. Replaces the complex nested dictionary approach.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    
    # Missing data handling
    missing_data_strategy: str = "complete_case"
    missing_threshold: float = 0.1
    
    # Standardization
    standardization_method: str = "2sd"  # 2 standard deviations as per Gelman
    
    # Response variable
    response_transform: str = "none"  # for count data
    
    # Biotic predictors
    biotic_transform: str = "log"  # log(x+1) transformation
    
    # Climate variables to standardize
    climate_variables: list = field(default_factory=lambda: [
        'bio1', 'bio4', 'bio12', 'bio15', 'elevation'
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SpatialConfig:
    """Configuration for spatial analysis (MESF)."""
    
    # MESF settings
    mesf_enabled: bool = True
    mesf_threshold: float = 0.25
    max_eigenvectors: int = 200
    min_eigenvectors: int = 10
    
    # Nystrom approximation for large datasets
    nystrom_enabled: bool = True
    nystrom_threshold: int = 100000
    nystrom_samples: int = 5000
    
    # Coordinate columns
    longitude_col: str = "longitude"
    latitude_col: str = "latitude"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass 
class GLMConfig:
    """Configuration for GLM models."""
    
    # Model settings
    distribution: str = "negative_binomial"  # or "poisson"
    overdispersion_threshold: float = 1.2
    fit_intercept: bool = True
    
    # Solver settings
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6
    
    # Regularization
    regularization: Optional[str] = None  # None, "l1", "l2", "elastic"
    alpha: float = 0.0  # regularization strength
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DMLConfig:
    """Configuration for Double Machine Learning."""
    
    # Cross-fitting
    n_folds: int = 5
    
    # Bootstrap inference
    n_bootstrap: int = 100
    confidence_level: float = 0.95
    
    # Nuisance function estimation
    nuisance_model: str = "gbm"  # or "random_forest"
    
    # GBM settings for nuisance functions
    gbm_n_trees: int = 200
    gbm_max_depth: int = 15
    gbm_learning_rate: float = 0.05
    gbm_sample_rate: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class HypothesisConfig:
    """Configuration for hypothesis testing."""
    
    # Significance level
    alpha: float = 0.05
    
    # Multiple testing correction
    correction_method: Optional[str] = None  # None, "bonferroni", "fdr"
    
    # Hypothesis definitions
    h0_description: str = "No biodiversity effect beyond climate"
    h1_description: str = "Biodiversity begets biodiversity"
    h2_description: str = "Temperate mismatch hypothesis"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ComputeConfig:
    """Configuration for computational resources."""
    
    # H2O settings
    h2o_max_memory: str = "128G"
    h2o_min_memory: str = "8G"
    h2o_n_threads: Optional[int] = None  # None means use all available
    
    # Parallelization
    use_parallel: bool = True
    n_jobs: int = -1  # -1 means use all available
    
    # Memory management
    chunk_size: Optional[int] = None  # for processing large datasets
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class UnifiedMLConfig:
    """Unified configuration for the entire ML module.
    
    This class provides a clean, hierarchical configuration system that's
    easy to understand and modify. It replaces the complex nested dictionary
    approach with typed configuration objects.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize unified configuration.
        
        Args:
            config_path: Path to YAML config file, or None for defaults
        """
        # Initialize with defaults
        self.preprocessing = PreprocessingConfig()
        self.spatial = SpatialConfig()
        self.glm = GLMConfig()
        self.dml = DMLConfig()
        self.hypothesis = HypothesisConfig()
        self.compute = ComputeConfig()
        
        # Load from file if provided
        if config_path:
            self.load_from_file(config_path)
        
        # Log configuration
        self._log_config()
    
    def load_from_file(self, config_path: Union[str, Path]) -> None:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return
        
        logger.info(f"Loading configuration from {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update preprocessing config
        if 'preprocessing' in config_dict:
            for key, value in config_dict['preprocessing'].items():
                if hasattr(self.preprocessing, key):
                    setattr(self.preprocessing, key, value)
        
        # Update spatial config
        if 'spatial' in config_dict:
            for key, value in config_dict['spatial'].items():
                if hasattr(self.spatial, key):
                    setattr(self.spatial, key, value)
        
        # Update GLM config
        if 'glm' in config_dict:
            for key, value in config_dict['glm'].items():
                if hasattr(self.glm, key):
                    setattr(self.glm, key, value)
        
        # Update DML config
        if 'dml' in config_dict:
            for key, value in config_dict['dml'].items():
                if hasattr(self.dml, key):
                    setattr(self.dml, key, value)
        
        # Update hypothesis config
        if 'hypothesis' in config_dict:
            for key, value in config_dict['hypothesis'].items():
                if hasattr(self.hypothesis, key):
                    setattr(self.hypothesis, key, value)
        
        # Update compute config
        if 'compute' in config_dict:
            for key, value in config_dict['compute'].items():
                if hasattr(self.compute, key):
                    setattr(self.compute, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            'preprocessing': self.preprocessing.to_dict(),
            'spatial': self.spatial.to_dict(),
            'glm': self.glm.to_dict(),
            'dml': self.dml.to_dict(),
            'hypothesis': self.hypothesis.to_dict(),
            'compute': self.compute.to_dict()
        }
    
    def save_to_file(self, output_path: Union[str, Path]) -> None:
        """Save configuration to YAML file.
        
        Args:
            output_path: Path to save configuration
        """
        output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to {output_path}")
    
    def _log_config(self) -> None:
        """Log key configuration settings."""
        logger.info(f"Preprocessing: {self.preprocessing.standardization_method} standardization, "
                   f"{self.preprocessing.missing_data_strategy} for missing data")
        logger.info(f"Spatial: MESF {'enabled' if self.spatial.mesf_enabled else 'disabled'}, "
                   f"max {self.spatial.max_eigenvectors} eigenvectors")
        logger.info(f"GLM: {self.glm.distribution} distribution, "
                   f"overdispersion threshold {self.glm.overdispersion_threshold}")
        logger.info(f"DML: {self.dml.n_folds}-fold cross-fitting, "
                   f"{self.dml.n_bootstrap} bootstrap samples")
        logger.info(f"Compute: {self.compute.h2o_max_memory} max memory, "
                   f"{'parallel' if self.compute.use_parallel else 'sequential'} processing")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated path.
        
        Args:
            key_path: Dot-separated path (e.g., 'preprocessing.missing_data_strategy')
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        try:
            keys = key_path.split('.')
            value = self
            
            for key in keys:
                if hasattr(value, key):
                    value = getattr(value, key)
                elif isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            
            return value
        except (AttributeError, KeyError):
            return default
    
    def update(self, key_path: str, value: Any) -> None:
        """Update configuration value by dot-separated path.
        
        Args:
            key_path: Dot-separated path (e.g., 'preprocessing.missing_data_strategy')
            value: New value to set
        """
        keys = key_path.split('.')
        
        if len(keys) == 2:
            section, key = keys
            if hasattr(self, section):
                section_obj = getattr(self, section)
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)
                    logger.info(f"Updated {key_path} = {value}")
                else:
                    logger.warning(f"Key {key} not found in {section}")
            else:
                logger.warning(f"Section {section} not found")
        else:
            logger.warning(f"Invalid key path: {key_path}")


# Adapter to make UnifiedMLConfig compatible with old MLRegressionConfig interface
class MLConfigAdapter:
    """Adapter to make UnifiedMLConfig compatible with the old interface.
    
    This allows gradual migration from the old config system to the new one.
    """
    
    def __init__(self, unified_config: Optional[UnifiedMLConfig] = None):
        """Initialize adapter.
        
        Args:
            unified_config: UnifiedMLConfig instance, or None for defaults
        """
        self.unified = unified_config or UnifiedMLConfig()
        
        # Create backwards-compatible nested structure
        self._build_legacy_structure()
    
    def _build_legacy_structure(self) -> None:
        """Build legacy nested dictionary structure for backwards compatibility."""
        self.config = {
            'regression_analysis': {
                'preprocessing': {
                    'missing_data': {
                        'strategy': self.unified.preprocessing.missing_data_strategy,
                        'threshold': self.unified.preprocessing.missing_threshold
                    },
                    'standardization': {
                        'method': self.unified.preprocessing.standardization_method
                    },
                    'response_variable': {
                        'transformation': self.unified.preprocessing.response_transform
                    },
                    'biotic_predictors': {
                        'transformation': self.unified.preprocessing.biotic_transform
                    }
                },
                'spatial': {
                    'mesf': {
                        'enabled': self.unified.spatial.mesf_enabled,
                        'threshold': self.unified.spatial.mesf_threshold,
                        'max_eigenvectors': self.unified.spatial.max_eigenvectors
                    },
                    'nystrom': {
                        'enable_for_large_data': self.unified.spatial.nystrom_enabled,
                        'threshold': self.unified.spatial.nystrom_threshold
                    }
                },
                'models': {
                    'glm': {
                        'distribution': self.unified.glm.distribution,
                        'overdispersion_threshold': self.unified.glm.overdispersion_threshold,
                        'fit_intercept': self.unified.glm.fit_intercept
                    },
                    'dml': {
                        'n_folds': self.unified.dml.n_folds,
                        'n_bootstrap': self.unified.dml.n_bootstrap
                    }
                }
            }
        }
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration (legacy interface)."""
        return self.config['regression_analysis']['preprocessing']
    
    def get_spatial_config(self) -> Dict[str, Any]:
        """Get spatial configuration (legacy interface)."""
        return self.config['regression_analysis']['spatial']
    
    def get_glm_config(self) -> Dict[str, Any]:
        """Get GLM configuration (legacy interface)."""
        return self.config['regression_analysis']['models']['glm']
    
    def get_dml_config(self) -> Dict[str, Any]:
        """Get DML configuration (legacy interface)."""
        return self.config['regression_analysis']['models']['dml']