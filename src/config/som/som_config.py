"""SOM Configuration Module

This module provides configuration management for the Self-Organizing Map (SOM) analysis.
It implements a singleton pattern to ensure consistent configuration across the application.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import threading
from dataclasses import dataclass, field
from src.abstractions.interfaces.method_config import IMethodConfig


@dataclass
class SOMConfigValidation:
    """Validation rules for SOM configuration parameters."""
    min_chunk_size: int = 1000
    max_chunk_size: int = 1000000
    min_grid_size: int = 2
    max_grid_size: int = 100
    min_learning_rate: float = 0.001
    max_learning_rate: float = 1.0
    min_epochs: int = 1
    max_epochs: int = 10000
    min_workers: int = 1
    max_workers: int = 256
    min_samples_for_parallel: int = 1000
    optimal_min_samples_for_parallel: int = 50000


class SOMConfigError(Exception):
    """Exception raised for SOM configuration errors."""
    pass


class SOMConfig(IMethodConfig):
    """
    SOM Configuration manager with singleton pattern.
    
    This class manages SOM-specific configuration parameters loaded from som_config_optimized.yml
    (or som_config.yml as fallback). It provides a thread-safe singleton implementation to ensure 
    configuration consistency.
    """
    
    _instance = None
    _lock = threading.Lock()
    _config = None
    _config_path = None
    _validation = SOMConfigValidation()
    
    def __new__(cls):
        """Implement thread-safe singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize configuration if not already done."""
        if self._initialized:
            return
            
        self._initialized = True
        self._load_config()
    
    def _find_config_file(self) -> Path:
        """
        Find the SOM configuration file.
        
        Returns:
            Path to the simplified config file
            
        Raises:
            SOMConfigError: If configuration file cannot be found
        """
        # Strategy 1: Explicit path via environment variable
        env_path = os.environ.get('SOM_CONFIG_PATH')
        if env_path and Path(env_path).exists():
            return Path(env_path)
        
        # Strategy 2: Use the simplified config (NEW DEFAULT)
        current_dir = Path(__file__).parent
        simplified_path = current_dir / 'som_config_simplified.yml'
        
        if simplified_path.exists():
            return simplified_path
        
        # Fallback for testing if unified config still exists (DEPRECATED)
        unified_path = current_dir / 'som_config_unified.yml'
        if unified_path.exists():
            import warnings
            warnings.warn(
                "DEPRECATED: som_config_unified.yml is deprecated. Using som_config_simplified.yml instead.",
                DeprecationWarning,
                stacklevel=2
            )
            # Don't use it, force using simplified
            if not simplified_path.exists():
                raise SOMConfigError(
                    "som_config_unified.yml found but is deprecated. "
                    "Please use som_config_simplified.yml instead."
                )
        
        raise SOMConfigError(
            f"Could not find SOM configuration. Searched for:\n" +
            f"  - Environment variable SOM_CONFIG_PATH: {env_path}\n" +
            f"  - Simplified config: {simplified_path}\n" +
            "Provide SOM_CONFIG_PATH or ensure som_config_simplified.yml exists"
        )
    
    def _load_config(self):
        """
        Load and validate configuration from YAML file.
        
        Raises:
            SOMConfigError: If configuration is invalid
        """
        try:
            self._config_path = self._find_config_file()
            
            with open(self._config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            if not isinstance(config, dict):
                raise SOMConfigError(f"Configuration file must contain a YAML dictionary, got {type(config)}")
            
            # Validate required sections for new simplified config
            required_sections = ['grid', 'data', 'distance', 'training']
            missing_sections = [s for s in required_sections if s not in config]
            if missing_sections:
                # Try to handle old config format gracefully
                if 'architecture_config' in config or 'training_config' in config:
                    logger.warning("Old config format detected. Please migrate to simplified config.")
                    # Convert old format to new format
                    config = self._convert_old_to_new_format(config)
                else:
                    raise SOMConfigError(f"Missing required configuration sections: {missing_sections}")
            
            self._config = config
            self._apply_env_overrides()  # Apply environment overrides immediately
            self._validate_config()
            
        except yaml.YAMLError as e:
            raise SOMConfigError(f"Error parsing YAML configuration: {e}")
        except Exception as e:
            raise SOMConfigError(f"Error loading configuration: {e}")
    
    def _convert_old_to_new_format(self, old_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert old config format to new simplified format."""
        new_config = {
            'grid': {
                'size': old_config.get('architecture_config', {}).get('grid_size', [20, 20]),
                'topology': old_config.get('architecture_config', {}).get('topology', 'rectangular')
            },
            'data': {
                'spatial_grid_step': old_config.get('data_config', {}).get('spatial_grid_step', 100),
                'features': old_config.get('data_config', {}).get('feature_cols', ['animal_richness', 'plant_richness', 'fungi_richness']),
                'coordinates': old_config.get('data_config', {}).get('coordinate_columns', ['x', 'y']),
                'min_valid_features': old_config.get('distance_config', {}).get('min_valid_features', 2)
            },
            'distance': {
                'feature_metric': old_config.get('distance_config', {}).get('input_space', 'bray_curtis'),
                'spatial_metric': old_config.get('architecture_config', {}).get('geographic_distance', 'haversine'),
                'spatial_weight': old_config.get('architecture_config', {}).get('spatial_weight', 0.2)
            },
            'training': old_config.get('training_config', {}),
            'vlrsom': {
                'enabled': True,
                'alpha': old_config.get('training_config', {}).get('vlrsom_alpha', 0.95),
                'gamma': old_config.get('training_config', {}).get('vlrsom_gamma', 0.001)
            },
            'convergence': {
                'patience': old_config.get('training_config', {}).get('patience', 50),
                'min_improvement': old_config.get('training_config', {}).get('qe_improvement_threshold', 0.001)
            },
            'monitoring': {
                'checkpoint': {
                    'frequency': old_config.get('validation_config', {}).get('progress_tracking', {}).get('checkpoint_frequency', 10),
                    'directory': old_config.get('validation_config', {}).get('progress_tracking', {}).get('checkpoint_dir', 'checkpoints/som')
                }
            },
            'output': old_config.get('output_config', {})
        }
        return new_config
    
    def _validate_config(self):
        """Validate configuration values against defined rules."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Validate chunk size
        chunk_size = self.get_chunk_size()
        if not (self._validation.min_chunk_size <= chunk_size <= self._validation.max_chunk_size):
            raise SOMConfigError(
                f"chunk_size must be between {self._validation.min_chunk_size} "
                f"and {self._validation.max_chunk_size}, got {chunk_size}"
            )
        
        # Validate grid size
        grid_size = self.get_grid_size()
        for dim in grid_size:
            if not (self._validation.min_grid_size <= dim <= self._validation.max_grid_size):
                raise SOMConfigError(
                    f"grid_size dimensions must be between {self._validation.min_grid_size} "
                    f"and {self._validation.max_grid_size}, got {dim}"
                )
        
        # Validate learning rate
        lr = self.get_initial_learning_rate()
        if not (self._validation.min_learning_rate <= lr <= self._validation.max_learning_rate):
            raise SOMConfigError(
                f"initial_learning_rate must be between {self._validation.min_learning_rate} "
                f"and {self._validation.max_learning_rate}, got {lr}"
            )
        
        # Validate parallel processing configuration
        parallel_config = self.get_parallel_config()
        if parallel_config.get('enabled', True):
            n_workers = parallel_config.get('n_workers')
            if n_workers is not None:
                if not isinstance(n_workers, int) or not (self._validation.min_workers <= n_workers <= self._validation.max_workers):
                    raise SOMConfigError(
                        f"n_workers must be between {self._validation.min_workers} "
                        f"and {self._validation.max_workers}, got {n_workers}"
                    )
            
            min_samples = parallel_config.get('min_samples_for_parallel', 100000)
            if not isinstance(min_samples, int) or min_samples < self._validation.min_samples_for_parallel:
                raise SOMConfigError(
                    f"min_samples_for_parallel must be at least {self._validation.min_samples_for_parallel}, "
                    f"got {min_samples}"
                )
            
            # Warn if suboptimal configuration
            if min_samples < self._validation.optimal_min_samples_for_parallel:
                logger.warning(
                    f"min_samples_for_parallel is {min_samples}, which is below the recommended "
                    f"threshold of {self._validation.optimal_min_samples_for_parallel}. "
                    "This may lead to parallel processing overhead exceeding benefits."
                )
            
            # Check performance config if exists
            perf_config = self._config.get('performance_config', {})
            batch_config = perf_config.get('parallel_batch_processing', {})
            
            if batch_config:
                backend = batch_config.get('backend', 'loky')
                if backend not in ['loky', 'threading', 'multiprocessing']:
                    logger.warning(
                        f"Unknown parallel backend '{backend}'. "
                        "Valid options are: 'loky', 'threading', 'multiprocessing'"
                    )
    
    def reload(self):
        """Reload configuration from file."""
        with self._lock:
            self._config = None
            self._load_config()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.
        
        Args:
            key: Configuration key (e.g., 'training_config.memory_management.chunk_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if self._config is None:
            raise SOMConfigError("Configuration not loaded")
        
        value = self._config
        for part in key.split('.'):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value
    
    # Convenience methods for common configurations
    def get_architecture_config(self) -> Dict[str, Any]:
        """Get architecture configuration (compatibility layer)."""
        # Map new config to old format for backward compatibility
        return {
            'grid_size': self._config.get('grid', {}).get('size', [20, 20]),
            'topology': self._config.get('grid', {}).get('topology', 'rectangular'),
            'spatial_weight': self._config.get('distance', {}).get('spatial_weight', 0.2),
            'geographic_distance': self._config.get('distance', {}).get('geographic_distance', 'haversine'),
            'neighborhood_function': self._config.get('spatial_metrics', {}).get('neighborhood_function', 'gaussian'),
            'initial_radius': self._config.get('training', {}).get('radius', {}).get('initial', 10.0),
            'final_radius': self._config.get('training', {}).get('radius', {}).get('final', 0.5),
            'radius_decay': self._config.get('training', {}).get('radius', {}).get('decay', 'linear')
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration (compatibility layer)."""
        # Map new config to old format for backward compatibility
        training = self._config.get('training', {})
        vlrsom = self._config.get('vlrsom', {})
        convergence = self._config.get('convergence', {})
        
        return {
            'max_epochs': training.get('max_epochs', 15),
            'n_workers': training.get('n_workers', 50),
            'initial_learning_rate': training.get('learning_rate', {}).get('initial', 0.4),
            'min_learning_rate': training.get('learning_rate', {}).get('minimum', 0.1),
            'max_learning_rate': training.get('learning_rate', {}).get('maximum', 1.0),
            'batch_size': training.get('batch', {}).get('size', 5000),
            'chunk_size': training.get('memory', {}).get('chunk_size', 50000),
            'patience': convergence.get('patience', 50),
            # VLRSOM parameters now properly read from vlrsom section
            'vlrsom_alpha': vlrsom.get('alpha', 0.95),
            'vlrsom_gamma': vlrsom.get('gamma', 0.001),
            'vlrsom_error_sample_size': vlrsom.get('error_sample_size', 10000),
            # VLRSOM adaptation factors
            'lr_increase_factor': vlrsom.get('increase_factor', 1.1),
            'lr_decrease_factor': vlrsom.get('decrease_factor', 0.95),
            'high_qe_lr_range': [0.01, 0.8],  # Default values for compatibility
            'low_qe_lr_range': [0.01, 0.8],   # Default values for compatibility
            # Convergence criteria - updated to new structure
            'geographic_coherence_threshold': convergence.get('geographic_coherence', {}).get('threshold', 0.6),
            'geographic_coherence_weight': convergence.get('geographic_coherence', {}).get('weight', 0.3),
            'lr_stability_threshold': convergence.get('lr_stability_threshold', 0.001),
            'qe_improvement_threshold': convergence.get('qe', {}).get('min_improvement', 0.001),
            'te_threshold': convergence.get('topographic_error', {}).get('threshold', 0.05),
            # Batch training config with momentum properly linked
            'batch_training': training.get('batch', {}),
            # Add momentum config for easy access
            'momentum': training.get('momentum', {})
        }
    
    def get_grid_size(self) -> list:
        """Get SOM grid size."""
        return self._config.get('grid', {}).get('size', [20, 20])
    
    def get_initial_learning_rate(self) -> float:
        """Get initial learning rate."""
        return self._config.get('training', {}).get('learning_rate', {}).get('initial', 0.4)
    
    def get_chunk_size(self) -> int:
        """Get chunk size for batch processing."""
        return self._config.get('training', {}).get('memory', {}).get('chunk_size', 50000)
    
    def get_sampling_config(self) -> Dict[str, Any]:
        """Get sampling configuration."""
        return self._config.get('monitoring', {}).get('sampling', {})
    
    def get_qe_sample_size(self) -> int:
        """Get sample size for QE calculation."""
        return self._config.get('monitoring', {}).get('sampling', {}).get('large_sample', 50000)
    
    def get_parallel_config(self) -> Dict[str, Any]:
        """Get parallel processing configuration."""
        training = self._config.get('training', {})
        return {
            'enabled': True,
            'n_workers': training.get('n_workers', 50),
            'min_samples_for_parallel': 100000
        }
    
    def is_parallel_enabled(self) -> bool:
        """Check if parallel processing is enabled."""
        parallel_config = self.get_parallel_config()
        return parallel_config.get('enabled', True)
    
    def get_n_workers(self) -> Optional[int]:
        """Get number of parallel workers."""
        parallel_config = self.get_parallel_config()
        return parallel_config.get('n_workers', None)
    
    def check_optimization_recommendations(self) -> Dict[str, Any]:
        """
        Check configuration for optimization opportunities.
        
        Returns:
            Dict containing recommendations and warnings
        """
        import multiprocessing
        recommendations = {
            'warnings': [],
            'suggestions': [],
            'performance_impact': []
        }
        
        # Check parallel processing settings
        parallel_config = self.get_parallel_config()
        if parallel_config.get('enabled', True):
            n_workers = parallel_config.get('n_workers')
            available_cpus = multiprocessing.cpu_count()
            
            if n_workers is None:
                recommendations['suggestions'].append(
                    f"Consider setting explicit n_workers. System has {available_cpus} CPUs available."
                )
            elif n_workers > available_cpus:
                recommendations['warnings'].append(
                    f"n_workers ({n_workers}) exceeds available CPUs ({available_cpus}). "
                    "This may cause resource contention."
                )
            elif n_workers < available_cpus // 4 and available_cpus > 8:
                recommendations['suggestions'].append(
                    f"n_workers ({n_workers}) seems low for a system with {available_cpus} CPUs. "
                    f"Consider increasing to {available_cpus // 2} for better performance."
                )
            
            # Check chunk size vs parallel threshold
            chunk_size = self.get_chunk_size()
            min_samples = parallel_config.get('min_samples_for_parallel', 100000)
            
            if chunk_size > min_samples // 2:
                recommendations['warnings'].append(
                    f"chunk_size ({chunk_size}) is large relative to min_samples_for_parallel ({min_samples}). "
                    "This may result in poor parallel efficiency."
                )
                recommendations['performance_impact'].append(
                    "Consider reducing chunk_size or increasing min_samples_for_parallel"
                )
        
        # Check memory settings
        train_config = self.get_training_config()
        memory_config = train_config.get('memory_management', {})
        chunk_if_exceeds = memory_config.get('chunk_if_exceeds', '8GB')
        
        # Parse memory string (e.g., '8GB' -> 8 * 1024 * 1024 * 1024)
        import re
        match = re.match(r'(\d+)(GB|MB)', chunk_if_exceeds)
        if match:
            size, unit = match.groups()
            size_bytes = int(size) * (1024**3 if unit == 'GB' else 1024**2)
            
            # Check if chunk size aligns with memory limit
            estimated_chunk_memory = chunk_size * 100 * 8  # Rough estimate
            if estimated_chunk_memory > size_bytes:
                recommendations['warnings'].append(
                    f"Estimated chunk memory usage may exceed chunk_if_exceeds limit ({chunk_if_exceeds})"
                )
        
        # Check QE sampling
        qe_sample_size = self.get_qe_sample_size()
        if qe_sample_size < 10000:
            recommendations['warnings'].append(
                f"QE sample size ({qe_sample_size}) is very low. "
                "This may lead to inaccurate convergence detection."
            )
        
        return recommendations
    
    @property
    def config_path(self) -> Optional[Path]:
        """Get the path to the loaded configuration file."""
        return self._config_path
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"SOMConfig(path={self._config_path}, loaded={self._config is not None})"
    
    # IMethodConfig interface implementation
    def load_config(self, config_name: Optional[str] = None,
                   preset: Optional[str] = None) -> Dict[str, Any]:
        """Load method configuration with environment overrides.
        
        Configuration Precedence Order (highest to lowest):
        1. Environment variables (SOM_* pattern)
        2. Preset-specific values (if preset specified)
        3. Config file values (from config_name or default)
        4. Hard-coded defaults in this class
        
        Environment Variables:
        - SOM_CONFIG_PATH: Path to config file (overrides config_name)
        - SOM_MAX_EPOCHS: Maximum training epochs (int)
        - SOM_MAX_WORKERS: Number of parallel workers (int)
        - SOM_CHUNK_SIZE: Samples per chunk (int)
        - SOM_SPATIAL_GRID_STEP: Spatial sampling step (int)
        - SOM_GRID_SIZE: Grid dimensions (comma-separated, e.g., "60,60")
        - SOM_INITIAL_LR: Initial learning rate (float)
        
        Args:
            config_name: Name of config file (without .yml)
            preset: Preset name to apply
            
        Returns:
            Configuration dictionary with all settings
        """
        if config_name:
            # Allow loading a specific config file
            old_path = os.environ.get('SOM_CONFIG_PATH', '')
            try:
                os.environ['SOM_CONFIG_PATH'] = f"{config_name}.yml"
                self.reload()
            finally:
                if old_path:
                    os.environ['SOM_CONFIG_PATH'] = old_path
                elif 'SOM_CONFIG_PATH' in os.environ:
                    del os.environ['SOM_CONFIG_PATH']
        
        # Apply environment variable overrides
        config = self._config.copy() if self._config else {}
        
        # Check for SOM-specific environment variables
        env_overrides = {
            'SOM_MAX_EPOCHS': ('training', 'max_epochs', int),
            'SOM_MAX_WORKERS': ('training', 'n_workers', int),
            'SOM_CHUNK_SIZE': ('training.memory', 'chunk_size', int),
            'SOM_SPATIAL_GRID_STEP': ('data', 'spatial_grid_step', int),
            'SOM_GRID_SIZE': ('grid', 'size', lambda x: [int(v) for v in x.split(',')]),
            'SOM_INITIAL_LR': ('training.learning_rate', 'initial', float),
        }
        
        for env_var, (section_path, key, converter) in env_overrides.items():
            if env_var in os.environ:
                try:
                    value = converter(os.environ[env_var])
                    # Handle nested paths like 'training.memory'
                    sections = section_path.split('.')
                    target = config
                    for section in sections[:-1]:
                        if section not in target:
                            target[section] = {}
                        target = target[section]
                    # Set the final value
                    if sections[-1] not in target:
                        target[sections[-1]] = {}
                    target[sections[-1]][key] = value
                except (ValueError, TypeError) as e:
                    import logging
                    logging.warning(f"Failed to parse {env_var}: {e}")
        
        return config
    
    def get_env_var_names(self) -> List[str]:
        """Return list of supported environment variables.
        
        Returns:
            List of environment variable names
        """
        return [
            'SOM_CONFIG_PATH',
            'SOM_MAX_EPOCHS',
            'SOM_MAX_WORKERS',
            'SOM_CHUNK_SIZE',
            'SOM_SPATIAL_GRID_STEP',
            'SOM_GRID_SIZE',
            'SOM_INITIAL_LR',
            'SOM_MIN_LR',
            'SOM_CONFIG_MODE',
        ]
    
    def get_env_var_descriptions(self) -> Dict[str, str]:
        """Return descriptions of environment variables.
        
        Returns:
            Dictionary mapping env var names to descriptions
        """
        return {
            'SOM_CONFIG_PATH': 'Path to SOM configuration YAML file',
            'SOM_MAX_EPOCHS': 'Maximum training epochs (int)',
            'SOM_MAX_WORKERS': 'Number of parallel workers (int)',
            'SOM_CHUNK_SIZE': 'Chunk size for batch processing (int)',
            'SOM_SPATIAL_GRID_STEP': 'Spatial grid sampling step (int, e.g., 100 = every 100th point)',
            'SOM_GRID_SIZE': 'SOM grid dimensions (comma-separated, e.g., "3,3")',
            'SOM_INITIAL_LR': 'Initial learning rate (float)',
            'SOM_MIN_LR': 'Minimum learning rate floor (float)',
            'SOM_CONFIG_MODE': 'Configuration mode: "test" or "production"',
        }
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration values.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        # This is already implemented in _validate_config
        # Just wrap it for the interface
        old_config = self._config
        try:
            self._config = config
            self._validate_config()
        except SOMConfigError as e:
            raise ValueError(str(e))
        finally:
            self._config = old_config
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides to the loaded config."""
        import logging
        logger = logging.getLogger(__name__)
        
        env_overrides = {
            'SOM_MAX_EPOCHS': ('training', 'max_epochs', int),
            'SOM_MAX_WORKERS': ('training', 'n_workers', int),
            'SOM_CHUNK_SIZE': ('training.memory', 'chunk_size', int),
            'SOM_SPATIAL_GRID_STEP': ('data', 'spatial_grid_step', int),
            'SOM_GRID_SIZE': ('grid', 'size', lambda x: [int(v) for v in x.split(',')]),
            'SOM_INITIAL_LR': ('training.learning_rate', 'initial', float),
        }
        
        for env_var, (section_path, key, converter) in env_overrides.items():
            if env_var in os.environ:
                try:
                    value = converter(os.environ[env_var])
                    # Handle nested paths like 'training.memory'
                    sections = section_path.split('.')
                    target = self._config
                    for section in sections[:-1]:
                        if section not in target:
                            target[section] = {}
                        target = target[section]
                    # Set the final value
                    if sections[-1] not in target:
                        target[sections[-1]] = {}
                    target[sections[-1]][key] = value
                    logger.info(f"Applied environment override: {env_var}={value}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse {env_var}: {e}")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration without any overrides.
        
        Returns:
            Default configuration dictionary
        """
        # Load config without environment variables
        if self._config is None:
            self._load_config()
        return self._config.copy() if self._config else {}


# Create singleton instance accessor
_som_config_instance = None

def get_som_config() -> SOMConfig:
    """
    Get the singleton SOM configuration instance.
    
    This is the recommended way to access SOM configuration throughout the application.
    
    Returns:
        SOMConfig: The singleton configuration instance
    """
    global _som_config_instance
    if _som_config_instance is None:
        _som_config_instance = SOMConfig()
    return _som_config_instance