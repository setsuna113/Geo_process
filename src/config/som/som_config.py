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
            Path to the unified config file
            
        Raises:
            SOMConfigError: If configuration file cannot be found
        """
        # Strategy 1: Explicit path via environment variable
        env_path = os.environ.get('SOM_CONFIG_PATH')
        if env_path and Path(env_path).exists():
            return Path(env_path)
        
        # Strategy 2: Use the unified config
        current_dir = Path(__file__).parent
        unified_path = current_dir / 'som_config_unified.yml'
        
        if unified_path.exists():
            return unified_path
        
        # Fallback: try som_config.yml (legacy name)
        legacy_path = current_dir / 'som_config.yml'
        if legacy_path.exists():
            import warnings
            warnings.warn(
                "Using legacy config name 'som_config.yml', consider renaming to 'som_config_unified.yml'",
                UserWarning,
                stacklevel=2
            )
            return legacy_path
        
        raise SOMConfigError(
            f"Could not find SOM configuration. Searched for:\n" +
            f"  - Environment variable SOM_CONFIG_PATH: {env_path}\n" +
            f"  - Unified config: {unified_path}\n" +
            f"  - Legacy config: {legacy_path}\n" +
            "Provide SOM_CONFIG_PATH or ensure som_config_unified.yml exists"
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
            
            # Validate required sections
            required_sections = ['architecture_config', 'training_config']
            missing_sections = [s for s in required_sections if s not in config]
            if missing_sections:
                raise SOMConfigError(f"Missing required configuration sections: {missing_sections}")
            
            self._config = config
            self._validate_config()
            
        except yaml.YAMLError as e:
            raise SOMConfigError(f"Error parsing YAML configuration: {e}")
        except Exception as e:
            raise SOMConfigError(f"Error loading configuration: {e}")
    
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
        """Get architecture configuration section."""
        return self._config.get('architecture_config', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration section."""
        return self._config.get('training_config', {})
    
    def get_grid_size(self) -> list:
        """Get SOM grid size."""
        arch_config = self.get_architecture_config()
        return arch_config.get('grid_size', [20, 20])
    
    def get_initial_learning_rate(self) -> float:
        """Get initial learning rate."""
        train_config = self.get_training_config()
        return train_config.get('initial_learning_rate', 0.5)
    
    def get_chunk_size(self) -> int:
        """Get chunk size for batch processing."""
        train_config = self.get_training_config()
        memory_mgmt = train_config.get('memory_management', {})
        return memory_mgmt.get('chunk_size', 50000)
    
    def get_sampling_config(self) -> Dict[str, Any]:
        """Get sampling configuration."""
        train_config = self.get_training_config()
        return train_config.get('sampling', {})
    
    def get_qe_sample_size(self) -> int:
        """Get sample size for QE calculation."""
        sampling_config = self.get_sampling_config()
        qe_config = sampling_config.get('qe_calculation', {})
        return qe_config.get('sample_size', 100000)
    
    def get_parallel_config(self) -> Dict[str, Any]:
        """Get parallel processing configuration."""
        train_config = self.get_training_config()
        return train_config.get('parallel', {
            'enabled': True,
            'n_workers': None,
            'min_samples_for_parallel': 100000
        })
    
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
            'SOM_MAX_EPOCHS': ('training_config', 'max_epochs', int),
            'SOM_MAX_WORKERS': ('training_config', 'n_workers', int),
            'SOM_CHUNK_SIZE': ('training_config', 'chunk_size', int),
            'SOM_SPATIAL_GRID_STEP': ('data_config', 'spatial_grid_step', int),
            'SOM_GRID_SIZE': ('architecture_config', 'grid_size', lambda x: [int(v) for v in x.split(',')]),
            'SOM_INITIAL_LR': ('training_config', 'initial_learning_rate', float),
        }
        
        for env_var, (section, key, converter) in env_overrides.items():
            if env_var in os.environ:
                try:
                    value = converter(os.environ[env_var])
                    if section not in config:
                        config[section] = {}
                    config[section][key] = value
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