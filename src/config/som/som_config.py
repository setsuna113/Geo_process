"""SOM Configuration Module

This module provides configuration management for the Self-Organizing Map (SOM) analysis.
It implements a singleton pattern to ensure consistent configuration across the application.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import threading
from dataclasses import dataclass, field


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


class SOMConfigError(Exception):
    """Exception raised for SOM configuration errors."""
    pass


class SOMConfig:
    """
    SOM Configuration manager with singleton pattern.
    
    This class manages SOM-specific configuration parameters loaded from som_config.yml.
    It provides a thread-safe singleton implementation to ensure configuration consistency.
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
        Find the SOM configuration file with improved path resolution.
        
        Returns:
            Path to som_config.yml
            
        Raises:
            SOMConfigError: If configuration file cannot be found
        """
        # Strategy 1: Check environment variable
        env_path = os.environ.get('SOM_CONFIG_PATH')
        if env_path and Path(env_path).exists():
            return Path(env_path)
        
        # Strategy 2: Check relative to this file
        current_dir = Path(__file__).parent
        config_path = current_dir / 'som_config.yml'
        if config_path.exists():
            return config_path
        
        # Strategy 3: Check project root
        project_root = current_dir.parent.parent.parent
        config_path = project_root / 'som_config.yml'
        if config_path.exists():
            return config_path
        
        # Strategy 4: Check common locations
        common_paths = [
            Path.cwd() / 'som_config.yml',
            Path.cwd() / 'config' / 'som_config.yml',
            Path.cwd() / 'src' / 'config' / 'som' / 'som_config.yml',
        ]
        
        for path in common_paths:
            if path.exists():
                return path
        
        raise SOMConfigError(
            "Could not find som_config.yml. Searched in:\n" +
            f"  - Environment variable SOM_CONFIG_PATH: {env_path}\n" +
            f"  - Current directory: {current_dir}\n" +
            f"  - Project root: {project_root}\n" +
            f"  - Working directory: {Path.cwd()}"
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
    
    @property
    def config_path(self) -> Optional[Path]:
        """Get the path to the loaded configuration file."""
        return self._config_path
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"SOMConfig(path={self._config_path}, loaded={self._config is not None})"


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