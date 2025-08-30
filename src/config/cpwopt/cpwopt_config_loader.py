"""Configuration loader for CP-WOPT tensor factorization.

This module handles loading and merging of CP-WOPT configurations,
including support for presets and environment variable overrides.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from copy import deepcopy

from src.core_abstractions import IMethodConfig

logger = logging.getLogger(__name__)


class CPWOPTConfigLoader(IMethodConfig):
    """Loads and manages CP-WOPT configuration."""
    
    def __init__(self):
        """Initialize config loader."""
        self.config_dir = Path(__file__).parent
        self.config = {}
        self.defaults = {}
        self.presets = {}
        
    def load_config(self, config_name: Optional[str] = None,
                   preset: Optional[str] = None) -> Dict[str, Any]:
        """Load CP-WOPT configuration.
        
        Args:
            config_name: Name of config file (without .yml)
            preset: Preset name to apply
            
        Returns:
            Merged configuration dictionary
        """
        # Start with default values
        config = deepcopy(self.get_default_config())
        
        # Apply preset if specified
        if preset and preset in self.presets:
            logger.info(f"Applying CP-WOPT preset: {preset}")
            self._merge_configs(config, self.presets[preset])
        
        # Load main config file
        if config_name:
            config_file = self.config_dir / f"{config_name}.yml"
        else:
            config_file = self.config_dir / "cpwopt_config.yml"
            
        if config_file.exists():
            logger.info(f"Loading CP-WOPT config from: {config_file}")
            with open(config_file, 'r') as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    self._merge_configs(config, user_config)
        
        # Apply environment variable overrides
        config = self._apply_env_overrides(config)
        
        # Validate configuration
        self.validate_config(config)
        
        self.config = config
        return config
    
    def _load_defaults(self):
        """Load default configuration and presets."""
        defaults_file = self.config_dir / "cpwopt_config_defaults.yml"
        
        if defaults_file.exists():
            with open(defaults_file, 'r') as f:
                defaults_data = yaml.safe_load(f)
                self.defaults = defaults_data.get('defaults', {})
                self.presets = defaults_data.get('presets', {})
        else:
            logger.warning("CP-WOPT defaults file not found")
    
    def _merge_configs(self, base: Dict[str, Any], 
                      updates: Dict[str, Any]) -> None:
        """Recursively merge configuration dictionaries.
        
        Args:
            base: Base configuration to update
            updates: Updates to apply
        """
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides.
        
        Environment variables follow pattern: CPWOPT_SECTION_KEY
        For example: CPWOPT_ALGORITHM_RANK=50
        """
        env_prefix = "CPWOPT_"
        
        for env_key, env_value in os.environ.items():
            if env_key.startswith(env_prefix):
                # Parse the key structure
                key_parts = env_key[len(env_prefix):].lower().split('_')
                
                # Navigate to the right config section
                current = config
                for part in key_parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Set the value with type conversion
                final_key = key_parts[-1]
                current[final_key] = self._convert_env_value(env_value)
                
                logger.info(f"Applied env override: {env_key}={env_value}")
        
        return config
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type.
        
        Args:
            value: String value from environment
            
        Returns:
            Converted value
        """
        # Try to convert to number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Check for boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        elif value.lower() in ('false', 'no', '0'):
            return False
        
        # Return as string
        return value
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration values.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate algorithm parameters
        if 'algorithm' in config:
            alg = config['algorithm']
            
            rank = alg.get('rank', 1)
            if rank != 'auto' and isinstance(rank, (int, float)) and rank <= 0:
                raise ValueError("Rank must be positive or 'auto'")
                
            if alg.get('max_iterations', 1) <= 0:
                raise ValueError("Max iterations must be positive")
                
            threshold = alg.get('convergence_threshold', 0.1)
            if isinstance(threshold, str):
                try:
                    threshold = float(threshold)
                except ValueError:
                    raise ValueError("Convergence threshold must be a number")
            if not 0 < threshold < 1:
                raise ValueError("Convergence threshold must be between 0 and 1")
        
        # Validate tensor construction
        if 'tensor_construction' in config:
            tc = config['tensor_construction']
            
            if tc.get('lat_bins', 1) <= 0 or tc.get('lon_bins', 1) <= 0:
                raise ValueError("Bin counts must be positive")
                
            valid_transforms = ['none', 'log1p', 'sqrt', 'asinh']
            transform = tc.get('transform')
            if transform and transform not in valid_transforms:
                raise ValueError(f"Transform must be one of {valid_transforms}")
        
        # Validate memory settings
        if 'memory' in config:
            mem = config['memory']
            
            if mem.get('max_tensor_gb', 1) <= 0:
                raise ValueError("Memory limit must be positive")
    
    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """Get a specific configuration value.
        
        Args:
            key_path: Dot-separated path to key (e.g., 'algorithm.rank')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        current = self.config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
                
        return current
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        self._merge_configs(self.config, updates)
        self.validate_config(self.config)
    
    def save_config(self, output_path: Path) -> None:
        """Save current configuration to file.
        
        Args:
            output_path: Path to save configuration
        """
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"Saved CP-WOPT config to: {output_path}")
    
    def get_env_var_names(self) -> List[str]:
        """Return list of supported environment variables.
        
        Returns:
            List of environment variable names
        """
        # Build list of all possible env vars based on config structure
        env_vars = []
        
        def _build_env_vars(config: Dict[str, Any], prefix: str = "CPWOPT"):
            for key, value in config.items():
                env_key = f"{prefix}_{key.upper()}"
                if isinstance(value, dict):
                    _build_env_vars(value, env_key)
                else:
                    env_vars.append(env_key)
        
        # Get default config structure
        default_config = self.get_default_config()
        _build_env_vars(default_config)
        
        return sorted(env_vars)
    
    def get_env_var_descriptions(self) -> Dict[str, str]:
        """Return descriptions of environment variables.
        
        Returns:
            Dictionary mapping env var names to descriptions
        """
        return {
            'CPWOPT_ALGORITHM_RANK': 'Tensor rank (number of components)',
            'CPWOPT_ALGORITHM_MAX_ITERATIONS': 'Maximum iterations for optimization',
            'CPWOPT_ALGORITHM_CONVERGENCE_THRESHOLD': 'Convergence threshold',
            'CPWOPT_ALGORITHM_OPTIMIZATION_METHOD': 'Optimization method (als, lbfgs)',
            'CPWOPT_TENSOR_CONSTRUCTION_LAT_BINS': 'Number of latitude bins',
            'CPWOPT_TENSOR_CONSTRUCTION_LON_BINS': 'Number of longitude bins',
            'CPWOPT_TENSOR_CONSTRUCTION_TRANSFORM': 'Data transformation (none, log1p, sqrt)',
            'CPWOPT_SPARSE_SPARSITY_THRESHOLD': 'Threshold for sparse tensor (0-1)',
            'CPWOPT_SPARSE_FORCE_SPARSE': 'Force sparse tensor format (true/false)',
            'CPWOPT_MINIBATCH_ENABLED': 'Enable mini-batch processing (true/false)',
            'CPWOPT_MINIBATCH_BATCH_SIZE': 'Mini-batch size',
        }
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration without any overrides.
        
        Returns:
            Default configuration dictionary
        """
        # Always return hardcoded defaults for reliability
        return {
            'algorithm': {
                'rank': 20,
                'max_iterations': 1000,
                'convergence_threshold': 1e-6,
                'optimization_method': 'als'
            },
            'tensor_construction': {
                'lat_bins': 20,
                'lon_bins': 20,
                'transform': 'log1p',
                'aggregation': 'sum'
            },
            'sparse': {
                'sparsity_threshold': 0.95,
                'force_sparse': False
            },
            'minibatch': {
                'enabled': True,
                'batch_size': 10000
            },
            'preprocessing': {
                'lat_bins': 20,
                'lon_bins': 20,
                'transform': 'log1p'
            }
        }


# Global instance
_cpwopt_config_loader: Optional[CPWOPTConfigLoader] = None


def get_cpwopt_config(config_name: Optional[str] = None,
                     preset: Optional[str] = None) -> Dict[str, Any]:
    """Get CP-WOPT configuration.
    
    Args:
        config_name: Config file name
        preset: Preset to apply
        
    Returns:
        Configuration dictionary
    """
    global _cpwopt_config_loader
    
    if _cpwopt_config_loader is None:
        _cpwopt_config_loader = CPWOPTConfigLoader()
    
    return _cpwopt_config_loader.load_config(config_name, preset)


def register_cpwopt_config():
    """Register CP-WOPT with the biodiversity config system."""
    from src.config.biodiversity_config import get_biodiversity_config
    
    # Get default CP-WOPT config
    cpwopt_config = get_cpwopt_config()
    
    # Register with biodiversity config manager
    bio_config = get_biodiversity_config()
    bio_config.register_custom_method('cpwopt', {
        'algorithm': cpwopt_config.get('algorithm', {}),
        'tensor_construction': cpwopt_config.get('tensor_construction', {}),
        'minibatch': cpwopt_config.get('minibatch', {}),
        'sparse': cpwopt_config.get('sparse', {})
    })
    
    logger.info("Registered CP-WOPT with biodiversity config system")