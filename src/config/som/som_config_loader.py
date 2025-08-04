"""Configuration loader for SOM analysis.

This module provides configuration loading for Self-Organizing Maps.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from copy import deepcopy

from src.abstractions.interfaces.method_config import IMethodConfig

logger = logging.getLogger(__name__)


class SOMConfigLoader(IMethodConfig):
    """Loads and manages SOM configuration."""
    
    def __init__(self):
        """Initialize config loader."""
        self.config_dir = Path(__file__).parent
        self.config = {}
        
    def load_config(self, config_name: Optional[str] = None,
                   preset: Optional[str] = None) -> Dict[str, Any]:
        """Load SOM configuration.
        
        Args:
            config_name: Name of config file (without .yml)
            preset: Preset name to apply
            
        Returns:
            Merged configuration dictionary
        """
        # Load main config file
        if config_name:
            config_file = self.config_dir / f"{config_name}.yml"
        else:
            config_file = self.config_dir / "som_config_optimized.yml"
            
        if config_file.exists():
            logger.info(f"Loading SOM config from: {config_file}")
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        else:
            self.config = self.get_default_config()
        
        # Apply environment variable overrides
        self.config = self._apply_env_overrides(self.config)
        
        # Validate configuration
        self.validate_config(self.config)
        
        return self.config
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        env_prefix = "SOM_"
        
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
        """Convert environment variable string to appropriate type."""
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
    
    def get_env_var_names(self) -> List[str]:
        """Return list of supported environment variables."""
        return [
            'SOM_GRID_SIZE',
            'SOM_GRID_TYPE',
            'SOM_LEARNING_RATE',
            'SOM_EPOCHS',
            'SOM_CHUNK_SIZE',
            'SOM_MAX_WORKERS',
            'SOM_ALGORITHM_TYPE',
            'SOM_NEIGHBORHOOD_INITIAL_RADIUS',
            'SOM_NEIGHBORHOOD_FINAL_RADIUS',
            'SOM_NEIGHBORHOOD_FUNCTION',
        ]
    
    def get_env_var_descriptions(self) -> Dict[str, str]:
        """Return descriptions of environment variables."""
        return {
            'SOM_GRID_SIZE': 'Size of the SOM grid (e.g., 50 for 50x50)',
            'SOM_GRID_TYPE': 'Grid topology (hexagonal or rectangular)',
            'SOM_LEARNING_RATE': 'Initial learning rate',
            'SOM_EPOCHS': 'Number of training epochs',
            'SOM_CHUNK_SIZE': 'Batch size for processing',
            'SOM_MAX_WORKERS': 'Maximum parallel workers',
            'SOM_ALGORITHM_TYPE': 'Algorithm variant (standard or vlrsom)',
            'SOM_NEIGHBORHOOD_INITIAL_RADIUS': 'Initial neighborhood radius',
            'SOM_NEIGHBORHOOD_FINAL_RADIUS': 'Final neighborhood radius',
            'SOM_NEIGHBORHOOD_FUNCTION': 'Neighborhood function (gaussian, mexican_hat)',
        }
    
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration values."""
        # Validate grid size
        grid_size = config.get('grid_size', 50)
        if not isinstance(grid_size, int) or grid_size <= 0:
            raise ValueError("Grid size must be a positive integer")
        
        # Validate learning rate
        lr = config.get('learning_rate', 0.4)
        if not 0 < lr <= 1:
            raise ValueError("Learning rate must be between 0 and 1")
        
        # Validate epochs
        epochs = config.get('epochs', 1000)
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("Epochs must be a positive integer")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration without any overrides."""
        return {
            'grid_size': 50,
            'grid_type': 'hexagonal',
            'learning_rate': 0.4,
            'epochs': 1000,
            'chunk_size': 50000,
            'max_workers': 100,
            'algorithm_type': 'vlrsom',
            'neighborhood': {
                'initial_radius': 25.0,
                'final_radius': 1.0,
                'function': 'gaussian'
            }
        }