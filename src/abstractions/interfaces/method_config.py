"""Interface for method-specific configuration loaders.

This interface defines the contract that all method configuration loaders
must implement to ensure consistent configuration management.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class IMethodConfig(ABC):
    """Interface for method-specific configuration loaders."""
    
    @abstractmethod
    def load_config(self, config_name: Optional[str] = None,
                   preset: Optional[str] = None) -> Dict[str, Any]:
        """Load method configuration with environment overrides.
        
        Args:
            config_name: Name of config file (without .yml)
            preset: Preset name to apply
            
        Returns:
            Configuration dictionary with all settings
        """
        pass
    
    @abstractmethod
    def get_env_var_names(self) -> List[str]:
        """Return list of supported environment variables.
        
        Returns:
            List of environment variable names (e.g., ['CPWOPT_ALGORITHM_RANK'])
        """
        pass
    
    @abstractmethod
    def get_env_var_descriptions(self) -> Dict[str, str]:
        """Return descriptions of environment variables.
        
        Returns:
            Dictionary mapping env var names to descriptions
        """
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration values.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration without any overrides.
        
        Returns:
            Default configuration dictionary
        """
        pass