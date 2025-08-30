from src.abstractions import get_logger
"""Registry for method-specific configuration loaders.

This module provides a central registry for all method configuration loaders,
ensuring consistent access to method configurations across the system.
"""

from typing import Dict, Type, List, Optional
import logging

from src.abstractions import IMethodConfig

logger = get_logger(__name__)


class MethodConfigRegistry:
    """Registry for method configuration loaders."""
    
    _loaders: Dict[str, Type[IMethodConfig]] = {}
    _instances: Dict[str, IMethodConfig] = {}  # Cache instances
    
    @classmethod
    def register(cls, method: str, loader_class: Type[IMethodConfig]) -> None:
        """Register a config loader for a method.
        
        Args:
            method: Method name (e.g., 'cpwopt', 'som')
            loader_class: Class implementing IMethodConfig interface
        """
        if not issubclass(loader_class, IMethodConfig):
            raise TypeError(f"{loader_class} must implement IMethodConfig interface")
        
        cls._loaders[method] = loader_class
        logger.info(f"Registered config loader for method: {method}")
    
    @classmethod
    def get_loader(cls, method: str) -> IMethodConfig:
        """Get config loader instance for a method.
        
        Args:
            method: Method name
            
        Returns:
            Config loader instance
            
        Raises:
            ValueError: If method is not registered
        """
        if method not in cls._loaders:
            available = ', '.join(cls._loaders.keys())
            raise ValueError(
                f"No config loader registered for method: '{method}'. "
                f"Available methods: {available}"
            )
        
        # Return cached instance or create new one
        if method not in cls._instances:
            cls._instances[method] = cls._loaders[method]()
        
        return cls._instances[method]
    
    @classmethod
    def list_methods(cls) -> List[str]:
        """List all registered methods.
        
        Returns:
            List of registered method names
        """
        return sorted(cls._loaders.keys())
    
    @classmethod
    def is_registered(cls, method: str) -> bool:
        """Check if a method is registered.
        
        Args:
            method: Method name to check
            
        Returns:
            True if method is registered
        """
        return method in cls._loaders
    
    @classmethod
    def get_method_config(cls, method: str, 
                         config_name: Optional[str] = None,
                         preset: Optional[str] = None) -> Dict[str, any]:
        """Convenience method to get configuration for a method.
        
        Args:
            method: Method name
            config_name: Config file name
            preset: Preset to apply
            
        Returns:
            Configuration dictionary
        """
        loader = cls.get_loader(method)
        return loader.load_config(config_name, preset)
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the instance cache."""
        cls._instances.clear()
        logger.debug("Cleared config loader instance cache")