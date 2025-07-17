"""Component registry for dynamic component management."""

import threading
from typing import Dict, Type, TypeVar, Optional, List, Any, Callable
from abc import ABC
import inspect
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

class Registry:
    """Thread-safe registry for component classes."""
    
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type] = {}
        self._instances: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._validators: List[Callable] = []
        
    def register(self, cls: Type[T], force: bool = False) -> Type[T]:
        """
        Register a class in the registry.
        
        Args:
            cls: Class to register
            force: Force registration even if already exists
            
        Returns:
            The registered class (for decorator usage)
            
        Raises:
            ValueError: If class already registered and force=False
        """
        with self._lock:
            class_name = cls.__name__
            
            if class_name in self._registry and not force:
                existing = self._registry[class_name]
                if existing is not cls:
                    raise ValueError(
                        f"Class '{class_name}' already registered in {self.name} registry. "
                        f"Existing: {existing.__module__}.{existing.__name__}, "
                        f"New: {cls.__module__}.{cls.__name__}"
                    )
                else:
                    # Same class, but still raise error to be explicit
                    raise ValueError(
                        f"Class '{class_name}' already registered in {self.name} registry. "
                        f"Use force=True to re-register."
                    )
                
            # Run validators
            for validator in self._validators:
                validator(cls)
                
            self._registry[class_name] = cls
            logger.debug(f"Registered {class_name} in {self.name} registry")
            
        return cls
    
    def register_decorator(self, force: bool = False) -> Callable:
        """Decorator for registering classes."""
        def decorator(cls: Type[T]) -> Type[T]:
            return self.register(cls, force=force)
        return decorator
    
    def get(self, name: str) -> Type:
        """
        Get a registered class by name.
        
        Args:
            name: Class name
            
        Returns:
            The registered class
            
        Raises:
            KeyError: If class not found
        """
        with self._lock:
            if name not in self._registry:
                available = list(self._registry.keys())
                raise KeyError(
                    f"'{name}' not found in {self.name} registry. "
                    f"Available: {available}"
                )
            return self._registry[name]
    
    def get_instance(self, name: str, *args, **kwargs) -> Any:
        """Get or create an instance of a registered class."""
        with self._lock:
            # Create a hashable key from args and kwargs
            import hashlib
            import json
            
            # Convert args and kwargs to a reproducible string
            try:
                args_str = json.dumps(args, sort_keys=True, default=str)
                kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
                key_data = f"{name}:{args_str}:{kwargs_str}"
                instance_key = hashlib.md5(key_data.encode()).hexdigest()
            except (TypeError, ValueError):
                # Fallback for non-serializable objects - always create new instance
                cls = self.get(name)
                return cls(*args, **kwargs)
            
            if instance_key not in self._instances:
                cls = self.get(name)
                self._instances[instance_key] = cls(*args, **kwargs)
                
            return self._instances[instance_key]
    
    def get_by_base_class(self, base_class: Type[T]) -> Dict[str, Type[T]]:
        """Get all registered classes that inherit from base_class."""
        with self._lock:
            return {
                name: cls
                for name, cls in self._registry.items()
                if issubclass(cls, base_class)
            }
    
    def list_registered(self) -> List[str]:
        """List all registered class names."""
        with self._lock:
            return list(self._registry.keys())
    
    def clear(self):
        """Clear the registry (mainly for testing)."""
        with self._lock:
            self._registry.clear()
            self._instances.clear()
            
    def add_validator(self, validator: Callable[[Type], None]):
        """Add a validator function that checks classes during registration."""
        self._validators.append(validator)
    
    def __contains__(self, name: str) -> bool:
        """Check if a class is registered."""
        with self._lock:
            return name in self._registry
    
    def __len__(self) -> int:
        """Get number of registered classes."""
        with self._lock:
            return len(self._registry)


class ComponentRegistry:
    """Central registry for all biodiversity pipeline components."""
    
    def __init__(self):
        # Create registries for different component types
        self.grids = Registry("grids")
        self.processors = Registry("processors")
        self.extractors = Registry("extractors")
        self.loaders = Registry("loaders")
        
        # Add validators
        self.grids.add_validator(self._validate_grid_class)
        self.processors.add_validator(self._validate_processor_class)
        
    def _validate_grid_class(self, cls: Type):
        """Validate grid classes have required methods."""
        required_methods = ['generate_grid', 'get_cell_size', 'get_cell_count']
        for method in required_methods:
            if not hasattr(cls, method):
                raise ValueError(
                    f"Grid class {cls.__name__} missing required method: {method}"
                )
    
    def _validate_processor_class(self, cls: Type):
        """Validate processor classes have required methods."""
        required_methods = ['process_single', 'process']
        for method in required_methods:
            if not hasattr(cls, method):
                raise ValueError(
                    f"Processor class {cls.__name__} missing required method: {method}"
                )
    
    def auto_register_module(self, module, registry_name: str, base_class: Optional[Type] = None):
        """Auto-register all classes from a module."""
        registry = getattr(self, registry_name)
        
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                # Skip if base_class specified and not a subclass
                if base_class and not issubclass(obj, base_class):
                    continue
                    
                # Skip abstract classes
                if inspect.isabstract(obj):
                    continue
                    
                try:
                    registry.register(obj)
                except ValueError as e:
                    logger.warning(f"Skipping {name}: {e}")


# Global component registry instance
component_registry = ComponentRegistry()