"""Component registry for dynamic component management."""

import threading
from typing import Dict, Type, TypeVar, Optional, List, Any, Callable, Union, Set
from abc import ABC
import inspect
import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class MemoryUsage(Enum):
    """Memory usage levels for components."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class ComponentMetadata:
    """Metadata for registered components."""
    name: str
    component_class: Type
    
    # Capability flags
    supports_lazy_loading: bool = False
    supports_tiles: bool = False
    supports_streaming: bool = False
    supports_memory_mapping: bool = False
    supports_skip_resampling: bool = False  # New capability
    
    # Format support
    supported_formats: Set[str] = field(default_factory=set)
    input_formats: Set[str] = field(default_factory=set)
    output_formats: Set[str] = field(default_factory=set)
    
    # Resource requirements
    memory_usage: MemoryUsage = MemoryUsage.MEDIUM
    memory_estimate_mb: Optional[int] = None
    cpu_intensive: bool = False
    requires_gpu: bool = False
    
    # Data type compatibility
    supported_data_types: Set[str] = field(default_factory=set)  # Int32, UInt16, Float64, etc.
    preferred_data_types: Set[str] = field(default_factory=set)
    
    # Resolution and scale support
    min_resolution: Optional[float] = None
    max_resolution: Optional[float] = None
    optimal_tile_size: Optional[int] = None
    
    # Additional metadata
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    tags: Set[str] = field(default_factory=set)


class EnhancedRegistry:
    """Enhanced thread-safe registry with metadata support."""
    
    def __init__(self, name: str):
        self.name = name
        self._components: Dict[str, ComponentMetadata] = {}
        self._instances: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._validators: List[Callable] = []
        
    def register(self, 
                 cls: Type[T], 
                 metadata: Optional[ComponentMetadata] = None,
                 force: bool = False,
                 **metadata_kwargs) -> Type[T]:
        """
        Register a class with metadata in the registry.
        
        Args:
            cls: Class to register
            metadata: Pre-built metadata object
            force: Force registration even if already exists
            **metadata_kwargs: Metadata fields to set
            
        Returns:
            The registered class (for decorator usage)
        """
        with self._lock:
            class_name = cls.__name__
            
            if class_name in self._components and not force:
                existing = self._components[class_name]
                if existing.component_class is not cls:
                    raise ValueError(
                        f"Class '{class_name}' already registered in {self.name} registry. "
                        f"Existing: {existing.component_class.__module__}.{existing.component_class.__name__}, "
                        f"New: {cls.__module__}.{cls.__name__}"
                    )
                else:
                    raise ValueError(
                        f"Class '{class_name}' already registered in {self.name} registry. "
                        f"Use force=True to re-register."
                    )
                
            # Create metadata if not provided
            if metadata is None:
                metadata = ComponentMetadata(
                    name=class_name,
                    component_class=cls,
                    **metadata_kwargs
                )
            
            # Run validators
            for validator in self._validators:
                validator(cls)
                
            self._components[class_name] = metadata
            logger.debug(f"Registered {class_name} in {self.name} registry")
            
        return cls
    
    def register_decorator(self, 
                          metadata: Optional[ComponentMetadata] = None,
                          force: bool = False,
                          **metadata_kwargs) -> Callable:
        """Decorator for registering classes with metadata."""
        def decorator(cls: Type[T]) -> Type[T]:
            return self.register(cls, metadata=metadata, force=force, **metadata_kwargs)
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
            if name not in self._components:
                # Try to auto-register grid components on first access
                if self.name == "grids" and name in ["CubicGrid", "HexagonalGrid"]:
                    try:
                        self._auto_register_grid_components()
                    except ImportError:
                        pass  # Grid systems not available
                
                # Check again after auto-registration
                if name not in self._components:
                    available = list(self._components.keys())
                    raise KeyError(
                        f"'{name}' not found in {self.name} registry. "
                        f"Available: {available}"
                    )
            return self._components[name].component_class
    
    def _auto_register_grid_components(self):
        """Auto-register grid components to avoid circular imports."""
        try:
            # Use lazy imports to avoid circular dependencies
            # Only import when explicitly requested (on-demand loading)
            
            if "CubicGrid" not in self._components:
                from ..domain.grid_systems.cubic_grid import CubicGrid
                self.register(CubicGrid)
                logger.debug("Auto-registered CubicGrid")
            
            if "HexagonalGrid" not in self._components:
                from ..domain.grid_systems.hexagonal_grid import HexagonalGrid
                self.register(HexagonalGrid)
                logger.debug("Auto-registered HexagonalGrid")
                
        except ImportError as e:
            logger.debug(f"Could not auto-register grid components: {e}")
            raise
    
    def get_metadata(self, name: str) -> ComponentMetadata:
        """Get metadata for a registered component."""
        with self._lock:
            if name not in self._components:
                available = list(self._components.keys())
                raise KeyError(
                    f"'{name}' not found in {self.name} registry. "
                    f"Available: {available}"
                )
            return self._components[name]
    
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
    
    def get_by_base_class(self, base_class: Type[T]) -> Dict[str, ComponentMetadata]:
        """Get all registered components that inherit from base_class."""
        with self._lock:
            return {
                name: metadata
                for name, metadata in self._components.items()
                if issubclass(metadata.component_class, base_class)
            }
    
    def find_by_format(self, 
                      format_name: str, 
                      operation: str = "input") -> List[ComponentMetadata]:
        """Find components that support a specific format."""
        with self._lock:
            results = []
            for metadata in self._components.values():
                formats = metadata.input_formats if operation == "input" else metadata.output_formats
                if format_name.lower() in {f.lower() for f in formats}:
                    results.append(metadata)
            return results
    
    def find_by_capability(self, **capabilities) -> List[ComponentMetadata]:
        """Find components with specific capabilities."""
        with self._lock:
            results = []
            for metadata in self._components.values():
                matches = True
                for capability, required_value in capabilities.items():
                    if hasattr(metadata, capability):
                        actual_value = getattr(metadata, capability)
                        if actual_value != required_value:
                            matches = False
                            break
                    else:
                        matches = False
                        break
                
                if matches:
                    results.append(metadata)
            return results
    
    def find_optimal_for_data_type(self, 
                                  data_type: str, 
                                  memory_limit: Optional[MemoryUsage] = None) -> Optional[ComponentMetadata]:
        """Find the optimal component for a given data type."""
        with self._lock:
            candidates = []
            
            for metadata in self._components.values():
                # Check data type support
                if data_type not in metadata.supported_data_types:
                    continue
                
                # Check memory requirements
                if memory_limit and self._memory_usage_value(metadata.memory_usage) > self._memory_usage_value(memory_limit):
                    continue
                
                candidates.append(metadata)
            
            if not candidates:
                return None
            
            # Prefer components that list this data type as preferred
            preferred = [c for c in candidates if data_type in c.preferred_data_types]
            if preferred:
                candidates = preferred
            
            # Sort by memory efficiency (lower is better)
            candidates.sort(key=lambda c: self._memory_usage_value(c.memory_usage))
            
            return candidates[0]
    
    def find_for_resolution(self, 
                           resolution: float,
                           format_name: Optional[str] = None) -> List[ComponentMetadata]:
        """Find components suitable for a specific resolution."""
        with self._lock:
            results = []
            
            for metadata in self._components.values():
                # Check resolution constraints
                if metadata.min_resolution is not None and resolution < metadata.min_resolution:
                    continue
                if metadata.max_resolution is not None and resolution > metadata.max_resolution:
                    continue
                
                # Check format if specified
                if format_name:
                    formats = metadata.input_formats.union(metadata.supported_formats)
                    if format_name.lower() not in {f.lower() for f in formats}:
                        continue
                
                results.append(metadata)
            
            return results
    
    def get_fallback_chain(self, 
                          primary_name: str, 
                          fallback_criteria: Optional[Dict] = None) -> List[ComponentMetadata]:
        """Get a fallback chain for a component."""
        with self._lock:
            chain = []
            
            # Start with primary component if it exists
            if primary_name in self._components:
                chain.append(self._components[primary_name])
            
            # Add fallbacks based on criteria
            if fallback_criteria:
                fallbacks = self.find_by_capability(**fallback_criteria)
                for fallback in fallbacks:
                    if fallback.name != primary_name and fallback not in chain:
                        chain.append(fallback)
            
            return chain
    
    def _memory_usage_value(self, usage: MemoryUsage) -> int:
        """Convert memory usage enum to comparable integer."""
        mapping = {
            MemoryUsage.LOW: 1,
            MemoryUsage.MEDIUM: 2,
            MemoryUsage.HIGH: 3,
            MemoryUsage.EXTREME: 4
        }
        return mapping.get(usage, 2)
    
    def list_registered(self) -> List[str]:
        """List all registered class names."""
        with self._lock:
            return list(self._components.keys())
    
    def clear(self):
        """Clear the registry (mainly for testing)."""
        with self._lock:
            self._components.clear()
            self._instances.clear()
            
    def add_validator(self, validator: Callable[[Type], None]):
        """Add a validator function that checks classes during registration."""
        self._validators.append(validator)
    
    def __contains__(self, name: str) -> bool:
        """Check if a class is registered."""
        with self._lock:
            return name in self._components
    
    def __len__(self) -> int:
        """Get number of registered classes."""
        with self._lock:
            return len(self._components)


class RegistryFactory:
    """Factory for creating and managing multiple registry instances."""
    
    def __init__(self):
        self._registries: Dict[str, EnhancedRegistry] = {}
        self._lock = threading.RLock()
    
    def create_registry(self, name: str) -> EnhancedRegistry:
        """Create a new registry instance."""
        with self._lock:
            if name in self._registries:
                return self._registries[name]
            
            registry = EnhancedRegistry(name)
            self._registries[name] = registry
            logger.debug(f"Created registry: {name}")
            return registry
    
    def get_registry(self, name: str) -> EnhancedRegistry:
        """Get an existing registry."""
        with self._lock:
            if name not in self._registries:
                raise KeyError(f"Registry '{name}' not found")
            return self._registries[name]
    
    def list_registries(self) -> List[str]:
        """List all registry names."""
        with self._lock:
            return list(self._registries.keys())
    
    def clear_all(self):
        """Clear all registries."""
        with self._lock:
            for registry in self._registries.values():
                registry.clear()
            self._registries.clear()


# Legacy Registry class for backward compatibility
class Registry(EnhancedRegistry):
    """Backward compatibility wrapper."""
    
    def register(self, 
                 cls: Type[T], 
                 metadata: Optional[ComponentMetadata] = None,
                 force: bool = False,
                 **metadata_kwargs) -> Type[T]:
        """Legacy register method without metadata."""
        # Pass through all parameters to maintain compatibility
        return super().register(cls, metadata=metadata, force=force, **metadata_kwargs)
    
    def register_decorator(self, 
                          metadata: Optional[ComponentMetadata] = None,
                          force: bool = False,
                          **metadata_kwargs) -> Callable:
        """Legacy decorator method."""
        # Pass through all parameters to maintain compatibility  
        return super().register_decorator(metadata=metadata, force=force, **metadata_kwargs)


class ComponentRegistry:
    """Enhanced central registry for all biodiversity pipeline components."""
    
    def __init__(self):
        # Create factory for managing registries
        self.factory = RegistryFactory()
        
        # Create registries for different component types
        self.grids = self.factory.create_registry("grids")
        self.processors = self.factory.create_registry("processors")
        self.extractors = self.factory.create_registry("extractors")
        self.loaders = self.factory.create_registry("loaders")
        
        # New specialized registries
        self.data_sources = self.factory.create_registry("data_sources")
        self.raster_sources = self.factory.create_registry("raster_sources")
        self.vector_sources = self.factory.create_registry("vector_sources")
        self.resamplers = self.factory.create_registry("resamplers")
        self.tile_processors = self.factory.create_registry("tile_processors")
        
        # Machine Learning registries
        self.ml_models = self.factory.create_registry("ml_models")
        self.feature_builders = self.factory.create_registry("feature_builders")
        self.imputation_strategies = self.factory.create_registry("imputation_strategies")
        self.cv_strategies = self.factory.create_registry("cv_strategies")
        
        # Add validators
        self.grids.add_validator(self._validate_grid_class)
        self.processors.add_validator(self._validate_processor_class)
        self.resamplers.add_validator(self._validate_resampler_class)
        self.tile_processors.add_validator(self._validate_tile_processor_class)
        
        # ML validators
        self.ml_models.add_validator(self._validate_ml_model_class)
        self.feature_builders.add_validator(self._validate_feature_builder_class)
        self.cv_strategies.add_validator(self._validate_cv_strategy_class)
        
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
    
    def _validate_resampler_class(self, cls: Type):
        """Validate resampler classes have required methods."""
        required_methods = ['resample', 'get_supported_methods']
        for method in required_methods:
            if not hasattr(cls, method):
                raise ValueError(
                    f"Resampler class {cls.__name__} missing required method: {method}"
                )
    
    def _validate_tile_processor_class(self, cls: Type):
        """Validate tile processor classes have required methods."""
        required_methods = ['process_tile', 'get_processing_strategy']
        for method in required_methods:
            if not hasattr(cls, method):
                raise ValueError(
                    f"Tile processor class {cls.__name__} missing required method: {method}"
                )
    
    def _validate_ml_model_class(self, cls: Type):
        """Validate ML model classes have required methods."""
        required_methods = ['fit', 'predict', 'evaluate', 'save_model', 'load_model']
        for method in required_methods:
            if not hasattr(cls, method):
                raise ValueError(
                    f"ML model class {cls.__name__} missing required method: {method}"
                )
    
    def _validate_feature_builder_class(self, cls: Type):
        """Validate feature builder classes have required methods."""
        required_methods = ['build_features', 'get_feature_names', 'get_required_columns']
        for method in required_methods:
            if not hasattr(cls, method):
                raise ValueError(
                    f"Feature builder class {cls.__name__} missing required method: {method}"
                )
    
    def _validate_cv_strategy_class(self, cls: Type):
        """Validate CV strategy classes have required methods."""
        required_methods = ['split', 'get_n_splits']
        for method in required_methods:
            if not hasattr(cls, method):
                raise ValueError(
                    f"CV strategy class {cls.__name__} missing required method: {method}"
                )
    
    def auto_register_module(self, module, registry_name: str, base_class: Optional[Type] = None):
        """Auto-register all classes from a module."""
        registry = self.factory.get_registry(registry_name)
        
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
    
    def find_optimal_resampler(self, 
                              data_type: str, 
                              method: str = "bilinear",
                              memory_limit: Optional[MemoryUsage] = None) -> Optional[ComponentMetadata]:
        """Find optimal resampler for given data type and method."""
        candidates = []
        
        for metadata in self.resamplers._components.values():
            # Check if method is supported
            if hasattr(metadata.component_class, 'get_supported_methods'):
                try:
                    instance = metadata.component_class()
                    supported_methods = instance.get_supported_methods()
                    if method not in supported_methods:
                        continue
                except:
                    continue
            
            # Check data type support
            if data_type not in metadata.supported_data_types:
                continue
            
            # Check memory requirements
            if memory_limit and self._memory_usage_value(metadata.memory_usage) > self._memory_usage_value(memory_limit):
                continue
            
            candidates.append(metadata)
        
        if not candidates:
            return None
        
        # Prefer components optimized for this data type
        preferred = [c for c in candidates if data_type in c.preferred_data_types]
        if preferred:
            return preferred[0]
        
        return candidates[0]
    
    def find_data_source_for_format(self, format_name: str) -> Optional[ComponentMetadata]:
        """Find appropriate data source for a file format."""
        # Check raster sources first
        raster_candidates = self.raster_sources.find_by_format(format_name, "input")
        if raster_candidates:
            return raster_candidates[0]
        
        # Check vector sources
        vector_candidates = self.vector_sources.find_by_format(format_name, "input")
        if vector_candidates:
            return vector_candidates[0]
        
        # Check generic data sources
        generic_candidates = self.data_sources.find_by_format(format_name, "input")
        if generic_candidates:
            return generic_candidates[0]
        
        return None
    
    def find_tile_processor_for_strategy(self, 
                                        strategy: str,
                                        memory_limit: Optional[MemoryUsage] = None) -> List[ComponentMetadata]:
        """Find tile processors that support a specific processing strategy."""
        results = []
        
        for metadata in self.tile_processors._components.values():
            # Check if processor supports the strategy
            if hasattr(metadata.component_class, 'get_processing_strategy'):
                try:
                    instance = metadata.component_class()
                    supported_strategies = instance.get_processing_strategy()
                    if isinstance(supported_strategies, str):
                        supported_strategies = [supported_strategies]
                    
                    if strategy not in supported_strategies:
                        continue
                except:
                    continue
            
            # Check memory requirements
            if memory_limit and self._memory_usage_value(metadata.memory_usage) > self._memory_usage_value(memory_limit):
                continue
            
            results.append(metadata)
        
        return results
    
    def find_models_for_task(self, 
                           task_type: str,
                           handles_missing: Optional[bool] = None,
                           requires_scaling: Optional[bool] = None) -> List[ComponentMetadata]:
        """
        Find ML models suitable for a specific task type.
        
        Args:
            task_type: Type of ML task ("regression", "classification", etc.)
            handles_missing: Filter by ability to handle missing values
            requires_scaling: Filter by scaling requirements
            
        Returns:
            List of suitable model metadata
        """
        results = []
        
        for metadata in self.ml_models._components.values():
            # Check task type
            if hasattr(metadata, 'model_type') and metadata.model_type != task_type:
                continue
            
            # Check missing value handling
            if handles_missing is not None:
                if hasattr(metadata, 'handles_missing_values'):
                    if metadata.handles_missing_values != handles_missing:
                        continue
                elif handles_missing:
                    # If we need missing handling but metadata doesn't specify, skip
                    continue
            
            # Check scaling requirements
            if requires_scaling is not None:
                if hasattr(metadata, 'requires_scaling'):
                    if metadata.requires_scaling != requires_scaling:
                        continue
            
            results.append(metadata)
        
        return results
    
    def find_feature_builders_by_category(self, 
                                        category: str) -> List[ComponentMetadata]:
        """
        Find feature builders by category.
        
        Args:
            category: Feature category ("spatial", "ecological", "richness", etc.)
            
        Returns:
            List of feature builder metadata
        """
        results = []
        
        for metadata in self.feature_builders._components.values():
            if hasattr(metadata, 'feature_categories'):
                if category in metadata.feature_categories:
                    results.append(metadata)
        
        return results
    
    def find_spatial_cv_strategies(self,
                                 spatial_aware: bool = True,
                                 min_folds: Optional[int] = None) -> List[ComponentMetadata]:
        """
        Find cross-validation strategies.
        
        Args:
            spatial_aware: Whether to filter for spatial-aware strategies
            min_folds: Minimum number of folds supported
            
        Returns:
            List of CV strategy metadata
        """
        results = []
        
        for metadata in self.cv_strategies._components.values():
            # Check spatial awareness
            if spatial_aware:
                if not hasattr(metadata, 'spatial_aware') or not metadata.spatial_aware:
                    continue
            
            # Check minimum folds
            if min_folds is not None:
                if hasattr(metadata, 'min_folds') and metadata.min_folds < min_folds:
                    continue
            
            results.append(metadata)
        
        return results
    
    def get_format_compatibility_matrix(self) -> Dict[str, Dict[str, List[str]]]:
        """Get format compatibility matrix for all components."""
        matrix: Dict[str, Dict[str, List[str]]] = {}
        
        for registry_name in self.factory.list_registries():
            registry = self.factory.get_registry(registry_name)
            matrix[registry_name] = {
                'input': [],
                'output': [],
                'supported': []
            }
            
            for metadata in registry._components.values():
                matrix[registry_name]['input'].extend(metadata.input_formats)
                matrix[registry_name]['output'].extend(metadata.output_formats)
                matrix[registry_name]['supported'].extend(metadata.supported_formats)
            
            # Remove duplicates and sort
            for key in matrix[registry_name]:
                matrix[registry_name][key] = sorted(list(set(matrix[registry_name][key])))
        
        return matrix
    
    def _memory_usage_value(self, usage: MemoryUsage) -> int:
        """Convert memory usage enum to comparable integer."""
        mapping = {
            MemoryUsage.LOW: 1,
            MemoryUsage.MEDIUM: 2,
            MemoryUsage.HIGH: 3,
            MemoryUsage.EXTREME: 4
        }
        return mapping.get(usage, 2)


# Global component registry instance
component_registry = ComponentRegistry()


# Convenience functions for common operations
def register_raster_source(cls: Type[T], 
                          supported_formats: List[str],
                          memory_usage: MemoryUsage = MemoryUsage.MEDIUM,
                          **kwargs) -> Type[T]:
    """Convenience function to register a raster source."""
    metadata = ComponentMetadata(
        name=cls.__name__,
        component_class=cls,
        supported_formats=set(supported_formats),
        input_formats=set(supported_formats),
        memory_usage=memory_usage,
        **kwargs
    )
    return component_registry.raster_sources.register(cls, metadata=metadata)


def register_resampler(cls: Type[T],
                      supported_data_types: List[str],
                      preferred_data_types: Optional[List[str]] = None,
                      memory_usage: MemoryUsage = MemoryUsage.MEDIUM,
                      **kwargs) -> Type[T]:
    """Convenience function to register a resampler."""
    metadata = ComponentMetadata(
        name=cls.__name__,
        component_class=cls,
        supported_data_types=set(supported_data_types),
        preferred_data_types=set(preferred_data_types or []),
        memory_usage=memory_usage,
        **kwargs
    )
    return component_registry.resamplers.register(cls, metadata=metadata)


def register_tile_processor(cls: Type[T],
                           supports_streaming: bool = False,
                           supports_memory_mapping: bool = False,
                           memory_usage: MemoryUsage = MemoryUsage.MEDIUM,
                           optimal_tile_size: Optional[int] = None,
                           **kwargs) -> Type[T]:
    """Convenience function to register a tile processor."""
    metadata = ComponentMetadata(
        name=cls.__name__,
        component_class=cls,
        supports_streaming=supports_streaming,
        supports_memory_mapping=supports_memory_mapping,
        supports_tiles=True,
        memory_usage=memory_usage,
        optimal_tile_size=optimal_tile_size,
        **kwargs
    )
    return component_registry.tile_processors.register(cls, metadata=metadata)


# Decorator versions
def raster_source(*formats, memory_usage: MemoryUsage = MemoryUsage.MEDIUM, **kwargs):
    """Decorator for registering raster sources."""
    def decorator(cls):
        return register_raster_source(cls, list(formats), memory_usage=memory_usage, **kwargs)
    return decorator


def resampler(*data_types, preferred_types: Optional[List[str]] = None, 
              memory_usage: MemoryUsage = MemoryUsage.MEDIUM, **kwargs):
    """Decorator for registering resamplers."""
    def decorator(cls):
        return register_resampler(cls, list(data_types), preferred_types, memory_usage=memory_usage, **kwargs)
    return decorator


def tile_processor(supports_streaming: bool = False, 
                  supports_memory_mapping: bool = False,
                  memory_usage: MemoryUsage = MemoryUsage.MEDIUM,
                  optimal_tile_size: Optional[int] = None, **kwargs):
    """Decorator for registering tile processors."""
    def decorator(cls):
        return register_tile_processor(cls, supports_streaming, supports_memory_mapping, 
                                     memory_usage, optimal_tile_size, **kwargs)
    return decorator


# ML-specific registration functions
def register_ml_model(cls: Type[T],
                     model_type: str,
                     requires_scaling: bool = False,
                     handles_missing_values: bool = False,
                     memory_usage: MemoryUsage = MemoryUsage.MEDIUM,
                     **kwargs) -> Type[T]:
    """Convenience function to register an ML model."""
    # Extend ComponentMetadata with ML-specific fields
    metadata = ComponentMetadata(
        name=cls.__name__,
        component_class=cls,
        memory_usage=memory_usage,
        **kwargs
    )
    # Add ML-specific attributes
    metadata.model_type = model_type
    metadata.requires_scaling = requires_scaling
    metadata.handles_missing_values = handles_missing_values
    
    return component_registry.ml_models.register(cls, metadata=metadata)


def register_feature_builder(cls: Type[T],
                           feature_categories: List[str],
                           required_columns: Set[str],
                           memory_usage: MemoryUsage = MemoryUsage.LOW,
                           **kwargs) -> Type[T]:
    """Convenience function to register a feature builder."""
    metadata = ComponentMetadata(
        name=cls.__name__,
        component_class=cls,
        memory_usage=memory_usage,
        **kwargs
    )
    # Add feature builder specific attributes
    metadata.feature_categories = set(feature_categories)
    metadata.required_columns = required_columns
    
    return component_registry.feature_builders.register(cls, metadata=metadata)


def register_cv_strategy(cls: Type[T],
                        spatial_aware: bool = False,
                        min_samples_per_fold: int = 10,
                        **kwargs) -> Type[T]:
    """Convenience function to register a CV strategy."""
    metadata = ComponentMetadata(
        name=cls.__name__,
        component_class=cls,
        memory_usage=MemoryUsage.LOW,
        **kwargs
    )
    # Add CV-specific attributes
    metadata.spatial_aware = spatial_aware
    metadata.min_samples_per_fold = min_samples_per_fold
    
    return component_registry.cv_strategies.register(cls, metadata=metadata)


def register_imputation_strategy(cls: Type[T],
                               strategy_name: str,
                               handles_spatial: bool = False,
                               **kwargs) -> Type[T]:
    """Convenience function to register an imputation strategy."""
    metadata = ComponentMetadata(
        name=cls.__name__,
        component_class=cls,
        memory_usage=MemoryUsage.LOW,
        **kwargs
    )
    # Add imputation-specific attributes
    metadata.strategy_name = strategy_name
    metadata.handles_spatial = handles_spatial
    
    return component_registry.imputation_strategies.register(cls, metadata=metadata)


# ML Decorator versions
def ml_model(model_type: str,
            requires_scaling: bool = False,
            handles_missing_values: bool = False,
            memory_usage: MemoryUsage = MemoryUsage.MEDIUM,
            **kwargs):
    """Decorator for registering ML models."""
    def decorator(cls):
        return register_ml_model(cls, model_type, requires_scaling, 
                               handles_missing_values, memory_usage, **kwargs)
    return decorator


def feature_builder(*categories, required_columns: Set[str] = None,
                   memory_usage: MemoryUsage = MemoryUsage.LOW, **kwargs):
    """Decorator for registering feature builders."""
    def decorator(cls):
        return register_feature_builder(cls, list(categories), 
                                      required_columns or set(), 
                                      memory_usage, **kwargs)
    return decorator


def cv_strategy(spatial_aware: bool = False,
               min_samples_per_fold: int = 10,
               **kwargs):
    """Decorator for registering CV strategies."""
    def decorator(cls):
        return register_cv_strategy(cls, spatial_aware, min_samples_per_fold, **kwargs)
    return decorator


def imputation_strategy(strategy_name: str,
                       handles_spatial: bool = False,
                       **kwargs):
    """Decorator for registering imputation strategies."""
    def decorator(cls):
        return register_imputation_strategy(cls, strategy_name, handles_spatial, **kwargs)
    return decorator