"""
Core module for the biodiversity geoprocessing framework.

This module provides the foundational components for building extensible, 
configurable processing pipelines:

- Registry system for dynamic component management
- Builder system for creating components from configuration  
- Thread-safe component registries with validation
- Plugin loading and auto-registration capabilities

Key Features:
- Thread-safe registries with RLock protection
- Component validation during registration
- Instance caching for performance
- Configuration-driven component building
- Plugin system for extensibility
- Multiprocessing-aware design (with caveats)

Usage:
    from src.core import component_registry, ComponentBuilder
    
    # Register a component
    @component_registry.grids.register_decorator()
    class MyGrid:
        def generate_grid(self): pass
        def get_cell_size(self): pass
        def get_cell_count(self): pass
    
    # Build from config
    builder = ComponentBuilder()
    grid = builder.build_from_config('grid', {
        'class': 'MyGrid',
        'params': {'resolution': 1000}
    })

Compatibility:
- ✅ Config module integration
- ✅ Database module integration  
- ✅ Thread safety (threading.RLock)
- ⚠️ Multiprocessing: Registry locks not pickle-able
- ✅ Plugin loading via importlib
- ✅ Future module extensibility

For multiprocessing workflows, consider:
- Creating fresh registries in worker processes
- Using process-local component instances
- Or implementing pickle-able registry alternatives
"""

from .registry import (
    Registry,
    ComponentRegistry, 
    component_registry
)

from .build import (
    ComponentBuilder,
    build_grid,
    build_processor,
    register_default_components
)

# Version info
__version__ = "1.0.0"
__author__ = "Jason"

# Public API
__all__ = [
    # Registry classes and instance
    'Registry',
    'ComponentRegistry',
    'component_registry',
    
    # Builder classes and functions
    'ComponentBuilder', 
    'build_grid',
    'build_processor',
    'register_default_components',
]

# Module-level initialization
def _initialize_core():
    """Initialize core module on import."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Grid systems will be registered when first needed
        # This avoids circular import issues during module initialization
        logger.debug("Core module initialized (grid systems will be registered on-demand)")
    except Exception as e:
        logger.warning(f"Core module initialization issue: {e}")

# Initialize on import
_initialize_core()
