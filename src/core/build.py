"""Builder utilities for creating components from configuration."""

from typing import Dict, Any, Type, Optional, List
from pathlib import Path
import yaml
import importlib
import importlib.util
import logging
from .registry import component_registry, ComponentRegistry
from src.config import config

logger = logging.getLogger(__name__)

class ComponentBuilder:
    """Build components from configuration."""
    
    def __init__(self, registry: Optional[ComponentRegistry] = None):
        self.registry = registry or component_registry
        
    def build_from_config(self, component_type: str, config_dict: Dict[str, Any]) -> Any:
        """
        Build a component from configuration dictionary.
        
        Args:
            component_type: Type of component ('grid', 'processor', etc.)
            config_dict: Configuration with 'class' and optional parameters
            
        Returns:
            Instantiated component
        """
        if 'class' not in config_dict:
            raise ValueError(f"Configuration missing 'class' key: {config_dict}")
            
        class_name = config_dict['class']
        params = config_dict.get('params', {})
        
        # Get the appropriate registry
        registry_map = {
            'grid': self.registry.grids,
            'processor': self.registry.processors,
            'extractor': self.registry.extractors,
            'loader': self.registry.loaders
        }
        
        if component_type not in registry_map:
            raise ValueError(f"Unknown component type: {component_type}")
            
        registry = registry_map[component_type]
        
        # Get and instantiate the class
        try:
            cls = registry.get(class_name)
            instance = cls(**params)
            logger.info(f"Built {component_type} component: {class_name}")
            return instance
            
        except KeyError as e:
            logger.error(f"Component class not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to build component {class_name}: {e}")
            raise
    
    def build_pipeline(self, pipeline_config: List[Dict[str, Any]]) -> List[Any]:
        """Build a pipeline of components from configuration."""
        components = []
        
        for step_config in pipeline_config:
            component_type = step_config.get('type')
            if not component_type:
                raise ValueError(f"Pipeline step missing 'type': {step_config}")
                
            component = self.build_from_config(component_type, step_config)
            components.append(component)
            
        return components
    
    def load_and_register_plugins(self, plugin_dir: Path):
        """Load and register plugin components from a directory."""
        if not plugin_dir.exists():
            logger.warning(f"Plugin directory not found: {plugin_dir}")
            return
            
        for py_file in plugin_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
                
            module_name = py_file.stem
            try:
                # Import the module
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec is None or spec.loader is None:
                    logger.warning(f"Could not create module spec for {py_file}")
                    continue
                    
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Auto-register components
                if hasattr(module, 'COMPONENT_TYPE'):
                    component_type = module.COMPONENT_TYPE
                    base_class = getattr(module, 'BASE_CLASS', None)
                    
                    self.registry.auto_register_module(
                        module, 
                        f"{component_type}s",  # pluralize
                        base_class
                    )
                    
                logger.info(f"Loaded plugin: {module_name}")
                
            except Exception as e:
                logger.error(f"Failed to load plugin {py_file}: {e}")


# Convenience functions
def build_grid(grid_config: Dict[str, Any]) -> Any:
    """Build a grid from configuration."""
    builder = ComponentBuilder()
    
    # Merge with defaults from config
    grid_type = grid_config.get('grid_type', grid_config.get('class', '').lower())
    default_config = config.get(f'grids.{grid_type}', {})
    
    merged_config = {
        'class': grid_config.get('class', f"{grid_type.title()}Grid"),
        'params': {**default_config, **grid_config.get('params', {})}
    }
    
    return builder.build_from_config('grid', merged_config)


def build_processor(processor_config: Dict[str, Any]) -> Any:
    """Build a processor from configuration."""
    builder = ComponentBuilder()
    return builder.build_from_config('processor', processor_config)


def register_default_components():
    """Register all default components from the project."""
    # Import to trigger registration decorators
    try:
        from ..grid_systems import CubicGrid, HexagonalGrid
        component_registry.grids.register(CubicGrid)
        component_registry.grids.register(HexagonalGrid)
        logger.info("Registered default grid systems")
    except ImportError as e:
        logger.warning(f"Could not register grid systems: {e}")