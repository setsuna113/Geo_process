from src.abstractions import get_logger
"""
Import Helpers for Biodiversity Analysis

Provides robust import mechanisms with fallbacks for development environments.
"""

import sys
import importlib
import logging
from pathlib import Path
from typing import Any, Optional, Union

logger = get_logger(__name__)


def safe_import(module_path: str, fallback_paths: Optional[list] = None) -> Any:
    """
    Safely import a module with fallback mechanisms.
    
    Args:
        module_path: Full module path (e.g., 'src.config.stat.paths_config')
        fallback_paths: List of paths to add to sys.path as fallbacks
        
    Returns:
        Imported module
        
    Raises:
        ImportError: If module cannot be imported with any method
    """
    
    # Try direct import first (works when package is installed)
    try:
        return importlib.import_module(module_path)
    except ImportError as e:
        logger.debug(f"Direct import failed for {module_path}: {e}")
    
    # Try with fallback paths
    if fallback_paths:
        original_path = sys.path.copy()
        try:
            for path in fallback_paths:
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
            
            return importlib.import_module(module_path)
        except ImportError as e:
            logger.debug(f"Fallback import failed for {module_path}: {e}")
        finally:
            # Restore original path
            sys.path[:] = original_path
    
    # Final attempt with project root discovery
    try:
        project_root = discover_project_root()
        if project_root and str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        return importlib.import_module(module_path)
    except ImportError as e:
        logger.error(f"All import attempts failed for {module_path}: {e}")
        raise ImportError(f"Cannot import {module_path}. Is the package installed or PYTHONPATH set?")


def discover_project_root(start_path: Optional[Path] = None) -> Optional[Path]:
    """
    Discover the project root by looking for common indicators.
    
    Args:
        start_path: Path to start searching from (default: current file location)
        
    Returns:
        Project root path or None if not found
    """
    if start_path is None:
        start_path = Path(__file__).parent
    
    # Look for common project root indicators
    indicators = [
        'pyproject.toml',
        'setup.py',
        'setup.cfg', 
        '.git',
        'src',
        'README.md'
    ]
    
    current = start_path.resolve()
    
    # Walk up the directory tree
    for _ in range(10):  # Limit search depth
        for indicator in indicators:
            if (current / indicator).exists():
                logger.debug(f"Found project root at {current} (indicator: {indicator})")
                return current
        
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent
    
    logger.warning("Could not discover project root")
    return None


def safe_import_class(module_path: str, class_name: str, 
                      alternative_names: Optional[list] = None) -> Any:
    """
    Safely import a class with alternative names.
    
    Args:
        module_path: Module path containing the class
        class_name: Primary class name to import
        alternative_names: List of alternative class names to try
        
    Returns:
        Imported class
        
    Raises:
        ImportError: If class cannot be imported with any name
    """
    
    try:
        module = safe_import(module_path)
    except ImportError:
        raise ImportError(f"Cannot import module {module_path}")
    
    # Try primary class name
    if hasattr(module, class_name):
        return getattr(module, class_name)
    
    # Try alternative names
    if alternative_names:
        for alt_name in alternative_names:
            if hasattr(module, alt_name):
                logger.info(f"Using alternative class name '{alt_name}' instead of '{class_name}'")
                return getattr(module, alt_name)
    
    # List available attributes for debugging
    available_attrs = [name for name in dir(module) if not name.startswith('_')]
    raise ImportError(
        f"Class '{class_name}' not found in module {module_path}. "
        f"Available: {available_attrs}"
    )


class ImportContext:
    """Context manager for temporary sys.path modifications"""
    
    def __init__(self, paths: Union[str, Path, list]):
        """
        Initialize with paths to temporarily add to sys.path
        
        Args:
            paths: Single path or list of paths to add
        """
        if isinstance(paths, (str, Path)):
            paths = [paths]
        
        self.paths = [str(p) for p in paths]
        self.original_path = None
    
    def __enter__(self):
        self.original_path = sys.path.copy()
        
        for path in self.paths:
            if path not in sys.path:
                sys.path.insert(0, path)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_path is not None:
            sys.path[:] = self.original_path