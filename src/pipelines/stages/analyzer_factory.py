"""Factory for creating analyzer instances."""

from typing import Dict, Tuple, Type, List
from importlib import import_module
import logging

from src.abstractions.interfaces.analyzer import IAnalyzer
from src.config.config import Config
from src.database.connection import DatabaseManager

logger = logging.getLogger(__name__)


class AnalyzerFactory:
    """
    Factory for creating analyzer instances.
    
    Decouples the analysis stage from concrete analyzer implementations,
    allowing dynamic loading and easy addition of new analysis methods.
    """
    
    # Registry of available analyzers
    # Format: method_name -> (module_path, class_name)
    _analyzers: Dict[str, Tuple[str, str]] = {
        'som': ('src.biodiversity_analysis.methods.som.analyzer_adapter', 'SOMAnalyzer'),
        # 'gwpca': ('src.biodiversity_analysis.methods.gwpca.analyzer', 'GWPCAAnalyzer'),
        # 'maxp_regions': ('src.biodiversity_analysis.methods.maxp.analyzer', 'MaxPAnalyzer')
    }
    
    @classmethod
    def create(cls, method: str, config: Config, db: DatabaseManager) -> IAnalyzer:
        """
        Create analyzer instance for the specified method.
        
        Args:
            method: Analysis method name (e.g., 'som', 'gwpca', 'maxp_regions')
            config: Configuration object
            db: Database connection manager
            
        Returns:
            Analyzer instance implementing IAnalyzer interface
            
        Raises:
            ValueError: If method is not registered
            ImportError: If analyzer module cannot be imported
            TypeError: If analyzer doesn't implement IAnalyzer interface
        """
        # Validate method
        if method not in cls._analyzers:
            available = ', '.join(cls._analyzers.keys())
            raise ValueError(
                f"Unknown analysis method: '{method}'. "
                f"Available methods: {available}"
            )
        
        module_path, class_name = cls._analyzers[method]
        
        try:
            # Dynamic import
            logger.debug(f"Importing {class_name} from {module_path}")
            module = import_module(module_path)
            analyzer_class = getattr(module, class_name)
            
            # Verify it implements IAnalyzer
            if not issubclass(analyzer_class, IAnalyzer):
                raise TypeError(
                    f"{class_name} does not implement IAnalyzer interface"
                )
            
            # Create instance with standardized constructor
            logger.info(f"Creating {class_name} instance for method '{method}'")
            return analyzer_class(config, db)
            
        except ImportError as e:
            logger.error(f"Failed to import analyzer for '{method}': {e}")
            raise ImportError(
                f"Failed to import {method} analyzer from {module_path}: {e}"
            )
        except AttributeError as e:
            logger.error(f"Analyzer class not found: {e}")
            raise AttributeError(
                f"Analyzer class {class_name} not found in {module_path}: {e}"
            )
        except Exception as e:
            logger.error(f"Failed to create analyzer: {e}")
            raise
    
    @classmethod
    def register(cls, method: str, module_path: str, class_name: str) -> None:
        """
        Register a new analyzer type.
        
        Args:
            method: Method name to register
            module_path: Python module path containing the analyzer
            class_name: Name of the analyzer class
            
        Example:
            AnalyzerFactory.register(
                'custom_method',
                'src.custom.analyzers',
                'CustomAnalyzer'
            )
        """
        logger.info(f"Registering analyzer '{method}' -> {module_path}.{class_name}")
        cls._analyzers[method] = (module_path, class_name)
    
    @classmethod
    def unregister(cls, method: str) -> None:
        """
        Unregister an analyzer type.
        
        Args:
            method: Method name to unregister
        """
        if method in cls._analyzers:
            logger.info(f"Unregistering analyzer '{method}'")
            del cls._analyzers[method]
    
    @classmethod
    def available_methods(cls) -> List[str]:
        """
        Get list of available analysis methods.
        
        Returns:
            List of registered method names
        """
        return list(cls._analyzers.keys())
    
    @classmethod
    def get_analyzer_info(cls, method: str) -> Tuple[str, str]:
        """
        Get module and class information for a method.
        
        Args:
            method: Analysis method name
            
        Returns:
            Tuple of (module_path, class_name)
            
        Raises:
            ValueError: If method is not registered
        """
        if method not in cls._analyzers:
            raise ValueError(f"Unknown analysis method: '{method}'")
        
        return cls._analyzers[method]