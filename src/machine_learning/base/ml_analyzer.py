"""Base class for machine learning analyzers."""

import logging
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseMLAnalyzer:
    """Base class for all ML analyzers in the biodiversity framework."""
    
    def __init__(self, 
                 config: Union[Dict[str, Any], Any],
                 db_connection: Optional[Any] = None,
                 model_type: Optional[Any] = None,
                 **kwargs):
        """Initialize base ML analyzer.
        
        Args:
            config: Configuration dict or object
            db_connection: Optional database connection
            model_type: Type of model (regression, classification, etc.)
            **kwargs: Additional parameters
        """
        # Store config - handle both dict and object
        if isinstance(config, dict):
            self.config = config
        else:
            # Assume it's a config object with to_dict or similar
            self.config = config if hasattr(config, 'get') else {'config': config}
        
        self.db_connection = db_connection
        self.model_type = model_type
        self.is_fitted = False
        
        # Store any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def safe_get_config(self, key_path: str, default: Any = None) -> Any:
        """Safely get nested configuration value.
        
        Args:
            key_path: Dot-separated path to config value
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            value = self.config
            for key in key_path.split('.'):
                if isinstance(value, dict):
                    value = value.get(key, default)
                else:
                    value = getattr(value, key, default)
                if value is None:
                    return default
            return value
        except (KeyError, AttributeError):
            return default
    
    def fit(self, X, y, **kwargs):
        """Fit the model. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement fit()")
    
    def predict(self, X, **kwargs):
        """Make predictions. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement predict()")
    
    def evaluate(self, X, y, **kwargs):
        """Evaluate model performance. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement evaluate()")