"""Unified base class for all machine learning components.

This replaces the dual hierarchy of BaseRegressionMethod and BaseMLAnalyzer.
"""

import logging
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from pathlib import Path

from ..diagnostics import DHARMaDiagnostics
from ..config_loader import MLRegressionConfig
from ..abstractions.types.ml_types import ModelType

logger = logging.getLogger(__name__)


class BaseMLComponent(ABC):
    """Unified base class for all ML components in the biodiversity framework.
    
    Combines functionality from BaseRegressionMethod and BaseMLAnalyzer into
    a single coherent base class hierarchy.
    
    This class provides:
    - Unified configuration handling
    - Model diagnostics
    - Result summarization
    - Database connection support
    - Consistent interface for all ML components
    """
    
    def __init__(self,
                 config: Optional[Union[Dict[str, Any], MLRegressionConfig]] = None,
                 db_connection: Optional[Any] = None,
                 model_type: ModelType = ModelType.REGRESSION,
                 **kwargs):
        """Initialize base ML component.
        
        Args:
            config: Configuration (dict, MLRegressionConfig object, or None for defaults)
            db_connection: Optional database connection
            model_type: Type of model (regression, classification, etc.)
            **kwargs: Additional parameters stored as attributes
        """
        # Handle different config types
        if config is None:
            # Use default configuration
            self.config = MLRegressionConfig()
            self.config_dict = self.config.config
        elif isinstance(config, MLRegressionConfig):
            # Already a config object
            self.config = config
            self.config_dict = config.config
        elif isinstance(config, dict):
            # Raw dictionary - create config object
            self.config_dict = config
            # Try to create MLRegressionConfig if possible
            try:
                self.config = MLRegressionConfig(validate=False)
                self.config.config = config
            except:
                # Fallback to just storing the dict
                self.config = None
        else:
            # Assume it's some other config object
            self.config = config
            self.config_dict = config.config if hasattr(config, 'config') else {}
        
        # Core attributes
        self.db_connection = db_connection
        self.model_type = model_type
        self.is_fitted = False
        
        # Model and data storage
        self.model = None
        self.data_processed = None
        
        # Diagnostics tool
        self.diagnostics_tool = DHARMaDiagnostics()
        
        # Store additional kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def safe_get_config(self, key_path: str, default: Any = None) -> Any:
        """Safely get nested configuration value.
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'regression_analysis.models.glm')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            # Try config object first
            if self.config and hasattr(self.config, 'get'):
                return self._nested_get(self.config.config, key_path, default)
            # Fall back to config dict
            elif self.config_dict:
                return self._nested_get(self.config_dict, key_path, default)
            else:
                return default
        except (KeyError, AttributeError, TypeError):
            return default
    
    def _nested_get(self, data: Union[Dict, Any], key_path: str, default: Any) -> Any:
        """Helper for nested dictionary/object access."""
        keys = key_path.split('.')
        value = data
        
        for key in keys:
            if value is None:
                return default
            
            if isinstance(value, dict):
                value = value.get(key)
            else:
                value = getattr(value, key, None)
        
        return value if value is not None else default
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self.safe_get_config('regression_analysis.preprocessing', {})
    
    def get_glm_config(self) -> Dict[str, Any]:
        """Get GLM model configuration."""
        return self.safe_get_config('regression_analysis.models.glm', {})
    
    def get_dml_config(self) -> Dict[str, Any]:
        """Get DML configuration."""
        return self.safe_get_config('regression_analysis.models.dml', {})
    
    def get_spatial_config(self) -> Dict[str, Any]:
        """Get spatial configuration."""
        return self.safe_get_config('regression_analysis.spatial', {})
    
    def run_diagnostics(self, 
                       model: Optional[Any] = None,
                       data: Optional[Any] = None,
                       coords: Optional[Any] = None) -> Dict[str, Any]:
        """Run model diagnostics using DHARMaDiagnostics.
        
        Args:
            model: Model to diagnose (defaults to self.model)
            data: Data used for fitting (defaults to self.data_processed)
            coords: Optional coordinates for spatial tests
            
        Returns:
            Dictionary with diagnostic results
        """
        model = model or self.model
        data = data or self.data_processed
        
        if model is None:
            logger.warning(f"{self.__class__.__name__}: No model fitted yet for diagnostics")
            return {
                'model_adequate': False,
                'error': 'No model fitted yet',
                'checks_performed': []
            }
        
        return self.diagnostics_tool.run_diagnostics(model, data, coords)
    
    def summarize_results(self) -> Dict[str, Any]:
        """Generate summary of results.
        
        Returns:
            Dictionary with summary information
        """
        summary = {
            'component': self.__class__.__name__,
            'model_type': self.model_type.value if self.model_type else 'unknown',
            'is_fitted': self.is_fitted,
            'has_model': self.model is not None
        }
        
        # Add diagnostic results if available
        if self.diagnostics_tool.diagnostics_results:
            summary['diagnostics'] = self.diagnostics_tool.summarize_results()
        
        # Subclasses should override to add specific summaries
        return summary
    
    def create_visualization(self, save_path: Optional[str] = None):
        """Create visualization.
        
        Args:
            save_path: Path to save visualization
        """
        # Default implementation - subclasses should override
        logger.info(f"{self.__class__.__name__}: Visualization would be saved to {save_path}")
    
    # Abstract methods that subclasses must implement
    @abstractmethod
    def fit(self, *args, **kwargs) -> Any:
        """Fit the model/component.
        
        This is the primary method for training/fitting.
        Subclasses must implement based on their specific needs.
        """
        pass
    
    def predict(self, *args, **kwargs) -> Any:
        """Make predictions (optional for non-predictive components)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement predict()")
    
    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
        """Evaluate performance (optional)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement evaluate()")


class BaseRegressionAnalysis(BaseMLComponent):
    """Base class specifically for regression analysis methods.
    
    Extends BaseMLComponent with regression-specific functionality.
    This replaces the old BaseRegressionMethod.
    """
    
    def __init__(self, config: Optional[Union[Dict[str, Any], MLRegressionConfig]] = None, **kwargs):
        """Initialize regression analysis.
        
        Args:
            config: Configuration
            **kwargs: Additional parameters
        """
        super().__init__(config, model_type=ModelType.REGRESSION, **kwargs)
    
    @abstractmethod
    def test_hypothesis(self, hypothesis: str) -> Dict[str, Any]:
        """Test specific hypothesis.
        
        Args:
            hypothesis: Which hypothesis to test ('h0', 'h1', 'h2')
            
        Returns:
            Hypothesis test results
        """
        pass
    
    def fit_model(self, data: Any) -> Dict[str, Any]:
        """Fit the regression model.
        
        This wraps the generic fit() method for backwards compatibility.
        
        Args:
            data: Input data
            
        Returns:
            Model fitting results
        """
        return self.fit(data)
    
    @abstractmethod
    def fit(self, data: Any) -> Dict[str, Any]:
        """Fit the regression model.
        
        Args:
            data: Input data
            
        Returns:
            Model fitting results
        """
        pass


class BaseModelImplementation(BaseMLComponent):
    """Base class specifically for model implementations (GLM, GBM, etc.).
    
    Extends BaseMLComponent with model-specific functionality.
    This replaces the old BaseMLAnalyzer.
    """
    
    def __init__(self, 
                 config: Optional[Union[Dict[str, Any], MLRegressionConfig]] = None,
                 **kwargs):
        """Initialize model implementation.
        
        Args:
            config: Configuration
            **kwargs: Additional parameters
        """
        super().__init__(config, **kwargs)
        self.feature_names = None
    
    @abstractmethod
    def fit(self, X, y, **kwargs):
        """Fit the model.
        
        Args:
            X: Features
            y: Target
            **kwargs: Additional fitting parameters
        """
        pass
    
    def get_coefficient(self, name: str) -> Dict[str, Any]:
        """Get coefficient information for a specific variable."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement get_coefficient()")
    
    def get_all_coefficients(self) -> Dict[str, float]:
        """Get all model coefficients."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement get_all_coefficients()")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement get_feature_importance()")