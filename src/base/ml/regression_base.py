"""Base class for regression methods with common functionality.

Provides shared implementations for diagnostics, summarization, and visualization.
"""

import logging
from typing import Dict, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import numpy as np
import h2o

logger = logging.getLogger(__name__)


class BaseRegressionMethod(ABC):
    """Base class for all regression methods.
    
    Provides common functionality:
    - Diagnostics via DHARMaDiagnostics
    - Summary generation
    - Visualization stubs
    - Type-safe interfaces
    """
    
    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize base method.
        
        Args:
            config: ML regression configuration (MLRegressionConfig instance)
        """
        # Avoid circular import by importing here
        from src.machine_learning.config_loader import MLRegressionConfig
        from src.machine_learning.diagnostics import DHARMaDiagnostics
        
        self.config: MLRegressionConfig = config or MLRegressionConfig()
        self.diagnostics_tool: DHARMaDiagnostics = DHARMaDiagnostics()
        self.model: Optional[Any] = None  # Model instance (H2O or native)
        self.data_processed: Optional[Union[pd.DataFrame, h2o.H2OFrame]] = None
        self.results: Dict[str, Any] = {}
    
    def run_diagnostics(
        self, 
        model: Optional[Any] = None,
        data: Optional[Union[pd.DataFrame, h2o.H2OFrame]] = None,
        coords: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Run model diagnostics using DHARMaDiagnostics.
        
        Args:
            model: Model to diagnose (defaults to self.model)
            data: Data used for fitting (defaults to self.data_processed)
            coords: Optional coordinates for spatial tests (Nx2 array)
            
        Returns:
            Dictionary with diagnostic results including:
            - model_adequate: bool
            - checks_performed: List[str]
            - issues: List[str]
        """
        model = model or self.model
        data = data or self.data_processed
        
        if model is None:
            logger.warning(f"{self.__class__.__name__}: No model fitted yet for diagnostics")
            return {
                'model_adequate': False,
                'error': 'No model fitted yet',
                'checks_performed': [],
                'issues': ['No model available']
            }
        
        return self.diagnostics_tool.run_diagnostics(model, data, coords)
    
    def summarize_results(self) -> Dict[str, Any]:
        """Generate summary of method results.
        
        Returns:
            Dictionary with summary information including:
            - method: str (class name)
            - model_fitted: bool
            - diagnostics: Optional[Dict]
            - results: Dict (method-specific)
        """
        summary: Dict[str, Any] = {
            'method': self.__class__.__name__,
            'model_fitted': self.model is not None,
            'results': self.results
        }
        
        # Add diagnostic results if available
        if hasattr(self.diagnostics_tool, 'diagnostics_results') and self.diagnostics_tool.diagnostics_results:
            summary['diagnostics'] = self.diagnostics_tool.summarize_results()
        
        return summary
    
    def create_visualization(self, save_path: Optional[Union[str, Path]] = None) -> None:
        """Create method-specific visualization.
        
        Args:
            save_path: Path to save visualization
        """
        # Default implementation - subclasses should override
        if save_path:
            logger.info(f"{self.__class__.__name__}: Visualization would be saved to {save_path}")
    
    @abstractmethod
    def fit_model(self, data: Union[pd.DataFrame, h2o.H2OFrame]) -> Dict[str, Any]:
        """Fit the model - must be implemented by subclasses.
        
        Args:
            data: Input data (pandas DataFrame or H2O Frame)
            
        Returns:
            Model fitting results including:
            - success: bool
            - metrics: Dict[str, float]
            - model_params: Dict[str, Any]
        """
        pass
    
    @abstractmethod
    def load_data(
        self, 
        data_source: Union[str, Path, pd.DataFrame, h2o.H2OFrame],
        **kwargs
    ) -> Union[pd.DataFrame, h2o.H2OFrame]:
        """Load and prepare data for analysis.
        
        Args:
            data_source: Path to data or data object
            **kwargs: Additional loading parameters
            
        Returns:
            Loaded and prepared data
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        data: Union[pd.DataFrame, h2o.H2OFrame],
        **kwargs
    ) -> np.ndarray:
        """Make predictions using fitted model.
        
        Args:
            data: Input data for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Predictions as numpy array
        """
        pass