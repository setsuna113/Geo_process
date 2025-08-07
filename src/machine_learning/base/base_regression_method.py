"""Base class for regression methods with common functionality.

Provides shared implementations for diagnostics, summarization, and visualization.
"""

import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from ..diagnostics import DHARMaDiagnostics
from ..config_loader import MLRegressionConfig

logger = logging.getLogger(__name__)


class BaseRegressionMethod(ABC):
    """Base class for all regression methods.
    
    Provides common functionality:
    - Diagnostics via DHARMaDiagnostics
    - Summary generation
    - Visualization stubs
    """
    
    def __init__(self, config: Optional[MLRegressionConfig] = None):
        """Initialize base method.
        
        Args:
            config: ML regression configuration
        """
        self.config = config or MLRegressionConfig()
        self.diagnostics_tool = DHARMaDiagnostics()
        self.model = None  # To be set by subclasses
        self.data_processed = None
    
    def run_diagnostics(self, model: Optional[Any] = None, 
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
        """Generate summary of method results.
        
        Returns:
            Dictionary with summary information
        """
        summary = {
            'method': self.__class__.__name__,
            'model_fitted': self.model is not None
        }
        
        # Add diagnostic results if available
        if self.diagnostics_tool.diagnostics_results:
            summary['diagnostics'] = self.diagnostics_tool.summarize_results()
        
        # Subclasses should override to add method-specific summaries
        return summary
    
    def create_visualization(self, save_path: Optional[str] = None):
        """Create method-specific visualization.
        
        Args:
            save_path: Path to save visualization
        """
        # Default implementation - subclasses should override
        logger.info(f"{self.__class__.__name__}: Visualization would be saved to {save_path}")
        pass
    
    @abstractmethod
    def fit_model(self, data: Any) -> Dict[str, Any]:
        """Fit the model - must be implemented by subclasses.
        
        Args:
            data: Input data
            
        Returns:
            Model fitting results
        """
        pass
    
    @abstractmethod
    def test_hypothesis(self, hypothesis: str) -> Dict[str, Any]:
        """Test specific hypothesis - must be implemented by subclasses.
        
        Args:
            hypothesis: Which hypothesis to test ('h0', 'h1', 'h2')
            
        Returns:
            Hypothesis test results
        """
        pass