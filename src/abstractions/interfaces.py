"""
Interface definitions - only interfaces with multiple implementations.

Usage counts:
- IAnalyzer: 4 implementations (CPWOPT, SOM, KMeans, Base)
- IMethodConfig: 7 implementations (various config loaders)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class IAnalyzer(ABC):
    """Base analyzer interface - 4 implementations."""
    
    @abstractmethod
    def analyze(self, data: Any) -> 'AnalysisResult':
        """Perform analysis on data."""
        pass


class IMethodConfig(ABC):
    """Configuration interface - 7 implementations."""
    
    @abstractmethod
    def load_config(self, preset: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration for the method."""
        pass