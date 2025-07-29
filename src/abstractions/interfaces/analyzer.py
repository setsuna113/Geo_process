# src/foundations/interfaces/analyzer.py
"""Pure analyzer interface - NO IMPLEMENTATIONS!"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple, List, Callable
from pathlib import Path
import numpy as np
import xarray as xr
from dataclasses import dataclass

@dataclass
class AnalysisMetadata:
    """Metadata for spatial analysis runs."""
    analysis_type: str
    input_shape: Tuple[int, ...]
    input_bands: List[str]
    parameters: Dict[str, Any]
    processing_time: float
    timestamp: str
    data_source: Optional[str] = None
    normalization_applied: bool = False
    coordinate_system: str = "EPSG:4326"

@dataclass
class AnalysisResult:
    """Container for analysis results."""
    labels: np.ndarray  # Cluster/region assignments
    metadata: AnalysisMetadata
    statistics: Dict[str, Any]
    spatial_output: Optional[Union[xr.Dataset, xr.DataArray]] = None
    additional_outputs: Optional[Dict[str, Any]] = None

class IAnalyzer(ABC):
    """
    Pure analyzer interface - defines the contract for spatial analysis methods.
    
    This interface contains NO implementations, only abstract method definitions.
    Concrete implementations should be in infrastructure or domain layers.
    """
    
    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Get default parameters for this analysis method.
        
        Returns:
            Dictionary of default parameter values
        """
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate analysis parameters.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        pass
    
    @abstractmethod
    def analyze(self, 
                data: Union[xr.Dataset, xr.DataArray, np.ndarray],
                **kwargs) -> AnalysisResult:
        """
        Perform the spatial analysis.
        
        Args:
            data: Input data for analysis
            **kwargs: Analysis-specific parameters
            
        Returns:
            AnalysisResult containing labels, metadata, and statistics
        """
        pass
    
    @abstractmethod
    def validate_input_data(self, data: Union[xr.Dataset, xr.DataArray, np.ndarray]) -> Tuple[bool, List[str]]:
        """
        Validate input data.
        
        Args:
            data: Input data to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        pass
    
    @abstractmethod
    def save_results(self, result: AnalysisResult, output_name: str, 
                     output_dir: Path = None) -> Path:
        """
        Save analysis results to disk.
        
        Args:
            result: Analysis result to save
            output_name: Base name for output files
            output_dir: Directory to save results (optional)
            
        Returns:
            Path to the primary saved file
        """
        pass
    
    @abstractmethod
    def set_progress_callback(self, callback: Callable[[int, int, str], None]) -> None:
        """
        Set callback for progress updates.
        
        Args:
            callback: Function that accepts (current, total, message)
        """
        pass