"""Adapter to make GeoSOMAnalyzer compatible with IAnalyzer interface."""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List, Callable

from src.abstractions.interfaces.analyzer import IAnalyzer, AnalysisResult
from .analyzer import GeoSOMAnalyzer


class SOMAnalyzer(IAnalyzer):
    """Adapter for GeoSOMAnalyzer to implement IAnalyzer interface.
    
    This adapter allows GeoSOMAnalyzer to work with the analyzer factory
    while maintaining compatibility with the existing pipeline.
    """
    
    def __init__(self, config: Any, db: Any):
        """Initialize SOM analyzer adapter.
        
        Args:
            config: Configuration object (used by factory)
            db: Database manager (used by factory, ignored here)
        """
        # Extract SOM-specific config if available
        som_config = None
        if hasattr(config, 'get'):
            som_config = config.get('spatial_analysis', {}).get('som', {})
        
        # Initialize the actual GeoSOM analyzer
        self._analyzer = GeoSOMAnalyzer(som_config)
        self._last_data_path = None
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for SOM analysis."""
        return self._analyzer.get_default_parameters()
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate analysis parameters."""
        return self._analyzer.validate_parameters(parameters)
    
    def analyze(self, 
                data: Union[str, xr.Dataset, xr.DataArray, np.ndarray],
                **kwargs) -> AnalysisResult:
        """Perform SOM analysis.
        
        Args:
            data: Input data - can be a file path (string) or data array
            **kwargs: Analysis parameters
            
        Returns:
            AnalysisResult from the analysis
        """
        # Handle different input types
        if isinstance(data, str):
            # Direct file path - what GeoSOMAnalyzer expects
            # Set default coordinate columns if not provided
            if 'coordinate_columns' not in kwargs:
                kwargs['coordinate_columns'] = ['x', 'y']
            return self._analyzer.analyze(data, **kwargs)
        else:
            # Need to handle array data - not implemented yet
            raise NotImplementedError(
                "SOMAnalyzer currently only supports file path inputs. "
                "Array inputs will be supported in future versions."
            )
    
    def validate_input_data(self, data: Union[str, xr.Dataset, xr.DataArray, np.ndarray]) -> Tuple[bool, List[str]]:
        """Validate input data."""
        if isinstance(data, str):
            return self._analyzer.validate_input_data(data)
        else:
            return False, ["SOMAnalyzer currently only supports file path inputs"]
    
    def save_results(self, result: AnalysisResult, output_name: str, 
                     output_dir: Path = None) -> Path:
        """Save analysis results to disk."""
        if output_dir is None:
            output_dir = Path("./results")
        
        output_path = str(output_dir)
        self._analyzer.save_results(result, output_path)
        return output_dir / f"{output_name}_metadata.json"
    
    def set_progress_callback(self, callback: Callable[[int, int, str], None]) -> None:
        """Set callback for progress updates.
        
        Args:
            callback: Function that accepts (current, total, message)
        """
        # Adapt the callback to match what GeoSOMAnalyzer expects
        def adapted_callback(message: str, progress: float):
            # Convert progress (0-1) to current/total
            total = 100
            current = int(progress * total)
            callback(current, total, message)
        
        self._analyzer.set_progress_callback(adapted_callback)