"""Adapter to make KMeansAnalyzer compatible with IAnalyzer interface."""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List

from src.abstractions.interfaces.analyzer import IAnalyzer, AnalysisResult
from .analyzer import KMeansAnalyzer


class KMeansAnalyzerAdapter(IAnalyzer):
    """Adapter for KMeansAnalyzer to implement IAnalyzer interface.
    
    This adapter allows KMeansAnalyzer to work with the analyzer factory
    while maintaining compatibility with the existing pipeline.
    """
    
    def __init__(self, config: Any, db: Any):
        """Initialize k-means analyzer adapter.
        
        Args:
            config: Configuration object (used by factory)
            db: Database manager (used by factory, ignored here)
        """
        # Initialize the actual k-means analyzer
        self._analyzer = KMeansAnalyzer()
        self._config = config
        self._db = db
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for k-means analysis."""
        return {
            'n_clusters': 20,
            'determine_k': False,
            'k_range': [5, 30],
            'save_results': True,
            'output_dir': './outputs/kmeans_results'
        }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate analysis parameters."""
        errors = []
        
        # Validate n_clusters
        if 'n_clusters' in parameters:
            n_clusters = parameters['n_clusters']
            if not isinstance(n_clusters, int) or n_clusters < 2:
                errors.append("n_clusters must be an integer >= 2")
        
        # Validate k_range if determine_k is True
        if parameters.get('determine_k', False):
            k_range = parameters.get('k_range', [5, 30])
            if not isinstance(k_range, (list, tuple)) or len(k_range) != 2:
                errors.append("k_range must be a list or tuple of 2 integers")
            elif not all(isinstance(k, int) for k in k_range):
                errors.append("k_range values must be integers")
            elif k_range[0] >= k_range[1]:
                errors.append("k_range[0] must be less than k_range[1]")
        
        return len(errors) == 0, errors
    
    def analyze(self, 
                data: Union[str, xr.Dataset, xr.DataArray, np.ndarray],
                **kwargs) -> AnalysisResult:
        """Perform k-means analysis.
        
        Args:
            data: Input data - can be a file path (string) or data array
            **kwargs: Analysis parameters
            
        Returns:
            AnalysisResult from the analysis
        """
        # Handle different input types
        if isinstance(data, str):
            # Direct file path - what KMeansAnalyzer expects
            return self._analyzer.analyze(data, **kwargs)
        else:
            # Need to handle array data - not implemented yet
            raise NotImplementedError(
                "KMeansAnalyzer currently only supports file path inputs. "
                "Array inputs will be supported in future versions."
            )
    
    def validate_input_data(self, data: Union[str, xr.Dataset, xr.DataArray, np.ndarray]) -> Tuple[bool, List[str]]:
        """Validate input data."""
        errors = []
        
        if isinstance(data, str):
            # Check if file exists
            path = Path(data)
            if not path.exists():
                errors.append(f"Input file does not exist: {data}")
            elif not path.suffix.lower() in ['.parquet', '.csv']:
                errors.append(f"Unsupported file format: {path.suffix}. Use .parquet or .csv")
        else:
            errors.append("KMeansAnalyzer currently only supports file path inputs")
        
        return len(errors) == 0, errors
    
    def save_results(self, result: AnalysisResult, output_name: str, 
                     output_dir: Path = None) -> Path:
        """Save analysis results to disk.
        
        Args:
            result: Analysis result to save
            output_name: Base name for output files
            output_dir: Directory to save results
            
        Returns:
            Path to saved results
        """
        if output_dir is None:
            output_dir = Path("./results/kmeans")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # The analyzer already saves results internally
        # This is just for interface compatibility
        return output_dir
    
    def set_progress_callback(self, callback: Any):
        """Set progress callback function.
        
        Args:
            callback: Progress callback function
        """
        if hasattr(self._analyzer, 'set_progress_callback'):
            self._analyzer.set_progress_callback(callback)