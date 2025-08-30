from src.abstractions import IAnalyzer, AnalysisResult, validate_parameters
from .hierarchical_analyzer import KMeansHierarchicalAnalyzer
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
import numpy as np
import xarray as xr




class KMeansHierarchicalAnalyzerAdapter(IAnalyzer):
    """Adapter for KMeansHierarchicalAnalyzer to implement IAnalyzer interface.
    
    This adapter allows KMeansHierarchicalAnalyzer to work with the analyzer factory
    while maintaining compatibility with the existing pipeline.
    """
    
    def __init__(self, config: Any):
        """Initialize k-means hierarchical analyzer adapter.
        
        Args:
            config: Configuration object (used by factory)
        """
        # Initialize the actual hierarchical analyzer
        self._analyzer = KMeansHierarchicalAnalyzer()
        self._config = config
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for k-means hierarchical analysis."""
        return self._analyzer.get_default_parameters()
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate analysis parameters for k-means hierarchical clustering."""
        errors = []
        
        # Validate n_pre_clusters
        if 'n_pre_clusters' in parameters:
            n = parameters['n_pre_clusters']
            if not isinstance(n, int) or n < 10:
                errors.append("n_pre_clusters must be an integer >= 10")
        
        # Validate final_k_range
        if 'final_k_range' in parameters:
            k_range = parameters['final_k_range']
            if not isinstance(k_range, (list, tuple)) or len(k_range) != 2:
                errors.append("final_k_range must be a list or tuple of 2 integers")
            elif not all(isinstance(k, int) for k in k_range):
                errors.append("final_k_range values must be integers")
            elif k_range[0] >= k_range[1]:
                errors.append("final_k_range[0] must be less than final_k_range[1]")
        
        # Validate linkage_method
        if 'linkage_method' in parameters:
            valid_methods = ['ward', 'complete', 'average', 'single']
            if parameters['linkage_method'] not in valid_methods:
                errors.append(f"linkage_method must be one of {valid_methods}")
        
        # Validate clustering_approach
        if 'clustering_approach' in parameters:
            valid_approaches = ['geometric', 'ecological']
            if parameters['clustering_approach'] not in valid_approaches:
                errors.append(f"clustering_approach must be one of {valid_approaches}")
        
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
        # Delegate to the analyzer's save_results method
        return self._analyzer.save_results(result, output_name, output_dir)
    
    def set_progress_callback(self, callback: Any):
        """Set progress callback function.
        
        Args:
            callback: Progress callback function
        """
        if hasattr(self._analyzer, 'set_progress_callback'):
            self._analyzer.set_progress_callback(callback)
