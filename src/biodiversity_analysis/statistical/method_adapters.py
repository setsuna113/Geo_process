"""
Method adapters - PROPER type conversion, not bullshit forwarding.

"Bad programmers worry about the code. Good programmers worry about data structures."
Each method gets EXACTLY the data type it needs. No guessing.
"""

from pathlib import Path
from typing import Dict, Any, Union
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# Data type converters - ONE place for conversion logic
def to_dataframe(data: Union[pd.DataFrame, Dict[str, np.ndarray]]) -> pd.DataFrame:
    """Convert any data to DataFrame for statistical methods."""
    if isinstance(data, pd.DataFrame):
        return data
    
    if isinstance(data, dict):
        # Dict of arrays -> DataFrame
        # Assume all arrays have same shape
        first_array = next(iter(data.values()))
        height, width = first_array.shape
        
        # Create coordinates
        y_coords, x_coords = np.meshgrid(
            np.arange(height), np.arange(width), 
            indexing='ij'
        )
        
        # Flatten everything
        df_data = {
            'x': x_coords.ravel(),
            'y': y_coords.ravel()
        }
        
        for name, array in data.items():
            df_data[name] = array.ravel()
        
        df = pd.DataFrame(df_data)
        
        # Remove invalid pixels
        mask = pd.Series(True, index=df.index)
        for col in data.keys():
            mask &= ~df[col].isna()
        
        return df[mask].reset_index(drop=True)
    
    raise TypeError(f"Cannot convert {type(data)} to DataFrame")


def to_arrays(data: Union[pd.DataFrame, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Convert any data to spatial arrays for map methods."""
    if isinstance(data, dict):
        # Already in the right format
        return data
    
    if isinstance(data, pd.DataFrame):
        # DataFrame -> spatial arrays (would need shape info)
        # This is lossy - better to load from source
        raise NotImplementedError(
            "DataFrame to spatial array conversion not supported. "
            "Load spatial data directly from rasters."
        )
    
    raise TypeError(f"Cannot convert {type(data)} to spatial arrays")


# Method adapters - CLEAN signatures, PROPER conversion
def analyze_correlation(data: Union[pd.DataFrame, Dict[str, np.ndarray]], 
                       output_dir: Path, **config) -> Dict[str, Any]:
    """Correlation needs DataFrame - convert if necessary."""
    from .correlation_analyzer import analyze_correlation as impl
    
    # Convert to DataFrame
    df = to_dataframe(data)
    
    # Now call with CLEAN signature
    return impl(df, output_dir, **config)


def analyze_coverage(data: Dict[str, np.ndarray], 
                    output_dir: Path, **config) -> Dict[str, Any]:
    """Coverage needs spatial arrays."""
    from .coverage import analyze_coverage as impl
    
    # Already correct type from loader
    return impl(data, output_dir, **config)


def analyze_overlap(data: Dict[str, np.ndarray], 
                   output_dir: Path, **config) -> Dict[str, Any]:
    """Overlap needs spatial arrays."""
    from .overlap import analyze_overlap as impl
    
    # Already correct type from loader
    return impl(data, output_dir, **config)


def analyze_latitudinal(data: Dict[str, np.ndarray], 
                       output_dir: Path, **config) -> Dict[str, Any]:
    """Latitudinal needs spatial arrays."""
    from .latitudinal import analyze_latitudinal as impl
    
    # Already correct type from loader
    return impl(data, output_dir, **config)


def analyze_global_maps(data: Dict[str, np.ndarray], 
                       output_dir: Path, **config) -> Dict[str, Any]:
    """Global maps need spatial arrays."""
    from .global_map import analyze_global_maps as impl
    
    # Already correct type from loader
    return impl(data, output_dir, **config)


def analyze_trust_maps(data: Dict[str, np.ndarray], 
                      output_dir: Path, **config) -> Dict[str, Any]:
    """Trust maps need spatial arrays."""
    from .trust_map import analyze_trust_maps as impl
    
    # Already correct type from loader
    return impl(data, output_dir, **config)


def analyze_biome_analysis(data: pd.DataFrame, 
                          output_dir: Path, **config) -> Dict[str, Any]:
    """Biome analysis needs DataFrame with climate data."""
    from .biome_analysis import analyze_by_biome as impl
    
    # Already correct type from loader
    return impl(data, output_dir, **config)


def analyze_partial_correlation(data: pd.DataFrame, 
                               output_dir: Path, **config) -> Dict[str, Any]:
    """Partial correlation needs DataFrame."""
    from .partial_correlation import analyze_partial_correlations as impl
    
    # Extract config properly
    target_pairs = config.get('target_pairs', [
        ('animal_richness', 'am_fungi_richness'),
        ('animal_richness', 'ecm_fungi_richness'),
        ('plant_richness', 'am_fungi_richness'),
        ('plant_richness', 'ecm_fungi_richness'),
        ('am_fungi_richness', 'ecm_fungi_richness')
    ])
    
    control_vars = config.get('control_vars', 
                              ['bio01', 'bio04', 'bio12', 'bio15', 'dem'])
    
    return impl(data, output_dir, target_pairs, control_vars, **config)


# Registry - maps method names to adapters
ADAPTER_REGISTRY = {
    'correlation': analyze_correlation,
    'coverage': analyze_coverage,
    'overlap': analyze_overlap,
    'latitudinal': analyze_latitudinal,
    'global_maps': analyze_global_maps,
    'trust_maps': analyze_trust_maps,
    'biome_analysis': analyze_biome_analysis,
    'partial_correlation': analyze_partial_correlation
}

# Method data requirements - THE SINGLE SOURCE OF TRUTH
METHOD_DATA_TYPE = {
    'correlation': 'tabular',
    'coverage': 'spatial',
    'overlap': 'spatial',
    'latitudinal': 'spatial',
    'global_maps': 'spatial',
    'trust_maps': 'spatial',
    'biome_analysis': 'tabular',
    'partial_correlation': 'tabular'
}


def call_method(method: str, data: Any, output_dir: Path, 
                config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Universal method caller with PROPER type conversion.
    
    Args:
        method: Method name
        data: Data from unified loader (DataFrame or Dict[str, ndarray])
        output_dir: Output directory
        config: Method config
        
    Returns:
        Method results
    """
    if method not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown method: {method}. Register it in ADAPTER_REGISTRY.")
    
    adapter = ADAPTER_REGISTRY[method]
    return adapter(data, output_dir, **config)