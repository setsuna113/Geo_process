from src.abstractions import get_logger
"""
Common data utilities for statistical analysis.

Following Linus's principle: "Bad programmers worry about the code. 
Good programmers worry about data structures."
"""

import json
from src.abstractions import save_json
import logging
from pathlib import Path
from typing import Dict, Any, Union, List
import numpy as np
import pandas as pd

logger = get_logger(__name__)


def load_data(source: Union[str, Dict[str, str], pd.DataFrame, np.ndarray], 
              parquet_loader=None, 
              raster_loader=None) -> Union[pd.DataFrame, np.ndarray]:
    """Load data from various sources.
    
    Args:
        source: Data source (path, dict of paths, or pre-loaded data)
        parquet_loader: ParquetLoader instance (created if needed)
        raster_loader: RasterLoader instance (created if needed)
        
    Returns:
        Loaded data as DataFrame or ndarray
    """
    # Already loaded
    if isinstance(source, (pd.DataFrame, np.ndarray)):
        return source
    
    # Parquet file
    if isinstance(source, str) and source.endswith('.parquet'):
        if parquet_loader is None:
            from src.biodiversity_analysis.shared.data.parquet_loader import ParquetLoader
            parquet_loader = ParquetLoader()
        biodiv_data = parquet_loader.load(source)
        return pd.DataFrame(biodiv_data.features)
    
    # Raster files (dict)
    if isinstance(source, dict):
        if raster_loader is None:
            from src.biodiversity_analysis.shared.data.raster_loader import RasterLoader
            raster_loader = RasterLoader()
        return raster_loader.load(source, align=True)
    
    # Single raster file
    if isinstance(source, str) and source.endswith('.tif'):
        if raster_loader is None:
            from src.biodiversity_analysis.shared.data.raster_loader import RasterLoader
            raster_loader = RasterLoader()
        return raster_loader.load_single(source)
    
    raise ValueError(f"Unsupported data source: {type(source)}")


def save_results(results: Dict[str, Any], output_path: Union[str, Path]) -> Path:
    """Save results to JSON file.
    
    Args:
        results: Results dictionary
        output_path: Output file path
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types for JSON
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    with open(output_path, 'w') as f:
        json.dump(convert_numpy(results), f, indent=2)
    
    logger.info(f"Saved results to {output_path}")
    return output_path


def save_figure(fig, output_path: Union[str, Path], dpi: int = 150) -> Path:
    """Save matplotlib figure.
    
    Args:
        fig: Matplotlib figure
        output_path: Output file path
        dpi: Resolution for saving
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"Saved figure to {output_path}")
    return output_path


def get_enabled_methods(config: Dict[str, Any]) -> List[str]:
    """Get list of enabled methods from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of enabled method names
    """
    methods = config.get('methods', {})
    return [name for name, cfg in methods.items() if cfg.get('enabled', False)]


def get_sample_factor(config: Dict[str, Any], method: str = None) -> int:
    """Get sample factor for a method.
    
    Args:
        config: Configuration dictionary
        method: Method name (optional)
        
    Returns:
        Sample factor value
    """
    if method and 'methods' in config:
        method_config = config['methods'].get(method, {})
        return method_config.get('sample_factor', 1)
    
    # Return most common sample factor as default
    factors = []
    for method_config in config.get('methods', {}).values():
        if 'sample_factor' in method_config:
            factors.append(method_config['sample_factor'])
    
    return max(set(factors), key=factors.count) if factors else 1