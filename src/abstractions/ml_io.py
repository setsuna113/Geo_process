#!/usr/bin/env python3
"""
Machine Learning I/O utilities for EconML pipelines
====================================================

Proper separation of concerns:
1. Model objects (pickle) - for predictions/SHAP/sensitivity
2. JSON results - for statistics and reporting  
3. Efficient residual storage - for large matrices (Python + R interop)
4. Unified save interface - orchestrates all saves

Author: Refactored for clarity
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


def save_model_objects(models: Dict[str, Any], output_dir: Path, prefix: str = "model") -> Path:
    """
    Save EconML model objects for later use (predictions, SHAP, sensitivity).
    
    Args:
        models: Dictionary of model objects (e.g., {'dml': LinearDML, 'forest': CausalForestDML})
        output_dir: Directory to save in
        prefix: Filename prefix
        
    Returns:
        Path to saved pickle file
        
    Notes:
        - Uses pickle protocol 4 for better numpy support
        - Models must be fitted before saving
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / f"{prefix}_objects.pkl"
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(models, f, protocol=4)  # Protocol 4 for numpy arrays
        logger.info(f"Saved model objects to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save model objects: {e}")
        raise


def save_json_results(results: Dict[str, Any], output_dir: Path, prefix: str = "results") -> Path:
    """
    Save JSON-serializable results (coefficients, p-values, CIs, etc.).
    
    Args:
        results: Dictionary of results (must be JSON-serializable)
        output_dir: Directory to save in
        prefix: Filename prefix
        
    Returns:
        Path to saved JSON file
        
    Notes:
        - Handles numpy types conversion
        - Pretty-prints for readability
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / f"{prefix}_stats.json"
    
    # Convert numpy types for JSON
    def numpy_converter(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    try:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=numpy_converter)
        logger.info(f"Saved JSON results to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save JSON results: {e}")
        raise


def save_residual_matrix(
    residual_matrix: np.ndarray, 
    taxa_names: list,
    output_dir: Path, 
    prefix: str = "residuals"
) -> Path:
    """
    Save residual matrix as Arrow format for efficient Python/R interop.
    
    Args:
        residual_matrix: numpy array of shape (n_samples, n_taxa)
        taxa_names: List of column names
        output_dir: Directory to save in
        prefix: Filename prefix
        
    Returns:
        Path to saved Arrow file
        
    Notes:
        - Arrow format provides zero-copy reads (memory mapping)
        - Native support in both Python (pyarrow) and R (arrow package)
        - No decompression needed unlike Parquet
        - Preserves float64 precision
    """
    import pyarrow as pa
    import pyarrow.feather as feather
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame with proper column names
    df = pd.DataFrame(residual_matrix, columns=taxa_names)
    
    # Add site_id for reference
    df.insert(0, 'site_id', range(len(df)))
    
    # Save as Arrow/Feather format (IPC format)
    filepath = output_dir / f"{prefix}.arrow"
    feather.write_feather(df, filepath, compression='uncompressed')  # No compression for memory mapping
    logger.info(f"Saved residuals to {filepath} (Arrow format)")
    
    return filepath


def save_pipeline_results(
    mode: str,
    results: Dict[str, Any],
    output_dir: Union[str, Path],
    config: Dict[str, Any]
) -> Dict[str, Path]:
    """
    Unified save function for pipeline results.
    
    Args:
        mode: 'homogeneous' or 'heterogeneous'
        results: Complete results from fit_homogeneous() or fit_heterogeneous()
        output_dir: Base output directory (or read from config)
        config: Pipeline configuration
        
    Returns:
        Dictionary of saved file paths
        
    Structure:
        output_dir/
            homogeneous/  (or heterogeneous/)
                model_objects.pkl     # EconML models for predictions
                results_stats.json    # Coefficients, p-values, etc.
                residuals.arrow       # (homogeneous only) Residual matrix
    """
    # Get output directory from config if not provided
    if output_dir is None:
        output_dir = config['output_dir']  # Required - no fallback
    
    output_dir = Path(output_dir) / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = {}
    
    # Separate data by type
    json_data = {}
    model_data = {}
    residual_matrix = None
    taxa_names = None
    
    for key, value in results.items():
        if key == 'residual_matrix':
            residual_matrix = value
        elif key == 'taxa':
            taxa_names = value
        elif key == 'models':
            model_data = value
        elif isinstance(value, (dict, list, str, int, float, bool, type(None))):
            json_data[key] = value
        elif isinstance(value, np.ndarray):
            # Convert small arrays to lists for JSON
            if value.size < 1000:
                json_data[key] = value.tolist()
            else:
                # Large arrays go to model pickle
                model_data[key] = value
        else:
            # Unknown types go to pickle
            model_data[key] = value
    
    # 1. Save model objects (if any)
    if model_data:
        saved_paths['models'] = save_model_objects(model_data, output_dir)
    
    # 2. Save JSON results
    if json_data:
        saved_paths['stats'] = save_json_results(json_data, output_dir)
    
    # 3. Save residual matrix (homogeneous mode only)
    if mode == 'homogeneous' and residual_matrix is not None:
        if taxa_names is None:
            taxa_names = ['fungi_richness', 'plants_richness', 'vertebrates_richness']
        
        saved_paths['residuals'] = save_residual_matrix(
            residual_matrix, taxa_names, output_dir
        )
    
    # Log summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Saved {mode} results to {output_dir}")
    for name, path in saved_paths.items():
        logger.info(f"  {name}: {path.name}")
    logger.info(f"{'='*60}\n")
    
    return saved_paths


def load_pipeline_results(
    mode: str,
    input_dir: Union[str, Path]
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[np.ndarray]]:
    """
    Load saved pipeline results.
    
    Args:
        mode: 'homogeneous' or 'heterogeneous'
        input_dir: Base directory containing saved results
        
    Returns:
        Tuple of (models_dict, stats_dict, residual_matrix or None)
    """
    input_dir = Path(input_dir) / mode
    
    # Load model objects
    model_path = input_dir / "model_objects.pkl"
    if model_path.exists():
        with open(model_path, 'rb') as f:
            models = pickle.load(f)
    else:
        models = {}
        logger.warning(f"No model objects found at {model_path}")
    
    # Load JSON stats
    stats_path = input_dir / "results_stats.json"
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)
    else:
        stats = {}
        logger.warning(f"No stats found at {stats_path}")
    
    # Load residuals (if homogeneous)
    residuals = None
    if mode == 'homogeneous':
        import pyarrow.feather as feather
        arrow_path = input_dir / "residuals.arrow"
        
        if arrow_path.exists():
            df = feather.read_feather(arrow_path)
            residuals = df.drop('site_id', axis=1).values
        else:
            logger.warning("No residual matrix found")
    
    return models, stats, residuals