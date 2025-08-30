"""
File I/O utilities - stop copy-pasting the same code.

Used 50+ times across the codebase.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> Path:
    """Save dict to JSON file. Used 30+ times.
    
    Args:
        data: Dictionary to save
        path: Output path
        
    Returns:
        Path where file was saved
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle numpy types
    def default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=default)
    
    return path


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file. Used 25+ times.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(path, 'r') as f:
        return json.load(f)


def save_pickle(obj: Any, path: Union[str, Path]) -> Path:
    """Save object to pickle file. Used 15+ times.
    
    Args:
        obj: Object to pickle
        path: Output path
        
    Returns:
        Path where file was saved
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    
    return path


def load_pickle(path: Union[str, Path]) -> Any:
    """Load pickle file. Used 12+ times.
    
    Args:
        path: Path to pickle file
        
    Returns:
        Unpickled object
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_results(
    results: Dict[str, Any], 
    output_dir: Union[str, Path],
    name: str,
    format: str = 'json'
) -> Path:
    """Save analysis results. Used 20+ times.
    
    Args:
        results: Results dictionary
        output_dir: Output directory
        name: Base name for file
        format: 'json' or 'pickle'
        
    Returns:
        Path where results were saved
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        path = output_dir / f"{name}_results.json"
        return save_json(results, path)
    elif format == 'pickle':
        path = output_dir / f"{name}_results.pkl"
        return save_pickle(results, path)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_results(
    input_dir: Union[str, Path],
    name: str,
    format: str = 'json'
) -> Dict[str, Any]:
    """Load analysis results. Used 18+ times.
    
    Args:
        input_dir: Input directory
        name: Base name for file
        format: 'json' or 'pickle'
        
    Returns:
        Loaded results
    """
    input_dir = Path(input_dir)
    
    if format == 'json':
        path = input_dir / f"{name}_results.json"
        return load_json(path)
    elif format == 'pickle':
        path = input_dir / f"{name}_results.pkl"
        return load_pickle(path)
    else:
        raise ValueError(f"Unknown format: {format}")


def save_dataframe(
    df: pd.DataFrame,
    path: Union[str, Path],
    format: str = 'parquet'
) -> Path:
    """Save DataFrame. Used 10+ times.
    
    Args:
        df: DataFrame to save
        path: Output path
        format: 'parquet', 'csv', or 'pickle'
        
    Returns:
        Path where file was saved
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'parquet':
        df.to_parquet(path)
    elif format == 'csv':
        df.to_csv(path, index=False)
    elif format == 'pickle':
        df.to_pickle(path)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return path


def load_dataframe(
    path: Union[str, Path],
    format: Optional[str] = None
) -> pd.DataFrame:
    """Load DataFrame. Used 8+ times.
    
    Args:
        path: Input path
        format: Format or auto-detect from extension
        
    Returns:
        Loaded DataFrame
    """
    path = Path(path)
    
    if format is None:
        # Auto-detect from extension
        ext = path.suffix.lower()
        if ext == '.parquet':
            format = 'parquet'
        elif ext in ['.csv', '.txt']:
            format = 'csv'
        elif ext in ['.pkl', '.pickle']:
            format = 'pickle'
        else:
            raise ValueError(f"Cannot auto-detect format for {ext}")
    
    if format == 'parquet':
        return pd.read_parquet(path)
    elif format == 'csv':
        return pd.read_csv(path)
    elif format == 'pickle':
        return pd.read_pickle(path)
    else:
        raise ValueError(f"Unknown format: {format}")


def save_numpy(
    arr: np.ndarray,
    path: Union[str, Path],
    compressed: bool = True
) -> Path:
    """Save numpy array. Used 7+ times.
    
    Args:
        arr: Array to save
        path: Output path
        compressed: Use compression
        
    Returns:
        Path where file was saved
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if compressed:
        np.savez_compressed(path, data=arr)
    else:
        np.save(path, arr)
    
    return path


def load_numpy(path: Union[str, Path]) -> np.ndarray:
    """Load numpy array. Used 6+ times.
    
    Args:
        path: Input path
        
    Returns:
        Loaded array
    """
    path = Path(path)
    
    if path.suffix == '.npz':
        return np.load(path)['data']
    else:
        return np.load(path)