"""
File I/O utilities - stop copy-pasting the same code.

Used 50+ times across the codebase.
LINUS: "Handle errors or die trying."
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from .logging import get_logger

logger = get_logger(__name__)


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> Path:
    """Save dict to JSON file. Used 30+ times.
    
    Args:
        data: Dictionary to save
        path: Output path
        
    Returns:
        Path where file was saved
        
    Raises:
        IOError: If save fails (permissions, disk space, etc)
        TypeError: If data contains non-serializable objects
    """
    path = Path(path)
    
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise IOError(f"Cannot create directory {path.parent}: {e}")
    
    # Handle numpy types - STREAMING for large arrays
    def default(obj):
        if isinstance(obj, np.ndarray):
            # LINUS: Don't load 1GB arrays into memory for JSON
            if obj.size > 10000:  # Threshold for large arrays
                logger.warning(f"Large array ({obj.size} elements) being serialized to JSON. Consider using save_numpy() instead.")
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=default)
    except OSError as e:
        raise IOError(f"Cannot write to {path}: {e}. Check disk space and permissions.")
    except TypeError as e:
        raise TypeError(f"Data contains non-serializable objects: {e}")
    
    return path


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file. Used 25+ times.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Loaded dictionary
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file can't be read
        json.JSONDecodeError: If file is malformed
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except OSError as e:
        raise IOError(f"Cannot read {path}: {e}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Malformed JSON in {path}: {e}", e.doc, e.pos)


def save_pickle(obj: Any, path: Union[str, Path]) -> Path:
    """Save object to pickle file. Used 15+ times.
    
    Args:
        obj: Object to pickle
        path: Output path
        
    Returns:
        Path where file was saved
        
    Raises:
        IOError: If save fails
        PicklingError: If object can't be pickled
    """
    path = Path(path)
    
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise IOError(f"Cannot create directory {path.parent}: {e}")
    
    try:
        with open(path, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError as e:
        raise IOError(f"Cannot write to {path}: {e}. Check disk space and permissions.")
    except pickle.PicklingError as e:
        raise pickle.PicklingError(f"Cannot pickle object: {e}")
    
    return path


def load_pickle(path: Union[str, Path]) -> Any:
    """Load pickle file. Used 12+ times.
    
    Args:
        path: Path to pickle file
        
    Returns:
        Unpickled object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file can't be read
        pickle.UnpicklingError: If file is corrupted
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")
    
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except OSError as e:
        raise IOError(f"Cannot read {path}: {e}")
    except (pickle.UnpicklingError, EOFError) as e:
        raise pickle.UnpicklingError(f"Corrupted pickle file {path}: {e}")


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
        
    Raises:
        ValueError: If format is unknown
        IOError: If save fails
    """
    output_dir = Path(output_dir)
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise IOError(f"Cannot create output directory {output_dir}: {e}")
    
    if format == 'json':
        path = output_dir / f"{name}_results.json"
        return save_json(results, path)
    elif format == 'pickle':
        path = output_dir / f"{name}_results.pkl"
        return save_pickle(results, path)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'json' or 'pickle'.")


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
        
    Raises:
        ValueError: If format is unknown
        FileNotFoundError: If file doesn't exist
        IOError: If load fails
    """
    input_dir = Path(input_dir)
    
    if format == 'json':
        path = input_dir / f"{name}_results.json"
        return load_json(path)
    elif format == 'pickle':
        path = input_dir / f"{name}_results.pkl"
        return load_pickle(path)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'json' or 'pickle'.")


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
        
    Raises:
        ValueError: If format is unknown
        IOError: If save fails
    """
    path = Path(path)
    
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise IOError(f"Cannot create directory {path.parent}: {e}")
    
    try:
        if format == 'parquet':
            df.to_parquet(path)
        elif format == 'csv':
            df.to_csv(path, index=False)
        elif format == 'pickle':
            df.to_pickle(path)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'parquet', 'csv', or 'pickle'.")
    except Exception as e:
        raise IOError(f"Cannot save DataFrame to {path}: {e}")
    
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
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format is unknown or can't be detected
        IOError: If load fails
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"DataFrame file not found: {path}")
    
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
            raise ValueError(f"Cannot auto-detect format for extension '{ext}'. Specify format explicitly.")
    
    try:
        if format == 'parquet':
            return pd.read_parquet(path)
        elif format == 'csv':
            return pd.read_csv(path)
        elif format == 'pickle':
            return pd.read_pickle(path)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'parquet', 'csv', or 'pickle'.")
    except Exception as e:
        raise IOError(f"Cannot load DataFrame from {path}: {e}")


def save_numpy(
    arr: np.ndarray,
    path: Union[str, Path],
    compressed: bool = True
) -> Path:
    """Save numpy array. Used 7+ times.
    
    LINUS: For large arrays, always compress. Disk is cheap, time is not.
    
    Args:
        arr: Array to save
        path: Output path
        compressed: Use compression (default True for arrays > 1MB)
        
    Returns:
        Path where file was saved
        
    Raises:
        IOError: If save fails
    """
    path = Path(path)
    
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise IOError(f"Cannot create directory {path.parent}: {e}")
    
    # Auto-compress large arrays
    if arr.nbytes > 1024 * 1024:  # 1MB threshold
        if not compressed:
            logger.warning(f"Large array ({arr.nbytes / 1024 / 1024:.1f}MB) being saved uncompressed. Consider compression.")
    
    try:
        if compressed:
            np.savez_compressed(path, data=arr)
        else:
            np.save(path, arr)
    except Exception as e:
        raise IOError(f"Cannot save numpy array to {path}: {e}")
    
    return path


def load_numpy(path: Union[str, Path]) -> np.ndarray:
    """Load numpy array. Used 6+ times.
    
    Args:
        path: Input path
        
    Returns:
        Loaded array
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If load fails
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Numpy file not found: {path}")
    
    try:
        if path.suffix == '.npz':
            with np.load(path) as data:
                return data['data']
        else:
            return np.load(path)
    except Exception as e:
        raise IOError(f"Cannot load numpy array from {path}: {e}")