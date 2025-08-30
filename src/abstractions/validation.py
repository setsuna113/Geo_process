"""
Validation utilities - stop copy-pasting validation code.

Used 40+ times across the codebase.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd


def validate_parameters(
    params: Dict[str, Any],
    required: List[str],
    optional: Optional[Dict[str, Any]] = None
) -> Tuple[bool, List[str]]:
    """Validate parameters dict. Used 40+ times.
    
    Args:
        params: Parameters to validate
        required: List of required parameter names
        optional: Dict of optional params with defaults
        
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required parameters
    for key in required:
        if key not in params:
            errors.append(f"Missing required parameter: {key}")
        elif params[key] is None:
            errors.append(f"Parameter '{key}' cannot be None")
    
    # Apply defaults for optional parameters
    if optional:
        for key, default in optional.items():
            if key not in params:
                params[key] = default
    
    return len(errors) == 0, errors


def validate_input_data(
    data: Any,
    expected_type: Optional[type] = None,
    min_shape: Optional[Tuple] = None,
    max_shape: Optional[Tuple] = None,
    allow_empty: bool = False
) -> Tuple[bool, List[str]]:
    """Validate input data. Used 35+ times.
    
    Args:
        data: Data to validate
        expected_type: Expected data type
        min_shape: Minimum shape for arrays/dataframes
        max_shape: Maximum shape for arrays/dataframes  
        allow_empty: Whether to allow empty data
        
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # Check if data exists
    if data is None:
        errors.append("Data is None")
        return False, errors
    
    # Type validation
    if expected_type and not isinstance(data, expected_type):
        errors.append(f"Expected type {expected_type.__name__}, got {type(data).__name__}")
        return False, errors
    
    # Shape validation for arrays
    if isinstance(data, np.ndarray):
        if not allow_empty and data.size == 0:
            errors.append("Data array is empty")
        
        if min_shape:
            for i, min_dim in enumerate(min_shape):
                if min_dim and i < len(data.shape) and data.shape[i] < min_dim:
                    errors.append(f"Dimension {i} size {data.shape[i]} < minimum {min_dim}")
        
        if max_shape:
            for i, max_dim in enumerate(max_shape):
                if max_dim and i < len(data.shape) and data.shape[i] > max_dim:
                    errors.append(f"Dimension {i} size {data.shape[i]} > maximum {max_dim}")
    
    # Shape validation for DataFrames
    elif isinstance(data, pd.DataFrame):
        if not allow_empty and data.empty:
            errors.append("DataFrame is empty")
        
        if min_shape:
            if min_shape[0] and len(data) < min_shape[0]:
                errors.append(f"DataFrame has {len(data)} rows, minimum is {min_shape[0]}")
            if len(min_shape) > 1 and min_shape[1] and len(data.columns) < min_shape[1]:
                errors.append(f"DataFrame has {len(data.columns)} columns, minimum is {min_shape[1]}")
    
    return len(errors) == 0, errors


def validate_numeric_range(
    value: Union[int, float],
    name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    allow_none: bool = False
) -> Tuple[bool, Optional[str]]:
    """Validate numeric value is in range. Used 20+ times.
    
    Args:
        value: Value to check
        name: Parameter name for error messages
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        allow_none: Whether None is valid
        
    Returns:
        (is_valid, error_message)
    """
    if value is None:
        if allow_none:
            return True, None
        else:
            return False, f"{name} cannot be None"
    
    if not isinstance(value, (int, float)):
        return False, f"{name} must be numeric, got {type(value).__name__}"
    
    if min_val is not None and value < min_val:
        return False, f"{name} = {value} is below minimum {min_val}"
    
    if max_val is not None and value > max_val:
        return False, f"{name} = {value} is above maximum {max_val}"
    
    return True, None


def validate_path(
    path: Union[str, Path],
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    create_parent: bool = False
) -> Tuple[bool, Optional[str]]:
    """Validate file/directory path. Used 15+ times.
    
    Args:
        path: Path to validate
        must_exist: Path must already exist
        must_be_file: Path must be a file (not directory)
        must_be_dir: Path must be a directory (not file)
        create_parent: Create parent directory if missing
        
    Returns:
        (is_valid, error_message)
    """
    try:
        path = Path(path)
    except Exception as e:
        return False, f"Invalid path: {e}"
    
    if must_exist and not path.exists():
        return False, f"Path does not exist: {path}"
    
    if path.exists():
        if must_be_file and not path.is_file():
            return False, f"Path is not a file: {path}"
        if must_be_dir and not path.is_dir():
            return False, f"Path is not a directory: {path}"
    
    if create_parent and not path.parent.exists():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return False, f"Could not create parent directory: {e}"
    
    return True, None


def validate_config(
    config: Dict[str, Any],
    schema: Dict[str, Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    """Validate config against schema. Used 12+ times.
    
    Args:
        config: Configuration dictionary
        schema: Schema definition with keys as param names and values as:
                {'type': type, 'required': bool, 'min': val, 'max': val}
                
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    for param_name, rules in schema.items():
        is_required = rules.get('required', False)
        expected_type = rules.get('type')
        min_val = rules.get('min')
        max_val = rules.get('max')
        
        # Check existence
        if param_name not in config:
            if is_required:
                errors.append(f"Missing required config parameter: {param_name}")
            continue
        
        value = config[param_name]
        
        # Check type
        if expected_type and not isinstance(value, expected_type):
            errors.append(f"Config {param_name}: expected {expected_type.__name__}, got {type(value).__name__}")
            continue
        
        # Check numeric range
        if isinstance(value, (int, float)):
            valid, error = validate_numeric_range(value, param_name, min_val, max_val)
            if not valid:
                errors.append(error)
    
    return len(errors) == 0, errors


def check_data_compatibility(
    datasets: List[Union[np.ndarray, pd.DataFrame]],
    require_same_shape: bool = False,
    require_same_columns: bool = False
) -> Tuple[bool, List[str]]:
    """Check if multiple datasets are compatible. Used 8+ times.
    
    Args:
        datasets: List of datasets to check
        require_same_shape: All must have same shape
        require_same_columns: All DataFrames must have same columns
        
    Returns:
        (is_compatible, list_of_errors)
    """
    errors = []
    
    if not datasets:
        errors.append("No datasets provided")
        return False, errors
    
    # Check same types
    types = [type(d) for d in datasets]
    if len(set(types)) > 1:
        errors.append(f"Mixed dataset types: {[t.__name__ for t in types]}")
    
    # Check shapes for arrays
    if all(isinstance(d, np.ndarray) for d in datasets):
        shapes = [d.shape for d in datasets]
        if require_same_shape and len(set(shapes)) > 1:
            errors.append(f"Different array shapes: {shapes}")
    
    # Check columns for DataFrames
    if all(isinstance(d, pd.DataFrame) for d in datasets):
        if require_same_columns:
            column_sets = [set(d.columns) for d in datasets]
            if len(set(map(frozenset, column_sets))) > 1:
                errors.append("DataFrames have different columns")
    
    return len(errors) == 0, errors