# src/utils/json_utils.py
"""JSON utilities for handling special types."""

import json
import numpy as np
from datetime import datetime, date
from pathlib import Path
from decimal import Decimal
from functools import lru_cache


class ExtendedJSONEncoder(json.JSONEncoder):
    """Optimized JSON encoder that handles additional types.
    
    Performance optimizations:
    - Type checking order optimized for most common types first
    - Cached type conversions where possible
    - Avoid numpy array serialization for large arrays
    """
    
    def __init__(self, *args, max_array_size=1000, **kwargs):
        """Initialize encoder with array size limit."""
        super().__init__(*args, **kwargs)
        self.max_array_size = max_array_size
    
    def default(self, obj):
        """Convert non-serializable objects to serializable format."""
        # Most common types first for performance
        obj_type = type(obj)
        
        # Handle numpy scalar types (most common in scientific computing)
        if obj_type in (np.float64, np.float32):
            return None if np.isnan(obj) else float(obj)
        if obj_type in (np.int64, np.int32):
            return int(obj)
        if obj_type == np.bool_:
            return bool(obj)
        
        # Handle numpy arrays with size limit
        if isinstance(obj, np.ndarray):
            if obj.size > self.max_array_size:
                # For large arrays, return summary instead
                return {
                    "_type": "numpy_array_summary",
                    "shape": obj.shape,
                    "dtype": str(obj.dtype),
                    "size": obj.size,
                    "min": float(np.nanmin(obj)) if obj.size > 0 else None,
                    "max": float(np.nanmax(obj)) if obj.size > 0 else None,
                    "mean": float(np.nanmean(obj)) if obj.size > 0 else None
                }
            return obj.tolist()
        
        # Handle datetime types
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        
        # Handle Path objects
        if isinstance(obj, Path):
            return str(obj)
        
        # Handle Decimal
        if isinstance(obj, Decimal):
            return float(obj)
        
        # Handle generic numpy types
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return None if np.isnan(obj) else float(obj)
        
        # Let the base class default method raise the TypeError
        return super().default(obj)


def clean_for_json(data):
    """Clean data structure for JSON serialization."""
    if isinstance(data, dict):
        return {k: clean_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_for_json(v) for v in data]
    elif isinstance(data, tuple):
        return [clean_for_json(v) for v in data]
    elif isinstance(data, (datetime, date)):
        return data.isoformat()
    elif isinstance(data, Path):
        return str(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.integer, np.int64, np.int32)):
        return int(data)
    elif isinstance(data, (np.floating, np.float64, np.float32)):
        if np.isnan(data):
            return None
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, Decimal):
        return float(data)
    else:
        return data