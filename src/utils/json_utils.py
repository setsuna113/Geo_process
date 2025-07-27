# src/utils/json_utils.py
"""JSON utilities for handling special types."""

import json
import numpy as np
from datetime import datetime, date
from pathlib import Path
from decimal import Decimal


class ExtendedJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles additional types."""
    
    def default(self, obj):
        """Convert non-serializable objects to serializable format."""
        # Handle datetime types
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        
        # Handle Path objects
        if isinstance(obj, Path):
            return str(obj)
        
        # Handle numpy types
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            if np.isnan(obj):
                return None
            return float(obj)
        
        if isinstance(obj, np.bool_):
            return bool(obj)
        
        # Handle Decimal
        if isinstance(obj, Decimal):
            return float(obj)
        
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