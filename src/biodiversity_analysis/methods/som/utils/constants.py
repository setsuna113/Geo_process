"""Constants and error handling conventions for SOM module."""

import numpy as np
import os

# Error handling conventions:
# 1. For distance/metric calculations that fail: return np.inf (infinite distance)
# 2. For index lookups that fail: return INVALID_INDEX (-1)
# 3. For operations that cannot proceed: raise appropriate exception

# Sentinel values
INVALID_DISTANCE = np.inf  # For failed distance calculations (infinite distance)
INVALID_INDEX = -1         # For failed index lookups (e.g., BMU not found)
INVALID_COORDINATE = -1    # For invalid grid coordinates

# Validation thresholds
MIN_VALID_FEATURES = 2     # Minimum features for valid distance calculation
MIN_SAMPLES = 10           # Minimum samples for various operations

# Distance normalization
DEFAULT_GEO_DISTANCE_NORM = float(os.environ.get('SOM_GEO_DISTANCE_NORM', '20000.0'))
DEFAULT_FEATURE_DISTANCE_NORM = float(os.environ.get('SOM_FEATURE_DISTANCE_NORM', '1.0'))

# Memory optimization defaults
DEFAULT_BATCH_SIZE = int(os.environ.get('SOM_BATCH_SIZE', '5000'))
DEFAULT_CHUNK_SIZE = int(os.environ.get('SOM_CHUNK_SIZE', '50000'))
DEFAULT_QE_SAMPLE_SIZE = int(os.environ.get('SOM_QE_SAMPLE_SIZE', '50000'))
DEFAULT_PROGRESS_LOG_INTERVAL = int(os.environ.get('SOM_PROGRESS_LOG_INTERVAL', '50000'))
MEMORY_THRESHOLD_BYTES = int(os.environ.get('SOM_MEMORY_THRESHOLD', '100000000'))
DEFAULT_MINIBATCH_SIZE = 1000
DEFAULT_INTERNAL_CHUNK_SIZE = 1000

# Performance defaults
DEFAULT_MAX_CACHE_SIZE = 100
DEFAULT_DISTANCE_CALC_BATCH_SIZE = 1000
DEFAULT_K_NEIGHBORS = int(os.environ.get('SOM_K_NEIGHBORS', '50'))  # Read from env or use config default
DEFAULT_MAX_DISTANCE_KM = float(os.environ.get('SOM_MAX_DISTANCE_KM', '500.0'))  # For spatial calculations
DEFAULT_STRATIFIED_GRID_DIMS = (50, 50)
DEFAULT_MAX_SPATIAL_SAMPLES = 100000

# Error messages
INSUFFICIENT_FEATURES_MSG = "Insufficient valid features for distance calculation (need at least {}, got {})"
INVALID_COORDINATES_MSG = "Invalid coordinates shape: expected (n_samples, 2), got {}"
NO_VALID_BMU_MSG = "No valid BMU found for sample"