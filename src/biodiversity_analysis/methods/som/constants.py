"""Constants and error handling conventions for SOM module."""

import numpy as np

# Error handling conventions:
# 1. For distance/metric calculations that fail: return np.nan
# 2. For index lookups that fail: return INVALID_INDEX (-1)
# 3. For operations that cannot proceed: raise appropriate exception

# Sentinel values
INVALID_DISTANCE = np.nan  # For failed distance calculations
INVALID_INDEX = -1         # For failed index lookups (e.g., BMU not found)
INVALID_COORDINATE = -1    # For invalid grid coordinates

# Validation thresholds
MIN_VALID_FEATURES = 2     # Minimum features for valid distance calculation
MIN_SAMPLES = 10           # Minimum samples for various operations

# Error messages
INSUFFICIENT_FEATURES_MSG = "Insufficient valid features for distance calculation (need at least {}, got {})"
INVALID_COORDINATES_MSG = "Invalid coordinates shape: expected (n_samples, 2), got {}"
NO_VALID_BMU_MSG = "No valid BMU found for sample"