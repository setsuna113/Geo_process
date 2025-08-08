"""Constants and default values for CP-WOPT implementation."""

import numpy as np

# Algorithm defaults
DEFAULT_RANK = 20
DEFAULT_MAX_ITERATIONS = 1000
DEFAULT_CONVERGENCE_THRESHOLD = 1e-6
DEFAULT_INIT_METHOD = 'random'
DEFAULT_OPTIMIZER = 'als'

# Mini-batch processing
DEFAULT_MINIBATCH_SIZE = 10000
MIN_MINIBATCH_SIZE = 1000
MAX_MINIBATCH_SIZE = 100000

# Tensor construction
DEFAULT_LAT_BINS = 50
DEFAULT_LON_BINS = 50
DEFAULT_AGGREGATION = 'sum'
DEFAULT_TRANSFORM = 'log1p'

# Sparsity handling
SPARSITY_THRESHOLD = 0.95  # Convert to sparse format above this
MIN_OBSERVATIONS_PER_CELL = 1

# Memory management
MEMORY_LIMIT_GB = 8.0  # Chunk if tensor exceeds this
CHUNK_MEMORY_FRACTION = 0.8  # Use 80% of limit per chunk

# Numerical stability
EPSILON = 1e-10
MIN_COMPONENT_NORM = 1e-8
REGULARIZATION_LAMBDA = 1e-9
NAN_REPLACEMENT_VALUE = 0.0

# Validation
MIN_VALID_FEATURES = 1
MAX_MISSING_RATIO = 0.99  # Maximum allowed missing data

# Progress tracking
PROGRESS_INTERVAL = 10  # Log progress every N iterations
CHECKPOINT_INTERVAL = 50  # Save checkpoint every N iterations

# Convergence criteria
PATIENCE = 10  # Iterations to wait for improvement
MIN_IMPROVEMENT = 1e-4  # Minimum relative improvement

# Component interpretation thresholds
TOP_N_FEATURES = 10  # Number of top features to show per component
SIGNIFICANCE_THRESHOLD = 0.1  # Minimum loading to be considered significant

# Reporting
FIGURE_DPI = 150
FIGURE_SIZE = (10, 8)
COLORMAP = 'viridis'

# HiCOO Sparse Format Configuration
HICOO_BLOCK_SIZE = 128
HICOO_MEMORY_FACTOR = 0.9

# Error messages
ERROR_INVALID_RANK = "Rank must be positive integer, got {}"
ERROR_INVALID_SHAPE = "Invalid tensor shape: {}"
ERROR_CONVERGENCE_FAILED = "Failed to converge after {} iterations"
ERROR_MEMORY_LIMIT = "Tensor size {:.1f}GB exceeds memory limit {:.1f}GB"
ERROR_CUDA_OOM = "CUDA out of memory. Try reducing rank or using smaller batches"
ERROR_CUDA_DEVICE = "CUDA device error: {}"
ERROR_NO_MODEL = "Model has not been fitted yet. Call fit() first"

# Warning messages  
WARNING_HIGH_SPARSITY = "Very high sparsity ({:.1%}) detected, results may be unstable"
WARNING_FEW_ITERATIONS = "Only {} iterations completed, results may not be converged"
WARNING_RANK_TOO_HIGH = "Rank {} may be too high for tensor with shape {}"