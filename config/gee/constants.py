"""GEE Pipeline Constants

Centralized constants to eliminate magic numbers scattered across the codebase.
All numeric constants, thresholds, and fixed values should be defined here.
"""

# ============================================================================
# GEOGRAPHIC CONSTANTS
# ============================================================================

# Base resolution in degrees (approximately 1.85km at equator)
BASE_RESOLUTION_DEGREES = 0.016666666666667  # 1/60 degree

# Conversion factor from degrees to meters at equator
# Note: This varies by latitude, but we use equatorial value as standard
DEGREES_TO_METERS = 111320  # More accurate than 111319.5

# Maximum resolution downsampling factor
MAX_RESOLUTION_FACTOR = 16  # Maximum 16x downsampling (1x, 2x, 4x, 8x, 16x)

# Recursive downsampling factor
DOWNSAMPLE_FACTOR = 2  # Always downsample by 2x recursively

# ============================================================================
# LATITUDE/LONGITUDE BOUNDS
# ============================================================================

# Default latitude bounds for processing (excluding extreme polar regions)
DEFAULT_LAT_MIN = -60  # Antarctic coast
DEFAULT_LAT_MAX = 75   # Arctic treeline

# Global bounds
GLOBAL_LON_MIN = -180
GLOBAL_LON_MAX = 180
GLOBAL_LAT_MIN = -90
GLOBAL_LAT_MAX = 90

# ============================================================================
# GEE PROCESSING PARAMETERS
# ============================================================================

# Maximum pixels for GEE operations
MAX_PIXELS = 1e13  # 10 trillion pixels
MAX_PIXELS_REDUCE = 1024  # For reduceResolution operations

# Tile scale for GEE exports
TILE_SCALE = 16  # Controls memory usage in exports

# Export dimensions limits
MAX_EXPORT_DIMENSION = 65536  # Maximum dimension for single export

# ============================================================================
# OCEAN MASK PARAMETERS
# ============================================================================

# MODIS water mask values
WATER_VALUE = 1  # Water pixels in MODIS mask
LAND_VALUE = 0   # Land pixels in MODIS mask

# Ocean mask threshold for coarse resolutions (meters)
COARSE_MASK_THRESHOLD = 5000  # Use simplified mask above 5km resolution

# Default ocean mask asset
WATER_MASK_ASSET = "MODIS/006/MOD44W/2015_01_01"

# ============================================================================
# CLOUD STORAGE
# ============================================================================

# Default GCS bucket
DEFAULT_GCS_BUCKET = 'biodiv-gee-yl998'

# Default export folders
DEFAULT_EXPORT_FOLDER = 'merged_outputs'
DEFAULT_TEMP_FOLDER = 'temp'

# Chunk sizes
CHUNK_SIZE_BYTES = 100 * 1024 * 1024  # 100MB
PARQUET_CHUNK_SIZE = 100000  # Number of rows per chunk

# ============================================================================
# TIMEOUTS AND INTERVALS
# ============================================================================

# Export operation timeouts (seconds)
EXPORT_TIMEOUT = 7200  # 2 hours
TASK_TIMEOUT = 3600    # 1 hour

# Check intervals (seconds)
CHECK_INTERVAL = 30    # Check task status every 30 seconds
TASK_INIT_DELAY = 10   # Wait before first status check

# Retry parameters
RETRY_DELAY = 2.0      # Delay between retries
GSUTIL_RETRY_DELAY = 5.0  # Delay for gsutil operations

# ============================================================================
# PARALLEL PROCESSING
# ============================================================================

# Default parallel settings
PARALLEL_UPLOADS = 4   # Number of parallel asset uploads
PARALLEL_DOWNLOADS = 4  # Number of parallel downloads

# ============================================================================
# DATA PARAMETERS
# ============================================================================

# NoData values
NODATA_VALUE = -9999   # Standard NoData value
NODATA_ZERO = 0        # Use 0 for NoData in some cases

# Data type ranges
INT16_MIN = -32768
INT16_MAX = 32767
UINT16_MAX = 65535

# ============================================================================
# WORLDCLIM SCALING FACTORS
# ============================================================================

# WorldClim data requires scaling to get actual values
WORLDCLIM_TEMP_SCALE = 10      # Divide by 10 for temperature in °C
WORLDCLIM_SEASONALITY_SCALE = 100  # Divide by 100 for standard deviation

# ============================================================================
# FILE SYSTEM PATHS
# ============================================================================

# Default directories (relative to project root)
DEFAULT_STATE_DIR = '~/.biodiversity'
DEFAULT_TEMP_DIR = '/tmp/biodiversity'
DEFAULT_PARQUET_DIR = 'data/merged'
DEFAULT_DOWNLOAD_DIR = 'data/downloads'

# ============================================================================
# ASSET PATHS
# ============================================================================

# Default asset prefix for GEE
DEFAULT_ASSET_PREFIX = 'projects/earthengine-legacy/assets/users/geebiodiversity'

# Production asset folder
PRODUCTION_ASSET_FOLDER = 'production'
TEST_ASSET_FOLDER = 'test'