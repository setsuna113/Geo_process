# src/config/defaults.py
"""Default configuration values with progress and checkpoint settings"""

import os
from pathlib import Path

# Project path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAWDATA_DIR = PROJECT_ROOT / 'gpkg_data'
DATA_DIR = PROJECT_ROOT / 'data' / 'richness_maps'  # Point to richness_maps for testing
LOGS_DIR = PROJECT_ROOT / 'logs'

# Path configuration for pipeline components
PATHS = {
    'project_root': str(PROJECT_ROOT),
    'data_dir': str(DATA_DIR),
    'rawdata_dir': str(RAWDATA_DIR),
    'logs_dir': str(LOGS_DIR),
    'checkpoint_dir': str(PROJECT_ROOT / 'checkpoint_outputs'),
    'output_dir': str(PROJECT_ROOT / 'outputs'),
    'temp_dir': str(PROJECT_ROOT / 'temp')
}

# Pipeline configuration - test mode defaults (production overrides via config.yml)
PIPELINE = {
    'memory_limit_gb': 800.0,  # Conservative test mode limit
    'enable_memory_monitoring': True,
    'memory_check_interval': 5.0,  # seconds
    'auto_adjust_chunk_size': True,
    'quality_thresholds': {
        'max_nan_ratio': 0.10,  # Default 10% - will be overridden for biodiversity data
        'min_completeness': 0.80,  # 80% data completeness
        'max_outlier_ratio': 0.05,  # 5% outliers
        'min_coverage': 0.90  # 90% spatial coverage
    }
}

# Enhanced processing configuration - test mode defaults (production overrides via config.yml)
PROCESSING = {
    'batch_size': 500,  # Smaller for test mode
    'max_workers': 2,   # Fewer workers for test mode
    'chunk_size': 1000,  # Smaller chunks for test mode
    'memory_limit_mb': 2048,  # 2GB limit for test mode
    'enable_progress': True,
    'enable_checkpoints': True,
    'checkpoint_interval': 50,  # More frequent checkpoints in test
    'supports_chunking': True,
    'enable_chunking': True,  # Enable chunked processing by default
    'processing_strategy': {
        'small_datasets_mb': 50,   # Smaller thresholds for test mode
        'medium_datasets_mb': 200,  # Use chunking earlier
        'large_datasets_mb': 1024,  # Use streaming earlier
        'strategy_selection': 'auto'  # auto, memory, chunked, streaming
    },
    # Memory-aware subsampling configuration
    'subsampling': {
        'enabled': True,
        'max_samples': 50000,  # Much smaller for test mode
        'strategy': 'stratified',  # 'random', 'stratified', 'grid'
        'random_seed': 42,
        'spatial_block_size': 50,  # Smaller blocks for test
        'min_samples_per_class': 50,  # Smaller minimum
        'memory_limit_gb': 2.0  # Trigger subsampling earlier in test mode
    }
}

# Database configuration - adaptive to environment
DATABASE = {
    'host': os.getenv('DB_HOST', '/var/run/postgresql'),  # Use Unix socket for local connection
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'geoprocess_db'),  # Use existing geoprocess_db with PostGIS
    'user': os.getenv('DB_USER', os.getenv('USER', 'jason')),  # Default to current user, not postgres
    'password': os.getenv('DB_PASSWORD', ''),  # No password for local connections
    'auto_create_database': True,  # Create database if it doesn't exist
    'fallback_databases': ['postgres', 'template1'],  # Try these if main DB fails
}

# Database schema mapping - allows flexibility across different database designs
DATABASE_SCHEMA_MAPPING = {
    'raster_sources': {
        'geometry_column': 'spatial_extent',  # Primary geometry column
        'fallback_geometry_columns': ['bounds', 'geometry', 'geom', 'shape'],  # Fallbacks to try
        'active_column': 'active',  # Column indicating if record is active
        'status_column': 'processing_status',  # Column for processing status
        'metadata_column': 'metadata'  # JSON metadata column
    },
    'grid_cells': {
        'geometry_column': 'geometry',
        'fallback_geometry_columns': ['geom', 'shape', 'bounds'],
        'active_column': None,  # No active column
        'metadata_column': None
    },
    # Add other tables as needed
}

# Progress monitoring configuration
PROGRESS_MONITORING = {
    'enabled': True,
    'reporting_interval_seconds': 1.0,  # Minimum time between progress updates
    'console_output': True,
    'database_logging': True,
    'file_logging': True,
    'event_queue_size': 10000,
    'hierarchical_tracking': {
        'pipeline': {
            'report_frequency': 5.0,  # Report every 5 seconds
            'minimum_change_percent': 1.0  # Report if change > 1%
        },
        'phase': {
            'report_frequency': 2.0,
            'minimum_change_percent': 0.5
        },
        'step': {
            'report_frequency': 1.0,
            'minimum_change_percent': 0.1
        },
        'substep': {
            'report_frequency': 0.5,
            'minimum_change_percent': 0.01
        }
    },
    'memory_reporting': {
        'enabled': True,
        'interval_seconds': 10.0,
        'threshold_mb': 100  # Report if memory change > 100MB
    }
}

# Checkpoint configuration
CHECKPOINTING = {
    'enabled': True,
    'checkpoint_dir': PROJECT_ROOT / 'checkpoint_outputs',
    'compression': 'gzip',  # 'gzip', 'lz4', or None
    'intervals': {
        'time_based': 300,  # Checkpoint every 5 minutes
        'item_based': 10000,  # Checkpoint every 10k items
        'memory_based_mb': 1024,  # Checkpoint if memory usage > 1GB
        'percentage_based': 10  # Checkpoint every 10% progress
    },
    'retention': {
        'max_checkpoints': 100,
        'max_age_days': 7,
        'min_keep_per_level': {
            'pipeline': 5,
            'phase': 3,
            'step': 2,
            'substep': 1
        }
    },
    'validation': {
        'checksum_algorithm': 'sha256',
        'verify_on_load': True,
        'corruption_recovery': True
    },
    'database_backup': {
        'enabled': True,
        'include_data_tables': False,  # Only metadata by default
        'compress_backup': True
    }
}

# Timeout settings for various operations
TIMEOUTS = {
    'database_connection': 30,  # seconds
    'database_query': 300,  # 5 minutes for large queries
    'file_operations': {
        'read': 60,
        'write': 120,
        'delete': 30
    },
    'gdal_operations': 300,  # 5 minutes for GDAL operations
    'resampling_operations': 600,  # 10 minutes for resampling
    'processing_operations': {
        'default': 3600,  # 1 hour default
        'species_intersection': 7200,  # 2 hours
        'feature_extraction': 3600,
        'data_export': 1800  # 30 minutes
    },
    'network_operations': {
        'download': 300,
        'upload': 600
    },
    'checkpoint_operations': {
        'save': 60,
        'load': 120,
        'validate': 30
    }
}

# Process management configuration
PROCESS_MANAGEMENT = {
    'signals': {
        'graceful_shutdown': ['SIGTERM', 'SIGINT'],  # Signals for graceful shutdown
        'pause_processing': 'SIGUSR1',  # Signal to pause
        'resume_processing': 'SIGUSR2',  # Signal to resume
        'report_progress': 'SIGUSR1',  # Also report on SIGUSR1
        'checkpoint_now': 'SIGHUP'  # Force checkpoint
    },
    'graceful_shutdown': {
        'timeout_seconds': 30,  # Max time to wait for graceful shutdown
        'save_checkpoint': True,
        'complete_current_batch': True,
        'drain_queues': True
    },
    'daemon_mode': {
        'enabled': False,
        'pid_file': str(Path.home() / '.biodiversity' / 'pipeline.pid'),
        'log_file': LOGS_DIR / 'daemon.log',
        'working_dir': PROJECT_ROOT,
        'umask': 0o022
    },
    'health_monitoring': {
        'enabled': True,
        'check_interval_seconds': 60,
        'max_memory_mb': 8192,  # Restart if exceeds
        'max_cpu_percent': 90,  # Alert if exceeds
        'heartbeat_interval': 30,
        'auto_restart': {
            'enabled': True,
            'max_restarts': 3,
            'restart_delay_seconds': 10,
            'backoff_multiplier': 2.0
        }
    },
    'resource_limits': {
        'max_open_files': 1024,
        'max_threads': 100,
        'max_memory_mb': 8192,
        'nice_level': 10  # Lower priority
    }
}

# Memory management configuration
MEMORY_MANAGEMENT = {
    'monitoring': {
        'enabled': True,
        'interval_seconds': 5.0,
        'history_size': 1000
    },
    'pressure_thresholds': {
        'warning': 70.0,  # % of system memory
        'high': 85.0,
        'critical': 95.0
    },
    'pressure_responses': {
        'warning': {
            'log_warning': True,
            'reduce_batch_size': False,
            'force_gc': False
        },
        'high': {
            'log_warning': True,
            'reduce_batch_size': True,
            'force_gc': True,
            'clear_caches': True,
            'checkpoint': True
        },
        'critical': {
            'log_error': True,
            'pause_processing': True,
            'emergency_gc': True,
            'dump_memory_profile': True,
            'checkpoint_and_exit': True
        }
    },
    'allocation_tracking': {
        'enabled': True,
        'track_large_allocations_mb': 10,
        'report_top_allocations': 10
    },
    'cache_management': {
        'max_cache_size_mb': 1024,
        'eviction_policy': 'lru',  # least recently used
        'gc_threshold_mb': 512
    }
}

# Note: PROCESSING configuration moved to top of file near PIPELINE config

# Grid configurations
GRIDS = {
    'cubic': {
        'resolutions': [1000, 5000, 10000],  # meters
        'crs': 'EPSG:3857',  # Projected CRS for accurate area calculations
        'default_resolution': 5000
    },
    'hexagonal': {
        'resolutions': [7, 8, 9],  # H3 levels
        'crs': 'EPSG:4326',
        'default_resolution': 8
    }
}

SPECIES_CLASSIFICATION_DEFAULTS = {
    'classification_file': 'data/species_classification.csv',
    'unknown_category': 'unknown',
    'valid_categories': ['plant', 'animal', 'fungi']
}

# Data files configuration (for test mode)
DATA_FILES = {
    'plants_richness': 'daru-plants-richness.tif',
    'terrestrial_richness': 'iucn-terrestrial-richness.tif'
}

# Dataset definitions (for unified resampling pipeline)
# Supports both legacy path_key and new direct path approaches
DATASETS = {
    'target_datasets': [
        {
            'name': 'plants-richness',
            'path': str(DATA_DIR / 'daru-plants-richness.tif'),  # New: Direct path (preferred)
            'path_key': 'plants_richness',  # Legacy: For backward compatibility
            'data_type': 'richness_data',
            'band_name': 'plants_richness',
            'enabled': True
        },
        {
            'name': 'terrestrial-richness', 
            'path': str(DATA_DIR / 'iucn-terrestrial-richness.tif'),  # New: Direct path (preferred)
            'path_key': 'terrestrial_richness',  # Legacy: For backward compatibility
            'data_type': 'richness_data',
            'band_name': 'terrestrial_richness',
            'enabled': True
        }
    ]
}

# Resampling configuration for test mode
RESAMPLING = {
    'target_resolution': 0.016667,  # Exactly match test data resolution
    'target_crs': 'EPSG:4326',
    'allow_skip_resampling': True,  # Skip resampling when source resolution matches target
    'resolution_tolerance': 0.000001,  # Very tight tolerance for precision match
    'strategies': {
        'richness_data': 'sum',
        'continuous_data': 'bilinear',
        'categorical_data': 'majority'
    },
    'chunk_size': 1000,
    'validate_output': True,
    'preserve_sum': True,
    'cache_resampled': True,
    'engine': 'numpy'
}

# Storage configuration for memory-efficient database operations
STORAGE = {
    'chunk_size': 1000000,  # 1M pixels threshold for chunked storage (production: 1M, test: 100k)
    'chunk_rows': 1000,  # Process 1000 rows at a time for chunked storage
    'aggregate_to_grid': False,  # Set to True for fungi datasets to aggregate pixels to grid cells
    'grid_cell_size': 0.1,  # Grid cell size in degrees for aggregation (0.1° ≈ 11km)
    'batch_insert_size': 10000,  # Number of records per batch insert
    'enable_progress_logging': True,  # Log progress during chunked storage
    'memory_cleanup_interval': 10  # Trigger memory cleanup every N chunks
}

# Raster processing configuration
RASTER_PROCESSING = {
    'tile_size': 1000,  # pixels per tile
    'cache_ttl_days': 30,
    'memory_limit_mb': 4096,
    'parallel_workers': 4,
    'gdal_timeout': 30,  # Timeout for GDAL operations (seconds)
    'file_load_timeout': 60,  # Timeout for loading large raster files (seconds)
    'lightweight_metadata': True,  # Use lightweight metadata extraction by default
    'lazy_loading': {
        'chunk_size_mb': 100,
        'prefetch_tiles': 2
    },
    'resampling_methods': {
        'default': 'bilinear',
        'categorical': 'nearest',
        'continuous': 'bilinear'
    },
    'supported_formats': ['tif', 'tiff', 'nc', 'hdf5'],
    'compression': {
        'method': 'lzw',
        'level': 6
    }
}

# Feature configuration
FEATURES = {
    'climate_variables': ['bio_1', 'bio_12'],  # Temperature, precipitation
    'richness_types': ['present', 'absent', 'fossil'],
}

# Output formats configuration
OUTPUT_FORMATS = {
    'csv': True,
    'parquet': True, 
    'geojson': False
}

# Processing bounds configuration with regional presets
PROCESSING_BOUNDS = {
    'global': [-180, -90, 180, 90],
    'test_tiny': [0.0, 0.0, 1.0, 1.0],  # Tiny 1x1 degree subset for quick testing
    'europe': [-25.0, 35.0, 50.0, 75.0],
    'north_america': [-170.0, 15.0, -50.0, 75.0],
    'south_america': [-85.0, -60.0, -30.0, 15.0],
    'africa': [-25.0, -40.0, 55.0, 40.0],
    'asia': [60.0, -15.0, 180.0, 75.0],
    'oceania': [110.0, -50.0, 180.0, -10.0]
}

# Species filters configuration for data cleaning rules
SPECIES_FILTERS = {
    'min_occurrence_count': 5,
    'exclude_uncertain_coordinates': True,
    'coordinate_precision_threshold': 0.01,  # degrees
    'exclude_cultivated': True,
    'exclude_fossil': False,
    'valid_basis_of_record': [
        'HUMAN_OBSERVATION',
        'MACHINE_OBSERVATION', 
        'PRESERVED_SPECIMEN',
        'LIVING_SPECIMEN'
    ],
    'exclude_invalid_coordinates': True,
    'max_coordinate_uncertainty': 10000,  # meters
    'taxonomic_filters': {
        'exclude_hybrids': True,
        'require_species_level': True,
        'exclude_subspecies': False
    },
    'temporal_filters': {
        'min_year': 1950,
        'max_year': 2024,
        'exclude_future_dates': True
    }
}

# Logging configuration
LOGGING = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': LOGS_DIR / 'biodiversity.log',
}

# SOM-specific analysis configuration
SOM_ANALYSIS = {
    'max_pixels_in_memory': 10000,  # Very small for quick testing
    'chunk_overlap': 5,  # Minimal overlap
    'subsample_ratio': 0.01,  # Heavy subsampling for speed
    'min_samples': 100,  # Very few samples for quick testing
    'use_memory_mapping': True,  # Use memmap for large arrays
    'batch_training': {
        'enabled': False,  # Disable for quick testing
        'batch_size': 1000,
        'epochs_per_batch': 1
    },
    'memory_overhead_factor': 2.0,  # Reduced overhead
    'default_grid_size': [3, 3],  # Very small SOM grid
    'iterations': 10,  # Very few iterations for speed
    'sigma': 1.0,
    'learning_rate': 0.5
}

# Data preparation configuration
DATA_PREPARATION = {
    'chunk_size': 1000,  # Size for chunked processing
    'resolution_tolerance': 1e-6,  # Tolerance for resolution matching
    'bounds_tolerance': 1e-4,  # Tolerance for bounds matching
}

# Data cleaning configuration
DATA_CLEANING = {
    'log_operations': True,  # Whether to log cleaning operations
    'outlier_removal': True,
    'outlier_std_threshold': 4,  # Standard deviations for outlier detection
}

# Machine Learning configuration
MACHINE_LEARNING = {
    'models': {
        'linear_regression': {
            'alpha': 1.0,  # Ridge regularization parameter
            'fit_intercept': True,
            'normalize': False,
            'max_iter': 1000,
            'solver': 'auto'
        },
        'lightgbm': {
            'num_leaves': 31,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'reg_alpha': 0.0,  # L1 regularization
            'reg_lambda': 0.0,  # L2 regularization
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'regression',
            'metric': 'rmse',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    },
    'preprocessing': {
        'imputation': {
            'strategy': 'auto',  # 'auto', 'knn', 'iterative', 'mean', 'median'
            'knn_neighbors': 5,
            'missing_threshold': 0.3,  # Max missing rate before dropping feature
            'spatial_aware': True,
            'iterative_max_iter': 10
        },
        'scaling': {
            'method': 'standard',  # 'standard', 'robust', 'minmax', 'none'
            'clip_outliers': True,
            'outlier_threshold': 4.0  # Standard deviations
        },
        'feature_selection': {
            'variance_threshold': 0.01,
            'correlation_threshold': 0.95,
            'importance_threshold': 0.001
        }
    },
    'validation': {
        'cv_folds': 5,
        'spatial_block_size_km': 100,
        'buffer_distance_km': 50,
        'stratify_by': 'latitude',  # 'latitude', 'biome', 'none'
        'min_samples_per_fold': 100,
        'test_size': 0.2  # For single train-test split
    },
    'feature_engineering': {
        'richness_features': {
            'compute_ratios': True,
            'log_transform': True,
            'diversity_metrics': ['shannon', 'simpson']
        },
        'spatial_features': {
            'polynomial_degree': 2,
            'include_interactions': True,
            'binning_strategy': 'quantile',
            'n_bins': 10
        },
        'ecological_features': {
            'climate_interactions': True,
            'seasonality_metrics': True,
            'placeholder_ndvi': False  # For future implementation
        }
    },
    'output': {
        'save_predictions': True,
        'save_model': True,
        'save_feature_importance': True,
        'formats': ['parquet', 'pickle'],
        'predictions_dir': 'outputs/ml/predictions',
        'models_dir': 'outputs/ml/models',
        'metrics_dir': 'outputs/ml/metrics'
    },
    'performance': {
        'chunk_size': 10000,  # Rows to process at once
        'memory_limit_gb': 4.0,
        'use_dask': False,  # For very large datasets
        'cache_features': True
    }
}

# Testing configuration - DISABLED by default for safety
TESTING = {
    'enabled': os.getenv('FORCE_TEST_MODE', '').lower() in ('true', '1', 'yes'),  # Enable when forced
    'cleanup_after_test': True,
    'test_data_retention_hours': 1,
    'allowed_cleanup_tables': [
        'experiments',
        'processing_jobs',
        'features',
        'species_grid_intersections',
        'climate_data',
        'resampling_cache',
        'processing_queue',
        'pipeline_checkpoints',  # New
        'processing_steps',  # New
        'file_processing_status'  # New
    ],
    'safety_checks': {
        'require_pytest_environment': True,
        'require_test_database_name': True,
        'require_test_prefix': True,
        'max_retention_hours': 24,
        'allowed_database_patterns': ['test', 'dev', 'staging', 'tmp']
    },
    'test_data_markers': {
        'name_prefix': 'TEST_',
        'metadata_key': '__test_data__',
        'created_by': 'pytest'
    }
}