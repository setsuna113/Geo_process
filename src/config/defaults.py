# src/config/defaults.py
"""Default configuration values"""

import os
from pathlib import Path

# Project path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAWDATA_DIR = PROJECT_ROOT / 'gpkg_data'
DATA_DIR = PROJECT_ROOT / 'data'
LOGS_DIR = PROJECT_ROOT / 'logs'

# Database configuration
DATABASE = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'geoprocess_db'),
    'user': os.getenv('DB_USER', os.getenv('USER')),
    'password': os.getenv('DB_PASSWORD', '123456'),
}

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

# Processing configuration
PROCESSING = {
    'batch_size': 1000,
    'max_workers': 4,
    'chunk_size': 10000,
}

# Raster processing configuration
RASTER_PROCESSING = {
    'tile_size': 1000,  # pixels per tile
    'cache_ttl_days': 30,
    'memory_limit_mb': 4096,
    'parallel_workers': 4,
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
