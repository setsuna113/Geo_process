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

# Feature configuration
FEATURES = {
    'climate_variables': ['bio_1', 'bio_12'],  # Temperature, precipitation
    'richness_types': ['present', 'absent', 'fossil'],
}

# Logging configuration
LOGGING = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': LOGS_DIR / 'biodiversity.log',
}