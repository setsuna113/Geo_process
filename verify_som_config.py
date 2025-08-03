#!/usr/bin/env python
"""Verify SOM configuration is loaded correctly."""

import sys
sys.path.append('/home/yl998/dev/geo')

from src.config.som import get_som_config
from src.biodiversity_analysis.methods.som.analyzer import GeoSOMAnalyzer

# Check what config is loaded
print("=== Verifying SOM Configuration ===")
som_config = get_som_config()

# Check grid size in config
grid_size = som_config.get_grid_size()
print(f"Grid size from som_config_optimized.yml: {grid_size}")

# Check what the analyzer would use
analyzer = GeoSOMAnalyzer()
print(f"Analyzer config loaded: {analyzer.config.get('architecture_config', {}).get('grid_size', 'NOT FOUND')}")

# Check the actual grid size that would be determined
n_samples = 11385954  # From your data
determined_size = analyzer._determine_grid_size(n_samples, 'auto')
print(f"Auto-determined grid size for {n_samples} samples: {determined_size}")

# Verify it's using optimized config
config_path = som_config._config_path
print(f"\nConfig loaded from: {config_path}")
print(f"Is optimized config: {'som_config_optimized.yml' in str(config_path)}")