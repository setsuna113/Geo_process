#!/usr/bin/env python
"""Test if SOM config is loading properly."""

import os
os.environ['SOM_CONFIG_PATH'] = '/home/yl998/dev/geo/src/config/som/som_config_optimized.yml'

from src.biodiversity_analysis.methods.som.analyzer import GeoSOMAnalyzer

# Create analyzer
analyzer = GeoSOMAnalyzer()

print(f"Use advanced preprocessing: {analyzer.use_advanced_preprocessing}")
print(f"Config preprocessing section: {analyzer.config.get('preprocessing_config', {})}")