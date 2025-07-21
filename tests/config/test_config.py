#!/usr/bin/env python3
"""
Comprehensive test module for configuration system.
Merged from test_config_integration.py and test_config_updates.py
"""

import pytest
import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from src.config.config import config


class TestConfigurationSystem:
    """Test the configuration system functionality."""
    
    def test_configuration_import(self):
        """Test that configuration can be imported and loaded successfully."""
        
        # Test importing the config
        from src.config.config import config
        assert config is not None
        print("✅ Successfully imported config")
        
        # Test that all new sections are accessible
        output_formats = config.output_formats
        processing_bounds = config.processing_bounds
        species_filters = config.species_filters
        
        assert output_formats is not None
        assert processing_bounds is not None
        assert species_filters is not None
        print("✅ All new configuration sections are accessible")
        
        # Test a few key values
        assert config.get('output_formats.csv') is True
        assert len(config.get('processing_bounds.global')) == 4
        assert config.get('species_filters.min_occurrence_count') == 5
        
        print("✅ Configuration values are correctly loaded")

    def test_output_formats_section(self):
        """Test the output_formats configuration section."""
        output_formats = config.output_formats
        
        # Test that the section exists
        assert output_formats is not None
        assert isinstance(output_formats, dict)
        
        # Test specific format settings
        assert output_formats['csv'] is True
        assert output_formats['parquet'] is True
        assert output_formats['geojson'] is False
        
        print("✅ Output formats configuration test passed")
    
    def test_processing_bounds_section(self):
        """Test the processing_bounds configuration section."""
        processing_bounds = config.processing_bounds
        
        # Test that the section exists
        assert processing_bounds is not None
        assert isinstance(processing_bounds, dict)
        
        # Test global bounds
        global_bounds = processing_bounds['global']
        assert global_bounds == [-180, -90, 180, 90]
        
        # Test regional presets
        europe_bounds = processing_bounds['europe']
        assert len(europe_bounds) == 4
        assert all(isinstance(x, (int, float)) for x in europe_bounds)
        
        # Test other regions exist
        assert 'north_america' in processing_bounds
        assert 'asia' in processing_bounds
        assert 'africa' in processing_bounds
        
        print("✅ Processing bounds configuration test passed")
    
    def test_species_filters_section(self):
        """Test the species_filters configuration section."""
        species_filters = config.species_filters
        
        # Test that the section exists
        assert species_filters is not None
        assert isinstance(species_filters, dict)
        
        # Test filter settings
        assert species_filters['min_occurrence_count'] == 5
        assert species_filters['max_coordinate_uncertainty'] == 10000  # meters, not km
        assert species_filters['exclude_uncertain_coordinates'] is True
        assert species_filters['exclude_cultivated'] is True
        
        # Test valid basis of record filters
        valid_records = species_filters['valid_basis_of_record']
        assert isinstance(valid_records, list)
        assert 'HUMAN_OBSERVATION' in valid_records
        assert 'PRESERVED_SPECIMEN' in valid_records
        
        print("✅ Species filters configuration test passed")
    
    def test_raster_processing_section(self):
        """Test the raster_processing configuration section."""
        raster_config = config.raster_processing
        
        # Test that the section exists
        assert raster_config is not None
        assert isinstance(raster_config, dict)
        
        # Test key settings with correct values
        assert raster_config['tile_size'] == 1000
        assert raster_config['memory_limit_mb'] == 4096  # Actual default value
        assert raster_config['cache_ttl_days'] == 30
        assert raster_config['parallel_workers'] == 4
        
        # Test nested lazy_loading config
        lazy_config = raster_config['lazy_loading']
        assert lazy_config['chunk_size_mb'] == 100
        assert lazy_config['prefetch_tiles'] == 2
        
        print("✅ Raster processing configuration test passed")
    
    def test_config_access_methods(self):
        """Test different ways of accessing configuration values."""
        
        # Test direct attribute access
        assert hasattr(config, 'output_formats')
        assert hasattr(config, 'processing_bounds')
        
        # Test get method with dot notation
        csv_enabled = config.get('output_formats.csv')
        assert csv_enabled is True
        
        # Test get method with default value
        non_existent = config.get('non_existent.key', 'default_value')
        assert non_existent == 'default_value'
        
        # Test nested access
        global_bounds = config.get('processing_bounds.global')
        assert isinstance(global_bounds, list)
        assert len(global_bounds) == 4
        
        print("✅ Configuration access methods test passed")


def test_configuration_integration():
    """Standalone integration test function."""
    
    # Test importing the config
    from src.config.config import config
    print("✅ Successfully imported config")
    
    # Test that all new sections are accessible
    output_formats = config.output_formats
    processing_bounds = config.processing_bounds
    species_filters = config.species_filters
    
    print("✅ All new configuration sections are accessible")
    
    # Test a few key values
    assert config.get('output_formats.csv') is True
    assert len(config.get('processing_bounds.global')) == 4
    assert config.get('species_filters.min_occurrence_count') == 5
    
    print("✅ Configuration values are correctly loaded")


if __name__ == "__main__":
    try:
        test_configuration_integration()
        success = True
    except Exception as e:
        print(f"💥 Configuration integration test failed: {e}")
        success = False
        
    print("\n" + "="*50)
    if success:
        print("🎉 Configuration integration test PASSED!")
    else:
        print("💥 Configuration integration test FAILED!")
    sys.exit(0 if success else 1)
