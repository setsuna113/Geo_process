#!/usr/bin/env python3
"""
Test script for the new configuration updates.
Tests the new output_formats, processing_bounds, and species_filters sections.
"""

import pytest
import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config.config import config


class TestConfigUpdates:
    """Test the new configuration sections."""
    
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
        assert all(isinstance(coord, (int, float)) for coord in europe_bounds)
        
        # Test that all expected regions exist
        expected_regions = ['global', 'europe', 'north_america', 'south_america', 
                          'africa', 'asia', 'oceania']
        for region in expected_regions:
            assert region in processing_bounds
            bounds = processing_bounds[region]
            assert len(bounds) == 4  # [min_lon, min_lat, max_lon, max_lat]
            
        print("✅ Processing bounds configuration test passed")
    
    def test_species_filters_section(self):
        """Test the species_filters configuration section."""
        species_filters = config.species_filters
        
        # Test that the section exists
        assert species_filters is not None
        assert isinstance(species_filters, dict)
        
        # Test basic filter settings
        assert isinstance(species_filters['min_occurrence_count'], int)
        assert species_filters['min_occurrence_count'] == 5
        
        assert isinstance(species_filters['exclude_uncertain_coordinates'], bool)
        assert species_filters['exclude_uncertain_coordinates'] is True
        
        assert isinstance(species_filters['coordinate_precision_threshold'], float)
        assert species_filters['coordinate_precision_threshold'] == 0.01
        
        # Test valid_basis_of_record list
        valid_records = species_filters['valid_basis_of_record']
        assert isinstance(valid_records, list)
        expected_records = ['HUMAN_OBSERVATION', 'MACHINE_OBSERVATION', 
                          'PRESERVED_SPECIMEN', 'LIVING_SPECIMEN']
        assert all(record in valid_records for record in expected_records)
        
        # Test nested taxonomic filters
        taxonomic_filters = species_filters['taxonomic_filters']
        assert isinstance(taxonomic_filters, dict)
        assert taxonomic_filters['exclude_hybrids'] is True
        assert taxonomic_filters['require_species_level'] is True
        assert taxonomic_filters['exclude_subspecies'] is False
        
        # Test nested temporal filters
        temporal_filters = species_filters['temporal_filters']
        assert isinstance(temporal_filters, dict)
        assert temporal_filters['min_year'] == 1950
        assert temporal_filters['max_year'] == 2024
        assert temporal_filters['exclude_future_dates'] is True
        
        print("✅ Species filters configuration test passed")
    
    def test_config_get_method_with_new_sections(self):
        """Test that the config.get() method works with new sections."""
        
        # Test dot notation access to new sections
        assert config.get('output_formats.csv') is True
        assert config.get('output_formats.geojson') is False
        
        assert config.get('processing_bounds.global') == [-180, -90, 180, 90]
        assert config.get('processing_bounds.europe') is not None
        
        assert config.get('species_filters.min_occurrence_count') == 5
        assert config.get('species_filters.taxonomic_filters.exclude_hybrids') is True
        assert config.get('species_filters.temporal_filters.min_year') == 1950
        
        # Test with defaults for non-existent keys
        assert config.get('output_formats.nonexistent', False) is False
        assert config.get('processing_bounds.mars', []) == []
        
        print("✅ Config get method test passed")
    
    def test_existing_config_still_works(self):
        """Test that existing configuration sections still work."""
        
        # Test existing database config
        assert config.database is not None
        assert 'host' in config.database
        
        # Test existing grids config
        assert config.grids is not None
        assert 'cubic' in config.grids
        assert 'hexagonal' in config.grids
        
        # Test existing features config
        features = config.get('features')
        assert features is not None
        assert 'climate_variables' in features
        
        print("✅ Existing configuration compatibility test passed")
    
    def test_yaml_override_compatibility(self):
        """Test that YAML override functionality still works with new sections."""
        
        # Create a temporary config file with overrides
        temp_config_content = """
output_formats:
  csv: false
  parquet: true
  geojson: true

processing_bounds:
  custom_region: [-10, -10, 10, 10]

species_filters:
  min_occurrence_count: 10
  taxonomic_filters:
    exclude_hybrids: false
"""
        
        temp_config_path = Path("temp_test_config.yml")
        with open(temp_config_path, 'w') as f:
            f.write(temp_config_content)
        
        try:
            # Create a new config instance with override
            from config.config import Config
            test_config = Config(temp_config_path)
            
            # Test that overrides were applied
            assert test_config.get('output_formats.csv') is False
            assert test_config.get('output_formats.geojson') is True
            assert test_config.get('processing_bounds.custom_region') == [-10, -10, 10, 10]
            assert test_config.get('species_filters.min_occurrence_count') == 10
            assert test_config.get('species_filters.taxonomic_filters.exclude_hybrids') is False
            
            print("✅ YAML override compatibility test passed")
            
        finally:
            # Clean up
            if temp_config_path.exists():
                temp_config_path.unlink()


def run_all_tests():
    """Run all configuration tests."""
    print("🔬 Testing configuration updates...")
    print("=" * 50)
    
    test_instance = TestConfigUpdates()
    
    try:
        test_instance.test_output_formats_section()
        test_instance.test_processing_bounds_section()
        test_instance.test_species_filters_section()
        test_instance.test_config_get_method_with_new_sections()
        test_instance.test_existing_config_still_works()
        test_instance.test_yaml_override_compatibility()
        
        print("=" * 50)
        print("🎉 All configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
