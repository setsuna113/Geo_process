#!/usr/bin/env python3
"""
Integration test to verify the configuration system works correctly.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_configuration_import():
    """Test that configuration can be imported and loaded successfully."""
    
    try:
        # Test importing the config
        from config.config import config
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
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_configuration_import()
    print("\n" + "="*50)
    if success:
        print("🎉 Configuration integration test PASSED!")
    else:
        print("💥 Configuration integration test FAILED!")
    sys.exit(0 if success else 1)
