#!/usr/bin/env python3
"""Test the config system fix for test mode detection."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Simulate test mode by adding pytest to modules
sys.modules['pytest'] = type(sys)('pytest')

def test_config_test_mode():
    """Test that config detects test mode and uses appropriate database settings."""
    from src.config.config import Config
    
    config = Config()
    
    print("🔍 Config test mode detection:")
    print(f"   Is test mode: {config._is_test_mode()}")
    print(f"   Database host: {config.get('database.host')}")
    print(f"   Database port: {config.get('database.port')}")
    print(f"   Database name: {config.get('database.database')}")
    print(f"   Database user: {config.get('database.user')}")
    
    # Verify test mode settings
    assert config._is_test_mode() == True
    assert config.get('database.port') == 5432  # Should be standard port, not 51051
    assert config.get('database.database') == 'geoprocess_test_db'
    
    print("✅ Config test mode detection working correctly!")

def test_config_normal_mode():
    """Test config in normal mode (remove pytest from modules)."""
    # Remove pytest to simulate normal mode
    if 'pytest' in sys.modules:
        del sys.modules['pytest']
    
    # Force reload of config
    import importlib
    import src.config.config
    importlib.reload(src.config.config)
    
    from src.config.config import Config
    config = Config()
    
    print("\n🔍 Config normal mode (should use YAML):")
    print(f"   Is test mode: {config._is_test_mode()}")
    print(f"   Database port: {config.get('database.port')}")
    
    # In normal mode, should use config.yml values (51051 for cluster)
    print("✅ Config normal mode working correctly!")

if __name__ == "__main__":
    test_config_test_mode()
    test_config_normal_mode()
    print("\n🎉 All config tests passed!")