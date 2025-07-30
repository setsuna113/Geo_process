#!/usr/bin/env python3
"""Test configuration validation for streaming export."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config import Config
from src.database.connection import DatabaseManager
from src.pipelines.stages.merge_stage import MergeStage


def create_test_config(settings):
    """Create a mock config with specific settings."""
    mock_config = MagicMock()
    mock_config.get = lambda key, default=None: settings.get(key, default)
    return mock_config


def test_streaming_validation():
    """Test that streaming configuration is properly validated."""
    print("Testing Streaming Configuration Validation")
    print("="*60)
    
    # Test 1: Valid configuration (CSV only)
    print("\n1. Testing valid configuration (CSV only)...")
    test_config = create_test_config({
        'merge.enable_streaming': True,
        'export.formats': ['csv'],
        'merge.streaming_chunk_size': 5000
    })
    
    merge_stage = MergeStage()
    merge_stage.config = test_config
    is_valid, errors = merge_stage.validate()
    print(f"   Valid: {is_valid}")
    print(f"   Errors: {errors}")
    assert is_valid, "Valid configuration should pass"
    
    # Test 2: Invalid format with streaming
    print("\n2. Testing invalid format with streaming...")
    test_config = create_test_config({
        'merge.enable_streaming': True,
        'export.formats': ['csv', 'parquet'],
        'merge.streaming_chunk_size': 5000
    })
    
    merge_stage = MergeStage()
    merge_stage.config = test_config
    is_valid, errors = merge_stage.validate()
    print(f"   Valid: {is_valid}")
    print(f"   Errors: {errors}")
    assert not is_valid, "Should fail with non-CSV formats"
    assert any("only supports CSV" in err for err in errors)
    
    # Test 3: Chunk size too small
    print("\n3. Testing chunk size too small...")
    test_config = create_test_config({
        'merge.enable_streaming': True,
        'export.formats': ['csv'],
        'merge.streaming_chunk_size': 50
    })
    
    merge_stage = MergeStage()
    merge_stage.config = test_config
    is_valid, errors = merge_stage.validate()
    print(f"   Valid: {is_valid}")
    print(f"   Errors: {errors}")
    assert not is_valid, "Should fail with small chunk size"
    assert any("too small" in err for err in errors)
    
    # Test 4: Chunk size too large
    print("\n4. Testing chunk size too large...")
    test_config = create_test_config({
        'merge.enable_streaming': True,
        'export.formats': ['csv'],
        'merge.streaming_chunk_size': 2000000
    })
    
    merge_stage = MergeStage()
    merge_stage.config = test_config
    is_valid, errors = merge_stage.validate()
    print(f"   Valid: {is_valid}")
    print(f"   Errors: {errors}")
    assert not is_valid, "Should fail with large chunk size"
    assert any("too large" in err for err in errors)
    
    # Test 5: Streaming disabled (should pass with any format)
    print("\n5. Testing with streaming disabled...")
    test_config = create_test_config({
        'merge.enable_streaming': False,
        'export.formats': ['csv', 'parquet', 'netcdf'],
        'merge.streaming_chunk_size': 5000
    })
    
    merge_stage = MergeStage()
    merge_stage.config = test_config
    is_valid, errors = merge_stage.validate()
    print(f"   Valid: {is_valid}")
    print(f"   Errors: {errors}")
    assert is_valid, "Should pass when streaming is disabled"
    
    print("\n✅ All validation tests passed!")


if __name__ == "__main__":
    test_streaming_validation()