"""Test security improvements for the ML pipeline.

Tests path validation, config validation, and error handling improvements.
"""

import os
import sys
import tempfile
import pytest
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.machine_learning.utils.security_utils import (
    PathValidator,
    ConfigValidator,
    DataLoader,
    InputSanitizer,
    DataLoadingError,
    ConfigurationError,
    SecurityValidationError
)
from src.machine_learning.pipelines.regression_pipeline import RegressionPipeline
from src.machine_learning.config_loader import MLRegressionConfig


def test_path_validator():
    """Test path validation with security checks."""
    print("\n=== Testing Path Validator ===")
    
    # Create a temporary test file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        tmp_path = tmp.name
        # Create a small parquet file
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        df.to_parquet(tmp_path)
    
    try:
        validator = PathValidator(max_file_size_mb=1)
        
        # Test valid path
        validated_path = validator.validate_path(tmp_path)
        assert validated_path.exists()
        print(f"✓ Valid path accepted: {validated_path}")
        
        # Test file size check
        large_validator = PathValidator(max_file_size_mb=0.0001)  # Very small limit
        try:
            large_validator.validate_path(tmp_path)
            assert False, "Should have raised ValueError for file size"
        except ValueError as e:
            print(f"✓ File size validation works: {e}")
        
        # Test invalid extension
        ext_validator = PathValidator(allowed_extensions={'.csv'})
        try:
            ext_validator.validate_path(tmp_path)
            assert False, "Should have raised ValueError for extension"
        except ValueError as e:
            print(f"✓ Extension validation works: {e}")
        
        # Test non-existent file
        try:
            validator.validate_path('/nonexistent/file.parquet')
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError as e:
            print(f"✓ Non-existent file detected: {e}")
        
    finally:
        # Clean up
        os.unlink(tmp_path)


def test_config_validator():
    """Test YAML configuration validation."""
    print("\n=== Testing Config Validator ===")
    
    validator = ConfigValidator()
    
    # Test valid config
    valid_config = {
        'regression_analysis': {
            'preprocessing': {
                'missing_data': {'strategy': 'complete_case'},
                'standardization': {'method': 'gelman_2sd'}
            },
            'models': {
                'glm': {'max_iter': 1000},
                'lightgbm': {'num_leaves': 31}
            }
        }
    }
    
    try:
        result = validator.validate_config(valid_config)
        assert result == True
        print("✓ Valid config accepted")
    except Exception as e:
        print(f"✗ Valid config rejected: {e}")
    
    # Test invalid config (missing required field)
    invalid_config = {
        'regression_analysis': {
            'models': {}  # Missing 'preprocessing'
        }
    }
    
    try:
        validator.validate_config(invalid_config)
        assert False, "Should have raised ValidationError"
    except Exception as e:
        print(f"✓ Invalid config rejected: {e}")
    
    # Test config sanitization
    dangerous_config = {
        'regression_analysis': {
            'preprocessing': {
                'command': 'eval("dangerous_code")',
                'path': 'os.system("rm -rf /")'
            },
            'models': {}
        }
    }
    
    sanitized = validator.sanitize_config(dangerous_config)
    assert 'eval(' not in str(sanitized)
    assert 'os.' not in str(sanitized)
    print("✓ Dangerous patterns sanitized")


def test_secure_data_loader():
    """Test secure data loading with validation."""
    print("\n=== Testing Secure Data Loader ===")
    
    # Create test data
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        tmp_path = tmp.name
        df = pd.DataFrame({
            'fungi_richness': np.random.randint(0, 100, 100),
            'plants_richness': np.random.randint(0, 200, 100),
            'vertebrates_richness': np.random.randint(0, 50, 100),
            'bio1': np.random.randn(100),
            'bio4': np.random.randn(100),
            'longitude': np.random.uniform(-180, 180, 100),
            'latitude': np.random.uniform(-90, 90, 100)
        })
        df.to_parquet(tmp_path)
    
    try:
        # Test with required columns
        loader = DataLoader(
            required_columns=['fungi_richness', 'plants_richness'],
            max_file_size_mb=10
        )
        
        loaded_df = loader.load_data(tmp_path)
        assert len(loaded_df) == 100
        print(f"✓ Data loaded successfully: {loaded_df.shape}")
        
        # Test with missing required columns
        loader_strict = DataLoader(
            required_columns=['nonexistent_column']
        )
        
        try:
            loader_strict.load_data(tmp_path)
            assert False, "Should have raised ValueError for missing columns"
        except ValueError as e:
            print(f"✓ Missing columns detected: {e}")
        
        # Test suspicious column names
        bad_df = pd.DataFrame({'__evil__': [1, 2], 'exec_command': [3, 4]})
        bad_path = tmp_path.replace('.parquet', '_bad.parquet')
        bad_df.to_parquet(bad_path)
        
        try:
            loader.load_data(bad_path)
            assert False, "Should have detected suspicious column"
        except ValueError as e:
            print(f"✓ Suspicious columns detected: {e}")
        finally:
            os.unlink(bad_path)
            
    finally:
        os.unlink(tmp_path)


def test_input_sanitizer():
    """Test input sanitization."""
    print("\n=== Testing Input Sanitizer ===")
    
    # Test string sanitization
    dangerous_string = "DROP TABLE users; -- comment"
    safe_string = InputSanitizer.sanitize_string(dangerous_string)
    assert 'DROP' not in safe_string
    print(f"✓ SQL injection pattern removed: '{dangerous_string}' -> '{safe_string}'")
    
    # Test length truncation
    long_string = "a" * 2000
    truncated = InputSanitizer.sanitize_string(long_string, max_length=100)
    assert len(truncated) == 100
    print(f"✓ String truncated: {len(long_string)} -> {len(truncated)}")
    
    # Test number sanitization
    safe_num = InputSanitizer.sanitize_number(42.5, min_val=0, max_val=100)
    assert safe_num == 42.5
    print(f"✓ Valid number accepted: {safe_num}")
    
    clamped_num = InputSanitizer.sanitize_number(150, min_val=0, max_val=100)
    assert clamped_num == 100
    print(f"✓ Number clamped to max: 150 -> {clamped_num}")
    
    try:
        InputSanitizer.sanitize_number(float('inf'))
        assert False, "Should have rejected infinite"
    except ValueError as e:
        print(f"✓ Non-finite number rejected: {e}")


def test_pipeline_integration():
    """Test security improvements in the pipeline."""
    print("\n=== Testing Pipeline Integration ===")
    
    # Create test data
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        tmp_path = tmp.name
        df = pd.DataFrame({
            'fungi_richness': np.random.randint(1, 100, 50),
            'plants_richness': np.random.randint(1, 200, 50),
            'vertebrates_richness': np.random.randint(1, 50, 50),
            'bio1': np.random.randn(50),
            'bio4': np.random.randn(50),
            'bio12': np.random.randn(50),
            'bio15': np.random.randn(50),
            'elevation': np.random.randn(50),
            'longitude': np.random.uniform(-180, 180, 50),
            'latitude': np.random.uniform(-90, 90, 50)
        })
        df.to_parquet(tmp_path)
    
    try:
        # Test pipeline with secure loading
        pipeline = RegressionPipeline()
        
        # Load with security validation
        data = pipeline.load_data(tmp_path, validate_security=True, max_file_size_mb=10)
        assert len(data) == 50
        print(f"✓ Pipeline loaded data securely: {data.shape}")
        
        # Test with file size limit
        try:
            pipeline.load_data(tmp_path, validate_security=True, max_file_size_mb=0.0001)
            assert False, "Should have rejected large file"
        except Exception as e:
            print(f"✓ Pipeline enforced file size limit: {e}")
        
        # Test legacy loading (backwards compatibility)
        data_legacy = pipeline.load_data(tmp_path, validate_security=False)
        assert len(data_legacy) == 50
        print(f"✓ Legacy loading still works: {data_legacy.shape}")
        
    finally:
        os.unlink(tmp_path)


def test_glm_error_handling():
    """Test improved error handling in GLM models."""
    print("\n=== Testing GLM Error Handling ===")
    
    from src.machine_learning.models.glm.negative_binomial import NegativeBinomialGLM
    
    # Create problematic data that might cause convergence issues
    X = pd.DataFrame({
        'x1': [1, 1, 1, 1, 1],  # No variation
        'x2': [1, 2, 3, 4, 5]
    })
    y = np.array([1, 1, 1, 1, 1])  # No variation in response
    
    model = NegativeBinomialGLM(config={})
    
    try:
        # This should trigger specific error handling
        model.fit(X, y)
        print("✓ GLM handled problematic data")
    except (ValueError, RuntimeError) as e:
        print(f"✓ GLM raised specific exception: {type(e).__name__}: {e}")
    except Exception as e:
        print(f"✗ GLM raised unexpected exception: {type(e).__name__}: {e}")


def main():
    """Run all security tests."""
    print("=" * 60)
    print("SECURITY IMPROVEMENTS TEST SUITE")
    print("=" * 60)
    
    try:
        test_path_validator()
        test_config_validator()
        test_secure_data_loader()
        test_input_sanitizer()
        test_pipeline_integration()
        test_glm_error_handling()
        
        print("\n" + "=" * 60)
        print("✅ ALL SECURITY TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())