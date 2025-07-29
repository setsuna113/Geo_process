"""Unit tests for coordinate integrity validators."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.domain.validators.coordinate_integrity import (
    BoundsConsistencyValidator, CoordinateTransformValidator, ParquetValueValidator
)
from src.abstractions.interfaces.validator import ValidationSeverity


class TestBoundsConsistencyValidator:
    """Test bounds consistency validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = BoundsConsistencyValidator(tolerance=1e-6)
    
    def test_valid_bounds(self):
        """Test validation of valid bounds."""
        data = {
            'bounds': (-180.0, -90.0, 180.0, 90.0),
            'crs': 'EPSG:4326'
        }
        
        result = self.validator.validate(data)
        
        assert result.is_valid
        assert result.error_count == 0
        assert len(result.issues) == 0
    
    def test_invalid_bounds_ordering(self):
        """Test detection of invalid bounds ordering."""
        data = {
            'bounds': (180.0, -90.0, -180.0, 90.0),  # minx > maxx
            'crs': 'EPSG:4326'
        }
        
        result = self.validator.validate(data)
        
        assert not result.is_valid
        assert result.error_count > 0
        assert any("minx" in issue.message and "maxx" in issue.message 
                  for issue in result.issues)
    
    def test_out_of_range_coordinates(self):
        """Test detection of out-of-range coordinates."""
        data = {
            'bounds': (-200.0, -100.0, 200.0, 100.0),  # Outside valid lat/lon
            'crs': 'EPSG:4326'
        }
        
        result = self.validator.validate(data)
        
        assert not result.is_valid
        assert any("Longitude" in issue.message for issue in result.issues)
        assert any("Latitude" in issue.message for issue in result.issues)
    
    def test_bounds_mismatch_warning(self):
        """Test bounds mismatch generates warning."""
        data = {
            'bounds': (-180.0, -90.0, 180.0, 90.0),
            'metadata_bounds': (-180.0, -90.0, 180.0, 90.00001),  # Small mismatch
            'crs': 'EPSG:4326'
        }
        
        result = self.validator.validate(data)
        
        assert result.is_valid  # Still valid, just warning
        assert result.warning_count > 0
        assert any(issue.severity == ValidationSeverity.WARNING 
                  for issue in result.issues)
    
    def test_zero_area_bounds(self):
        """Test detection of zero-area bounds."""
        data = {
            'bounds': (0.0, 0.0, 0.0, 0.0),  # Zero area
            'crs': 'EPSG:4326'
        }
        
        result = self.validator.validate(data)
        
        assert not result.is_valid
        assert any("zero or negative area" in issue.message 
                  for issue in result.issues)
    
    def test_missing_bounds(self):
        """Test handling of missing bounds."""
        data = {
            'crs': 'EPSG:4326'
        }
        
        result = self.validator.validate(data)
        
        assert not result.is_valid
        assert any("No bounds provided" in issue.message 
                  for issue in result.issues)


class TestCoordinateTransformValidator:
    """Test coordinate transformation validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = CoordinateTransformValidator(max_error_meters=1.0)
    
    def test_valid_crs_transformation(self):
        """Test validation of valid CRS transformation."""
        data = {
            'source_crs': 'EPSG:4326',
            'target_crs': 'EPSG:3857',
            'sample_points': [(0.0, 0.0), (1.0, 1.0)]
        }
        
        result = self.validator.validate(data)
        
        assert result.is_valid
        assert result.error_count == 0
    
    def test_same_crs_transformation(self):
        """Test transformation between same CRS."""
        data = {
            'source_crs': 'EPSG:4326',
            'target_crs': 'EPSG:4326',
            'sample_points': [(0.0, 0.0)]
        }
        
        result = self.validator.validate(data)
        
        assert result.is_valid
    
    def test_invalid_crs(self):
        """Test handling of invalid CRS."""
        data = {
            'source_crs': 'INVALID:9999',
            'target_crs': 'EPSG:4326',
            'sample_points': [(0.0, 0.0)]
        }
        
        result = self.validator.validate(data)
        
        assert not result.is_valid
        assert any("Invalid CRS" in issue.message for issue in result.issues)
    
    def test_datum_shift_warning(self):
        """Test detection of datum shifts."""
        data = {
            'source_crs': 'EPSG:4326',  # WGS84
            'target_crs': 'EPSG:4269',  # NAD83
            'sample_points': [(0.0, 0.0)]
        }
        
        result = self.validator.validate(data)
        
        # Should be valid but with warning about datum shift
        assert result.is_valid
        assert result.warning_count > 0
        assert any("Datum shift" in issue.message for issue in result.issues)
    
    def test_missing_crs(self):
        """Test handling of missing CRS."""
        data = {
            'sample_points': [(0.0, 0.0)]
        }
        
        result = self.validator.validate(data)
        
        assert not result.is_valid
        assert any("Source and target CRS must be provided" in issue.message 
                  for issue in result.issues)


class TestParquetValueValidator:
    """Test Parquet value validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ParquetValueValidator(
            max_null_percentage=10.0,
            outlier_std_threshold=3.0
        )
    
    def test_valid_dataframe(self):
        """Test validation of valid DataFrame."""
        df = pd.DataFrame({
            'x': np.arange(100),
            'y': np.arange(100),
            'value': np.random.normal(0, 1, 100)
        })
        
        result = self.validator.validate(df)
        
        assert result.is_valid
        assert result.error_count == 0
    
    def test_high_null_percentage(self):
        """Test detection of high null percentage."""
        df = pd.DataFrame({
            'value': [1.0, 2.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        })
        
        result = self.validator.validate(df)
        
        assert result.is_valid  # Warnings only
        assert result.warning_count > 0
        assert any("80.0% null values" in issue.message for issue in result.issues)
    
    def test_value_out_of_range(self):
        """Test detection of values out of configured range."""
        df = pd.DataFrame({
            'temperature': np.array([20.0, 25.0, 30.0, 150.0])  # 150 is unrealistic
        })
        
        variable_config = {
            'temperature': {'min': -50, 'max': 60}
        }
        
        result = self.validator.validate({
            'dataframe': df,
            'variable_config': variable_config
        })
        
        assert not result.is_valid
        assert any("out of range" in issue.message for issue in result.issues)
    
    def test_outlier_detection(self):
        """Test outlier detection."""
        # Create data with clear outliers
        normal_data = np.random.normal(0, 1, 95)
        outliers = np.array([100, -100, 150, -150, 200])  # Clear outliers
        df = pd.DataFrame({
            'value': np.concatenate([normal_data, outliers])
        })
        
        result = self.validator.validate(df)
        
        assert result.is_valid  # Outliers generate warnings, not errors
        assert result.warning_count > 0
        assert any("outliers" in issue.message for issue in result.issues)
    
    def test_coordinate_validation(self):
        """Test validation of coordinate columns."""
        df = pd.DataFrame({
            'x': [0, 45, 90, 180, 200],  # 200 is out of range
            'y': [0, 45, 90, 100, -100]  # 100 and -100 are out of range
        })
        
        result = self.validator.validate(df)
        
        assert not result.is_valid
        assert any("Longitude" in issue.message and "outside [-180, 180]" in issue.message 
                  for issue in result.issues)
        assert any("Latitude" in issue.message and "outside [-90, 90]" in issue.message 
                  for issue in result.issues)
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        
        result = self.validator.validate(df)
        
        assert not result.is_valid
        assert any("No data provided" in issue.message for issue in result.issues)
    
    def test_all_nan_column(self):
        """Test handling of all-NaN column."""
        df = pd.DataFrame({
            'value': [np.nan] * 10
        })
        
        result = self.validator.validate(df)
        
        assert result.is_valid  # Should handle gracefully
        assert result.warning_count > 0


class TestValidatorIntegration:
    """Test validator integration scenarios."""
    
    def test_multiple_validators_on_same_data(self):
        """Test running multiple validators on the same dataset."""
        bounds_validator = BoundsConsistencyValidator()
        value_validator = ParquetValueValidator()
        
        # Create test data with coordinates and bounds
        df = pd.DataFrame({
            'x': np.linspace(-180, 180, 100),
            'y': np.linspace(-90, 90, 100),
            'value': np.random.normal(0, 1, 100)
        })
        
        bounds_data = {
            'bounds': (-180.0, -90.0, 180.0, 90.0),
            'crs': 'EPSG:4326'
        }
        
        # Run both validators
        bounds_result = bounds_validator.validate(bounds_data)
        value_result = value_validator.validate(df)
        
        assert bounds_result.is_valid
        assert value_result.is_valid
    
    def test_validation_with_real_world_bounds(self):
        """Test validation with real-world coordinate bounds."""
        validator = BoundsConsistencyValidator()
        
        # Real-world example: Continental US bounds
        data = {
            'bounds': (-125.0, 24.0, -66.0, 49.0),
            'crs': 'EPSG:4326'
        }
        
        result = validator.validate(data)
        
        assert result.is_valid
        assert result.error_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])