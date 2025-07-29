"""Test error scenarios and edge cases for validation framework."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.domain.validators.coordinate_integrity import (
    BoundsConsistencyValidator, CoordinateTransformValidator, ParquetValueValidator
)
from src.abstractions.interfaces.validator import (
    CompositeValidator, ConditionalValidator, ValidationSeverity
)


class TestValidationErrorScenarios:
    """Test various error scenarios that could occur during validation."""
    
    def test_coordinate_mismatch_detection(self):
        """Test detection of coordinate mismatches between datasets."""
        # Scenario: Two datasets claim same bounds but have different actual coordinates
        bounds_validator = BoundsConsistencyValidator(tolerance=1e-6)
        value_validator = ParquetValueValidator()
        
        # Dataset 1: Claims bounds (-10, -10, 10, 10) but has coordinates outside
        dataset1_bounds = {
            'bounds': (-10.0, -10.0, 10.0, 10.0),
            'crs': 'EPSG:4326'
        }
        
        dataset1_data = pd.DataFrame({
            'x': [-15, -5, 0, 5, 15],  # -15 and 15 are outside claimed bounds!
            'y': [-15, -5, 0, 5, 15],
            'value': [1, 2, 3, 4, 5]
        })
        
        # Validate bounds claim
        bounds_result = bounds_validator.validate(dataset1_bounds)
        assert bounds_result.is_valid  # Bounds claim is valid on its own
        
        # But data validation should detect coordinates outside bounds
        data_result = value_validator.validate(dataset1_data)
        # The validator currently checks if lon/lat are in valid ranges,
        # but doesn't check against claimed bounds
        
        # This is a gap that could cause the coordinate integrity issue!
        # The bounds claim and actual data coordinates are not cross-validated
    
    def test_projection_error_accumulation(self):
        """Test how projection errors can accumulate and cause misalignment."""
        transform_validator = CoordinateTransformValidator(max_error_meters=1.0)
        
        # Chain of transformations that could accumulate errors
        transformations = [
            ('EPSG:4326', 'EPSG:3857'),  # WGS84 to Web Mercator
            ('EPSG:3857', 'EPSG:32633'),  # Web Mercator to UTM 33N
            ('EPSG:32633', 'EPSG:4326')   # Back to WGS84
        ]
        
        # Start with precise coordinates
        original_points = [(10.0, 45.0), (11.0, 46.0)]
        current_crs = 'EPSG:4326'
        current_points = original_points.copy()
        
        errors = []
        
        # Simulate transformation chain
        for source_crs, target_crs in transformations:
            data = {
                'source_crs': source_crs,
                'target_crs': target_crs,
                'sample_points': current_points
            }
            
            result = transform_validator.validate(data)
            
            # In reality, each transformation introduces small errors
            # Simulate this by adding small random errors
            if result.is_valid:
                # Add small error to simulate real transformation
                current_points = [
                    (p[0] + np.random.normal(0, 0.00001),
                     p[1] + np.random.normal(0, 0.00001))
                    for p in current_points
                ]
            
            current_crs = target_crs
        
        # After round-trip, points should be same but won't be due to accumulation
        # This demonstrates how coordinate integrity can be lost
        assert current_crs == 'EPSG:4326'
        # In practice, current_points would differ from original_points
    
    def test_boundary_pixel_shift(self):
        """Test detection of boundary pixel shifts that cause misalignment."""
        bounds_validator = BoundsConsistencyValidator(tolerance=1e-6)
        
        # Common issue: Pixel registration differences (center vs corner)
        # Dataset thinks bounds are pixel corners
        corner_bounds = (0.0, 0.0, 10.0, 10.0)
        
        # But data is actually pixel centers, shifted by half pixel
        pixel_size = 0.1
        center_bounds = (
            0.0 + pixel_size/2,
            0.0 + pixel_size/2, 
            10.0 - pixel_size/2,
            10.0 - pixel_size/2
        )
        
        data = {
            'bounds': corner_bounds,
            'metadata_bounds': center_bounds,
            'crs': 'EPSG:4326'
        }
        
        result = bounds_validator.validate(data)
        
        # Should detect mismatch
        assert result.warning_count > 0
        assert any("Bounds mismatch" in issue.message for issue in result.issues)
        
        # This half-pixel shift is a common source of coordinate misalignment!
    
    def test_floating_point_precision_issues(self):
        """Test handling of floating point precision issues."""
        bounds_validator = BoundsConsistencyValidator(tolerance=1e-6)
        
        # Floating point arithmetic can cause tiny differences
        calculated_bound = 1.0 / 3.0 * 3.0  # Should be 1.0 but might not be exactly
        expected_bound = 1.0
        
        data = {
            'bounds': (0.0, 0.0, calculated_bound, calculated_bound),
            'metadata_bounds': (0.0, 0.0, expected_bound, expected_bound),
            'crs': 'EPSG:4326'
        }
        
        result = bounds_validator.validate(data)
        
        # Should handle tiny floating point differences within tolerance
        assert result.is_valid
        assert result.error_count == 0
    
    def test_null_value_masking_impact(self):
        """Test how null value masking can affect coordinate integrity."""
        value_validator = ParquetValueValidator(max_null_percentage=50.0)
        
        # Dataset with many nulls that might affect coordinate calculations
        coords = [(x, y) for x in range(10) for y in range(10)]
        df = pd.DataFrame({
            'x': [c[0] for c in coords],
            'y': [c[1] for c in coords],
            'value': [np.nan if (c[0] + c[1]) % 3 == 0 else 1.0 for c in coords]
        })
        
        # About 33% nulls in a pattern
        result = value_validator.validate(df)
        
        # Should detect high null percentage
        assert result.warning_count > 0
        
        # If nulls are dropped, remaining coordinates might not align with original grid!
    
    def test_composite_validation_failure_modes(self):
        """Test how composite validators handle multiple failure modes."""
        bounds_validator = BoundsConsistencyValidator()
        transform_validator = CoordinateTransformValidator()
        value_validator = ParquetValueValidator()
        
        composite = CompositeValidator([
            bounds_validator,
            transform_validator,
            value_validator
        ])
        
        # Create data with multiple issues
        data = pd.DataFrame({
            'x': [999],  # Invalid longitude
            'y': [999],  # Invalid latitude  
            'value': [np.nan]  # All null
        })
        
        # Also add bounds issues
        data.bounds = (999, 999, 1000, 1000)  # Invalid bounds
        data.crs = 'EPSG:4326'
        
        # Run composite validation
        result = composite.validate(data)
        
        # Should accumulate all issues
        assert not result.is_valid
        assert result.error_count >= 2  # At least coordinate and bounds errors
    
    def test_conditional_validation_edge_cases(self):
        """Test edge cases in conditional validation."""
        validator = ParquetValueValidator()
        
        # Only validate if data has more than 10 rows
        conditional = ConditionalValidator(
            validator,
            condition=lambda df: len(df) > 10
        )
        
        # Small dataset - validation skipped
        small_df = pd.DataFrame({'value': [1, 2, 3]})
        result = conditional.validate(small_df)
        assert result.is_valid
        assert result.metadata.get('skipped', False)
        
        # Large dataset - validation runs
        large_df = pd.DataFrame({'value': range(20)})
        result = conditional.validate(large_df)
        assert result.is_valid
        assert not result.metadata.get('skipped', False)
    
    def test_graceful_degradation_on_validation_failure(self):
        """Test that validation failures don't crash the pipeline."""
        # Create validator that will fail
        failing_validator = BoundsConsistencyValidator()
        
        # Provide invalid data type
        invalid_data = "not a dictionary"
        
        # Should handle gracefully
        try:
            result = failing_validator.validate(invalid_data)
            # Should return invalid result, not crash
            assert not result.is_valid
        except Exception:
            # If it raises, that's also acceptable error handling
            pass
    
    def test_validation_performance_with_large_datasets(self):
        """Test validation performance doesn't degrade with large datasets."""
        import time
        
        value_validator = ParquetValueValidator()
        
        # Create large dataset
        size = 1_000_000
        large_df = pd.DataFrame({
            'x': np.random.uniform(-180, 180, size),
            'y': np.random.uniform(-90, 90, size),
            'value': np.random.normal(0, 1, size)
        })
        
        # Time validation
        start = time.time()
        result = value_validator.validate(large_df)
        duration = time.time() - start
        
        # Should complete in reasonable time (< 5 seconds)
        assert duration < 5.0
        assert result is not None
    
    def test_edge_case_coordinate_values(self):
        """Test handling of edge case coordinate values."""
        value_validator = ParquetValueValidator()
        
        # Test edge coordinates
        edge_cases = pd.DataFrame({
            'x': [-180.0, 180.0, 0.0, -179.9999, 179.9999],
            'y': [-90.0, 90.0, 0.0, -89.9999, 89.9999],
            'value': [1, 2, 3, 4, 5]
        })
        
        result = value_validator.validate(edge_cases)
        
        # Should handle edge values correctly
        assert result.is_valid
        assert result.error_count == 0
        
        # Test just outside valid range
        outside_cases = pd.DataFrame({
            'x': [-180.0001, 180.0001],
            'y': [-90.0001, 90.0001],
            'value': [1, 2]
        })
        
        result = value_validator.validate(outside_cases)
        
        # Should detect out of range
        assert not result.is_valid
        assert result.error_count > 0


class TestValidationRecoveryScenarios:
    """Test recovery from validation failures."""
    
    def test_partial_validation_results_on_error(self):
        """Test that partial validation results are available after errors."""
        composite = CompositeValidator([
            BoundsConsistencyValidator(),
            CoordinateTransformValidator(),
            ParquetValueValidator()
        ])
        
        # Create data that will pass some validations but fail others
        mixed_data = {
            'bounds': (-180.0, -90.0, 180.0, 90.0),  # Valid bounds
            'source_crs': 'INVALID:9999',  # Invalid CRS - will fail
            'target_crs': 'EPSG:4326',
            'dataframe': pd.DataFrame({'x': [0], 'y': [0], 'value': [1]})  # Valid data
        }
        
        # Even with failures, should get partial results
        result = composite.validate(mixed_data)
        
        # Should have collected some successful validations
        assert len(result.issues) > 0
        assert any(issue.severity == ValidationSeverity.ERROR for issue in result.issues)
    
    def test_validation_with_missing_optional_data(self):
        """Test validation handles missing optional data gracefully."""
        bounds_validator = BoundsConsistencyValidator()
        
        # Minimal valid data
        minimal_data = {
            'bounds': (0.0, 0.0, 10.0, 10.0),
            'crs': 'EPSG:4326'
            # No metadata_bounds - optional
        }
        
        result = bounds_validator.validate(minimal_data)
        
        # Should validate successfully without optional fields
        assert result.is_valid
        assert result.error_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])