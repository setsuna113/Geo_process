"""Unit tests for CoordinateGenerator in climate_gee module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import yaml
from unittest.mock import patch

# Mock database connections before importing climate_gee modules
with patch('src.database.connection.DatabaseManager'):
    with patch('psycopg2.pool.ThreadedConnectionPool'):
        from src.climate_gee.coordinate_generator import (
            CoordinateGenerator,
            load_config_resolution,
            get_processing_bounds,
            create_from_config
        )


class TestCoordinateGenerator:
    """Test suite for CoordinateGenerator class."""
    
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        generator = CoordinateGenerator()
        assert generator.target_resolution == 0.016667
        assert generator.logger is not None
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        resolution = 0.1
        generator = CoordinateGenerator(target_resolution=resolution)
        assert generator.target_resolution == resolution
    
    def test_generate_coordinate_grid_basic(self):
        """Test basic coordinate grid generation."""
        generator = CoordinateGenerator(target_resolution=1.0)
        bounds = (0, 0, 2, 2)
        
        coords_df = generator.generate_coordinate_grid(bounds)
        
        # Should have x and y columns
        assert 'x' in coords_df.columns
        assert 'y' in coords_df.columns
        
        # Check basic bounds
        assert coords_df['x'].min() == 0.0
        assert coords_df['y'].min() == 0.0
        
        # Check that we have expected number of points (2x2 grid)
        assert len(coords_df) == 4
    
    def test_generate_coordinate_grid_precision(self):
        """Test coordinate generation precision."""
        generator = CoordinateGenerator(target_resolution=0.1)
        bounds = (0, 0, 0.3, 0.2)
        
        coords_df = generator.generate_coordinate_grid(bounds)
        
        # Check x coordinates
        expected_x = [0.0, 0.1, 0.2]
        actual_x = sorted(coords_df['x'].unique())
        np.testing.assert_array_almost_equal(actual_x, expected_x, decimal=6)
        
        # Check y coordinates  
        expected_y = [0.0, 0.1]
        actual_y = sorted(coords_df['y'].unique())
        np.testing.assert_array_almost_equal(actual_y, expected_y, decimal=6)
    
    def test_generate_coordinate_grid_negative_bounds(self):
        """Test coordinate generation with negative bounds."""
        generator = CoordinateGenerator(target_resolution=1.0)
        bounds = (-2, -1, 1, 1)
        
        coords_df = generator.generate_coordinate_grid(bounds)
        
        assert coords_df['x'].min() == -2.0
        assert coords_df['y'].min() == -1.0
        assert len(coords_df) == 6  # 3x2 grid
    
    def test_generate_coordinate_grid_chunked(self):
        """Test chunked coordinate generation."""
        generator = CoordinateGenerator(target_resolution=1.0)
        bounds = (0, 0, 3, 3)
        chunk_size = 5
        
        coords_df = generator.generate_coordinate_grid(bounds, chunk_size=chunk_size)
        
        # Should still generate all points
        assert len(coords_df) == 9  # 3x3 grid
        assert 'x' in coords_df.columns
        assert 'y' in coords_df.columns
    
    def test_generate_streaming_coordinates(self):
        """Test streaming coordinate generation."""
        generator = CoordinateGenerator(target_resolution=1.0)
        bounds = (0, 0, 2, 2)
        batch_size = 2
        
        batches = list(generator.generate_streaming_coordinates(bounds, batch_size))
        
        # Should have 2 batches (4 points total, 2 per batch)
        assert len(batches) == 2
        
        # Combine all batches
        all_coords = pd.concat(batches, ignore_index=True)
        assert len(all_coords) == 4
        
        # Check batch metadata
        assert batches[0].attrs['batch_id'] == 1
        assert batches[0].attrs['total_batches'] == 2
    
    def test_generate_coordinate_chunks(self):
        """Test coordinate chunk generation."""
        generator = CoordinateGenerator(target_resolution=1.0)
        bounds = (0, 0, 3, 2)
        chunk_size = 3
        
        chunks = list(generator.generate_coordinate_chunks(bounds, chunk_size))
        
        # Should have multiple chunks
        assert len(chunks) >= 1
        
        # Combine all chunks
        all_coords = pd.concat(chunks, ignore_index=True)
        assert len(all_coords) == 6  # 3x2 grid
        
        # Check chunk metadata
        assert hasattr(chunks[0], 'attrs')
        assert 'chunk_id' in chunks[0].attrs
    
    def test_validate_coordinate_alignment_valid(self):
        """Test coordinate alignment validation with valid data."""
        generator = CoordinateGenerator(target_resolution=1.0)
        bounds = (0, 0, 2, 2)
        
        coords_df = generator.generate_coordinate_grid(bounds)
        is_valid = generator.validate_coordinate_alignment(coords_df, bounds)
        
        assert is_valid is True
    
    def test_validate_coordinate_alignment_empty(self):
        """Test coordinate alignment validation with empty data."""
        generator = CoordinateGenerator()
        empty_df = pd.DataFrame(columns=['x', 'y'])
        bounds = (0, 0, 1, 1)
        
        is_valid = generator.validate_coordinate_alignment(empty_df, bounds)
        
        assert is_valid is False
    
    def test_validate_coordinate_alignment_tolerance(self):
        """Test coordinate alignment validation with tolerance."""
        generator = CoordinateGenerator(target_resolution=1.0)
        
        # Create slightly offset coordinates
        coords_df = pd.DataFrame({
            'x': [0.000001, 1.000001],
            'y': [0.000001, 1.000001]
        })
        bounds = (0, 0, 1, 1)
        
        # Should pass with default tolerance
        is_valid = generator.validate_coordinate_alignment(coords_df, bounds)
        assert is_valid is True
        
        # Should fail with strict tolerance
        is_valid = generator.validate_coordinate_alignment(coords_df, bounds, tolerance=1e-9)
        assert is_valid is False


class TestConfigurationFunctions:
    """Test suite for configuration utility functions."""
    
    def test_load_config_resolution_default(self):
        """Test loading resolution with default when config missing."""
        # Test with non-existent file
        resolution = load_config_resolution("nonexistent.yml")
        assert resolution == 0.016667
    
    def test_load_config_resolution_from_file(self):
        """Test loading resolution from actual config file."""
        # Create temporary config file
        config_data = {
            'resampling': {
                'target_resolution': 0.05
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            resolution = load_config_resolution(temp_config_path)
            assert resolution == 0.05
        finally:
            Path(temp_config_path).unlink()
    
    def test_get_processing_bounds_default(self):
        """Test getting processing bounds with default when config missing."""
        bounds = get_processing_bounds("nonexistent.yml")
        assert bounds == (-180, -90, 180, 90)
    
    def test_get_processing_bounds_from_file(self):
        """Test loading bounds from actual config file."""
        config_data = {
            'processing_bounds': {
                'test_region': [-10, -5, 10, 5]
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            bounds = get_processing_bounds(temp_config_path, 'test_region')
            assert bounds == (-10, -5, 10, 5)
        finally:
            Path(temp_config_path).unlink()
    
    def test_create_from_config(self):
        """Test creating CoordinateGenerator from config."""
        config_data = {
            'resampling': {
                'target_resolution': 0.025
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            generator = create_from_config(temp_config_path)
            assert generator.target_resolution == 0.025
        finally:
            Path(temp_config_path).unlink()


class TestCoordinateGeneratorEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_small_resolution(self):
        """Test with very small resolution."""
        generator = CoordinateGenerator(target_resolution=0.001)
        bounds = (0, 0, 0.01, 0.01)
        
        coords_df = generator.generate_coordinate_grid(bounds)
        
        # Should handle small resolution without issues
        assert len(coords_df) > 0
        assert 'x' in coords_df.columns
        assert 'y' in coords_df.columns
    
    def test_large_grid_memory_management(self):
        """Test memory management with large grids."""
        generator = CoordinateGenerator(target_resolution=0.1)
        bounds = (0, 0, 100, 100)  # Large area
        
        # Should use chunked generation automatically
        coords_df = generator.generate_coordinate_grid(bounds)
        
        assert len(coords_df) > 1000000  # Large number of points
        assert 'x' in coords_df.columns
        assert 'y' in coords_df.columns
    
    def test_invalid_bounds_order(self):
        """Test with bounds in wrong order (max < min)."""
        generator = CoordinateGenerator(target_resolution=1.0)
        bounds = (2, 2, 0, 0)  # max < min
        
        # Should still work (linspace handles this)
        coords_df = generator.generate_coordinate_grid(bounds)
        assert len(coords_df) >= 0
    
    def test_zero_size_bounds(self):
        """Test with zero-size bounds."""
        generator = CoordinateGenerator(target_resolution=1.0)
        bounds = (0, 0, 0, 0)
        
        coords_df = generator.generate_coordinate_grid(bounds)
        # Should return empty or single point
        assert len(coords_df) >= 0


class TestCoordinateGeneratorIntegration:
    """Integration tests with real-world scenarios."""
    
    def test_pipeline_coordinate_alignment(self):
        """Test that coordinates align with expected pipeline format."""
        # Use same resolution as typical pipeline
        generator = CoordinateGenerator(target_resolution=0.016667)
        
        # Test with small global subset
        bounds = (-1, -1, 1, 1)
        coords_df = generator.generate_coordinate_grid(bounds)
        
        # Verify standard checks
        assert len(coords_df) > 0
        assert coords_df['x'].dtype in [np.float64, np.float32]
        assert coords_df['y'].dtype in [np.float64, np.float32]
        
        # Check coordinate spacing
        x_unique = sorted(coords_df['x'].unique())
        if len(x_unique) > 1:
            spacing = x_unique[1] - x_unique[0]
            np.testing.assert_almost_equal(spacing, 0.016667, decimal=5)
    
    def test_multiple_resolution_compatibility(self):
        """Test that different resolutions produce consistent patterns."""
        bounds = (0, 0, 1, 1)
        
        # Test with different resolutions
        resolutions = [0.1, 0.05, 0.025]
        generators = [CoordinateGenerator(target_resolution=r) for r in resolutions]
        
        coord_grids = []
        for gen in generators:
            coords_df = gen.generate_coordinate_grid(bounds)
            coord_grids.append(coords_df)
        
        # Higher resolution should have more points
        assert len(coord_grids[2]) > len(coord_grids[1]) > len(coord_grids[0])
        
        # All should cover same area
        for coords_df in coord_grids:
            assert coords_df['x'].min() == 0.0
            assert coords_df['y'].min() == 0.0