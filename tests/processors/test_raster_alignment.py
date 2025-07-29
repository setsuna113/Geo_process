"""Unit tests for RasterAligner grid shift calculations."""

import unittest
from pathlib import Path
import numpy as np
from unittest.mock import Mock, MagicMock

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.processors.data_preparation.raster_alignment import (
    RasterAligner, GridAlignment
)


class TestRasterAlignerGridShifts(unittest.TestCase):
    """Test RasterAligner grid shift calculation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.aligner = RasterAligner()
        
    def create_mock_dataset_info(self, name, bounds, resolution):
        """Create a mock dataset info object."""
        info = Mock()
        info.name = name
        info.bounds = bounds
        info.actual_bounds = bounds
        info.target_resolution = resolution
        return info
        
    def test_no_shift_required_aligned_datasets(self):
        """Test that aligned datasets require no shift."""
        # Create two perfectly aligned datasets
        datasets = [
            self.create_mock_dataset_info(
                "dataset1",
                (-180, -90, 180, 90),
                1.0
            ),
            self.create_mock_dataset_info(
                "dataset2", 
                (-180, -90, 180, 90),
                1.0
            )
        ]
        
        alignments = self.aligner.calculate_grid_shifts(datasets)
        
        # Both should have no shift
        self.assertEqual(len(alignments), 2)
        for alignment in alignments:
            self.assertFalse(alignment.requires_shift)
            self.assertEqual(alignment.x_shift, 0.0)
            self.assertEqual(alignment.y_shift, 0.0)
            
    def test_fractional_pixel_shift_detection(self):
        """Test detection of fractional pixel shifts."""
        # Dataset 2 is shifted by half a pixel
        # Note: Y shift is negative because max_y is shifted up
        datasets = [
            self.create_mock_dataset_info(
                "dataset1",
                (-180, -90, 180, 90),
                1.0
            ),
            self.create_mock_dataset_info(
                "dataset2",
                (-179.5, -89.5, 180.5, 90.5),  # 0.5 degree shift
                1.0
            )
        ]
        
        alignments = self.aligner.calculate_grid_shifts(datasets)
        
        # First dataset is reference (no shift)
        self.assertEqual(alignments[0].aligned_dataset, "dataset1")
        self.assertFalse(alignments[0].requires_shift)
        
        # Second dataset should have shift
        self.assertEqual(alignments[1].aligned_dataset, "dataset2")
        self.assertTrue(alignments[1].requires_shift)
        self.assertAlmostEqual(alignments[1].x_shift, -0.5)
        self.assertAlmostEqual(alignments[1].y_shift, -0.5)  # Negative because max_y is higher
        self.assertAlmostEqual(alignments[1].shift_pixels_x, 0.5)  # Pixel fraction is positive
        self.assertAlmostEqual(alignments[1].shift_pixels_y, 0.5)  # Pixel fraction is positive
        
    def test_multiple_datasets_alignment(self):
        """Test alignment of multiple datasets with different shifts."""
        datasets = [
            self.create_mock_dataset_info(
                "reference",
                (-180, -90, 180, 90),
                1.0
            ),
            self.create_mock_dataset_info(
                "shifted_x",
                (-179.25, -90, 180.75, 90),  # 0.75 degree shift in X
                1.0
            ),
            self.create_mock_dataset_info(
                "shifted_y",
                (-180, -90.33, 180, 89.67),  # 0.33 degree shift in Y
                1.0
            ),
            self.create_mock_dataset_info(
                "shifted_both",
                (-179.1, -89.2, 180.9, 90.8),  # Shift in both directions
                1.0
            )
        ]
        
        alignments = self.aligner.calculate_grid_shifts(datasets)
        
        # Check each alignment
        alignment_dict = {a.aligned_dataset: a for a in alignments}
        
        # Reference should have no shift
        self.assertFalse(alignment_dict["reference"].requires_shift)
        
        # shifted_x should have X shift only
        self.assertTrue(alignment_dict["shifted_x"].requires_shift)
        self.assertAlmostEqual(alignment_dict["shifted_x"].x_shift, 0.25)  # To align back to ref
        self.assertAlmostEqual(alignment_dict["shifted_x"].y_shift, 0.0)
        
        # shifted_y should have Y shift only  
        self.assertTrue(alignment_dict["shifted_y"].requires_shift)
        self.assertAlmostEqual(alignment_dict["shifted_y"].x_shift, 0.0)
        self.assertAlmostEqual(alignment_dict["shifted_y"].y_shift, 0.33, places=2)  # To align back
        
        # shifted_both should have both shifts
        self.assertTrue(alignment_dict["shifted_both"].requires_shift)
        self.assertAlmostEqual(alignment_dict["shifted_both"].x_shift, 0.1, places=1)  # To align back
        self.assertAlmostEqual(alignment_dict["shifted_both"].y_shift, 0.2, places=1)  # Positive shift
        
    def test_different_resolutions(self):
        """Test alignment detection with different resolutions."""
        # Different resolution datasets
        datasets = [
            self.create_mock_dataset_info(
                "res_1deg",
                (-180, -90, 180, 90),
                1.0
            ),
            self.create_mock_dataset_info(
                "res_0.5deg",
                (-180, -90, 180, 90),
                0.5
            )
        ]
        
        alignments = self.aligner.calculate_grid_shifts(datasets)
        
        # Should handle different resolutions gracefully
        self.assertEqual(len(alignments), 2)
        
    def test_empty_dataset_list(self):
        """Test handling of empty dataset list."""
        alignments = self.aligner.calculate_grid_shifts([])
        self.assertEqual(alignments, [])
        
    def test_single_dataset(self):
        """Test handling of single dataset."""
        datasets = [
            self.create_mock_dataset_info(
                "single",
                (-180, -90, 180, 90),
                1.0
            )
        ]
        
        alignments = self.aligner.calculate_grid_shifts(datasets)
        self.assertEqual(alignments, [])
        
    def test_create_aligned_coordinate_query(self):
        """Test SQL query generation with alignment shifts."""
        # Create alignment info
        alignment = GridAlignment(
            reference_dataset="reference",
            aligned_dataset="shifted",
            x_shift=0.5,
            y_shift=-0.25,
            requires_shift=True,
            shift_pixels_x=0.5,
            shift_pixels_y=-0.25
        )
        
        # Test query generation
        query = self.aligner.create_aligned_coordinate_query(
            table_name="test_table",
            alignment=alignment,
            chunk_bounds=(-180, -90, 180, 90),
            name_column="test_dataset"
        )
        
        # Check that query includes shift adjustments
        self.assertIn("x_coord + 0.5", query)
        self.assertIn("y_coord + -0.25", query)
        self.assertIn("as x", query.lower())
        self.assertIn("as y", query.lower())
        self.assertIn("test_dataset", query)
        
    def test_create_query_no_shift(self):
        """Test SQL query generation when no shift is needed."""
        # Create alignment info with no shift
        alignment = GridAlignment(
            reference_dataset="reference",
            aligned_dataset="aligned",
            x_shift=0.0,
            y_shift=0.0,
            requires_shift=False,
            shift_pixels_x=0.0,
            shift_pixels_y=0.0
        )
        
        query = self.aligner.create_aligned_coordinate_query(
            table_name="test_table",
            alignment=alignment,
            chunk_bounds=(-180, -90, 180, 90),
            name_column="test_dataset"
        )
        
        # Should select coordinates directly without adjustment
        self.assertIn("x_coord as x", query)
        self.assertIn("y_coord as y", query)
        self.assertNotIn("+", query)  # No addition for shifts


if __name__ == '__main__':
    unittest.main()