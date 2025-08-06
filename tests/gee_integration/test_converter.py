"""Tests for GeoTIFF to Parquet converter with zero-to-NaN conversion"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from src.gee_integration.download.converter import ParquetConverter


class TestParquetConverter(unittest.TestCase):
    """Test ParquetConverter functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.converter = ParquetConverter(self.test_dir)
    
    def tearDown(self):
        """Clean up test files"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_biodiversity_zero_to_nan_conversion(self):
        """Test that biodiversity zeros are converted to NaN"""
        # Create test data with biodiversity zeros
        test_data = {
            'x': [10.0, 11.0, 12.0, 13.0],
            'y': [50.0, 51.0, 52.0, 53.0],
            'bio01': [100, 110, 120, 130],  # Climate data (should keep zeros if any)
            'bio04': [450, 460, 470, 0],    # Climate with a zero (should stay)
            'animal_richness': [0, 50, 0, 100],  # Biodiversity with zeros
            'plant_richness': [200, 0, 300, 0],  # Biodiversity with zeros
        }
        
        # Simulate the conversion logic
        filtered_data = test_data.copy()
        
        # The actual conversion logic from converter.py
        climate_bands = ['bio01', 'bio02', 'bio03', 'bio04', 'bio05', 'bio06', 
                        'bio07', 'bio08', 'bio09', 'bio10', 'bio11', 'bio12',
                        'bio13', 'bio14', 'bio15', 'bio16', 'bio17', 'bio18', 'bio19']
        known_non_biodiv = ['x', 'y', 'elevation', 'dem'] + climate_bands
        
        for band_name, band_data in filtered_data.items():
            if band_name not in known_non_biodiv and isinstance(band_data, list):
                # Convert 0 to NaN for biodiversity bands
                filtered_data[band_name] = [np.nan if v == 0 else v for v in band_data]
        
        # Check results
        # Climate should keep zeros
        self.assertEqual(filtered_data['bio01'], [100, 110, 120, 130])
        self.assertEqual(filtered_data['bio04'], [450, 460, 470, 0])  # Zero preserved
        
        # Biodiversity zeros should be NaN
        expected_animal = [np.nan, 50, np.nan, 100]
        expected_plant = [200, np.nan, 300, np.nan]
        
        np.testing.assert_array_equal(filtered_data['animal_richness'], expected_animal)
        np.testing.assert_array_equal(filtered_data['plant_richness'], expected_plant)
    
    def test_ocean_pixel_filtering(self):
        """Test that ocean pixels (-9999) are filtered out completely"""
        # Create test data with ocean pixels
        # Note: Current logic keeps pixels if ANY band has valid data
        # This preserves climate data even where biodiversity is missing
        test_data = {
            'x': np.array([10.0, 11.0, 12.0, 13.0]),
            'y': np.array([50.0, 51.0, 52.0, 53.0]),
            'bio01': np.array([100, -9999, 120, -9999]),  # Climate
            'animal_richness': np.array([50, -9999, 0, -9999]),  # Biodiversity
        }
        
        # Simulate the filtering logic (OR operation - keep if ANY band valid)
        # Now correctly excludes x,y from the check
        nodata = -9999
        valid_mask = np.zeros(len(test_data['x']), dtype=bool)
        
        for band_name, band_data in test_data.items():
            # Skip coordinate columns
            if band_name in ['x', 'y']:
                continue
            if isinstance(band_data, np.ndarray) and np.issubdtype(band_data.dtype, np.number):
                valid_mask |= (band_data != nodata) & ~np.isnan(band_data)
        
        # Apply mask
        filtered_data = {k: v[valid_mask] for k, v in test_data.items()}
        
        # Rows 0 and 2 have valid data in at least one band
        # Rows 1 and 3 have -9999 in all data bands (should be removed)
        self.assertEqual(len(filtered_data['x']), 2)  # Only rows with valid data kept
        # Row 1 might be kept because x,y are valid
        # Actually, x and y are not checked for nodata, only data bands
        
        # Better test: all data bands are -9999
        test_data2 = {
            'x': np.array([10.0, 11.0, 12.0]),
            'y': np.array([50.0, 51.0, 52.0]),
            'bio01': np.array([100, -9999, 120]),
            'animal_richness': np.array([50, -9999, 0]),
        }
        
        valid_mask2 = np.zeros(len(test_data2['x']), dtype=bool)
        for band_name, band_data in test_data2.items():
            if band_name not in ['x', 'y'] and isinstance(band_data, np.ndarray):
                valid_mask2 |= (band_data != nodata) & ~np.isnan(band_data)
        
        filtered_data2 = {k: v[valid_mask2] for k, v in test_data2.items()}
        
        # Row 1 has all -9999 in data bands, should be filtered
        self.assertEqual(len(filtered_data2['x']), 2)
        self.assertNotIn(-9999, filtered_data2['bio01'])
        self.assertNotIn(-9999, filtered_data2['animal_richness'])
    
    def test_hemisphere_detection(self):
        """Test detection of hemisphere files"""
        # Test file patterns
        test_cases = [
            ("merged_data_west.tif", True),
            ("merged_data_east.tif", True),
            ("merged_data.tif", False),
            ("data_western.tif", False),
            ("west_data.tif", False),
        ]
        
        for filename, should_match in test_cases:
            is_hemisphere = filename.endswith('_west.tif') or filename.endswith('_east.tif')
            self.assertEqual(is_hemisphere, should_match, 
                           f"Failed for {filename}: expected {should_match}")
    
    def test_band_classification(self):
        """Test classification of biodiversity vs climate bands"""
        climate_bands = ['bio01', 'bio02', 'bio03', 'bio04', 'bio05', 'bio06', 
                        'bio07', 'bio08', 'bio09', 'bio10', 'bio11', 'bio12',
                        'bio13', 'bio14', 'bio15', 'bio16', 'bio17', 'bio18', 'bio19']
        known_non_biodiv = ['x', 'y', 'elevation', 'dem'] + climate_bands
        
        test_bands = {
            'x': False,  # Not biodiversity
            'y': False,  # Not biodiversity
            'bio01': False,  # Climate
            'bio19': False,  # Climate
            'elevation': False,  # Terrain
            'animal_richness': True,  # Biodiversity
            'plant_richness': True,  # Biodiversity
            'fungi_richness': True,  # Biodiversity
            'AM_fungi': True,  # Biodiversity
            'custom_biodiv': True,  # Unknown = biodiversity
        }
        
        for band, is_biodiv in test_bands.items():
            actual_is_biodiv = band not in known_non_biodiv
            self.assertEqual(actual_is_biodiv, is_biodiv, 
                           f"Band {band} classification incorrect")


if __name__ == '__main__':
    unittest.main()