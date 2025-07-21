# tests/processors/data_preparation/test_raster_merger.py
import pytest
import numpy as np
import xarray as xr
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.processors.data_preparation.raster_merger import RasterMerger, AlignmentCheck
from src.raster_data.catalog import RasterEntry

class TestAlignmentCheck:
    """Test AlignmentCheck dataclass."""
    
    def test_aligned_check(self):
        check = AlignmentCheck(
            aligned=True,
            same_resolution=True,
            same_bounds=True,
            same_crs=True,
            resolution_diff=None,
            bounds_diff=None,
            crs_mismatch=None
        )
        assert check.aligned == True
    
    def test_misaligned_check(self):
        check = AlignmentCheck(
            aligned=False,
            same_resolution=False,
            same_bounds=True,
            same_crs=True,
            resolution_diff=0.01,
            bounds_diff=None,
            crs_mismatch=None
        )
        assert check.aligned == False
        assert check.resolution_diff == 0.01

class TestRasterMerger:
    """Test RasterMerger with mocked dependencies."""
    
    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.get.return_value = {
            'resolution_tolerance': 1e-6,
            'bounds_tolerance': 1e-4
        }
        return config
    
    @pytest.fixture
    def mock_db(self):
        db = Mock()
        mock_conn = MagicMock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=None)
        db.get_connection.return_value = mock_conn
        return db
    
    @pytest.fixture
    def mock_catalog(self):
        catalog = Mock()
        
        # Create mock raster entries
        plants_entry = Mock(spec=RasterEntry)
        plants_entry.id = 1
        plants_entry.path = Path("plants.tif")
        plants_entry.name = "plants"
        
        animals_entry = Mock(spec=RasterEntry)
        animals_entry.id = 2
        animals_entry.path = Path("animals.tif")
        animals_entry.name = "animals"
        
        fungi_entry = Mock(spec=RasterEntry)
        fungi_entry.id = 3
        fungi_entry.path = Path("fungi.tif")
        fungi_entry.name = "fungi"
        
        catalog.get_raster.side_effect = lambda name: {
            'plants': plants_entry,
            'animals': animals_entry,
            'fungi': fungi_entry
        }.get(name)
        
        return catalog
    
    @pytest.fixture
    def merger(self, mock_config, mock_db, mock_catalog):
        with patch('src.processors.data_preparation.raster_merger.RasterCatalog', return_value=mock_catalog):
            with patch('src.processors.data_preparation.raster_merger.GeoTIFFLoader'):
                merger = RasterMerger(mock_config, mock_db)
                merger.catalog = mock_catalog
                return merger
    
    def test_initialization(self, merger):
        assert merger.resolution_tolerance == 1e-6
        assert merger.bounds_tolerance == 1e-4
    
    def test_load_raster_entries(self, merger):
        raster_names = {
            'plants': 'plants',
            'animals': 'animals',
            'fungi': 'fungi'
        }
        
        entries = merger._load_raster_entries(raster_names)
        
        assert len(entries) == 3
        assert 'plants' in entries
        assert entries['plants'].name == 'plants'
    
    def test_load_raster_entries_missing(self, merger):
        merger.catalog.get_raster.return_value = None
        
        with pytest.raises(ValueError, match="not found in catalog"):
            merger._load_raster_entries({'missing': 'nonexistent'})
    
    def test_check_alignment_single_raster(self, merger):
        entries = [Mock(spec=RasterEntry, path=Path("test.tif"))]
        
        check = merger._check_alignment(entries)
        
        assert check.aligned == True
        assert check.same_resolution == True
    
    def test_check_alignment_mismatched_resolution(self, merger):
        # Create mock metadata with different resolutions
        mock_metadata1 = Mock()
        mock_metadata1.resolution_degrees = 0.1
        mock_metadata1.bounds = (-10, 40, 10, 60)
        mock_metadata1.crs = "EPSG:4326"
        
        mock_metadata2 = Mock()
        mock_metadata2.resolution_degrees = 0.2  # Different resolution
        mock_metadata2.bounds = (-10, 40, 10, 60)
        mock_metadata2.crs = "EPSG:4326"
        
        merger.loader.extract_metadata.side_effect = [mock_metadata1, mock_metadata2]
        
        entries = [
            Mock(spec=RasterEntry, path=Path("raster1.tif")),
            Mock(spec=RasterEntry, path=Path("raster2.tif"))
        ]
        
        check = merger._check_alignment(entries)
        
        assert check.aligned == False
        assert check.same_resolution == False
        assert check.resolution_diff == 0.1
    
    def test_can_fix_alignment_small_bounds_diff(self, merger):
        alignment = AlignmentCheck(
            aligned=False,
            same_resolution=True,
            same_bounds=False,
            same_crs=True,
            resolution_diff=None,
            bounds_diff=(0.01, 0.01, 0.01, 0.01),  # Small differences
            crs_mismatch=None
        )
        
        assert merger._can_fix_alignment(alignment) == True
    
    def test_can_fix_alignment_large_bounds_diff(self, merger):
        alignment = AlignmentCheck(
            aligned=False,
            same_resolution=True,
            same_bounds=False,
            same_crs=True,
            resolution_diff=None,
            bounds_diff=(0.5, 0.5, 0.5, 0.5),  # Large differences
            crs_mismatch=None
        )
        
        assert merger._can_fix_alignment(alignment) == False
    
    @patch('src.processors.data_preparation.raster_merger.xr')
    def test_merge_raster_data(self, mock_xr, merger):
        # Create mock data arrays
        mock_da1 = Mock(spec=xr.DataArray)
        mock_da1.coords = Mock()
        mock_da1.attrs = {'crs': 'EPSG:4326'}
        
        mock_da2 = Mock(spec=xr.DataArray)
        mock_da2.coords = Mock()
        mock_da2.coords.equals.return_value = True
        
        # Mock loading function
        merger._load_as_xarray = Mock(side_effect=[mock_da1, mock_da2])
        
        # Mock xarray Dataset creation
        mock_dataset = Mock(spec=xr.Dataset)
        mock_dataset.attrs = {}
        mock_xr.Dataset.return_value = mock_dataset
        
        # Test merging
        rasters = {
            'band1': Mock(path=Path("band1.tif")),
            'band2': Mock(path=Path("band2.tif"))
        }
        
        result = merger._merge_raster_data(rasters)
        
        assert merger._load_as_xarray.call_count == 2
        assert mock_xr.Dataset.called