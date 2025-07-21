# tests/raster_data/test_metadata_extractor.py
import pytest
import json
from datetime import datetime

from src.raster_data.loaders.metadata_extractor import RasterMetadataExtractor


class TestRasterMetadataExtractor:
    """Test metadata extraction functionality."""
    
    def test_extract_file_info(self, test_db, sample_raster):
        extractor = RasterMetadataExtractor(test_db)
        file_info = extractor._extract_file_info(sample_raster)
        
        assert file_info['name'] == 'test_raster.tif'
        assert file_info['path'] == str(sample_raster)
        assert file_info['size_mb'] > 0
        assert file_info['format'] == '.tif'
        assert 'modified' in file_info
        
    def test_extract_full_metadata(self, test_db, sample_raster):
        extractor = RasterMetadataExtractor(test_db)
        metadata = extractor.extract_full_metadata(sample_raster)
        
        # Check all sections present
        assert 'file_info' in metadata
        assert 'spatial_info' in metadata
        assert 'data_info' in metadata
        assert 'statistics' in metadata
        assert 'pyramid_info' in metadata
        
        # Check spatial info
        spatial = metadata['spatial_info']
        assert spatial['dimensions']['width'] == 200
        assert spatial['dimensions']['height'] == 200
        assert 'crs' in spatial
        assert 'extent' in spatial
        assert 'corners' in spatial
        
        # Check data info
        data_info = metadata['data_info']
        assert len(data_info['bands']) == 1
        assert data_info['bands'][0]['data_type'] in ['Int32', 'Float32']
        
    def test_compute_sample_statistics(self, test_db, sample_raster):
        from osgeo import gdal
        
        extractor = RasterMetadataExtractor(test_db)
        dataset = gdal.Open(str(sample_raster))
        band = dataset.GetRasterBand(1)
        
        stats = extractor._compute_sample_statistics(band, sample_size=1000)
        
        assert stats[0] is not None  # min
        assert stats[1] is not None  # max
        assert stats[2] is not None  # mean
        assert stats[3] is not None  # std
        assert stats[0] <= stats[2] <= stats[1]  # min <= mean <= max
        
        dataset = None
        
    def test_store_in_database(self, test_db, sample_raster):
        extractor = RasterMetadataExtractor(test_db)
        metadata = extractor.extract_full_metadata(sample_raster)
        
        # Store in database
        raster_id = extractor.store_in_database(metadata, "test_raster")
        
        assert isinstance(raster_id, str)
        assert raster_id  # Check that UUID is not empty
        
        # Verify stored correctly
        with test_db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name, file_path FROM raster_sources WHERE id = %s", (raster_id,))
            row = cur.fetchone()
            
            assert row[0] == "test_raster"
            assert row[1] == str(sample_raster)