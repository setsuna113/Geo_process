# tests/raster_data/test_raster_catalog.py
import pytest
from pathlib import Path
import json

from src.domain.raster.catalog import RasterCatalog, RasterEntry

class TestRasterCatalog:
    """Test raster catalog functionality."""
    
    @pytest.fixture
    def catalog(self, test_db, real_config):
        """Use real test database and config instead of mock."""
        return RasterCatalog(test_db, real_config)
    
    @pytest.fixture
    def mock_catalog_with_db(self, test_db, test_config_file):
        """Create catalog with real test database if available."""
        from src.config.config import Config
        config = Config(test_config_file)
        return RasterCatalog(test_db, config)
    
    def test_add_raster(self, catalog, sample_raster):
        entry = catalog.add_raster(sample_raster, dataset_type="test", validate=False)
        
        assert isinstance(entry, RasterEntry)
        assert entry.name == "test_raster"
        assert entry.path == sample_raster
        assert entry.dataset_type == "test"
        assert entry.is_active == True
        assert entry.id  # Check that UUID is not empty (changed from > 0)
        
    def test_scan_directory(self, catalog, raster_helper, test_data_dir):
        # Create multiple test rasters
        for i in range(3):
            raster_helper.create_test_raster(
                test_data_dir / f"scan_test_{i}.tif",
                pattern="random"
            )
        
        entries = catalog.scan_directory(test_data_dir, pattern="scan_test_*.tif")
        
        assert len(entries) == 3
        assert all(isinstance(e, RasterEntry) for e in entries)
        assert all(e.name.startswith("scan_test_") for e in entries)
        
    def test_get_raster(self, catalog, sample_raster):
        # Add raster first
        added = catalog.add_raster(sample_raster, validate=False)
        
        # Retrieve by name
        retrieved = catalog.get_raster(added.name)
        
        assert retrieved is not None
        assert retrieved.id == added.id
        assert retrieved.name == added.name
        
        # Try non-existent
        assert catalog.get_raster("non_existent") is None
        
    def test_list_rasters(self, catalog, raster_helper, test_data_dir):
        # Add multiple rasters
        for i, dtype in enumerate(["plants", "vertebrates", "unknown"]):
            path = test_data_dir / f"{dtype}_{i}.tif"
            raster_helper.create_test_raster(path)
            catalog.add_raster(path, dataset_type=dtype, validate=False)
        
        # List all
        all_entries = catalog.list_rasters()
        assert len(all_entries) >= 3
        
        # List by type
        plant_entries = catalog.list_rasters(dataset_type="plants")
        assert all(e.dataset_type == "plants" for e in plant_entries)
        
    def test_validate_catalog(self, catalog, sample_raster, test_data_dir, raster_helper):
        # Add valid raster
        catalog.add_raster(sample_raster, validate=False)
        
        # Create a real raster that we can then delete to test missing file detection
        fake_raster = test_data_dir / "fake.tif"
        raster_helper.create_test_raster(fake_raster, width=50, height=50)
        
        # Add entry while file exists
        fake_entry = catalog.add_raster(fake_raster, validate=False)
        
        # Now delete the file to simulate missing file
        fake_raster.unlink()
        
        # Validate catalog
        results = catalog.validate_catalog(fix_issues=False)
        
        assert results['total'] >= 2
        assert len(results['issues']) >= 1
        assert any(issue['type'] == 'missing_file' for issue in results['issues'])
        
    def test_deactivate_raster(self, catalog, sample_raster):
        entry = catalog.add_raster(sample_raster, validate=False)
        
        # Deactivate
        catalog.deactivate_raster(entry.id)
        
        # Should not appear in active list
        active = catalog.list_rasters(active_only=True)
        assert not any(e.id == entry.id for e in active)
        
    def test_generate_report(self, catalog, sample_raster, tmp_path):
        # Add some rasters
        catalog.add_raster(sample_raster, dataset_type="plants", validate=False)
        
        report_path = tmp_path / "catalog_report.json"
        catalog.generate_report(report_path)
        
        assert report_path.exists()
        
        with open(report_path) as f:
            report = json.load(f)
            
        assert 'total_rasters' in report
        assert 'by_type' in report
        assert 'rasters' in report
        assert report['total_rasters'] >= 1