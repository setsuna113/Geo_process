# tests/raster_data/test_integration.py
import pytest
from pathlib import Path
import numpy as np
import psutil
import gc

from src.domain.raster.catalog import RasterCatalog
from src.domain.raster.loaders.geotiff_loader import GeoTIFFLoader
from src.grid_systems.grid_factory import GridFactory
# from src.processors.richness_processor import RichnessProcessor  # TODO: Implement in Phase 5

class TestRasterDataIntegration:
    """Integration tests for raster data module."""
    
    def test_end_to_end_workflow(self, real_config, test_db, raster_helper, test_data_dir):
        """Test complete workflow from catalog to processing."""
        # Step 1: Create small test rasters for quick validation
        plant_raster = test_data_dir / "plants.tif"
        vert_raster = test_data_dir / "vertebrates.tif"
        
        # Use small dimensions for fast testing
        raster_helper.create_test_raster(
            plant_raster,
            pattern="hotspots",
            width=50,
            height=50
        )
        
        raster_helper.create_test_raster(
            vert_raster,
            pattern="gradient",
            width=50,
            height=50
        )
        
        # Step 2: Catalog rasters
        catalog = RasterCatalog(test_db, real_config)
        plant_entry = catalog.add_raster(plant_raster, "plants", validate=True)
        vert_entry = catalog.add_raster(vert_raster, "vertebrates", validate=True)
        
        # Step 3: Create small grid for quick testing
        factory = GridFactory()
        spec = {
            'grid_type': 'cubic',
            'resolution': 5000.0,  # Large resolution = fewer cells
            'bounds': (-1, 49, 1, 51),  # Small bounds for minimal grid
            'crs': 'EPSG:4326'
        }
        grid = factory.create_grid(spec)
        grid.generate_grid()
        
        # TODO: Step 4 & 5 - Processing will be implemented in Phase 5
        # processor = RichnessProcessor(real_config)
        # plant_result = processor.process_raster(plant_entry.path, grid)
        # vert_result = processor.process_raster(vert_entry.path, grid)
        
        # Basic validation that catalog and grid work
        assert plant_entry.is_active
        assert vert_entry.is_active
        assert len(grid.get_cells()) > 0
        
        # Skip the rest - requires RichnessProcessor from Phase 5
        pytest.skip("RichnessProcessor not implemented yet - Phase 5")
    
    @pytest.mark.skip(reason="RichnessProcessor not implemented yet - Phase 5")
    def test_multi_resolution_processing(self, real_config, sample_raster):
        """Test processing at multiple resolutions."""
        # Skip this test - requires RichnessProcessor from Phase 5
        pass
        
    def test_memory_efficiency_large_processing(self, real_config, large_raster):
        """Test memory efficiency with large raster."""
        gc.collect()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        loader = GeoTIFFLoader(real_config)
        loader.tile_size = 100  # Small tiles
        
        # Process in tiles
        tile_count = 0
        max_memory = initial_memory
        
        for window, data in loader.iter_tiles(large_raster):
            # Simulate processing
            result = np.mean(data)
            
            current_memory = process.memory_info().rss / 1024 / 1024
            max_memory = max(max_memory, current_memory)
            
            tile_count += 1
            
            # Force garbage collection periodically
            if tile_count % 10 == 0:
                gc.collect()
        
        memory_increase = max_memory - initial_memory
        
        # Should process 1 tile (100/100)^2 = 1
        assert tile_count == 1
        
        # Memory increase should be modest (not loading full raster)
        assert memory_increase < 10  # Less than 10MB for 100x100 raster
        
    def test_concurrent_raster_access(self, real_config, sample_raster):
        """Test concurrent access to rasters."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        loader = GeoTIFFLoader(real_config)
        metadata = loader.extract_metadata(sample_raster)
        
        def read_random_point(idx):
            """Read a random point from raster."""
            x = np.random.uniform(metadata.bounds[0], metadata.bounds[2])
            y = np.random.uniform(metadata.bounds[1], metadata.bounds[3])
            
            with loader.open_lazy(sample_raster) as reader:
                value = reader.read_point(x, y)
                
            return idx, value
        
        # Read 100 random points concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(read_random_point, i) for i in range(100)]
            
            results = {}
            for future in as_completed(futures):
                idx, value = future.result()
                results[idx] = value
        
        # Should have read all points
        assert len(results) == 100
        
        # Most should have valid values (some might hit nodata)
        valid_count = sum(1 for v in results.values() if v is not None)
        assert valid_count > 90
        
    def test_error_recovery(self, real_config, test_db, test_data_dir):
        """Test error handling and recovery."""
        catalog = RasterCatalog(test_db, real_config)
        loader = GeoTIFFLoader(real_config)
        
        # Test handling of corrupted file
        bad_file = test_data_dir / "corrupted.tif"
        bad_file.write_text("This is not a valid GeoTIFF")
        
        # Should not crash on bad file
        with pytest.raises(ValueError):
            loader.extract_metadata(bad_file)
        
        # Catalog should handle gracefully
        entries = catalog.scan_directory(test_data_dir, pattern="corrupted.tif")
        assert len(entries) == 0  # Should skip bad file
        
        # Test handling of missing file
        missing = Path("/non/existent/file.tif")
        with pytest.raises(ValueError):
            loader.extract_metadata(missing)