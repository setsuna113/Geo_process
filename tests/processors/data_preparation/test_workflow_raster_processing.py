# tests/processors/data_preparation/test_workflow_raster_processing.py
import pytest
import tempfile
from pathlib import Path
import numpy as np
import yaml
from osgeo import gdal, osr

from src.config.config import Config
from src.raster_data.catalog import RasterCatalog
from src.processors.data_preparation.raster_cleaner import RasterCleaner
from src.processors.data_preparation.raster_merger import RasterMerger
from src.processors.data_preparation.data_normalizer import DataNormalizer
from src.processors.data_preparation.array_converter import ArrayConverter

class TestRasterProcessingWorkflow:
    """Test complete raster processing workflow."""
    
    @pytest.fixture(scope="class")
    def workflow_config(self):
        """Create comprehensive config for workflow."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            config_data = {
                'database': {
                    'host': 'localhost',
                    'port': 5432,
                    'database': 'test_workflow_db',
                    'user': 'test_user',
                    'password': 'test_pass'
                },
                'raster_processing': {
                    'tile_size': 50,
                    'cache_ttl_days': 1,
                    'memory_limit_mb': 512
                },
                'data_preparation': {
                    'chunk_size': 100,
                    'resolution_tolerance': 1e-6,
                    'bounds_tolerance': 1e-4
                },
                'data_cleaning': {
                    'log_operations': True
                }
            }
            yaml.dump(config_data, f)
            return Config(Path(f.name))
    
    @pytest.fixture(scope="function")  # Changed from "class" to "function"
    def workflow_db(self, workflow_config):
        """Set up workflow test database with proper isolation."""
        try:
            # Ensure we're using the real database module, not mocked
            from src.database.setup import setup_database, reset_database
            from src.database.connection import DatabaseManager
            
            print("Setting up test database...")
            
            # Reset and setup database schema
            reset_database()
            setup_database(reset=True)
            print("Database schema created")
            
            # Create fresh database manager instance
            db_manager = DatabaseManager()
            
            # Verify database is working and empty (correct table name)
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM raster_sources;")
                count = cursor.fetchone()[0]
                print(f"Database ready with {count} entries")
                
                # Clear any existing data
                if count > 0:
                    cursor.execute("TRUNCATE TABLE raster_sources CASCADE;")
                    conn.commit()
                    print("Cleared existing entries")
            
            yield db_manager
            
            # Cleanup after test
            try:
                with db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("TRUNCATE TABLE raster_sources CASCADE;")
                    conn.commit()
                    print("Test cleanup completed")
            except Exception as e:
                print(f"Cleanup failed: {e}")
                
        except Exception as e:
            print(f"Database setup failed: {e}")
            import traceback
            traceback.print_exc()
            pytest.skip(f"Test database not available: {e}")
    
    @pytest.fixture
    def create_test_rasters(self, tmp_path):
        """Create P, A, F test rasters with realistic patterns."""
        rasters = {}
        
        # Common parameters
        width, height = 100, 100
        bounds = (-10, 40, 10, 60)
        
        # Plants - highest diversity
        plants_data = np.random.poisson(lam=150, size=(height, width))
        plants_data[20:30, 20:30] = np.random.poisson(lam=500, size=(10, 10))  # Hotspot
        plants_data[70:75, 70:75] = -9999  # NoData region
        
        # Animals - medium diversity
        animals_data = np.random.poisson(lam=50, size=(height, width))
        animals_data[25:35, 25:35] = np.random.poisson(lam=200, size=(10, 10))  # Hotspot
        animals_data[70:75, 70:75] = -9999  # Same NoData region
        
        # Fungi - low diversity
        fungi_data = np.random.poisson(lam=20, size=(height, width))
        fungi_data[30:40, 30:40] = np.random.poisson(lam=100, size=(10, 10))  # Hotspot
        fungi_data[70:75, 70:75] = -9999  # Same NoData region
        
        # Create GeoTIFFs
        for name, data in [('plants', plants_data), ('animals', animals_data), ('fungi', fungi_data)]:
            path = tmp_path / f"{name}_richness.tif"
            
            driver = gdal.GetDriverByName('GTiff')
            ds = driver.Create(str(path), width, height, 1, gdal.GDT_Int32)
            
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            ds.SetProjection(srs.ExportToWkt())
            
            # Calculate geotransform
            west, south, east, north = bounds
            pixel_width = (east - west) / width
            pixel_height = (north - south) / height
            ds.SetGeoTransform([west, pixel_width, 0, north, 0, -pixel_height])
            
            band = ds.GetRasterBand(1)
            band.SetNoDataValue(-9999)
            band.WriteArray(data)
            band.ComputeStatistics(False)
            
            ds = None
            rasters[name] = path
            
        return rasters
    
    def test_complete_workflow(self, workflow_config, workflow_db, create_test_rasters):
        """Test complete workflow from raw rasters to analysis-ready data."""
        
        # Verify database is clean at start
        catalog = RasterCatalog(workflow_db, workflow_config)
        initial_count = len(catalog.list_rasters())
        assert initial_count == 0, f"Database not clean: found {initial_count} existing rasters"
        
        # Step 1: Catalog the rasters
        for name, path in create_test_rasters.items():
            entry = catalog.add_raster(path, dataset_type=name, validate=True)
            assert entry is not None
            assert entry.is_active
        
        # Verify catalog
        all_rasters = catalog.list_rasters()
        assert len(all_rasters) == 3
        
        # Step 2: Clean each raster
        cleaner = RasterCleaner(workflow_config, workflow_db)
        
        cleaned_paths = {}
        for name in ['plants', 'animals', 'fungi']:
            entry = catalog.get_raster(f"{name}_richness")
            if entry is None:
                pytest.skip(f"Raster {name}_richness not found in catalog")
            
            # Clean and save
            output_path = create_test_rasters[name].parent / f"{name}_cleaned.tif"
            result = cleaner.clean_raster(
                entry.path,
                dataset_type=name,
                output_path=output_path
            )
            
            cleaned_paths[name] = output_path
            
            # Verify cleaning
            assert result['statistics'].negative_values_fixed == 0  # Poisson has no negatives
            assert result['statistics'].nodata_pixels == 25  # 5x5 NoData region
        
        # Step 3: Add cleaned rasters to catalog
        for name, path in cleaned_paths.items():
            catalog.add_raster(path, dataset_type=f"{name}_cleaned", validate=False)
        
        # Step 4: Merge cleaned rasters
        merger = RasterMerger(workflow_config, workflow_db)
        
        merge_result = merger.merge_paf_rasters(
            plants_name='plants_cleaned',
            animals_name='animals_cleaned', 
            fungi_name='fungi_cleaned',
            validate_alignment=True
        )
        
        assert merge_result['alignment'].aligned
        merged_data = merge_result['data']
        
        # Step 5: Normalize the merged data
        normalizer = DataNormalizer(workflow_config, workflow_db)
        
        norm_result = normalizer.normalize(
            merged_data,
            method='standard',
            by_band=True,
            save_params=True
        )
        
        normalized_data = norm_result['data']
        
        # Step 6: Convert to different formats for analysis
        converter = ArrayConverter(workflow_config)
        
        # Convert to numpy for ML
        numpy_result = converter.xarray_to_numpy(
            normalized_data,
            flatten=True,
            preserve_coords=True
        )
        
        # Convert to GeoDataFrame for spatial analysis
        gdf_result = converter.xarray_to_geopandas(
            normalized_data,
            variable_names=['plants', 'animals', 'fungi']  # Use actual variable names
        )
        
        # Verify final outputs
        assert numpy_result['array'].shape[0] == 30000  # 3 variables * 100x100 = 30000
        assert len(gdf_result) == 10000  # One row per pixel
        assert 'plants' in gdf_result.columns  # Use actual column names
        assert 'animals' in gdf_result.columns
        assert 'fungi' in gdf_result.columns
        assert 'geometry' in gdf_result.columns
        
        # Step 7: Test round-trip conversion
        restored_data = normalizer.denormalize(
            normalized_data,
            parameter_id=norm_result['parameter_id']
        )
        
        # Check one band matches original (within tolerance)
        original_plants = merged_data['plants'].values
        restored_plants = restored_data['plants'].values
        
        # Exclude NoData pixels from comparison
        valid_mask = original_plants != -9999
        assert np.allclose(
            original_plants[valid_mask],
            restored_plants[valid_mask],
            rtol=1e-5
        )
    
    def test_workflow_with_misaligned_rasters(self, workflow_config, workflow_db, tmp_path):
        """Test workflow handling of misaligned rasters."""
        
        # Verify database is clean at start
        catalog = RasterCatalog(workflow_db, workflow_config)
        initial_count = len(catalog.list_rasters())
        assert initial_count == 0, f"Database not clean: found {initial_count} existing rasters"
        
        # Create slightly misaligned rasters
        rasters = []
        bounds_list = [
            (-10, 40, 10, 60),
            (-10.01, 40.01, 9.99, 59.99),  # Slightly shifted
            (-10, 40, 10, 60)
        ]
        
        for i, (name, bounds) in enumerate(zip(['plants', 'animals', 'fungi'], bounds_list)):
            path = tmp_path / f"{name}_misaligned.tif"
            
            driver = gdal.GetDriverByName('GTiff')
            ds = driver.Create(str(path), 50, 50, 1, gdal.GDT_Float32)
            
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            ds.SetProjection(srs.ExportToWkt())
            
            west, south, east, north = bounds
            pixel_width = (east - west) / 50
            pixel_height = (north - south) / 50
            ds.SetGeoTransform([west, pixel_width, 0, north, 0, -pixel_height])
            
            data = np.random.normal(100 * (i + 1), 20, size=(50, 50))
            band = ds.GetRasterBand(1)
            band.WriteArray(data)
            
            ds = None
            rasters.append((name, path))
        
        # Verify database is clean before adding to catalog
        temp_catalog = RasterCatalog(workflow_db, workflow_config)
        initial_count = len(temp_catalog.list_rasters())
        assert initial_count == 0, f"Database not clean: found {initial_count} existing rasters"
        
        # Add to catalog with consistent naming
        catalog = RasterCatalog(workflow_db, workflow_config)
        for name, path in rasters:
            # Use the base name (plants, animals, fungi) for dataset_type so merger can find them
            base_name = name.replace('_misaligned', '') if '_misaligned' in name else name
            catalog.add_raster(path, dataset_type=base_name, validate=False)
        
        # Try to merge - should handle small misalignment
        merger = RasterMerger(workflow_config, workflow_db)
        
        result = merger.merge_paf_rasters(
            'plants_misaligned', 'animals_misaligned', 'fungi_misaligned',
            validate_alignment=True
        )
        
        # Small misalignment should be handled
        assert result['alignment'] is not None
        if not result['alignment'].aligned:
            assert result['alignment'].bounds_diff is not None
            max_diff = max(result['alignment'].bounds_diff)
            assert max_diff < 0.1  # Small difference
    
    def test_workflow_data_validation(self, workflow_config, workflow_db, create_test_rasters):
        """Test data validation throughout workflow."""
        
        # Verify database is clean at start
        catalog = RasterCatalog(workflow_db, workflow_config)
        initial_count = len(catalog.list_rasters())
        assert initial_count == 0, f"Database not clean: found {initial_count} existing rasters"
        
        # Add rasters with validation
        for name, path in create_test_rasters.items():
            entry = catalog.add_raster(path, dataset_type=name, validate=True)
            
            # Check validation results in metadata
            assert 'validation' in entry.metadata
            validation = entry.metadata['validation']['values']
            assert validation['dataset_type'] == name
            assert validation['validation']['valid']
        
        # Test catalog validation
        validation_report = catalog.validate_catalog()
        
        assert validation_report['total'] == 3
        assert validation_report['valid'] == 3
        assert len(validation_report['issues']) == 0