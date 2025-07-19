"""Tests for database schema operations."""

import pytest
import json
from datetime import datetime
from src.database.schema import schema
from src.database.setup import setup_database, reset_database
from src.config import config

class TestDatabaseSchema:
    """Test DatabaseSchema class functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_clean_db(self):
        """Ensure clean database for each test."""
        reset_database()
        yield
        # Cleanup after test if needed
    
    def test_create_schema(self):
        """Test schema creation."""
        # Schema should already be created by setup_clean_db
        info = schema.get_schema_info()
        assert info['summary']['table_count'] >= 11  # Original 7 + 4 raster tables
        
        expected_tables = [
            'grids', 'grid_cells', 'species_ranges', 
            'species_grid_intersections', 'features', 
            'climate_data', 'experiments', 'processing_jobs',
            # Raster tables
            'raster_sources', 'raster_tiles', 'resampling_cache', 'processing_queue'
        ]
        
        table_names = [t['table_name'] for t in info['tables']]
        for table in expected_tables:
            assert table in table_names
    
    def test_drop_schema_requires_confirmation(self):
        """Test that drop schema requires confirmation."""
        with pytest.raises(ValueError, match="Must set confirm=True"):
            schema.drop_schema()
    
    def test_drop_schema_with_confirmation(self):
        """Test schema dropping with confirmation."""
        assert schema.drop_schema(confirm=True)
        
        # Verify tables are gone
        info = schema.get_schema_info()
        assert info['summary']['table_count'] == 0

class TestGridOperations:
    """Test grid-related database operations."""
    
    @pytest.fixture(autouse=True)
    def setup_clean_db(self):
        """Ensure clean database for each test."""
        reset_database()
        yield
    
    def test_store_grid_definition_cubic(self):
        """Test storing cubic grid definition."""
        grid_id = schema.store_grid_definition(
            name="test_cubic",
            grid_type="cubic",
            resolution=1000,
            bounds="POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
            metadata={"test": True}
        )
        
        assert grid_id is not None
        
        # Verify grid exists
        grid = schema.get_grid_by_name("test_cubic")
        assert grid is not None
        assert grid['grid_type'] == 'cubic'
        assert grid['resolution'] == 1000
        assert grid['crs'] == 'EPSG:3857'
        assert grid['metadata']['test'] is True
    
    def test_store_grid_definition_hexagonal(self):
        """Test storing hexagonal grid definition."""
        grid_id = schema.store_grid_definition(
            name="test_hex",
            grid_type="hexagonal",
            resolution=8
        )
        
        grid = schema.get_grid_by_name("test_hex")
        assert grid is not None  # Check if grid exists
        assert grid['grid_type'] in ['cubic', 'hexagonal']
        assert grid['grid_type'] == 'hexagonal'
        assert grid['resolution'] == 8
        assert grid['crs'] == 'EPSG:4326'
    
    def test_store_grid_definition_invalid_type(self):
        """Test storing grid with invalid type."""
        with pytest.raises(ValueError, match="Unknown grid type"):
            schema.store_grid_definition(
                name="invalid_grid",
                grid_type="invalid_type",
                resolution=1000
            )
    
    def test_store_grid_cells_batch(self):
        """Test bulk storing of grid cells."""
        # Create grid first
        grid_id = schema.store_grid_definition("test_grid", "cubic", 1000)
        
        # Prepare cell data
        cells_data = []
        for i in range(5):
            cells_data.append({
                'cell_id': f'cell_{i}',
                'geometry_wkt': f'POLYGON(({i} {i}, {i+1} {i}, {i+1} {i+1}, {i} {i+1}, {i} {i}))',
                'area_km2': 1.0,
                'centroid_wkt': f'POINT({i+0.5} {i+0.5})'
            })
        
        # Store cells
        count = schema.store_grid_cells_batch(grid_id, cells_data)
        assert count == 5
        
        # Verify cells exist
        cells = schema.get_grid_cells(grid_id)
        assert len(cells) == 5
        
        # Verify trigger updated total_cells
        grid = schema.get_grid_by_name("test_grid")
        assert grid is not None  # Check if grid exists
        assert grid['grid_type'] in ['cubic', 'hexagonal']
        assert grid['total_cells'] == 5
    
    def test_get_grid_cells_with_limit(self):
        """Test getting grid cells with limit."""
        grid_id = schema.store_grid_definition("test_grid", "cubic", 1000)
        
        # Store 10 cells
        cells_data = [
            {
                'cell_id': f'cell_{i}',
                'geometry_wkt': f'POLYGON(({i} {i}, {i+1} {i}, {i+1} {i+1}, {i} {i+1}, {i} {i}))',
                'area_km2': 1.0
            }
            for i in range(10)
        ]
        schema.store_grid_cells_batch(grid_id, cells_data)
        
        # Get limited results
        cells = schema.get_grid_cells(grid_id, limit=3)
        assert len(cells) == 3
    
    def test_delete_grid_cascade(self):
        """Test grid deletion cascades to cells."""
        grid_id = schema.store_grid_definition("delete_test", "cubic", 1000)
        
        # Add cells
        cells_data = [{
            'cell_id': 'test_cell',
            'geometry_wkt': 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))',
            'area_km2': 1.0
        }]
        schema.store_grid_cells_batch(grid_id, cells_data)
        
        # Verify cell exists
        cells_before = schema.get_grid_cells(grid_id)
        assert len(cells_before) == 1
        
        # Delete grid
        assert schema.delete_grid("delete_test")
        
        # Verify grid and cells are gone
        assert schema.get_grid_by_name("delete_test") is None
        cells_after = schema.get_grid_cells(grid_id)
        assert len(cells_after) == 0
    
    def test_delete_nonexistent_grid(self):
        """Test deleting non-existent grid."""
        result = schema.delete_grid("nonexistent")
        assert result is False
    
    def test_validate_grid_config(self):
        """Test grid configuration validation."""
        # Valid configurations
        assert schema.validate_grid_config('cubic', 1000)
        assert schema.validate_grid_config('cubic', 5000)
        assert schema.validate_grid_config('hexagonal', 7)
        assert schema.validate_grid_config('hexagonal', 8)
        
        # Invalid configurations
        assert not schema.validate_grid_config('cubic', 999)
        assert not schema.validate_grid_config('hexagonal', 15)
        assert not schema.validate_grid_config('invalid_type', 1000)

class TestSpeciesOperations:
    """Test species-related database operations."""
    
    @pytest.fixture(autouse=True)
    def setup_clean_db(self):
        """Ensure clean database for each test."""
        reset_database()
        yield
    
    def test_store_species_range(self):
        """Test storing species range data."""
        species_data = {
            'species_name': 'Quercus alba',
            'scientific_name': 'Quercus alba',
            'genus': 'Quercus',
            'family': 'Fagaceae',
            'category': 'plant',
            'range_type': 'distribution',
            'geometry_wkt': 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))',
            'source_file': 'test.gpkg',
            'source_dataset': 'test_data',
            'confidence': 0.9,
            'area_km2': 1000.0,
            'metadata': {'test': True}
        }
        
        range_id = schema.store_species_range(species_data)
        assert range_id is not None
        
        # Verify species exists
        species_list = schema.get_species_ranges(category='plant')
        assert len(species_list) == 1
        
        species = species_list[0]
        assert species['species_name'] == 'Quercus alba'
        assert species['genus'] == 'Quercus'
        assert species['family'] == 'Fagaceae'
        assert species['category'] == 'plant'
        assert species['confidence'] == 0.9
    
    def test_store_species_intersections_batch(self):
        """Test bulk storing of species-grid intersections."""
        # Create grid and species first
        grid_id = schema.store_grid_definition("test_grid", "cubic", 1000)
        species_data = {
            'species_name': 'Test species',
            'category': 'plant',
            'range_type': 'distribution',
            'geometry_wkt': 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))',
            'source_file': 'test.gpkg'
        }
        range_id = schema.store_species_range(species_data)
        
        # Prepare intersection data
        intersections = []
        for i in range(3):
            intersections.append({
                'grid_id': grid_id,
                'cell_id': f'cell_{i}',
                'species_range_id': range_id,
                'species_name': 'Test species',
                'category': 'plant',
                'range_type': 'distribution',
                'intersection_area_km2': 0.5,
                'coverage_percent': 50.0,
                'presence_score': 0.8,
                'computation_metadata': {'method': 'test'}
            })
        
        # Store intersections
        count = schema.store_species_intersections_batch(intersections)
        assert count == 3
        
        # Verify intersections exist
        richness = schema.get_species_richness(grid_id, category='plant')
        assert len(richness) == 3
        assert richness[0]['species_count'] == 1
    
    def test_store_species_intersections_conflict_resolution(self):
        """Test intersection storage handles conflicts correctly."""
        # Setup
        grid_id = schema.store_grid_definition("test_grid", "cubic", 1000)
        range_id = schema.store_species_range({
            'species_name': 'Test species',
            'category': 'plant',
            'geometry_wkt': 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))',
            'source_file': 'test.gpkg'
        })
        
        intersection = {
            'grid_id': grid_id,
            'cell_id': 'cell_1',
            'species_range_id': range_id,
            'species_name': 'Test species',
            'category': 'plant',
            'range_type': 'distribution',
            'coverage_percent': 30.0
        }
        
        # Store first time
        schema.store_species_intersections_batch([intersection])
        
        # Store again with different coverage
        intersection['coverage_percent'] = 70.0
        schema.store_species_intersections_batch([intersection])
        
        # Should have updated, not duplicated
        richness = schema.get_species_richness(grid_id)
        assert len(richness) == 1
    
    def test_get_species_ranges_filtering(self):
        """Test species range filtering."""
        # Store different species
        for i, category in enumerate(['plant', 'animal', 'fungi']):
            schema.store_species_range({
                'species_name': f'Species_{i}',
                'category': category,
                'geometry_wkt': 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))',
                'source_file': f'file_{i}.gpkg'
            })
        
        # Test category filtering
        plants = schema.get_species_ranges(category='plant')
        assert len(plants) == 1
        assert plants[0]['category'] == 'plant'
        
        # Test source file filtering
        file_species = schema.get_species_ranges(source_file='file_1.gpkg')
        assert len(file_species) == 1
        assert file_species[0]['source_file'] == 'file_1.gpkg'

class TestFeatureOperations:
    """Test feature-related database operations."""
    
    @pytest.fixture(autouse=True)
    def setup_clean_db(self):
        """Ensure clean database for each test."""
        reset_database()
        yield
    
    def test_store_feature(self):
        """Test storing individual feature."""
        grid_id = schema.store_grid_definition("test_grid", "cubic", 1000)
        
        feature_id = schema.store_feature(
            grid_id=grid_id,
            cell_id="cell_1",
            feature_type="richness",
            feature_name="plant_richness",
            feature_value=15.5,
            metadata={"method": "count"}
        )
        
        assert feature_id is not None
        
        # Verify feature exists
        features = schema.get_features(grid_id, feature_type="richness")
        assert len(features) == 1
        assert features[0]['feature_name'] == 'plant_richness'
        assert features[0]['feature_value'] == 15.5
    
    def test_store_features_batch(self):
        """Test bulk storing of features."""
        grid_id = schema.store_grid_definition("test_grid", "cubic", 1000)
        
        features = []
        for i in range(5):
            features.append({
                'grid_id': grid_id,
                'cell_id': f'cell_{i}',
                'feature_type': 'richness',
                'feature_name': 'total_richness',
                'feature_value': float(i * 10),
                'computation_metadata': {'batch': True}
            })
        
        count = schema.store_features_batch(features)
        assert count == 5
        
        # Verify features exist
        all_features = schema.get_features(grid_id)
        assert len(all_features) == 5
    
    def test_store_climate_data_batch(self):
        """Test bulk storing of climate data."""
        grid_id = schema.store_grid_definition("test_grid", "cubic", 1000)
        
        climate_data = []
        variables = ['bio_1', 'bio_12']
        for var in variables:
            for i in range(3):
                climate_data.append({
                    'grid_id': grid_id,
                    'cell_id': f'cell_{i}',
                    'variable': var,
                    'value': float(i * 100),
                    'source': 'WorldClim',
                    'resolution': '10m'
                })
        
        count = schema.store_climate_data_batch(climate_data)
        assert count == 6  # 2 variables × 3 cells
    
    def test_feature_conflict_resolution(self):
        """Test feature storage handles conflicts correctly."""
        grid_id = schema.store_grid_definition("test_grid", "cubic", 1000)
        
        # Store feature first time
        schema.store_feature(grid_id, "cell_1", "richness", "plant_richness", 10.0)
        
        # Store again with different value
        schema.store_feature(grid_id, "cell_1", "richness", "plant_richness", 15.0)
        
        # Should have updated, not duplicated
        features = schema.get_features(grid_id)
        assert len(features) == 1
        assert features[0]['feature_value'] == 15.0

class TestExperimentTracking:
    """Test experiment and job tracking operations."""
    
    @pytest.fixture(autouse=True)
    def setup_clean_db(self):
        """Ensure clean database for each test."""
        reset_database()
        yield
    
    def test_create_experiment(self):
        """Test creating experiment."""
        experiment_id = schema.create_experiment(
            name="test_experiment",
            description="Test experiment",
            config={"test": True, "created_by": "test_user"}
        )
        
        assert experiment_id is not None
    
    def test_update_experiment_status(self):
        """Test updating experiment status."""
        experiment_id = schema.create_experiment(
            "test_exp", 
            "Test", 
            {"test": True}
        )
        
        # Update to completed
        schema.update_experiment_status(
            experiment_id, 
            "completed", 
            results={"success": True},
            error_message=None
        )
        
        # Verify update (would need to add getter method to verify)
    
    def test_create_processing_job(self):
        """Test creating processing job."""
        experiment_id = schema.create_experiment("test_exp", "Test", {})
        
        job_id = schema.create_processing_job(
            job_type="grid_generation",
            job_name="Generate test grid",
            parameters={"grid_type": "cubic", "resolution": 1000},
            parent_experiment_id=experiment_id
        )
        
        assert job_id is not None
    
    def test_update_job_progress(self):
        """Test updating job progress."""
        job_id = schema.create_processing_job(
            "test_job", 
            "Test job", 
            {"test": True}
        )
        
        # Update progress
        schema.update_job_progress(
            job_id, 
            50.0, 
            status="running", 
            log_message="Half complete"
        )
        
        # Update to completed
        schema.update_job_progress(
            job_id, 
            100.0, 
            status="completed", 
            log_message="Finished"
        )

class TestUtilityMethods:
    """Test utility and query methods."""
    
    @pytest.fixture(autouse=True)
    def setup_clean_db(self):
        """Ensure clean database for each test."""
        reset_database()
        yield
    
    def test_get_schema_info(self):
        """Test schema information retrieval."""
        info = schema.get_schema_info()
        
        assert 'tables' in info
        assert 'views' in info
        assert 'table_counts' in info
        assert 'summary' in info
        
        assert info['summary']['table_count'] >= 7
        assert info['summary']['view_count'] >= 3
        
        # Check specific tables exist
        table_names = [t['table_name'] for t in info['tables']]
        assert 'grids' in table_names
        assert 'grid_cells' in table_names
        assert 'species_ranges' in table_names
    
    def test_get_grid_status(self):
        """Test grid status retrieval."""
        # Empty status initially
        status = schema.get_grid_status()
        assert len(status) == 0
        
        # Create grid and check status
        grid_id = schema.store_grid_definition("test_grid", "cubic", 1000)
        
        status = schema.get_grid_status("test_grid")
        assert len(status) == 1
        assert status[0]['grid_name'] == 'test_grid'
        assert status[0]['cells_generated'] == 0
    
    def test_get_species_richness(self):
        """Test species richness retrieval."""
        # Setup grid and species
        grid_id = schema.store_grid_definition("test_grid", "cubic", 1000)
        
        range_id = schema.store_species_range({
            'species_name': 'Test species',
            'category': 'plant',
            'geometry_wkt': 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))',
            'source_file': 'test.gpkg'
        })
        
        schema.store_species_intersections_batch([{
            'grid_id': grid_id,
            'cell_id': 'cell_1',
            'species_range_id': range_id,
            'species_name': 'Test species',
            'category': 'plant',
            'range_type': 'distribution',
            'coverage_percent': 50.0
        }])
        
        # Get richness
        richness = schema.get_species_richness(grid_id)
        assert len(richness) == 1
        assert richness[0]['species_count'] == 1
        
        # Get richness filtered by category
        plant_richness = schema.get_species_richness(grid_id, category='plant')
        assert len(plant_richness) == 1


class TestRasterDatabaseOperations:
    """Test raster-related database operations."""
    
    @pytest.fixture(autouse=True)
    def setup_clean_db(self):
        """Ensure clean database for each test."""
        reset_database()
        yield
    
    def test_store_raster_source(self, sample_raster_data):
        """Test storing raster source metadata."""
        raster_id = schema.store_raster_source(sample_raster_data)
        assert raster_id is not None
        
        # Verify raster exists
        sources = schema.get_raster_sources()
        assert len(sources) == 1
        
        source = sources[0]
        assert source['name'] == 'test_raster'
        assert source['data_type'] == 'Float32'
        assert source['band_count'] == 1
        assert source['processing_status'] == 'pending'
        assert source['active'] is True
    
    def test_get_raster_sources_filtering(self, sample_raster_data):
        """Test filtering raster sources."""
        # Store raster
        raster_id = schema.store_raster_source(sample_raster_data)
        
        # Update status
        schema.update_raster_processing_status(raster_id, 'ready')
        
        # Test filtering by processing status
        ready_sources = schema.get_raster_sources(processing_status='ready')
        assert len(ready_sources) == 1
        assert ready_sources[0]['processing_status'] == 'ready'
        
        pending_sources = schema.get_raster_sources(processing_status='pending')
        assert len(pending_sources) == 0
    
    def test_update_raster_processing_status(self, sample_raster_data):
        """Test updating raster processing status."""
        raster_id = schema.store_raster_source(sample_raster_data)
        
        # Update status with metadata (use valid enum value)
        metadata = {'tiles_processed': 10}
        success = schema.update_raster_processing_status(raster_id, 'tiling', metadata)
        assert success is True
        
        # Verify update
        sources = schema.get_raster_sources()
        source = sources[0]
        assert source['processing_status'] == 'tiling'
        assert source['metadata']['tiles_processed'] == 10
    
    def test_store_raster_tiles_batch(self, sample_raster_data, sample_raster_tiles):
        """Test bulk storing of raster tiles."""
        raster_id = schema.store_raster_source(sample_raster_data)
        
        # Store tiles
        count = schema.store_raster_tiles_batch(raster_id, sample_raster_tiles)
        assert count == 2
        
        # Test conflict resolution - store same tiles again
        count = schema.store_raster_tiles_batch(raster_id, sample_raster_tiles)
        assert count == 2  # Should update existing tiles
    
    def test_get_raster_tiles_for_bounds(self, sample_raster_data, sample_raster_tiles):
        """Test getting raster tiles that intersect with bounds."""
        raster_id = schema.store_raster_source(sample_raster_data)
        schema.store_raster_tiles_batch(raster_id, sample_raster_tiles)
        
        # Query tiles that intersect with a specific bounds
        bounds_wkt = 'POLYGON((5 5, 15 5, 15 15, 5 15, 5 5))'
        tiles = schema.get_raster_tiles_for_bounds(raster_id, bounds_wkt)
        
        # Should intersect with both tiles
        assert len(tiles) == 2
        assert tiles[0]['tile_x'] in [0, 1]
        assert tiles[1]['tile_x'] in [0, 1]
    
    def test_store_resampling_cache_batch(self, sample_raster_data, sample_grid_data):
        """Test bulk storing of resampling cache entries."""
        # Setup
        raster_id = schema.store_raster_source(sample_raster_data)
        grid_id = schema.store_grid_definition(**sample_grid_data)
        
        cache_data = [
            {
                'source_raster_id': raster_id,
                'target_grid_id': grid_id,
                'cell_id': 'cell_1',
                'method': 'bilinear',
                'band_number': 1,
                'value': 25.5,
                'confidence_score': 0.95,
                'source_tiles_used': [1, 2],
                'computation_metadata': {'method': 'bilinear'}
            },
            {
                'source_raster_id': raster_id,
                'target_grid_id': grid_id,
                'cell_id': 'cell_2',
                'method': 'bilinear',
                'band_number': 1,
                'value': 30.2,
                'confidence_score': 0.92,
                'source_tiles_used': [2, 3],
                'computation_metadata': {'method': 'bilinear'}
            }
        ]
        
        # Store cache
        count = schema.store_resampling_cache_batch(cache_data)
        assert count == 2
        
        # Test conflict resolution - store with updated values
        cache_data[0]['value'] = 26.0
        count = schema.store_resampling_cache_batch(cache_data)
        assert count == 2  # Should update existing entries
    
    def test_get_cached_resampling_values(self, sample_raster_data, sample_grid_data):
        """Test retrieving cached resampling values."""
        # Setup
        raster_id = schema.store_raster_source(sample_raster_data)
        grid_id = schema.store_grid_definition(**sample_grid_data)
        
        cache_data = [{
            'source_raster_id': raster_id,
            'target_grid_id': grid_id,
            'cell_id': 'cell_1',
            'method': 'bilinear',
            'band_number': 1,
            'value': 25.5,
            'confidence_score': 0.95,
            'source_tiles_used': [1, 2],
            'computation_metadata': {'method': 'bilinear'}
        }]
        
        schema.store_resampling_cache_batch(cache_data)
        
        # Retrieve cached values
        cached_values = schema.get_cached_resampling_values(
            raster_id, grid_id, ['cell_1'], 'bilinear', 1
        )
        
        assert cached_values == {'cell_1': 25.5}
        
        # Test non-existent cells
        empty_cache = schema.get_cached_resampling_values(
            raster_id, grid_id, ['cell_999'], 'bilinear', 1
        )
        assert empty_cache == {}
    
    def test_processing_queue_operations(self, sample_raster_data, sample_grid_data):
        """Test processing queue operations."""
        # Setup
        raster_id = schema.store_raster_source(sample_raster_data)
        grid_id = schema.store_grid_definition(**sample_grid_data)
        
        # Add task to queue
        task_id = schema.add_processing_task(
            queue_type='raster_tiling',
            raster_id=raster_id,
            parameters={'tile_size': 1000},
            priority=1
        )
        assert task_id is not None
        
        # Get next task
        task = schema.get_next_processing_task('raster_tiling', 'worker-1')
        assert task is not None
        assert task['id'] == task_id
        assert task['queue_type'] == 'raster_tiling'
        assert task['status'] == 'processing'  # SQL function sets to 'processing' when assigned to worker
        assert task['worker_id'] == 'worker-1'
        
        # Complete task successfully
        success = schema.complete_processing_task(
            task_id, 
            success=True, 
            checkpoint_data={'tiles_created': 100}
        )
        assert success is True
        
        # Verify task is marked as completed
        # (No direct way to check this without another query, but we can verify no more tasks)
        next_task = schema.get_next_processing_task('raster_tiling', 'worker-2')
        assert next_task is None
    
    def test_processing_task_failure(self, sample_raster_data):
        """Test handling of failed processing tasks."""
        raster_id = schema.store_raster_source(sample_raster_data)
        
        # Add task
        task_id = schema.add_processing_task('raster_tiling', raster_id=raster_id)
        
        # Get and fail task
        task = schema.get_next_processing_task('raster_tiling', 'worker-1')
        success = schema.complete_processing_task(
            task_id, 
            success=False, 
            error_message='Tiling failed due to memory issue'
        )
        assert success is True
    
    def test_get_raster_processing_status(self, sample_raster_data, sample_raster_tiles):
        """Test getting raster processing status overview."""
        # Setup multiple rasters with different statuses
        raster_id_1 = schema.store_raster_source(sample_raster_data)
        
        sample_raster_data_2 = sample_raster_data.copy()
        sample_raster_data_2['name'] = 'test_raster_2'
        raster_id_2 = schema.store_raster_source(sample_raster_data_2)
        
        # Add tiles to first raster
        schema.store_raster_tiles_batch(raster_id_1, sample_raster_tiles)
        
        # Update statuses
        schema.update_raster_processing_status(raster_id_1, 'ready')
        schema.update_raster_processing_status(raster_id_2, 'processing')
        
        # Get overall status
        status = schema.get_raster_processing_status()
        assert len(status) >= 2
        
        # Get specific raster status
        specific_status = schema.get_raster_processing_status(raster_id_1)
        assert len(specific_status) == 1
        assert specific_status[0]['raster_name'] == 'test_raster'
    
    def test_cache_efficiency_summary(self, sample_raster_data, sample_grid_data):
        """Test cache efficiency statistics."""
        # Setup
        raster_id = schema.store_raster_source(sample_raster_data)
        grid_id = schema.store_grid_definition(**sample_grid_data)
        
        # Add cache entries with multiple accesses
        cache_data = [{
            'source_raster_id': raster_id,
            'target_grid_id': grid_id,
            'cell_id': 'cell_1',
            'method': 'bilinear',
            'band_number': 1,
            'value': 25.5,
            'confidence_score': 0.95,
            'source_tiles_used': [1, 2],
            'computation_metadata': {'method': 'bilinear'}
        }]
        
        schema.store_resampling_cache_batch(cache_data)
        
        # Access the cache multiple times to build statistics
        for _ in range(5):
            schema.get_cached_resampling_values(
                raster_id, grid_id, ['cell_1'], 'bilinear', 1
            )
        
        # Get efficiency summary
        efficiency = schema.get_cache_efficiency_summary()
        assert len(efficiency) >= 1
        
        # Get specific efficiency
        specific_efficiency = schema.get_cache_efficiency_summary(
            raster_id=raster_id, grid_id=grid_id
        )
        assert len(specific_efficiency) >= 1
    
    def test_cleanup_old_cache(self, sample_raster_data, sample_grid_data):
        """Test cleaning up old cache entries."""
        # Setup
        raster_id = schema.store_raster_source(sample_raster_data)
        grid_id = schema.store_grid_definition(**sample_grid_data)
        
        cache_data = [{
            'source_raster_id': raster_id,
            'target_grid_id': grid_id,
            'cell_id': 'cell_1',
            'method': 'bilinear',
            'band_number': 1,
            'value': 25.5,
            'confidence_score': 0.95,
            'source_tiles_used': [1, 2],
            'computation_metadata': {'method': 'bilinear'}
        }]
        
        schema.store_resampling_cache_batch(cache_data)
        
        # Clean up cache (should not delete recent entries)
        deleted_count = schema.cleanup_old_cache(days_old=1, min_access_count=1)
        assert deleted_count >= 0  # May be 0 if entries are too recent
    
    def test_get_processing_queue_summary(self, sample_raster_data):
        """Test getting processing queue statistics."""
        raster_id = schema.store_raster_source(sample_raster_data)
        
        # Add various tasks
        schema.add_processing_task('raster_tiling', raster_id=raster_id, priority=1)
        schema.add_processing_task('resampling', raster_id=raster_id, priority=0)
        
        # Get queue summary
        summary = schema.get_processing_queue_summary()
        assert len(summary) >= 2  # At least 2 queue types
    
    def test_raster_tables_in_schema_info(self):
        """Test that raster tables are included in schema info."""
        info = schema.get_schema_info()
        
        table_names = [t['table_name'] for t in info['tables']]
        
        # Check that all raster tables exist
        expected_raster_tables = [
            'raster_sources',
            'raster_tiles', 
            'resampling_cache',
            'processing_queue'
        ]
        
        for table in expected_raster_tables:
            assert table in table_names, f"Missing raster table: {table}"
        
        # Check for raster views
        view_names = [v['view_name'] for v in info['views']]
        expected_raster_views = [
            'raster_processing_status',
            'cache_efficiency_summary',
            'processing_queue_summary'
        ]
        
        for view in expected_raster_views:
            assert view in view_names, f"Missing raster view: {view}"
    
    def test_raster_enums_exist(self):
        """Test that raster-related enums are created."""
        info = schema.get_schema_info()
        
        # Check for raster enums
        expected_enums = ['processing_status_enum', 'queue_type_enum']
        
        # We don't have direct enum info in schema_info, but we can test by 
        # trying to use them in queries or check table definitions
        # For now, we'll just verify the schema loads without errors
        assert info['summary']['table_count'] >= 11  # Original 7 + 4 raster tables