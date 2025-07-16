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
        assert info['summary']['table_count'] >= 7
        
        expected_tables = [
            'grids', 'grid_cells', 'species_ranges', 
            'species_grid_intersections', 'features', 
            'climate_data', 'experiments', 'processing_jobs'
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
        assert json.loads(grid['metadata'])['test'] is True
    
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
        assert len(richness) == 1
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