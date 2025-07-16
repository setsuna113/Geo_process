"""Tests for data integrity, constraints, and triggers."""

import pytest
from src.database.schema import schema
from src.database.connection import db
from src.database.setup import reset_database

class TestConstraints:
    """Test database constraints and validation."""
    
    @pytest.fixture(autouse=True)
    def setup_clean_db(self):
        """Ensure clean database for each test."""
        reset_database()
        yield
    
    def test_grid_unique_name_constraint(self):
        """Test that grid names must be unique."""
        schema.store_grid_definition("unique_grid", "cubic", 1000)
        
        with pytest.raises(Exception):  # Should violate unique constraint
            schema.store_grid_definition("unique_grid", "hexagonal", 8)
    
    def test_grid_type_check_constraint(self):
        """Test grid type check constraint."""
        with pytest.raises(Exception):
            with db.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO grids (name, grid_type, resolution, crs)
                    VALUES ('invalid_grid', 'invalid_type', 1000, 'EPSG:4326')
                """)
    
    def test_grid_crs_check_constraint(self):
        """Test CRS check constraint."""
        with pytest.raises(Exception):
            with db.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO grids (name, grid_type, resolution, crs)
                    VALUES ('invalid_crs_grid', 'cubic', 1000, 'EPSG:9999')
                """)
    
    def test_species_category_check_constraint(self):
        """Test species category check constraint."""
        with pytest.raises(Exception):
            with db.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO species_ranges (species_name, category, geometry, source_file)
                    VALUES ('Test', 'invalid_category', ST_Point(0, 0), 'test.gpkg')
                """)
    
    def test_species_confidence_check_constraint(self):
        """Test species confidence check constraint."""
        with pytest.raises(Exception):
            with db.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO species_ranges (species_name, confidence, geometry, source_file)
                    VALUES ('Test', 2.0, ST_Point(0, 0), 'test.gpkg')
                """)
        
        with pytest.raises(Exception):
            with db.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO species_ranges (species_name, confidence, geometry, source_file)
                    VALUES ('Test', -0.5, ST_Point(0, 0), 'test.gpkg')
                """)
    
    def test_coverage_percent_check_constraint(self):
        """Test coverage percentage check constraint."""
        grid_id = schema.store_grid_definition("test_grid", "cubic", 1000)
        range_id = schema.store_species_range({
            'species_name': 'Test',
            'geometry_wkt': 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))',
            'source_file': 'test.gpkg'
        })
        
        with pytest.raises(Exception):
            with db.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO species_grid_intersections 
                    (grid_id, cell_id, species_range_id, species_name, category, 
                     range_type, coverage_percent)
                    VALUES (%s, 'cell_1', %s, 'Test', 'unknown', 'distribution', 150.0)
                """, (grid_id, range_id))
    
    def test_experiment_status_check_constraint(self):
        """Test experiment status check constraint."""
        with pytest.raises(Exception):
            with db.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO experiments (name, config, status)
                    VALUES ('test', '{}', 'invalid_status')
                """)
    
    def test_job_progress_check_constraint(self):
        """Test job progress check constraint."""
        with pytest.raises(Exception):
            with db.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO processing_jobs (job_type, parameters, progress_percent)
                    VALUES ('test', '{}', 150.0)
                """)

class TestForeignKeyConstraints:
    """Test foreign key constraint behavior."""
    
    @pytest.fixture(autouse=True)
    def setup_clean_db(self):
        """Ensure clean database for each test."""
        reset_database()
        yield
    
    def test_grid_cells_cascade_delete(self):
        """Test that deleting grid cascades to grid cells."""
        grid_id = schema.store_grid_definition("cascade_test", "cubic", 1000)
        
        # Add grid cells
        cells_data = [{
            'cell_id': 'test_cell',
            'geometry_wkt': 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))',
            'area_km2': 1.0
        }]
        schema.store_grid_cells_batch(grid_id, cells_data)
        
        # Verify cell exists
        with db.get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM grid_cells WHERE grid_id = %s", (grid_id,))
            count_before = cursor.fetchone()['count']
            assert count_before == 1
        
        # Delete grid
        schema.delete_grid("cascade_test")
        
        # Verify cell is deleted
        with db.get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM grid_cells WHERE grid_id = %s", (grid_id,))
            count_after = cursor.fetchone()['count']
            assert count_after == 0
    
    def test_species_intersections_cascade_delete(self):
        """Test that deleting species range cascades to intersections."""
        grid_id = schema.store_grid_definition("test_grid", "cubic", 1000)
        range_id = schema.store_species_range({
            'species_name': 'Test species',
            'geometry_wkt': 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))',
            'source_file': 'test.gpkg'
        })
        
        # Add intersection
        schema.store_species_intersections_batch([{
            'grid_id': grid_id,
            'cell_id': 'cell_1',
            'species_range_id': range_id,
            'species_name': 'Test species',
            'category': 'unknown',
            'range_type': 'distribution'
        }])
        
        # Verify intersection exists
        with db.get_cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) as count FROM species_grid_intersections 
                WHERE species_range_id = %s
            """, (range_id,))
            count_before = cursor.fetchone()['count']
            assert count_before == 1
        
        # Delete species range
        with db.get_cursor() as cursor:
            cursor.execute("DELETE FROM species_ranges WHERE id = %s", (range_id,))
        
        # Verify intersection is deleted
        with db.get_cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) as count FROM species_grid_intersections 
                WHERE species_range_id = %s
            """, (range_id,))
            count_after = cursor.fetchone()['count']
            assert count_after == 0
    
    def test_features_cascade_delete(self):
        """Test that deleting grid cascades to features."""
        grid_id = schema.store_grid_definition("test_grid", "cubic", 1000)
        
        # Add feature
        schema.store_feature(grid_id, "cell_1", "richness", "plant_richness", 10.0)
        
        # Verify feature exists
        with db.get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM features WHERE grid_id = %s", (grid_id,))
            count_before = cursor.fetchone()['count']
            assert count_before == 1
        
        # Delete grid
        schema.delete_grid("test_grid")
        
        # Verify feature is deleted
        with db.get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM features WHERE grid_id = %s", (grid_id,))
            count_after = cursor.fetchone()['count']
            assert count_after == 0

class TestTriggers:
    """Test database triggers."""
    
    @pytest.fixture(autouse=True)
    def setup_clean_db(self):
        """Ensure clean database for each test."""
        reset_database()
        yield
    
    def test_grid_cell_count_trigger_insert(self):
        """Test that inserting grid cells updates total_cells."""
        grid_id = schema.store_grid_definition("trigger_test", "cubic", 1000)
        
        # Initially should be 0
        grid = schema.get_grid_by_name("trigger_test")
        assert grid is not None  # Check if grid exists
        assert grid['grid_type'] in ['cubic', 'hexagonal']
        assert grid['total_cells'] == 0
        
        # Add cells
        cells_data = [
            {
                'cell_id': f'cell_{i}',
                'geometry_wkt': f'POLYGON(({i} {i}, {i+1} {i}, {i+1} {i+1}, {i} {i+1}, {i} {i}))',
                'area_km2': 1.0
            }
            for i in range(3)
        ]
        schema.store_grid_cells_batch(grid_id, cells_data)
        
        # Should update total_cells
        grid = schema.get_grid_by_name("trigger_test")
        assert grid is not None  # Check if grid exists
        assert grid['grid_type'] in ['cubic', 'hexagonal']
        assert grid['total_cells'] == 3
    
    def test_grid_cell_count_trigger_delete(self):
        """Test that deleting grid cells updates total_cells."""
        grid_id = schema.store_grid_definition("trigger_test", "cubic", 1000)
        
        # Add cells
        cells_data = [
            {
                'cell_id': f'cell_{i}',
                'geometry_wkt': f'POLYGON(({i} {i}, {i+1} {i}, {i+1} {i+1}, {i} {i+1}, {i} {i}))',
                'area_km2': 1.0
            }
            for i in range(5)
        ]
        schema.store_grid_cells_batch(grid_id, cells_data)
        
        # Verify count
        grid = schema.get_grid_by_name("trigger_test")
        assert grid is not None  # Check if grid exists
        assert grid['grid_type'] in ['cubic', 'hexagonal']
        assert grid['total_cells'] == 5
        
        # Delete some cells
        with db.get_cursor() as cursor:
            cursor.execute("""
                DELETE FROM grid_cells 
                WHERE grid_id = %s AND cell_id IN ('cell_0', 'cell_1')
            """, (grid_id,))
        
        # Should update total_cells
        grid = schema.get_grid_by_name("trigger_test")
        assert grid is not None  # Check if grid exists
        assert grid['grid_type'] in ['cubic', 'hexagonal']
        assert grid['total_cells'] == 3

class TestUniqueConstraints:
    """Test unique constraint behavior."""
    
    @pytest.fixture(autouse=True)
    def setup_clean_db(self):
        """Ensure clean database for each test."""
        reset_database()
        yield
    
    def test_grid_cell_unique_constraint(self):
        """Test unique constraint on grid_id, cell_id."""
        grid_id = schema.store_grid_definition("test_grid", "cubic", 1000)
        
        # Add cell
        cells_data = [{
            'cell_id': 'duplicate_cell',
            'geometry_wkt': 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))',
            'area_km2': 1.0
        }]
        schema.store_grid_cells_batch(grid_id, cells_data)
        
        # Try to add same cell again
        with pytest.raises(Exception):
            schema.store_grid_cells_batch(grid_id, cells_data)
    
    def test_species_intersection_unique_constraint(self):
        """Test unique constraint on species intersections."""
        grid_id = schema.store_grid_definition("test_grid", "cubic", 1000)
        range_id = schema.store_species_range({
            'species_name': 'Test species',
            'geometry_wkt': 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))',
            'source_file': 'test.gpkg'
        })
        
        intersection = {
            'grid_id': grid_id,
            'cell_id': 'cell_1',
            'species_range_id': range_id,
            'species_name': 'Test species',
            'category': 'unknown',
            'range_type': 'distribution'
        }
        
        # Add intersection
        schema.store_species_intersections_batch([intersection])
        
        # Should handle conflict with upsert (not raise error)
        schema.store_species_intersections_batch([intersection])
        
        # Verify only one record exists
        with db.get_cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) as count FROM species_grid_intersections
                WHERE grid_id = %s AND cell_id = %s AND species_range_id = %s
            """, (grid_id, 'cell_1', range_id))
            count = cursor.fetchone()['count']
            assert count == 1
    
    def test_feature_unique_constraint(self):
        """Test unique constraint on features."""
        grid_id = schema.store_grid_definition("test_grid", "cubic", 1000)
        
        # Add feature
        schema.store_feature(grid_id, "cell_1", "richness", "plant_richness", 10.0)
        
        # Should handle conflict with upsert (not raise error)
        schema.store_feature(grid_id, "cell_1", "richness", "plant_richness", 15.0)
        
        # Verify only one record exists with updated value
        features = schema.get_features(grid_id)
        assert len(features) == 1
        assert features[0]['feature_value'] == 15.0