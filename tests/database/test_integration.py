"""Integration tests for the complete database system."""

import pytest
from src.database.setup import setup_database, reset_database, check_database_status
from src.database.schema import schema
from src.config import config

class TestDatabaseSetup:
    """Test database setup and initialization."""
    
    def test_initial_database_setup(self):
        """Test initial database setup from scratch."""
        # Drop everything first
        try:
            schema.drop_schema(confirm=True)
        except:
            pass  # Ignore if schema doesn't exist
        
        # Setup database
        success = setup_database()
        assert success
        
        # Verify setup
        status = check_database_status()
        assert status['ready']
        assert status['connection']
        assert status['schema']
    
    def test_database_reset(self):
        """Test database reset functionality."""
        # Add some data first
        grid_id = schema.store_grid_definition("test_grid", "cubic", 1000)
        
        # Reset database
        success = reset_database()
        assert success
        
        # Verify data is gone
        grid = schema.get_grid_by_name("test_grid")
        assert grid is None
        
        # Verify schema still exists
        info = schema.get_schema_info()
        assert info['summary']['table_count'] > 0
    
    def test_database_status_check(self):
        """Test database status checking."""
        status = check_database_status()
        
        assert 'status' in status
        assert 'connection' in status
        assert 'schema' in status
        assert 'ready' in status
        
        # Should be ready after setup
        assert status['ready']

class TestCompleteWorkflow:
    """Test complete workflow scenarios."""
    
    @pytest.fixture(autouse=True)
    def setup_clean_db(self):
        """Ensure clean database for each test."""
        reset_database()
        yield
    
    def test_grid_creation_workflow(self):
        """Test complete grid creation workflow."""
        # 1. Create grid definition
        grid_id = schema.store_grid_definition(
            name="workflow_cubic",
            grid_type="cubic",
            resolution=5000,
            bounds="POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))",
            metadata={"workflow_test": True}
        )
        
        # 2. Verify grid exists
        grid = schema.get_grid_by_name("workflow_cubic")
        assert grid is not None
        assert grid['grid_type'] == 'cubic'
        assert grid['resolution'] == 5000
        
        # 3. Add grid cells
        cells_data = []
        for i in range(10):
            for j in range(10):
                cells_data.append({
                    'cell_id': f'cell_{i}_{j}',
                    'geometry_wkt': f'POLYGON(({i} {j}, {i+1} {j}, {i+1} {j+1}, {i} {j+1}, {i} {j}))',
                    'area_km2': 1.0,
                    'centroid_wkt': f'POINT({i+0.5} {j+0.5})'
                })
        
        count = schema.store_grid_cells_batch(grid_id, cells_data)
        assert count == 100
        
        # 4. Verify grid status
        status = schema.get_grid_status("workflow_cubic")
        assert len(status) == 1
        assert status[0]['cells_generated'] == 100
        assert status[0]['total_cells'] == 100
    
    def test_species_processing_workflow(self):
        """Test complete species processing workflow."""
        # 1. Create grid
        grid_id = schema.store_grid_definition("species_grid", "cubic", 1000)
        
        # Add some grid cells
        cells_data = [
            {
                'cell_id': f'cell_{i}',
                'geometry_wkt': f'POLYGON(({i} {i}, {i+1} {i}, {i+1} {i+1}, {i} {i+1}, {i} {i}))',
                'area_km2': 1.0
            }
            for i in range(5)
        ]
        schema.store_grid_cells_batch(grid_id, cells_data)
        
        # 2. Load species ranges
        species_list = []
        for i, category in enumerate(['plant', 'animal', 'fungi']):
            for j in range(3):
                species_name = f'{category.title()}_species_{j}'
                range_id = schema.store_species_range({
                    'species_name': species_name,
                    'scientific_name': f'{species_name} scientificus',
                    'genus': f'Genus_{j}',
                    'family': f'{category.title()}aceae',
                    'category': category,
                    'range_type': 'distribution',
                    'geometry_wkt': f'POLYGON(({j} {j}, {j+2} {j}, {j+2} {j+2}, {j} {j+2}, {j} {j}))',
                    'source_file': f'{category}_species_{j}.gpkg',
                    'confidence': 0.9,
                    'area_km2': 4.0
                })
                species_list.append((range_id, species_name, category))
        
        # 3. Compute intersections
        intersections = []
        for range_id, species_name, category in species_list:
            for cell_id in ['cell_0', 'cell_1', 'cell_2']:
                intersections.append({
                    'grid_id': grid_id,
                    'cell_id': cell_id,
                    'species_range_id': range_id,
                    'species_name': species_name,
                    'category': category,
                    'range_type': 'distribution',
                    'intersection_area_km2': 0.5,
                    'coverage_percent': 50.0,
                    'presence_score': 0.8
                })
        
        count = schema.store_species_intersections_batch(intersections)
        assert count == len(intersections)
        
        # 4. Verify richness calculations
        richness = schema.get_species_richness(grid_id)
        assert len(richness) > 0
        
        # Should have richness for each category
        categories = {r['category'] for r in richness}
        assert 'plant' in categories
        assert 'animal' in categories
        assert 'fungi' in categories
    
    def test_feature_computation_workflow(self):
        """Test complete feature computation workflow."""
        # 1. Setup grid and species data
        grid_id = schema.store_grid_definition("feature_grid", "cubic", 1000)
        
        # Add cells
        for i in range(3):
            schema.store_grid_cells_batch(grid_id, [{
                'cell_id': f'cell_{i}',
                'geometry_wkt': f'POLYGON(({i} {i}, {i+1} {i}, {i+1} {i+1}, {i} {i+1}, {i} {i}))',
                'area_km2': 1.0
            }])
        
        # 2. Compute richness features
        richness_features = []
        for i in range(3):
            for category in ['plant', 'animal']:
                richness_features.append({
                    'grid_id': grid_id,
                    'cell_id': f'cell_{i}',
                    'feature_type': 'richness',
                    'feature_name': f'{category}_richness',
                    'feature_value': float(i * 5 + (1 if category == 'plant' else 2)),
                    'computation_metadata': {'method': 'count_distinct'}
                })
        
        schema.store_features_batch(richness_features)
        
        # 3. Add climate features
        climate_data = []
        for i in range(3):
            for var in ['bio_1', 'bio_12']:
                climate_data.append({
                    'grid_id': grid_id,
                    'cell_id': f'cell_{i}',
                    'variable': var,
                    'value': float(i * 10 + (1 if var == 'bio_1' else 100)),
                    'source': 'WorldClim',
                    'resolution': '10m'
                })
        
        schema.store_climate_data_batch(climate_data)
        
        # 4. Compute interaction features
        interaction_features = []
        for i in range(3):
            # Get plant richness and bio_1 for this cell
            plant_richness = i * 5 + 1
            bio_1 = i * 10 + 1
            
            interaction_features.append({
                'grid_id': grid_id,
                'cell_id': f'cell_{i}',
                'feature_type': 'interaction',
                'feature_name': 'P_x_T',
                'feature_value': plant_richness * bio_1,
                'computation_metadata': {
                    'components': ['plant_richness', 'bio_1'],
                    'operation': 'multiply'
                }
            })
        
        schema.store_features_batch(interaction_features)
        
        # 5. Verify all features exist
        all_features = schema.get_features(grid_id)
        
        # Should have richness, climate (via climate_data), and interaction features
        feature_types = {f['feature_type'] for f in all_features}
        assert 'richness' in feature_types
        assert 'interaction' in feature_types
        
        # Verify specific features
        feature_names = {f['feature_name'] for f in all_features}
        assert 'plant_richness' in feature_names
        assert 'animal_richness' in feature_names
        assert 'P_x_T' in feature_names
    
    def test_experiment_tracking_workflow(self):
        """Test complete experiment tracking workflow."""
        # 1. Create experiment
        experiment_id = schema.create_experiment(
            name="integration_test_experiment",
            description="Testing complete workflow tracking",
            config={
                "grid_type": "cubic",
                "resolution": 1000,
                "species_files": ["test1.gpkg", "test2.gpkg"],
                "features": ["richness", "climate", "interactions"],
                "created_by": "integration_test"
            }
        )
        
        # 2. Create processing jobs
        jobs = []
        job_types = [
            ("grid_generation", "Generate grid cells"),
            ("species_loading", "Load species data"),
            ("intersection_computation", "Compute intersections"),
            ("feature_computation", "Compute features")
        ]
        
        for job_type, job_name in job_types:
            job_id = schema.create_processing_job(
                job_type=job_type,
                job_name=job_name,
                parameters={"experiment_id": experiment_id, "test": True},
                parent_experiment_id=experiment_id
            )
            jobs.append(job_id)
        
        # 3. Simulate job progress
        for i, job_id in enumerate(jobs):
            # Start job
            schema.update_job_progress(
                job_id, 0.0, status="running", 
                log_message=f"Starting job {i+1}"
            )
            
            # Progress updates
            for progress in [25.0, 50.0, 75.0]:
                schema.update_job_progress(
                    job_id, progress, 
                    log_message=f"Progress: {progress}%"
                )
            
            # Complete job
            schema.update_job_progress(
                job_id, 100.0, status="completed",
                log_message=f"Job {i+1} completed successfully"
            )
        
        # 4. Complete experiment
        schema.update_experiment_status(
            experiment_id, "completed",
            results={
                "grids_created": 1,
                "species_processed": 10,
                "features_computed": 100,
                "total_runtime_minutes": 15.5
            }
        )

class TestConfigurationIntegration:
    """Test integration with configuration system."""
    
    def test_grid_config_validation(self):
        """Test that database validates against configuration."""
        # Get valid configurations from config
        cubic_resolutions = config.get('grids.cubic.resolutions')
        hex_resolutions = config.get('grids.hexagonal.resolutions')
        
        # Test valid configurations
        for resolution in cubic_resolutions:
            assert schema.validate_grid_config('cubic', resolution)
        
        for resolution in hex_resolutions:
            assert schema.validate_grid_config('hexagonal', resolution)
        
        # Test invalid configurations
        assert not schema.validate_grid_config('cubic', 999)
        assert not schema.validate_grid_config('hexagonal', 15)
        assert not schema.validate_grid_config('invalid_type', 1000)
    
    def test_species_categories_integration(self):
        """Test species categories match configuration."""
        reset_database()
        
        valid_categories = config.get('species_classification.valid_categories', ['plant', 'animal', 'fungi'])
        
        # Test storing species with valid categories
        for category in valid_categories:
            range_id = schema.store_species_range({
                'species_name': f'Test_{category}',
                'category': category,
                'geometry_wkt': 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))',
                'source_file': 'test.gpkg'
            })
            assert range_id is not None
        
        # Verify all categories stored correctly
        species_list = schema.get_species_ranges()
        stored_categories = {s['category'] for s in species_list}
        
        for category in valid_categories:
            assert category in stored_categories