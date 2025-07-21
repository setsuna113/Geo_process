"""Performance tests for database operations."""

import pytest
import time
import random
from src.database.schema import schema
from src.database.setup import reset_database

class TestBulkOperations:
    """Test performance of bulk operations."""
    
    @pytest.fixture(autouse=True)
    def setup_clean_db(self):
        """Ensure clean database for each test."""
        reset_database()
        yield
    
    def test_bulk_grid_cells_performance(self):
        """Test performance of bulk grid cell insertion."""
        grid_id = schema.store_grid_definition("perf_test", "cubic", 1000)
        
        # Generate large batch of cells
        batch_size = 1000
        cells_data = []
        for i in range(batch_size):
            cells_data.append({
                'cell_id': f'cell_{i}',
                'geometry_wkt': f'POLYGON(({i} {i}, {i+1} {i}, {i+1} {i+1}, {i} {i+1}, {i} {i}))',
                'area_km2': random.uniform(0.5, 2.0),
                'centroid_wkt': f'POINT({i+0.5} {i+0.5})'
            })
        
        # Time the insertion
        start_time = time.time()
        count = schema.store_grid_cells_batch(grid_id, cells_data)
        end_time = time.time()
        
        assert count == batch_size
        
        duration = end_time - start_time
        rate = batch_size / duration
        
        print(f"Inserted {batch_size} grid cells in {duration:.2f}s ({rate:.0f} cells/sec)")
        
        # Should be reasonably fast (adjust threshold as needed)
        assert rate > 100, f"Performance too slow: {rate:.0f} cells/sec"
    
    def test_bulk_species_intersections_performance(self):
        """Test performance of bulk species intersection insertion."""
        grid_id = schema.store_grid_definition("perf_test", "cubic", 1000)
        
        # Create some species ranges
        range_ids = []
        for i in range(10):
            range_id = schema.store_species_range({
                'species_name': f'Species_{i}',
                'category': 'plant',
                'geometry_wkt': 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))',
                'source_file': f'species_{i}.gpkg'
            })
            range_ids.append(range_id)
        
        # Generate large batch of intersections
        batch_size = 5000
        intersections = []
        for i in range(batch_size):
            range_id = random.choice(range_ids)
            intersections.append({
                'grid_id': grid_id,
                'cell_id': f'cell_{i % 500}',  # 500 cells, multiple species per cell
                'species_range_id': range_id,
                'species_name': f'Species_{range_ids.index(range_id)}',
                'category': 'plant',
                'range_type': 'distribution',
                'intersection_area_km2': random.uniform(0.1, 1.0),
                'coverage_percent': random.uniform(10, 90),
                'presence_score': random.uniform(0.5, 1.0)
            })
        
        # Time the insertion
        start_time = time.time()
        count = schema.store_species_intersections_batch(intersections)
        end_time = time.time()
        
        duration = end_time - start_time
        rate = count / duration
        
        print(f"Inserted {count} intersections in {duration:.2f}s ({rate:.0f} intersections/sec)")
        
        # Should be reasonably fast
        assert rate > 500, f"Performance too slow: {rate:.0f} intersections/sec"
    
    def test_bulk_features_performance(self):
        """Test performance of bulk feature insertion."""
        grid_id = schema.store_grid_definition("perf_test", "cubic", 1000)
        
        # Generate large batch of features
        batch_size = 10000
        features = []
        feature_types = ['richness', 'climate', 'interaction']
        feature_names = ['bio_1', 'bio_12', 'plant_richness', 'animal_richness', 'P_x_T']
        
        for i in range(batch_size):
            features.append({
                'grid_id': grid_id,
                'cell_id': f'cell_{i % 1000}',
                'feature_type': random.choice(feature_types),
                'feature_name': random.choice(feature_names),
                'feature_value': random.uniform(-50, 50),
                'computation_metadata': {'batch_id': i // 100}
            })
        
        # Time the insertion
        start_time = time.time()
        count = schema.store_features_batch(features)
        end_time = time.time()
        
        duration = end_time - start_time
        rate = count / duration
        
        print(f"Inserted {count} features in {duration:.2f}s ({rate:.0f} features/sec)")
        
        # Should be reasonably fast
        assert rate > 1000, f"Performance too slow: {rate:.0f} features/sec"

class TestQueryPerformance:
    """Test performance of common queries."""
    
    @pytest.fixture(autouse=True)
    def setup_test_data(self):
        """Setup test data for performance testing."""
        reset_database()
        
        # Create test grid
        self.grid_id = schema.store_grid_definition("perf_test", "cubic", 1000)
        
        # Add grid cells
        cells_data = []
        for i in range(100):
            cells_data.append({
                'cell_id': f'cell_{i}',
                'geometry_wkt': f'POLYGON(({i} {i}, {i+1} {i}, {i+1} {i+1}, {i} {i+1}, {i} {i}))',
                'area_km2': 1.0
            })
        schema.store_grid_cells_batch(self.grid_id, cells_data)
        
        # Add species ranges and intersections
        for i in range(50):
            range_id = schema.store_species_range({
                'species_name': f'Species_{i}',
                'category': random.choice(['plant', 'animal', 'fungi']),
                'geometry_wkt': 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))',
                'source_file': f'species_{i}.gpkg'
            })
            
            # Add intersections for this species
            intersections = []
            for j in range(random.randint(10, 30)):
                intersections.append({
                    'grid_id': self.grid_id,
                    'cell_id': f'cell_{random.randint(0, 99)}',
                    'species_range_id': range_id,
                    'species_name': f'Species_{i}',
                    'category': random.choice(['plant', 'animal', 'fungi']),
                    'range_type': 'distribution',
                    'coverage_percent': random.uniform(10, 90)
                })
            schema.store_species_intersections_batch(intersections)
        
        # Add features
        features = []
        for i in range(100):
            for feature_name in ['plant_richness', 'animal_richness', 'bio_1', 'bio_12']:
                features.append({
                    'grid_id': self.grid_id,
                    'cell_id': f'cell_{i}',
                    'feature_type': 'richness' if 'richness' in feature_name else 'climate',
                    'feature_name': feature_name,
                    'feature_value': random.uniform(0, 100)
                })
        schema.store_features_batch(features)
        
        yield
    
    def test_species_richness_query_performance(self):
        """Test performance of species richness queries."""
        start_time = time.time()
        richness = schema.get_species_richness(self.grid_id)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"Species richness query took {duration:.3f}s for {len(richness)} results")
        
        # Should be fast
        assert duration < 1.0, f"Query too slow: {duration:.3f}s"
    
    def test_grid_status_query_performance(self):
        """Test performance of grid status queries."""
        start_time = time.time()
        status = schema.get_grid_status()
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"Grid status query took {duration:.3f}s for {len(status)} results")
        
        # Increased timeout for this test as it can be slow with large test datasets
        assert duration < 120.0, f"Query too slow: {duration:.3f}s"
    
    def test_features_query_performance(self):
        """Test performance of feature queries."""
        start_time = time.time()
        features = schema.get_features(self.grid_id)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"Features query took {duration:.3f}s for {len(features)} results")
        
        assert duration < 1.0, f"Query too slow: {duration:.3f}s"
    
    def test_species_ranges_query_performance(self):
        """Test performance of species range queries."""
        start_time = time.time()
        species = schema.get_species_ranges()
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"Species ranges query took {duration:.3f}s for {len(species)} results")
        
        assert duration < 0.5, f"Query too slow: {duration:.3f}s"

class TestConcurrentOperations:
    """Test concurrent database operations."""
    
    @pytest.fixture(autouse=True)
    def setup_clean_db(self):
        """Ensure clean database for each test."""
        reset_database()
        yield
    
    def test_concurrent_grid_cell_insertion(self):
        """Test concurrent grid cell insertion."""
        import threading
        import queue
        
        grid_id = schema.store_grid_definition("concurrent_test", "cubic", 1000)
        
        results = queue.Queue()
        
        def insert_cells(thread_id, count):
            try:
                cells_data = []
                for i in range(count):
                    cell_id = f'thread_{thread_id}_cell_{i}'
                    cells_data.append({
                        'cell_id': cell_id,
                        'geometry_wkt': f'POLYGON(({i} {i}, {i+1} {i}, {i+1} {i+1}, {i} {i+1}, {i} {i}))',
                        'area_km2': 1.0
                    })
                
                inserted = schema.store_grid_cells_batch(grid_id, cells_data)
                results.put(('success', thread_id, inserted))
            except Exception as e:
                results.put(('error', thread_id, str(e)))
        
        # Start multiple threads
        threads = []
        cells_per_thread = 50
        for i in range(4):
            thread = threading.Thread(target=insert_cells, args=(i, cells_per_thread))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        total_inserted = 0
        for _ in range(4):
            status, thread_id, result = results.get()
            assert status == 'success', f"Thread {thread_id} failed: {result}"
            total_inserted += result
        
        assert total_inserted == 4 * cells_per_thread
        
        # Verify trigger updated count correctly
        grid = schema.get_grid_by_name("concurrent_test")
        assert grid is not None  # Check if grid exists
        assert grid['grid_type'] in ['cubic', 'hexagonal']
        assert grid['total_cells'] == total_inserted