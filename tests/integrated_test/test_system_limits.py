# tests/integration/test_system_limits.py
import pytest
import psutil
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from pathlib import Path

from tests.fixtures.data_generator import TestDataGenerator
from src.config.config import Config
from src.parallel.pool_manager import PoolManager

class TestSystemLimits:
    """Test system behavior at limits."""
    
    @pytest.fixture
    def memory_limited_config(self, tmp_path):
        """Create config with strict memory limits."""
        config_data = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'database': 'test_biodiversity',
                'user': 'test_user',
                'password': 'test_pass',
                'pool_size': 5,
                'max_overflow': 10
            },
            'raster_processing': {
                'tile_size': 100,
                'memory_limit_mb': 100,  # Very low limit
                'parallel_workers': 4
            }
        }
        
        config_path = tmp_path / "config.yml"
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
            
        return Config(config_path)
    
    def test_memory_limit_enforcement(self, memory_limited_config, tmp_path):
        """Test system behavior when approaching memory limits."""
        generator = TestDataGenerator(tmp_path)
        
        # Create large raster that would exceed memory if fully loaded
        with generator.mock_large_raster(width=10000, height=10000) as large_raster:
            from src.raster_data.loaders.geotiff_loader import GeoTIFFLoader
            loader = GeoTIFFLoader(memory_limited_config)
            
            # Track memory during loading
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Load with lazy loading - should not exceed limit
            raster_data = loader.load(large_raster, lazy=True)
            
            # Process in tiles
            tile_count = 0
            max_memory = initial_memory
            
            for tile in raster_data.iter_tiles():
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                max_memory = max(max_memory, current_memory)
                
                # Verify memory limit is respected
                memory_increase = current_memory - initial_memory
                assert memory_increase < memory_limited_config.raster_processing.memory_limit_mb, \
                    f"Memory limit exceeded: {memory_increase:.1f}MB used"
                
                tile_count += 1
                if tile_count > 10:  # Process first 10 tiles
                    break
            
            assert tile_count > 0, "No tiles were processed"
    
    def test_database_connection_pool_stress(self, memory_limited_config):
        """Test database connection pool under stress."""
        from src.database.connection import DatabaseConnection
        
        db = DatabaseConnection(memory_limited_config)
        max_workers = 20  # More than pool size
        
        def worker_task(worker_id: int) -> Dict[str, Any]:
            """Simulate database operations."""
            start_time = time.time()
            
            # Try to get connection
            with db.get_connection() as conn:
                cur = conn.cursor()
                
                # Simulate work
                cur.execute("SELECT pg_sleep(0.1)")
                cur.execute("SELECT COUNT(*) FROM raster_sources")
                count = cur.fetchone()[0]
                
                # Check active connections
                cur.execute("SELECT COUNT(*) FROM pg_stat_activity WHERE state = 'active'")
                active = cur.fetchone()[0]
                
            return {
                'worker_id': worker_id,
                'duration': time.time() - start_time,
                'result_count': count,
                'active_connections': active
            }
        
        # Run workers in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker_task, i) for i in range(max_workers)]
            
            results = []
            connection_counts = []
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                connection_counts.append(result['active_connections'])
        
        # Verify connection pool worked correctly
        max_connections = max(connection_counts)
        pool_config = memory_limited_config.database.pool_size + memory_limited_config.database.max_overflow
        
        assert max_connections <= pool_config, \
            f"Connection pool exceeded: {max_connections} > {pool_config}"
        
        # All workers should complete
        assert len(results) == max_workers
    
    def test_cache_overflow_behavior(self, memory_limited_config, tmp_path):
        """Test cache behavior when reaching limits."""
        from src.resampling.cache_manager import ResamplingCache
        
        cache = ResamplingCache(memory_limited_config)
        generator = TestDataGenerator(tmp_path)
        
        # Fill cache beyond limit
        cache_entries = []
        cache_size_mb = 0
        target_size_mb = 50  # Half of memory limit
        
        while cache_size_mb < target_size_mb:
            # Generate unique cache entry
            grid_id = f"grid_{len(cache_entries)}"
            cell_id = f"cell_{np.random.randint(1000)}"
            value = np.random.rand()
            
            cache.put(
                source_raster_id=1,
                target_grid_id=grid_id,
                cell_id=cell_id,
                value=value,
                method='bilinear'
            )
            
            cache_entries.append((grid_id, cell_id))
            cache_size_mb = cache.get_size_mb()
        
        initial_entries = len(cache_entries)
        
        # Add more entries - should trigger eviction
        for i in range(100):
            cache.put(
                source_raster_id=1,
                target_grid_id=f"grid_overflow_{i}",
                cell_id=f"cell_{i}",
                value=np.random.rand(),
                method='bilinear'
            )
        
        # Verify cache size is still within limits
        final_size = cache.get_size_mb()
        assert final_size <= target_size_mb * 1.1, \
            f"Cache size exceeded limit: {final_size:.1f}MB > {target_size_mb}MB"
        
        # Verify LRU eviction - oldest entries should be gone
        hit_count = 0
        for grid_id, cell_id in cache_entries[:10]:  # Check first 10 entries
            if cache.get(1, grid_id, cell_id) is not None:
                hit_count += 1
        
        assert hit_count < 5, "LRU eviction not working properly"
    
    def test_parallel_processing_scaling(self, memory_limited_config, tmp_path):
        """Test scaling with different worker counts."""
        generator = TestDataGenerator(tmp_path)
        
        # Create test raster
        test_raster = generator.create_test_raster(
            width=1000,
            height=1000,
            pattern="gradient"
        )
        
        # Test with different worker counts
        worker_counts = [1, 2, 4, 8]
        processing_times = {}
        
        for workers in worker_counts:
            # Update config
            memory_limited_config.raster_processing.parallel_workers = workers
            
            # Time processing
            from src.parallel.pool_manager import PoolManager
            pool_manager = PoolManager(memory_limited_config)
            
            start_time = time.time()
            
            # Process raster in parallel
            tasks = []
            for i in range(100):  # 100 tiles
                task = {
                    'tile_id': i,
                    'bounds': (i * 10, 0, (i + 1) * 10, 10)
                }
                tasks.append(task)
            
            results = pool_manager.process_tasks(tasks, self._process_tile)
            
            duration = time.time() - start_time
            processing_times[workers] = duration
            
            # Verify all tasks completed
            assert len(results) == 100
        
        # Verify reasonable scaling
        # Should see improvement up to number of cores
        speedup = processing_times[1] / processing_times[4]
        assert speedup > 2.0, f"Poor parallel scaling: {speedup:.1f}x speedup with 4 workers"
    
    @staticmethod
    def _process_tile(task: Dict[str, Any]) -> Dict[str, Any]:
        """Mock tile processing function."""
        # Simulate CPU-intensive work
        data = np.random.rand(100, 100)
        result = np.sum(data ** 2)
        time.sleep(0.01)  # Simulate I/O
        
        return {
            'tile_id': task['tile_id'],
            'result': result
        }