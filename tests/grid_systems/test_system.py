"""End-to-end system tests for grid system."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

from src.grid_systems import (
    BoundsManager, GridFactory,
    create_standard_grids, get_or_create_grid
)


class TestGridSystemE2E:
    """End-to-end tests for complete grid system workflows."""
    
    @patch('src.grid_systems.grid_factory.schema')
    @patch('src.base.grid.schema')
    def test_complete_grid_workflow(self, mock_base_schema, mock_factory_schema, test_config):
        """Test complete workflow from creation to storage."""
        # Mock database operations for both schema instances
        mock_base_schema.get_grid_by_name.return_value = None
        mock_base_schema.store_grid_definition.return_value = 'grid_123'
        mock_base_schema.store_grid_cells_batch.return_value = None
        
        mock_factory_schema.get_grid_by_name.return_value = None
        mock_factory_schema.store_grid_definition.return_value = 'grid_123'
        mock_factory_schema.store_grid_cells_batch.return_value = None
        
        # 1. Initialize system
        bounds_manager = BoundsManager()
        factory = GridFactory()
        
        # 2. Define study area
        study_area = bounds_manager.get_bounds('africa')
        
        # 3. Create multi-resolution grids
        grids = factory.create_multi_resolution_grids(
            grid_type='cubic',
            resolutions=[100000, 50000, 25000],
            bounds=study_area,
            base_name='africa_biodiversity'
        )
        
        # 4. Generate cells for each resolution
        cells_by_resolution = {}
        for resolution, grid in grids.items():
            cells = grid.get_cells()
            cells_by_resolution[resolution] = cells
            
        # 5. Validate grid properties
        assert len(cells_by_resolution[25000]) > len(cells_by_resolution[50000])
        assert len(cells_by_resolution[50000]) > len(cells_by_resolution[100000])
        
        # 6. Store grids
        for grid in grids.values():
            grid_id = factory.store_grid(grid)
            assert grid_id == 'grid_123'
            
        # 7. Verify storage operations
        assert mock_base_schema.store_grid_definition.call_count == 3
        assert mock_base_schema.store_grid_cells_batch.call_count > 0
        
    def test_global_grid_generation_scenario(self, test_config):
        """Test generating global grids for biodiversity analysis."""
        factory = GridFactory()
        bounds_manager = BoundsManager()
        
        # Simulate breaking global bounds into chunks
        global_bounds = bounds_manager.get_bounds('global')
        chunks = bounds_manager.subdivide_bounds(global_bounds, max_size_degrees=60)
        
        total_cells = 0
        
        for chunk in chunks[:3]:  # Test first 3 chunks only
            # Create grid for chunk
            chunk_grid = factory.create_grid({
                'grid_type': 'cubic',
                'resolution': 100000,  # 100km
                'bounds': chunk,
                'name': f'global_chunk_{chunk.name}'
            })
            
            cells = chunk_grid.get_cells()
            total_cells += len(cells)
            
        assert total_cells > 0
        assert len(chunks) > 1  # Global should be subdivided
        
    def test_mixed_grid_types_workflow(self, test_config):
        """Test workflow with both cubic and hexagonal grids."""
        factory = GridFactory()
        
        # Create cubic grid for initial analysis
        cubic_grid = factory.create_grid({
            'grid_type': 'cubic',
            'resolution': 50000,
            'bounds': 'europe',
            'name': 'europe_cubic_50k'
        })
        
        # Create hexagonal grid for detailed analysis
        hex_grid = factory.create_grid({
            'grid_type': 'hexagonal',
            'resolution': 50000,
            'bounds': 'europe',
            'name': 'europe_hex_50k'
        })
        
        # Compare grids
        cubic_cells = cubic_grid.get_cells()
        hex_cells = hex_grid.get_cells()
        
        # Both should exist and cover similar area
        assert len(cubic_cells) > 0
        assert len(hex_cells) > 0
        
        # Test getting specific cells
        test_point = (10, 50)  # Somewhere in Europe
        cubic_cell_id = cubic_grid.get_cell_id(*test_point)
        hex_cell_id = hex_grid.get_cell_id(*test_point)
        
        assert cubic_cell_id.startswith('C50000_')
        assert hex_cell_id.startswith('H')
        
    def test_regional_analysis_workflow(self, test_config):
        """Test workflow for regional biodiversity analysis."""
        factory = GridFactory()
        bounds_manager = BoundsManager()
        
        # Define regions of interest
        regions = ['africa', 'south_america', 'asia']
        
        regional_grids = {}
        
        for region in regions:
            # Get region bounds
            region_bounds = bounds_manager.get_bounds(region)
            
            # Create appropriate resolution based on region size
            if region_bounds.area_km2 > 10_000_000:  # Large region
                resolution = 100000  # 100km
            else:
                resolution = 50000   # 50km
                
            # Create grid
            grid = factory.create_grid({
                'grid_type': 'cubic',
                'resolution': resolution,
                'bounds': region_bounds,
                'name': f'{region}_biodiversity_{resolution}m'
            })
            
            regional_grids[region] = grid
            
        # Verify grids created
        assert len(regional_grids) == 3
        assert all(grid is not None for grid in regional_grids.values())
        
    def test_error_handling_workflow(self, test_config):
        """Test error handling in grid system workflow."""
        factory = GridFactory()
        bounds_manager = BoundsManager()
        
        # Test invalid bounds
        with pytest.raises(ValueError):
            bounds_manager.get_bounds('invalid_region')
            
        # Test invalid grid type
        with pytest.raises(ValueError):
            factory.create_grid({
                'grid_type': 'invalid_type',
                'resolution': 10000,
                'bounds': 'global'
            })
            
        # Test incompatible parameters
        with pytest.raises(ValueError):
            # Hexagonal grid with non-WGS84 CRS
            factory.create_grid({
                'grid_type': 'hexagonal',
                'resolution': 10000,
                'bounds': 'global',
                'crs': 'EPSG:3857'
            })
            
    def test_performance_characteristics(self, test_config):
        """Test performance with different grid sizes."""
        import time
        
        factory = GridFactory()
        
        resolutions = [100000, 50000, 25000]
        times = {}
        
        for resolution in resolutions:
            start = time.time()
            
            grid = factory.create_grid({
                'grid_type': 'cubic',
                'resolution': resolution,
                'bounds': (0, 0, 10, 10)  # Small area
            })
            
            cells = grid.get_cells()
            
            times[resolution] = time.time() - start
            
            # Basic performance check
            assert times[resolution] < 5.0  # Should complete in 5 seconds
            
        # Finer resolution should take more time
        assert times[25000] >= times[100000]


# Performance and stress tests
class TestGridSystemPerformance:
    """Performance tests for grid system."""
    
    @pytest.mark.slow
    def test_large_grid_generation(self, test_config):
        """Test generating large grid."""
        factory = GridFactory()
        
        # Generate 10km grid for large area
        grid = factory.create_grid({
            'grid_type': 'cubic',
            'resolution': 10000,
            'bounds': (-180, -60, 180, 60)  # Most of Earth
        })
        
        # Should handle large grid efficiently
        cells = grid.get_cells()
        
        # Rough estimate: ~40,000 x 13,000 km area / 10km cells
        expected_cells = (40000 / 10) * (13000 / 10)
        
        # Should be in right ballpark
        assert len(cells) > expected_cells * 0.5
        assert len(cells) < expected_cells * 2
        
    @pytest.mark.slow 
    def test_memory_efficiency(self, test_config):
        """Test memory usage stays reasonable."""
        import psutil
        import gc
        
        factory = GridFactory()
        process = psutil.Process()
        
        # Get baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large grid
        grid = factory.create_grid({
            'grid_type': 'hexagonal',
            'resolution': 50000,
            'bounds': 'global'
        })
        
        # Generate cells (should use chunking)
        cells = grid.get_cells()
        
        # Check memory usage
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = current_memory - baseline_memory
        
        # Should not use excessive memory (< 1GB for global 50km grid)
        assert memory_used < 1024
        
        # Cleanup
        del cells
        del grid
        gc.collect()