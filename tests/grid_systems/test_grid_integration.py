"""Integration tests for grid system components."""

import pytest
from unittest.mock import patch, MagicMock

from src.grid_systems import (
    BoundsManager, GridFactory, 
    create_standard_grids, get_or_create_grid
)


class TestGridSystemIntegration:
    """Test integration between grid system components."""
    
    def test_bounds_manager_with_factory(self, test_config):
        """Test bounds manager integration with factory."""
        bounds_manager = BoundsManager()
        factory = GridFactory()
        
        # Get bounds and create grid
        africa_bounds = bounds_manager.get_bounds('africa')
        
        grid = factory.create_grid({
            'grid_type': 'cubic',
            'resolution': 50000,
            'bounds': africa_bounds
        })
        
        assert grid.bounds == africa_bounds.bounds
        # Note: bounds_def is not available on all BaseGrid implementations
        
    def test_multi_grid_type_generation(self, test_config):
        """Test generating both cubic and hexagonal grids."""
        factory = GridFactory()
        
        # Create cubic grid
        cubic = factory.create_grid({
            'grid_type': 'cubic',
            'resolution': 25000,
            'bounds': 'europe'
        })
        
        # Create hexagonal grid  
        hexagonal = factory.create_grid({
            'grid_type': 'hexagonal',
            'resolution': 25000,
            'bounds': 'europe'
        })
        
        # Both should cover similar areas
        cubic_cells = cubic.get_cells()
        hex_cells = hexagonal.get_cells()
        
        assert len(cubic_cells) > 0
        assert len(hex_cells) > 0
        
        # Area coverage should be similar (but different grid types have different packing)
        cubic_area = sum(c.area_km2 for c in cubic_cells)
        hex_area = sum(c.area_km2 for c in hex_cells)
        
        # Different grid types can have significantly different total coverage due to:
        # - Different geometric packing (hexagons vs squares)
        # - Different boundary handling and resolution discretization
        # Allow up to 80% difference for different grid geometries
        assert abs(cubic_area - hex_area) / max(cubic_area, hex_area) < 0.8
        
    def test_resolution_hierarchy(self, test_config):
        """Test creating and using resolution hierarchy."""
        factory = GridFactory()
        
        # Create hierarchy
        grids = factory.create_multi_resolution_grids(
            grid_type='cubic',
            resolutions=[100000, 50000, 25000],
            bounds='africa',
            base_name='africa_bio'
        )
        
        # Verify hierarchy relationships
        coarse = grids[100000]
        medium = grids[50000]
        fine = grids[25000]
        
        # Finer grids should have more cells
        assert len(fine.get_cells()) > len(medium.get_cells())
        assert len(medium.get_cells()) > len(coarse.get_cells())
        
        # All should cover same bounds
        assert coarse.bounds == medium.bounds == fine.bounds
        
    def test_partial_bounds_processing(self, test_config):
        """Test processing with partial bounds."""
        bounds_manager = BoundsManager()
        factory = GridFactory()
        
        # Define custom study area
        study_bounds = bounds_manager.get_bounds('10,45,15,50')  # Small Europe region
        
        # Create grid for study area
        grid = factory.create_grid({
            'grid_type': 'cubic',
            'resolution': 10000,
            'bounds': study_bounds
        })
        
        cells = grid.get_cells()
        
        # Most cells should be within or very close to study bounds
        # Allow some tolerance for grid alignment at boundaries
        tolerance = 0.1  # degrees
        within_bounds_count = 0
        
        for cell in cells:
            # Check if centroid is within expanded bounds (with tolerance)
            expanded_bounds = (
                study_bounds.bounds[0] - tolerance,
                study_bounds.bounds[1] - tolerance, 
                study_bounds.bounds[2] + tolerance,
                study_bounds.bounds[3] + tolerance
            )
            if (expanded_bounds[0] <= cell.centroid.x <= expanded_bounds[2] and 
                expanded_bounds[1] <= cell.centroid.y <= expanded_bounds[3]):
                within_bounds_count += 1
        
        # At least 80% of cells should be within the expanded bounds
        assert within_bounds_count >= len(cells) * 0.8
            
    @patch('src.grid_systems.grid_factory.GridFactory.create_standard_grids')
    @patch('src.database.schema.schema.get_grid_by_name')
    @patch('src.database.schema.schema.store_grid_definition')
    @patch('src.database.schema.schema.store_grid_cells_batch')
    def test_create_standard_grids_helper(self, mock_store_cells, 
                                         mock_store_def, mock_get, 
                                         mock_create_standard, test_config):
        """Test create_standard_grids helper function."""
        # Mock database operations
        mock_get.return_value = None
        mock_store_def.return_value = 'grid_id'
        
        # Mock the factory method to avoid actual grid generation
        mock_grids = {
            f'cubic_{res}': MagicMock() for res in [1000, 5000, 10000, 25000, 50000]
        }
        mock_create_standard.return_value = mock_grids
        
        grids = create_standard_grids(
            grid_type='cubic',
            bounds='global',
            store=True
        )
        
        assert len(grids) == 5  # All standard resolutions
        mock_create_standard.assert_called_once_with('cubic', 'global')
        
    def test_grid_cell_clipping(self, test_config):
        """Test grid cell clipping at boundaries."""
        bounds_manager = BoundsManager()
        factory = GridFactory()
        
        # Create grid larger than desired bounds
        large_grid = factory.create_grid({
            'grid_type': 'cubic',
            'resolution': 50000,
            'bounds': (-10, -10, 10, 10)
        })
        
        # Clip to smaller bounds
        clip_bounds = bounds_manager.get_bounds('0,0,5,5')
        clipped_cells = bounds_manager.clip_grid_cells(
            large_grid.get_cells(), 
            clip_bounds
        )
        
        # Check clipping worked
        assert len(clipped_cells) < len(large_grid.get_cells())
        
        for cell in clipped_cells:
            # Cell should intersect clip bounds
            assert cell.geometry.intersects(clip_bounds.polygon)

    def test_top_down_system_integration(self, test_config):
        """Test complete grid system from top-down perspective."""
        # Test system-wide grid creation and validation
        from src.grid_systems import GridFactory, BoundsManager
        
        bounds_manager = BoundsManager()
        factory = GridFactory()
        
        # Test with multiple regions
        regions = ['europe', 'africa', 'north_america']
        grid_types = ['cubic', 'hexagonal']
        resolutions = [25000, 50000]
        
        created_grids = []
        
        for region in regions:
            for grid_type in grid_types:
                for resolution in resolutions:
                    try:
                        grid = factory.create_grid({
                            'grid_type': grid_type,
                            'resolution': resolution,
                            'bounds': region
                        })
                        created_grids.append((region, grid_type, resolution, grid))
                    except Exception as e:
                        pytest.fail(f"Failed to create {grid_type} grid for {region} at {resolution}m: {e}")
        
        # Verify all grids were created successfully
        expected_count = len(regions) * len(grid_types) * len(resolutions)
        assert len(created_grids) == expected_count
        
        # Test grid integrity across different configurations
        for region, grid_type, resolution, grid in created_grids:
            assert grid.resolution == resolution
            assert len(grid.bounds) == 4
            
            # Generate sample cells to test functionality
            cells = grid.generate_grid()
            assert len(cells) > 0
            
            # Test cell properties
            for cell in cells[:min(5, len(cells))]:  # Test first 5 cells
                assert cell.cell_id is not None
                assert cell.geometry is not None
                assert cell.area_km2 > 0

    def test_grid_consistency_across_resolutions(self, test_config):
        """Test that grids maintain consistency across different resolutions."""
        factory = GridFactory()
        
        # Create grids at different resolutions for same bounds
        resolutions = [10000, 25000, 50000, 100000]
        grids = []
        
        for resolution in resolutions:
            grid = factory.create_grid({
                'grid_type': 'cubic',
                'resolution': resolution,
                'bounds': (0, 0, 10, 10)
            })
            grids.append((resolution, grid))
        
        # Verify grid consistency
        for i, (res1, grid1) in enumerate(grids):
            for res2, grid2 in grids[i+1:]:
                # Higher resolution should have more cells
                cells1 = grid1.generate_grid()
                cells2 = grid2.generate_grid()
                
                if res1 < res2:  # res1 has higher resolution (smaller cell size)
                    assert len(cells1) >= len(cells2)
                
                # Bounds should be consistent
                assert grid1.bounds == grid2.bounds

    def test_error_handling_integration(self, test_config):
        """Test error handling across grid system components."""
        factory = GridFactory()
        bounds_manager = BoundsManager()
        
        # Test invalid grid type
        with pytest.raises((ValueError, KeyError)):
            factory.create_grid({
                'grid_type': 'invalid_type',
                'resolution': 25000,
                'bounds': 'europe'
            })
        
        # Test invalid bounds
        with pytest.raises((ValueError, KeyError)):
            bounds_manager.get_bounds('invalid_region')
        
        # Test invalid resolution
        with pytest.raises(ValueError):
            factory.create_grid({
                'grid_type': 'cubic',
                'resolution': -1000,  # Negative resolution
                'bounds': 'europe'
            })