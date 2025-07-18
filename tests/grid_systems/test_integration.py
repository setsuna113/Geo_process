"""Integration tests for grid system components."""

import pytest
from unittest.mock import patch

from src.grid_systems import (
    BoundsManager, GridFactory, 
    create_standard_grids, get_or_create_grid
)


class TestGridSystemIntegration:
    """Test integration between grid system components."""
    
    def test_bounds_manager_with_factory(self, mock_config):
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
        assert grid.bounds_def == africa_bounds
        
    def test_multi_grid_type_generation(self, mock_config):
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
        
        # Area coverage should be similar
        cubic_area = sum(c.area_km2 for c in cubic_cells)
        hex_area = sum(c.area_km2 for c in hex_cells)
        
        assert abs(cubic_area - hex_area) / cubic_area < 0.2  # Within 20%
        
    def test_resolution_hierarchy(self, mock_config):
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
        
    def test_partial_bounds_processing(self, mock_config):
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
        
        # All cells should be within study bounds
        for cell in cells:
            assert study_bounds.contains(cell.centroid.x, cell.centroid.y)
            
    @patch('src.database.schema.schema.get_grid_by_name')
    @patch('src.database.schema.schema.store_grid_definition')
    @patch('src.database.schema.schema.store_grid_cells_batch')
    def test_create_standard_grids_helper(self, mock_store_cells, 
                                         mock_store_def, mock_get, 
                                         mock_config):
        """Test create_standard_grids helper function."""
        # Mock database operations
        mock_get.return_value = None
        mock_store_def.return_value = 'grid_id'
        
        grids = create_standard_grids(
            grid_type='cubic',
            bounds='global',
            store=True
        )
        
        assert len(grids) == 5  # All standard resolutions
        assert mock_store_def.call_count == 5  # Stored all grids
        
    def test_grid_cell_clipping(self, mock_config):
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