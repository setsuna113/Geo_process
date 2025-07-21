"""Tests for hexagonal grid implementation."""

import pytest
from shapely.geometry import Point, Polygon

from src.grid_systems import HexagonalGrid, BoundsDefinition
from src.base import GridCell


class TestHexagonalGrid:
    """Test HexagonalGrid class."""
    
    def test_grid_creation(self, test_config):
        """Test basic grid creation."""
        grid = HexagonalGrid(
            resolution=10000,  # 10km
            bounds=(0, 0, 10, 10)
        )
        
        assert grid.resolution == 10000
        assert grid.bounds == (0, 0, 10, 10)
        assert grid.crs == "EPSG:4326"  # H3 requirement
        
    def test_invalid_crs(self, test_config):
        """Test that non-WGS84 CRS raises error."""
        with pytest.raises(ValueError) as exc_info:
            HexagonalGrid(
                resolution=10000,
                bounds=(0, 0, 10, 10),
                crs="EPSG:3857"
            )
        
        assert "H3 only supports EPSG:4326" in str(exc_info.value)
        
    def test_h3_resolution_selection(self, test_config):
        """Test H3 resolution selection for target sizes."""
        grid = HexagonalGrid(resolution=100000, bounds=(0, 0, 1, 1))
        assert grid.h3_resolution in [2, 3, 4]  # Coarse resolution
        
        grid = HexagonalGrid(resolution=10000, bounds=(0, 0, 1, 1))
        assert grid.h3_resolution in [5, 6, 7]  # Medium resolution
        
        grid = HexagonalGrid(resolution=1000, bounds=(0, 0, 1, 1))
        assert grid.h3_resolution in [7, 8, 9]  # Fine resolution
        
    def test_generate_small_grid(self, test_config):
        """Test generation of small hexagonal grid."""
        grid = HexagonalGrid(
            resolution=50000,  # 50km
            bounds=(0, 0, 2, 2)
        )
        
        cells = grid.generate_grid()
        
        assert len(cells) > 0
        
        # Check cell properties
        for cell in cells:
            assert isinstance(cell, GridCell)
            assert cell.cell_id.startswith(f'H{grid.h3_resolution}_')
            assert isinstance(cell.geometry, Polygon)
            assert cell.area_km2 > 0
            assert cell.metadata is not None
            assert cell.metadata['grid_type'] == 'hexagonal'
            assert 'h3_id' in cell.metadata
            
    def test_cell_id_operations(self, test_config):
        """Test cell ID related operations."""
        grid = HexagonalGrid(
            resolution=50000,
            bounds=(0, 0, 5, 5)
        )
        
        # Get cell ID for coordinate
        cell_id = grid.get_cell_id(2.5, 2.5)
        assert cell_id.startswith(f'H{grid.h3_resolution}_')
        
        # Get cell by ID
        cell = grid.get_cell_by_id(cell_id)
        assert cell is not None
        assert cell.cell_id == cell_id
        
        # Invalid cell ID
        assert grid.get_cell_by_id('invalid') is None
        assert grid.get_cell_by_id('C10000_00000_00000') is None  # Cubic ID
        
    def test_neighbor_operations(self, test_config):
        """Test hexagonal neighbor operations."""
        grid = HexagonalGrid(
            resolution=50000,
            bounds=(0, 0, 5, 5)
        )
        
        # Get a cell
        cell_id = grid.get_cell_id(2.5, 2.5)
        
        # Hexagons have 6 neighbors
        neighbors = grid.get_neighbor_ids(cell_id)
        assert len(neighbors) == 6
        assert all(n.startswith(f'H{grid.h3_resolution}_') for n in neighbors)
        
    def test_resolution_hierarchy(self, test_config):
        """Test getting cells at different resolutions."""
        grid = HexagonalGrid(
            resolution=50000,
            bounds=(0, 0, 5, 5),
            h3_resolution=5
        )
        
        # Get a cell
        cell_id = grid.get_cell_id(2.5, 2.5)
        
        # Get children (finer resolution)
        children = grid.get_cells_at_resolution(cell_id, 6)
        assert len(children) == 7  # H3 has 7 children per hexagon
        
        # Get parent (coarser resolution)
        parents = grid.get_cells_at_resolution(cell_id, 4)
        assert len(parents) == 1
        
    def test_bounds_handling(self, test_config):
        """Test hexagonal grid with different bounds."""
        # Small bounds
        small_grid = HexagonalGrid(
            resolution=10000,
            bounds=BoundsDefinition('small', (0, 0, 1, 1))
        )
        
        small_cells = small_grid.generate_grid()
        assert len(small_cells) > 0
        
        # Check that cells are roughly within bounds (allowing for hexagon edge overflow)
        # Hexagons can extend slightly beyond rectangular bounds which is expected
        buffer = 0.1  # Allow 0.1 degree buffer for hexagon edges
        for cell in small_cells:
            centroid = cell.centroid
            assert -buffer <= centroid.x <= 1 + buffer
            assert -buffer <= centroid.y <= 1 + buffer
            
    def test_memory_chunking(self, test_config):
        """Test that large grids are processed in chunks."""
        # Set small chunk size in the config's grids settings
        test_config.grids['hexagonal']['chunk_size'] = 10
        
        grid = HexagonalGrid(
            resolution=100000,
            bounds=(0, 0, 10, 10)
        )
        
        # Should still generate all cells despite chunking
        cells = grid.generate_grid()
        assert len(cells) > 10  # More than chunk size
        
        # Check for duplicates
        cell_ids = [cell.cell_id for cell in cells]
        assert len(cell_ids) == len(set(cell_ids))