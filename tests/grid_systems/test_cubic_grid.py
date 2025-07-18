# tests/grid_systems/test_cubic_grid.py
"""Tests for cubic grid implementation."""

import pytest
from shapely.geometry import Point, Polygon

from src.grid_systems import CubicGrid, BoundsDefinition
from src.base import GridCell


class TestCubicGrid:
    """Test CubicGrid class."""
    
    def test_grid_creation(self, mock_config):
        """Test basic grid creation."""
        grid = CubicGrid(
            resolution=10000,  # 10km
            bounds=(0, 0, 10, 10),
            use_postgis=False
        )
        
        assert grid.resolution == 10000
        assert grid.bounds == (0, 0, 10, 10)
        assert grid.crs == "EPSG:4326"
        
    def test_cell_size_calculation(self, mock_config):
        """Test cell size calculation in degrees."""
        # At equator
        grid = CubicGrid(
            resolution=111000,  # ~111km = ~1 degree
            bounds=(-1, -1, 1, 1),
            use_postgis=False
        )
        
        # Should be approximately 1 degree
        assert 0.9 < grid.cell_size_degrees < 1.1
        
    def test_generate_grid_python(self, mock_config):
        """Test grid generation using Python implementation."""
        grid = CubicGrid(
            resolution=111000,  # ~1 degree cells
            bounds=(0, 0, 3, 3),
            use_postgis=False
        )
        
        cells = grid.generate_grid()
        
        # Should have 3x3 = 9 cells
        assert len(cells) == 9
        
        # Check cell properties
        for cell in cells:
            assert isinstance(cell, GridCell)
            assert cell.cell_id.startswith('C111000_')
            assert isinstance(cell.geometry, Polygon)
            assert cell.area_km2 > 0
            assert cell.metadata['grid_type'] == 'cubic'
            
    def test_cell_id_generation(self, mock_config):
        """Test cell ID generation."""
        grid = CubicGrid(
            resolution=10000,
            bounds=(0, 0, 10, 10),
            use_postgis=False
        )
        
        # Test various positions
        assert grid.get_cell_id(0, 0) == 'C10000_00000_00000'
        assert grid.get_cell_id(5, 5).startswith('C10000_')
        
        # Test outside bounds
        with pytest.raises(ValueError):
            grid.get_cell_id(-1, 5)
            
    def test_get_cell_by_id(self, mock_config):
        """Test retrieving cell by ID."""
        grid = CubicGrid(
            resolution=111000,
            bounds=(0, 0, 10, 10),
            use_postgis=False
        )
        
        # Valid cell
        cell = grid.get_cell_by_id('C111000_00001_00001')
        assert cell is not None
        assert cell.cell_id == 'C111000_00001_00001'
        assert isinstance(cell.geometry, Polygon)
        
        # Invalid cell ID formats
        assert grid.get_cell_by_id('invalid') is None
        assert grid.get_cell_by_id('H3_12345') is None  # Wrong type
        assert grid.get_cell_by_id('C50000_00001_00001') is None  # Wrong resolution
        
    def test_get_neighbor_ids(self, mock_config):
        """Test getting neighbor cell IDs."""
        grid = CubicGrid(
            resolution=111000,
            bounds=(0, 0, 10, 10),
            use_postgis=False
        )
        
        # Center cell should have 8 neighbors
        neighbors = grid.get_neighbor_ids('C111000_00001_00001')
        assert len(neighbors) == 8
        
        # Corner cell should have 3 neighbors
        neighbors = grid.get_neighbor_ids('C111000_00000_00000')
        assert len(neighbors) == 3
        
        # Edge cell should have 5 neighbors
        neighbors = grid.get_neighbor_ids('C111000_00001_00000')
        assert len(neighbors) == 5
        
    def test_bounds_integration(self, mock_config):
        """Test integration with BoundsDefinition."""
        bounds_def = BoundsDefinition('test', (10, 20, 30, 40))
        
        grid = CubicGrid(
            resolution=111000,
            bounds=bounds_def,
            use_postgis=False
        )
        
        assert grid.bounds == bounds_def.bounds
        assert grid.bounds_def == bounds_def
        
    def test_spatial_queries(self, mock_config):
        """Test spatial query methods."""
        grid = CubicGrid(
            resolution=111000,
            bounds=(0, 0, 5, 5),
            use_postgis=False
        )
        
        # Get cells in smaller bounds
        cells = grid.get_cells_in_bounds((1, 1, 3, 3))
        assert len(cells) > 0
        assert all(cell.geometry.intersects(box(1, 1, 3, 3)) for cell in cells)
        
        # Get cells for polygon
        test_poly = Polygon([(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)])
        cells = grid.get_cells_for_geometry(test_poly)
        assert len(cells) > 0
        assert all(cell.geometry.intersects(test_poly) for cell in cells)
        
    def test_large_grid_generation(self, mock_config):
        """Test generation of large grids."""
        # This would normally use PostGIS
        grid = CubicGrid(
            resolution=1000000,  # 1000km cells
            bounds=(-180, -90, 180, 90),
            use_postgis=False
        )
        
        cells = grid.generate_grid()
        
        # Should have reasonable number of cells
        assert len(cells) > 10
        assert len(cells) < 1000
        
        # Check total coverage
        total_area = sum(cell.area_km2 for cell in cells)
        earth_area = 510072000  # km²
        
        # Should cover significant portion of Earth
        assert total_area > earth_area * 0.5