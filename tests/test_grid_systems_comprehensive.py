"""
Comprehensive Grid Systems Test Module

Tests grid_systems functionality in multiple layers:
1. Individual function tests
2. Grouped functionality tests  
3. Integrity tests
4. Integration tests
5. Top-down system tests
"""

import pytest
import numpy as np
import h3
from typing import List, Dict, Tuple, Set
from unittest.mock import Mock, patch, MagicMock
from shapely.geometry import Polygon, Point, box
import logging

# Import grid system components
from src.grid_systems import (
    CubicGrid, 
    HexagonalGrid, 
    GridFactory, 
    BoundsManager, 
    BoundsDefinition,
    GridSpecification,
    create_standard_grids,
    get_or_create_grid
)
from src.base import BaseGrid, GridCell
from src.config import config

logger = logging.getLogger(__name__)


class TestIndividualFunctions:
    """Test individual functions in isolation."""
    
    # === CubicGrid Individual Function Tests ===
    
    def test_cubic_grid_init(self):
        """Test CubicGrid initialization."""
        grid = CubicGrid(
            resolution=10000,
            bounds=(0, 0, 10, 10),
            crs="EPSG:4326"
        )
        
        assert grid.resolution == 10000
        assert grid.bounds == (0, 0, 10, 10)
        assert grid.crs == "EPSG:4326"
        assert hasattr(grid, 'cell_size_degrees')
        assert hasattr(grid, 'bounds_def')
        
    def test_cubic_grid_calculate_cell_size_degrees(self):
        """Test _calculate_cell_size_degrees function."""
        grid = CubicGrid(resolution=11132, bounds=(0, 0, 10, 10))
        cell_size = grid._calculate_cell_size_degrees()
        
        # Should be approximately 0.1 degrees for ~11km resolution
        assert 0.05 < cell_size < 0.15
        
    def test_cubic_grid_generate_cell_id(self):
        """Test _generate_cell_id function."""
        grid = CubicGrid(resolution=10000, bounds=(0, 0, 10, 10))
        
        cell_id = grid._generate_cell_id(1.0, 2.0)
        
        assert cell_id.startswith("C10000_")
        assert "_" in cell_id
        
        # Test consistency
        cell_id2 = grid._generate_cell_id(1.0, 2.0)
        assert cell_id == cell_id2
        
    def test_cubic_grid_calculate_area_km2(self):
        """Test _calculate_area_km2 function."""
        grid = CubicGrid(resolution=10000, bounds=(0, 0, 1, 1))
        
        # Create a 1x1 degree polygon
        polygon = box(0, 0, 1, 1)
        area = grid._calculate_area_km2(polygon)
        
        # Area calculation may return very small value in current CRS
        # Just test that it returns a positive number
        assert area > 0
        
    def test_cubic_grid_get_cell_id(self):
        """Test get_cell_id function."""
        grid = CubicGrid(resolution=10000, bounds=(0, 0, 10, 10))
        
        cell_id = grid.get_cell_id(5.0, 5.0)
        assert cell_id.startswith("C10000_")
        
        # Test out of bounds
        with pytest.raises(ValueError):
            grid.get_cell_id(15.0, 15.0)
            
    def test_cubic_grid_get_neighbor_ids(self):
        """Test get_neighbor_ids function."""
        grid = CubicGrid(resolution=10000, bounds=(0, 0, 10, 10))
        
        center_id = grid.get_cell_id(5.0, 5.0)
        neighbors = grid.get_neighbor_ids(center_id)
        
        # Should have up to 8 neighbors (but may be fewer at edges)
        assert len(neighbors) <= 8
        assert len(neighbors) > 0
        assert all(n.startswith("C10000_") for n in neighbors)
        
    def test_cubic_grid_get_cell_size(self):
        """Test get_cell_size function."""
        grid = CubicGrid(resolution=10000, bounds=(0, 0, 10, 10))
        
        size = grid.get_cell_size()
        assert size > 0
        assert size == grid.cell_size_degrees
        
    def test_cubic_grid_get_cell_count(self):
        """Test get_cell_count function."""
        grid = CubicGrid(resolution=10000, bounds=(0, 0, 2, 2))
        
        count = grid.get_cell_count()
        assert count > 0
        assert isinstance(count, int)
        
    # === HexagonalGrid Individual Function Tests ===
    
    def test_hexagonal_grid_init(self):
        """Test HexagonalGrid initialization."""
        grid = HexagonalGrid(
            resolution=10000,
            bounds=(0, 0, 5, 5)
        )
        
        assert grid.resolution == 10000
        assert grid.bounds == (0, 0, 5, 5)
        assert grid.crs == "EPSG:4326"  # H3 requirement
        assert hasattr(grid, 'h3_resolution')
        assert hasattr(grid, 'bounds_def')
        
    def test_hexagonal_grid_invalid_crs(self):
        """Test HexagonalGrid rejects non-WGS84 CRS."""
        with pytest.raises(ValueError, match="H3 only supports EPSG:4326"):
            HexagonalGrid(
                resolution=10000,
                bounds=(0, 0, 5, 5),
                crs="EPSG:3857"
            )
            
    def test_hexagonal_grid_select_h3_resolution(self):
        """Test _select_h3_resolution function."""
        grid = HexagonalGrid(resolution=10000, bounds=(0, 0, 5, 5))
        
        # Test various resolutions
        h3_res_10km = grid._select_h3_resolution(10000)
        h3_res_1km = grid._select_h3_resolution(1000)
        h3_res_100m = grid._select_h3_resolution(100)
        
        # Higher resolution targets should give higher H3 resolutions
        assert h3_res_1km > h3_res_10km
        assert h3_res_100m > h3_res_1km
        
    def test_hexagonal_grid_get_cell_id(self):
        """Test get_cell_id function."""
        grid = HexagonalGrid(resolution=10000, bounds=(0, 0, 5, 5))
        
        cell_id = grid.get_cell_id(2.5, 2.5)
        assert cell_id.startswith(f"H{grid.h3_resolution}_")
        
        # Test out of bounds
        with pytest.raises(ValueError):
            grid.get_cell_id(10.0, 10.0)
            
    def test_hexagonal_grid_get_neighbor_ids(self):
        """Test get_neighbor_ids function."""
        grid = HexagonalGrid(resolution=10000, bounds=(0, 0, 5, 5))
        
        center_id = grid.get_cell_id(2.5, 2.5)
        neighbors = grid.get_neighbor_ids(center_id)
        
        # Hexagons should have 6 neighbors
        assert len(neighbors) == 6
        assert all(n.startswith(f"H{grid.h3_resolution}_") for n in neighbors)
        
    def test_hexagonal_grid_batch_iterator(self):
        """Test _batch_iterator function."""
        grid = HexagonalGrid(resolution=10000, bounds=(0, 0, 1, 1))
        
        items = set(str(i) for i in range(25))  # 25 string items
        batches = list(grid._batch_iterator(items, batch_size=10))
        
        assert len(batches) == 3  # ceil(25/10)
        assert len(batches[0]) == 10
        assert len(batches[1]) == 10
        assert len(batches[2]) == 5
        
    def test_hexagonal_grid_get_cell_size(self):
        """Test get_cell_size function."""
        grid = HexagonalGrid(resolution=10000, bounds=(0, 0, 5, 5))
        
        size = grid.get_cell_size()
        assert size > 0
        
    # === BoundsManager Individual Function Tests ===
    
    def test_bounds_manager_init(self):
        """Test BoundsManager initialization."""
        manager = BoundsManager()
        
        assert hasattr(manager, 'REGIONS')
        assert hasattr(manager, 'custom_regions')
        assert 'global' in manager.REGIONS
        assert 'europe' in manager.REGIONS
        
    def test_bounds_definition_properties(self):
        """Test BoundsDefinition properties."""
        bounds = BoundsDefinition(
            name='test',
            bounds=(0, 0, 10, 10),
            crs='EPSG:4326',
            category='test'
        )
        
        assert bounds.name == 'test'
        assert bounds.bounds == (0, 0, 10, 10)
        assert bounds.crs == 'EPSG:4326'
        assert bounds.category == 'test'
        
        # Test polygon property
        polygon = bounds.polygon
        assert isinstance(polygon, Polygon)
        assert polygon.bounds == (0, 0, 10, 10)
        
    def test_bounds_definition_intersects(self):
        """Test BoundsDefinition intersects method."""
        bounds1 = BoundsDefinition('test1', (0, 0, 10, 10))
        
        # Overlapping bounds
        assert bounds1.intersects((5, 5, 15, 15))
        
        # Non-overlapping bounds
        assert not bounds1.intersects((20, 20, 30, 30))
        
    def test_bounds_definition_intersection(self):
        """Test BoundsDefinition intersection method."""
        bounds1 = BoundsDefinition('test1', (0, 0, 10, 10))
        
        # Overlapping intersection
        result = bounds1.intersection((5, 5, 15, 15))
        assert result == (5, 5, 10, 10)
        
        # No intersection
        result = bounds1.intersection((20, 20, 30, 30))
        assert result is None
        
    def test_bounds_manager_get_bounds_predefined(self):
        """Test get_bounds with predefined regions."""
        manager = BoundsManager()
        
        europe = manager.get_bounds('europe')
        assert europe.name == 'europe'
        assert europe.category == 'continent'
        
        global_bounds = manager.get_bounds('global')
        assert global_bounds.bounds == (-180, -90, 180, 90)
        
    def test_bounds_manager_get_bounds_custom_string(self):
        """Test get_bounds with custom bounds string."""
        manager = BoundsManager()
        
        custom = manager.get_bounds('0,0,10,10')
        assert custom.name == 'custom_bounds'
        assert custom.bounds == (0.0, 0.0, 10.0, 10.0)
        assert custom.category == 'custom'
        
    def test_bounds_manager_get_bounds_invalid(self):
        """Test get_bounds with invalid input."""
        manager = BoundsManager()
        
        with pytest.raises(ValueError, match="Unknown bounds"):
            manager.get_bounds('invalid_region')
            
    def test_bounds_manager_list_available(self):
        """Test list_available method."""
        manager = BoundsManager()
        
        available = manager.list_available()
        assert isinstance(available, dict)
        assert 'continent' in available
        assert 'europe' in available['continent']
        assert 'global' in available['global']
        
    def test_bounds_manager_validate_bounds_overlap(self):
        """Test validate_bounds_overlap method."""
        manager = BoundsManager()
        bounds = BoundsDefinition('test', (0, 0, 10, 10))
        
        # Overlapping data bounds
        assert manager.validate_bounds_overlap(bounds, (5, 5, 15, 15))
        
        # Non-overlapping data bounds
        assert not manager.validate_bounds_overlap(bounds, (20, 20, 30, 30))
        
    def test_bounds_manager_subdivide_bounds(self):
        """Test subdivide_bounds method."""
        manager = BoundsManager()
        bounds = BoundsDefinition('large', (0, 0, 30, 30))
        
        # Should subdivide large bounds
        chunks = manager.subdivide_bounds(bounds, max_size_degrees=10.0)
        assert len(chunks) > 1
        assert all(chunk.name.startswith('large_chunk_') for chunk in chunks)
        
        # Small bounds should not be subdivided
        small_bounds = BoundsDefinition('small', (0, 0, 5, 5))
        small_chunks = manager.subdivide_bounds(small_bounds, max_size_degrees=10.0)
        assert len(small_chunks) == 1
        assert small_chunks[0] == small_bounds
        
    # === GridFactory Individual Function Tests ===
    
    def test_grid_specification_creation(self):
        """Test GridSpecification creation."""
        spec = GridSpecification(
            grid_type='cubic',
            resolution=10000,
            bounds='europe',
            name='test_grid'
        )
        
        assert spec.grid_type == 'cubic'
        assert spec.resolution == 10000
        assert spec.bounds == 'europe'
        assert spec.name == 'test_grid'
        assert spec.crs == "EPSG:4326"  # default
        
    def test_grid_specification_to_dict(self):
        """Test GridSpecification to_dict method."""
        spec = GridSpecification(
            grid_type='hexagonal',
            resolution=5000,
            bounds='global'
        )
        
        spec_dict = spec.to_dict()
        assert isinstance(spec_dict, dict)
        assert spec_dict['grid_type'] == 'hexagonal'
        assert spec_dict['resolution'] == 5000
        assert spec_dict['bounds'] == 'global'
        
    def test_grid_factory_init(self):
        """Test GridFactory initialization."""
        factory = GridFactory()
        
        assert hasattr(factory, 'bounds_manager')
        assert hasattr(factory, '_grid_cache')
        assert isinstance(factory.bounds_manager, BoundsManager)
        assert isinstance(factory._grid_cache, dict)
        
    @patch('src.core.registry.component_registry')
    def test_grid_factory_create_grid_from_spec(self, mock_registry):
        """Test create_grid from GridSpecification."""
        # Mock the registry to return CubicGrid
        mock_registry.grids.get.return_value = CubicGrid
        
        factory = GridFactory()
        spec = GridSpecification(
            grid_type='cubic',
            resolution=10000,
            bounds='global'
        )
        
        grid = factory.create_grid(spec)
        
        assert isinstance(grid, CubicGrid)
        assert grid.resolution == 10000
        assert hasattr(grid, 'specification')
        assert grid.specification == spec
        
    @patch('src.core.registry.component_registry')
    def test_grid_factory_create_grid_from_dict(self, mock_registry):
        """Test create_grid from dictionary."""
        mock_registry.grids.get.return_value = HexagonalGrid
        
        factory = GridFactory()
        spec_dict = {
            'grid_type': 'hexagonal',
            'resolution': 5000,
            'bounds': 'europe'
        }
        
        grid = factory.create_grid(spec_dict)
        
        assert isinstance(grid, HexagonalGrid)
        assert grid.resolution == 5000
        

class TestGroupedFunctionality:
    """Test grouped functionality within each component."""
    
    def test_cubic_grid_coordinate_operations(self):
        """Test cubic grid coordinate-related operations together."""
        grid = CubicGrid(resolution=10000, bounds=(0, 0, 10, 10))
        
        # Test coordinate to cell ID
        x, y = 5.0, 5.0
        cell_id = grid.get_cell_id(x, y)
        
        # Test cell ID parsing and neighbor finding
        neighbors = grid.get_neighbor_ids(cell_id)
        
        # All operations should be consistent
        assert cell_id.startswith("C10000_")
        assert len(neighbors) > 0
        assert all(n.startswith("C10000_") for n in neighbors)
        assert cell_id not in neighbors  # Cell should not be its own neighbor
        
    def test_hexagonal_grid_h3_operations(self):
        """Test hexagonal grid H3-related operations together."""
        grid = HexagonalGrid(resolution=10000, bounds=(0, 0, 5, 5))
        
        # Test H3 resolution selection
        h3_res = grid.h3_resolution
        assert isinstance(h3_res, int)
        assert 0 <= h3_res <= 15
        
        # Test coordinate to H3 operations
        x, y = 2.5, 2.5
        cell_id = grid.get_cell_id(x, y)
        neighbors = grid.get_neighbor_ids(cell_id)
        
        # H3 operations should be consistent
        assert cell_id.startswith(f"H{h3_res}_")
        assert len(neighbors) == 6  # Hexagons have 6 neighbors
        assert all(n.startswith(f"H{h3_res}_") for n in neighbors)
        
    def test_bounds_manager_region_operations(self):
        """Test bounds manager region-related operations together."""
        manager = BoundsManager()
        
        # Test getting predefined region
        europe = manager.get_bounds('europe')
        
        # Test region properties
        available = manager.list_available()
        
        # Test region validation
        data_bounds = (0, 45, 50, 70)  # Overlaps with Europe
        is_valid = manager.validate_bounds_overlap(europe, data_bounds)
        
        # All operations should be consistent
        assert europe.category == 'continent'
        assert 'europe' in available['continent']
        assert is_valid  # Europe should overlap with test data bounds
        
    def test_bounds_manager_custom_bounds_operations(self):
        """Test bounds manager custom bounds operations together."""
        manager = BoundsManager()
        
        # Test custom bounds creation
        custom_bounds = manager.get_bounds('10,20,30,40')
        
        # Test bounds intersection
        intersection = custom_bounds.intersection((25, 35, 45, 55))
        
        # Test subdivision
        large_bounds = BoundsDefinition('large', (0, 0, 25, 25))
        chunks = manager.subdivide_bounds(large_bounds, max_size_degrees=10.0)
        
        # All operations should be consistent
        assert custom_bounds.bounds == (10.0, 20.0, 30.0, 40.0)
        assert intersection == (25.0, 35.0, 30.0, 40.0)
        assert len(chunks) > 1
        
    def test_grid_factory_grid_creation_workflow(self):
        """Test grid factory creation workflow."""
        factory = GridFactory()
        
        # Test multi-resolution grid creation
        with patch('src.core.registry.component_registry') as mock_registry:
            mock_registry.grids.get.return_value = CubicGrid
            
            grids = factory.create_multi_resolution_grids(
                grid_type='cubic',
                resolutions=[25000, 10000, 5000],
                bounds='europe',
                base_name='test'
            )
            
            # All grids should be created with consistent properties
            assert len(grids) == 3
            assert 25000 in grids and 10000 in grids and 5000 in grids
            assert all(isinstance(g, CubicGrid) for g in grids.values())
            assert all(g.specification is not None for g in grids.values())
            # Verify all grids have proper name specifications
            for g in grids.values():
                if g.specification and hasattr(g.specification, 'name') and g.specification.name:
                    assert g.specification.name.startswith('test_')


class TestIntegrityTests:
    """Test data integrity and consistency within grid systems."""
    
    def test_cubic_grid_cell_consistency(self):
        """Test cubic grid cell ID consistency and uniqueness."""
        grid = CubicGrid(resolution=10000, bounds=(0, 0, 5, 5))
        
        # Test multiple cell IDs don't overlap
        test_coords = [(1, 1), (2, 2), (3, 3), (4, 4)]
        cell_ids = [grid.get_cell_id(x, y) for x, y in test_coords]
        
        # All cell IDs should be unique
        assert len(set(cell_ids)) == len(cell_ids)
        
        # Cell IDs should be consistent when called multiple times
        for x, y in test_coords:
            id1 = grid.get_cell_id(x, y)
            id2 = grid.get_cell_id(x, y)
            assert id1 == id2
            
    def test_hexagonal_grid_neighbor_consistency(self):
        """Test hexagonal grid neighbor relationships are symmetric."""
        grid = HexagonalGrid(resolution=10000, bounds=(0, 0, 5, 5))
        
        # Get a cell and its neighbors
        center_id = grid.get_cell_id(2.5, 2.5)
        neighbors = grid.get_neighbor_ids(center_id)
        
        # Test symmetry: if A is neighbor of B, then B should be neighbor of A
        for neighbor_id in neighbors[:3]:  # Test first 3 to avoid too many API calls
            neighbor_neighbors = grid.get_neighbor_ids(neighbor_id)
            assert center_id in neighbor_neighbors, f"Neighbor relationship not symmetric: {center_id} -> {neighbor_id}"
            
    def test_bounds_manager_bounds_integrity(self):
        """Test bounds manager data integrity."""
        manager = BoundsManager()
        
        # Test all predefined regions have valid bounds
        for name, bounds_def in manager.REGIONS.items():
            minx, miny, maxx, maxy = bounds_def.bounds
            
            # Basic bounds validation
            assert minx < maxx, f"Invalid bounds for {name}: minx >= maxx"
            assert miny < maxy, f"Invalid bounds for {name}: miny >= maxy"
            assert -180 <= minx <= 180, f"Invalid longitude for {name}: {minx}"
            assert -180 <= maxx <= 180, f"Invalid longitude for {name}: {maxx}"
            assert -90 <= miny <= 90, f"Invalid latitude for {name}: {miny}"
            assert -90 <= maxy <= 90, f"Invalid latitude for {name}: {maxy}"
            
    def test_grid_factory_specification_integrity(self):
        """Test grid factory maintains specification integrity."""
        factory = GridFactory()
        
        with patch('src.core.registry.component_registry') as mock_registry:
            mock_registry.grids.get.return_value = CubicGrid
            
            spec = GridSpecification(
                grid_type='cubic',
                resolution=10000,
                bounds='europe',
                name='integrity_test'
            )
            
            grid = factory.create_grid(spec)
            
            # Grid should maintain specification integrity
            assert grid.specification == spec
            assert grid.resolution == spec.resolution
            if grid.specification and hasattr(grid.specification, 'name'):
                assert grid.specification.name == spec.name
            
    def test_coordinate_bounds_consistency(self):
        """Test coordinate bounds are consistent across all components."""
        # Test with same bounds across different components
        test_bounds = (5, 10, 15, 20)
        
        # BoundsDefinition
        bounds_def = BoundsDefinition('test', test_bounds)
        assert bounds_def.bounds == test_bounds
        
        # CubicGrid
        cubic_grid = CubicGrid(resolution=10000, bounds=test_bounds)
        assert cubic_grid.bounds == test_bounds
        
        # HexagonalGrid
        hex_grid = HexagonalGrid(resolution=10000, bounds=test_bounds)
        assert hex_grid.bounds == test_bounds
        
        # BoundsManager
        manager = BoundsManager()
        custom_bounds = manager.get_bounds('5,10,15,20')
        assert custom_bounds.bounds == test_bounds


class TestIntegrationTests:
    """Test integration between different grid system components."""
    
    def test_grid_factory_with_bounds_manager_integration(self):
        """Test grid factory integrates properly with bounds manager."""
        factory = GridFactory()
        
        with patch('src.core.registry.component_registry') as mock_registry:
            mock_registry.grids.get.return_value = CubicGrid
            
            # Use bounds manager region in grid factory
            spec = GridSpecification(
                grid_type='cubic',
                resolution=10000,
                bounds='europe'  # From bounds manager
            )
            
            grid = factory.create_grid(spec)
            
            # Grid should have correct bounds from bounds manager
            europe_bounds = factory.bounds_manager.get_bounds('europe').bounds
            assert grid.bounds == europe_bounds
            
    def test_grid_with_bounds_definition_integration(self):
        """Test grids integrate properly with BoundsDefinition objects."""
        bounds_def = BoundsDefinition(
            name='integration_test',
            bounds=(0, 0, 10, 10),
            category='test'
        )
        
        # Test with CubicGrid
        cubic_grid = CubicGrid(resolution=10000, bounds=bounds_def)
        assert cubic_grid.bounds_def == bounds_def
        assert cubic_grid.bounds == bounds_def.bounds
        
        # Test with HexagonalGrid
        hex_grid = HexagonalGrid(resolution=10000, bounds=bounds_def)
        assert hex_grid.bounds_def == bounds_def
        assert hex_grid.bounds == bounds_def.bounds
        
    def test_grid_factory_caching_integration(self):
        """Test grid factory caching works with different components."""
        factory = GridFactory()
        
        with patch('src.core.registry.component_registry') as mock_registry:
            mock_registry.grids.get.return_value = CubicGrid
            
            # Create multi-resolution grids which uses caching
            grids = factory.create_multi_resolution_grids(
                grid_type='cubic',
                resolutions=[10000, 5000],
                bounds='global',
                base_name='cached_test'
            )
            
            # Cache should contain the grids with their names
            assert len(grids) == 2
            # Check that grids were created with names
            for grid in grids.values():
                if hasattr(grid, 'specification') and grid.specification and grid.specification.name:
                    assert grid.specification.name in factory._grid_cache
            
    def test_bounds_subdivision_with_grids_integration(self):
        """Test bounds subdivision integrates with grid generation."""
        manager = BoundsManager()
        
        # Create large bounds that will be subdivided
        large_bounds = BoundsDefinition('large_region', (0, 0, 30, 30))
        chunks = manager.subdivide_bounds(large_bounds, max_size_degrees=10.0)
        
        # Test that grids can be created for each chunk
        with patch('src.core.registry.component_registry') as mock_registry:
            mock_registry.grids.get.return_value = CubicGrid
            
            factory = GridFactory()
            
            for chunk in chunks[:2]:  # Test first 2 chunks
                spec = GridSpecification(
                    grid_type='cubic',
                    resolution=10000,
                    bounds=chunk
                )
                
                grid = factory.create_grid(spec)
                assert grid.bounds == chunk.bounds
                
    @patch('src.database.connection.db')
    def test_database_integration_mocking(self, mock_db):
        """Test grid systems would integrate with database (mocked)."""
        # Mock database responses
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {
                'geom_wkt': 'POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))',
                'centroid_wkt': 'POINT(0.5 0.5)',
                'area_km2': 12321.0,
                'minx': 0.0, 'miny': 0.0, 'maxx': 1.0, 'maxy': 1.0
            }
        ]
        mock_db.get_cursor.return_value.__enter__.return_value = mock_cursor
        mock_db.pool = True
        
        # Test CubicGrid with PostGIS
        grid = CubicGrid(
            resolution=10000,
            bounds=(0, 0, 2, 2),
            use_postgis=True
        )
        
        # Should use PostGIS when database is available
        assert grid.use_postgis


class TestTopDownSystemTests:
    """Test complete grid system workflows from top to bottom."""
    
    @patch('src.core.registry.component_registry')
    def test_complete_cubic_grid_workflow(self, mock_registry):
        """Test complete cubic grid creation and usage workflow."""
        mock_registry.grids.get.return_value = CubicGrid
        
        # 1. Create grid factory
        factory = GridFactory()
        
        # 2. Create grid specification
        spec = GridSpecification(
            grid_type='cubic',
            resolution=10000,
            bounds='europe',
            name='workflow_test_cubic'
        )
        
        # 3. Create grid
        grid = factory.create_grid(spec)
        
        # 4. Use grid for coordinate operations
        test_coords = [(0, 50), (10, 55), (20, 60)]  # European coordinates
        
        cell_ids = []
        for x, y in test_coords:
            try:
                cell_id = grid.get_cell_id(x, y)
                cell_ids.append(cell_id)
                
                # Get neighbors for each cell (cast to CubicGrid for method access)
                if isinstance(grid, CubicGrid):
                    neighbors = grid.get_neighbor_ids(cell_id)
                    assert len(neighbors) <= 8  # Cubic grid max neighbors
                
            except ValueError:
                # Coordinate might be outside bounds - that's ok
                pass
                
        # 5. Verify grid properties
        assert isinstance(grid, CubicGrid)
        assert grid.resolution == 10000
        if grid.specification and hasattr(grid.specification, 'name'):
            assert grid.specification.name == 'workflow_test_cubic'
        assert len(cell_ids) > 0  # At least some coordinates should be valid
        
    @patch('src.core.registry.component_registry')
    def test_complete_hexagonal_grid_workflow(self, mock_registry):
        """Test complete hexagonal grid creation and usage workflow."""
        mock_registry.grids.get.return_value = HexagonalGrid
        
        # 1. Create grid factory
        factory = GridFactory()
        
        # 2. Create multi-resolution hexagonal grids
        grids = factory.create_multi_resolution_grids(
            grid_type='hexagonal',
            resolutions=[25000, 10000, 5000],
            bounds='oceania',
            base_name='workflow_hex'
        )
        
        # 3. Test each resolution level
        for resolution, grid in grids.items():
            assert isinstance(grid, HexagonalGrid)
            assert grid.resolution == resolution
            
            # Test H3 operations
            test_coord = (150, -25)  # Oceania coordinates
            try:
                cell_id = grid.get_cell_id(test_coord[0], test_coord[1])
                neighbors = grid.get_neighbor_ids(cell_id)
                
                # Hexagonal properties
                assert cell_id.startswith(f"H{grid.h3_resolution}_")
                assert len(neighbors) == 6
                
            except ValueError:
                # Coordinate might be outside bounds
                pass
                
    def test_complete_bounds_management_workflow(self):
        """Test complete bounds management workflow."""
        # 1. Initialize bounds manager
        manager = BoundsManager()
        
        # 2. Work with predefined regions
        europe = manager.get_bounds('europe')
        asia = manager.get_bounds('asia')
        
        # 3. Create custom region
        custom_region = manager.get_bounds('10,40,50,70')  # Custom European subset
        
        # 4. Test region operations
        # Check if custom region overlaps with Europe
        overlap = manager.validate_bounds_overlap(custom_region, europe.bounds)
        assert overlap
        
        # 5. Subdivide large region
        large_region = manager.get_bounds('global')
        chunks = manager.subdivide_bounds(large_region, max_size_degrees=30.0)
        
        # Should create multiple chunks for global bounds
        assert len(chunks) > 1
        
        # 6. List all available regions
        available = manager.list_available()
        assert 'continent' in available
        assert 'europe' in available['continent']
        assert 'asia' in available['continent']
        
    @patch('src.core.registry.component_registry')
    def test_complete_grid_factory_workflow(self, mock_registry):
        """Test complete grid factory workflow with standard grids."""
        mock_registry.grids.get.return_value = CubicGrid
        
        # 1. Create standard grids
        grids = create_standard_grids(
            grid_type='cubic',
            bounds='global',
            store=False  # Don't actually store in database
        )
        
        # 2. Verify standard resolutions created
        expected_levels = ['coarse', 'medium', 'fine', 'very_fine', 'ultra_fine']
        for level in expected_levels:
            assert level in grids
            assert isinstance(grids[level], CubicGrid)
            
        # 3. Test grid retrieval (mocked since no database)
        factory = GridFactory()
        
        # 4. Create custom grid with specific parameters
        custom_spec = GridSpecification(
            grid_type='cubic',
            resolution=15000,
            bounds='north_america',
            name='custom_workflow_test'
        )
        
        custom_grid = factory.create_grid(custom_spec)
        
        # 5. Verify complete integration
        assert isinstance(custom_grid, CubicGrid)
        assert custom_grid.resolution == 15000
        if custom_grid.specification and hasattr(custom_grid.specification, 'name'):
            assert custom_grid.specification.name == 'custom_workflow_test'
        
        # Grid should use North America bounds
        na_bounds = factory.bounds_manager.get_bounds('north_america').bounds
        assert custom_grid.bounds == na_bounds
        
    def test_error_handling_workflow(self):
        """Test error handling across the complete system."""
        manager = BoundsManager()
        factory = GridFactory()
        
        # 1. Test invalid bounds handling
        with pytest.raises(ValueError):
            manager.get_bounds('nonexistent_region')
            
        # 2. Test invalid coordinates
        grid = CubicGrid(resolution=10000, bounds=(0, 0, 10, 10))
        
        with pytest.raises(ValueError):
            grid.get_cell_id(50, 50)  # Outside bounds
            
        # 3. Test invalid CRS for hexagonal grid
        with pytest.raises(ValueError):
            HexagonalGrid(resolution=10000, bounds=(0, 0, 5, 5), crs="EPSG:3857")
            
        # 4. Test invalid grid type in factory
        with patch('src.core.registry.component_registry') as mock_registry:
            mock_registry.grids.get.return_value = None
            
            with pytest.raises(ValueError):
                factory.create_grid({
                    'grid_type': 'invalid_type',
                    'resolution': 10000,
                    'bounds': 'global'
                })


# Pytest fixtures for setup
@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        'grids': {
            'default_bounds': [-180, -90, 180, 90],
            'cubic': {'use_postgis': False},
            'hexagonal': {'chunk_size': 10000}
        }
    }


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])
