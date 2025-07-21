"""Tests for bounds manager."""

import pytest
from shapely.geometry import box, Point, Polygon

from src.grid_systems import BoundsManager, BoundsDefinition
from src.base import GridCell


class TestBoundsDefinition:
    """Test BoundsDefinition class."""
    
    def test_bounds_creation(self):
        """Test creating bounds definition."""
        bounds = BoundsDefinition('test', (0, 0, 10, 10))
        
        assert bounds.name == 'test'
        assert bounds.bounds == (0, 0, 10, 10)
        assert bounds.crs == "EPSG:4326"
        assert bounds.category == "custom"
        
    def test_polygon_property(self):
        """Test polygon generation."""
        bounds = BoundsDefinition('test', (0, 0, 10, 10))
        poly = bounds.polygon
        
        assert poly.bounds == (0, 0, 10, 10)
        assert poly.area == 100
        
    def test_area_calculation(self):
        """Test area calculation in km²."""
        # Approximate 1 degree = 111 km at equator
        bounds = BoundsDefinition('test', (0, 0, 1, 1))
        area = bounds.area_km2
        
        assert area > 12000  # Should be ~12,321 km²
        assert area < 13000
        
    def test_contains_point(self):
        """Test point containment."""
        bounds = BoundsDefinition('test', (0, 0, 10, 10))
        
        assert bounds.contains(5, 5)      # Inside
        assert bounds.contains(0, 0)      # Corner
        assert bounds.contains(10, 10)    # Opposite corner
        assert not bounds.contains(-1, 5)  # Outside
        assert not bounds.contains(5, 11)  # Outside
        
    def test_intersects(self):
        """Test bounds intersection."""
        bounds1 = BoundsDefinition('test1', (0, 0, 10, 10))
        
        # Overlapping
        assert bounds1.intersects((5, 5, 15, 15))
        
        # Adjacent
        assert bounds1.intersects((10, 0, 20, 10))
        
        # Disjoint
        assert not bounds1.intersects((20, 20, 30, 30))
        
        # Contained
        assert bounds1.intersects((2, 2, 8, 8))
        
    def test_intersection(self):
        """Test bounds intersection calculation."""
        bounds1 = BoundsDefinition('test1', (0, 0, 10, 10))
        
        # Overlapping
        intersection = bounds1.intersection((5, 5, 15, 15))
        assert intersection == (5, 5, 10, 10)
        
        # Disjoint
        intersection = bounds1.intersection((20, 20, 30, 30))
        assert intersection is None
        
    def test_buffer(self):
        """Test bounds buffering."""
        bounds = BoundsDefinition('test', (0, 0, 10, 10))
        
        # Buffer by 111 km (~1 degree)
        buffered = bounds.buffer(111)
        assert buffered is not None  # Type hint for pyright
        
        assert buffered.bounds[0] < bounds.bounds[0]
        assert buffered.bounds[1] < bounds.bounds[1]
        assert buffered.bounds[2] > bounds.bounds[2]
        assert buffered.bounds[3] > bounds.bounds[3]
        assert buffered.metadata is not None
        assert buffered.metadata['buffered_km'] == 111


class TestBoundsManager:
    """Test BoundsManager class."""
    
    def test_predefined_regions(self, bounds_manager):
        """Test predefined regions are available."""
        assert 'global' in bounds_manager.REGIONS
        assert 'africa' in bounds_manager.REGIONS
        assert 'europe' in bounds_manager.REGIONS
        
        # Check region properties
        global_bounds = bounds_manager.REGIONS['global']
        assert global_bounds.bounds == (-180, -90, 180, 90)
        assert global_bounds.category == 'global'
        
    def test_get_bounds_predefined(self, bounds_manager):
        """Test getting predefined bounds."""
        # Get predefined region
        africa = bounds_manager.get_bounds('africa')
        assert africa.name == 'africa'
        assert africa.category == 'continent'
        
    def test_get_bounds_string_parse(self, bounds_manager):
        """Test parsing bounds from string."""
        # Parse comma-separated bounds
        bounds = bounds_manager.get_bounds('10,20,30,40')
        assert bounds.bounds == (10, 20, 30, 40)
        assert bounds.name == 'custom_bounds'
        
    def test_get_bounds_invalid(self, bounds_manager):
        """Test getting invalid bounds raises error."""
        with pytest.raises(ValueError) as exc_info:
            bounds_manager.get_bounds('nonexistent')
        
        assert "Unknown bounds" in str(exc_info.value)
        
    def test_custom_regions(self, bounds_manager, test_config):
        """Test loading custom regions from config."""
        # Manually add the test region since BoundsManager uses global config
        from src.grid_systems.bounds_manager import BoundsDefinition
        bounds_manager.custom_regions['test_region'] = BoundsDefinition(
            name='test_region',
            bounds=(0, 0, 10, 10),
            category='custom'
        )
        
        # Should have test_region
        custom = bounds_manager.get_bounds('test_region')
        assert custom.bounds == (0, 0, 10, 10)
        assert custom.category == 'custom'
        
    def test_list_available(self, bounds_manager):
        """Test listing available bounds."""
        available = bounds_manager.list_available()
        
        assert 'global' in available
        assert 'continent' in available
        assert 'africa' in available['continent']
        assert 'europe' in available['continent']
        
    def test_validate_bounds_overlap(self, bounds_manager):
        """Test bounds overlap validation."""
        bounds = BoundsDefinition('test', (0, 0, 10, 10))
        
        # Overlapping data
        assert bounds_manager.validate_bounds_overlap(bounds, (5, 5, 15, 15))
        
        # Non-overlapping data
        assert not bounds_manager.validate_bounds_overlap(bounds, (20, 20, 30, 30))
        
    def test_clip_grid_cells(self, bounds_manager):
        """Test clipping grid cells to bounds."""
        from src.base import GridCell
        from shapely.geometry import Polygon
        
        # Create test cells
        cells = [
            GridCell(
                cell_id='cell_1',
                geometry=Polygon([(0, 0), (5, 0), (5, 5), (0, 5), (0, 0)]),
                centroid=Point(2.5, 2.5),
                area_km2=25,
                bounds=(0, 0, 5, 5)
            ),
            GridCell(
                cell_id='cell_2',
                geometry=Polygon([(5, 0), (10, 0), (10, 5), (5, 5), (5, 0)]),
                centroid=Point(7.5, 2.5),
                area_km2=25,
                bounds=(5, 0, 10, 5)
            ),
            GridCell(
                cell_id='cell_3',
                geometry=Polygon([(10, 0), (15, 0), (15, 5), (10, 5), (10, 0)]),
                centroid=Point(12.5, 2.5),
                area_km2=25,
                bounds=(10, 0, 15, 5)
            )
        ]
        
        # Clip to bounds that excludes third cell
        clip_bounds = BoundsDefinition('clip', (0, 0, 8, 10))
        clipped = bounds_manager.clip_grid_cells(cells, clip_bounds)
        
        assert len(clipped) == 2
        assert clipped[0].cell_id == 'cell_1'  # Fully contained
        assert 'clipped' in clipped[1].cell_id  # Partially clipped
        
    def test_subdivide_bounds(self, bounds_manager):
        """Test bounds subdivision."""
        # Large bounds that need subdivision
        large_bounds = BoundsDefinition('large', (-180, -90, 180, 90))
        
        chunks = bounds_manager.subdivide_bounds(large_bounds, max_size_degrees=30)
        
        assert len(chunks) > 1
        
        # Check chunks cover original bounds
        total_area = 0
        for chunk in chunks:
            assert chunk.category == 'chunk'
            assert 'parent' in chunk.metadata
            assert chunk.metadata['parent'] == 'large'
            total_area += chunk.polygon.area
            
        # Total area should approximately equal original
        assert abs(total_area - large_bounds.polygon.area) < 0.1
        
    def test_subdivide_small_bounds(self, bounds_manager):
        """Test subdivision of small bounds returns single chunk."""
        small_bounds = BoundsDefinition('small', (0, 0, 5, 5))
        
        chunks = bounds_manager.subdivide_bounds(small_bounds, max_size_degrees=10)
        
        assert len(chunks) == 1
        assert chunks[0].bounds == small_bounds.bounds