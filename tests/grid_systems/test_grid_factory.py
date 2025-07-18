"""Tests for grid factory."""

import pytest
from unittest.mock import Mock, patch

from src.grid_systems import (
    GridFactory, GridSpecification, 
    CubicGrid, HexagonalGrid,
    BoundsDefinition
)


class TestGridSpecification:
    """Test GridSpecification dataclass."""
    
    def test_specification_creation(self):
        """Test creating grid specification."""
        spec = GridSpecification(
            grid_type='cubic',
            resolution=10000,
            bounds='global',
            name='test_grid'
        )
        
        assert spec.grid_type == 'cubic'
        assert spec.resolution == 10000
        assert spec.bounds == 'global'
        assert spec.name == 'test_grid'
        assert spec.crs == 'EPSG:4326'
        
    def test_specification_to_dict(self):
        """Test converting specification to dict."""
        spec = GridSpecification(
            grid_type='hexagonal',
            resolution=25000,
            bounds=BoundsDefinition('test', (0, 0, 10, 10)),
            metadata={'key': 'value'}
        )
        
        spec_dict = spec.to_dict()
        assert spec_dict['grid_type'] == 'hexagonal'
        assert spec_dict['resolution'] == 25000
        assert spec_dict['bounds'] == 'test'  # BoundsDefinition name
        assert spec_dict['metadata'] == {'key': 'value'}


class TestGridFactory:
    """Test GridFactory class."""
    
    def test_factory_creation(self):
        """Test factory initialization."""
        factory = GridFactory()
        
        assert factory.bounds_manager is not None
        assert hasattr(factory, '_grid_cache')
        assert len(factory.STANDARD_RESOLUTIONS) == 5
        
    def test_create_grid_from_spec(self, mock_config):
        """Test creating grid from specification."""
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
        
    def test_create_grid_from_dict(self, mock_config):
        """Test creating grid from dictionary."""
        factory = GridFactory()
        
        grid = factory.create_grid({
            'grid_type': 'hexagonal',
            'resolution': 25000,
            'bounds': 'africa'
        })
        
        assert isinstance(grid, HexagonalGrid)
        assert grid.resolution == 25000
        
    def test_create_grid_invalid_type(self):
        """Test creating grid with invalid type."""
        factory = GridFactory()
        
        with pytest.raises(ValueError) as exc_info:
            factory.create_grid({
                'grid_type': 'invalid',
                'resolution': 10000,
                'bounds': 'global'
            })
        
        assert "Unknown grid type" in str(exc_info.value)
        
    def test_create_multi_resolution_grids(self, mock_config):
        """Test creating multiple resolution grids."""
        factory = GridFactory()
        
        grids = factory.create_multi_resolution_grids(
            grid_type='cubic',
            resolutions=[100000, 50000, 25000],
            bounds='europe',
            base_name='test_multi'
        )
        
        assert len(grids) == 3
        assert 100000 in grids
        assert 50000 in grids
        assert 25000 in grids
        
        # Check grids are properly configured
        assert grids[100000].resolution == 100000
        assert grids[50000].resolution == 50000
        assert grids[25000].resolution == 25000
        
        # Check naming
        assert grids[100000].specification.name == 'test_multi_100000m'
        
    @patch('src.database.schema.schema.get_grid_by_name')
    @patch.object(CubicGrid, 'store_grid')
    def test_store_grid(self, mock_store, mock_get, mock_config):
        """Test storing grid in database."""
        factory = GridFactory()
        
        # Mock database responses
        mock_get.return_value = None  # Grid doesn't exist
        mock_store.return_value = 'grid_123'
        
        # Create and store grid
        grid = factory.create_grid({
            'grid_type': 'cubic',
            'resolution': 10000,
            'bounds': 'global',
            'name': 'test_store'
        })
        
        grid_id = factory.store_grid(grid)
        
        assert grid_id == 'grid_123'
        mock_store.assert_called_once()
        
    @patch('src.database.schema.schema.get_grid_by_name')
    def test_store_grid_exists(self, mock_get, mock_config):
        """Test storing grid that already exists."""
        factory = GridFactory()
        
        # Mock existing grid
        mock_get.return_value = {'id': 'existing_grid'}
        
        grid = factory.create_grid({
            'grid_type': 'cubic',
            'resolution': 10000,
            'bounds': 'global',
            'name': 'existing'
        })
        
        # Should raise error
        with pytest.raises(ValueError) as exc_info:
            factory.store_grid(grid)
        
        assert "already exists" in str(exc_info.value)
        
    def test_create_standard_grids(self, mock_config):
        """Test creating standard resolution grids."""
        factory = GridFactory()
        
        grids = factory.create_standard_grids(
            grid_type='cubic',
            bounds='global',
            name_prefix='test'
        )
        
        # Should have all standard resolutions
        assert 'coarse' in grids
        assert 'medium' in grids
        assert 'fine' in grids
        assert 'very_fine' in grids
        assert 'ultra_fine' in grids
        
        # Check resolutions match
        assert grids['coarse'].resolution == 100000
        assert grids['fine'].resolution == 25000
        
    def test_upscale_data(self, mock_config):
        """Test upscaling data between resolutions."""
        factory = GridFactory()
        
        # Create two grids
        fine_grid = factory.create_grid({
            'grid_type': 'cubic',
            'resolution': 10000,
            'bounds': (0, 0, 2, 2)
        })
        
        coarse_grid = factory.create_grid({
            'grid_type': 'cubic',
            'resolution': 100000,
            'bounds': (0, 0, 2, 2)
        })
        
        # Create sample data for fine grid
        fine_cells = fine_grid.get_cells()[:4]
        data = {cell.cell_id: float(i) for i, cell in enumerate(fine_cells)}
        
        # Upscale
        upscaled = factory.upscale_data(
            data, fine_grid, coarse_grid, 
            aggregation='mean'
        )
        
        assert len(upscaled) <= len(coarse_grid.get_cells())
        assert all(isinstance(v, float) for v in upscaled.values())
        
    def test_validate_grid_compatibility(self, mock_config):
        """Test grid compatibility validation."""
        factory = GridFactory()
        
        # Compatible grids (same CRS, overlapping bounds)
        grid1 = factory.create_grid({
            'grid_type': 'cubic',
            'resolution': 10000,
            'bounds': (0, 0, 10, 10)
        })
        
        grid2 = factory.create_grid({
            'grid_type': 'hexagonal',
            'resolution': 25000,
            'bounds': (5, 5, 15, 15)
        })
        
        assert factory.validate_grid_compatibility(grid1, grid2)
        
        # Non-overlapping grids
        grid3 = factory.create_grid({
            'grid_type': 'cubic',
            'resolution': 10000,
            'bounds': (20, 20, 30, 30)
        })
        
        assert not factory.validate_grid_compatibility(grid1, grid3)