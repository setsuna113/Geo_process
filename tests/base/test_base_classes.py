"""Test suite for base classes."""

import pytest
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from shapely.geometry import Point, Polygon

from src.base import (
    BaseProcessor, ProcessingResult,
    BaseGrid, GridCell,
    BaseFeature, FeatureResult,
    BaseDataset, DatasetInfo
)
from src.base.feature import SourceType
from src.base.dataset import DataType

class TestBaseProcessor:
    """Test BaseProcessor class."""
    
    class DummyProcessor(BaseProcessor):
        """Dummy processor for testing."""
        
        def process_single(self, item: int) -> int:
            """Double the input."""
            time.sleep(0.01)  # Simulate processing
            return item * 2
            
        def validate_input(self, item: Any) -> Tuple[bool, Optional[str]]:
            """Validate item is positive integer."""
            if not isinstance(item, int):
                return False, "Item must be integer"
            if item < 0:
                return False, "Item must be positive"
            return True, None
            
    def test_process_single_item(self):
        """Test processing single items."""
        processor = self.DummyProcessor()
        
        # Valid item
        result = processor.process_single(5)
        assert result == 10
        
    def test_batch_processing(self):
        """Test batch processing."""
        processor = self.DummyProcessor(batch_size=10)
        items = list(range(20))
        
        result = processor.process_batch(items)
        
        assert result.success
        assert result.items_processed == 20
        assert result.items_failed == 0
        assert result.elapsed_time > 0
        
    def test_validation(self):
        """Test input validation."""
        processor = self.DummyProcessor()
        
        # Valid
        is_valid, error = processor.validate_input(5)
        assert is_valid
        assert error is None
        
        # Invalid type
        is_valid, error = processor.validate_input("5")
        assert not is_valid
        assert error is not None, "Expected an error message, but got None."
        assert "must be integer" in error
        
        # Invalid value
        is_valid, error = processor.validate_input(-5)
        assert not is_valid
        assert error is not None, "Expected an error message, but got None."
        assert "must be positive" in error
        
    def test_memory_tracking(self):
        """Test memory tracking."""
        processor = self.DummyProcessor()
        
        with processor.track_memory("test_operation"):
            # Allocate some memory
            data = [i for i in range(100000)]
            
        summary = processor.memory_tracker.get_summary()
        assert summary['total_mb'] >= 0
        assert len(summary['operations']) == 1
        
    def test_error_handling(self):
        """Test error handling in batch processing."""
        processor = self.DummyProcessor()
        
        # Mix valid and invalid items
        items = [1, "invalid", 3, -1, 5]
        
        result = processor.process_batch(items)
        
        assert result.items_processed == 3  # 1, 3, 5
        assert result.items_failed == 2     # "invalid", -1
        assert result.errors is not None, "Expected a list of errors, but got None."
        assert len(result.errors) == 2
        

class TestBaseGrid:
    """Test BaseGrid class."""
    
    class DummyGrid(BaseGrid):
        """Dummy grid for testing."""
        
        def generate_grid(self) -> List[GridCell]:
            """Generate a simple 2x2 grid."""
            cells = []
            cell_size = self.resolution / 111000  # Approximate degrees
            
            for i in range(2):
                for j in range(2):
                    minx = self.bounds[0] + i * cell_size
                    miny = self.bounds[1] + j * cell_size
                    maxx = minx + cell_size
                    maxy = miny + cell_size
                    
                    cell = GridCell(
                        cell_id=f"{i}_{j}",
                        geometry=Polygon([
                            (minx, miny), (maxx, miny),
                            (maxx, maxy), (minx, maxy),
                            (minx, miny)
                        ]),
                        centroid=Point((minx + maxx) / 2, (miny + maxy) / 2),
                        area_km2=(cell_size * 111) ** 2,
                        bounds=(minx, miny, maxx, maxy)
                    )
                    cells.append(cell)
                    
            return cells
            
        def get_cell_id(self, x: float, y: float) -> str:
            """Get cell ID for coordinate."""
            cell_size = self.resolution / 111000
            i = int((x - self.bounds[0]) / cell_size)
            j = int((y - self.bounds[1]) / cell_size)
            return f"{i}_{j}"
            
        def get_cell_by_id(self, cell_id: str) -> Optional[GridCell]:
            """Get cell by ID."""
            for cell in self.get_cells():
                if cell.cell_id == cell_id:
                    return cell
            return None
            
    def test_grid_generation(self):
        """Test grid generation."""
        grid = self.DummyGrid(
            resolution=1000,
            bounds=(0, 0, 0.02, 0.02)
        )
        
        cells = grid.generate_grid()
        assert len(cells) == 4
        
        # Check cell properties
        for cell in cells:
            assert isinstance(cell.geometry, Polygon)
            assert isinstance(cell.centroid, Point)
            assert cell.area_km2 > 0
            
    def test_cell_lookup(self):
        """Test cell lookup methods."""
        grid = self.DummyGrid(
            resolution=1000,
            bounds=(0, 0, 0.02, 0.02)
        )
        
        # By coordinate
        cell_id = grid.get_cell_id(0.015, 0.015)
        assert cell_id == "1_1"
        
        # By ID
        cell = grid.get_cell_by_id("0_1")
        assert cell is not None
        assert cell.cell_id == "0_1"
        
    def test_spatial_queries(self):
        """Test spatial queries."""
        grid = self.DummyGrid(
            resolution=1000,
            bounds=(0, 0, 0.02, 0.02)
        )
        
        # Get cells in bounds
        cells = grid.get_cells_in_bounds((0, 0, 0.01, 0.01))
        assert len(cells) > 0
        
        # Get cells for geometry
        test_polygon = Polygon([
            (0.005, 0.005), (0.015, 0.005),
            (0.015, 0.015), (0.005, 0.015),
            (0.005, 0.005)
        ])
        cells = grid.get_cells_for_geometry(test_polygon)
        assert len(cells) >= 1
        

class TestBaseFeature:
    """Test BaseFeature class."""
    
class DummyFeature(BaseFeature):
    """Test feature extractor implementation."""
    
    def __init__(self):
        super().__init__(feature_type="test", store_results=False)
        self.test_data = "dummy"
    
    @property
    def source_type(self) -> SourceType:
        """Get source type for testing."""
        return SourceType.RASTER
    
    def extract_single(self, grid_cell_id: str, data: Dict[str, Any]) -> List[FeatureResult]:
        """Extract dummy features for a single cell."""
        return [FeatureResult(
            feature_name="dummy_feature",
            feature_type="test",
            value=1.0,
            metadata={"test": True}
        )]
    
    def get_required_data(self) -> List[str]:
        """Get required data types."""
        return ["test_data"]
    
    def get_feature_count(self) -> int:
        """Get number of features."""
        return 3

    def test_feature_extraction(self):
        """Test feature extraction."""
        extractor = DummyFeature()
        
        # Single cell
        features = extractor.extract_single("cell_1", {'items': [1, 2, 3]})
        assert len(features) == 2
        assert features[0].value == 3.0  # count
        assert features[1].value == 6.0  # sum
        
    def test_batch_extraction(self):
        """Test batch feature extraction."""
        extractor = DummyFeature()
        
        cell_data = {
            'cell_1': {'items': [1, 2, 3]},
            'cell_2': {'items': [4, 5]},
            'cell_3': {'invalid': 'data'}  # Missing required data
        }
        
        results = extractor.extract_batch("grid_1", cell_data)
        
        assert len(results) == 2  # Only valid cells
        assert 'cell_1' in results
        assert 'cell_2' in results
        assert 'cell_3' not in results
        
    def test_normalization(self):
        """Test feature normalization."""
        extractor = DummyFeature()
        
        features = [
            FeatureResult("test", "test", 0.0),
            FeatureResult("test", "test", 50.0),
            FeatureResult("test", "test", 100.0)
        ]
        
        normalized = extractor._normalize_features(features)
        
        assert normalized[0].value == 0.0
        assert normalized[1].value == 0.5
        assert normalized[2].value == 1.0


class DummyDataset(BaseDataset):
    """Dummy dataset for testing."""
    
    @property
    def data_type(self) -> DataType:
        """Get the data type of this dataset."""
        return DataType.TABULAR
    
    def load_info(self) -> DatasetInfo:
        """Load dataset info."""
        return DatasetInfo(
            name="test_dataset",
            source=str(self.source),
            format="dummy",
            size_mb=10.5,
            record_count=1000,
            bounds=(0, 0, 10, 10),
            crs="EPSG:4326",
            metadata={'test': True},
            data_type=DataType.TABULAR
        )
        
    def read_records(self):
        """Read dummy records."""
        for i in range(10):
            yield {'id': i, 'value': i * 10}
            
    def read_chunks(self):
        """Read in chunks."""
        chunk = []
        for i, record in enumerate(self.read_records()):
            chunk.append(record)
            if len(chunk) >= self.chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk
            
    def validate_record(self, record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate record has required fields."""
        if 'id' not in record:
            return False, "Missing 'id' field"
        if 'value' not in record:
            return False, "Missing 'value' field"
        return True, None
        

class TestBaseDataset:
    """Test BaseDataset class."""
            
    def test_dataset_info(self):
        """Test dataset info loading."""
        dataset = DummyDataset("dummy.txt")
        
        info = dataset.get_info()
        assert info.name == "test_dataset"
        assert info.record_count == 1000
        assert info.size_mb == 10.5
        
    def test_record_reading(self):
        """Test record reading."""
        dataset = DummyDataset("dummy.txt")
        
        records = list(dataset.read_records())
        assert len(records) == 10
        assert records[0]['id'] == 0
        assert records[9]['value'] == 90
        
    def test_chunked_reading(self):
        """Test chunked reading."""
        dataset = DummyDataset("dummy.txt", chunk_size=3)
        
        chunks = list(dataset.read_chunks())
        assert len(chunks) == 4  # 3, 3, 3, 1
        assert len(chunks[0]) == 3
        assert len(chunks[-1]) == 1
        
    def test_filtering(self):
        """Test record filtering."""
        dataset = DummyDataset("dummy.txt")
        
        # Filter even values
        filtered = list(dataset.filter_records(
            lambda r: r['value'] % 20 == 0
        ))
        
        assert len(filtered) == 5  # 0, 20, 40, 60, 80
        assert all(r['value'] % 20 == 0 for r in filtered)
        
    def test_sampling(self):
        """Test record sampling."""
        dataset = DummyDataset("dummy.txt")
        
        sample = dataset.sample_records(5, seed=42)
        assert len(sample) == 5
        assert all(isinstance(r['id'], int) for r in sample)