# tests/test_exporters/test_csv_exporter.py
"""Tests for CSV exporter."""

import pytest
import csv
import gzip
from pathlib import Path
import numpy as np

from src.processors.exporters.csv_exporter import CSVExporter, ExportConfig


class TestCSVExporter:
    """Test CSV exporter functionality."""
    
    def test_export_basic(self, mock_db, temp_dir):
        """Test basic CSV export."""
        exporter = CSVExporter(mock_db)
        
        # Mock database responses
        with mock_db.get_connection() as conn:
            cur = conn.cursor()
            
            # Mock dataset query
            cur.execute = Mock(side_effect=[
                # First call - get datasets
                Mock(fetchall=lambda: [('test-plants', 'plants_richness')]),
                # Second call - get table info
                Mock(fetchone=lambda: ('resampled_test_plants', [10, 10], [-5, -5, 5, 5])),
                # Subsequent calls for values
                *[Mock(fetchone=lambda: (i * 0.1,)) for i in range(100)]
            ])
        
        config = ExportConfig(
            output_path=temp_dir / "test.csv",
            chunk_size=50
        )
        
        dataset_info = {
            'experiment_id': 'test_exp',
            'dataset_names': ['test-plants']
        }
        
        output_file = exporter.export(dataset_info, config)
        
        # Verify file created
        assert output_file.exists()
        
        # Verify content
        with open(output_file, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            
            assert headers == ['cell_id', 'x', 'y', 'plants_richness']
            
            # Check first row
            first_row = next(reader)
            assert first_row[0] == 'cell_0'
            assert len(first_row) == 4
    
    def test_export_with_compression(self, mock_db, temp_dir):
        """Test CSV export with gzip compression."""
        exporter = CSVExporter(mock_db)
        
        config = ExportConfig(
            output_path=temp_dir / "test.csv",
            compression='gzip'
        )
        
        # ... setup mocks ...
        
        output_file = exporter.export({'experiment_id': 'test'}, config)
        
        # Verify compressed file
        assert output_file.suffix == '.gz'
        assert output_file.exists()
        
        # Verify can read compressed content
        with gzip.open(output_file, 'rt') as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert 'cell_id' in headers
    
    def test_validate_export(self, temp_dir):
        """Test export validation."""
        exporter = CSVExporter(None)
        
        # Create valid CSV
        valid_csv = temp_dir / "valid.csv"
        with open(valid_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['cell_id', 'x', 'y', 'value'])
            writer.writerow(['cell_0', '0.0', '0.0', '1.0'])
        
        assert exporter.validate_export(valid_csv)
        
        # Test invalid cases
        assert not exporter.validate_export(temp_dir / "nonexistent.csv")
        
        # Empty file
        empty_csv = temp_dir / "empty.csv"
        empty_csv.touch()
        assert not exporter.validate_export(empty_csv)