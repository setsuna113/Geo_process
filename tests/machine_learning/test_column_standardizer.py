"""Tests for Column Standardizer."""

import pytest
import tempfile
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.machine_learning.preprocessing import ColumnStandardizer


class TestColumnStandardizer:
    """Test suite for Column Standardizer."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def sample_parquet_needs_standardization(self, temp_dir):
        """Create parquet file that needs standardization."""
        df = pd.DataFrame({
            'x': [1.0, 2.0, 3.0],
            'y': [4.0, 5.0, 6.0],
            'animal_richness': [10, 20, 30],
            'plant_richness': [100, 200, 300],
            'am_fungi_richness': [5, 10, 15],
            'ecm_fungi_richness': [7, 14, 21],
            'bio01': [15.5, 16.0, 16.5],
            'bio04': [1.2, 1.3, 1.4],
            'bio12': [800, 900, 1000],
            'bio15': [50, 55, 60],
            'elevation': [100, 200, 300]
        })
        file_path = temp_dir / "needs_standardization.parquet"
        df.to_parquet(file_path)
        return str(file_path)
    
    @pytest.fixture
    def sample_parquet_already_standardized(self, temp_dir):
        """Create parquet file already standardized."""
        df = pd.DataFrame({
            'longitude': [1.0, 2.0, 3.0],
            'latitude': [4.0, 5.0, 6.0],
            'vertebrates_richness': [10, 20, 30],
            'plants_richness': [100, 200, 300],
            'am_fungi_richness': [5, 10, 15],
            'ecm_fungi_richness': [7, 14, 21],
            'fungi_richness': [12, 24, 36],  # sum of am + ecm
            'bio1': [15.5, 16.0, 16.5],
            'bio4': [1.2, 1.3, 1.4],
            'bio12': [800, 900, 1000],
            'bio15': [50, 55, 60],
            'elevation': [100, 200, 300]
        })
        file_path = temp_dir / "already_standardized.parquet"
        df.to_parquet(file_path)
        return str(file_path)
    
    @pytest.fixture
    def standardizer(self):
        """Create standardizer instance."""
        return ColumnStandardizer(batch_size=100)  # Small batch for testing
    
    def test_standardize_parquet_renames_columns(self, standardizer, sample_parquet_needs_standardization, temp_dir):
        """Test that standardize_parquet renames columns correctly."""
        output_path = str(temp_dir / "output.parquet")
        
        result_path = standardizer.standardize_parquet(
            sample_parquet_needs_standardization,
            output_path,
            compute_fungi_richness=False
        )
        
        # Check result
        assert result_path == output_path
        pf = pq.ParquetFile(output_path)
        columns = pf.schema.names
        
        # Check renamed columns
        assert 'longitude' in columns  # x -> longitude
        assert 'latitude' in columns   # y -> latitude
        assert 'vertebrates_richness' in columns  # animal_richness -> vertebrates_richness
        assert 'plants_richness' in columns  # plant_richness -> plants_richness (keeps s)
        assert 'bio1' in columns  # bio01 -> bio1
        assert 'bio4' in columns  # bio04 -> bio4
        
        # Check old names are gone
        assert 'x' not in columns
        assert 'y' not in columns
        assert 'animal_richness' not in columns
        assert 'bio01' not in columns
    
    def test_standardize_parquet_computes_fungi_richness(self, standardizer, sample_parquet_needs_standardization, temp_dir):
        """Test that fungi_richness is computed as sum of am + ecm."""
        output_path = str(temp_dir / "output_with_fungi.parquet")
        
        result_path = standardizer.standardize_parquet(
            sample_parquet_needs_standardization,
            output_path,
            compute_fungi_richness=True
        )
        
        # Read result and check fungi_richness
        df = pd.read_parquet(result_path)
        
        assert 'fungi_richness' in df.columns
        # Check it's the sum (not max as tests previously assumed)
        expected = [12, 24, 36]  # [5+7, 10+14, 15+21]
        np.testing.assert_array_equal(df['fungi_richness'].values, expected)
    
    def test_standardize_parquet_no_changes_returns_original(self, standardizer, sample_parquet_already_standardized):
        """Test that files needing no changes return original path."""
        result_path = standardizer.standardize_parquet(
            sample_parquet_already_standardized,
            compute_fungi_richness=False  # Already has fungi_richness
        )
        
        # Should return original path
        assert result_path == sample_parquet_already_standardized
    
    def test_add_fungi_richness_handles_nulls(self, standardizer, temp_dir):
        """Test that fungi_richness handles null values correctly."""
        # Create data with nulls
        df = pd.DataFrame({
            'am_fungi_richness': [5.0, 10.0, None, None],
            'ecm_fungi_richness': [7.0, None, 15.0, None]
        })
        table = pa.Table.from_pandas(df)
        
        # Add fungi_richness
        result_table = standardizer._add_fungi_richness(table)
        
        # Check results
        result_df = result_table.to_pandas()
        assert 'fungi_richness' in result_df.columns
        
        # Nulls are filled with 0, so we get sum with 0 for missing values
        expected = [12.0, 10.0, 15.0, 0.0]  # [5+7, 10+0, 0+15, 0+0]
        np.testing.assert_array_equal(result_df['fungi_richness'].values, expected)
    
    def test_check_required_columns_all_methods(self, standardizer, sample_parquet_already_standardized):
        """Test checking required columns for all methods."""
        status = standardizer.check_required_columns(
            sample_parquet_already_standardized,
            method='all'
        )
        
        # All required columns should be present
        assert status['longitude'] == True
        assert status['latitude'] == True
        assert status['bio1'] == True
        assert status['bio4'] == True
        assert status['bio12'] == True
        assert status['bio15'] == True
        assert status['elevation'] == True
        assert status['vertebrates_richness'] == True
        assert status['plants_richness'] == True
        assert status['fungi_richness'] == True
    
    def test_check_required_columns_accepts_original_names(self, standardizer, sample_parquet_needs_standardization):
        """Test that check accepts both original and standardized names."""
        status = standardizer.check_required_columns(
            sample_parquet_needs_standardization,
            method='all'
        )
        
        # Should recognize x/y as longitude/latitude
        assert status['longitude'] == True  # x is present
        assert status['latitude'] == True   # y is present
        assert status['vertebrates_richness'] == True  # animal_richness is present
        assert status['bio1'] == True  # bio01 is present
    
    def test_get_standardized_path(self, standardizer):
        """Test generation of standardized file path."""
        input_path = "/path/to/data.parquet"
        result = standardizer.get_standardized_path(input_path)
        
        assert result == "/path/to/data_standardized.parquet"
    
    def test_standardize_if_needed_creates_standardized(self, standardizer, sample_parquet_needs_standardization):
        """Test that standardize_if_needed creates standardized file when needed."""
        result_path = standardizer.standardize_if_needed(sample_parquet_needs_standardization)
        
        # Should create new standardized file
        assert "_standardized" in result_path
        assert Path(result_path).exists()
        
        # Check standardization happened
        df = pd.read_parquet(result_path)
        assert 'longitude' in df.columns
        assert 'vertebrates_richness' in df.columns
        assert 'fungi_richness' in df.columns  # Should be computed
    
    def test_standardize_if_needed_returns_original_when_standardized(self, standardizer, sample_parquet_already_standardized):
        """Test that standardize_if_needed returns original for already standardized files."""
        result_path = standardizer.standardize_if_needed(sample_parquet_already_standardized)
        
        # Should return original path (no changes needed)
        assert result_path == sample_parquet_already_standardized
    
    def test_standardize_if_needed_uses_cache(self, standardizer, sample_parquet_needs_standardization):
        """Test that standardize_if_needed uses existing standardized file."""
        # First call creates standardized file
        first_result = standardizer.standardize_if_needed(sample_parquet_needs_standardization)
        
        # Modify the standardized file to add a marker
        df = pd.read_parquet(first_result)
        df['test_marker'] = 999
        df.to_parquet(first_result)
        
        # Second call should return cached file
        second_result = standardizer.standardize_if_needed(sample_parquet_needs_standardization)
        
        assert first_result == second_result
        
        # Verify it used the cached file (has our marker)
        df_cached = pd.read_parquet(second_result)
        assert 'test_marker' in df_cached.columns
        assert df_cached['test_marker'].iloc[0] == 999
    
    def test_batch_processing(self, standardizer, temp_dir):
        """Test that large files are processed in batches."""
        # Create larger dataset
        n_rows = 500
        df = pd.DataFrame({
            'x': np.arange(n_rows, dtype=float),
            'y': np.arange(n_rows, dtype=float),
            'animal_richness': np.random.randint(0, 100, n_rows),
            'plant_richness': np.random.randint(0, 1000, n_rows),
            'am_fungi_richness': np.random.randint(0, 50, n_rows),
            'ecm_fungi_richness': np.random.randint(0, 50, n_rows),
            'bio01': np.random.uniform(10, 20, n_rows),
            'elevation': np.random.uniform(0, 3000, n_rows)
        })
        
        input_path = temp_dir / "large_file.parquet"
        df.to_parquet(input_path)
        
        output_path = temp_dir / "large_output.parquet"
        
        # Process with small batch size
        standardizer_small_batch = ColumnStandardizer(batch_size=100)
        result_path = standardizer_small_batch.standardize_parquet(
            str(input_path),
            str(output_path)
        )
        
        # Verify all rows processed
        df_result = pd.read_parquet(result_path)
        assert len(df_result) == n_rows
        assert 'fungi_richness' in df_result.columns
        assert 'longitude' in df_result.columns
    
    def test_preserve_data_integrity(self, standardizer, sample_parquet_needs_standardization):
        """Test that data values are preserved during standardization."""
        # Read original
        df_original = pd.read_parquet(sample_parquet_needs_standardization)
        
        # Standardize
        output_path = standardizer.standardize_parquet(
            sample_parquet_needs_standardization,
            compute_fungi_richness=True
        )
        df_standardized = pd.read_parquet(output_path)
        
        # Check values are preserved (just renamed)
        np.testing.assert_array_equal(df_standardized['longitude'].values, df_original['x'].values)
        np.testing.assert_array_equal(df_standardized['latitude'].values, df_original['y'].values)
        np.testing.assert_array_equal(df_standardized['vertebrates_richness'].values, df_original['animal_richness'].values)
        np.testing.assert_array_equal(df_standardized['bio1'].values, df_original['bio01'].values)