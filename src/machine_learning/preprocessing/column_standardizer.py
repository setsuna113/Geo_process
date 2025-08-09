"""
Column standardizer for ML regression analysis.

Handles column name standardization and memory-efficient data preprocessing,
including fungi_richness calculation from am_fungi_richness and ecm_fungi_richness.

Uses memory-efficient techniques inspired by MESF implementation:
1. In-place operations when possible
2. Batch processing for large datasets
3. PyArrow for efficient parquet operations
"""

import logging
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import tempfile
import os

logger = logging.getLogger(__name__)


class ColumnStandardizer:
    """Standardizes column names and creates derived columns efficiently."""
    
    # Column name mappings
    COLUMN_MAPPINGS = {
        # Coordinate columns
        'x': 'longitude',
        'y': 'latitude',
        
        # Biodiversity columns  
        'animal_richness': 'vertebrates_richness',
        'plant_richness': 'plants_richness',  # Keep as is
        
        # Climate columns (standardize bio01 -> bio1, etc.)
        'bio01': 'bio1',
        'bio04': 'bio4',
        'bio12': 'bio12',  # Already correct
        'bio15': 'bio15',  # Already correct
        
        # Keep elevation as is
        'elevation': 'elevation'
    }
    
    def __init__(self, batch_size: int = 1000000):
        """
        Initialize column standardizer.
        
        Args:
            batch_size: Size of batches for processing large files
        """
        self.batch_size = batch_size
        
    def standardize_parquet(self, 
                           input_path: str,
                           output_path: Optional[str] = None,
                           compute_fungi_richness: bool = True) -> str:
        """
        Standardize columns in a parquet file.
        
        Args:
            input_path: Path to input parquet file
            output_path: Path to output parquet file (if None, creates temp file)
            compute_fungi_richness: Whether to compute total fungi_richness
            
        Returns:
            Path to standardized parquet file
        """
        logger.info(f"Standardizing columns in {input_path}")
        
        # Open parquet file for reading
        parquet_file = pq.ParquetFile(input_path)
        schema = parquet_file.schema
        
        # Check which columns need standardization
        columns_to_rename = {}
        for old_name in schema.names:
            if old_name in self.COLUMN_MAPPINGS:
                new_name = self.COLUMN_MAPPINGS[old_name]
                if old_name != new_name:
                    columns_to_rename[old_name] = new_name
                    
        # Check if fungi columns exist for computing total
        has_am_fungi = 'am_fungi_richness' in schema.names
        has_ecm_fungi = 'ecm_fungi_richness' in schema.names
        needs_fungi_total = compute_fungi_richness and has_am_fungi and has_ecm_fungi
        
        # If no changes needed, return original
        if not columns_to_rename and not needs_fungi_total:
            logger.info("No column standardization needed")
            return input_path
            
        # Create output path if not specified
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='_standardized.parquet', delete=False)
            output_path = temp_file.name
            temp_file.close()
            
        logger.info(f"Processing {parquet_file.num_row_groups} row groups...")
        
        # Process in batches to manage memory
        tables = []
        for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=self.batch_size)):
            if batch_idx % 10 == 0:
                logger.info(f"Processing batch {batch_idx+1}...")
                
            # Convert to PyArrow table
            table = pa.Table.from_batches([batch])
            
            # Rename columns
            for old_name, new_name in columns_to_rename.items():
                if old_name in table.column_names:
                    table = table.rename_columns({old_name: new_name})
            
            # Compute fungi_richness if needed
            if needs_fungi_total and 'fungi_richness' not in table.column_names:
                table = self._add_fungi_richness(table)
                
            tables.append(table)
            
        # Combine all tables
        logger.info("Combining processed batches...")
        final_table = pa.concat_tables(tables)
        
        # Write to output
        logger.info(f"Writing standardized data to {output_path}")
        pq.write_table(final_table, output_path, compression='snappy')
        
        # Log summary
        logger.info(f"Standardization complete:")
        logger.info(f"  - Renamed columns: {list(columns_to_rename.keys())}")
        if needs_fungi_total:
            logger.info(f"  - Added fungi_richness column (sum of am_fungi and ecm_fungi)")
        logger.info(f"  - Output rows: {final_table.num_rows:,}")
        logger.info(f"  - Output columns: {len(final_table.column_names)}")
        
        return output_path
    
    def _add_fungi_richness(self, table: pa.Table) -> pa.Table:
        """
        Add fungi_richness column as sum of am_fungi_richness and ecm_fungi_richness.
        
        Uses PyArrow compute functions for efficiency and memory safety.
        
        Args:
            table: PyArrow table
            
        Returns:
            Table with added fungi_richness column
        """
        # Get the fungi columns
        am_fungi = table.column('am_fungi_richness')
        ecm_fungi = table.column('ecm_fungi_richness')
        
        # Compute sum using PyArrow compute (memory efficient)
        # Handle nulls by filling with 0
        am_filled = pc.fill_null(am_fungi, 0)
        ecm_filled = pc.fill_null(ecm_fungi, 0)
        
        # Add the two columns
        fungi_total = pc.add(am_filled, ecm_filled)
        
        # Add as new column to table
        table = table.append_column('fungi_richness', fungi_total)
        
        return table
    
    def check_required_columns(self, parquet_path: str, method: str = 'all') -> Dict[str, bool]:
        """
        Check if required columns exist after standardization.
        
        Args:
            parquet_path: Path to parquet file
            method: Which method to check for ('method1', 'method2', 'method3', 'all')
            
        Returns:
            Dictionary of column existence status
        """
        # Define required columns for each method
        base_columns = [
            'longitude', 'latitude',  # Or 'x', 'y'
            'bio1', 'bio4', 'bio12', 'bio15',
            'elevation'
        ]
        
        method1_columns = base_columns + [
            'fungi_richness',
            'plants_richness', 
            'vertebrates_richness'
        ]
        
        method2_columns = method1_columns  # Same requirements
        
        method3_columns = method1_columns  # Same requirements
        
        # Select columns to check
        if method == 'method1':
            required = method1_columns
        elif method == 'method2':
            required = method2_columns
        elif method == 'method3':
            required = method3_columns
        else:  # 'all'
            required = list(set(method1_columns + method2_columns + method3_columns))
            
        # Check parquet schema
        parquet_file = pq.ParquetFile(parquet_path)
        available_columns = parquet_file.schema.names
        
        # Check each required column
        status = {}
        for col in required:
            # Check both standardized and original names
            if col in available_columns:
                status[col] = True
            elif col == 'longitude' and 'x' in available_columns:
                status[col] = True
            elif col == 'latitude' and 'y' in available_columns:
                status[col] = True
            elif col == 'vertebrates_richness' and 'animal_richness' in available_columns:
                status[col] = True
            elif col == 'bio1' and 'bio01' in available_columns:
                status[col] = True
            elif col == 'bio4' and 'bio04' in available_columns:
                status[col] = True
            else:
                status[col] = False
                
        # Log missing columns
        missing = [col for col, exists in status.items() if not exists]
        if missing:
            logger.warning(f"Missing required columns: {missing}")
            
        return status
    
    def get_standardized_path(self, input_path: str, suffix: str = '_standardized') -> str:
        """
        Get the path for standardized file.
        
        Args:
            input_path: Original file path
            suffix: Suffix to add to filename
            
        Returns:
            Path for standardized file
        """
        input_path = Path(input_path)
        return str(input_path.parent / f"{input_path.stem}{suffix}{input_path.suffix}")
    
    def standardize_if_needed(self, parquet_path: str) -> str:
        """
        Check if standardization is needed and perform if necessary.
        
        This is the main entry point for the preprocessing pipeline.
        
        Args:
            parquet_path: Path to input parquet file
            
        Returns:
            Path to standardized parquet (may be same as input if no changes needed)
        """
        # Check if already standardized
        status = self.check_required_columns(parquet_path)
        
        # Check if we have fungi_richness or can compute it
        parquet_file = pq.ParquetFile(parquet_path)
        columns = parquet_file.schema.names
        
        has_fungi_richness = 'fungi_richness' in columns
        can_compute_fungi = ('am_fungi_richness' in columns and 
                            'ecm_fungi_richness' in columns)
        
        # Determine if standardization is needed
        needs_standardization = (
            not all(status.values()) or  # Missing required columns
            (not has_fungi_richness and can_compute_fungi)  # Can compute fungi_richness
        )
        
        if not needs_standardization:
            logger.info("Data already standardized")
            return parquet_path
            
        # Perform standardization
        standardized_path = self.get_standardized_path(parquet_path)
        
        # Check if standardized version already exists
        if Path(standardized_path).exists():
            logger.info(f"Using existing standardized file: {standardized_path}")
            return standardized_path
            
        # Create standardized version
        return self.standardize_parquet(
            parquet_path,
            standardized_path,
            compute_fungi_richness=can_compute_fungi
        )