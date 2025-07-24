# src/processors/exporters/csv_exporter.py
"""CSV exporter for resampled datasets."""

import csv
import logging
from typing import Dict, Any, Optional, Iterator, List, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime
import gzip

from .base_exporter import BaseExporter, ExportConfig

logger = logging.getLogger(__name__)


class CSVExporter(BaseExporter):
    """Export resampled datasets to CSV format."""
    
    def __init__(self, db_connection):
        super().__init__(db_connection)
        self.supported_compressions = {'gzip', 'gz', None}
    
    def export(self,
               dataset_info: Dict[str, Any],
               config: ExportConfig,
               progress_callback: Optional[callable] = None) -> Path:
        """
        Export resampled dataset to CSV.
        
        Args:
            dataset_info: Should contain:
                - 'experiment_id': ID of the experiment
                - 'dataset_names': List of dataset names to export
                - 'bounds': Optional spatial bounds to filter
            config: Export configuration
            progress_callback: Optional progress callback
            
        Returns:
            Path to exported CSV file
        """
        self.export_stats['start_time'] = datetime.now()
        
        try:
            # Validate configuration
            self._validate_config(config)
            
            # Ensure output directory exists
            config.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine output file path
            output_file = self._get_output_path(config)
            
            # Export data
            if config.compression == 'gzip':
                with gzip.open(output_file, 'wt', newline='', encoding='utf-8') as f:
                    self._export_to_file(f, dataset_info, config, progress_callback)
            else:
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    self._export_to_file(f, dataset_info, config, progress_callback)
            
            self.export_stats['end_time'] = datetime.now()
            
            # Create metadata file if requested
            if config.include_metadata:
                self._create_metadata_file(output_file, dataset_info, config)
            
            logger.info(f"Export completed: {output_file}")
            logger.info(f"Exported {self.export_stats['rows_exported']:,} rows")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise
    
    def export_merged_dataset(self,
                             experiment_id: str,
                             config: ExportConfig,
                             progress_callback: Optional[callable] = None) -> Path:
        """
        Export merged resampled datasets from an experiment.
        
        This is the main method for exporting pipeline results.
        """
        # Get dataset information from experiment
        dataset_info = self._get_experiment_datasets(experiment_id)
        dataset_info['experiment_id'] = experiment_id
        
        return self.export(dataset_info, config, progress_callback)
    
    def _export_to_file(self, 
                       file_handle,
                       dataset_info: Dict[str, Any],
                       config: ExportConfig,
                       progress_callback: Optional[callable] = None):
        """Export data to file handle."""
        writer = None
        total_rows = 0
        
        # Get data iterator
        for chunk_num, (headers, data_chunk) in enumerate(
            self._iterate_dataset_chunks(dataset_info, config)
        ):
            # Initialize CSV writer with headers on first chunk
            if writer is None:
                writer = csv.writer(file_handle)
                writer.writerow(headers)
            
            # Write data rows
            writer.writerows(data_chunk)
            
            # Update statistics
            chunk_size = len(data_chunk)
            total_rows += chunk_size
            self.export_stats['rows_exported'] = total_rows
            self.export_stats['chunks_processed'] = chunk_num + 1
            
            # Progress callback
            if progress_callback:
                progress_callback({
                    'rows_exported': total_rows,
                    'current_chunk': chunk_num + 1,
                    'chunk_size': chunk_size
                })
    
    def _iterate_dataset_chunks(self, 
                               dataset_info: Dict[str, Any],
                               config: ExportConfig) -> Iterator[Tuple[List[str], List[List[Any]]]]:
        """
        Iterate through dataset in chunks.
        
        Yields:
            Tuple of (headers, data_rows)
        """
        experiment_id = dataset_info.get('experiment_id')
        dataset_names = dataset_info.get('dataset_names', [])
        bounds = dataset_info.get('bounds')
        
        # Build query for merged data
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            
            # First, get the structure from one dataset to build headers
            if not dataset_names:
                # Get all datasets for experiment
                cur.execute("""
                    SELECT DISTINCT rd.name, rd.band_name
                    FROM resampled_datasets rd
                    WHERE rd.experiment_id = %s
                    ORDER BY rd.name
                """, (experiment_id,))
                dataset_names = [(row[0], row[1]) for row in cur.fetchall()]
            else:
                # Get band names for specified datasets
                placeholders = ','.join(['%s'] * len(dataset_names))
                cur.execute(f"""
                    SELECT name, band_name
                    FROM resampled_datasets
                    WHERE experiment_id = %s AND name IN ({placeholders})
                    ORDER BY name
                """, (experiment_id, *dataset_names))
                dataset_names = [(row[0], row[1]) for row in cur.fetchall()]
            
            if not dataset_names:
                logger.warning("No datasets found for export")
                return
            
            # Build headers
            headers = ['cell_id', 'x', 'y']
            headers.extend([band_name for _, band_name in dataset_names])
            
            # Get data from first dataset to establish grid
            first_dataset, first_band = dataset_names[0]
            cur.execute("""
                SELECT data_table_name, shape, bounds
                FROM resampled_datasets
                WHERE experiment_id = %s AND name = %s
            """, (experiment_id, first_dataset))
            
            result = cur.fetchone()
            if not result:
                logger.error(f"Dataset {first_dataset} not found")
                return
            
            data_table_name, shape, full_bounds = result
            
            # Apply spatial bounds filter if provided
            if bounds:
                export_bounds = self._intersect_bounds(full_bounds, bounds)
            else:
                export_bounds = full_bounds
            
            # Calculate grid indices for bounds
            indices = self._calculate_grid_indices(shape, full_bounds, export_bounds)
            
            # Export in chunks
            chunk_size = config.chunk_size
            total_cells = len(indices)
            
            for chunk_start in range(0, total_cells, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_cells)
                chunk_indices = indices[chunk_start:chunk_end]
                
                # Get coordinates for this chunk
                coords = self._get_coordinates_for_indices(
                    chunk_indices, shape, full_bounds
                )
                
                # Build data rows
                data_rows = []
                
                for idx, (x, y) in zip(chunk_indices, coords):
                    row = [f"cell_{idx}", x, y]
                    
                    # Get values from each dataset
                    for dataset_name, band_name in dataset_names:
                        value = self._get_cell_value(
                            cur, experiment_id, dataset_name, idx
                        )
                        row.append(value)
                    
                    data_rows.append(row)
                
                yield headers, data_rows
    
    def _get_cell_value(self, cursor, experiment_id: str, 
                       dataset_name: str, cell_index: int) -> Any:
        """Get value for specific cell from dataset."""
        # Get table name
        cursor.execute("""
            SELECT data_table_name
            FROM resampled_datasets
            WHERE experiment_id = %s AND name = %s
        """, (experiment_id, dataset_name))
        
        result = cursor.fetchone()
        if not result:
            return None
        
        table_name = result[0]
        
        # Get value
        cursor.execute(f"""
            SELECT value
            FROM {table_name}
            WHERE cell_index = %s
        """, (cell_index,))
        
        result = cursor.fetchone()
        return result[0] if result else None
    
    def _calculate_grid_indices(self, shape: List[int], 
                               full_bounds: List[float],
                               export_bounds: List[float]) -> List[int]:
        """Calculate grid indices within export bounds."""
        height, width = shape
        west, south, east, north = full_bounds
        exp_west, exp_south, exp_east, exp_north = export_bounds
        
        # Calculate pixel size
        pixel_width = (east - west) / width
        pixel_height = (north - south) / height
        
        # Calculate index ranges
        col_start = max(0, int((exp_west - west) / pixel_width))
        col_end = min(width, int((exp_east - west) / pixel_width) + 1)
        row_start = max(0, int((north - exp_north) / pixel_height))
        row_end = min(height, int((north - exp_south) / pixel_height) + 1)
        
        # Generate indices
        indices = []
        for row in range(row_start, row_end):
            for col in range(col_start, col_end):
                indices.append(row * width + col)
        
        return indices
    
    def _get_coordinates_for_indices(self, indices: List[int],
                                    shape: List[int],
                                    bounds: List[float]) -> List[Tuple[float, float]]:
        """Convert grid indices to coordinates."""
        height, width = shape
        west, south, east, north = bounds
        
        pixel_width = (east - west) / width
        pixel_height = (north - south) / height
        
        coords = []
        for idx in indices:
            row = idx // width
            col = idx % width
            
            # Center coordinates of pixel
            x = west + (col + 0.5) * pixel_width
            y = north - (row + 0.5) * pixel_height
            
            coords.append((x, y))
        
        return coords
    
    def _intersect_bounds(self, bounds1: List[float], 
                         bounds2: List[float]) -> List[float]:
        """Calculate intersection of two bounds."""
        west = max(bounds1[0], bounds2[0])
        south = max(bounds1[1], bounds2[1])
        east = min(bounds1[2], bounds2[2])
        north = min(bounds1[3], bounds2[3])
        
        if west >= east or south >= north:
            raise ValueError("Bounds do not intersect")
        
        return [west, south, east, north]
    
    def _get_experiment_datasets(self, experiment_id: str) -> Dict[str, Any]:
        """Get dataset information for an experiment."""
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            
            # Get all resampled datasets for this experiment
            cur.execute("""
                SELECT name, band_name, shape, bounds
                FROM resampled_datasets
                WHERE experiment_id = %s
                ORDER BY name
            """, (experiment_id,))
            
            datasets = []
            common_bounds = None
            
            for row in cur.fetchall():
                name, band_name, shape, bounds = row
                datasets.append(name)
                
                # Calculate common bounds (intersection)
                if common_bounds is None:
                    common_bounds = bounds
                else:
                    common_bounds = self._intersect_bounds(common_bounds, bounds)
            
            return {
                'dataset_names': datasets,
                'bounds': common_bounds
            }
    
    def _validate_config(self, config: ExportConfig):
        """Validate export configuration."""
        if config.compression and config.compression not in self.supported_compressions:
            raise ValueError(
                f"Unsupported compression: {config.compression}. "
                f"Supported: {self.supported_compressions}"
            )
    
    def _get_output_path(self, config: ExportConfig) -> Path:
        """Determine output file path with appropriate extension."""
        output_path = config.output_path
        
        # Add .csv extension if not present
        if output_path.suffix != '.csv':
            output_path = output_path.with_suffix('.csv')
        
        # Add compression extension
        if config.compression == 'gzip':
            output_path = output_path.with_suffix('.csv.gz')
        
        return output_path
    
    def _create_metadata_file(self, output_file: Path,
                             dataset_info: Dict[str, Any],
                             config: ExportConfig):
        """Create metadata file for the export."""
        metadata = {
            'export_timestamp': datetime.now().isoformat(),
            'output_file': str(output_file),
            'export_stats': self.get_export_stats(),
            'dataset_info': dataset_info,
            'configuration': {
                'chunk_size': config.chunk_size,
                'compression': config.compression,
                'include_metadata': config.include_metadata
            }
        }
        
        metadata_file = output_file.with_suffix('.meta.json')
        
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Metadata saved to: {metadata_file}")
    
    def validate_export(self, output_path: Path) -> bool:
        """Validate the exported CSV file."""
        try:
            # Check file exists
            if not output_path.exists():
                logger.error(f"Export file not found: {output_path}")
                return False
            
            # Check file size
            file_size = output_path.stat().st_size
            if file_size == 0:
                logger.error("Export file is empty")
                return False
            
            # Try reading first few lines
            if output_path.suffix == '.gz':
                opener = gzip.open
                mode = 'rt'
            else:
                opener = open
                mode = 'r'
            
            with opener(output_path, mode) as f:
                reader = csv.reader(f)
                headers = next(reader)
                
                # Check headers
                if len(headers) < 3:  # At minimum: cell_id, x, y
                    logger.error("Invalid headers in CSV")
                    return False
                
                # Check a few data rows
                for i, row in enumerate(reader):
                    if i >= 5:  # Check first 5 rows
                        break
                    
                    if len(row) != len(headers):
                        logger.error(f"Row {i+1} has incorrect number of columns")
                        return False
            
            logger.info(f"Export validation passed: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Export validation failed: {e}")
            return False


# Convenience function for simple export
def export_experiment_to_csv(experiment_id: str, 
                            output_path: Path,
                            db_connection,
                            **kwargs) -> Path:
    """
    Convenience function to export experiment results to CSV.
    
    Args:
        experiment_id: ID of the experiment to export
        output_path: Path for output CSV file
        db_connection: Database connection
        **kwargs: Additional options for ExportConfig
        
    Returns:
        Path to exported file
    """
    exporter = CSVExporter(db_connection)
    config = ExportConfig(output_path, **kwargs)
    
    return exporter.export_merged_dataset(experiment_id, config)