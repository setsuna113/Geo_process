"""Parquet Converter for GEE Climate Data

Converts CSV data from GEE exports to parquet format matching
the exact schema used by export_stage.py for pipeline compatibility.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path
import json
import time

try:
    from src.infrastructure.logging import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)


class ParquetConverter:
    """Converts GEE CSV exports to pipeline-compatible parquet format."""
    
    # Expected schema matching export_stage.py output
    TARGET_SCHEMA = {
        'x': 'float64',  # longitude
        'y': 'float64',  # latitude
        'bio01': 'float32',  # Annual Mean Temperature
        'bio04': 'float32',  # Temperature Seasonality  
        'bio12': 'float32'   # Annual Precipitation
    }
    
    def __init__(self, logger=None):
        """
        Initialize parquet converter.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger if logger else get_logger(__name__)
        
    def convert_to_parquet(self,
                          climate_data: pd.DataFrame,
                          output_path: Union[str, Path],
                          experiment_id: Optional[str] = None,
                          validate_schema: bool = True,
                          include_metadata: bool = True) -> Dict[str, any]:
        """
        Convert climate DataFrame to parquet format.
        
        Args:
            climate_data: DataFrame with coordinates and climate variables
            output_path: Output parquet file path
            experiment_id: Optional experiment ID for filename
            validate_schema: Whether to validate schema before export
            include_metadata: Whether to create metadata file
            
        Returns:
            Dict with export statistics and metadata
        """
        if climate_data.empty:
            raise ValueError("Cannot convert empty DataFrame to parquet")
        
        output_path = Path(output_path)
        
        # Generate filename with timestamp like export_stage.py
        if experiment_id:
            timestamp = int(time.time())
            output_filename = f"climate_data_{experiment_id}_{timestamp}.parquet"
            output_path = output_path.parent / output_filename
        
        self.logger.info(f"Converting {len(climate_data)} rows to parquet: {output_path}")
        
        # Prepare DataFrame with correct schema
        df = self._prepare_dataframe(climate_data, validate_schema)
        
        # Track conversion time
        start_time = time.time()
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to parquet with snappy compression (matching export_stage.py)
        df.to_parquet(output_path, index=False, compression='snappy')
        
        # Validate export
        if not output_path.exists():
            raise RuntimeError("Parquet export failed - output file not created")
        
        file_size = output_path.stat().st_size
        if file_size == 0:
            raise RuntimeError("Parquet export failed - output file is empty")
        
        conversion_duration = time.time() - start_time
        
        # Create export statistics
        export_stats = {
            'output_path': str(output_path),
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'file_size_mb': file_size / (1024**2),
            'conversion_duration': conversion_duration,
            'compression': 'snappy',
            'schema': dict(df.dtypes.astype(str))
        }
        
        self.logger.info(f"Parquet conversion complete:")
        self.logger.info(f"  Rows: {export_stats['total_rows']:,}")
        self.logger.info(f"  Size: {export_stats['file_size_mb']:.2f} MB")
        self.logger.info(f"  Duration: {export_stats['conversion_duration']:.2f}s")
        
        # Create metadata file if requested
        if include_metadata:
            metadata_path = output_path.with_suffix('.json')
            self._create_metadata_file(metadata_path, export_stats, df)
            export_stats['metadata_path'] = str(metadata_path)
        
        return export_stats
    
    def _prepare_dataframe(self, df: pd.DataFrame, validate_schema: bool) -> pd.DataFrame:
        """Prepare DataFrame with correct schema and data types."""
        
        # Start with a copy
        result_df = df.copy()
        
        # Ensure required coordinate columns exist
        if 'x' not in result_df.columns or 'y' not in result_df.columns:
            raise ValueError("DataFrame must contain 'x' and 'y' coordinate columns")
        
        # Apply target schema data types
        for column, dtype in self.TARGET_SCHEMA.items():
            if column in result_df.columns:
                try:
                    result_df[column] = result_df[column].astype(dtype)
                except Exception as e:
                    self.logger.warning(f"Failed to convert {column} to {dtype}: {e}")
            else:
                if column in ['bio01', 'bio04', 'bio12']:
                    if self.logger:
                        self.logger.warning(f"Missing climate variable: {column}")
                    # Add column with NaN values
                    result_df[column] = np.nan
                    result_df[column] = result_df[column].astype(self.TARGET_SCHEMA[column])
        
        # Remove any columns not in target schema (optional - keep extra columns)
        # This preserves any additional data that might be useful
        
        # Validate schema if requested
        if validate_schema:
            self._validate_schema(result_df)
        
        # Reset index to match export_stage.py behavior
        result_df = result_df.reset_index(drop=True)
        
        return result_df
    
    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Validate DataFrame schema matches expected format."""
        
        # Check required columns
        required_columns = ['x', 'y']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check data types for known columns
        for column in df.columns:
            if column in self.TARGET_SCHEMA:
                expected_dtype = self.TARGET_SCHEMA[column]
                actual_dtype = str(df[column].dtype)
                
                # Allow some flexibility in float precision
                if expected_dtype.startswith('float') and actual_dtype.startswith('float'):
                    continue
                if expected_dtype.startswith('int') and actual_dtype.startswith('int'):
                    continue
                    
                if actual_dtype != expected_dtype:
                    if self.logger:
                        self.logger.warning(f"Schema mismatch for {column}: expected {expected_dtype}, got {actual_dtype}")
        
        # Check for valid coordinate ranges
        if df['x'].min() < -180 or df['x'].max() > 180:
            raise ValueError("Longitude (x) values outside valid range [-180, 180]")
        
        if df['y'].min() < -90 or df['y'].max() > 90:
            raise ValueError("Latitude (y) values outside valid range [-90, 90]")
        
        # Check for missing coordinates
        coord_nulls = df[['x', 'y']].isnull().any(axis=1).sum()
        if coord_nulls > 0:
            if self.logger:
                self.logger.warning(f"{coord_nulls} rows have missing coordinates")
        
        if self.logger:
            self.logger.debug("Schema validation passed")
    
    def _create_metadata_file(self, metadata_path: Path, export_stats: Dict, df: pd.DataFrame) -> None:
        """Create metadata file with export information."""
        
        # Calculate additional statistics
        metadata = {
            'export_info': export_stats.copy(),
            'data_summary': {
                'coordinate_bounds': {
                    'x_min': float(df['x'].min()),
                    'x_max': float(df['x'].max()),
                    'y_min': float(df['y'].min()),
                    'y_max': float(df['y'].max())
                },
                'climate_variables': {}
            },
            'pipeline_compatibility': {
                'compatible_with_export_stage': True,
                'schema_version': '1.0',
                'compression': 'snappy'
            }
        }
        
        # Add climate variable statistics
        for var in ['bio01', 'bio04', 'bio12']:
            if var in df.columns:
                var_data = df[var].dropna()
                if len(var_data) > 0:
                    metadata['data_summary']['climate_variables'][var] = {
                        'count': len(var_data),
                        'min': float(var_data.min()),
                        'max': float(var_data.max()),
                        'mean': float(var_data.mean()),
                        'null_count': df[var].isnull().sum()
                    }
        
        # Write metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if self.logger:
            self.logger.debug(f"Metadata written to {metadata_path}")
    
    def convert_csv_to_parquet(self,
                              csv_path: Union[str, Path],
                              parquet_path: Union[str, Path],
                              **kwargs) -> Dict[str, any]:
        """
        Convert CSV file to parquet format.
        
        Args:
            csv_path: Input CSV file path
            parquet_path: Output parquet file path
            **kwargs: Additional arguments for convert_to_parquet
            
        Returns:
            Dict with export statistics
        """
        csv_path = Path(csv_path)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        if self.logger:
            self.logger.info(f"Reading CSV file: {csv_path}")
        
        # Read CSV file
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV file {csv_path}: {e}")
        
        return self.convert_to_parquet(df, parquet_path, **kwargs)
    
    def validate_parquet_compatibility(self, parquet_path: Union[str, Path]) -> Dict[str, any]:
        """
        Validate that parquet file is compatible with pipeline.
        
        Args:
            parquet_path: Path to parquet file
            
        Returns:
            Dict with validation results
        """
        parquet_path = Path(parquet_path)
        
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        
        try:
            # Read parquet file
            df = pd.read_parquet(parquet_path)
            
            # Validate schema
            validation_results = {
                'file_path': str(parquet_path),
                'rows': len(df),
                'columns': len(df.columns),
                'schema_valid': True,
                'coordinate_valid': True,
                'issues': []
            }
            
            # Check schema
            try:
                self._validate_schema(df)
            except Exception as e:
                validation_results['schema_valid'] = False
                validation_results['issues'].append(f"Schema validation failed: {e}")
            
            # Check coordinates
            if 'x' in df.columns and 'y' in df.columns:
                coord_bounds = {
                    'x_min': float(df['x'].min()),
                    'x_max': float(df['x'].max()),
                    'y_min': float(df['y'].min()),
                    'y_max': float(df['y'].max())
                }
                validation_results['coordinate_bounds'] = coord_bounds
                
                # Check for reasonable coordinate ranges
                if coord_bounds['x_min'] < -180 or coord_bounds['x_max'] > 180:
                    validation_results['coordinate_valid'] = False
                    validation_results['issues'].append("Invalid longitude range")
                
                if coord_bounds['y_min'] < -90 or coord_bounds['y_max'] > 90:
                    validation_results['coordinate_valid'] = False
                    validation_results['issues'].append("Invalid latitude range")
            else:
                validation_results['coordinate_valid'] = False
                validation_results['issues'].append("Missing coordinate columns")
            
            validation_results['valid'] = (validation_results['schema_valid'] and 
                                         validation_results['coordinate_valid'])
            
            if self.logger:
                if validation_results['valid']:
                    self.logger.info(f"Parquet validation passed: {parquet_path}")
                else:
                    self.logger.warning(f"Parquet validation issues: {validation_results['issues']}")
            
            return validation_results
            
        except Exception as e:
            return {
                'file_path': str(parquet_path),
                'valid': False,
                'issues': [f"Failed to read parquet file: {e}"]
            }