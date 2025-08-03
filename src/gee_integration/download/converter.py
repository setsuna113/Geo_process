"""Parquet Converter Module

Converts GEE CSV exports to parquet format.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import glob

from src.infrastructure.logging import get_logger
from src.infrastructure.logging.decorators import log_operation

logger = get_logger(__name__)


class ParquetConverter:
    """Converts CSV exports from GEE to parquet format"""
    
    # Expected columns from GEE export
    COORD_COLUMNS = ['x', 'y']
    
    def __init__(self, output_dir: Path):
        """
        Initialize converter.
        
        Args:
            output_dir: Directory for parquet outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
    
    @log_operation("convert_csv_to_parquet")
    def convert_csv_files(self, csv_files: List[Path], 
                         resolution_name: str,
                         chunk_size: int = 1000000) -> Path:
        """
        Convert multiple CSV files to a single parquet file.
        
        Args:
            csv_files: List of CSV file paths
            resolution_name: Resolution identifier
            chunk_size: Rows to process at once
            
        Returns:
            Path to output parquet file
        """
        if not csv_files:
            raise ValueError("No CSV files provided")
        
        output_path = self.output_dir / f"biodiversity_climate_{resolution_name}.parquet"
        
        self.logger.info(f"Converting {len(csv_files)} CSV files to parquet...")
        
        # Process files
        all_dfs = []
        total_rows = 0
        
        for csv_file in sorted(csv_files):
            self.logger.debug(f"Processing {csv_file.name}...")
            
            try:
                # Read CSV in chunks for memory efficiency
                for chunk_df in pd.read_csv(csv_file, chunksize=chunk_size):
                    # Clean and validate chunk
                    chunk_df = self._clean_dataframe(chunk_df)
                    
                    if len(chunk_df) > 0:
                        all_dfs.append(chunk_df)
                        total_rows += len(chunk_df)
                    
                    # Write incrementally if getting large
                    if sum(len(df) for df in all_dfs) > chunk_size * 2:
                        self._write_parquet_chunk(all_dfs, output_path, append=output_path.exists())
                        all_dfs = []
                        
            except Exception as e:
                self.logger.error(f"Failed to process {csv_file}: {e}")
                raise
        
        # Write remaining data
        if all_dfs:
            self._write_parquet_chunk(all_dfs, output_path, append=output_path.exists())
        
        # Validate output
        if not output_path.exists():
            raise RuntimeError("Failed to create parquet file")
        
        # Get final statistics
        final_df = pd.read_parquet(output_path)
        stats = self._compute_statistics(final_df)
        
        self.logger.info(f"✅ Converted to parquet: {output_path.name}")
        self.logger.info(f"   Rows: {len(final_df):,}")
        self.logger.info(f"   Columns: {len(final_df.columns)}")
        self.logger.info(f"   Size: {output_path.stat().st_size/(1024*1024):.1f} MB")
        
        # Save metadata
        self._save_metadata(output_path, stats, resolution_name)
        
        return output_path
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate dataframe"""
        # Remove any system columns from GEE
        system_cols = [col for col in df.columns if col.startswith('system:') or col == '.geo']
        if system_cols:
            df = df.drop(columns=system_cols)
        
        # Ensure coordinate columns exist
        if not all(col in df.columns for col in self.COORD_COLUMNS):
            missing = [col for col in self.COORD_COLUMNS if col not in df.columns]
            raise ValueError(f"Missing coordinate columns: {missing}")
        
        # Convert data types
        for col in self.COORD_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with invalid coordinates
        df = df.dropna(subset=self.COORD_COLUMNS)
        
        # Sort by coordinates for better compression
        df = df.sort_values(by=self.COORD_COLUMNS)
        
        return df
    
    def _write_parquet_chunk(self, dfs: List[pd.DataFrame], 
                           output_path: Path, 
                           append: bool = False) -> None:
        """Write dataframes to parquet"""
        # Combine dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        if append and output_path.exists():
            # Read existing data
            existing_df = pd.read_parquet(output_path)
            combined_df = pd.concat([existing_df, combined_df], ignore_index=True)
        
        # Write with compression
        combined_df.to_parquet(
            output_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
    
    def _compute_statistics(self, df: pd.DataFrame) -> Dict:
        """Compute statistics for the dataset"""
        stats = {
            'total_rows': len(df),
            'columns': list(df.columns),
            'coordinate_bounds': {
                'min_x': float(df['x'].min()),
                'max_x': float(df['x'].max()),
                'min_y': float(df['y'].min()),
                'max_y': float(df['y'].max())
            },
            'data_columns': {}
        }
        
        # Stats for each data column
        data_cols = [col for col in df.columns if col not in self.COORD_COLUMNS]
        
        for col in data_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                stats['data_columns'][col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'non_null_count': int(df[col].notna().sum()),
                    'null_count': int(df[col].isna().sum())
                }
            else:
                stats['data_columns'][col] = {
                    'unique_values': int(df[col].nunique()),
                    'non_null_count': int(df[col].notna().sum())
                }
        
        return stats
    
    def _save_metadata(self, parquet_path: Path, stats: Dict, resolution: str) -> None:
        """Save metadata file"""
        import json
        from datetime import datetime
        
        metadata = {
            'file': parquet_path.name,
            'resolution': resolution,
            'created': datetime.now().isoformat(),
            'statistics': stats,
            'source': 'GEE biodiversity-climate merge',
            'coordinate_system': 'EPSG:4326'
        }
        
        metadata_path = parquet_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.debug(f"Saved metadata to {metadata_path}")
    
    def convert_all_resolutions(self, downloads: Dict[str, List[Path]]) -> Dict[str, Path]:
        """
        Convert downloads for all resolutions.
        
        Args:
            downloads: Dict mapping resolution name to CSV files
            
        Returns:
            Dict mapping resolution name to parquet path
        """
        outputs = {}
        
        for res_name, csv_files in downloads.items():
            if not csv_files:
                self.logger.warning(f"No files to convert for {res_name}")
                continue
            
            self.logger.info(f"Converting {res_name} data...")
            
            try:
                output_path = self.convert_csv_files(csv_files, res_name)
                outputs[res_name] = output_path
            except Exception as e:
                self.logger.error(f"Failed to convert {res_name}: {e}")
                raise
        
        return outputs