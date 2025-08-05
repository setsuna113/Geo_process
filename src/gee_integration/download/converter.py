"""Parquet Converter Module

Converts GEE exports (CSV or GeoTIFF) to parquet format.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import glob
import numpy as np
import pandas as pd

from src.infrastructure.logging import get_logger
from src.infrastructure.logging.decorators import log_operation

logger = get_logger(__name__)


class ParquetConverter:
    """Converts GEE exports to parquet format"""
    
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
        self.band_names = None  # Will be set from config or metadata
    
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
            # For testing with GeoTIFF exports, create dummy parquet
            self.logger.warning(f"No CSV files for {resolution_name}, creating placeholder parquet")
            output_path = self.output_dir / f"biodiversity_climate_{resolution_name}.parquet"
            # Create a minimal DataFrame for testing
            test_df = pd.DataFrame({
                'x': [0.0],
                'y': [0.0],
                'test_data': [1.0],
                'note': ['GeoTIFF export test - placeholder data']
            })
            test_df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
            return output_path
        
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
    
    def _clean_dataframe(self, df) -> 'pd.DataFrame':
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
    
    def _write_parquet_chunk(self, dfs: List['pd.DataFrame'], 
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
    
    def _compute_statistics(self, df) -> Dict:
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
    
    def convert(self, source_files: Union[Path, List[Path]], 
                resolution_name: str,
                metadata: Optional[Dict] = None) -> Path:
        """
        Convert source files to parquet based on format.
        
        Args:
            source_files: Path to GeoTIFF or list of CSV paths
            resolution_name: Resolution identifier
            metadata: Optional metadata including band names
            
        Returns:
            Path to output parquet file
        """
        # Handle both single file and list
        if isinstance(source_files, Path):
            source_files = [source_files]
        elif not isinstance(source_files, list):
            source_files = list(source_files)
        
        if not source_files:
            raise ValueError("No source files provided")
        
        # Check file format
        first_file = source_files[0]
        
        if first_file.suffix.lower() in ['.tif', '.tiff']:
            # Check if we have hemisphere files that need merging
            if len(source_files) == 2:
                # Check if these are hemisphere files
                file_names = [f.stem for f in source_files]
                if any('west' in name for name in file_names) and any('east' in name for name in file_names):
                    self.logger.info("Detected hemisphere files, merging before conversion...")
                    
                    # Find west and east files
                    west_file = next(f for f in source_files if 'west' in f.stem)
                    east_file = next(f for f in source_files if 'east' in f.stem)
                    
                    # Merge hemispheres
                    from src.gee_integration.processing.hemisphere_merger import merge_hemispheres
                    merged_path = self.output_dir / f"merged_{resolution_name}.tif"
                    merge_hemispheres(west_file, east_file, merged_path)
                    
                    # Convert the merged file
                    source_files = [merged_path]
            
            # GeoTIFF conversion
            if len(source_files) > 1:
                self.logger.warning("Multiple GeoTIFF files provided, using first one")
            
            # Set band names from metadata if available
            if metadata and 'band_names' in metadata:
                self.band_names = metadata['band_names']
            
            return self._convert_geotiff_to_parquet(
                source_files[0], 
                resolution_name,
                chunk_size=metadata.get('chunk_size', 100000) if metadata else 100000,
                metadata=metadata
            )
        else:
            # CSV conversion (existing method)
            return self.convert_csv_files(source_files, resolution_name)
    
    @log_operation("convert_geotiff_to_parquet")
    def _convert_geotiff_to_parquet(self, tif_path: Path, 
                                   resolution_name: str,
                                   chunk_size: int = 100000,
                                   metadata: Optional[Dict] = None) -> Path:
        """
        Convert GeoTIFF to parquet with streaming.
        
        Args:
            tif_path: Path to GeoTIFF file
            resolution_name: Resolution identifier
            chunk_size: Number of pixels to process at once
            
        Returns:
            Path to output parquet file
        """
        import rasterio
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        output_path = self.output_dir / f"biodiversity_climate_{resolution_name}.parquet"
        
        self.logger.info(f"Converting GeoTIFF to parquet: {tif_path.name}")
        
        with rasterio.open(tif_path) as src:
            # Get raster info
            width = src.width
            height = src.height
            n_bands = src.count
            transform = src.transform
            nodata = src.nodata or -9999
            
            self.logger.info(f"   Dimensions: {width}x{height}")
            self.logger.info(f"   Bands: {n_bands}")
            self.logger.info(f"   NoData value: {nodata}")
            
            # Prepare band names
            if not self.band_names:
                # Get default band names from metadata if available
                if metadata and 'band_names' in metadata:
                    self.band_names = metadata['band_names'][:n_bands]
                else:
                    # Fallback defaults if no metadata provided
                    self.logger.warning("No band names in metadata, using minimal defaults")
                    self.band_names = [f'band_{i+1}' for i in range(n_bands)]
            
            # Define schema
            schema_fields = [
                ('x', pa.float64()),
                ('y', pa.float64())
            ]
            for band_name in self.band_names:
                schema_fields.append((band_name, pa.float32()))
            schema = pa.schema(schema_fields)
            
            # Process in windows for memory efficiency
            total_pixels = 0
            valid_pixels = 0
            
            with pq.ParquetWriter(output_path, schema, compression='snappy') as writer:
                
                # Process each window
                for ji, window in src.block_windows(1):
                    # Read all bands for this window
                    data = src.read(window=window)  # Shape: (bands, height, width)
                    
                    # Get coordinates for this window
                    # Memory optimization: process in chunks defined by rasterio blocks
                    rows, cols = np.meshgrid(
                        np.arange(window.row_off, window.row_off + window.height),
                        np.arange(window.col_off, window.col_off + window.width),
                        indexing='ij'
                    )
                    
                    # Convert pixel indices to coordinates
                    xs, ys = rasterio.transform.xy(transform, rows.flatten(), cols.flatten())
                    xs = np.array(xs)
                    ys = np.array(ys)
                    
                    # Build record batch
                    batch_data = {
                        'x': xs,
                        'y': ys
                    }
                    
                    # Add each band
                    for i, band_name in enumerate(self.band_names):
                        if i < n_bands:
                            batch_data[band_name] = data[i].flatten().astype(np.float32)
                    
                    # Create mask for valid pixels
                    # Include ALL pixels that have valid data in ANY band (not just biodiversity)
                    # This ensures we keep climate and elevation data even where biodiversity is missing
                    valid_mask = np.zeros(len(xs), dtype=bool)
                    
                    # Check ALL bands for valid data
                    for band_name, band_data in batch_data.items():
                        if isinstance(band_data, np.ndarray) and np.issubdtype(band_data.dtype, np.number):
                            # Include pixels that have non-NoData values in this band
                            valid_mask |= (band_data != nodata) & ~np.isnan(band_data)
                    
                    # Apply mask
                    if np.any(valid_mask):
                        filtered_data = {k: v[valid_mask] for k, v in batch_data.items()}
                        
                        # Convert zeros to NaN for biodiversity bands
                        # (0 represents nodata from unmask operation)
                        # Identify biodiversity bands (exclude climate/elevation bands)
                        climate_bands = ['bio01', 'bio02', 'bio03', 'bio04', 'bio05', 'bio06', 
                                       'bio07', 'bio08', 'bio09', 'bio10', 'bio11', 'bio12',
                                       'bio13', 'bio14', 'bio15', 'bio16', 'bio17', 'bio18', 'bio19']
                        known_non_biodiv = ['x', 'y', 'elevation', 'dem'] + climate_bands
                        
                        for band_name, band_data in filtered_data.items():
                            # If it's not a known non-biodiversity band, it's biodiversity
                            if band_name not in known_non_biodiv and isinstance(band_data, np.ndarray):
                                # Convert 0 to NaN for biodiversity bands
                                filtered_data[band_name] = np.where(band_data == 0, np.nan, band_data)
                        
                        # Ensure all schema fields are present (fill missing with NaN)
                        for field in schema.names:
                            if field not in filtered_data:
                                # Fill missing bands with NaN
                                filtered_data[field] = np.full(np.sum(valid_mask), np.nan, dtype=np.float32)
                        
                        # Write batch
                        batch = pa.RecordBatch.from_pydict(filtered_data, schema=schema)
                        writer.write_batch(batch)
                        
                        valid_pixels += np.sum(valid_mask)
                    
                    total_pixels += len(xs)
                    
                    # Progress update
                    if ji[0] % 10 == 0:
                        self.logger.debug(f"   Processed window {ji}: {valid_pixels:,}/{total_pixels:,} valid pixels")
        
        # Validate output
        if not output_path.exists():
            raise RuntimeError("Failed to create parquet file")
        
        # Get final statistics
        final_df = pd.read_parquet(output_path)
        stats = self._compute_statistics(final_df)
        
        self.logger.info(f"✅ Converted GeoTIFF to parquet: {output_path.name}")
        self.logger.info(f"   Total pixels: {total_pixels:,}")
        self.logger.info(f"   Valid pixels: {valid_pixels:,} ({valid_pixels/total_pixels*100:.1f}%)")
        self.logger.info(f"   Output rows: {len(final_df):,}")
        self.logger.info(f"   Size: {output_path.stat().st_size/(1024*1024):.1f} MB")
        
        # Save metadata
        self._save_metadata(output_path, stats, resolution_name)
        
        return output_path