"""Hemisphere merger for GEE exports that split at the antimeridian."""

from pathlib import Path
import rasterio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Optional, Dict
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


def merge_hemispheres_to_parquet(west_file: Path, east_file: Path, 
                                 output_path: Path,
                                 metadata: Optional[Dict] = None) -> Path:
    """
    Merge hemisphere files directly to parquet using streaming.
    
    Args:
        west_file: Path to western hemisphere GeoTIFF
        east_file: Path to eastern hemisphere GeoTIFF
        output_path: Output parquet path
        metadata: Optional metadata with band names
        
    Returns:
        Path to output parquet file
    """
    logger.info(f"Merging hemispheres to parquet: {west_file.name} + {east_file.name}")
    
    # Process both files in sequence
    files = [west_file, east_file]
    writer = None
    total_pixels = 0
    schema = None
    band_names = None
    
    try:
        for file_idx, file_path in enumerate(files):
            hemisphere = "western" if file_idx == 0 else "eastern"
            logger.info(f"Processing {hemisphere} hemisphere: {file_path.name}")
            
            with rasterio.open(file_path) as src:
                width = src.width
                height = src.height
                n_bands = src.count
                transform = src.transform
                nodata = src.nodata or -9999
                
                # On first file, setup schema
                if writer is None:
                    # Get band names from metadata or use defaults
                    if metadata and 'band_names' in metadata:
                        band_names = metadata['band_names'][:n_bands]
                    else:
                        band_names = [f'band_{i+1}' for i in range(n_bands)]
                    
                    # Define schema
                    schema_fields = [
                        ('x', pa.float64()),
                        ('y', pa.float64())
                    ]
                    for band_name in band_names:
                        schema_fields.append((band_name, pa.float32()))
                    schema = pa.schema(schema_fields)
                    
                    # Create writer
                    writer = pq.ParquetWriter(
                        output_path,
                        schema,
                        compression='snappy',
                        use_dictionary=True,
                        write_statistics=True
                    )
                    
                    logger.info(f"   Initialized schema with {len(band_names)} bands")
                
                # Process in windows for memory efficiency
                hemisphere_pixels = 0
                
                for ji, window in src.block_windows(1):
                    # Read all bands for this window
                    data = src.read(window=window)
                    
                    # Get coordinates
                    rows, cols = np.meshgrid(
                        np.arange(window.row_off, window.row_off + window.height),
                        np.arange(window.col_off, window.col_off + window.width),
                        indexing='ij'
                    )
                    
                    # Convert to geographic coordinates
                    xs, ys = rasterio.transform.xy(transform, rows.flatten(), cols.flatten())
                    xs = np.array(xs)
                    ys = np.array(ys)
                    
                    # Build batch data
                    batch_data = {
                        'x': xs,
                        'y': ys
                    }
                    
                    # Add band data
                    for i, band_name in enumerate(band_names):
                        if i < n_bands:
                            batch_data[band_name] = data[i].flatten().astype(np.float32)
                    
                    # Create valid pixel mask
                    valid_mask = np.zeros(len(xs), dtype=bool)
                    for band_name, band_data in batch_data.items():
                        if band_name not in ['x', 'y']:
                            valid_mask |= (band_data != nodata) & ~np.isnan(band_data)
                    
                    # Write valid pixels
                    if np.any(valid_mask):
                        filtered_data = {k: v[valid_mask] for k, v in batch_data.items()}
                        
                        # Convert zeros to NaN for biodiversity bands
                        climate_bands = ['bio01', 'bio02', 'bio03', 'bio04', 'bio05', 'bio06',
                                       'bio07', 'bio08', 'bio09', 'bio10', 'bio11', 'bio12',
                                       'bio13', 'bio14', 'bio15', 'bio16', 'bio17', 'bio18', 'bio19']
                        known_non_biodiv = ['x', 'y', 'elevation', 'dem'] + climate_bands
                        
                        for band_name, band_data in filtered_data.items():
                            if band_name not in known_non_biodiv and isinstance(band_data, np.ndarray):
                                filtered_data[band_name] = np.where(band_data == 0, np.nan, band_data)
                        
                        # Ensure all schema fields present
                        for field in schema.names:
                            if field not in filtered_data:
                                filtered_data[field] = np.full(np.sum(valid_mask), np.nan, dtype=np.float32)
                        
                        # Write batch
                        batch = pa.RecordBatch.from_pydict(filtered_data, schema=schema)
                        writer.write_batch(batch)
                        
                        hemisphere_pixels += len(filtered_data['x'])
                
                logger.info(f"   {hemisphere} hemisphere: wrote {hemisphere_pixels:,} pixels")
                total_pixels += hemisphere_pixels
        
        # Close writer
        if writer:
            writer.close()
            logger.info(f"✅ Successfully merged hemispheres: {total_pixels:,} total pixels")
            
            # Verify output
            if output_path.exists():
                file_size = output_path.stat().st_size / (1024**3)
                logger.info(f"   Output file size: {file_size:.2f} GB")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to merge hemispheres: {e}")
        if writer:
            try:
                writer.close()
            except:
                pass
        raise


def merge_hemispheres(west_file: Path, east_file: Path, output_file: Path) -> None:
    """
    Legacy method - merge hemisphere GeoTIFFs into single GeoTIFF.
    Consider using merge_hemispheres_to_parquet for direct parquet output.
    
    Args:
        west_file: Western hemisphere file
        east_file: Eastern hemisphere file  
        output_file: Output merged GeoTIFF
    """
    from rasterio.merge import merge
    
    logger.info(f"Merging hemisphere GeoTIFFs: {west_file.name} + {east_file.name}")
    
    with rasterio.open(west_file) as west_src, rasterio.open(east_file) as east_src:
        # Merge the two files
        mosaic, out_trans = merge([west_src, east_src])
        
        # Update metadata
        out_meta = west_src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "compress": "lzw"
        })
        
        # Write merged file
        with rasterio.open(output_file, "w", **out_meta) as dest:
            dest.write(mosaic)
    
    logger.info(f"Merged hemispheres saved to: {output_file}")