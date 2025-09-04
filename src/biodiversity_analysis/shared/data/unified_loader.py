"""
Unified data loader with Ray zero-copy caching.

Linus says: "Bad programmers worry about the code. Good programmers worry about data structures."
This loader uses Ray for TRUE zero-copy between processes.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
from rasterio.merge import merge
from src.raster_merge.pipeline import merge_rasters
from src.raster_merge.config import RasterConfig
logger = logging.getLogger(__name__)

# Simple cache - no Ray bullshit
_CACHE = {}


def get_loader():
    """Get a loader instance. Returns self because it's just functions."""
    return UnifiedLoader()


class UnifiedLoader:
    """Not really a class, just a namespace for functions.
    
    Why a class? Because orchestrator expects loader.load_datasets().
    Why not make it functional? Because someone already wrote the orchestrator.
    """
    
    def __init__(self):
        """Nothing to initialize. This is basically a module."""
        pass
    
    def load_datasets(
        self,
        datasets: List[str],
        data_type: str,
        config: Dict[str, Any]
    ) -> Any:  # Returns DataFrame for tabular, Dict[str, np.ndarray] for spatial
        """Load the requested datasets. Simple switch based on data_type.
        
        Args:
            datasets: List of dataset names needed
            data_type: 'spatial' or 'tabular'
            config: Configuration with paths
            
        Returns:
            Dict of numpy arrays
        """
        if data_type == 'tabular':
            return self._load_tabular(datasets, config)
        elif data_type == 'spatial':
            return self._load_spatial(datasets, config)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
    
    def _load_tabular(self, datasets: List[str], config: Dict[str, Any]) -> Any:
        """Load tabular data and return Arrow Table directly.
        
        LINUS: "Keep it as Arrow Table. Let methods convert as needed."
        """
        # Get data path - flat config structure ONLY
        arrow_path = config.get('arrow_file')
        parquet_path = config.get('climate_parquet')
        
        data_path = arrow_path or parquet_path
        if not data_path:
            raise ValueError(
                "Tabular methods require 'arrow_file' or 'climate_parquet' in config!\n"
                "Got config keys: {}\n"
                .format(list(config.keys()))
            )
        
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data file does not exist: {data_path}")
        
        # Get sampling config - apply it ONCE here, nowhere else!
        spatial_step = config.get('spatial_grid_step', 1)
        
        # Check cache
        cache_key = f"tabular:{data_path}:step={spatial_step}"
        if cache_key in _CACHE:
            logger.info(f"Using cached data for {cache_key}")
            return _CACHE[cache_key]
        
        logger.info(f"Loading {data_path} with spatial_grid_step={spatial_step}")
        
        # Determine file type and load accordingly
        if arrow_path and Path(arrow_path).exists():
            # Load Arrow IPC file
            with pa.memory_map(arrow_path, 'r') as source:
                reader = pa.ipc.open_file(source)
                table = reader.read_all()
        elif parquet_path and Path(parquet_path).exists():
            # Load Parquet file
            table = pq.read_table(parquet_path)
        else:
            # This should never happen due to earlier checks
            raise ValueError(f"No valid data file found!")
        
        # Apply sampling if requested - ONCE AND ONLY HERE!
        if spatial_step > 1:
            original_len = len(table)
            indices = np.arange(0, len(table), spatial_step)
            table = table.take(indices)
            logger.info(f"Sampled every {spatial_step}th row: {original_len:,} → {len(table):,} rows")
        
        # Add total_fungi column - LINUS: "One line, no bullshit"
        table = table.append_column('total_fungi', pc.add(table['am_fungi_richness'], table['ecm_fungi_richness']))
        
        logger.info(f"Loaded Arrow Table: {len(table):,} rows, {len(table.column_names)} columns")
        
        # Cache the Arrow Table directly
        _CACHE[cache_key] = table
        return table
    
    def _load_spatial(self, datasets: List[str], config: Dict[str, Any]) -> Dict[str, Union[str, rasterio.DatasetReader]]:
        """Return raster file paths or opened datasets for spatial analysis.
        
        NO NUMPY LOADING! Let methods use rasterio directly.
        
        Returns:
            Dict mapping dataset name to file path (str) or opened rasterio dataset
        """
        result = {}
        dataset_paths = config.get('datasets', {})
        
        if not dataset_paths:
            raise ValueError("No 'datasets' section in config! Cannot load spatial data.")
        
        # Option to return opened datasets vs just paths
        return_handles = config.get('return_raster_handles', False)
        
        for dataset in datasets:
            if dataset == 'multiband':
                # SIMPLE: multiband means merge all biodiversity rasters
                required_biodiv = ['plants', 'animals', 'am_fungi', 'ecm_fungi']
                raster_paths = {}
                
                for name in required_biodiv:
                    if name in dataset_paths:
                        path = dataset_paths[name]
                        if Path(path).exists():
                            raster_paths[name] = str(path)
                
                if not raster_paths:
                    raise ValueError(
                        f"No biodiversity rasters found in config!\n"
                        f"Expected at least one of: {required_biodiv}\n"
                        f"Got datasets: {list(dataset_paths.keys())}"
                    )
                
                # Check cache
                cache_key = f"multiband:{sorted(raster_paths.keys())}"
                if cache_key in _CACHE:
                    logger.info(f"Using cached multiband raster")
                    return _CACHE[cache_key]
                
                # Create multiband raster
                logger.info(f"Creating multiband raster from {len(raster_paths)} rasters")
                merged_dataset = self.create_multiband_raster(
                    raster_paths=raster_paths,
                    target_resolution='coarse'
                )
                
                result = {
                    'multiband': merged_dataset,
                    'band_names': list(raster_paths.keys())
                }
                
                _CACHE[cache_key] = result
                # Don't return early - let it accumulate in the main result dict
                result = _CACHE[cache_key]
                    
            elif dataset == 'multiband_with_cv':
                # For trust_maps: need both multiband richness AND CV data
                # First get multiband richness data
                required_biodiv = ['plants', 'animals', 'am_fungi', 'ecm_fungi']
                richness_paths = {}
                for name in required_biodiv:
                    if name in dataset_paths:
                        path = dataset_paths[name]
                        if Path(path).exists():
                            richness_paths[name] = str(path)
                
                if richness_paths:
                    # Create multiband from richness rasters
                    merged_dataset = self.create_multiband_raster(
                        raster_paths=richness_paths,
                        target_resolution='coarse'
                    )
                    result = {
                        'multiband': merged_dataset,
                        'band_names': list(richness_paths.keys())
                    }
                    
                    # Also add CV data if available
                    cv_names = ['plants_cv', 'animals_cv', 'fungi_cv']
                    cv_data = {}
                    for name in cv_names:
                        if name in dataset_paths:
                            path = dataset_paths[name]
                            if Path(path).exists():
                                base_name = name.replace('_cv', '')
                                cv_data[f"{base_name}_cv"] = str(path)
                    
                    if cv_data:
                        result['cv_data'] = cv_data
                        logger.info(f"Loaded multiband with CV data: {len(cv_data)} CV rasters")
                else:
                    raise ValueError("No biodiversity rasters found for multiband_with_cv")
                    
            elif dataset == 'uncertainty':
                # Return paths to CV/uncertainty rasters
                cv_names = ['plants_cv', 'animals_cv', 'fungi_cv']
                found_any = False
                for name in cv_names:
                    if name in dataset_paths:
                        path = dataset_paths[name]
                        if not Path(path).exists():
                            raise FileNotFoundError(
                                f"Uncertainty raster does not exist: {path}\n"
                                f"trust_maps method requires uncertainty data!"
                            )
                        base_name = name.replace('_cv', '')
                        
                        if return_handles:
                            result[f"{base_name}_cv"] = rasterio.open(path)
                        else:
                            result[f"{base_name}_cv"] = str(path)
                        found_any = True
                
                if not found_any:
                    raise ValueError(
                        f"No uncertainty rasters found for trust_maps!\n"
                        f"Expected at least one of: {cv_names}\n"
                        f"Got datasets: {list(dataset_paths.keys())}"
                    )
                        
            elif dataset == 'coordinates':
                # Coordinates will be generated from first raster's transform
                # Methods should handle this themselves
                logger.info("Coordinate generation moved to methods for efficiency")
                    
            else:
                # Try direct lookup
                if dataset in dataset_paths:
                    path = dataset_paths[dataset]
                    if return_handles:
                        result[dataset] = rasterio.open(path)
                    else:
                        result[dataset] = str(path)
        
        return result
    
    def create_multiband_raster(self, raster_paths: Dict[str, Union[str, Path]], 
                                target_resolution: str = 'coarse') -> Union[Path, rasterio.DatasetReader]:
        """
        Create a multi-band raster from multiple input rasters using raster_merge pipeline.
        
        Args:
            raster_paths: Dict of name -> path for rasters to merge
            target_resolution: 'coarse', 'fine', or specific resolution
            
        Returns:
            Path to merged multi-band raster or opened dataset
        """
        if not raster_paths:
            raise ValueError("No raster paths provided")
        
        # Convert dict values to list of paths
        # Handle both string paths and rasterio.DatasetReader objects
        input_paths = []
        for p in raster_paths.values():
            if p is not None:
                if hasattr(p, 'name'):  # rasterio.DatasetReader
                    input_paths.append(Path(p.name))
                else:  # string path
                    input_paths.append(Path(p))
        
        if len(input_paths) < 2:
            # Single raster, no merge needed
            if len(input_paths) == 1:
                return rasterio.open(str(input_paths[0]))
            else:
                raise ValueError("No valid raster paths")
        
        # Create output path in a persistent cache directory
        # LINUS: Use persistent cache instead of temp to avoid premature cleanup!
        cache_dir = Path('/tmp/geo_cache')
        cache_dir.mkdir(exist_ok=True)
        import hashlib
        # Create unique name based on input files
        hash_key = hashlib.md5(str(sorted(input_paths)).encode()).hexdigest()[:8]
        output_path = cache_dir / f"merged_multiband_{hash_key}.tif"
        
        # If already exists, just open and return
        if output_path.exists():
            logger.info(f"Using existing merged file: {output_path}")
            return rasterio.open(str(output_path))
        
        # Configure merge based on resolution strategy
        config = RasterConfig(
            input_paths=input_paths,
            output_dir=cache_dir,
            output_name="merged_multiband",
            resolution_strategy=target_resolution if target_resolution in ['equal', 'average', 'lowest', 'highest'] else 'lowest',
            target_resolution='coarse' if target_resolution == 'coarse' else None,
            overlap_method='first',  # Just take first value for overlaps
            band_merge_strategy='auto',  # Create multi-band output
            compress='lzw'  # Use LZW compression (widely supported)
        )
        
        logger.info(f"Merging {len(input_paths)} rasters into multi-band raster at {output_path}")
        
        # Use the pipeline to merge
        merged_path = merge_rasters(
            input_paths=input_paths,
            output_path=output_path,
            config=config
        )
        
        # Open and return the merged multi-band raster
        merged_dataset = rasterio.open(merged_path)
        logger.info(f"Created multi-band raster: {merged_dataset.count} bands, shape: {merged_dataset.shape}")
        
        return merged_dataset
    
    def _load_raster_stats(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Compute raster statistics without loading full array.
        
        Memory efficient block-wise computation.
        """
        stats = {
            'sum': 0,
            'sum_sq': 0,
            'count': 0,
            'valid_count': 0,
            'min': float('inf'),
            'max': float('-inf')
        }
        
        with rasterio.open(path) as src:
            for ji, window in src.block_windows(1):
                data = src.read(1, window=window, masked=True)
                valid = data.compressed()
                
                if len(valid) > 0:
                    stats['sum'] += valid.sum()
                    stats['sum_sq'] += (valid ** 2).sum()
                    stats['count'] += data.size
                    stats['valid_count'] += len(valid)
                    stats['min'] = min(stats['min'], valid.min())
                    stats['max'] = max(stats['max'], valid.max())
            
            # Compute final statistics
            if stats['valid_count'] > 0:
                mean = stats['sum'] / stats['valid_count']
                variance = (stats['sum_sq'] / stats['valid_count']) - mean ** 2
                std = np.sqrt(variance) if variance > 0 else 0
                
                return {
                    'mean': float(mean),
                    'std': float(std),
                    'min': float(stats['min']),
                    'max': float(stats['max']),
                    'valid_pixels': int(stats['valid_count']),
                    'total_pixels': int(stats['count']),
                    'coverage': float(100 * stats['valid_count'] / stats['count']),
                    'shape': (src.height, src.width),
                    'bounds': src.bounds,
                    'crs': str(src.crs) if src.crs else None
                }
            else:
                return {
                    'error': 'No valid data',
                    'shape': (src.height, src.width)
                }
    
    def clear_cache(self):
        """Clear the cache. That's it."""
        global _CACHE
        _CACHE.clear()
    
    @staticmethod
    def downsample_on_read(
        src: rasterio.DatasetReader,
        band_num: int = 1,
        max_pixels: int = 2_000_000,
        resampling: str = 'nearest'
    ) -> np.ndarray:
        """Read and downsample raster data efficiently.
        
        LINUS: "Don't load what you don't need" - downsample DURING read.
        This should be the ONLY place where downsampling logic exists.
        
        Args:
            src: Rasterio dataset to read from
            band_num: Band number to read (1-indexed)
            max_pixels: Maximum pixels in output (default 2M for visualization)
            resampling: Resampling method ('nearest', 'bilinear', 'average')
            
        Returns:
            Downsampled array (or original if small enough)
        """
        import rasterio.enums
        
        total_pixels = src.height * src.width
        
        if total_pixels > max_pixels:
            # Calculate downsampling factor
            factor = int(np.sqrt(total_pixels / max_pixels))
            new_height = src.height // factor
            new_width = src.width // factor
            
            # Map string to resampling enum
            resampling_map = {
                'nearest': rasterio.enums.Resampling.nearest,
                'bilinear': rasterio.enums.Resampling.bilinear,
                'average': rasterio.enums.Resampling.average,
            }
            resampling_enum = resampling_map.get(resampling, rasterio.enums.Resampling.nearest)
            
            # Read with downsampling
            data = src.read(
                band_num,
                masked=True,
                out_shape=(new_height, new_width),
                resampling=resampling_enum
            )
            logger.info(f"Downsampled band {band_num} from {src.height}x{src.width} to {new_height}x{new_width}")
        else:
            # Small enough, read as-is
            data = src.read(band_num, masked=True)
            
        return data