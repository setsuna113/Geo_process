from src.abstractions import get_logger
"""
VRT-based merger following FarmVibes implementation exactly.

NO FALLBACKS - if VRT fails, it fails. No silent degradation to rioxarray.
"""

from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import logging
import dataclasses
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform
import numpy as np
from contextlib import ExitStack

try:
    from ..config import RasterConfig
except ImportError:
    # Fallback for when module is run directly
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import RasterConfig

logger = get_logger(__name__)


class VRTMerger:
    """
    VRT-based merger following FarmVibes patterns exactly.
    
    Key differences from original merger.py:
    1. Uses WarpedVRT for virtual alignment (no data loading)
    2. Direct merge() call with VRT objects
    3. Strict ZSTD compression with proper predictors
    4. No fallback to rioxarray - VRT or error
    5. Multi-band merging with resolution mismatch handling
    """
    
    def __init__(self, config: RasterConfig):
        """
        Initialize VRT merger with config.
        
        Args:
            config: RasterConfig object
        """
        self.config = config
        self._validate_config()
        self.band_merge_strategy = getattr(config, 'band_merge_strategy', 'auto')  # flat or auto
    
    def _validate_config(self):
        """Validate configuration for VRT merging."""
        if self.config.compress != "zstd":
            raise ValueError(
                f"FarmVibes uses ZSTD compression exclusively, got: {self.config.compress}"
            )
        
        if self.config.blocksize != 512:
            raise ValueError(
                f"FarmVibes uses 512x512 blocks exclusively, got: {self.config.blocksize}"
            )
    
    def check_resolution_mismatch(self, paths: List[Path]) -> Dict[str, Any]:
        """
        Check for resolution mismatches without loading data.
        
        Uses VRT or minimal reads to check resolution efficiently.
        
        Args:
            paths: List of raster paths to check
            
        Returns:
            Dict with resolution info and mismatch details
        """
        resolutions = []
        bounds_list = []
        crs_list = []
        band_counts = []
        
        for path in paths:
            # Use VRT for efficient metadata reading
            with rasterio.open(path, 'r') as src:
                resolutions.append(src.res)
                bounds_list.append(src.bounds)
                crs_list.append(str(src.crs))
                band_counts.append(src.count)
        
        # Check for mismatches
        unique_res = list(set(resolutions))
        unique_crs = list(set(crs_list))
        unique_bands = list(set(band_counts))
        
        return {
            'resolutions': resolutions,
            'unique_resolutions': unique_res,
            'has_resolution_mismatch': len(unique_res) > 1,
            'crs_list': unique_crs,
            'has_crs_mismatch': len(unique_crs) > 1,
            'band_counts': band_counts,
            'unique_band_counts': unique_bands,
            'has_band_mismatch': len(unique_bands) > 1,
            'bounds': bounds_list
        }
    
    def merge(self) -> Path:
        """
        Merge rasters using VRT pattern - now with proper batching for many rasters.
        
        Handles unlimited number of rasters by processing in batches to avoid
        file descriptor exhaustion.
        
        Returns:
            Path to merged output file
        
        Raises:
            ValueError: If rasters have incompatible CRS/bands
            RuntimeError: If VRT creation or merge fails
        """
        output_path = self.config.output_dir / f"{self.config.output_name}.tif"
        
        # Get actual paths (handles new DatasetConfig structure)
        input_paths = self.config.get_input_paths()
        
        # Single raster - just copy/convert
        if len(input_paths) == 1:
            logger.info("Single input raster - converting to output format")
            return self._convert_single_raster(input_paths[0], output_path)
        
        # Check for resolution mismatches first
        mismatch_info = self.check_resolution_mismatch(input_paths)
        
        if mismatch_info['has_resolution_mismatch']:
            logger.warning(f"Resolution mismatch detected: {mismatch_info['unique_resolutions']}")
            if self.config.target_resolution:
                logger.info(f"Will resample to target resolution: {self.config.target_resolution}")
            else:
                logger.info(f"Will resample to {self.config.resolution_strategy} resolution")
        
        # Process in batches if too many files
        max_open_files = 50  # Conservative limit to avoid "too many open files"
        
        if len(input_paths) > max_open_files:
            logger.info(f"Processing {len(input_paths)} rasters in batches of {max_open_files}")
            return self._merge_in_batches(input_paths, output_path, max_open_files, mismatch_info)
        
        # Small number of files - process all at once
        with ExitStack() as stack:
            # Step 1: Open all source rasters
            logger.info(f"Opening {len(input_paths)} source rasters")
            sources = []
            for path in input_paths:
                src = stack.enter_context(rasterio.open(path, "r"))
                sources.append(src)
            
            # Step 2: Validate compatibility (FarmVibes check_rasters_are_valid)
            self._validate_rasters(sources)
            
            # Step 3: Compute target resolution
            target_res = self._compute_target_resolution(sources)
            
            # Step 4: Get VRT options from first raster (reference)
            vrt_options = self._get_vrt_options(sources[0], target_res)
            
            # Step 5: Handle band merging based on strategy
            if self.band_merge_strategy == 'auto':
                # Smart band detection and merging
                logger.info("Using auto band strategy - detecting unique bands")
                output_path = self._merge_auto_bands(sources, stack, vrt_options, target_res, output_path)
            elif self.band_merge_strategy == 'flat':
                # Step 6: Create WarpedVRT for each source with resampling
                logger.info("Creating WarpedVRT for virtual alignment and resampling")
                vrts = []
                for i, src in enumerate(sources):
                    # Add resampling to VRT options for resolution mismatch
                    vrt_opts = vrt_options.copy()
                    if mismatch_info['has_resolution_mismatch']:
                        # Get per-dataset resampling if configured
                        src_path = input_paths[i]
                        # Use smart resampling based on data type
                        if hasattr(self.config, 'get_resampling_for_path'):
                            resampling = self.config.get_resampling_for_path(src_path)
                        else:
                            # Smart resampling based on dtype
                            resampling = self._get_smart_resampling(src.dtypes[0])
                        vrt_opts['resampling'] = resampling
                        logger.info(f"Using {resampling.name} resampling for {src_path.name}")
                    
                    # Pre-align all sources to the same grid using VRT
                    # This ensures merge() receives identically gridded inputs
                    vrt = stack.enter_context(WarpedVRT(src, **vrt_opts))
                    vrts.append(vrt)
                
                # Step 7: Get destination kwargs (compression settings)
                dtype = sources[0].meta.get("dtype", "float32")
                dst_kwds = self._get_dst_kwds(dtype)
                
                # Step 8: Perform merge with VRTs (FarmVibes pattern exactly)
                logger.info(f"Merging with method: {self.config.overlap_method}")
                
                # Get merge method - handle "mean" as custom callable
                merge_method = self._get_merge_method(self.config.overlap_method)
                
                # VRTs are pre-aligned with transform/dimensions, so no need for bounds/res
                # The VRT options already handle all alignment through transform, height, width
                merge(
                    vrts,
                    method=merge_method,
                    dst_path=str(output_path),
                    dst_kwds=dst_kwds
                )
        
        logger.info(f"Merge complete: {output_path}")
        return output_path
    
    def _validate_rasters(self, rasters: List[rasterio.DatasetReader]):
        """
        Validate raster compatibility (FarmVibes check_rasters_are_valid).
        
        Args:
            rasters: List of opened raster datasets
        
        Raises:
            ValueError: If rasters are incompatible
        """
        # Collect metadata for validation
        from collections import defaultdict
        assets_meta = defaultdict(set)
        
        FIELDS = ("crs", "dtype", "count")  # FarmVibes validation fields
        
        for r in rasters:
            for field in FIELDS:
                if field == "crs":
                    # Store CRS as string for comparison
                    assets_meta[field].add(str(r.meta[field]))
                else:
                    assets_meta[field].add(r.meta[field])
        
        # Check CRS compatibility
        if len(assets_meta["crs"]) > 1:
            raise ValueError(
                f"Cannot merge rasters with different CRS: {assets_meta['crs']}"
            )
        
        # Band count compatibility is now handled by auto/flat strategy
        # No need to check - auto mode handles any band configuration
        
        # Check resolution if using 'equal' strategy
        if self.config.resolution_strategy == "equal":
            resolutions = list({r.res for r in rasters})
            if len(resolutions) > 1:
                raise ValueError(
                    f"Resolution strategy 'equal' but found different resolutions: {resolutions}"
                )
    
    def _compute_target_resolution(self, rasters: List[rasterio.DatasetReader]) -> Optional[Tuple[float, float]]:
        """
        Compute target resolution based on config.
        
        Now supports:
        1. Explicit target_resolution from config
        2. "coarse" to align with coarsest resolution
        3. Legacy resolution_strategy for backward compatibility
        
        Args:
            rasters: List of opened raster datasets
        
        Returns:
            Target resolution tuple or None for equal
        """
        # Check if explicit target resolution is set
        if self.config.target_resolution is not None:
            if self.config.target_resolution == "coarse":
                # Use the coarsest (highest value) resolution
                resolutions = [r.res for r in rasters]
                coarsest_res = tuple(np.max(resolutions, axis=0))
                logger.info(f"Using coarsest resolution: {coarsest_res}")
                return coarsest_res
            elif isinstance(self.config.target_resolution, (int, float)):
                # Use explicit resolution value for both X and Y
                target_res = (float(self.config.target_resolution), float(self.config.target_resolution))
                logger.info(f"Using explicit target resolution: {target_res}")
                return target_res
        
        # Fall back to legacy resolution_strategy
        strategy = self.config.resolution_strategy
        
        if strategy == "equal":
            # No resampling needed
            return None
        
        # Collect all resolutions
        resolutions = [r.res for r in rasters]
        
        if len(set(resolutions)) > 1:
            logger.warning(
                f"Found multiple resolutions when merging, using {strategy}: {resolutions}"
            )
        
        # Apply resolution strategy (FarmVibes RESOLUTION_METHODS)
        if strategy == "average":
            return tuple(np.mean(resolutions, axis=0))
        elif strategy == "lowest":
            return tuple(np.min(resolutions, axis=0))
        elif strategy == "highest":
            return tuple(np.max(resolutions, axis=0))
        else:
            raise ValueError(f"Unknown resolution strategy: {strategy}")
    
    def _get_vrt_options(self, reference: rasterio.DatasetReader, target_res: Optional[Tuple[float, float]]) -> Dict[str, Any]:
        """
        Get VRT options for WarpedVRT creation.
        
        Args:
            reference: Reference raster (first in list)
            target_res: Target resolution or None
        
        Returns:
            Options dict for WarpedVRT
        """
        vrt_options = {
            "crs": reference.crs,
            "transform": reference.transform,
            "height": reference.height,
            "width": reference.width,
        }
        
        # Apply target resolution if specified
        if target_res is not None:
            # Recalculate dimensions for new resolution
            bounds = reference.bounds
            width = int((bounds.right - bounds.left) / target_res[0])
            height = int((bounds.top - bounds.bottom) / target_res[1])
            
            # Update VRT options
            from rasterio.transform import from_bounds
            vrt_options["transform"] = from_bounds(
                bounds.left, bounds.bottom, bounds.right, bounds.top,
                width, height
            )
            vrt_options["width"] = width
            vrt_options["height"] = height
        
        return vrt_options
    
    def _get_dst_kwds(self, dtype: str) -> Dict[str, Any]:
        """
        Get destination kwargs (FarmVibes FLOAT/INT_COMPRESSION_KWARGS).
        
        Args:
            dtype: Data type string
        
        Returns:
            Destination kwargs for merge
        """
        # Base kwargs (FarmVibes pattern)
        dst_kwds = {
            "driver": "GTiff",
            "tiled": True,
            "compress": "ZSTD",
            "zstd_level": self.config.compression_level,  # Configurable (default 9 for best compression)
            "blockxsize": self.config.blocksize,
            "blockysize": self.config.blocksize,
            "NUM_THREADS": "ALL_CPUS",  # Parallel compression
            "BIGTIFF": "IF_SAFER",  # Handle large files automatically
        }
        
        # Add predictor based on dtype (FarmVibes pattern)
        if 'int' in str(dtype).lower():
            # INT_COMPRESSION_KWARGS
            dst_kwds["predictor"] = 2  # Horizontal differencing
            dst_kwds["interleave"] = "pixel"
        else:
            # FLOAT_COMPRESSION_KWARGS
            dst_kwds["predictor"] = 3  # Floating point predictor
            dst_kwds["interleave"] = "band"
        
        # Add overviews if requested
        if self.config.build_overviews:
            dst_kwds["COPY_SRC_OVERVIEWS"] = True
        
        return dst_kwds
    
    def _get_smart_resampling(self, dtype: str = None) -> Resampling:
        """
        Choose appropriate resampling based on data type and config.
        
        Smart selection based on profiling results:
        - Integer data (likely categorical): Use nearest (5x faster)
        - Float data (continuous values): Use bilinear
        - Can be overridden by config
        
        Args:
            dtype: Data type string
            
        Returns:
            Resampling method
        """
        # Allow override from config
        if hasattr(self.config, 'resampling_method'):
            method = self.config.resampling_method.lower()
            if method == 'nearest':
                return Resampling.nearest
            elif method == 'bilinear':
                return Resampling.bilinear
            elif method == 'cubic':
                return Resampling.cubic
            else:
                return Resampling.bilinear
        
        # Smart default based on data type
        if dtype and 'int' in str(dtype).lower():
            # Integer data - likely categorical, use nearest (5x faster)
            logger.info(f"Using nearest resampling for integer data type: {dtype}")
            return Resampling.nearest
        else:
            # Float data - continuous values, use bilinear
            logger.info(f"Using bilinear resampling for float data type: {dtype}")
            return Resampling.bilinear
    
    def _get_merge_method(self, method_name: str):
        """
        Get merge method - convert string to callable if needed.
        
        Args:
            method_name: Method name (first, last, min, max, mean, sum, count)
            
        Returns:
            Method name or callable for merge
        """
        # Standard methods supported by rasterio
        if method_name in ['first', 'last', 'min', 'max', 'sum', 'count']:
            return method_name
        
        # Custom method for mean
        if method_name == 'mean':
            def mean_merger(old_data, new_data, old_nodata, new_nodata, index=None, roff=None, coff=None):
                """Custom mean merger for overlapping pixels."""
                # Create masks for valid data
                old_mask = ~np.isnan(old_data) if old_nodata is None else old_data != old_nodata
                new_mask = ~np.isnan(new_data) if new_nodata is None else new_data != new_nodata
                
                # Initialize output
                merged = old_data.copy()
                
                # Where both have data, take mean
                both_valid = old_mask & new_mask
                merged[both_valid] = (old_data[both_valid] + new_data[both_valid]) / 2.0
                
                # Where only new has data, use new
                only_new = new_mask & ~old_mask
                merged[only_new] = new_data[only_new]
                
                return merged
            
            return mean_merger
        
        # Unknown method
        raise ValueError(f"Unknown merge method: {method_name}")
    
    def _merge_auto_bands(self, sources: List[rasterio.DatasetReader], 
                         stack: ExitStack, vrt_options: Dict[str, Any], 
                         target_res: Optional[Tuple[float, float]], 
                         output_path: Path) -> Path:
        """
        Intelligently merge rasters by detecting and grouping bands by name.
        Each unique band name gets its own flat merge across all inputs that have it.
        
        Args:
            sources: List of opened raster datasets
            stack: ExitStack for resource management
            vrt_options: VRT options for alignment
            target_res: Target resolution
            output_path: Output file path
            
        Returns:
            Path to merged output file
        """
        from collections import defaultdict
        logger.info("Performing auto band merge - grouping by band names")
        
        # Get actual paths for per-dataset resampling
        input_paths = self.config.get_input_paths()
        
        # Group bands by their semantic names
        bands_by_name = defaultdict(list)  # band_name -> [(vrt, band_idx, source_idx), ...]
        
        # Create VRTs for all sources first
        vrts = []
        for i, src in enumerate(sources):
            logger.info(f"Processing source {i+1}: {input_paths[i].name} with {src.count} bands")
            
            # Create WarpedVRT for this source to handle resolution
            vrt_opts = vrt_options.copy()
            if target_res:
                src_path = input_paths[i]
                resampling = self._get_smart_resampling(src.dtypes[0])
                vrt_opts['resampling'] = resampling
                logger.info(f"Using {resampling.name} resampling for {src_path.name}")
                # Update dimensions for new resolution
                bounds = src.bounds
                width = int((bounds.right - bounds.left) / target_res[0])
                height = int((bounds.top - bounds.bottom) / target_res[1])
                vrt_opts['width'] = width
                vrt_opts['height'] = height
                
                from rasterio.transform import from_bounds
                vrt_opts['transform'] = from_bounds(
                    bounds.left, bounds.bottom, bounds.right, bounds.top,
                    width, height
                )
            
            vrt = stack.enter_context(WarpedVRT(src, **vrt_opts))
            vrts.append(vrt)
            
            # Group bands by their descriptions/names
            for band_idx in range(1, src.count + 1):
                # Try to get band description, fall back to file name
                band_description = src.descriptions[band_idx-1] if src.descriptions else None
                if not band_description:
                    # Use filename without extension as band name
                    band_description = input_paths[i].stem
                    if src.count > 1:
                        band_description = f"{band_description}_band{band_idx}"
                
                logger.info(f"  Band {band_idx}: {band_description}")
                bands_by_name[band_description].append((vrt, band_idx, i))
        
        # Now we have bands grouped by name
        unique_bands = list(bands_by_name.keys())
        logger.info(f"Detected {len(unique_bands)} unique bands: {unique_bands}")
        
        # Get output profile from first VRT
        first_vrt = vrts[0]
        profile = first_vrt.profile.copy()
        
        # Update profile for multi-band output with optimized compression
        profile.update({
            'count': len(unique_bands),
            'driver': 'GTiff',
            'compress': self.config.compress,
            'tiled': True,
            'blockxsize': self.config.blocksize,
            'blockysize': self.config.blocksize,
            'NUM_THREADS': 'ALL_CPUS',  # Parallel compression
            'BIGTIFF': 'IF_SAFER',  # Handle large files
        })
        
        # Add compression kwargs
        dtype = profile.get('dtype', 'float32')
        if 'int' in str(dtype).lower():
            profile['predictor'] = 2
        else:
            profile['predictor'] = 3
        
        if self.config.compress == 'ZSTD':
            profile['zstd_level'] = self.config.compression_level
        
        # Write output with intelligent band merging
        with rasterio.open(output_path, 'w', **profile) as dst:
            for output_band_idx, band_name in enumerate(unique_bands, 1):
                logger.info(f"Writing band {output_band_idx}/{len(unique_bands)}: {band_name}")
                
                # Get all sources for this band name
                band_sources = bands_by_name[band_name]
                
                if len(band_sources) == 1:
                    # Single source for this band - direct copy
                    vrt, src_band_idx, _ = band_sources[0]
                    windows = list(dst.block_windows())
                    for window_idx, window in windows:
                        data = vrt.read(src_band_idx, window=window)
                        dst.write(data, output_band_idx, window=window)
                else:
                    # Multiple sources for this band - flat merge them
                    logger.info(f"  Merging {len(band_sources)} sources for band '{band_name}'")
                    
                    # Create list of VRTs and band indices for this specific band
                    band_vrts = [(vrt, band_idx) for vrt, band_idx, _ in band_sources]
                    
                    # Use rasterio.merge for this specific band
                    from rasterio.merge import merge as rio_merge
                    
                    # Create temporary VRTs that expose only the specific band
                    temp_datasets = []
                    for vrt, band_idx in band_vrts:
                        # Read the specific band data
                        # Process in chunks for memory efficiency
                        windows = list(dst.block_windows())
                        for window_idx, window in windows:
                            arrays = []
                            for vrt, band_idx in band_vrts:
                                data = vrt.read(band_idx, window=window)
                                arrays.append(data)
                            
                            # Apply overlap method to merge arrays
                            if self.config.overlap_method == 'first':
                                merged = arrays[0]  # Use first valid
                                for arr in arrays[1:]:
                                    mask = np.isnan(merged) if dst.nodata is None else merged == dst.nodata
                                    merged[mask] = arr[mask]
                            elif self.config.overlap_method == 'max':
                                merged = np.nanmax(arrays, axis=0)
                            elif self.config.overlap_method == 'min':
                                merged = np.nanmin(arrays, axis=0)
                            elif self.config.overlap_method == 'mean':
                                merged = np.nanmean(arrays, axis=0)
                            else:  # last
                                merged = arrays[-1]
                                for arr in reversed(arrays[:-1]):
                                    mask = np.isnan(merged) if dst.nodata is None else merged == dst.nodata
                                    merged[mask] = arr[mask]
                            
                            dst.write(merged, output_band_idx, window=window)
                        break  # Only process windows once
                
                # Set band description
                dst.set_band_description(output_band_idx, band_name)
        
        logger.info(f"Multi-band merge complete: {output_path}")
        return output_path


    def _convert_single_raster(self, input_path: Path, output_path: Path) -> Path:
        """
        Convert single raster to output format with compression.
        """
        import shutil
        
        with rasterio.open(input_path) as src:
            # Get output kwargs
            dst_kwds = self._get_dst_kwds(src.meta.get("dtype", "float32"))
            
            # Update profile
            profile = src.profile.copy()
            profile.update(dst_kwds)
            
            # Write output
            with rasterio.open(output_path, 'w', **profile) as dst:
                for band in range(1, src.count + 1):
                    dst.write(src.read(band), band)
        
        logger.info(f"Converted single raster to: {output_path}")
        return output_path
    
    def _merge_in_batches(self, input_paths: List[Path], output_path: Path, 
                         batch_size: int, mismatch_info: Dict[str, Any]) -> Path:
        """
        Merge many rasters by processing in batches.
        
        First merges batches into temporary files, then merges those.
        """
        import tempfile
        import shutil
        
        temp_dir = Path(tempfile.mkdtemp(prefix="raster_merge_"))
        try:
            batch_outputs = []
            
            # Process in batches
            for i in range(0, len(input_paths), batch_size):
                batch = input_paths[i:i+batch_size]
                batch_num = i // batch_size + 1
                
                logger.info(f"Processing batch {batch_num} ({len(batch)} rasters)")
                
                # Create temp output for this batch
                batch_output = temp_dir / f"batch_{batch_num:03d}.tif"
                
                # Process this batch
                with ExitStack() as stack:
                    sources = [stack.enter_context(rasterio.open(p, "r")) for p in batch]
                    
                    # Validate compatibility within batch
                    self._validate_rasters(sources)
                    
                    # Get target resolution
                    target_res = self._compute_target_resolution(sources)
                    
                    # Get VRT options
                    vrt_options = self._get_vrt_options(sources[0], target_res)
                    
                    # Create VRTs
                    vrts = []
                    for j, src in enumerate(sources):
                        vrt_opts = vrt_options.copy()
                        if mismatch_info['has_resolution_mismatch'] and target_res:
                            src_path = batch[j]
                            # Smart resampling selection
                            if hasattr(self.config, 'get_resampling_for_path'):
                                resampling = self.config.get_resampling_for_path(src_path)
                            else:
                                resampling = self._get_smart_resampling(src.dtypes[0])
                            vrt_opts['resampling'] = resampling
                        vrt = stack.enter_context(WarpedVRT(src, **vrt_opts))
                        vrts.append(vrt)
                    
                    # Get destination kwargs
                    dtype = sources[0].meta.get("dtype", "float32")
                    dst_kwds = self._get_dst_kwds(dtype)
                    
                    # Merge this batch
                    merge_method = self._get_merge_method(self.config.overlap_method)
                    # VRTs are pre-aligned, merge them directly
                    merge(
                        vrts,
                        method=merge_method,
                        dst_path=str(batch_output),
                        dst_kwds=dst_kwds
                    )
                
                batch_outputs.append(batch_output)
            
            # Now merge all batch outputs
            logger.info(f"Merging {len(batch_outputs)} batch outputs into final result")
            
            if len(batch_outputs) == 1:
                # Single batch - just move it
                shutil.move(str(batch_outputs[0]), str(output_path))
            else:
                # Recursively merge batch outputs
                # Create a new config without mutating the original
                final_config = dataclasses.replace(
                    self.config,
                    input_paths=batch_outputs,
                    output_dir=output_path.parent,
                    output_name=output_path.stem
                )
                
                # Create new merger for final merge
                final_merger = VRTMerger(final_config)
                final_merger.merge()
        
        finally:
            # Clean up temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        
        logger.info(f"Batch merge complete: {output_path}")
        return output_path


def merge_rasters_vrt(config: RasterConfig) -> Path:
    """
    Convenience function to merge rasters using VRT.
    
    This replaces the old merge functionality with proper VRT implementation.
    
    Args:
        config: RasterConfig object
    
    Returns:
        Path to merged output file
    """
    merger = VRTMerger(config)
    return merger.merge()