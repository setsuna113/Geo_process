"""Main rasterio resampling implementation with windowed processing."""

import os
import sys
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window
from rasterio.crs import CRS
import numpy as np
from typing import Tuple, Optional, Iterator, Dict, Any
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import signal
import json

from .config import ResamplingConfig
from .monitor import ResamplingMonitor

logger = logging.getLogger(__name__)


class RasterioResampler:
    """Rasterio-based resampler with memory management and progress tracking."""
    
    # Mapping of string names to rasterio resampling methods
    RESAMPLING_METHODS = {
        'nearest': Resampling.nearest,
        'bilinear': Resampling.bilinear,
        'cubic': Resampling.cubic,
        'cubic_spline': Resampling.cubic_spline,
        'lanczos': Resampling.lanczos,
        'average': Resampling.average,
        'mode': Resampling.mode,
        'max': Resampling.max,
        'min': Resampling.min,
        'med': Resampling.med,
        'q1': Resampling.q1,
        'q3': Resampling.q3,
        'sum': Resampling.sum
    }
    
    def __init__(self, config: ResamplingConfig):
        """Initialize resampler with configuration."""
        self.config = config
        self.monitor = None
        self._setup_logging()
        
        # Handle interrupts gracefully
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self):
        """Configure logging."""
        log_level = getattr(logging, self.config.log_level.upper())
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # File handler if specified
        handlers = [console_handler]
        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setLevel(log_level)
            handlers.append(file_handler)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        logger.info(f"Received signal {signum}, cleaning up...")
        if self.monitor:
            self.monitor.cleanup()
        sys.exit(0)
    
    def resample(self, input_path: str, output_path: Optional[str] = None) -> str:
        """Resample a raster file to target resolution.
        
        Args:
            input_path: Path to input raster
            output_path: Path for output raster (auto-generated if None)
            
        Returns:
            Path to resampled raster
        """
        # Generate output path if not provided
        if output_path is None:
            base_name = os.path.basename(input_path)
            name, ext = os.path.splitext(base_name)
            output_path = os.path.join(
                self.config.output_dir,
                f"{name}_resampled_{self.config.target_resolution:.6f}{ext}"
            )
        
        logger.info(f"Starting resampling: {input_path} -> {output_path}")
        
        with rasterio.open(input_path) as src:
            # Calculate transform for target resolution
            dst_transform, dst_width, dst_height = self._calculate_target_transform(src)
            
            # Get resampling method
            resampling_method = self.RESAMPLING_METHODS.get(
                self.config.resampling_method, 
                Resampling.average
            )
            
            # Calculate optimal window size
            window_size = self.config.get_optimal_window_size(
                src.width, src.height, 
                dtype_size=np.dtype(src.dtypes[0]).itemsize
            )
            
            logger.info(f"Source: {src.width}x{src.height} @ {src.res}")
            logger.info(f"Target: {dst_width}x{dst_height} @ {self.config.target_resolution}")
            logger.info(f"Window size: {window_size}x{window_size}")
            logger.info(f"Resampling method: {self.config.resampling_method}")
            
            # Create output raster
            profile = src.profile.copy()
            profile.update({
                'crs': self.config.target_crs,
                'transform': dst_transform,
                'width': dst_width,
                'height': dst_height,
                'compress': self.config.compress,
                'tiled': self.config.tiled,
                'blockxsize': self.config.blockxsize,
                'blockysize': self.config.blockysize
            })
            
            # Calculate windows
            windows = list(self._generate_windows(dst_width, dst_height, window_size))
            total_windows = len(windows)
            
            # Initialize monitor
            self.monitor = ResamplingMonitor(self.config, total_windows)
            self.monitor.start_monitoring()
            
            try:
                with rasterio.open(output_path, 'w', **profile) as dst:
                    # Process windows in parallel
                    self._process_windows_parallel(
                        src, dst, windows, dst_transform, resampling_method
                    )
                
                # Success!
                summary = self.monitor.get_summary()
                logger.info(f"Resampling completed successfully!")
                logger.info(f"Summary: {json.dumps(summary, indent=2)}")
                
                return output_path
                
            except Exception as e:
                logger.error(f"Resampling failed: {e}")
                self.monitor.add_error(str(e))
                raise
            finally:
                self.monitor.cleanup()
    
    def _calculate_target_transform(self, src) -> Tuple[Any, int, int]:
        """Calculate transform for target resolution and CRS.
        
        Always reprojects bounds to target CRS first to ensure correct dimensions.
        """
        dst_crs = CRS.from_string(self.config.target_crs)
        
        # Always calculate transform properly, regardless of CRS
        # This ensures bounds are correctly reprojected
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds,
            resolution=self.config.target_resolution
        )
        
        # Validate dimensions
        if dst_width <= 0 or dst_height <= 0:
            raise ValueError(f"Invalid output dimensions: {dst_width}x{dst_height}")
        
        # Log transform details for debugging
        if self.config.verbose:
            from rasterio.transform import xy
            left, top = xy(dst_transform, 0, 0)
            right, bottom = xy(dst_transform, dst_width, dst_height)
            print(f"Output bounds: ({left:.6f}, {bottom:.6f}, {right:.6f}, {top:.6f})")
            print(f"Output dimensions: {dst_width}x{dst_height}")
            print(f"Pixel size: {dst_transform[0]:.6f}, {abs(dst_transform[4]):.6f}")
        
        return dst_transform, dst_width, dst_height
    
    def _generate_windows(self, width: int, height: int, window_size: int) -> Iterator[Tuple[Window, str]]:
        """Generate windows for processing with unique IDs."""
        for row in range(0, height, window_size):
            for col in range(0, width, window_size):
                window = Window(
                    col_off=col,
                    row_off=row,
                    width=min(window_size, width - col),
                    height=min(window_size, height - row)
                )
                window_id = f"window_{row}_{col}"
                
                # Skip if already processed (for resume)
                if self.monitor and self.monitor.is_window_completed(window_id):
                    continue
                    
                yield window, window_id
    
    def _process_windows_parallel(self, src, dst, windows, dst_transform, resampling_method):
        """Process windows in parallel with proper memory management."""
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit tasks
            futures = {}
            for window, window_id in windows:
                future = executor.submit(
                    self._process_single_window,
                    src.name,  # File path instead of object
                    window,
                    window_id,
                    dst_transform,
                    src.crs.to_string(),
                    self.config.target_crs,
                    resampling_method,
                    src.nodata
                )
                futures[future] = (window, window_id)
            
            # Process completed tasks
            for future in as_completed(futures):
                window, window_id = futures[future]
                try:
                    result_data, window_info = future.result()
                    
                    # Write result to output
                    dst.write(result_data, 1, window=window)
                    
                    # Update progress
                    self.monitor.update_progress(window_id, window_info)
                    
                except Exception as e:
                    self.monitor.add_error(str(e), window_id)
                    logger.error(f"Failed to process {window_id}: {e}")
                    if self.config.debug:
                        import traceback
                        traceback.print_exc()
    
    @staticmethod
    def _process_single_window(src_path: str, window: Window, window_id: str,
                              dst_transform, src_crs: str, dst_crs: str,
                              resampling_method, nodata) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a single window (runs in separate process)."""
        # Open source in this process
        with rasterio.open(src_path) as src:
            # Create array for destination window
            dst_array = np.empty((window.height, window.width), dtype=src.dtypes[0])
            
            # Fill with nodata if specified
            if nodata is not None:
                dst_array.fill(nodata)
            
            # Calculate source window bounds in world coordinates
            window_transform = rasterio.windows.transform(window, dst_transform)
            
            # Reproject
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_array,
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=window_transform,
                dst_crs=dst_crs,
                resampling=resampling_method,
                dst_nodata=nodata
            )
            
            # Window info for monitoring
            window_info = {
                'window_id': window_id,
                'shape': dst_array.shape,
                'valid_pixels': np.sum(dst_array != nodata) if nodata is not None else dst_array.size,
                'memory_mb': dst_array.nbytes / (1024**2)
            }
            
            return dst_array, window_info
    
    def validate_output(self, output_path: str, input_path: str) -> Dict[str, Any]:
        """Validate the resampled output."""
        validation_results = {
            'valid': True,
            'warnings': [],
            'info': {}
        }
        
        with rasterio.open(input_path) as src, rasterio.open(output_path) as dst:
            # Check resolution
            actual_res = dst.res[0]
            expected_res = self.config.target_resolution
            res_diff = abs(actual_res - expected_res)
            
            if res_diff > expected_res * 0.01:  # 1% tolerance
                validation_results['warnings'].append(
                    f"Resolution mismatch: expected {expected_res}, got {actual_res}"
                )
            
            # Check CRS
            if dst.crs.to_string() != self.config.target_crs:
                validation_results['warnings'].append(
                    f"CRS mismatch: expected {self.config.target_crs}, got {dst.crs}"
                )
                validation_results['valid'] = False
            
            # Check data integrity
            src_stats = src.statistics(1)
            dst_stats = dst.statistics(1)
            
            validation_results['info'] = {
                'source': {
                    'shape': (src.width, src.height),
                    'resolution': src.res,
                    'crs': src.crs.to_string(),
                    'min': src_stats.min,
                    'max': src_stats.max,
                    'mean': src_stats.mean
                },
                'destination': {
                    'shape': (dst.width, dst.height),
                    'resolution': dst.res,
                    'crs': dst.crs.to_string(),
                    'min': dst_stats.min,
                    'max': dst_stats.max,
                    'mean': dst_stats.mean
                }
            }
            
            # For average resampling, check if mean is preserved (approximately)
            if self.config.resampling_method == 'average':
                mean_diff = abs(src_stats.mean - dst_stats.mean) / src_stats.mean
                if mean_diff > 0.05:  # 5% tolerance
                    validation_results['warnings'].append(
                        f"Mean value changed significantly: {mean_diff*100:.1f}%"
                    )
        
        return validation_results