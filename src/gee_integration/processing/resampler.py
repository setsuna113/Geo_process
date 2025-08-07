"""GEE Resampling Module

Implements recursive 2x downsampling for robust multi-resolution exports.
Instead of large aggregation factors that fail in GEE, we chain multiple 2x downsamples.
"""

from typing import Dict, Optional, List, Tuple, TYPE_CHECKING
from src.infrastructure.logging import get_logger

if TYPE_CHECKING:
    import ee

logger = get_logger(__name__)


class GEEResampler:
    """Recursive 2x downsampling for multi-resolution processing"""
    
    def __init__(self, base_resolution: float, outputs: Dict[str, bool], target_crs: str = 'EPSG:4326'):
        """
        Initialize resampler.
        
        Args:
            base_resolution: Base resolution for alignment (degrees)
            outputs: Dict of resolution multipliers to export flags (e.g., {'1x': True, '2x': False, '4x': True})
            target_crs: Target coordinate reference system
        """
        self.base_resolution = base_resolution
        self.outputs = outputs
        self.target_crs = target_crs
        self.logger = logger
        
        # Convert base resolution to meters (approximate at equator)
        self.base_scale = base_resolution * 111320
        
        # Parse output configuration
        self.resolutions_to_process = self._parse_outputs(outputs)
        
    def _parse_outputs(self, outputs: Dict[str, any]) -> List[Tuple[str, int, bool]]:
        """
        Parse output configuration to determine processing chain.
        Expects dict format with 'enabled' key.
        
        Returns:
            List of (name, downsample_factor, export_flag) tuples
        """
        resolutions = []
        
        # Parse each resolution key
        for key, value in outputs.items():
            if key.endswith('x'):
                try:
                    factor = int(key[:-1])
                    
                    # Expect dict format with 'enabled' key
                    if not isinstance(value, dict):
                        raise ValueError(f"Resolution config for {key} must be a dict with 'enabled' key, got {type(value)}")
                    export = value.get('enabled', False)
                    
                    # Validate factor is power of 2 and reasonable
                    if factor > 0 and (factor & (factor - 1)) == 0 and factor <= 16:
                        resolutions.append((key, factor, export))
                    else:
                        self.logger.warning(f"Skipping invalid resolution factor: {key} (must be power of 2, ≤16)")
                except ValueError:
                    self.logger.warning(f"Skipping invalid resolution key: {key}")
        
        # Sort by factor to ensure proper processing order
        resolutions.sort(key=lambda x: x[1])
        
        self.logger.info(f"Resolution processing chain: {[(r[0], r[2]) for r in resolutions]}")
        return resolutions
    
    def process_dataset(self, image: 'ee.Image', config: Dict) -> Dict[str, 'ee.Image']:
        """
        Process a single dataset through recursive downsampling.
        
        Args:
            image: Input Earth Engine image
            config: Dataset configuration with resampling method
            
        Returns:
            Dict mapping resolution names to processed images
        """
        import ee
        
        dataset_name = config.get('name', 'unknown')
        resampling_method = config.get('resampling', 'bilinear')
        band_name = config.get('band_name', 'value')
        
        self.logger.info(f"Processing {dataset_name} with method: {resampling_method}")
        
        # Step 1: Align to base resolution
        aligned_image = self._align_to_base(image, config)
        
        # Step 2: Build resolution pyramid through recursive downsampling
        resolution_images = {'1x': aligned_image}  # Start with base resolution
        
        # Get the appropriate reducer for this dataset
        reducer = self._get_reducer(resampling_method)
        
        # Process each resolution level
        current_image = aligned_image
        current_factor = 1
        
        for res_name, target_factor, should_export in self.resolutions_to_process:
            if target_factor == 1:
                # Already have base resolution
                continue
                
            # Calculate how many 2x downsamples needed from current position
            downsample_steps = self._calculate_steps(current_factor, target_factor)
            
            self.logger.debug(f"  {res_name}: {downsample_steps} steps from {current_factor}x to {target_factor}x")
            
            # Apply recursive 2x downsampling
            for step in range(downsample_steps):
                current_image = self._downsample_2x(current_image, reducer)
                current_factor *= 2
                
                # Store intermediate result if it matches a target resolution
                for check_name, check_factor, _ in self.resolutions_to_process:
                    if check_factor == current_factor and check_name not in resolution_images:
                        resolution_images[check_name] = current_image
                        self.logger.debug(f"    Stored {check_name} at factor {current_factor}")
        
        # Filter to only return resolutions that should be exported
        export_images = {
            name: img for name, img in resolution_images.items()
            if any(r[0] == name and r[2] for r in self.resolutions_to_process)
        }
        
        self.logger.info(f"  Generated {len(resolution_images)} resolutions, exporting {len(export_images)}")
        return export_images
    
    def _align_to_base(self, image: 'ee.Image', config: Dict) -> 'ee.Image':
        """
        Align image to base resolution grid.
        
        This ensures all datasets share the same pixel grid before merging.
        """
        import ee
        
        # Handle nodata values
        no_data = config.get('no_data_value') or config.get('no_data')
        if no_data is not None:
            # Mask nodata values
            image = image.updateMask(image.neq(no_data))
        
        # Reproject to base resolution
        # Use projection at scale to ensure proper alignment
        # CRITICAL: Use consistent Y-axis orientation (positive yScale) for all bands
        # This prevents projection mismatches during export
        base_transform = [
            self.base_scale / 111320,  # xScale (degrees)
            0,                          # xShearing
            0,                          # xTranslation  
            0,                          # yShearing
            self.base_scale / 111320,  # yScale (POSITIVE for consistent orientation)
            0                           # yTranslation
        ]
        target_projection = ee.Projection(self.target_crs, base_transform)
        
        # Use appropriate resampling for alignment
        resampling_method = config.get('resampling', 'bilinear')
        if resampling_method in ['bilinear', 'bicubic']:
            aligned = image.resample(resampling_method)
        else:
            # For discrete data, use nearest neighbor for alignment
            aligned = image
        
        # Reproject to ensure alignment
        aligned = aligned.reproject(
            crs=target_projection,
            scale=self.base_scale
        )
        
        # Convert to Float32 to ensure consistent data types across all bands
        # This prevents "incompatible data types" errors during export
        aligned = aligned.toFloat()
        
        # Preserve band names if specified
        band_name = config.get('band_name')
        if band_name and isinstance(band_name, str):
            aligned = aligned.rename(band_name)
        
        # Copy properties
        aligned = aligned.copyProperties(image, ['system:time_start'])
        
        return aligned
    
    def _get_reducer(self, method: str) -> 'ee.Reducer':
        """
        Get the appropriate reducer for a resampling method.
        """
        import ee
        
        reducers = {
            'max': ee.Reducer.max(),
            'min': ee.Reducer.min(),
            'mean': ee.Reducer.mean(),
            'mode': ee.Reducer.mode(),
            'median': ee.Reducer.median(),
            'bilinear': ee.Reducer.mean(),  # Use mean for bilinear approximation
            'bicubic': ee.Reducer.mean(),   # Use mean for bicubic approximation
            'rms': ee.Reducer.stdDev(),     # RMS for coefficient of variation
            'sum': ee.Reducer.sum(),
            'count': ee.Reducer.count()
        }
        
        return reducers.get(method, ee.Reducer.mean())
    
    def _downsample_2x(self, image: 'ee.Image', reducer: 'ee.Reducer') -> 'ee.Image':
        """
        Downsample image by factor of 2 using the specified reducer.
        
        This is the core operation that works reliably in GEE.
        """
        import ee
        
        # Get current scale
        current_projection = image.projection()
        current_scale = current_projection.nominalScale()
        
        # Target scale is 2x current
        target_scale = current_scale.multiply(2)
        
        # Create target projection with consistent Y-axis orientation
        # Use scale() method which preserves the CRS and orientation
        target_projection = current_projection.scale(2, 2)
        
        # Apply reduceResolution with conservative settings
        # 2x downsampling means 4 pixels -> 1 pixel, well within limits
        downsampled = image.reduceResolution(
            reducer=reducer,
            maxPixels=4,  # 2x2 pixels per output pixel
            bestEffort=False  # We know this should work
        )
        
        # Reproject to target scale
        # CRITICAL: Must extract and pass the scale from the projection
        # Otherwise GEE won't actually downsample
        downsampled = downsampled.reproject(
            crs=target_projection,
            scale=target_scale  # Pass the target scale explicitly
        )
        
        return downsampled
    
    def _calculate_steps(self, current_factor: int, target_factor: int) -> int:
        """
        Calculate number of 2x downsample steps needed.
        
        Args:
            current_factor: Current downsample factor (e.g., 1 for base)
            target_factor: Target downsample factor (e.g., 4 for 4x)
            
        Returns:
            Number of 2x downsample steps needed
        """
        import math
        
        # Calculate the additional factor needed
        additional_factor = target_factor // current_factor
        
        # Convert to number of 2x steps (log base 2)
        if additional_factor <= 1:
            return 0
        
        steps = int(math.log2(additional_factor))
        return steps
    
    def process_collection(self, images: List['ee.Image'], 
                          configs: List[Dict]) -> Dict[str, List['ee.Image']]:
        """
        Process multiple datasets through recursive downsampling.
        
        Args:
            images: List of Earth Engine images
            configs: List of dataset configurations
            
        Returns:
            Dict mapping resolution names to lists of processed images
        """
        # Process each dataset
        all_results = {}
        
        for image, config in zip(images, configs):
            dataset_results = self.process_dataset(image, config)
            
            # Organize by resolution
            for res_name, processed_image in dataset_results.items():
                if res_name not in all_results:
                    all_results[res_name] = []
                all_results[res_name].append(processed_image)
        
        # Log summary
        for res_name, images_list in all_results.items():
            self.logger.info(f"Resolution {res_name}: {len(images_list)} images ready")
        
        return all_results