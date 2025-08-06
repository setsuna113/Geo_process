"""GEE Resampling Module

Handles resampling of different data types with appropriate methods.
"""

from typing import Dict, Optional, List, TYPE_CHECKING
from src.infrastructure.logging import get_logger

if TYPE_CHECKING:
    import ee

logger = get_logger(__name__)


class GEEResampler:
    """Handle resampling with different methods per dataset"""
    
    def __init__(self, target_resolution: float, target_crs: str = 'EPSG:4326'):
        """
        Initialize resampler.
        
        Args:
            target_resolution: Target resolution in degrees
            target_crs: Target coordinate reference system
        """
        self.target_resolution = target_resolution
        self.target_crs = target_crs
        self.logger = logger
        # Convert degrees to meters (approximate at equator)
        self.target_scale = target_resolution * 111320  # degrees to meters
        
        # Store configuration - ee will be imported when needed
        self.RESAMPLING_METHODS = None  # Will be initialized on first use
    
    def resample_image(self, image: 'ee.Image', config: Dict) -> 'ee.Image':
        """
        Resample image based on configuration.
        
        Args:
            image: Earth Engine image
            config: Dataset configuration with resampling method
            
        Returns:
            Resampled image
        """
        # Get resampling method from config - use dataset-specific method if available
        method = config.get('resampling', 'bilinear')
        data_type = config.get('data_type', 'continuous')
        band_name = config.get('band_name', 'value')
        
        self.logger.debug(f"Resampling {band_name} with method: {method}")
        
        # Handle nodata values and sparse data
        no_data = config.get('no_data')
        
        # Handle masking differently based on data type
        # For biodiversity data uploaded without nodata flag, no masking needed
        # For climate data with proper nodata values, mask them
        data_type = config.get('data_type', 'continuous')
        if data_type == 'discrete' and band_name != 'climate':
            # Biodiversity data - no unmasking needed since uploaded without nodata flag
            # This preserves the distinction between ocean mask and missing data values
            self.logger.debug(f"Processing {band_name} without additional masking")
        elif no_data is not None and any(keyword in str(band_name).lower() for keyword in ['climate', 'worldclim', 'elevation']):
            # Only mask nodata for climate/elevation data that was uploaded with nodata flag
            self.logger.debug(f"Masking nodata values ({no_data}) for {band_name}")
            image = image.updateMask(image.neq(no_data))
        
        # Apply resampling based on method
        if method in ['max', 'min', 'mean', 'mode', 'median']:
            # Use reducer for aggregation methods
            resampled = self._resample_with_reducer(image, method)
        else:
            # Use standard resampling
            resampled = self._resample_standard(image, method)
        
        # Set band name only if it's a single string (not for multi-band images)
        if isinstance(band_name, str) and not band_name.startswith('climate'):
            resampled = resampled.rename(band_name)
        
        # Copy properties
        resampled = resampled.copyProperties(image, ['system:time_start'])
        
        # Ensure we return an ee.Image
        try:
            import ee
            return ee.Image(resampled)
        except ImportError as e:
            raise ImportError("Earth Engine module not available. "
                              "Please install with: pip install earthengine-api") from e
    
    def _resample_with_reducer(self, image: 'ee.Image', method: str) -> 'ee.Image':
        """Resample using reducer (for aggregation methods)"""
        # Initialize resampling methods if not done
        if self.RESAMPLING_METHODS is None:
            try:
                import ee
                self.RESAMPLING_METHODS = {
                    'bilinear': None,  # Default resampling
                    'bicubic': 'bicubic',
                    'max': ee.Reducer.max(),
                    'min': ee.Reducer.min(),
                    'mean': ee.Reducer.mean(),
                    'mode': ee.Reducer.mode(),
                    'median': ee.Reducer.median(),
                    'stdDev': ee.Reducer.stdDev(),
                    'rms': ee.Reducer.stdDev(),  # RMS for coefficient of variation
                    'sum': ee.Reducer.sum(),
                    'count': ee.Reducer.count()
                }
            except ImportError as e:
                raise ImportError("Earth Engine module not available. "
                                  "Please install with: pip install earthengine-api") from e
        
        reducer = self.RESAMPLING_METHODS.get(method)
        if reducer is None:
            # For bilinear/bicubic, use standard resampling
            return self._resample_standard(image, method)
        
        # APPROACH 1: Aggregate then export without scale
        # This gives us control over aggregation method
        import ee
        
        # CRITICAL: We must reproject to target scale AFTER reduceResolution
        # The working pattern from GEE docs is:
        # 1. Get target projection at desired scale
        # 2. Apply reduceResolution with low maxPixels and bestEffort
        # 3. Reproject to target projection
        
        # Create target projection at desired scale
        target_projection = ee.Projection(self.target_crs).atScale(self.target_scale)
        
        # Apply reduceResolution for aggregation
        aggregated = image.reduceResolution(
            reducer=reducer,
            maxPixels=64,  # Low value to avoid "output too large" errors
            bestEffort=True  # Handle edge cases gracefully
        )
        
        # Reproject to target projection 
        # This is necessary to properly set the output resolution
        reprojected = aggregated.reproject(
            crs=target_projection
        )
        
        return reprojected
    
    def _resample_standard(self, image: 'ee.Image', method: str) -> 'ee.Image':
        """Standard resampling (bilinear, bicubic)"""
        if method == 'bicubic':
            # Apply bicubic resampling
            resampled = image.resample('bicubic')
        else:
            # Default bilinear
            resampled = image.resample('bilinear')
        
        # DON'T reproject here - let the export handle the scale
        # Reprojecting before export can cause coordinate transform errors
        # The export will automatically resample to the specified scale
        
        # COMMENTED OUT to fix hemisphere export issues
        # resampled = resampled.reproject(
        #     crs=self.target_crs,
        #     scale=self.target_scale
        # )
        
        return resampled
    
    def resample_collection(self, images: List['ee.Image'], 
                          configs: List[Dict]) -> List['ee.Image']:
        """
        Resample multiple images.
        
        Args:
            images: List of Earth Engine images
            configs: List of dataset configurations
            
        Returns:
            List of resampled images
        """
        resampled_images = []
        
        for image, config in zip(images, configs):
            try:
                resampled = self.resample_image(image, config)
                resampled_images.append(resampled)
                self.logger.info(f"✅ Resampled {config['name']} to {self.target_resolution}°")
            except Exception as e:
                self.logger.error(f"Failed to resample {config['name']}: {e}")
                raise
        
        return resampled_images