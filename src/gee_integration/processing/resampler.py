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
        method = config.get('resampling', 'bilinear')
        data_type = config.get('data_type', 'continuous')
        band_name = config.get('band_name', 'value')
        
        self.logger.debug(f"Resampling {band_name} with method: {method}")
        
        # Handle nodata values
        no_data = config.get('no_data')
        if no_data is not None:
            # Mask nodata values
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
                    'median': ee.Reducer.median()
                }
            except ImportError as e:
                raise ImportError("Earth Engine module not available. "
                                  "Please install with: pip install earthengine-api") from e
        
        reducer = self.RESAMPLING_METHODS[method]
        
        # Use reduceResolution for proper aggregation
        resampled = image.reduceResolution(
            reducer=reducer,
            maxPixels=65536
        ).reproject(
            crs=self.target_crs,
            scale=self.target_scale
        )
        
        return resampled
    
    def _resample_standard(self, image: 'ee.Image', method: str) -> 'ee.Image':
        """Standard resampling (bilinear, bicubic)"""
        if method == 'bicubic':
            # Apply bicubic resampling
            resampled = image.resample('bicubic')
        else:
            # Default bilinear
            resampled = image.resample('bilinear')
        
        # Reproject to target resolution
        resampled = resampled.reproject(
            crs=self.target_crs,
            scale=self.target_scale
        )
        
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