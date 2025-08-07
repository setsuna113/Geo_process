"""GEE Merger Module

Handles merging of biodiversity and climate data in GEE.
"""

from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from src.infrastructure.logging import get_logger
from src.infrastructure.logging.decorators import log_operation
from config.gee.constants import (
    COARSE_MASK_THRESHOLD, 
    MAX_PIXELS_REDUCE,
    DEFAULT_LAT_MIN,
    DEFAULT_LAT_MAX,
    GLOBAL_LON_MIN,
    GLOBAL_LON_MAX,
    GLOBAL_LAT_MIN,
    GLOBAL_LAT_MAX
)

if TYPE_CHECKING:
    import ee

logger = get_logger(__name__)


class GEEMerger:
    """Merge biodiversity and climate data in GEE"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize merger"""
        self.logger = logger
        self.config = config or {}
    
    @log_operation("merge_datasets")
    def merge_datasets(self, 
                      biodiversity_images: List['ee.Image'],
                      climate_image: 'ee.Image',
                      merge_strategy: str = "addBands",
                      mask_ocean: bool = True,
                      include_elevation: bool = True,
                      elevation_image: Optional['ee.Image'] = None,
                      additional_layers: Optional[Dict[str, 'ee.Image']] = None,
                      scale: Optional[float] = None,
                      ocean_mask_config: Optional[Dict] = None) -> 'ee.Image':
        """
        Merge biodiversity and climate datasets.
        
        Args:
            biodiversity_images: List of biodiversity images
            climate_image: Climate data image (multi-band)
            merge_strategy: How to merge (currently only 'addBands')
            mask_ocean: Whether to mask out ocean areas (default True)
            include_elevation: Whether to include elevation data (default True)
            additional_layers: Optional dict of layer_name -> ee.Image for extra layers
            
        Returns:
            Merged multi-band image
        """
        import ee
        self.logger.info(f"Merging {len(biodiversity_images)} biodiversity images with climate data")
        
        # Debug: Check what we're getting
        for i, img in enumerate(biodiversity_images):
            self.logger.debug(f"Image {i}: type={type(img)}, class={img.__class__.__name__ if hasattr(img, '__class__') else 'unknown'}")
        
        # First, combine all biodiversity images
        # IMPORTANT: Unmask biodiversity data to prevent climate from inheriting sparse masks
        # Even though we upload without --nodata_value flag, GEE reads nodata from GeoTIFF metadata
        
        # Ensure all biodiversity images are ee.Image objects
        biodiversity_images = [ee.Image(img) for img in biodiversity_images]
        
        if len(biodiversity_images) == 1:
            biodiv_combined = biodiversity_images[0].unmask()
        else:
            # Stack biodiversity bands and unmask
            biodiv_combined = biodiversity_images[0].unmask()
            for img in biodiversity_images[1:]:
                # Type check and raise error if incorrect
                if not isinstance(img, ee.Image):
                    raise TypeError(f"Expected ee.Image but got {type(img)}. "
                                    f"This indicates an issue in the image loading process.")
                biodiv_combined = biodiv_combined.addBands(img.unmask())
        
        # Ensure we have ee.Image objects
        biodiv_combined = ee.Image(biodiv_combined)
        climate_image = ee.Image(climate_image)
        
        # Convert all images to Float32 for consistent export
        self.logger.info("Converting all bands to Float32 for export compatibility")
        biodiv_combined = biodiv_combined.toFloat()
        climate_image = climate_image.toFloat()
        
        # CRITICAL: DO NOT reproject here - the recursive resampler has already aligned everything!
        # All bands should already have consistent projections from the resampler
        # Reprojecting here causes Y-axis flips and projection mismatches
        self.logger.info("Skipping reprojection - bands already aligned by recursive resampler")
        
        # Get band info (lazy - don't call getInfo)
        biodiv_bands = biodiv_combined.bandNames()
        climate_bands = climate_image.bandNames()
        
        self.logger.info(f"Merging biodiversity bands with climate data")
        self.logger.info("Note: Biodiversity has been unmasked to ensure climate retains full land coverage")
        
        # Merge with climate data
        # CRITICAL: Band order must match BandConfig.get_ordered_bands()
        # The order here determines the band indices in the exported GeoTIFF
        if merge_strategy == "addBands":
            # Order: biodiversity first, then climate (as defined in BandConfig)
            merged = biodiv_combined.addBands(climate_image)
            
            # Add elevation if requested
            if include_elevation:
                if elevation_image is not None:
                    # Use provided pre-processed elevation
                    self.logger.info("Adding pre-processed elevation data")
                    merged = merged.addBands(elevation_image)
                else:
                    # Load elevation directly - MERIT DEM has native pyramids
                    # GEE will automatically use the appropriate pyramid level
                    self.logger.info("Adding MERIT DEM elevation (using native pyramids)")
                    elevation = self.load_elevation().toFloat()
                    # No reprojection needed - GEE handles pyramid selection at export
                    merged = merged.addBands(elevation)
            
            # Add any additional layers
            if additional_layers:
                for layer_name, layer_image in additional_layers.items():
                    self.logger.info(f"Adding additional layer: {layer_name}")
                    merged = merged.addBands(layer_image.toFloat())
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")
        
        # Add metadata (keep as server-side objects)
        # Note: bands are now ordered as biodiversity first, then climate
        total_bands = biodiv_bands.size().add(climate_bands.size())
        if include_elevation:
            total_bands = total_bands.add(1)
            
        merged = merged.set({
            'biodiversity_bands': biodiv_bands,
            'climate_bands': climate_bands,
            'elevation_included': include_elevation,
            'total_bands': total_bands,
            'merge_strategy': merge_strategy,
            'band_order': 'biodiversity_first'  # Track that biodiversity comes first
        })
        
        # Apply ocean mask if requested
        if mask_ocean:
            self.logger.info("Applying ocean mask to exclude water areas")
            # Get latitude bounds from config if available
            lat_bounds = None
            if hasattr(self, 'config') and 'masking' in self.config:
                mask_config = self.config['masking']
                if mask_config.get('enabled', True) and 'latitude_bounds' in mask_config:
                    bounds = mask_config['latitude_bounds']
                    lat_bounds = (bounds.get('min', DEFAULT_LAT_MIN), bounds.get('max', DEFAULT_LAT_MAX))
            merged = self.apply_land_mask(merged, latitude_bounds=lat_bounds, scale=scale, ocean_mask_config=ocean_mask_config)
        
        self.logger.info(f"✅ Created merged image")
        
        return merged
    
    def load_worldclim_v2(self, variables: Optional[List[str]] = None) -> 'ee.Image':
        """
        Load WorldClim V2 bioclimatic variables.
        
        Args:
            variables: List of variables to load (e.g., ['bio01', 'bio12'])
                      If None, loads all 19 variables
                      
        Returns:
            Multi-band image with WorldClim data
        """
        import ee
        self.logger.info("Loading WorldClim V2 data...")
        
        # WorldClim V2 is a single image with 19 bands
        # Each band represents a bioclimatic variable
        if variables is None:
            # Load all 19 bioclimatic variables
            variables = [f'bio{i:02d}' for i in range(1, 20)]
        
        # Load the WorldClim image with all bands
        # Note: Using V1 as it's the version available in GEE catalog
        # Config may specify V2 for clarity but we use V1 for actual loading
        worldclim_all = ee.Image("WORLDCLIM/V1/BIO")
        
        # Get the band names to select (bio01, bio02, etc.)
        band_indices = []
        for var in variables:
            # Extract number from 'bio01' format
            num = int(var[3:])
            band_indices.append(num)
        
        # WorldClim bands are named bio01, bio02, etc.
        band_names = [f'bio{i:02d}' for i in band_indices]
        
        # Select specified bands
        worldclim_combined = worldclim_all.select(band_names)
        
        # Debug: check how many bands we got
        self.logger.info(f"Selected bands: {band_names}")
        self.logger.info(f"Variables requested: {variables}")
        
        # No need to rename - WorldClim bands already have correct names
        
        # Add metadata
        worldclim_combined = worldclim_combined.set({
            'source': 'WORLDCLIM/V1/BIO',  # Actual source used
            'variables': variables,
            'version': '1.4',  # WorldClim V1 version
            'resolution': '30 arc-seconds (~1km)'
        })
        
        self.logger.info(f"✅ Loaded {len(variables)} WorldClim variables")
        
        return worldclim_combined
    
    def apply_land_mask(self, image: 'ee.Image', 
                       latitude_bounds: Optional[Tuple[float, float]] = None,
                       scale: Optional[float] = None,
                       ocean_mask_config: Optional[Dict] = None) -> 'ee.Image':
        """
        Apply a land mask to exclude ocean areas and optionally polar regions.
        
        Args:
            image: Image to mask
            latitude_bounds: Optional (min_lat, max_lat) to exclude polar regions
                           Common boundaries:
                           - Arctic Circle: 66.5°N
                           - Antarctic Circle: -66.5°S
                           - Tree line: ~70°N, -55°S
                           - Tundra limit: ~75°N, -60°S
            scale: Optional scale in meters for creating coarse mask
            
        Returns:
            Image with ocean areas (and optionally polar regions) masked out
        """
        import ee
        
        # Use MODIS land/water mask
        # 0 = land, 1 = water (ocean/lakes), 2 = shallow inland water (rare)
        water_mask_img = ee.Image("MODIS/006/MOD44W/2015_01_01").select('water_mask')
        
        # Handle resolution-specific ocean mask strategies from config
        if ocean_mask_config:
            method = ocean_mask_config.get('method', 'original')
            
            if method == 'disabled':
                # Skip ocean mask entirely - use only latitude bounds
                self.logger.info("Ocean mask disabled - using latitude bounds only")
                # Set a dummy mask that keeps all land pixels
                water_mask_img = ee.Image.constant(0)  # All pixels are "land"
                
            elif method == 'original':
                # Use raw MODIS mask (500m) - good for fine resolutions
                self.logger.info("Using original MODIS mask (500m resolution)")
                # No modification needed
                
            elif method == 'reduceResolution':
                # Aggregate mask for coarser resolutions
                reducer_type = ocean_mask_config.get('reducer', 'mean')
                threshold = ocean_mask_config.get('threshold', 0.5)
                max_pixels = ocean_mask_config.get('maxPixels', 256)
                
                self.logger.info(f"Aggregating ocean mask with {reducer_type} reducer")
                
                # Select appropriate reducer
                if reducer_type == 'mean':
                    reducer = ee.Reducer.mean()
                elif reducer_type == 'mode':
                    reducer = ee.Reducer.mode()
                else:
                    reducer = ee.Reducer.mean()
                
                # Apply aggregation
                water_mask_img = water_mask_img.reduceResolution(
                    reducer=reducer,
                    maxPixels=max_pixels
                )
                
                # If using mean reducer, apply threshold
                if reducer_type == 'mean':
                    water_mask_img = water_mask_img.gt(threshold)
                    self.logger.info(f"Applied threshold {threshold} for water classification")
                
                # Reproject to target scale if provided
                if scale:
                    water_mask_img = water_mask_img.reproject(
                        crs='EPSG:4326',
                        scale=scale
                    )
        else:
            # No ocean mask config provided - use default behavior
            if scale and scale > COARSE_MASK_THRESHOLD:  # > 5km resolution
                self.logger.info(f"Using default coarse ocean mask for {scale}m resolution")
                water_mask_img = water_mask_img.reduceResolution(
                    reducer=ee.Reducer.mode(),
                    maxPixels=MAX_PIXELS_REDUCE
                ).reproject(
                    crs='EPSG:4326',
                    scale=scale
                )
                self.logger.info("Using simplified coastlines for coarse resolution")
        
        # Create water mask (water=1, land=0)
        # We want to mask out water, so we check where water_mask equals 1
        water_pixels = water_mask_img.eq(1)
        
        # Apply latitude bounds if specified
        if latitude_bounds:
            min_lat, max_lat = latitude_bounds
            self.logger.info(f"Creating unified mask: ocean OR outside [{min_lat}°, {max_lat}°]")
            
            # Create latitude image
            lat_img = ee.Image.pixelLonLat().select('latitude')
            
            # Create out-of-bounds mask
            out_of_bounds = lat_img.lt(min_lat).Or(lat_img.gt(max_lat))
            
            # Create unified exclusion mask: water OR out-of-bounds
            exclusion_mask = water_pixels.Or(out_of_bounds)
            
            # Final mask is the inverse (keep pixels that are NOT excluded)
            final_mask = exclusion_mask.Not()
            
            self.logger.info(f"Applied unified mask - excluding water and areas outside [{min_lat}, {max_lat}]")
        else:
            # Just exclude water
            final_mask = water_pixels.Not()
            self.logger.info("Applied water mask only")
        
        # Apply the mask to all bands
        masked = image.updateMask(final_mask)
        
        # Add metadata
        masked = masked.set({
            'water_masked': True,
            'latitude_bounds': latitude_bounds if latitude_bounds else 'none'
        })
        
        return masked
    
    def load_elevation(self) -> 'ee.Image':
        """
        Load MERIT DEM elevation data.
        
        Returns:
            Single-band image with elevation in meters
        """
        import ee
        self.logger.info("Loading MERIT DEM elevation data...")
        
        # Load MERIT DEM - error-removed elevation model
        elevation = ee.Image("MERIT/DEM/v1_0_3").select('dem')
        
        # Add metadata
        elevation = elevation.set({
            'source': 'MERIT/DEM/v1_0_3',
            'units': 'meters',
            'resolution': '~90m',
            'coverage': '90N to 60S'
        })
        
        # Rename band for clarity
        elevation = elevation.rename('elevation')
        
        self.logger.info("✅ Loaded MERIT DEM elevation data")
        
        return elevation
    
    def create_sampling_grid(self, 
                           bounds: Tuple[float, float, float, float],
                           resolution: float) -> 'ee.FeatureCollection':
        """
        Create a grid of points for sampling.
        
        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat)
            resolution: Grid resolution in degrees
            
        Returns:
            FeatureCollection of grid points
        """
        import ee
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Create grid coordinates
        lon_count = int((max_lon - min_lon) / resolution)
        lat_count = int((max_lat - min_lat) / resolution)
        
        self.logger.info(f"Creating {lon_count} x {lat_count} sampling grid")
        
        # Generate points
        points = []
        for i in range(lon_count):
            lon = min_lon + (i * resolution)
            for j in range(lat_count):
                lat = min_lat + (j * resolution)
                
                # Create point feature
                point = ee.Feature(
                    ee.Geometry.Point([lon, lat]),
                    {'x': lon, 'y': lat}
                )
                points.append(point)
        
        # Create feature collection
        grid = ee.FeatureCollection(points)
        
        return grid
    
    def get_existing_data_bounds(self, parquet_bounds: Optional[Dict] = None) -> 'ee.Geometry':
        """
        Get bounds where biodiversity data exists.
        
        Args:
            parquet_bounds: Optional bounds from parquet metadata
            
        Returns:
            ee.Geometry representing data bounds
        """
        import ee
        if parquet_bounds:
            # Use provided bounds
            bounds = ee.Geometry.Rectangle([
                parquet_bounds['min_x'],
                parquet_bounds['min_y'],
                parquet_bounds['max_x'],
                parquet_bounds['max_y']
            ])
        else:
            # Default to global
            bounds = ee.Geometry.Rectangle([GLOBAL_LON_MIN, GLOBAL_LAT_MIN, GLOBAL_LON_MAX, GLOBAL_LAT_MAX])
        
        return bounds
    
    def get_data_bounds(self, image: 'ee.Image', scale: float = 10000) -> 'ee.Geometry':
        """
        Automatically detect the bounds of valid data in an image.
        
        Args:
            image: The image to get bounds from
            scale: Scale in meters for bounds calculation (coarser = faster)
            
        Returns:
            ee.Geometry representing the bounds of valid data
        """
        import ee
        
        self.logger.info("Detecting data bounds from image...")
        
        # Method 1: Try to get geometry directly from the image
        try:
            # This works if the image has a defined geometry
            image_geom = image.geometry()
            bounds = image_geom.bounds()
            
            # Check if it's not just a default global bounds
            bounds_info = bounds.getInfo()
            if bounds_info and 'coordinates' in bounds_info:
                coords = bounds_info['coordinates'][0]
                min_lon = min(c[0] for c in coords)
                max_lon = max(c[0] for c in coords)
                min_lat = min(c[1] for c in coords)
                max_lat = max(c[1] for c in coords)
                
                # If it's not global bounds, use it
                if not (min_lon < -179 and max_lon > 179 and min_lat < -89 and max_lat > 89):
                    self.logger.info(f"Using image geometry bounds: ({min_lon:.2f}, {min_lat:.2f}, {max_lon:.2f}, {max_lat:.2f})")
                    return bounds
        except Exception as e:
            self.logger.warning(f"Could not get bounds from image geometry: {e}")
        
        # Method 2: Use reduceRegion to find min/max coordinates where data exists
        try:
            # Create a mask of valid pixels (not null and not nodata)
            valid_mask = image.select(0).mask()
            
            # Create coordinate images
            coords = ee.Image.pixelLonLat()
            
            # Mask coordinates to only include valid data areas
            masked_coords = coords.updateMask(valid_mask)
            
            # Find min/max of coordinates where data exists
            bounds_dict = masked_coords.reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=ee.Geometry.Rectangle([GLOBAL_LON_MIN, GLOBAL_LAT_MIN, GLOBAL_LON_MAX, GLOBAL_LAT_MAX]),
                scale=scale,
                maxPixels=1e9,
                tileScale=4
            )
            
            # Extract bounds
            bounds_info = bounds_dict.getInfo()
            if bounds_info:
                min_lon = bounds_info.get('longitude_min', GLOBAL_LON_MIN)
                max_lon = bounds_info.get('longitude_max', GLOBAL_LON_MAX)
                min_lat = bounds_info.get('latitude_min', -90)
                max_lat = bounds_info.get('latitude_max', 90)
                
                # Check if we got valid values
                if all(v is not None for v in [min_lon, max_lon, min_lat, max_lat]):
                    self.logger.info(f"Detected data bounds: ({min_lon:.2f}, {min_lat:.2f}, {max_lon:.2f}, {max_lat:.2f})")
                    
                    # Create bounds geometry
                    bounds = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
                    return bounds
                else:
                    self.logger.warning("Got None values in bounds detection")
        except Exception as e:
            self.logger.error(f"Error in reduceRegion bounds detection: {e}")
            self.logger.debug(f"Full traceback:", exc_info=True)
        
        # Method 3: Fallback to using the footprint of the first band
        try:
            footprint = image.select(0).geometry()
            bounds = footprint.bounds()
            self.logger.info("Using image footprint as bounds")
            return bounds
        except Exception as e:
            self.logger.error(f"Error getting image footprint: {e}")
        
        # Final fallback: return None to indicate bounds detection failed
        self.logger.error("Failed to detect data bounds - all methods failed")
        self.logger.warning("Export will proceed without explicit bounds (may use image default)")
        return None