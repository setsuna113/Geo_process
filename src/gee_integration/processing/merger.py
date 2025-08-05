"""GEE Merger Module

Handles merging of biodiversity and climate data in GEE.
"""

from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from src.infrastructure.logging import get_logger
from src.infrastructure.logging.decorators import log_operation

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
                      include_elevation: bool = True) -> 'ee.Image':
        """
        Merge biodiversity and climate datasets.
        
        Args:
            biodiversity_images: List of biodiversity images
            climate_image: Climate data image (multi-band)
            merge_strategy: How to merge (currently only 'addBands')
            mask_ocean: Whether to mask out ocean areas (default True)
            include_elevation: Whether to include elevation data (default True)
            
        Returns:
            Merged multi-band image
        """
        import ee
        self.logger.info(f"Merging {len(biodiversity_images)} biodiversity images with climate data")
        
        # Debug: Check what we're getting
        for i, img in enumerate(biodiversity_images):
            self.logger.debug(f"Image {i}: type={type(img)}, class={img.__class__.__name__ if hasattr(img, '__class__') else 'unknown'}")
        
        # First, combine all biodiversity images
        # Note: Don't unmask here as resampler will re-mask NoData values
        if len(biodiversity_images) == 1:
            biodiv_combined = biodiversity_images[0]
        else:
            # Stack biodiversity bands
            biodiv_combined = biodiversity_images[0]
            for img in biodiversity_images[1:]:
                # Type check and raise error if incorrect
                if not isinstance(img, ee.Image):
                    raise TypeError(f"Expected ee.Image but got {type(img)}. "
                                    f"This indicates an issue in the image loading process.")
                biodiv_combined = biodiv_combined.addBands(img)
        
        # Ensure we have ee.Image objects
        biodiv_combined = ee.Image(biodiv_combined)
        climate_image = ee.Image(climate_image)
        
        # Convert all images to Float32 for consistent export
        self.logger.info("Converting all bands to Float32 for export compatibility")
        biodiv_combined = biodiv_combined.toFloat()
        climate_image = climate_image.toFloat()
        
        # Get band info (lazy - don't call getInfo)
        biodiv_bands = biodiv_combined.bandNames()
        climate_bands = climate_image.bandNames()
        
        self.logger.info(f"Merging biodiversity bands with climate data")
        
        # Merge with climate data
        if merge_strategy == "addBands":
            merged = biodiv_combined.addBands(climate_image)
            
            # Add elevation if requested
            if include_elevation:
                self.logger.info("Adding elevation data")
                elevation = self.load_elevation().toFloat()
                merged = merged.addBands(elevation)
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")
        
        # Add metadata (keep as server-side objects)
        total_bands = biodiv_bands.size().add(climate_bands.size())
        if include_elevation:
            total_bands = total_bands.add(1)
            
        merged = merged.set({
            'biodiversity_bands': biodiv_bands,
            'climate_bands': climate_bands,
            'elevation_included': include_elevation,
            'total_bands': total_bands,
            'merge_strategy': merge_strategy
        })
        
        # Apply ocean mask if requested
        if mask_ocean:
            self.logger.info("Applying ocean mask to exclude water areas")
            # Get latitude bounds from config if available
            lat_bounds = None
            if hasattr(self, 'config') and 'latitude_bounds' in self.config.get('processing', {}):
                lat_bounds = tuple(self.config['processing']['latitude_bounds'])
            merged = self.apply_land_mask(merged, latitude_bounds=lat_bounds)
        
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
                       latitude_bounds: Optional[Tuple[float, float]] = None) -> 'ee.Image':
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
            
        Returns:
            Image with ocean areas (and optionally polar regions) masked out
        """
        import ee
        
        # Use MODIS land/water mask
        # 0 = water, 1 = land, 2 = shallow water
        land_mask = ee.Image("MODIS/006/MOD44W/2015_01_01").select('water_mask').eq(1)
        
        # Apply latitude bounds if specified
        if latitude_bounds:
            min_lat, max_lat = latitude_bounds
            self.logger.info(f"Applying latitude mask: {min_lat}° to {max_lat}°")
            
            # Create latitude image
            lat_img = ee.Image.pixelLonLat().select('latitude')
            
            # Create latitude mask
            lat_mask = lat_img.gte(min_lat).And(lat_img.lte(max_lat))
            
            # Combine with land mask
            final_mask = land_mask.And(lat_mask)
            
            self.logger.info(f"Applied land mask with latitude bounds [{min_lat}, {max_lat}]")
        else:
            final_mask = land_mask
            self.logger.info("Applied land mask - ocean areas excluded")
        
        # Apply the mask to all bands
        masked = image.updateMask(final_mask)
        
        # Add metadata
        masked = masked.set({
            'ocean_masked': True,
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
            bounds = ee.Geometry.Rectangle([-180, -90, 180, 90])
        
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
                geometry=ee.Geometry.Rectangle([-180, -90, 180, 90]),
                scale=scale,
                maxPixels=1e9,
                tileScale=4
            )
            
            # Extract bounds
            bounds_info = bounds_dict.getInfo()
            if bounds_info:
                min_lon = bounds_info.get('longitude_min', -180)
                max_lon = bounds_info.get('longitude_max', 180)
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