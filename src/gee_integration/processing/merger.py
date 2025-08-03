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
    
    def __init__(self):
        """Initialize merger"""
        self.logger = logger
    
    @log_operation("merge_datasets")
    def merge_datasets(self, 
                      biodiversity_images: List['ee.Image'],
                      climate_image: 'ee.Image',
                      merge_strategy: str = "addBands") -> 'ee.Image':
        """
        Merge biodiversity and climate datasets.
        
        Args:
            biodiversity_images: List of biodiversity images
            climate_image: Climate data image (multi-band)
            merge_strategy: How to merge (currently only 'addBands')
            
        Returns:
            Merged multi-band image
        """
        import ee
        self.logger.info(f"Merging {len(biodiversity_images)} biodiversity images with climate data")
        
        # Debug: Check what we're getting
        for i, img in enumerate(biodiversity_images):
            self.logger.debug(f"Image {i}: type={type(img)}, class={img.__class__.__name__ if hasattr(img, '__class__') else 'unknown'}")
        
        # First, combine all biodiversity images
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
        
        # Get band info (lazy - don't call getInfo)
        biodiv_bands = biodiv_combined.bandNames()
        climate_bands = climate_image.bandNames()
        
        self.logger.info(f"Merging biodiversity bands with climate data")
        
        # Merge with climate data
        if merge_strategy == "addBands":
            merged = biodiv_combined.addBands(climate_image)
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")
        
        # Add metadata (keep as server-side objects)
        merged = merged.set({
            'biodiversity_bands': biodiv_bands,
            'climate_bands': climate_bands,
            'total_bands': biodiv_bands.size().add(climate_bands.size()),
            'merge_strategy': merge_strategy
        })
        
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