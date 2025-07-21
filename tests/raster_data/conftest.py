# tests/raster_data/conftest.py
import pytest
import tempfile
from pathlib import Path
import numpy as np
from osgeo import gdal, osr
import shutil
from typing import Tuple, Optional
import psycopg2
import yaml
import logging

from src.config.config import Config
from src.database.connection import db
from src.database.setup import setup_database, reset_database
from src.database.schema import schema

logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def real_config(test_config_file):
    """Create a real Config instance instead of mock."""
    from src.config.config import Config
    return Config(test_config_file)

@pytest.fixture
def test_config_file(test_data_dir):
    """Create a test config file."""
    config_data = {
        'database': {
            'host': 'localhost',
            'port': 5432,
            'database': 'geoprocess_db',  # Use real database
            'user': 'jason',              # Use real user
            'password': '123456',         # Use real password
            'pool_size': 5,
            'max_overflow': 10
        },
        'raster_processing': {
            'tile_size': 100,
            'cache_ttl_days': 1,
            'memory_limit_mb': 512,
            'parallel_workers': 2
        },
        'lazy_loading': {
            'chunk_size_mb': 50,
            'prefetch_tiles': 2
        },
        'grids': {
            'cubic': {
                'resolutions': [10, 25, 50]
            },
            'hexagonal': {
                'resolutions': [10, 25]
            }
        }
    }
    
    config_path = test_data_dir / "test_config.yml"
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    return config_path

@pytest.fixture
def test_db(test_config_file):
    """Create a test database connection."""
    try:
        # Load test config
        from src.config.config import Config
        test_config = Config(test_config_file)
        
        # Test if database is accessible (don't reset - just verify connection)
        from src.database.connection import db
        if not db.test_connection():
            pytest.skip("Cannot connect to test database")
        
        yield db  # Use the global db object from database.connection
        
        # Clean up - drop all test data
        try:
            with db.get_connection() as conn:
                cur = conn.cursor()
                # Clean up test data but keep schema
                tables = [
                    'processing_queue',
                    'resampling_cache', 
                    'raster_tiles',
                    'grid_cells',
                    'raster_sources',
                    'experiments'
                ]
                for table in tables:
                    cur.execute(f"TRUNCATE TABLE {table} CASCADE")
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to clean up test data: {e}")
            
    except Exception as e:
        pytest.skip(f"Could not connect to test database: {e}")

@pytest.fixture
def test_db_connection(test_db):
    """Get a test database connection context manager."""
    return test_db.get_connection()

class RasterTestHelper:
    """Helper class for creating test rasters."""
    
    @staticmethod
    def create_test_raster(
        output_path: Path,
        width: int = 100,
        height: int = 100,
        bounds: Tuple[float, float, float, float] = (-10, 40, 10, 60),
        pixel_size: float = 0.0166667,  # ~1.85km at equator
        data_type: int = gdal.GDT_Int32,
        nodata_value: Optional[float] = 0,
        pattern: str = "gradient",
        band_count: int = 1
    ) -> Path:
        """Create a test GeoTIFF raster."""
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(
            str(output_path),
            width,
            height,
            band_count,
            data_type
        )
        
        # Set projection (WGS84)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        dataset.SetProjection(srs.ExportToWkt())
        
        # Set geotransform
        west, south, east, north = bounds
        pixel_width = (east - west) / width
        pixel_height = (north - south) / height
        
        geotransform = [west, pixel_width, 0, north, 0, -pixel_height]
        dataset.SetGeoTransform(geotransform)
        
        # Generate data based on pattern
        for band_idx in range(1, band_count + 1):
            data = RasterTestHelper._generate_pattern(width, height, pattern, band_idx)
            
            # Add some nodata values
            if nodata_value is not None:
                mask = np.random.random((height, width)) < 0.05
                data[mask] = nodata_value
            
            band = dataset.GetRasterBand(band_idx)
            band.SetNoDataValue(nodata_value)
            band.WriteArray(data)
            band.ComputeStatistics(False)
        
        dataset.FlushCache()
        dataset = None
        
        return output_path
    
    @staticmethod
    def _generate_pattern(width: int, height: int, pattern: str, band: int = 1) -> np.ndarray:
        """Generate different data patterns for testing."""
        if pattern == "gradient":
            # Linear gradient from NW to SE
            x = np.linspace(0, 100, width)
            y = np.linspace(0, 100, height)
            xx, yy = np.meshgrid(x, y)
            data = (xx + yy).astype(np.int32) * band
            
        elif pattern == "hotspots":
            # Random hotspots of species richness
            data = np.ones((height, width), dtype=np.int32) * 10
            # Add hotspots
            np.random.seed(42)  # For reproducibility
            for _ in range(5):
                cx, cy = np.random.randint(10, width-10), np.random.randint(10, height-10)
                y, x = np.ogrid[:height, :width]
                mask = (x - cx)**2 + (y - cy)**2 <= 10**2
                data[mask] = np.random.randint(100, 500)
                
        elif pattern == "uniform":
            # Uniform value
            data = np.full((height, width), 42 * band, dtype=np.int32)
            
        elif pattern == "random":
            # Random values (species counts)
            np.random.seed(42)  # For reproducibility
            data = np.random.randint(0, 1000, size=(height, width), dtype=np.int32)
            
        elif pattern == "edge_effects":
            # High values at edges, low in center
            data = np.ones((height, width), dtype=np.int32) * 50
            data[10:-10, 10:-10] = 10
            
        else:
            data = np.zeros((height, width), dtype=np.int32)
            
        return data

@pytest.fixture
def raster_helper():
    """Provide raster test helper."""
    return RasterTestHelper()

@pytest.fixture
def sample_raster(test_data_dir, raster_helper):
    """Create a sample test raster."""
    raster_path = test_data_dir / "test_raster.tif"
    return raster_helper.create_test_raster(
        raster_path,
        width=200,
        height=200,
        pattern="hotspots"
    )

@pytest.fixture
def large_raster(test_data_dir, raster_helper):
    """Create a larger test raster for memory tests (but still small for fast testing)."""
    raster_path = test_data_dir / "large_raster.tif"
    return raster_helper.create_test_raster(
        raster_path,
        width=100,  # Reduced from 1000 to 100
        height=100,  # Reduced from 1000 to 100
        pattern="gradient"
    )

# Mock fixtures removed - using real database connections