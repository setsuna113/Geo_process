# tests/fixtures/data_generator.py
import numpy as np
from pathlib import Path
import tempfile
from osgeo import gdal, osr
import geopandas as gpd
from shapely.geometry import box
from typing import Tuple, Optional, Dict, Any
import psycopg2
from contextlib import contextmanager

class TestDataGenerator:
    """Generate synthetic test data for integration tests."""
    
    def __init__(self, temp_dir: Optional[Path] = None):
        self.temp_dir = temp_dir or Path(tempfile.mkdtemp())
        self.temp_dir.mkdir(exist_ok=True)
        
    def create_test_raster(
        self,
        width: int = 100,
        height: int = 100,
        bounds: Tuple[float, float, float, float] = (-10, 40, 10, 60),
        pattern: str = "gradient",
        data_type: gdal.GDT = gdal.GDT_Int32,
        nodata_value: float = 0
    ) -> Path:
        """Create a synthetic raster with known patterns."""
        output_path = self.temp_dir / f"test_raster_{pattern}_{width}x{height}.tif"
        
        # Create the raster
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(
            str(output_path),
            width,
            height,
            1,
            data_type
        )
        
        # Set projection and geotransform
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        dataset.SetProjection(srs.ExportToWkt())
        
        # Calculate pixel size
        pixel_width = (bounds[2] - bounds[0]) / width
        pixel_height = (bounds[3] - bounds[1]) / height
        
        geotransform = [
            bounds[0],    # top left x
            pixel_width,  # pixel width
            0,           # rotation
            bounds[3],    # top left y
            0,           # rotation
            -pixel_height # pixel height (negative)
        ]
        dataset.SetGeoTransform(geotransform)
        
        # Generate pattern
        if pattern == "gradient":
            data = np.arange(width * height, dtype=np.int32).reshape(height, width)
        elif pattern == "checkerboard":
            data = np.zeros((height, width), dtype=np.int32)
            for i in range(height):
                for j in range(width):
                    if (i // 10 + j // 10) % 2:
                        data[i, j] = 100
        elif pattern == "hotspots":
            data = np.ones((height, width), dtype=np.int32) * 10
            # Add some hotspots
            for _ in range(5):
                cx, cy = np.random.randint(20, width-20), np.random.randint(20, height-20)
                y, x = np.ogrid[:height, :width]
                mask = (x - cx)**2 + (y - cy)**2 <= 15**2
                data[mask] = 200
        elif pattern == "nodata_edges":
            data = np.ones((height, width), dtype=np.int32) * 50
            data[:10, :] = nodata_value
            data[-10:, :] = nodata_value
            data[:, :10] = nodata_value
            data[:, -10:] = nodata_value
        else:
            data = np.random.randint(1, 255, size=(height, width), dtype=np.int32)
            
        # Add some nodata values
        if pattern != "nodata_edges":
            mask = np.random.random((height, width)) < 0.1
            data[mask] = nodata_value
        
        # Write data
        band = dataset.GetRasterBand(1)
        band.SetNoDataValue(nodata_value)
        band.WriteArray(data)
        band.ComputeStatistics(False)
        
        dataset.FlushCache()
        dataset = None
        
        return output_path
    
    def create_test_grid(
        self,
        grid_type: str = "cubic",
        resolution: float = 10.0,  # km
        bounds: Tuple[float, float, float, float] = (-10, 40, 10, 60),
        epsg: int = 4326
    ) -> gpd.GeoDataFrame:
        """Create a test grid."""
        from src.grid_systems.grid_factory import GridFactory
        
        # Create grid using your factory
        factory = GridFactory()
        grid_class = factory.get_grid_class(grid_type)
        
        # Create config for grid
        config = {
            'resolution': resolution,
            'bounds': bounds,
            'epsg': epsg
        }
        
        grid_instance = grid_class(config)
        return grid_instance.generate()
    
    def setup_test_database(self, connection_params: Dict[str, Any]) -> None:
        """Set up test database with sample data."""
        conn = psycopg2.connect(**connection_params)
        cur = conn.cursor()
        
        try:
            # Insert test raster metadata
            cur.execute("""
                INSERT INTO raster_sources 
                (name, file_path, pixel_size_degrees, data_type, nodata_value, 
                 bounds, file_size_mb, active)
                VALUES 
                ('test_plants', '/test/plants.tif', 0.0167, 'Int32', 0, 
                 ST_MakeEnvelope(-180, -90, 180, 90, 4326), 100.5, true),
                ('test_vertebrates', '/test/vertebrates.tif', 0.0167, 'UInt16', 0,
                 ST_MakeEnvelope(-180, -60, 180, 84, 4326), 85.3, true)
                ON CONFLICT (name) DO NOTHING
            """)
            
            # Insert test grids
            cur.execute("""
                INSERT INTO grids 
                (grid_type, resolution_meters, epsg, bounds)
                VALUES 
                ('cubic', 10000, 4326, ST_MakeEnvelope(-10, 40, 10, 60, 4326)),
                ('cubic', 25000, 4326, ST_MakeEnvelope(-10, 40, 10, 60, 4326)),
                ('hexagonal', 10000, 4326, ST_MakeEnvelope(-10, 40, 10, 60, 4326))
                ON CONFLICT DO NOTHING
            """)
            
            conn.commit()
        finally:
            cur.close()
            conn.close()
    
    @contextmanager
    def mock_large_raster(self, width: int = 21600, height: int = 10800):
        """Create a mock large raster using VRT for memory efficiency."""
        vrt_path = self.temp_dir / "large_mock.vrt"
        
        # Create small tile
        tile = self.create_test_raster(100, 100, pattern="gradient")
        
        # Build VRT that pretends to be large
        vrt_content = f"""<VRTDataset rasterXSize="{width}" rasterYSize="{height}">
            <GeoTransform>-180.0, 0.0166667, 0.0, 90.0, 0.0, -0.0166667</GeoTransform>
            <SRS>GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],
                PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]</SRS>
            <VRTRasterBand dataType="Int32" band="1">
                <NoDataValue>0</NoDataValue>
                <SimpleSource>
                    <SourceFilename relativeToVRT="0">{tile}</SourceFilename>
                    <SourceBand>1</SourceBand>
                    <SrcRect xOff="0" yOff="0" xSize="100" ySize="100"/>
                    <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>
                </SimpleSource>
            </VRTRasterBand>
        </VRTDataset>"""
        
        vrt_path.write_text(vrt_content)
        
        try:
            yield vrt_path
        finally:
            vrt_path.unlink(missing_ok=True)