#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, 'src')
sys.modules['pytest'] = type(sys)('pytest')  # Force test mode

import rasterio
from src.database.connection import DatabaseManager
from src.raster_data.loaders.metadata_extractor import RasterMetadataExtractor

# Create test data
test_data_dir = Path("data/test_richness_small")
test_data_dir.mkdir(exist_ok=True)

source_dir = Path("data/richness_maps")
plants_src = source_dir / "daru-plants-richness.tif"
plants_test = test_data_dir / "daru-plants-richness.tif"

if not plants_test.exists():
    with rasterio.open(plants_src) as src:
        width, height = src.width, src.height
        center_x, center_y = width // 2, height // 2
        
        window = rasterio.windows.Window(center_x - 50, center_y - 50, 100, 100)
        data = src.read(1, window=window)
        transform = src.window_transform(window)
        
        with rasterio.open(
            plants_test, 'w', driver='GTiff', height=100, width=100,
            count=1, dtype=data.dtype, crs=src.crs, transform=transform, nodata=src.nodata
        ) as dst:
            dst.write(data, 1)

print("=== Direct File Check ===")
with rasterio.open(plants_test) as src:
    bounds = src.bounds
    print(f"File bounds: {bounds}")
    print(f"  minx: {bounds.left}")
    print(f"  miny: {bounds.bottom}")  
    print(f"  maxx: {bounds.right}")
    print(f"  maxy: {bounds.top}")

print(f"\n=== Metadata Extractor Check ===")
db = DatabaseManager()
extractor = RasterMetadataExtractor(db)

try:
    full_metadata = extractor.extract_full_metadata(plants_test)
    extent = full_metadata['spatial_info']['extent']
    print(f"Metadata extent: {extent}")
    print(f"Extent type: {type(extent)}")
    
    if isinstance(extent, dict):
        print("Extent keys and values:")
        for key, value in extent.items():
            print(f"  {key}: {value}")
    
    extent_values = tuple(extent.values())
    print(f"extent.values() tuple: {extent_values}")
    
    # This is what gets stored as bounds
    print(f"This becomes catalog bounds: {extent_values}")
    
except Exception as e:
    print(f"❌ Metadata extraction failed: {e}")
    import traceback
    traceback.print_exc()