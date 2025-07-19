#!/usr/bin/env python3
"""
Example usage of the enhanced registry system.

This script demonstrates how to use the new registry features for 
data sources, resamplers, and tile processors.
"""

from src.core.registry import (
    component_registry, 
    raster_source, 
    resampler, 
    tile_processor,
    MemoryUsage,
    ComponentMetadata
)
from typing import List, Dict, Any, Optional
import numpy as np


# Example Raster Source Classes
@raster_source('geotiff', 'tif', 'tiff', 
               memory_usage=MemoryUsage.LOW,
               supports_lazy_loading=True,
               description="GeoTIFF file reader")
class GeoTIFFSource:
    """GeoTIFF raster data source."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def read_data(self):
        """Read raster data."""
        pass
    
    def get_metadata(self):
        """Get raster metadata."""
        pass


@raster_source('netcdf', 'nc', 
               memory_usage=MemoryUsage.MEDIUM,
               supports_lazy_loading=True,
               supports_streaming=True,
               description="NetCDF file reader")
class NetCDFSource:
    """NetCDF raster data source."""
    
    def __init__(self, file_path: str, variable: Optional[str] = None):
        self.file_path = file_path
        self.variable = variable
    
    def read_data(self):
        """Read raster data."""
        pass


@raster_source('hdf5', 'h5', 'hdf', 
               memory_usage=MemoryUsage.HIGH,
               supports_memory_mapping=True,
               description="HDF5 file reader")
class HDF5Source:
    """HDF5 raster data source."""
    
    def __init__(self, file_path: str, dataset: Optional[str] = None):
        self.file_path = file_path
        self.dataset = dataset
    
    def read_data(self):
        """Read raster data."""
        pass


# Example Resampler Classes
@resampler('Int32', 'UInt32', 'Int16', 'UInt16',
           preferred_types=['Int32', 'UInt32'],
           memory_usage=MemoryUsage.LOW,
           description="Nearest neighbor resampling for integer data")
class NearestNeighborResampler:
    """Fast nearest neighbor resampling for integer data types."""
    
    def get_supported_methods(self) -> List[str]:
        return ['nearest']
    
    def resample(self, data: np.ndarray, target_shape: tuple, method: str = 'nearest'):
        """Perform nearest neighbor resampling."""
        pass


@resampler('Float32', 'Float64',
           preferred_types=['Float32', 'Float64'],
           memory_usage=MemoryUsage.MEDIUM,
           description="Bilinear/bicubic resampling for floating point data")
class BilinearResampler:
    """High-quality resampling for floating point data."""
    
    def get_supported_methods(self) -> List[str]:
        return ['bilinear', 'bicubic']
    
    def resample(self, data: np.ndarray, target_shape: tuple, method: str = 'bilinear'):
        """Perform bilinear or bicubic resampling."""
        pass


@resampler('Int32', 'UInt32', 'Int16', 'UInt16', 'Float32', 'Float64',
           memory_usage=MemoryUsage.HIGH,
           cpu_intensive=True,
           description="High-quality Lanczos resampling")
class LanczosResampler:
    """High-quality Lanczos resampling for all data types."""
    
    def get_supported_methods(self) -> List[str]:
        return ['lanczos', 'lanczos3']
    
    def resample(self, data: np.ndarray, target_shape: tuple, method: str = 'lanczos'):
        """Perform Lanczos resampling."""
        pass


# Example Tile Processor Classes
@tile_processor(supports_streaming=False,
                supports_memory_mapping=False,
                memory_usage=MemoryUsage.LOW,
                optimal_tile_size=512,
                description="In-memory tile processing")
class InMemoryTileProcessor:
    """Fast in-memory tile processing."""
    
    def get_processing_strategy(self) -> List[str]:
        return ['in-memory']
    
    def process_tile(self, tile_data: np.ndarray) -> np.ndarray:
        """Process a single tile in memory."""
        # Example implementation - just return the input for now
        return tile_data


@tile_processor(supports_streaming=True,
                supports_memory_mapping=True,
                memory_usage=MemoryUsage.MEDIUM,
                optimal_tile_size=1024,
                description="Memory-mapped tile processing")
class MemoryMappedTileProcessor:
    """Memory-efficient tile processing using memory mapping."""
    
    def get_processing_strategy(self) -> List[str]:
        return ['memory-mapped', 'streaming']
    
    def process_tile(self, tile_data: np.ndarray) -> np.ndarray:
        """Process a tile using memory mapping."""
        # Example implementation - just return the input for now
        return tile_data


@tile_processor(supports_streaming=True,
                memory_usage=MemoryUsage.LOW,
                optimal_tile_size=2048,
                description="Streaming tile processor for large datasets")
class StreamingTileProcessor:
    """Streaming tile processor for very large datasets."""
    
    def get_processing_strategy(self) -> List[str]:
        return ['streaming']
    
    def process_tile(self, tile_data: np.ndarray) -> np.ndarray:
        """Process a tile in streaming mode."""
        # Example implementation - just return the input for now
        return tile_data


def demonstrate_registry_features():
    """Demonstrate the enhanced registry features."""
    
    print("=== Enhanced Registry System Demo ===\n")
    
    # 1. Find data sources by format
    print("1. Finding data sources by format:")
    geotiff_source = component_registry.find_data_source_for_format('geotiff')
    if geotiff_source:
        print(f"   GeoTIFF source: {geotiff_source.name}")
        print(f"   Description: {geotiff_source.description}")
        print(f"   Memory usage: {geotiff_source.memory_usage.value}")
        print(f"   Supports lazy loading: {geotiff_source.supports_lazy_loading}")
    
    netcdf_source = component_registry.find_data_source_for_format('netcdf')
    if netcdf_source:
        print(f"   NetCDF source: {netcdf_source.name}")
        print(f"   Supports streaming: {netcdf_source.supports_streaming}")
    print()
    
    # 2. Find optimal resampler for data type
    print("2. Finding optimal resamplers by data type:")
    int32_resampler = component_registry.find_optimal_resampler('Int32', 'nearest')
    if int32_resampler:
        print(f"   Int32 nearest: {int32_resampler.name}")
        print(f"   Description: {int32_resampler.description}")
    
    float32_resampler = component_registry.find_optimal_resampler('Float32', 'bilinear')
    if float32_resampler:
        print(f"   Float32 bilinear: {float32_resampler.name}")
        print(f"   Memory usage: {float32_resampler.memory_usage.value}")
    print()
    
    # 3. Find tile processors by strategy
    print("3. Finding tile processors by strategy:")
    streaming_processors = component_registry.find_tile_processor_for_strategy('streaming')
    print(f"   Streaming processors found: {len(streaming_processors)}")
    for processor in streaming_processors:
        print(f"   - {processor.name}: {processor.description}")
        print(f"     Optimal tile size: {processor.optimal_tile_size}")
    print()
    
    # 4. Get format compatibility matrix
    print("4. Format compatibility matrix:")
    matrix = component_registry.get_format_compatibility_matrix()
    for registry_name, formats in matrix.items():
        if formats['input'] or formats['supported']:
            print(f"   {registry_name}:")
            if formats['input']:
                print(f"     Input formats: {', '.join(formats['input'])}")
            if formats['supported']:
                print(f"     Supported formats: {', '.join(formats['supported'])}")
    print()
    
    # 5. Find components by capability
    print("5. Finding components by capability:")
    lazy_loading_components = component_registry.raster_sources.find_by_capability(
        supports_lazy_loading=True
    )
    print(f"   Components with lazy loading: {len(lazy_loading_components)}")
    for comp in lazy_loading_components:
        print(f"   - {comp.name}")
    
    memory_mapped_components = component_registry.tile_processors.find_by_capability(
        supports_memory_mapping=True
    )
    print(f"   Components with memory mapping: {len(memory_mapped_components)}")
    for comp in memory_mapped_components:
        print(f"   - {comp.name}")
    print()


if __name__ == "__main__":
    demonstrate_registry_features()
