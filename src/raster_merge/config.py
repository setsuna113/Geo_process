"""
Configuration for raster merging - minimal and explicit.

Following FarmVibes pattern but simpler - no inheritance hell.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Literal, Union, Dict, Any
from rasterio.enums import Resampling


# Kill the DatasetConfig - just use a tuple if you need per-file config
# (path, resampling) is clearer than a whole class


@dataclass
class RasterConfig:
    """
    Configuration for raster processing.
    
    Simple and explicit. No type confusion.
    """
    # Input/Output paths - just paths, nothing fancy
    input_paths: List[Path]
    output_dir: Path
    output_name: str = "merged_raster"
    
    # Per-file resampling if needed (path -> resampling method)
    per_file_resampling: Optional[Dict[Path, Resampling]] = None
    
    # Processing parameters
    resolution_strategy: Literal["equal", "average", "lowest", "highest"] = "average"
    
    # Target resolution - new fields as requested
    target_resolution: Optional[Union[float, Literal["coarse"]]] = None  # None = compute, "coarse" = use coarsest, or specific value
    
    overlap_method: Literal["first", "last", "min", "max", "mean"] = "first"
    resampling: Resampling = Resampling.bilinear  # Global default
    band_merge_strategy: Literal["flat", "auto"] = "auto"  # flat: force single band, auto: smart band detection
    
    # Memory optimization
    chunk_size: int = 2048  # Process in 2048x2048 chunks like FarmVibes
    use_dask: bool = True   # Enable lazy loading for large files
    n_workers: int = 4      # Parallel workers for processing
    
    # Output formats
    export_geotiff: bool = True
    export_arrow: bool = True
    compress: Literal["zstd", "deflate", "lzw", None] = "zstd"  # FarmVibes uses ZSTD
    compression_level: int = 9
    
    # COG (Cloud-Optimized GeoTIFF) settings
    build_overviews: bool = True
    overview_levels: List[int] = None  # Default: [2, 4, 8, 16, 32]
    blocksize: int = 512
    
    # Data handling
    nodata_value: Optional[float] = None  # Auto-detect if None
    dtype: Optional[str] = None  # Preserve input dtype if None
    
    def __post_init__(self):
        """Validate and set defaults."""
        # Convert to Path objects and validate
        self.input_paths = [Path(p) for p in self.input_paths]
        for path in self.input_paths:
            if not path.exists():
                raise FileNotFoundError(f"Input raster not found: {path}")
        
        self.output_dir = Path(self.output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set default overview levels
        if self.overview_levels is None:
            self.overview_levels = [2, 4, 8, 16, 32]
        
        # Validate target resolution
        if self.target_resolution is not None:
            if isinstance(self.target_resolution, str) and self.target_resolution != "coarse":
                raise ValueError(f"Invalid target_resolution string: {self.target_resolution}. Use 'coarse' or a numeric value.")
            elif isinstance(self.target_resolution, (int, float)) and self.target_resolution <= 0:
                raise ValueError(f"Target resolution must be positive: {self.target_resolution}")
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "RasterConfig":
        """Create config from dictionary."""
        # Handle resampling enum
        if "resampling" in config_dict and isinstance(config_dict["resampling"], str):
            config_dict["resampling"] = getattr(Resampling, config_dict["resampling"])
        
        # Handle per-file resampling if present
        if "per_file_resampling" in config_dict:
            per_file = {}
            for path_str, method in config_dict["per_file_resampling"].items():
                path = Path(path_str)
                if isinstance(method, str):
                    method = getattr(Resampling, method)
                per_file[path] = method
            config_dict["per_file_resampling"] = per_file
        
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """Export config to dictionary."""
        result = {
            "input_paths": [str(p) for p in self.input_paths],
            "output_dir": str(self.output_dir),
            "output_name": self.output_name,
            "resolution_strategy": self.resolution_strategy,
            "target_resolution": self.target_resolution,
            "overlap_method": self.overlap_method,
            "resampling": self.resampling.name,
            "band_merge_strategy": self.band_merge_strategy,
            "chunk_size": self.chunk_size,
            "use_dask": self.use_dask,
            "n_workers": self.n_workers,
            "export_geotiff": self.export_geotiff,
            "export_arrow": self.export_arrow,
            "compress": self.compress,
            "compression_level": self.compression_level,
            "build_overviews": self.build_overviews,
            "overview_levels": self.overview_levels,
            "blocksize": self.blocksize,
            "nodata_value": self.nodata_value,
            "dtype": self.dtype
        }
        
        # Add per-file resampling if present
        if self.per_file_resampling:
            result["per_file_resampling"] = {
                str(path): method.name 
                for path, method in self.per_file_resampling.items()
            }
        
        return result
    
    def get_input_paths(self) -> List[Path]:
        """Get input paths - simple accessor."""
        return self.input_paths
    
    def get_resampling_for_path(self, path: Path) -> Resampling:
        """Get the resampling method for a specific path."""
        if self.per_file_resampling and path in self.per_file_resampling:
            return self.per_file_resampling[path]
        return self.resampling