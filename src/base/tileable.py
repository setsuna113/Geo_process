"""Mixin for tile-based processing support."""

from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Optional, List, Any, Dict, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TileSpec:
    """Specification for a tile including bounds and overlap."""
    
    def __init__(self,
                 tile_id: str,
                 bounds: Tuple[int, int, int, int],  # (col_off, row_off, width, height)
                 overlap: int = 0,
                 priority: int = 0):
        """
        Initialize tile specification.
        
        Args:
            tile_id: Unique identifier for the tile
            bounds: Tile bounds in pixels (col_off, row_off, width, height)
            overlap: Overlap with neighboring tiles in pixels
            priority: Processing priority (higher = process first)
        """
        self.tile_id = tile_id
        self.bounds = bounds
        self.overlap = overlap
        self.priority = priority
        
    @property
    def col_off(self) -> int:
        """Column offset in pixels."""
        return self.bounds[0]
        
    @property
    def row_off(self) -> int:
        """Row offset in pixels."""
        return self.bounds[1]
        
    @property
    def width(self) -> int:
        """Tile width in pixels."""
        return self.bounds[2]
        
    @property
    def height(self) -> int:
        """Tile height in pixels."""
        return self.bounds[3]
        
    @property
    def area_pixels(self) -> int:
        """Tile area in pixels."""
        return self.width * self.height
        
    def expanded_bounds(self) -> Tuple[int, int, int, int]:
        """Get bounds expanded by overlap."""
        return (
            max(0, self.col_off - self.overlap),
            max(0, self.row_off - self.overlap),
            self.width + 2 * self.overlap,
            self.height + 2 * self.overlap
        )


class Tileable(ABC):
    """
    Mixin for tile-based processing.
    
    Provides tile generation, iteration, and overlap handling.
    """
    
    def __init__(self):
        self._tile_size = 512
        self._overlap = 0
        self._tile_cache: Dict[str, Any] = {}
        self._tile_stats = {
            'generated': 0,
            'processed': 0,
            'cached': 0,
            'errors': 0
        }
        
    @abstractmethod
    def get_dimensions(self) -> Tuple[int, int]:
        """Get data dimensions (width, height) in pixels."""
        pass
        
    def set_tile_size(self, tile_size: int) -> None:
        """Set default tile size."""
        if tile_size <= 0:
            raise ValueError("Tile size must be positive")
        self._tile_size = tile_size
        
    def set_overlap(self, overlap: int) -> None:
        """Set default overlap between tiles."""
        if overlap < 0:
            raise ValueError("Overlap cannot be negative")
        self._overlap = overlap
        
    def get_tile_size(self) -> int:
        """Get current tile size."""
        return self._tile_size
        
    def get_overlap(self) -> int:
        """Get current overlap."""
        return self._overlap
        
    def generate_tile_specs(self,
                           tile_size: Optional[int] = None,
                           overlap: Optional[int] = None,
                           bounds: Optional[Tuple[int, int, int, int]] = None) -> List[TileSpec]:
        """
        Generate tile specifications for processing.
        
        Args:
            tile_size: Size of tiles in pixels
            overlap: Overlap between tiles
            bounds: Bounding box to tile (col_off, row_off, width, height)
            
        Returns:
            List of TileSpec objects
        """
        tile_size = tile_size or self._tile_size
        overlap = overlap or self._overlap
        
        # Get data dimensions
        width, height = self.get_dimensions()
        
        # Use provided bounds or full extent
        if bounds is None:
            bounds = (0, 0, width, height)
            
        col_start, row_start, region_width, region_height = bounds
        col_end = col_start + region_width
        row_end = row_start + region_height
        
        tiles = []
        tile_row = 0
        
        for row in range(row_start, row_end, tile_size):
            tile_col = 0
            for col in range(col_start, col_end, tile_size):
                # Calculate actual tile dimensions
                tile_width = min(tile_size, col_end - col)
                tile_height = min(tile_size, row_end - row)
                
                tile_id = f"tile_{tile_row}_{tile_col}"
                
                tile_spec = TileSpec(
                    tile_id=tile_id,
                    bounds=(col, row, tile_width, tile_height),
                    overlap=overlap
                )
                
                tiles.append(tile_spec)
                tile_col += 1
                
            tile_row += 1
            
        self._tile_stats['generated'] = len(tiles)
        logger.debug(f"Generated {len(tiles)} tile specifications")
        
        return tiles
        
    def get_tile_iterator(self,
                         tile_size: Optional[int] = None,
                         overlap: Optional[int] = None,
                         bounds: Optional[Tuple[int, int, int, int]] = None,
                         prioritize: bool = False) -> Iterator[TileSpec]:
        """
        Get iterator over tile specifications.
        
        Args:
            tile_size: Size of tiles in pixels
            overlap: Overlap between tiles
            bounds: Bounding box to iterate over
            prioritize: Whether to sort by priority
            
        Yields:
            TileSpec objects
        """
        tiles = self.generate_tile_specs(tile_size, overlap, bounds)
        
        if prioritize:
            tiles.sort(key=lambda t: t.priority, reverse=True)
            
        for tile in tiles:
            yield tile
            
    def calculate_overlap_region(self,
                               tile1: TileSpec,
                               tile2: TileSpec) -> Optional[Tuple[int, int, int, int]]:
        """
        Calculate overlap region between two tiles.
        
        Args:
            tile1: First tile specification
            tile2: Second tile specification
            
        Returns:
            Overlap bounds (col_off, row_off, width, height) or None
        """
        # Get tile bounds
        t1_left = tile1.col_off
        t1_top = tile1.row_off
        t1_right = t1_left + tile1.width
        t1_bottom = t1_top + tile1.height
        
        t2_left = tile2.col_off
        t2_top = tile2.row_off
        t2_right = t2_left + tile2.width
        t2_bottom = t2_top + tile2.height
        
        # Calculate intersection
        left = max(t1_left, t2_left)
        top = max(t1_top, t2_top)
        right = min(t1_right, t2_right)
        bottom = min(t1_bottom, t2_bottom)
        
        # Check if there's actual overlap
        if left >= right or top >= bottom:
            return None
            
        return (left, top, right - left, bottom - top)
        
    def merge_overlapping_tiles(self,
                              tiles: List[Tuple[TileSpec, Any]],
                              merge_function: Optional[Any] = None) -> Any:
        """
        Merge overlapping tiles using specified function.
        
        Args:
            tiles: List of (TileSpec, data) tuples
            merge_function: Function to merge overlapping data
            
        Returns:
            Merged result
        """
        if not tiles:
            return None
            
        if len(tiles) == 1:
            return tiles[0][1]
            
        # Default merge function - override for specific data types
        if merge_function is None:
            merge_function = self._default_merge_function
            
        # Sort tiles by priority for merging order
        sorted_tiles = sorted(tiles, key=lambda x: x[0].priority, reverse=True)
        
        result = sorted_tiles[0][1]
        for tile_spec, tile_data in sorted_tiles[1:]:
            result = merge_function(result, tile_data, tile_spec)
            
        return result
        
    def _default_merge_function(self, base_data: Any, overlay_data: Any, overlay_spec: TileSpec) -> Any:
        """Default merge function - override in subclasses."""
        # Simple replacement - override for specific merge logic
        return overlay_data
        
    def get_tile_neighbors(self, tile_spec: TileSpec, all_tiles: List[TileSpec]) -> List[TileSpec]:
        """
        Get neighboring tiles for a given tile.
        
        Args:
            tile_spec: Target tile
            all_tiles: List of all available tiles
            
        Returns:
            List of neighboring tiles
        """
        neighbors = []
        
        for other_tile in all_tiles:
            if other_tile.tile_id == tile_spec.tile_id:
                continue
                
            # Check if tiles are adjacent or overlapping
            overlap = self.calculate_overlap_region(tile_spec, other_tile)
            if overlap is not None:
                neighbors.append(other_tile)
                continue
                
            # Check for adjacency (touching edges)
            t1_right = tile_spec.col_off + tile_spec.width
            t1_bottom = tile_spec.row_off + tile_spec.height
            t2_right = other_tile.col_off + other_tile.width
            t2_bottom = other_tile.row_off + other_tile.height
            
            # Check horizontal adjacency
            horizontal_adjacent = (
                (tile_spec.col_off == t2_right or t1_right == other_tile.col_off) and
                not (tile_spec.row_off >= t2_bottom or t1_bottom <= other_tile.row_off)
            )
            
            # Check vertical adjacency
            vertical_adjacent = (
                (tile_spec.row_off == t2_bottom or t1_bottom == other_tile.row_off) and
                not (tile_spec.col_off >= t2_right or t1_right <= other_tile.col_off)
            )
            
            if horizontal_adjacent or vertical_adjacent:
                neighbors.append(other_tile)
                
        return neighbors
        
    def estimate_tile_memory_usage(self,
                                 tile_spec: TileSpec,
                                 dtype: np.dtype,
                                 bands: int = 1) -> float:
        """
        Estimate memory usage for a tile in MB.
        
        Args:
            tile_spec: Tile specification
            dtype: Data type
            bands: Number of bands
            
        Returns:
            Estimated memory usage in MB
        """
        expanded_bounds = tile_spec.expanded_bounds()
        width, height = expanded_bounds[2], expanded_bounds[3]
        
        total_pixels = width * height * bands
        bytes_per_pixel = np.dtype(dtype).itemsize
        total_bytes = total_pixels * bytes_per_pixel
        
        return total_bytes / (1024 * 1024)
        
    def get_optimal_tile_count(self,
                             available_memory_mb: float,
                             dtype: np.dtype,
                             bands: int = 1) -> int:
        """
        Calculate optimal number of tiles to process simultaneously.
        
        Args:
            available_memory_mb: Available memory in MB
            dtype: Data type
            bands: Number of bands
            
        Returns:
            Optimal number of concurrent tiles
        """
        # Estimate memory per tile
        dummy_tile = TileSpec("dummy", (0, 0, self._tile_size, self._tile_size), self._overlap)
        memory_per_tile = self.estimate_tile_memory_usage(dummy_tile, dtype, bands)
        
        if memory_per_tile <= 0:
            return 1
            
        # Leave some memory for overhead
        usable_memory = available_memory_mb * 0.8
        max_tiles = int(usable_memory / memory_per_tile)
        
        return max(1, max_tiles)
        
    def get_tile_stats(self) -> Dict[str, int]:
        """Get tile processing statistics."""
        return self._tile_stats.copy()
        
    def reset_tile_stats(self) -> None:
        """Reset tile processing statistics."""
        self._tile_stats = {
            'generated': 0,
            'processed': 0,
            'cached': 0,
            'errors': 0
        }
