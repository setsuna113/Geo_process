# src/raster/validators/coverage_validator.py
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from shapely.geometry import box, Polygon
import geopandas as gpd

from ..loaders.base_loader import BaseRasterLoader, RasterMetadata
from src.base import BaseGrid

class CoverageValidator:
    """Validate raster coverage against grids and regions."""
    
    def __init__(self, loader: BaseRasterLoader):
        self.loader = loader
        
    def validate_coverage(self, raster_path: Path, grid: BaseGrid) -> Dict[str, Any]:
        """Validate that raster covers the grid adequately."""
        metadata = self.loader.extract_metadata(raster_path)
        
        # Get raster bounds as polygon
        raster_bounds = box(*metadata.bounds)
        
        # Get grid bounds
        grid_bounds = box(*grid.bounds)
        
        # Calculate coverage
        intersection = raster_bounds.intersection(grid_bounds)
        coverage_ratio = intersection.area / grid_bounds.area
        
        # Check resolution compatibility (convert grid resolution from meters to degrees)
        raster_resolution = metadata.resolution_degrees
        grid_resolution_m = grid.resolution
        # Approximate conversion: 1 degree ≈ 111 km at equator
        grid_resolution_deg = grid_resolution_m / 111000.0
        resolution_ratio = raster_resolution / grid_resolution_deg
        
        # Sample coverage check
        coverage_gaps = self._find_coverage_gaps(raster_path, grid, metadata)
        
        validation_result = {
            'coverage_ratio': coverage_ratio,
            'fully_covers': coverage_ratio >= 0.99,
            'resolution_ratio': resolution_ratio,
            'resolution_adequate': resolution_ratio <= 1.0,  # Raster res <= grid res
            'coverage_gaps': coverage_gaps,
            'recommendations': self._generate_recommendations(
                coverage_ratio, resolution_ratio, coverage_gaps
            )
        }
        
        return validation_result
    
    def _find_coverage_gaps(self, raster_path: Path, grid: BaseGrid, 
                           metadata: RasterMetadata, sample_rate: float = 0.1) -> List[Dict]:
        """Find areas where raster has no data within grid bounds."""
        gaps = []
        
        # Sample grid cells
        all_cells = grid.get_cells()
        sample_size = max(1, int(len(all_cells) * sample_rate))
        
        # Simple random sampling of cells
        import random
        sampled_cells = random.sample(all_cells, min(sample_size, len(all_cells)))
        
        with self.loader.open_lazy(raster_path) as reader:
            for cell in sampled_cells:
                # Check center point
                center = cell.centroid
                value = reader.read_point(center.x, center.y)
                
                if value is None or value == metadata.nodata_value:
                    gaps.append({
                        'cell_id': cell.cell_id,
                        'location': (center.x, center.y),
                        'type': 'nodata'
                    })
        
        return gaps
    
    def _generate_recommendations(self, coverage_ratio: float, 
                                resolution_ratio: float, 
                                gaps: List[Dict]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if coverage_ratio < 0.99:
            recommendations.append(
                f"Raster covers only {coverage_ratio:.1%} of grid area. "
                "Consider using a larger raster or restricting grid bounds."
            )
        
        if resolution_ratio > 1.0:
            recommendations.append(
                f"Raster resolution is {resolution_ratio:.1f}x coarser than grid. "
                "Consider using a higher resolution raster or coarser grid."
            )
        
        if len(gaps) > 0:
            gap_percent = len(gaps) / len(gaps) * 100  # This is from sample
            recommendations.append(
                f"Found {len(gaps)} cells with no data in sample. "
                "Consider filling gaps or excluding these areas."
            )
        
        if not recommendations:
            recommendations.append("Raster coverage is adequate for processing.")
        
        return recommendations
    
    def validate_multiple_rasters(self, raster_paths: List[Path], 
                                 check_consistency: bool = True) -> Dict[str, Any]:
        """Validate multiple rasters for consistency."""
        results = {}
        metadata_list = []
        
        for path in raster_paths:
            metadata = self.loader.extract_metadata(path)
            metadata_list.append(metadata)
            results[path.name] = metadata
        
        if check_consistency and len(metadata_list) > 1:
            # Check resolution consistency
            resolutions = [m.resolution_degrees for m in metadata_list]
            resolution_consistent = all(abs(r - resolutions[0]) < 1e-6 for r in resolutions)
            
            # Check CRS consistency
            crs_list = [m.crs for m in metadata_list]
            crs_consistent = all(c == crs_list[0] for c in crs_list)
            
            # Check bounds overlap
            bounds_list = [box(*m.bounds) for m in metadata_list]
            total_bounds = bounds_list[0]
            for b in bounds_list[1:]:
                total_bounds = total_bounds.union(b)
            
            overlap_area = bounds_list[0].area
            for b in bounds_list[1:]:
                overlap_area = bounds_list[0].intersection(b).area
            
            results['consistency'] = {
                'resolution_consistent': resolution_consistent,
                'crs_consistent': crs_consistent,
                'spatial_overlap': overlap_area > 0,
                'resolutions': {p.name: r for p, r in zip(raster_paths, resolutions)},
                'total_coverage': total_bounds.bounds
            }
        
        return results
