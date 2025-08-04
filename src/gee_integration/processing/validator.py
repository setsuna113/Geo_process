"""GEE Alignment and Validation Module"""

from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
import numpy as np

from src.infrastructure.logging import get_logger

if TYPE_CHECKING:
    import ee

logger = get_logger(__name__)


class GEEAlignmentValidator:
    """Validate alignment and bounds before processing"""
    
    def __init__(self, tolerance: float = 0.0001):
        """
        Initialize validator.
        
        Args:
            tolerance: Tolerance for bounds comparison (in degrees)
        """
        self.tolerance = tolerance
        self.logger = logger
    
    def validate_images(self, images: List['ee.Image'], 
                       configs: List[Dict]) -> Dict[str, any]:
        """
        Validate that images can be aligned and merged.
        
        Args:
            images: List of Earth Engine images
            configs: List of dataset configurations
            
        Returns:
            Validation result with issues and metadata
        """
        import ee
        self.logger.info(f"Validating {len(images)} images for alignment...")
        
        issues = []
        metadata = []
        
        # Get info for all images
        for i, (image, config) in enumerate(zip(images, configs)):
            try:
                info = image.getInfo()
                
                # Extract metadata
                img_metadata = {
                    'name': config['name'],
                    'bands': len(info['bands']),
                    'band_names': [b['id'] for b in info['bands']],
                    'crs': info['bands'][0].get('crs', 'unknown'),
                    'crs_transform': info['bands'][0].get('crs_transform'),
                    'dimensions': info['bands'][0].get('dimensions', [0, 0]),
                    'data_type': info['bands'][0].get('data_type', {}).get('precision', 'unknown')
                }
                
                # Get bounds (if available)
                if 'properties' in info and 'system:footprint' in info['properties']:
                    footprint = info['properties']['system:footprint']
                    if footprint and 'coordinates' in footprint:
                        img_metadata['bounds'] = self._extract_bounds(footprint['coordinates'])
                
                metadata.append(img_metadata)
                
            except Exception as e:
                issues.append(f"Failed to get info for {config['name']}: {e}")
        
        # Check compatibility
        if len(metadata) > 1:
            compatibility_issues = self._check_compatibility(metadata)
            issues.extend(compatibility_issues)
        
        # Check for required bands
        for meta in metadata:
            if meta['bands'] == 0:
                issues.append(f"{meta['name']} has no bands")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'metadata': metadata,
            'summary': self._create_summary(metadata)
        }
    
    def _extract_bounds(self, coordinates: List) -> Tuple[float, float, float, float]:
        """Extract bounds from footprint coordinates"""
        # Flatten nested coordinates
        coords = []
        for coord_list in coordinates:
            if isinstance(coord_list[0], list):
                coords.extend(coord_list)
            else:
                coords.append(coord_list)
        
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        
        return (min(lons), min(lats), max(lons), max(lats))
    
    def _check_compatibility(self, metadata_list: List[Dict]) -> List[str]:
        """Check if images are compatible for merging"""
        issues = []
        
        # Check CRS consistency
        crs_set = set(m['crs'] for m in metadata_list)
        if len(crs_set) > 1:
            issues.append(f"Multiple CRS found: {crs_set}")
            self.logger.warning("Images have different CRS - will need reprojection")
        
        # Check transform alignment (pixel grid alignment)
        if all('crs_transform' in m for m in metadata_list):
            transforms = [m['crs_transform'] for m in metadata_list]
            # Check if pixel origins align
            origins = [(t[2], t[5]) for t in transforms if t]  # x_origin, y_origin
            if len(set(origins)) > 1:
                self.logger.warning(f"Different pixel origins detected: {origins}")
                self.logger.info("Images may have sub-pixel misalignment")
        
        # Check bounds overlap (if available)
        bounds_list = [m.get('bounds') for m in metadata_list if 'bounds' in m]
        if len(bounds_list) > 1:
            # Check if all bounds overlap
            intersection = self._get_bounds_intersection(bounds_list)
            if intersection is None:
                issues.append("Images do not overlap")
            else:
                # Calculate overlap percentage
                overlap_areas = []
                for bounds in bounds_list:
                    area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
                    overlap_area = (intersection[2] - intersection[0]) * (intersection[3] - intersection[1])
                    overlap_pct = (overlap_area / area) * 100
                    overlap_areas.append(overlap_pct)
                
                min_overlap = min(overlap_areas)
                if min_overlap < 50:
                    issues.append(f"Low overlap between images: {min_overlap:.1f}%")
        
        return issues
    
    def _get_bounds_intersection(self, bounds_list: List[Tuple]) -> Optional[Tuple]:
        """Get intersection of multiple bounds"""
        if not bounds_list:
            return None
        
        # Start with first bounds
        min_lon = max(b[0] for b in bounds_list)
        min_lat = max(b[1] for b in bounds_list)
        max_lon = min(b[2] for b in bounds_list)
        max_lat = min(b[3] for b in bounds_list)
        
        # Check if valid intersection
        if min_lon >= max_lon or min_lat >= max_lat:
            return None
        
        return (min_lon, min_lat, max_lon, max_lat)
    
    def _create_summary(self, metadata_list: List[Dict]) -> Dict:
        """Create summary of validation results"""
        summary = {
            'total_images': len(metadata_list),
            'crs_list': list(set(m['crs'] for m in metadata_list)),
            'total_bands': sum(m['bands'] for m in metadata_list),
            'data_types': list(set(m['data_type'] for m in metadata_list))
        }
        
        # Add bounds summary if available
        bounds_list = [m.get('bounds') for m in metadata_list if 'bounds' in m]
        if bounds_list:
            # Get overall bounds
            summary['overall_bounds'] = (
                min(b[0] for b in bounds_list),
                min(b[1] for b in bounds_list),
                max(b[2] for b in bounds_list),
                max(b[3] for b in bounds_list)
            )
        
        return summary
    
    def validate_worldclim(self) -> bool:
        """Validate WorldClim data availability"""
        try:
            # Test WorldClim V2 availability
            worldclim = ee.ImageCollection("WORLDCLIM/V2/BIO")
            
            # Check if collection exists and has images
            size = worldclim.size()
            count = size.getInfo()
            
            if count > 0:
                self.logger.info(f"✅ WorldClim V2 available with {count} variables")
                return True
            else:
                self.logger.error("WorldClim V2 collection is empty")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to access WorldClim: {e}")
            return False