# src/processors/data_preparation/raster_alignment.py
"""
Robust raster alignment utilities for spatial data processing.

This module provides comprehensive tools for detecting, analyzing, and correcting
spatial alignment issues between rasters.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import from_bounds
import xarray as xr
import rioxarray
from src.infrastructure.logging import get_logger
from src.infrastructure.logging.decorators import log_operation

logger = get_logger(__name__)

class AlignmentStrategy(Enum):
    """Strategies for handling alignment issues."""
    STRICT = "strict"           # Fail on any misalignment
    REPROJECT = "reproject"     # Reproject to common grid
    INTERSECT = "intersect"     # Use intersection bounds
    SNAP = "snap"              # Snap to reference grid
    
class ResamplingMethod(Enum):
    """Resampling methods for reprojection."""
    NEAREST = Resampling.nearest
    BILINEAR = Resampling.bilinear
    CUBIC = Resampling.cubic
    LANCZOS = Resampling.lanczos

@dataclass
class AlignmentIssue:
    """Detailed description of an alignment issue."""
    type: str
    severity: str
    description: str
    values: Dict[str, Any]
    fixable: bool
    suggested_fix: Optional[str] = None

@dataclass
class AlignmentReport:
    """Comprehensive alignment analysis report."""
    aligned: bool
    issues: List[AlignmentIssue]
    reference_raster: str
    compared_rasters: List[str]
    summary: Dict[str, Any]
    
    def has_issues(self, severity: str = None) -> bool:
        """Check if report has issues of given severity."""
        if severity is None:
            return len(self.issues) > 0
        return any(issue.severity == severity for issue in self.issues)
    
    def get_fixable_issues(self) -> List[AlignmentIssue]:
        """Get list of issues that can be automatically fixed."""
        return [issue for issue in self.issues if issue.fixable]

@dataclass
class GridAlignment:
    """Grid alignment information between datasets."""
    reference_dataset: str
    aligned_dataset: str
    x_shift: float  # Shift in x direction (degrees or meters)
    y_shift: float  # Shift in y direction (degrees or meters)
    requires_shift: bool
    shift_pixels_x: float  # Shift in pixels
    shift_pixels_y: float  # Shift in pixels

@dataclass
class AlignmentConfig:
    """Configuration for alignment checking and fixing."""
    resolution_tolerance: float = 1e-6
    bounds_tolerance: float = 1e-4
    strategy: AlignmentStrategy = AlignmentStrategy.REPROJECT
    resampling_method: ResamplingMethod = ResamplingMethod.NEAREST
    target_resolution: Optional[float] = None
    target_bounds: Optional[Tuple[float, float, float, float]] = None
    target_crs: Optional[str] = None

class RasterAligner:
    """Robust raster alignment and correction utilities."""
    
    def __init__(self, config: AlignmentConfig = None):
        self.config = config or AlignmentConfig()
        
    def analyze_mixed_dataset_alignment(self, dataset_infos: List, reference_idx: int = 0) -> AlignmentReport:
        """
        Analyze alignment of mixed resampled and passthrough datasets.
        
        Args:
            dataset_infos: List of ResampledDatasetInfo objects
            reference_idx: Index of reference dataset
            
        Returns:
            Detailed alignment report
        """
        if len(dataset_infos) < 2:
            return AlignmentReport(
                aligned=True,
                issues=[],
                reference_raster=dataset_infos[0].name if dataset_infos else "",
                compared_rasters=[],
                summary={'message': 'Single or no dataset provided'}
            )
        
        reference_info = dataset_infos[reference_idx]
        compared_infos = [info for i, info in enumerate(dataset_infos) if i != reference_idx]
        
        logger.info(f"Analyzing mixed dataset alignment with reference: {reference_info.name}")
        
        issues = []
        
        # Extract reference metadata
        ref_meta = {
            'crs': reference_info.target_crs,
            'bounds': reference_info.bounds,
            'shape': reference_info.shape,
            'resolution': (reference_info.target_resolution, reference_info.target_resolution),
            'dataset_type': 'passthrough' if reference_info.metadata.get('passthrough') else 'resampled'
        }
        
        # Compare each dataset to reference
        for dataset_info in compared_infos:
            dataset_meta = {
                'crs': dataset_info.target_crs,
                'bounds': dataset_info.bounds,
                'shape': dataset_info.shape,
                'resolution': (dataset_info.target_resolution, dataset_info.target_resolution),
                'dataset_type': 'passthrough' if dataset_info.metadata.get('passthrough') else 'resampled'
            }
            
            dataset_issues = self._compare_dataset_metadata(
                ref_meta, dataset_meta, reference_info.name, dataset_info.name
            )
            issues.extend(dataset_issues)
        
        aligned = len(issues) == 0
        summary = self._generate_mixed_dataset_summary(issues, ref_meta, dataset_infos)
        
        return AlignmentReport(
            aligned=aligned,
            issues=issues,
            reference_raster=reference_info.name,
            compared_rasters=[info.name for info in compared_infos],
            summary=summary
        )

    def analyze_alignment(self, raster_paths: List[Union[str, Path]], 
                         reference_idx: int = 0) -> AlignmentReport:
        """
        Comprehensive alignment analysis of multiple rasters.
        
        Args:
            raster_paths: List of raster file paths
            reference_idx: Index of reference raster (default: first)
            
        Returns:
            Detailed alignment report
        """
        raster_paths = [Path(p) for p in raster_paths]
        
        if len(raster_paths) < 2:
            return AlignmentReport(
                aligned=True,
                issues=[],
                reference_raster=str(raster_paths[0]) if raster_paths else "",
                compared_rasters=[],
                summary={'message': 'Single or no raster provided'}
            )
        
        reference_path = raster_paths[reference_idx]
        compared_paths = [p for i, p in enumerate(raster_paths) if i != reference_idx]
        
        logger.info(f"Analyzing alignment with reference: {reference_path.name}")
        
        issues = []
        
        # Load reference metadata
        with rasterio.open(reference_path) as ref_src:
            ref_meta = self._extract_spatial_metadata(ref_src)
        
        # Compare each raster to reference
        for raster_path in compared_paths:
            with rasterio.open(raster_path) as src:
                meta = self._extract_spatial_metadata(src)
                raster_issues = self._compare_spatial_metadata(
                    ref_meta, meta, reference_path.name, raster_path.name
                )
                issues.extend(raster_issues)
        
        aligned = len(issues) == 0
        
        # Generate summary
        summary = self._generate_alignment_summary(issues, ref_meta)
        
        return AlignmentReport(
            aligned=aligned,
            issues=issues,
            reference_raster=str(reference_path),
            compared_rasters=[str(p) for p in compared_paths],
            summary=summary
        )
    
    def fix_alignment(self, raster_paths: List[Union[str, Path]], 
                     output_dir: Path,
                     reference_idx: int = 0,
                     force_fix: bool = False) -> Dict[str, Path]:
        """
        Fix alignment issues in raster collection.
        
        Args:
            raster_paths: List of raster file paths
            output_dir: Directory for aligned outputs
            reference_idx: Index of reference raster
            force_fix: Fix even if no issues detected
            
        Returns:
            Dictionary mapping original paths to aligned output paths
        """
        raster_paths = [Path(p) for p in raster_paths]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analyze alignment first
        report = self.analyze_alignment(raster_paths, reference_idx)
        
        if report.aligned and not force_fix:
            logger.info("Rasters are already aligned, returning original paths")
            return {str(p): p for p in raster_paths}
        
        if report.has_issues("critical") and self.config.strategy == AlignmentStrategy.STRICT:
            raise ValueError(f"Critical alignment issues found with STRICT strategy")
        
        # Determine target grid
        reference_path = raster_paths[reference_idx]
        target_grid = self._determine_target_grid(raster_paths, reference_path)
        
        aligned_paths = {}
        
        # Process each raster
        for i, raster_path in enumerate(raster_paths):
            if i == reference_idx and not force_fix:
                # Reference raster might not need changes
                aligned_path = output_dir / f"aligned_{raster_path.name}"
                self._copy_raster(raster_path, aligned_path)
            else:
                aligned_path = output_dir / f"aligned_{raster_path.name}"
                self._align_raster_to_grid(raster_path, aligned_path, target_grid)
            
            aligned_paths[str(raster_path)] = aligned_path
            logger.info(f"Aligned {raster_path.name} -> {aligned_path.name}")
        
        # Verify alignment of outputs
        aligned_report = self.analyze_alignment(list(aligned_paths.values()))
        if not aligned_report.aligned:
            logger.warning("Output rasters still have alignment issues!")
            logger.warning(f"Remaining issues: {len(aligned_report.issues)}")
        else:
            logger.info("✅ All rasters successfully aligned")
        
        return aligned_paths
    
    def create_aligned_subsets(self, raster_paths: List[Union[str, Path]],
                             output_dir: Path,
                             subset_size: int = 100,
                             bounds: Optional[Tuple[float, float, float, float]] = None) -> Dict[str, Path]:
        """
        Create perfectly aligned subsets from multiple rasters.
        
        Args:
            raster_paths: List of raster file paths
            output_dir: Directory for subset outputs
            subset_size: Size in pixels (creates square subsets)
            bounds: Optional specific bounds for subsets
            
        Returns:
            Dictionary mapping original paths to subset paths
        """
        raster_paths = [Path(p) for p in raster_paths]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine common bounds and resolution
        if bounds is None:
            bounds = self._calculate_intersection_bounds(raster_paths)
        
        logger.info(f"Intersection bounds: {bounds}")
        
        # Get reference resolution (use first raster)
        with rasterio.open(raster_paths[0]) as src:
            # Extract pixel size from transform: [0] is x-pixel size, [4] is y-pixel size
            pixel_size = abs(src.transform[0])
        
        logger.info(f"Pixel size: {pixel_size}")
        
        # Calculate subset bounds within intersection
        subset_width = subset_size * pixel_size
        subset_height = subset_size * pixel_size
        
        # Ensure we have valid bounds with non-zero dimensions
        bounds_width = bounds[2] - bounds[0]  # right - left
        bounds_height = bounds[3] - bounds[1]  # top - bottom
        
        if bounds_width <= 0 or bounds_height <= 0:
            raise ValueError(f"Invalid intersection bounds: {bounds} (width: {bounds_width}, height: {bounds_height})")
        
        # Limit subset size to available area
        actual_subset_width = min(subset_width, bounds_width * 0.8)  # Use 80% of available width
        actual_subset_height = min(subset_height, bounds_height * 0.8)
        
        # Position subset at center of bounds
        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2
        
        subset_bounds = (
            center_x - actual_subset_width / 2,  # left
            center_y - actual_subset_height / 2,  # bottom
            center_x + actual_subset_width / 2,   # right
            center_y + actual_subset_height / 2   # top
        )
        
        logger.info(f"Creating {subset_size}x{subset_size} subsets")
        logger.info(f"Subset bounds: {subset_bounds}")
        logger.info(f"Subset dimensions: {actual_subset_width:.6f} x {actual_subset_height:.6f}")
        
        # Create transform for subset
        subset_transform = from_bounds(*subset_bounds, subset_size, subset_size)
        
        subset_paths = {}
        
        # Process each raster
        for raster_path in raster_paths:
            subset_path = output_dir / f"subset_{raster_path.name}"
            
            with rasterio.open(raster_path) as src:
                # Prepare output metadata
                out_meta = src.meta.copy()
                out_meta.update({
                    'height': subset_size,
                    'width': subset_size,
                    'transform': subset_transform
                })
                
                # Create output array
                out_data = np.zeros((1, subset_size, subset_size), dtype=out_meta['dtype'])
                
                # Reproject to subset
                reproject(
                    source=rasterio.band(src, 1),
                    destination=out_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=subset_transform,
                    dst_crs=src.crs,
                    resampling=self.config.resampling_method.value
                )
                
                # Write subset
                with rasterio.open(subset_path, 'w', **out_meta) as dst:
                    dst.write(out_data)
            
            subset_paths[str(raster_path)] = subset_path
            logger.info(f"Created subset: {raster_path.name} -> {subset_path.name}")
        
        return subset_paths
    
    def _extract_spatial_metadata(self, src: rasterio.DatasetReader) -> Dict[str, Any]:
        """Extract spatial metadata from rasterio dataset."""
        return {
            'crs': src.crs,
            'transform': src.transform,
            'bounds': src.bounds,
            'shape': (src.height, src.width),
            'resolution': (abs(src.transform[0]), abs(src.transform[4])),
            'nodata': src.nodata,
            'dtype': src.dtypes[0]
        }
    
    def _compare_spatial_metadata(self, ref_meta: Dict[str, Any], 
                                 meta: Dict[str, Any],
                                 ref_name: str, 
                                 raster_name: str) -> List[AlignmentIssue]:
        """Compare spatial metadata and identify issues."""
        issues = []
        
        # Check CRS
        if ref_meta['crs'] != meta['crs']:
            issues.append(AlignmentIssue(
                type="crs_mismatch",
                severity="critical",
                description=f"CRS mismatch between {ref_name} and {raster_name}",
                values={'reference': str(ref_meta['crs']), 'raster': str(meta['crs'])},
                fixable=True,
                suggested_fix="reproject_to_common_crs"
            ))
        
        # Check resolution
        ref_res = ref_meta['resolution']
        raster_res = meta['resolution']
        res_diff = (abs(ref_res[0] - raster_res[0]), abs(ref_res[1] - raster_res[1]))
        
        if max(res_diff) > self.config.resolution_tolerance:
            severity = "critical" if max(res_diff) > 0.001 else "warning"
            issues.append(AlignmentIssue(
                type="resolution_mismatch",
                severity=severity,
                description=f"Resolution mismatch between {ref_name} and {raster_name}",
                values={'reference': ref_res, 'raster': raster_res, 'difference': res_diff},
                fixable=True,
                suggested_fix="resample_to_common_resolution"
            ))
        
        # Check bounds
        ref_bounds = ref_meta['bounds']
        raster_bounds = meta['bounds']
        bounds_diff = tuple(abs(r - m) for r, m in zip(ref_bounds, raster_bounds))
        
        if max(bounds_diff) > self.config.bounds_tolerance:
            severity = "critical" if max(bounds_diff) > 1.0 else "warning"
            issues.append(AlignmentIssue(
                type="bounds_mismatch",
                severity=severity,
                description=f"Bounds mismatch between {ref_name} and {raster_name}",
                values={'reference': ref_bounds, 'raster': raster_bounds, 'difference': bounds_diff},
                fixable=True,
                suggested_fix="reproject_to_common_bounds"
            ))
        
        # Check transform alignment (pixel grid alignment)
        ref_transform = ref_meta['transform']
        raster_transform = meta['transform']
        
        # Check if origins are aligned
        origin_diff = (
            abs(ref_transform[2] - raster_transform[2]),  # x origin
            abs(ref_transform[5] - raster_transform[5])   # y origin
        )
        
        pixel_size = max(abs(ref_transform[1]), abs(ref_transform[5]))
        if max(origin_diff) > pixel_size * 0.1:  # 10% of pixel size
            issues.append(AlignmentIssue(
                type="grid_misalignment",
                severity="warning",
                description=f"Pixel grid misalignment between {ref_name} and {raster_name}",
                values={'origin_difference': origin_diff, 'pixel_size': pixel_size},
                fixable=True,
                suggested_fix="snap_to_pixel_grid"
            ))
        
        return issues
    
    def _compare_dataset_metadata(self, ref_meta: Dict[str, Any], 
                                 dataset_meta: Dict[str, Any],
                                 ref_name: str, 
                                 dataset_name: str) -> List[AlignmentIssue]:
        """Compare dataset metadata and identify alignment issues."""
        issues = []
        
        # Check CRS
        if ref_meta['crs'] != dataset_meta['crs']:
            issues.append(AlignmentIssue(
                type="crs_mismatch",
                severity="critical",
                description=f"CRS mismatch between {ref_name} and {dataset_name}",
                values={'reference': ref_meta['crs'], 'dataset': dataset_meta['crs']},
                fixable=True,
                suggested_fix="reproject_to_common_crs"
            ))
        
        # Check resolution - allow small differences for mixed datasets
        ref_res = ref_meta['resolution']
        dataset_res = dataset_meta['resolution']
        res_diff = (abs(ref_res[0] - dataset_res[0]), abs(ref_res[1] - dataset_res[1]))
        
        # Use larger tolerance for mixed datasets
        tolerance = max(self.config.resolution_tolerance, 0.01)  # At least 0.01° tolerance
        
        if max(res_diff) > tolerance:
            severity = "warning" if max(res_diff) < 0.1 else "critical"
            issues.append(AlignmentIssue(
                type="resolution_mismatch",
                severity=severity,
                description=f"Resolution mismatch between {ref_name} ({ref_meta['dataset_type']}) and {dataset_name} ({dataset_meta['dataset_type']})",
                values={
                    'reference': ref_res, 
                    'dataset': dataset_res, 
                    'difference': res_diff,
                    'ref_type': ref_meta['dataset_type'],
                    'dataset_type': dataset_meta['dataset_type']
                },
                fixable=True,
                suggested_fix="align_to_common_grid"
            ))
        
        # Check bounds overlap
        ref_bounds = ref_meta['bounds']
        dataset_bounds = dataset_meta['bounds']
        
        # Calculate intersection
        intersection = (
            max(ref_bounds[0], dataset_bounds[0]),  # left
            max(ref_bounds[1], dataset_bounds[1]),  # bottom
            min(ref_bounds[2], dataset_bounds[2]),  # right
            min(ref_bounds[3], dataset_bounds[3])   # top
        )
        
        # Check if there's valid intersection
        if intersection[0] >= intersection[2] or intersection[1] >= intersection[3]:
            issues.append(AlignmentIssue(
                type="no_bounds_overlap",
                severity="critical",
                description=f"No geographic overlap between {ref_name} and {dataset_name}",
                values={'reference': ref_bounds, 'dataset': dataset_bounds},
                fixable=False,
                suggested_fix="check_dataset_coverage"
            ))
        else:
            # Check if overlap is significant
            ref_area = (ref_bounds[2] - ref_bounds[0]) * (ref_bounds[3] - ref_bounds[1])
            overlap_area = (intersection[2] - intersection[0]) * (intersection[3] - intersection[1])
            overlap_ratio = overlap_area / ref_area
            
            if overlap_ratio < 0.5:  # Less than 50% overlap
                issues.append(AlignmentIssue(
                    type="limited_bounds_overlap",
                    severity="warning",
                    description=f"Limited geographic overlap ({overlap_ratio:.1%}) between {ref_name} and {dataset_name}",
                    values={
                        'reference': ref_bounds, 
                        'dataset': dataset_bounds,
                        'intersection': intersection,
                        'overlap_ratio': overlap_ratio
                    },
                    fixable=True,
                    suggested_fix="crop_to_intersection"
                ))
        
        return issues
    
    def _generate_mixed_dataset_summary(self, issues: List[AlignmentIssue], 
                                       ref_meta: Dict[str, Any],
                                       dataset_infos: List) -> Dict[str, Any]:
        """Generate summary for mixed dataset alignment analysis."""
        passthrough_count = sum(1 for info in dataset_infos if info.metadata.get('passthrough', False))
        resampled_count = len(dataset_infos) - passthrough_count
        
        return {
            'total_issues': len(issues),
            'critical_issues': len([i for i in issues if i.severity == "critical"]),
            'warning_issues': len([i for i in issues if i.severity == "warning"]),
            'fixable_issues': len([i for i in issues if i.fixable]),
            'dataset_composition': {
                'total': len(dataset_infos),
                'passthrough': passthrough_count,
                'resampled': resampled_count
            },
            'reference_metadata': {
                'crs': ref_meta['crs'],
                'bounds': ref_meta['bounds'],
                'resolution': ref_meta['resolution'],
                'shape': ref_meta['shape'],
                'type': ref_meta['dataset_type']
            },
            'issue_types': list(set(issue.type for issue in issues)),
            'recommended_strategy': self._recommend_mixed_dataset_strategy(issues, dataset_infos)
        }
    
    def _recommend_mixed_dataset_strategy(self, issues: List[AlignmentIssue], dataset_infos: List) -> str:
        """Recommend alignment strategy for mixed datasets."""
        if not issues:
            return "no_action_needed"
        
        has_critical = any(issue.severity == "critical" for issue in issues)
        has_crs_issues = any(issue.type == "crs_mismatch" for issue in issues)
        has_no_overlap = any(issue.type == "no_bounds_overlap" for issue in issues)
        
        if has_no_overlap:
            return "check_dataset_coverage"
        elif has_crs_issues:
            return "reproject_to_common_crs"
        elif has_critical:
            return "align_to_common_grid"
        else:
            return "minor_adjustment_for_mixed_data"

    def _generate_alignment_summary(self, issues: List[AlignmentIssue], 
                                  ref_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of alignment analysis."""
        return {
            'total_issues': len(issues),
            'critical_issues': len([i for i in issues if i.severity == "critical"]),
            'warning_issues': len([i for i in issues if i.severity == "warning"]),
            'fixable_issues': len([i for i in issues if i.fixable]),
            'reference_metadata': {
                'crs': str(ref_meta['crs']),
                'bounds': ref_meta['bounds'],
                'resolution': ref_meta['resolution'],
                'shape': ref_meta['shape']
            },
            'issue_types': list(set(issue.type for issue in issues)),
            'recommended_strategy': self._recommend_strategy(issues)
        }
    
    def _recommend_strategy(self, issues: List[AlignmentIssue]) -> str:
        """Recommend alignment strategy based on issues found."""
        if not issues:
            return "no_action_needed"
        
        has_critical = any(issue.severity == "critical" for issue in issues)
        has_crs_issues = any(issue.type == "crs_mismatch" for issue in issues)
        has_bounds_issues = any(issue.type == "bounds_mismatch" for issue in issues)
        
        if has_crs_issues:
            return "reproject_to_common_crs"
        elif has_critical and has_bounds_issues:
            return "reproject_to_intersection"
        elif has_bounds_issues:
            return "crop_to_intersection"
        else:
            return "minor_adjustment"
    
    def _determine_target_grid(self, raster_paths: List[Path], 
                              reference_path: Path) -> Dict[str, Any]:
        """Determine target grid for alignment."""
        with rasterio.open(reference_path) as ref_src:
            if self.config.target_crs:
                target_crs = self.config.target_crs
            else:
                target_crs = ref_src.crs
            
            if self.config.target_resolution:
                pixel_size = self.config.target_resolution
            else:
                pixel_size = abs(ref_src.transform[0])
            
            if self.config.target_bounds:
                bounds = self.config.target_bounds
            else:
                # Calculate intersection or union bounds
                bounds = self._calculate_intersection_bounds(raster_paths)
            
            # Create target transform
            width = int((bounds[2] - bounds[0]) / pixel_size)
            height = int((bounds[3] - bounds[1]) / pixel_size)
            
            transform = from_bounds(*bounds, width, height)
            
            return {
                'crs': target_crs,
                'transform': transform,
                'bounds': bounds,
                'width': width,
                'height': height,
                'nodata': ref_src.nodata,
                'dtype': ref_src.dtypes[0]
            }
    
    def _calculate_intersection_bounds(self, raster_paths: List[Path]) -> Tuple[float, float, float, float]:
        """Calculate intersection bounds of multiple rasters."""
        all_bounds = []
        
        for raster_path in raster_paths:
            with rasterio.open(raster_path) as src:
                all_bounds.append(src.bounds)
        
        # Calculate intersection
        left = max(bounds.left for bounds in all_bounds)
        bottom = max(bounds.bottom for bounds in all_bounds)
        right = min(bounds.right for bounds in all_bounds)
        top = min(bounds.top for bounds in all_bounds)
        
        return (left, bottom, right, top)
    
    def _align_raster_to_grid(self, input_path: Path, output_path: Path, 
                             target_grid: Dict[str, Any]):
        """Align a raster to target grid specification."""
        with rasterio.open(input_path) as src:
            # Prepare output metadata
            out_meta = src.meta.copy()
            out_meta.update({
                'crs': target_grid['crs'],
                'transform': target_grid['transform'],
                'width': target_grid['width'],
                'height': target_grid['height']
            })
            
            # Create output array
            out_data = np.zeros((src.count, target_grid['height'], target_grid['width']), 
                              dtype=out_meta['dtype'])
            
            # Reproject each band
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=out_data[i-1],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=target_grid['transform'],
                    dst_crs=target_grid['crs'],
                    resampling=self.config.resampling_method.value
                )
            
            # Write aligned raster
            with rasterio.open(output_path, 'w', **out_meta) as dst:
                dst.write(out_data)
    
    def _copy_raster(self, input_path: Path, output_path: Path):
        """Copy raster file to new location."""
        import shutil
        shutil.copy2(input_path, output_path)
    
    @log_operation("calculate_grid_shifts")
    def calculate_grid_shifts(self, 
                            dataset_infos: List[Any],
                            reference_idx: int = 0) -> List[GridAlignment]:
        """
        Calculate shift vectors for aligning datasets without loading data.
        
        Args:
            dataset_infos: List of ResampledDatasetInfo objects
            reference_idx: Index of reference dataset
            
        Returns:
            List of GridAlignment objects with shift information
        """
        if len(dataset_infos) < 2:
            return []
            
        reference = dataset_infos[reference_idx]
        ref_resolution = reference.target_resolution
        ref_bounds = reference.bounds
        
        # Calculate reference grid origin
        ref_origin_x = ref_bounds[0]  # min_x
        ref_origin_y = ref_bounds[3]  # max_y (raster origin is top-left)
        
        alignments = []
        
        for i, dataset in enumerate(dataset_infos):
            if i == reference_idx:
                # Reference aligns with itself
                alignments.append(GridAlignment(
                    reference_dataset=reference.name,
                    aligned_dataset=dataset.name,
                    x_shift=0.0,
                    y_shift=0.0,
                    requires_shift=False,
                    shift_pixels_x=0.0,
                    shift_pixels_y=0.0
                ))
                continue
            
            # Calculate dataset grid origin
            ds_origin_x = dataset.bounds[0]
            ds_origin_y = dataset.bounds[3]
            
            # Calculate misalignment in coordinate units
            x_misalign = ds_origin_x - ref_origin_x
            y_misalign = ds_origin_y - ref_origin_y
            
            # Calculate misalignment in pixels
            x_misalign_pixels = x_misalign / ref_resolution
            y_misalign_pixels = y_misalign / ref_resolution
            
            # Check if misalignment is a whole pixel
            x_pixel_fraction = x_misalign_pixels - round(x_misalign_pixels)
            y_pixel_fraction = y_misalign_pixels - round(y_misalign_pixels)
            
            # If fractional pixel offset exists, calculate shift needed
            if abs(x_pixel_fraction) > 0.01 or abs(y_pixel_fraction) > 0.01:
                # Shift needed to align to reference grid
                x_shift = -x_pixel_fraction * ref_resolution
                y_shift = -y_pixel_fraction * ref_resolution
                
                alignments.append(GridAlignment(
                    reference_dataset=reference.name,
                    aligned_dataset=dataset.name,
                    x_shift=x_shift,
                    y_shift=y_shift,
                    requires_shift=True,
                    shift_pixels_x=x_pixel_fraction,
                    shift_pixels_y=y_pixel_fraction
                ))
                
                logger.info(f"Dataset {dataset.name} requires shift: "
                          f"({x_shift:.6f}, {y_shift:.6f}) degrees")
            else:
                # Already aligned
                alignments.append(GridAlignment(
                    reference_dataset=reference.name,
                    aligned_dataset=dataset.name,
                    x_shift=0.0,
                    y_shift=0.0,
                    requires_shift=False,
                    shift_pixels_x=0.0,
                    shift_pixels_y=0.0
                ))
                
        return alignments
    
    def create_aligned_coordinate_query(self,
                                      table_name: str,
                                      alignment: GridAlignment,
                                      chunk_bounds: Tuple[float, float, float, float],
                                      name_column: str = None) -> str:
        """
        Create SQL query that applies alignment shift.
        
        Args:
            table_name: Database table name
            alignment: Grid alignment information
            chunk_bounds: (min_x, min_y, max_x, max_y) for chunk
            name_column: Optional column name for value (default: 'value')
            
        Returns:
            SQL query string with alignment applied
        """
        value_col = name_column if name_column else 'value'
        
        if not alignment.requires_shift:
            # No shift needed
            return f"""
                SELECT 
                    x_coord as x,
                    y_coord as y,
                    value as {value_col}
                FROM {table_name}
                WHERE x_coord >= {chunk_bounds[0]}
                  AND x_coord < {chunk_bounds[2]}
                  AND y_coord >= {chunk_bounds[1]}
                  AND y_coord < {chunk_bounds[3]}
                  AND value IS NOT NULL
            """
        else:
            # Apply shift during query
            return f"""
                SELECT 
                    x_coord + {alignment.x_shift} as x,
                    y_coord + {alignment.y_shift} as y,
                    value as {value_col}
                FROM {table_name}
                WHERE x_coord >= {chunk_bounds[0] - alignment.x_shift}
                  AND x_coord < {chunk_bounds[2] - alignment.x_shift}
                  AND y_coord >= {chunk_bounds[1] - alignment.y_shift}
                  AND y_coord < {chunk_bounds[3] - alignment.y_shift}
                  AND value IS NOT NULL
            """