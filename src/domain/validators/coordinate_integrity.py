"""Coordinate integrity validators for geospatial data."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import box
import rasterio

from src.abstractions.interfaces.validator import (
    BaseValidator, ValidationResult, ValidationType, ValidationSeverity
)
from src.abstractions.types.validation_types import (
    BoundsValidation, CoordinateValidation, ValueRangeValidation,
    ValidationErrorType
)

logger = logging.getLogger(__name__)


class BoundsConsistencyValidator(BaseValidator):
    """
    Validates that geographic bounds are consistent across datasets and operations.
    
    Checks for:
    - Bounds within valid geographic ranges
    - Consistency between metadata and actual data bounds
    - Proper bounds ordering (min < max)
    - Bounds alignment with grid systems
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize bounds validator.
        
        Args:
            tolerance: Tolerance for floating point comparisons
        """
        super().__init__("BoundsConsistencyValidator")
        self.tolerance = tolerance
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate bounds consistency.
        
        Args:
            data: Dictionary with 'bounds' and optional 'metadata_bounds'
            
        Returns:
            ValidationResult with any issues found
        """
        issues = []
        
        # Extract bounds
        bounds = data.get('bounds')
        metadata_bounds = data.get('metadata_bounds', bounds)
        crs = data.get('crs', 'EPSG:4326')
        
        if not bounds:
            issues.append(self.create_issue(
                ValidationType.BOUNDS_CONSISTENCY,
                ValidationSeverity.ERROR,
                "No bounds provided for validation"
            ))
            return self.create_result(is_valid=False, issues=issues)
        
        # Validate bounds format
        if not self._validate_bounds_format(bounds):
            issues.append(self.create_issue(
                ValidationType.BOUNDS_CONSISTENCY,
                ValidationSeverity.ERROR,
                f"Invalid bounds format: {bounds}. Expected (minx, miny, maxx, maxy)"
            ))
            return self.create_result(is_valid=False, issues=issues)
        
        minx, miny, maxx, maxy = bounds
        
        # Check proper ordering
        if minx >= maxx:
            issues.append(self.create_issue(
                ValidationType.BOUNDS_CONSISTENCY,
                ValidationSeverity.ERROR,
                f"Invalid bounds: minx ({minx}) >= maxx ({maxx})"
            ))
        
        if miny >= maxy:
            issues.append(self.create_issue(
                ValidationType.BOUNDS_CONSISTENCY,
                ValidationSeverity.ERROR,
                f"Invalid bounds: miny ({miny}) >= maxy ({maxy})"
            ))
        
        # Check geographic validity for lat/lon
        if crs == 'EPSG:4326':
            if not (-180 - self.tolerance <= minx <= 180 + self.tolerance and 
                    -180 - self.tolerance <= maxx <= 180 + self.tolerance):
                issues.append(self.create_issue(
                    ValidationType.BOUNDS_CONSISTENCY,
                    ValidationSeverity.ERROR,
                    f"Longitude out of range [-180, 180]: min={minx}, max={maxx}"
                ))
            
            if not (-90 - self.tolerance <= miny <= 90 + self.tolerance and 
                    -90 - self.tolerance <= maxy <= 90 + self.tolerance):
                issues.append(self.create_issue(
                    ValidationType.BOUNDS_CONSISTENCY,
                    ValidationSeverity.ERROR,
                    f"Latitude out of range [-90, 90]: min={miny}, max={maxy}"
                ))
        
        # Check consistency with metadata bounds
        if metadata_bounds and metadata_bounds != bounds:
            validation = BoundsValidation(
                expected_bounds=metadata_bounds,
                actual_bounds=bounds,
                tolerance=self.tolerance,
                coordinate_system=crs
            )
            
            if not validation.is_consistent:
                issues.append(self.create_issue(
                    ValidationType.BOUNDS_CONSISTENCY,
                    ValidationSeverity.WARNING,
                    f"Bounds mismatch: metadata={metadata_bounds}, actual={bounds}, "
                    f"max deviation={validation.max_deviation:.6f}",
                    details={'validation': validation}
                ))
        
        # Check if bounds represent a valid area
        area = (maxx - minx) * (maxy - miny)
        if area <= 0:
            issues.append(self.create_issue(
                ValidationType.BOUNDS_CONSISTENCY,
                ValidationSeverity.ERROR,
                f"Bounds represent zero or negative area: {area}"
            ))
        
        is_valid = not any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        
        return self.create_result(
            is_valid=is_valid,
            issues=issues,
            metadata={'bounds': bounds, 'crs': crs}
        )
    
    def _validate_bounds_format(self, bounds: Any) -> bool:
        """Check if bounds have valid format."""
        if not isinstance(bounds, (list, tuple)):
            return False
        if len(bounds) != 4:
            return False
        try:
            for val in bounds:
                float(val)
            return True
        except (TypeError, ValueError):
            return False


class CoordinateTransformValidator(BaseValidator):
    """
    Validates coordinate transformations between different CRS.
    
    Checks for:
    - Valid transformation paths
    - Transformation accuracy
    - Datum shifts
    - Projection distortions
    """
    
    def __init__(self, max_error_meters: float = 1.0):
        """
        Initialize transform validator.
        
        Args:
            max_error_meters: Maximum acceptable transformation error
        """
        super().__init__("CoordinateTransformValidator")
        self.max_error_meters = max_error_meters
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """
        Validate coordinate transformation.
        
        Args:
            data: Dictionary with 'source_crs', 'target_crs', and 'sample_points'
            
        Returns:
            ValidationResult with any issues found
        """
        issues = []
        
        source_crs = data.get('source_crs')
        target_crs = data.get('target_crs')
        sample_points = data.get('sample_points', [])
        
        if not source_crs or not target_crs:
            issues.append(self.create_issue(
                ValidationType.COORDINATE_TRANSFORM,
                ValidationSeverity.ERROR,
                "Source and target CRS must be provided"
            ))
            return self.create_result(is_valid=False, issues=issues)
        
        # Validate CRS
        try:
            source_proj = pyproj.CRS(source_crs)
            target_proj = pyproj.CRS(target_crs)
        except Exception as e:
            issues.append(self.create_issue(
                ValidationType.COORDINATE_TRANSFORM,
                ValidationSeverity.ERROR,
                f"Invalid CRS: {e}"
            ))
            return self.create_result(is_valid=False, issues=issues)
        
        # Check if transformation is available
        try:
            transformer = pyproj.Transformer.from_crs(
                source_proj, target_proj, always_xy=True
            )
        except Exception as e:
            issues.append(self.create_issue(
                ValidationType.COORDINATE_TRANSFORM,
                ValidationSeverity.ERROR,
                f"Cannot create transformation: {e}"
            ))
            return self.create_result(is_valid=False, issues=issues)
        
        # Validate sample points if provided
        if sample_points:
            errors = self._validate_transformation_accuracy(
                transformer, sample_points, data.get('expected_points')
            )
            
            if errors:
                max_error = max(errors)
                if max_error > self.max_error_meters:
                    issues.append(self.create_issue(
                        ValidationType.COORDINATE_TRANSFORM,
                        ValidationSeverity.ERROR,
                        f"Transformation error ({max_error:.2f}m) exceeds "
                        f"threshold ({self.max_error_meters}m)",
                        details={'errors': errors}
                    ))
        
        # Check for datum shifts
        if source_proj.datum != target_proj.datum:
            issues.append(self.create_issue(
                ValidationType.COORDINATE_TRANSFORM,
                ValidationSeverity.WARNING,
                f"Datum shift from {source_proj.datum} to {target_proj.datum}"
            ))
        
        is_valid = not any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        
        return self.create_result(
            is_valid=is_valid,
            issues=issues,
            metadata={
                'source_crs': str(source_proj),
                'target_crs': str(target_proj),
                'has_datum_shift': source_proj.datum != target_proj.datum
            }
        )
    
    def _validate_transformation_accuracy(self,
                                        transformer: pyproj.Transformer,
                                        sample_points: List[Tuple[float, float]],
                                        expected_points: Optional[List[Tuple[float, float]]]) -> List[float]:
        """Calculate transformation errors."""
        if not expected_points:
            return []
        
        errors = []
        for (x, y), (ex, ey) in zip(sample_points, expected_points):
            tx, ty = transformer.transform(x, y)
            # Calculate error in meters (approximate for small distances)
            error = self._calculate_distance_meters(tx, ty, ex, ey, transformer.target_crs)
            errors.append(error)
        
        return errors
    
    def _calculate_distance_meters(self, x1: float, y1: float, 
                                 x2: float, y2: float, crs: pyproj.CRS) -> float:
        """Calculate distance in meters between two points."""
        if crs.is_geographic:
            # Use geodesic distance for geographic coordinates
            geod = pyproj.Geod(ellps=crs.ellipsoid.name)
            _, _, dist = geod.inv(x1, y1, x2, y2)
            return dist
        else:
            # Euclidean distance for projected coordinates
            return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


class ParquetValueValidator(BaseValidator):
    """
    Validates data values in Parquet files.
    
    Checks for:
    - Value ranges
    - Data types
    - Null/missing values
    - Statistical anomalies
    """
    
    def __init__(self, 
                 max_null_percentage: float = 10.0,
                 outlier_std_threshold: float = 3.0):
        """
        Initialize Parquet validator.
        
        Args:
            max_null_percentage: Maximum acceptable null percentage
            outlier_std_threshold: Standard deviations for outlier detection
        """
        super().__init__("ParquetValueValidator")
        self.max_null_percentage = max_null_percentage
        self.outlier_std_threshold = outlier_std_threshold
    
    def validate(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> ValidationResult:
        """
        Validate Parquet data values.
        
        Args:
            data: DataFrame or dict with 'dataframe' and 'variable_config'
            
        Returns:
            ValidationResult with any issues found
        """
        issues = []
        
        # Extract DataFrame and configuration
        if isinstance(data, pd.DataFrame):
            df = data
            variable_config = {}
        else:
            df = data.get('dataframe')
            variable_config = data.get('variable_config', {})
        
        if df is None or df.empty:
            issues.append(self.create_issue(
                ValidationType.VALUE_RANGE,
                ValidationSeverity.ERROR,
                "No data provided for validation"
            ))
            return self.create_result(is_valid=False, issues=issues)
        
        # Validate each numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_issues = self._validate_column(df[col], variable_config.get(col, {}))
            issues.extend(col_issues)
        
        # Check for coordinate columns
        coord_cols = ['x', 'y', 'lon', 'lat', 'longitude', 'latitude']
        found_coords = [col for col in coord_cols if col in df.columns]
        
        if found_coords:
            coord_issues = self._validate_coordinates(df, found_coords)
            issues.extend(coord_issues)
        
        is_valid = not any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        
        return self.create_result(
            is_valid=is_valid,
            issues=issues,
            metadata={
                'row_count': len(df),
                'column_count': len(df.columns),
                'numeric_columns': list(numeric_cols)
            }
        )
    
    def _validate_column(self, series: pd.Series, config: Dict[str, Any]) -> List[Any]:
        """Validate a single column."""
        issues = []
        col_name = series.name
        
        # Check null values
        null_count = series.isna().sum()
        total_count = len(series)
        null_percentage = (null_count / total_count) * 100 if total_count > 0 else 0
        
        if null_percentage > self.max_null_percentage:
            issues.append(self.create_issue(
                ValidationType.DATA_COMPLETENESS,
                ValidationSeverity.WARNING,
                f"Column '{col_name}' has {null_percentage:.1f}% null values",
                details={'null_count': null_count, 'total_count': total_count}
            ))
        
        # Check value range if configured
        if 'min' in config or 'max' in config:
            actual_min = series.min()
            actual_max = series.max()
            
            expected_min = config.get('min', -np.inf)
            expected_max = config.get('max', np.inf)
            
            if actual_min < expected_min or actual_max > expected_max:
                issues.append(self.create_issue(
                    ValidationType.VALUE_RANGE,
                    ValidationSeverity.ERROR,
                    f"Column '{col_name}' values out of range: "
                    f"[{actual_min:.2f}, {actual_max:.2f}] not in "
                    f"[{expected_min:.2f}, {expected_max:.2f}]",
                    details={
                        'validation': ValueRangeValidation(
                            variable_name=col_name,
                            expected_range=(expected_min, expected_max),
                            actual_min=actual_min,
                            actual_max=actual_max,
                            null_count=null_count,
                            total_count=total_count
                        )
                    }
                ))
        
        # Check for outliers
        if len(series.dropna()) > 10:  # Need sufficient data
            mean = series.mean()
            std = series.std()
            
            if std > 0:
                z_scores = np.abs((series - mean) / std)
                outlier_count = (z_scores > self.outlier_std_threshold).sum()
                
                if outlier_count > 0:
                    outlier_percentage = (outlier_count / total_count) * 100
                    if outlier_percentage > 1.0:  # More than 1% outliers
                        issues.append(self.create_issue(
                            ValidationType.VALUE_RANGE,
                            ValidationSeverity.WARNING,
                            f"Column '{col_name}' has {outlier_count} potential outliers "
                            f"({outlier_percentage:.1f}%)",
                            details={'outlier_threshold': self.outlier_std_threshold}
                        ))
        
        return issues
    
    def _validate_coordinates(self, df: pd.DataFrame, coord_cols: List[str]) -> List[Any]:
        """Validate coordinate columns."""
        issues = []
        
        # Check longitude columns
        lon_cols = [col for col in coord_cols if col in ['x', 'lon', 'longitude']]
        for col in lon_cols:
            if col in df.columns:
                lon_values = df[col].dropna()
                if len(lon_values) > 0:
                    if lon_values.min() < -180 or lon_values.max() > 180:
                        issues.append(self.create_issue(
                            ValidationType.COORDINATE_TRANSFORM,
                            ValidationSeverity.ERROR,
                            f"Longitude column '{col}' has values outside [-180, 180]"
                        ))
        
        # Check latitude columns
        lat_cols = [col for col in coord_cols if col in ['y', 'lat', 'latitude']]
        for col in lat_cols:
            if col in df.columns:
                lat_values = df[col].dropna()
                if len(lat_values) > 0:
                    if lat_values.min() < -90 or lat_values.max() > 90:
                        issues.append(self.create_issue(
                            ValidationType.COORDINATE_TRANSFORM,
                            ValidationSeverity.ERROR,
                            f"Latitude column '{col}' has values outside [-90, 90]"
                        ))
        
        return issues