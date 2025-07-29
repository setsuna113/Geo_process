"""Types for validation framework."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import numpy as np


@dataclass
class BoundsValidation:
    """Validation data for geographic bounds."""
    expected_bounds: Tuple[float, float, float, float]
    actual_bounds: Tuple[float, float, float, float]
    tolerance: float = 1e-6
    coordinate_system: str = "EPSG:4326"
    
    @property
    def is_consistent(self) -> bool:
        """Check if bounds are within tolerance."""
        for expected, actual in zip(self.expected_bounds, self.actual_bounds):
            if abs(expected - actual) > self.tolerance:
                return False
        return True
    
    @property
    def max_deviation(self) -> float:
        """Maximum deviation between expected and actual bounds."""
        deviations = [abs(e - a) for e, a in zip(self.expected_bounds, self.actual_bounds)]
        return max(deviations)


@dataclass
class CoordinateValidation:
    """Validation data for coordinate transformations."""
    source_crs: str
    target_crs: str
    sample_points: List[Tuple[float, float]]
    transformed_points: List[Tuple[float, float]]
    expected_points: Optional[List[Tuple[float, float]]] = None
    max_error_meters: float = 1.0
    
    def calculate_errors(self) -> List[float]:
        """Calculate transformation errors if expected points provided."""
        if not self.expected_points:
            return []
        
        errors = []
        for (tx, ty), (ex, ey) in zip(self.transformed_points, self.expected_points):
            # Simple Euclidean distance (assumes projected coordinates)
            error = np.sqrt((tx - ex)**2 + (ty - ey)**2)
            errors.append(error)
        
        return errors


@dataclass 
class ValueRangeValidation:
    """Validation data for value ranges."""
    variable_name: str
    expected_range: Tuple[float, float]
    actual_min: float
    actual_max: float
    null_count: int = 0
    total_count: int = 0
    
    @property
    def is_within_range(self) -> bool:
        """Check if actual values are within expected range."""
        return (self.actual_min >= self.expected_range[0] and 
                self.actual_max <= self.expected_range[1])
    
    @property
    def null_percentage(self) -> float:
        """Percentage of null values."""
        if self.total_count == 0:
            return 0.0
        return (self.null_count / self.total_count) * 100


@dataclass
class SpatialIntegrityValidation:
    """Validation data for spatial integrity checks."""
    dataset_name: str
    grid_alignment: bool
    projection_match: bool
    resolution_match: bool
    coverage_complete: bool
    overlapping_tiles: List[str] = None
    missing_tiles: List[str] = None
    
    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.overlapping_tiles is None:
            self.overlapping_tiles = []
        if self.missing_tiles is None:
            self.missing_tiles = []
    
    @property
    def is_valid(self) -> bool:
        """Check if all spatial integrity checks pass."""
        return (self.grid_alignment and 
                self.projection_match and 
                self.resolution_match and 
                self.coverage_complete and
                len(self.overlapping_tiles) == 0 and
                len(self.missing_tiles) == 0)


class ValidationErrorType(Enum):
    """Types of validation errors."""
    BOUNDS_MISMATCH = "bounds_mismatch"
    COORDINATE_ERROR = "coordinate_error"
    VALUE_OUT_OF_RANGE = "value_out_of_range"
    NULL_VALUE_EXCESS = "null_value_excess"
    GRID_MISALIGNMENT = "grid_misalignment"
    PROJECTION_MISMATCH = "projection_mismatch"
    RESOLUTION_MISMATCH = "resolution_mismatch"
    COVERAGE_GAP = "coverage_gap"
    TILE_OVERLAP = "tile_overlap"
    DATA_CORRUPTION = "data_corruption"


@dataclass
class ValidationContext:
    """Context information for validation operations."""
    stage: str  # Pipeline stage (e.g., "merge", "resample", "analysis")
    dataset_ids: List[str]
    operation: str
    timestamp: str
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.parameters is None:
            self.parameters = {}