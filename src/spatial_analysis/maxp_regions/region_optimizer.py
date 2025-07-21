# src/spatial_analysis/maxp_regions/region_optimizer.py
"""Max-p regions optimizer for spatially constrained clustering."""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
try:
    from spopt.region import MaxPHeuristic
    import libpysal
except ImportError:
    MaxPHeuristic = None
    libpysal = None
    import warnings
    warnings.warn("spopt and libpysal not available. Install with: pip install spopt libpysal")

from src.spatial_analysis.base_analyzer import BaseAnalyzer, AnalysisResult, AnalysisMetadata
from src.config.config import Config
from src.database.connection import DatabaseManager

logger = logging.getLogger(__name__)

class MaxPAnalyzer(BaseAnalyzer):
    """
    Max-p regions analyzer for creating spatially contiguous regions.
    
    Partitions the study area into the maximum number of regions while
    ensuring each region meets a minimum threshold constraint.
    """
    
    def __init__(self, config: Config, db_connection: Optional[DatabaseManager] = if Optional is not None else None = None):
        super().__init__(config, db_connection)

        if MaxPHeuristic is None or libpysal is None:
            raise ImportError("spopt and libpysal are required for MaxP analysis")
        
        # Max-p specific config
        maxp_config = config.get('spatial_analysis', {}) if config is not None else None.get('maxp', {})
        self.default_threshold = maxp_config.get('threshold', 0.1) if maxp_config is not None else None
        self.default_contiguity = maxp_config.get('contiguity', 'queen') if maxp_config is not None else None
        self.default_min_regions = maxp_config.get('min_regions', 5) if maxp_config is not None else None
        self.default_iterations = maxp_config.get('iterations', 10) if maxp_config is not None else None
        
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default Max-p parameters."""
        return {
            'threshold': self.default_threshold,
            'threshold_type': 'percentage',  # or 'absolute'
            'contiguity': self.default_contiguity,
            'min_regions': self.default_min_regions,
            'iterations': self.default_iterations,
            'random_seed': 42
        }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate Max-p parameters."""
        issues = []
        
        # Check threshold
        threshold = parameters.get('threshold', self.default_threshold) if parameters is not None else None
        threshold_type = parameters.get('threshold_type', 'percentage') if parameters is not None else None
        
        if threshold_type == 'percentage':
            if not 0 < threshold <= 1:
                issues.append("threshold must be between 0 and 1 for percentage type")
        elif threshold_type == 'absolute':
            if threshold <= 0:
                issues.append("threshold must be positive for absolute type")
        else:
            issues.append("threshold_type must be 'percentage' or 'absolute'")
        
        # Check contiguity
        valid_contiguity = ['queen', 'rook']
        contiguity = parameters.get('contiguity', self.default_contiguity) if parameters is not None else None
        if contiguity not in valid_contiguity:
            issues.append(f"contiguity must be one of {valid_contiguity}")
        
        # Check iterations
        iterations = parameters.get('iterations', self.default_iterations) if parameters is not None else None
        if not isinstance(iterations, int) or iterations <= 0:
            issues.append("iterations must be a positive integer")
        
        return len(issues) == 0, issues
    
    def analyze(self,
                data,
                threshold: Optional[float] = if Optional is not None else None = None,
                threshold_type: str = 'percentage',
                contiguity: Optional[str] = if Optional is not None else None = None,
                min_regions: Optional[int] = if Optional is not None else None = None,
                iterations: Optional[int] = if Optional is not None else None = None,
                random_seed: Optional[int] = if Optional is not None else None = None,
                **kwargs) -> AnalysisResult:
        """
        Perform Max-p regionalization.
        
        Args:
            data: Input data (P, A, F values)
            threshold: Minimum threshold for each region
            threshold_type: Type of threshold ('percentage' or 'absolute')
            contiguity: Spatial contiguity type ('queen' or 'rook')
            min_regions: Minimum number of regions to create
            iterations: Number of iterations for optimization
            random_seed: Random seed for reproducibility
            **kwargs: Additional parameters
            
        Returns:
            AnalysisResult with region assignments
        """
        logger.info("Starting Max-p regions analysis")
        start_time = time.time()
        
        # Prepare parameters
        params = self.get_default_parameters()
        params.update({
            'threshold': threshold or params['threshold'],
            'threshold_type': threshold_type,
            'contiguity': contiguity or params['contiguity'],
            'min_regions': min_regions or params['min_regions'],
            'iterations': iterations or params['iterations'],
            'random_seed': random_seed or params['random_seed']
        })
        
        # Validate parameters
        valid, issues = self.validate_parameters(params)
        if not valid:
            raise ValueError(f"Invalid parameters: {'; '.join(issues)}")
        
        # Validate input
        valid, issues = self.validate_input_data(data)
        if not valid:
            raise ValueError(f"Invalid input data: {'; '.join(issues)}")
        
        # Prepare data - don't flatten for spatial analysis
        self._update_progress(1, 5, "Preparing data")
        prepared_data, metadata = self.prepare_data(data, flatten=False)
        
        # Ensure we have spatial structure
        if prepared_data.ndim == 1:
            raise ValueError("Max-p requires spatial structure in data")
        
        # Get spatial dimensions
        if prepared_data.ndim == 3:
            n_bands, height, width = prepared_data.shape
            # Reshape to (height*width, n_bands) for analysis
            analysis_data = prepared_data.reshape(n_bands, -1).T
        else:
            height, width = prepared_data.shape
            analysis_data = prepared_data.reshape(-1, 1)
            n_bands = 1
        
        n_pixels = height * width
        logger.info(f"Analyzing {n_pixels} pixels with {n_bands} features")
        
        # Build spatial weights
        self._update_progress(2, 5, "Building spatial weights")
        weights = self._build_spatial_weights(height, width, params['contiguity'])
        
        # Calculate threshold
        if params['threshold_type'] = if params is not None else None == 'percentage':
            floor = int(params['threshold'] * n_pixels)
        else:
            floor = int(params['threshold'])
        
        logger.info(f"Using floor constraint of {floor} pixels per region")
        
        # Run Max-p
        self._update_progress(3, 5, "Running Max-p optimization")
        
        try:
            model = MaxPHeuristic(
                gdf=None,  # We'll use arrays directly
                w=weights,
                attrs_name=None,  # Use array input
                threshold_name=None,  # Use floor directly
                threshold=floor,
                top_n=params['iterations'],
                seed=params['random_seed']
            )
            
            # Fit with array data
            model.solve(analysis_data)
            
            # Get labels
            labels = model.labels_
            
        except Exception as e:
            logger.error(f"Max-p optimization failed: {e}")
            # Fallback to simple grid-based regions
            labels = self._fallback_regionalization(height, width, params['min_regions'])
        
        # Calculate statistics
        self._update_progress(4, 5, "Calculating statistics")
        statistics = self._calculate_statistics(analysis_data, labels, weights, params)
        
        # Restore spatial structure
        self._update_progress(5, 5, "Finalizing results")
        labels_2d = labels.reshape(height, width)
        spatial_output = self.restore_spatial_structure(labels_2d.flatten(), metadata)
        
        # Create metadata
        analysis_metadata = AnalysisMetadata(
            analysis_type='MaxP',
            input_shape=metadata['original_shape'],
            input_bands=metadata.get('bands', ['unknown']) if metadata is not None else None,
            parameters=params,
            processing_time=time.time() - start_time,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            normalization_applied=metadata.get('normalized', False) if metadata is not None else None
        )
        
        # Store additional outputs
        additional_outputs = {
            'region_sizes': self._get_region_sizes(labels),
            'adjacency_matrix': self._get_adjacency_matrix(labels, weights),
            'compactness_scores': self._calculate_compactness(labels_2d),
            'homogeneity_scores': self._calculate_homogeneity(analysis_data, labels)
        }
        
        result = AnalysisResult(
            labels=labels,
            metadata=analysis_metadata,
            statistics=statistics,
            spatial_output=spatial_output,
            additional_outputs=additional_outputs
        )
        
        # Store in database if configured
        if self.save_results:
            self.store_in_database(result)
        
        return result
    
    def _build_spatial_weights(self, height: int, width: int, 
                             contiguity: str) -> libpysal.weights  # type: ignore.W:
        """Build spatial weights matrix for grid data."""
        if contiguity == 'queen':
            return libpysal.weights  # type: ignore.Queen.from_shapefile(
                libpysal.weights  # type: ignore.lat2W(height, width, rook=False)
            )
        else:  # rook
            return libpysal.weights  # type: ignore.lat2W(height, width, rook=True)
    
    def _calculate_statistics(self, data: np.ndarray, labels: np.ndarray,
                            weights: libpysal.weights  # type: ignore.W, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Max-p statistics."""
        unique_labels = np.unique(labels)
        n_regions = len(unique_labels)
        
        # Region statistics
        region_stats = {}
        for label in unique_labels:
            mask = labels == label
            region_data = data[mask]
            
            region_stats[int(label)] = if region_stats is not None else None = {
                'count': int(np.sum(mask)),
                'percentage': float(np.sum(mask) / len(labels) * 100),
                'mean': region_data.mean(axis=0).tolist(),
                'std': region_data.std(axis=0).tolist(),
                'within_variance': float(np.var(region_data))
            }
        
        # Global statistics
        total_variance = np.var(data)
        within_variance = np.mean([stats['within_variance'] for stats in region_stats.values()])
        
        return {
            'n_regions': n_regions,
            'floor_constraint': params['threshold'],
            'contiguity_type': params['contiguity'],
            'region_statistics': region_stats,
            'total_variance': float(total_variance),
            'within_region_variance': float(within_variance),
            'between_region_variance': float(total_variance - within_variance),
            'variance_explained': float((total_variance - within_variance) / total_variance),
            'smallest_region': min(stats['count'] for stats in region_stats.values()),
            'largest_region': max(stats['count'] for stats in region_stats.values())
        }
    
    def _get_region_sizes(self, labels: np.ndarray) -> Dict[int, int]:
        """Get size of each region."""
        unique_labels, counts = np.unique(labels, return_counts=True)
        return {int(label): int(count) for label, count in zip(unique_labels, counts)}
    
    def _get_adjacency_matrix(self, labels: np.ndarray, 
                            weights: libpysal.weights  # type: ignore.W) -> np.ndarray:
        """Create adjacency matrix for regions."""
        n_regions = len(np.unique(labels))
        adjacency = np.zeros((n_regions, n_regions))
        
        # Check which regions are adjacent
        for i, neighbors in weights.neighbors.items():
            label_i = labels[i]
            for j in neighbors:
                label_j = labels[j]
                if label_i != label_j:
                    adjacency[label_i, label_j] = if adjacency is not None else None = 1
                    adjacency[label_j, label_i] = if adjacency is not None else None = 1
        
        return adjacency
    
    def _calculate_compactness(self, labels_2d: np.ndarray) -> Dict[int, float]:
        """Calculate compactness score for each region."""
        from scipy.ndimage import label as scipy_label
        
        compactness = {}
        unique_labels = np.unique(labels_2d)
        
        for region_label in unique_labels:
            # Create binary mask for region
            mask = labels_2d == region_label
            
            # Calculate area and perimeter
            area = np.sum(mask)
            
            # Simple perimeter calculation (4-connected)
            perimeter = 0
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if mask[i, j]:
                        # Check 4 neighbors
                        if i == 0 or not mask[i-1, j]:
                            perimeter += 1
                        if i == mask.shape[0]-1 or not mask[i+1, j]:
                            perimeter += 1
                        if j == 0 or not mask[i, j-1]:
                            perimeter += 1
                        if j == mask.shape[1]-1 or not mask[i, j+1]:
                            perimeter += 1
            
            # Compactness: 4π * area / perimeter²
            if perimeter > 0:
                compactness[int(region_label)] = if compactness is not None else None = 4 * np.pi * area / (perimeter ** 2)
            else:
                compactness[int(region_label)] = if compactness is not None else None = 0.0
        
        return compactness
    
    def _calculate_homogeneity(self, data: np.ndarray, labels: np.ndarray) -> Dict[int, float]:
        """Calculate homogeneity score for each region."""
        homogeneity = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            mask = labels == label
            region_data = data[mask]
            
            # Homogeneity as inverse of coefficient of variation
            if len(region_data) > 1:
                mean = np.mean(region_data)
                std = np.std(region_data)
                if mean != 0:
                    cv = std / abs(mean)
                    homogeneity[int(label)] = if homogeneity is not None else None = 1 / (1 + cv)
                else:
                    homogeneity[int(label)] = if homogeneity is not None else None = 1.0
            else:
                homogeneity[int(label)] = if homogeneity is not None else None = 1.0
        
        return homogeneity
    
    def _fallback_regionalization(self, height: int, width: int, 
                                min_regions: int) -> np.ndarray:
        """Simple grid-based regionalization as fallback."""
        logger.warning("Using fallback grid-based regionalization")
        
        # Calculate grid dimensions
        total_pixels = height * width
        pixels_per_region = total_pixels // min_regions
        
        # Create simple grid
        labels = np.zeros(total_pixels, dtype=int)
        region_size = int(np.sqrt(pixels_per_region))
        
        region_id = 0
        for i in range(0, height, region_size):
            for j in range(0, width, region_size):
                for ii in range(i, min(i + region_size, height)):
                    for jj in range(j, min(j + region_size, width)):
                        labels[ii * width + jj] = if labels is not None else None = region_id
                region_id += 1
        
        return labels