# src/spatial_analysis/maxp_regions/region_optimizer.py
"""Max-p regions optimizer for spatially constrained clustering."""

import logging
import time
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import pandas as pd
from spopt.region import MaxPHeuristic
import libpysal

from src.spatial_analysis.base_analyzer import BaseAnalyzer, AnalysisResult, AnalysisMetadata
from src.config import config
from src.database.connection import DatabaseManager

logger = logging.getLogger(__name__)

class MaxPAnalyzer(BaseAnalyzer):
    """
    Max-p regions analyzer for creating spatially contiguous regions.
    
    Partitions the study area into the maximum number of regions while
    ensuring each region meets a minimum area threshold.
    """
    
    def __init__(self, db_connection: Optional[DatabaseManager] = None):
        # Use global config instance
        super().__init__(config, db_connection)
        
        # Max-p specific config
        maxp_config = config.get('spatial_analysis', {}).get('maxp', {})
        
        # Default area thresholds for different ecological scales
        self.ecological_scales = {
            'biome': 50000,      # > 50,000 km²
            'ecoregion': 5000,   # 1,000 - 10,000 km²
            'landscape': 500     # 100 - 500 km²
        }
        
        # Grid resolution from config or default
        self.pixel_size_km = maxp_config.get('pixel_size_km', 1.8)
        self.pixel_area_km2 = self.pixel_size_km ** 2  # 3.24 km² for 1.8km pixels
        
        # Default parameters
        self.default_min_area_km2 = maxp_config.get('min_area_km2', 2500)  # Default ecoregion scale
        self.default_ecological_scale = maxp_config.get('ecological_scale', 'ecoregion')
        self.default_contiguity = maxp_config.get('contiguity', 'queen')
        self.default_min_regions = maxp_config.get('min_regions', 5)
        self.default_iterations = maxp_config.get('iterations', 10)
        
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default Max-p parameters."""
        return {
            'min_area_km2': self.default_min_area_km2,
            'ecological_scale': self.default_ecological_scale,
            'contiguity': self.default_contiguity,
            'min_regions': self.default_min_regions,
            'iterations': self.default_iterations,
            'random_seed': 42,
            'perturbation_range': [0.5, 1.0, 2.0]  # Multipliers for perturbation analysis
        }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate Max-p parameters."""
        issues = []
        
        # Check area threshold
        min_area = parameters.get('min_area_km2', self.default_min_area_km2)
        if not isinstance(min_area, (int, float)) or min_area <= 0:
            issues.append("min_area_km2 must be a positive number")
        
        # Check if area is reasonable
        min_pixels = min_area / self.pixel_area_km2
        if min_pixels < 4:
            issues.append(f"min_area_km2 ({min_area}) is too small - less than 4 pixels (pixel_area={self.pixel_area_km2:.2f} km², min_pixels={min_pixels:.1f})")
        
        # Check ecological scale
        valid_scales = list(self.ecological_scales.keys()) + ['custom']
        scale = parameters.get('ecological_scale', self.default_ecological_scale)
        if scale not in valid_scales:
            issues.append(f"ecological_scale must be one of {valid_scales}")
        
        # Check contiguity
        valid_contiguity = ['queen', 'rook']
        contiguity = parameters.get('contiguity', self.default_contiguity)
        if contiguity not in valid_contiguity:
            issues.append(f"contiguity must be one of {valid_contiguity}")
        
        # Check iterations
        iterations = parameters.get('iterations', self.default_iterations)
        if not isinstance(iterations, int) or iterations <= 0:
            issues.append("iterations must be a positive integer")
        
        # Check perturbation range
        pert_range = parameters.get('perturbation_range', [0.5, 1.0, 2.0])
        if not isinstance(pert_range, list) or not all(isinstance(x, (int, float)) and x > 0 for x in pert_range):
            issues.append("perturbation_range must be a list of positive numbers")
        
        return len(issues) == 0, issues
    
    def analyze(self,
                data,
                min_area_km2: Optional[float] = None,
                ecological_scale: Optional[str] = None,
                contiguity: Optional[str] = None,
                min_regions: Optional[int] = None,
                iterations: Optional[int] = None,
                random_seed: Optional[int] = None,
                run_perturbation: bool = True,
                perturbation_range: Optional[List[float]] = None,
                **kwargs) -> AnalysisResult:
        """
        Perform Max-p regionalization.
        
        Args:
            data: Input data (P, A, F values)
            min_area_km2: Minimum area threshold in square kilometers
            ecological_scale: Preset scale ('biome', 'ecoregion', 'landscape', 'custom')
            contiguity: Spatial contiguity type ('queen' or 'rook')
            min_regions: Minimum number of regions to create
            iterations: Number of iterations for optimization
            random_seed: Random seed for reproducibility
            run_perturbation: Whether to run perturbation analysis
            perturbation_range: Multipliers for perturbation analysis
            **kwargs: Additional parameters
            
        Returns:
            AnalysisResult with region assignments
        """
        logger.info("Starting Max-p regions analysis")
        start_time = time.time()
        
        # Prepare parameters
        params = self.get_default_parameters()
        
        # Handle ecological scale
        if ecological_scale and ecological_scale != 'custom':
            if ecological_scale in self.ecological_scales:
                min_area_km2 = self.ecological_scales[ecological_scale]
                logger.info(f"Using {ecological_scale} scale: {min_area_km2} km²")
        
        params.update({
            'min_area_km2': min_area_km2 or params['min_area_km2'],
            'ecological_scale': ecological_scale or params['ecological_scale'],
            'contiguity': contiguity or params['contiguity'],
            'min_regions': min_regions or params['min_regions'],
            'iterations': iterations or params['iterations'],
            'random_seed': random_seed or params['random_seed'],
            'perturbation_range': perturbation_range or params['perturbation_range']
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
        self._update_progress(1, 6, "Preparing data")
        prepared_data, metadata = self.prepare_data(data, flatten=False)
        
        # Get spatial dimensions
        if prepared_data.ndim == 3:
            n_bands, height, width = prepared_data.shape
            analysis_data = prepared_data.reshape(n_bands, -1).T
        else:
            height, width = prepared_data.shape
            analysis_data = prepared_data.reshape(-1, 1)
            n_bands = 1
        
        n_pixels = height * width
        total_area_km2 = n_pixels * self.pixel_area_km2
        
        logger.info(f"Analyzing {n_pixels} pixels ({total_area_km2:.1f} km²) with {n_bands} features")
        
        # Calculate floor constraint in pixels
        floor_pixels = int(params['min_area_km2'] / self.pixel_area_km2)
        logger.info(f"Minimum region size: {params['min_area_km2']} km² = {floor_pixels} pixels")
        
        # Check if threshold is feasible
        max_possible_regions = n_pixels // floor_pixels
        if max_possible_regions < params['min_regions']:
            logger.warning(f"Cannot create {params['min_regions']} regions with "
                         f"{params['min_area_km2']} km² threshold. "
                         f"Maximum possible: {max_possible_regions}")
        
        # Build spatial weights
        self._update_progress(2, 6, "Building spatial weights")
        weights = self._build_spatial_weights(height, width, params['contiguity'])
        
        # Run main Max-p analysis
        self._update_progress(3, 6, "Running Max-p optimization")
        main_labels = self._run_maxp(
            analysis_data, weights, floor_pixels, params
        )
        
        # Run perturbation analysis if requested
        perturbation_results = None
        if run_perturbation:
            self._update_progress(4, 6, "Running perturbation analysis")
            perturbation_results = self._run_perturbation_analysis(
                analysis_data, weights, params, main_labels
            )
        
        # Calculate statistics
        self._update_progress(5, 6, "Calculating statistics")
        statistics = self._calculate_statistics(
            analysis_data, main_labels, weights, params, floor_pixels
        )
        
        # Restore spatial structure
        self._update_progress(6, 6, "Finalizing results")
        labels_2d = main_labels.reshape(height, width)
        spatial_output = self.restore_spatial_structure(labels_2d.flatten(), metadata)
        
        # Create metadata
        analysis_metadata = AnalysisMetadata(
            analysis_type='MaxP',
            input_shape=metadata['original_shape'],
            input_bands=metadata.get('bands', ['unknown']),
            parameters=params,
            processing_time=time.time() - start_time,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            normalization_applied=metadata.get('normalized', False),
            data_source=f"Grid resolution: {self.pixel_size_km}km"
        )
        
        # Store additional outputs
        additional_outputs = {
            'region_sizes_km2': self._get_region_areas(main_labels),
            'adjacency_matrix': self._get_adjacency_matrix(main_labels, weights),
            'compactness_scores': self._calculate_compactness(labels_2d),
            'homogeneity_scores': self._calculate_homogeneity(analysis_data, main_labels),
            'perturbation_results': perturbation_results,
            'total_area_km2': total_area_km2,
            'pixel_area_km2': self.pixel_area_km2
        }
        
        result = AnalysisResult(
            labels=main_labels,
            metadata=analysis_metadata,
            statistics=statistics,
            spatial_output=spatial_output,
            additional_outputs=additional_outputs
        )
        
        # Store in database if configured
        if self.save_results_enabled:
            self.store_in_database(result)
        
        return result
    
    def _run_maxp(self, data: np.ndarray, weights: libpysal.weights.W,
                  floor_pixels: int, params: Dict[str, Any]) -> np.ndarray:
        """Run Max-p algorithm with error handling."""
        try:
            model = MaxPHeuristic(
                gdf=None,
                w=weights,
                attrs_name=None,
                threshold_name=None,
                threshold=floor_pixels,
                top_n=params['iterations'],
                verbose=False
            )
            
            model.solve()
            if hasattr(model, 'labels_') and model.labels_ is not None:
                return np.array(model.labels_)
            else:
                raise ValueError("MaxP solver did not produce valid labels")
            
        except Exception as e:
            logger.error(f"Max-p optimization failed: {e}")
            logger.warning("Using fallback grid-based regionalization")
            
            # Calculate grid size based on floor constraint
            pixels_per_side = int(np.sqrt(floor_pixels))
            return self._fallback_regionalization(
                int(np.sqrt(len(data))), 
                int(np.sqrt(len(data))),
                pixels_per_side
            )
    
    def _run_perturbation_analysis(self, data: np.ndarray, weights: libpysal.weights.W,
                                  params: Dict[str, Any], 
                                  baseline_labels: np.ndarray) -> Dict[str, Any]:
        """Run perturbation analysis to test stability."""
        base_area = params['min_area_km2']
        perturbation_range = params['perturbation_range']
        
        results = {
            'baseline_area_km2': base_area,
            'tested_areas_km2': [],
            'boundary_stability': [],
            'n_regions': [],
            'region_correspondence': []
        }
        
        for multiplier in perturbation_range:
            test_area = base_area * multiplier
            test_floor = int(test_area / self.pixel_area_km2)
            
            logger.info(f"Testing perturbation: {test_area:.0f} km² (×{multiplier})")
            
            # Run Max-p with perturbed threshold
            test_labels = self._run_maxp(
                data, weights, test_floor, 
                {**params, 'random_seed': params['random_seed'] + int(multiplier * 10)}
            )
            
            # Calculate stability metrics
            stability = self._calculate_boundary_stability(baseline_labels, test_labels)
            correspondence = self._calculate_region_correspondence(baseline_labels, test_labels)
            
            results['tested_areas_km2'].append(test_area)
            results['boundary_stability'].append(stability)
            results['n_regions'].append(len(np.unique(test_labels)))
            results['region_correspondence'].append(correspondence)
        
        # Overall stability assessment
        avg_stability = np.mean(results['boundary_stability'])
        if avg_stability > 0.8:
            results['stability_assessment'] = "High - regions are robust to threshold changes"
        elif avg_stability > 0.6:
            results['stability_assessment'] = "Moderate - core regions stable, boundaries variable"
        else:
            results['stability_assessment'] = "Low - regions sensitive to threshold selection"
        
        return results
    
    def _calculate_boundary_stability(self, labels1: np.ndarray, 
                                    labels2: np.ndarray) -> float:
        """Calculate stability of region boundaries between two labelings."""
        # Find pixels that maintain same neighbors
        stable_boundaries = 0
        total_boundaries = 0
        
        height = int(np.sqrt(len(labels1)))
        width = height
        
        labels1_2d = labels1.reshape(height, width)
        labels2_2d = labels2.reshape(height, width)
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                # Check if pixel is on boundary in labeling 1
                center = labels1_2d[i, j]
                neighbors1 = [
                    labels1_2d[i-1, j], labels1_2d[i+1, j],
                    labels1_2d[i, j-1], labels1_2d[i, j+1]
                ]
                
                if any(n != center for n in neighbors1):
                    total_boundaries += 1
                    
                    # Check if still on boundary in labeling 2
                    center2 = labels2_2d[i, j]
                    neighbors2 = [
                        labels2_2d[i-1, j], labels2_2d[i+1, j],
                        labels2_2d[i, j-1], labels2_2d[i, j+1]
                    ]
                    
                    if any(n != center2 for n in neighbors2):
                        stable_boundaries += 1
        
        return stable_boundaries / total_boundaries if total_boundaries > 0 else 1.0
    
    def _calculate_region_correspondence(self, labels1: np.ndarray, 
                                       labels2: np.ndarray) -> float:
        """Calculate correspondence between two region sets using Rand index."""
        from sklearn.metrics import rand_score
        return float(rand_score(labels1, labels2))
    
    def _calculate_statistics(self, data: np.ndarray, labels: np.ndarray,
                            weights: libpysal.weights.W, params: Dict[str, Any],
                            floor_pixels: int) -> Dict[str, Any]:
        """Calculate Max-p statistics with area information."""
        unique_labels = np.unique(labels)
        n_regions = len(unique_labels)
        
        # Region statistics
        region_stats = {}
        for label in unique_labels:
            mask = labels == label
            region_data = data[mask]
            pixel_count = int(np.sum(mask))
            area_km2 = pixel_count * self.pixel_area_km2
            
            region_stats[int(label)] = {
                'pixel_count': pixel_count,
                'area_km2': area_km2,
                'percentage_of_total': float(pixel_count / len(labels) * 100),
                'mean': region_data.mean(axis=0).tolist(),
                'std': region_data.std(axis=0).tolist(),
                'within_variance': float(np.var(region_data))
            }
        
        # Global statistics
        total_variance = np.var(data)
        within_variance = np.mean([stats['within_variance'] for stats in region_stats.values()])
        
        # Area-based statistics
        areas = [stats['area_km2'] for stats in region_stats.values()]
        
        return {
            'n_regions': n_regions,
            'min_area_threshold_km2': params['min_area_km2'],
            'floor_pixels': floor_pixels,
            'ecological_scale': params['ecological_scale'],
            'contiguity_type': params['contiguity'],
            'region_statistics': region_stats,
            'total_variance': float(total_variance),
            'within_region_variance': float(within_variance),
            'between_region_variance': float(total_variance - within_variance),
            'variance_explained': float((total_variance - within_variance) / total_variance),
            'smallest_region_km2': min(areas),
            'largest_region_km2': max(areas),
            'mean_region_area_km2': np.mean(areas),
            'area_coefficient_variation': np.std(areas) / np.mean(areas),
            'threshold_satisfied': min(areas) >= params['min_area_km2']
        }
    
    def _get_region_areas(self, labels: np.ndarray) -> Dict[int, float]:
        """Get area in km² for each region."""
        unique_labels, counts = np.unique(labels, return_counts=True)
        return {
            int(label): float(count * self.pixel_area_km2) 
            for label, count in zip(unique_labels, counts)
        }
    
    def _build_spatial_weights(self, height: int, width: int, 
                             contiguity: str) -> libpysal.weights.W:
        """Build spatial weights matrix for grid data."""
        if contiguity == 'queen':
            return libpysal.weights.lat2W(height, width, rook=False)
        else:  # rook
            return libpysal.weights.lat2W(height, width, rook=True)
    
    def _get_adjacency_matrix(self, labels: np.ndarray, 
                            weights: libpysal.weights.W) -> np.ndarray:
        """Create adjacency matrix for regions."""
        n_regions = len(np.unique(labels))
        adjacency = np.zeros((n_regions, n_regions))
        
        for i, neighbors in weights.neighbors.items():
            label_i = labels[i]
            for j in neighbors:
                label_j = labels[j]
                if label_i != label_j:
                    adjacency[label_i, label_j] = 1
                    adjacency[label_j, label_i] = 1
        
        return adjacency
    
    def _calculate_compactness(self, labels_2d: np.ndarray) -> Dict[int, float]:
        """Calculate compactness score for each region."""
        compactness = {}
        unique_labels = np.unique(labels_2d)
        
        for region_label in unique_labels:
            mask = labels_2d == region_label
            area = np.sum(mask)
            
            # Simple perimeter calculation (4-connected)
            perimeter = 0
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if mask[i, j]:
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
                compactness[int(region_label)] = 4 * np.pi * area / (perimeter ** 2)
            else:
                compactness[int(region_label)] = 0.0
        
        return compactness
    
    def _calculate_homogeneity(self, data: np.ndarray, labels: np.ndarray) -> Dict[int, float]:
        """Calculate homogeneity score for each region."""
        homogeneity = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            mask = labels == label
            region_data = data[mask]
            
            if len(region_data) > 1:
                mean = np.mean(region_data)
                std = np.std(region_data)
                if mean != 0:
                    cv = std / abs(mean)
                    homogeneity[int(label)] = 1 / (1 + cv)
                else:
                    homogeneity[int(label)] = 1.0
            else:
                homogeneity[int(label)] = 1.0
        
        return homogeneity
    
    def _fallback_regionalization(self, height: int, width: int, 
                                grid_size: int) -> np.ndarray:
        """Simple grid-based regionalization as fallback."""
        logger.warning("Using fallback grid-based regionalization")
        
        labels = np.zeros(height * width, dtype=int)
        region_id = 0
        
        for i in range(0, height, grid_size):
            for j in range(0, width, grid_size):
                for ii in range(i, min(i + grid_size, height)):
                    for jj in range(j, min(j + grid_size, width)):
                        labels[ii * width + jj] = region_id
                region_id += 1
        
        return labels