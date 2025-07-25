# src/spatial_analysis/gwpca/gwpca_analyzer.py
"""Geographically Weighted Principal Component Analysis."""

import logging
import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.spatial_analysis.base_analyzer import BaseAnalyzer, AnalysisResult, AnalysisMetadata
from src.config import config
from src.database.connection import DatabaseManager

logger = logging.getLogger(__name__)

class GWPCAAnalyzer(BaseAnalyzer):
    """
    Geographically Weighted PCA analyzer.
    
    Explores how the structure of biodiversity relationships varies
    across geographic space.
    """
    
    def __init__(self, config: Config, db_connection: Optional[DatabaseManager] = None):
        super().__init__(config, db_connection)
        
        # GWPCA specific config
        gwpca_config = config.get('spatial_analysis', {}).get('gwpca', {})
        self.default_bandwidth_method = gwpca_config.get('bandwidth_method', 'cv')
        self.default_adaptive = gwpca_config.get('adaptive', True)
        self.default_n_components = gwpca_config.get('n_components', 2)
        self.default_kernel = gwpca_config.get('kernel', 'bisquare')
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default GWPCA parameters."""
        return {
            'bandwidth': None,  # Auto-select if None
            'bandwidth_method': self.default_bandwidth_method,
            'adaptive': self.default_adaptive,
            'n_components': self.default_n_components,
            'kernel': self.default_kernel,
            'standardize': True
        }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate GWPCA parameters."""
        issues = []
        
        # Check bandwidth
        bandwidth = parameters.get('bandwidth')
        if bandwidth is not None:
            if parameters.get('adaptive', True):
                if not isinstance(bandwidth, int) or bandwidth <= 0:
                    issues.append("adaptive bandwidth must be a positive integer")
            else:
                if not isinstance(bandwidth, (int, float)) or bandwidth <= 0:
                    issues.append("fixed bandwidth must be a positive number")
        
        # Check bandwidth method
        valid_methods = ['cv', 'aic', 'bic']
        method = parameters.get('bandwidth_method', self.default_bandwidth_method)
        if method not in valid_methods:
            issues.append(f"bandwidth_method must be one of {valid_methods}")
        
        # Check kernel
        valid_kernels = ['gaussian', 'bisquare', 'exponential']
        kernel = parameters.get('kernel', self.default_kernel)
        if kernel not in valid_kernels:
            issues.append(f"kernel must be one of {valid_kernels}")
        
        # Check n_components
        n_comp = parameters.get('n_components', self.default_n_components)
        if not isinstance(n_comp, int) or n_comp <= 0:
            issues.append("n_components must be a positive integer")
        
        return len(issues) == 0, issues
    
    def analyze(self,
                data,
                bandwidth: Optional[float] = None,
                bandwidth_method: Optional[str] = None,
                adaptive: Optional[bool] = None,
                n_components: Optional[int] = None,
                kernel: Optional[str] = None,
                standardize: bool = True,
                **kwargs) -> AnalysisResult:
        """
        Perform GWPCA analysis.
        
        Args:
            data: Input data (P, A, F values)
            bandwidth: Spatial bandwidth (auto-selected if None)
            bandwidth_method: Method for bandwidth selection
            adaptive: Whether to use adaptive bandwidth
            n_components: Number of components to compute
            kernel: Spatial kernel type
            standardize: Whether to standardize data
            **kwargs: Additional parameters
            
        Returns:
            AnalysisResult with local PCA results
        """
        logger.info("Starting GWPCA analysis")
        start_time = time.time()
        
        # Prepare parameters
        params = self.get_default_parameters()
        params.update({
            'bandwidth': bandwidth,
            'bandwidth_method': bandwidth_method or params['bandwidth_method'],
            'adaptive': adaptive if adaptive is not None else params['adaptive'],
            'n_components': n_components or params['n_components'],
            'kernel': kernel or params['kernel'],
            'standardize': standardize
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
        prepared_data, metadata = self.prepare_data(data, flatten=False, normalize=False)
        
        # Get spatial structure
        if prepared_data.ndim == 3:
            n_bands, height, width = prepared_data.shape
            # Reshape to (n_pixels, n_bands)
            X = prepared_data.reshape(n_bands, -1).T
        else:
            height, width = prepared_data.shape
            X = prepared_data.reshape(-1, 1)
            n_bands = 1
        
        n_pixels = height * width
        
        # Create coordinate arrays
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        coords = np.column_stack([x_coords.ravel(), y_coords.ravel()])
        
        logger.info(f"Analyzing {n_pixels} pixels with {n_bands} features")
        
        # Standardize if requested
        if params['standardize']:
            self._update_progress(2, 6, "Standardizing data")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Select bandwidth if not provided
        if params['bandwidth'] is None:
            self._update_progress(3, 6, "Selecting optimal bandwidth")
            params['bandwidth'] = self._select_bandwidth(
                X_scaled, coords, params
            )
            logger.info(f"Selected bandwidth: {params['bandwidth']}")
        
        # Perform GWPCA
        self._update_progress(4, 6, "Computing local PCAs")
        local_results = self._compute_gwpca(
            X_scaled, coords, params
        )
        
        # Calculate statistics
        self._update_progress(5, 6, "Calculating statistics")
        statistics = self._calculate_statistics(
            X_scaled, local_results, params
        )
        
        # Create spatial outputs
        self._update_progress(6, 6, "Creating spatial outputs")
        spatial_outputs = self._create_spatial_outputs(
            local_results, height, width, metadata
        )
        
        # Create metadata
        analysis_metadata = AnalysisMetadata(
            analysis_type='GWPCA',
            input_shape=metadata['original_shape'],
            input_bands=metadata.get('bands', ['unknown']),
            parameters=params,
            processing_time=time.time() - start_time,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            normalization_applied=params['standardize']
        )
        
        # Primary output is local R²
        labels = local_results['local_r2']
        
        result = AnalysisResult(
            labels=labels,
            metadata=analysis_metadata,
            statistics=statistics,
            spatial_output=spatial_outputs['local_r2_map'],
            additional_outputs=spatial_outputs
        )
        
        # Store in database if configured
        if self.save_results_enabled:
            self.store_in_database(result)
        
        return result
    
    def _select_bandwidth(self, X: np.ndarray, coords: np.ndarray,
                        params: Dict[str, Any]) -> float:
        """Select optimal bandwidth using cross-validation."""
        from mgwr.sel_bw import Sel_BW
        
        # Use first PC for bandwidth selection (faster)
        pca_global = PCA(n_components=1)
        pc1 = pca_global.fit_transform(X)
        
        # Create a simple y variable (PC1) for bandwidth selection
        selector = Sel_BW(coords, pc1, X)
        
        if params['adaptive']:
            # Search adaptive bandwidth (number of neighbors)
            bw = selector.search(
                criterion=params['bandwidth_method'],
                search_method='golden_section'
            )
        else:
            # Search fixed bandwidth (distance)
            bw = selector.search(
                criterion=params['bandwidth_method'],
                search_method='golden_section'
            )
        
        return float(bw) if bw is not None else 50.0
    
    def _compute_gwpca(self, X: np.ndarray, coords: np.ndarray,
                     params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Compute GWPCA at each location."""
        n_pixels, n_features = X.shape
        n_components = min(params['n_components'], n_features)
        
        # Initialize output arrays
        local_r2 = np.zeros(n_pixels)
        local_eigenvalues = np.zeros((n_pixels, n_components))
        local_loadings = np.zeros((n_pixels, n_features, n_components))
        local_scores = np.zeros((n_pixels, n_components))
        
        # Compute weights for each location
        # Custom kernel functions (replacing mgwr imports)
        def gaussian_weights(dist, bandwidth):
            """Gaussian kernel function."""
            return np.exp(-0.5 * (dist / bandwidth) ** 2)
        
        def bisquare_weights(dist, bandwidth):
            """Bisquare kernel function."""
            weights = np.zeros_like(dist)
            valid = dist <= bandwidth
            if np.any(valid):
                weights[valid] = (1 - (dist[valid] / bandwidth) ** 2) ** 2
            return weights
        
        # Select kernel function
        if params['kernel'] == 'gaussian':
            kernel_func = gaussian_weights
        elif params['kernel'] == 'bisquare':
            kernel_func = bisquare_weights
        else:  # exponential - use gaussian as fallback
            kernel_func = gaussian_weights
        
        bandwidth = params['bandwidth']
        
        # Process each location
        for i in range(n_pixels):
            if i % 1000 == 0:
                progress = 3 + (i / n_pixels) * 1
                self._update_progress(progress, 6, f"Processing pixel {i}/{n_pixels}")
            
            # Calculate weights
            if params['adaptive']:
                # Adaptive: find k nearest neighbors
                distances = np.sqrt(np.sum((coords - coords[i]) ** 2, axis=1))
                sorted_indices = np.argsort(distances)
                bw_i = distances[sorted_indices[int(bandwidth)]]
                weights = kernel_func(distances, bw_i)
            else:
                # Fixed bandwidth
                distances = np.sqrt(np.sum((coords - coords[i]) ** 2, axis=1))
                weights = kernel_func(distances, bandwidth)
            
            # Weight threshold
            weight_threshold = 1e-5
            valid_weights = weights > weight_threshold
            
            if np.sum(valid_weights) < n_features + 1:
                # Not enough neighbors, use global PCA
                weights = np.ones(n_pixels) / n_pixels
                valid_weights = np.ones(n_pixels, dtype=bool)
            
            # Weighted PCA
            X_weighted = X[valid_weights] * np.sqrt(weights[valid_weights, np.newaxis])
            
            if len(X_weighted) > n_features:
                pca = PCA(n_components=n_components)
                pca.fit(X_weighted)
                
                # Store results
                local_eigenvalues[i] = pca.explained_variance_
                local_loadings[i] = pca.components_.T
                local_r2[i] = pca.explained_variance_ratio_[0]
                
                # Transform the point itself
                local_scores[i] = pca.transform(X[i:i+1])[0]
            else:
                # Fallback values
                local_r2[i] = 0
                local_eigenvalues[i] = 0
                local_loadings[i] = 0
        
        return {
            'local_r2': local_r2,
            'local_eigenvalues': local_eigenvalues,
            'local_loadings': local_loadings,
            'local_scores': local_scores,
            'bandwidth_used': bandwidth
        }
    
    def _calculate_statistics(self, X: np.ndarray, 
                            local_results: Dict[str, np.ndarray],
                            params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate GWPCA statistics."""
        # Global PCA for comparison
        pca_global = PCA(n_components=params['n_components'])
        pca_global.fit(X)
        
        # Local statistics
        local_r2 = local_results['local_r2']
        local_eigenvalues = local_results['local_eigenvalues']  # Will be used for enhanced statistics
        
        # Calculate variation in local structure
        loading_variation = np.std(local_results['local_loadings'], axis=0)
        
        return {
            'global_variance_explained': pca_global.explained_variance_ratio_.tolist(),
            'global_eigenvalues': pca_global.explained_variance_.tolist(),
            'global_loadings': pca_global.components_.tolist(),
            'local_r2_mean': float(np.mean(local_r2)),
            'local_r2_std': float(np.std(local_r2)),
            'local_r2_min': float(np.min(local_r2)),
            'local_r2_max': float(np.max(local_r2)),
            'bandwidth': local_results['bandwidth_used'],
            'loading_variation': loading_variation.tolist(),
            'n_components': params['n_components'],
            'spatial_heterogeneity': float(np.std(local_r2) / np.mean(local_r2))
        }
    
    def _create_spatial_outputs(self, local_results: Dict[str, np.ndarray],
                              height: int, width: int,
                              metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create spatial output arrays."""
        import xarray as xr
        
        # Create coordinate arrays
        if 'coords' in metadata and metadata['coords']:
            # Use original coordinates if available
            lat_coords = metadata['coords'].get('lat', range(height))
            lon_coords = metadata['coords'].get('lon', range(width))
        else:
            lat_coords = range(height)
            lon_coords = range(width)
        
        outputs = {}
        
        # Local R² map
        r2_2d = local_results['local_r2'].reshape(height, width)
        outputs['local_r2_map'] = xr.DataArray(
            r2_2d,
            coords={'lat': lat_coords, 'lon': lon_coords},
            dims=['lat', 'lon'],
            name='local_r2',
            attrs={'description': 'Local R² from first principal component'}
        )
        
        # Local eigenvalue maps
        n_components = local_results['local_eigenvalues'].shape[1]
        for i in range(n_components):
            eigenval_2d = local_results['local_eigenvalues'][:, i].reshape(height, width)
            outputs[f'eigenvalue_pc{i+1}'] = xr.DataArray(
                eigenval_2d,
                coords={'lat': lat_coords, 'lon': lon_coords},
                dims=['lat', 'lon'],
                name=f'eigenvalue_pc{i+1}',
                attrs={'description': f'Local eigenvalue for PC{i+1}'}
            )
        
        # Local loading maps for each variable and component
        band_names = metadata.get('bands', [f'var_{i}' for i in range(local_results['local_loadings'].shape[1])])
        
        for var_idx, var_name in enumerate(band_names[:local_results['local_loadings'].shape[1]]):
            for comp_idx in range(n_components):
                loading_2d = local_results['local_loadings'][:, var_idx, comp_idx].reshape(height, width)
                outputs[f'loading_{var_name}_pc{comp_idx+1}'] = xr.DataArray(
                    loading_2d,
                    coords={'lat': lat_coords, 'lon': lon_coords},
                    dims=['lat', 'lon'],
                    name=f'loading_{var_name}_pc{comp_idx+1}',
                    attrs={'description': f'Local loading of {var_name} on PC{comp_idx+1}'}
                )
        
        return outputs