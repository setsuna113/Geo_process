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
import xarray as xr

from src.spatial_analysis.base_analyzer import BaseAnalyzer, AnalysisResult, AnalysisMetadata
from src.config.config import Config
from src.database.connection import DatabaseManager

logger = logging.getLogger(__name__)

class GWPCAAnalyzer(BaseAnalyzer):
    """
    Geographically Weighted PCA analyzer with block aggregation support.
    
    Explores how the structure of biodiversity relationships varies
    across geographic space using aggregated blocks for computational efficiency.
    """
    
    def __init__(self, config: Config, db_connection: Optional[DatabaseManager] = None):
        super().__init__(config, db_connection)
        
        # GWPCA specific config
        gwpca_config = config.get('spatial_analysis', {}).get('gwpca', {})
        self.default_bandwidth_method = gwpca_config.get('bandwidth_method', 'cv')
        self.default_adaptive = gwpca_config.get('adaptive', True)
        self.default_n_components = gwpca_config.get('n_components', 2)
        self.default_kernel = gwpca_config.get('kernel', 'bisquare')
        
        # Block aggregation parameters
        self.default_block_size_km = gwpca_config.get('block_size_km', 50)
        self.pixel_size_km = gwpca_config.get('pixel_size_km', 1.8)
        self.use_block_aggregation = gwpca_config.get('use_block_aggregation', True)
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default GWPCA parameters."""
        return {
            'bandwidth': None,  # Auto-select if None
            'bandwidth_method': self.default_bandwidth_method,
            'adaptive': self.default_adaptive,
            'n_components': self.default_n_components,
            'kernel': self.default_kernel,
            'standardize': True,
            'block_size_km': self.default_block_size_km,
            'use_block_aggregation': self.use_block_aggregation,
            'aggregation_method': 'mean'  # or 'median', 'sum'
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
        
        # Check block size
        block_size = parameters.get('block_size_km', self.default_block_size_km)
        if not isinstance(block_size, (int, float)) or block_size <= 0:
            issues.append("block_size_km must be a positive number")
        
        # Check aggregation method
        valid_agg = ['mean', 'median', 'sum']
        agg_method = parameters.get('aggregation_method', 'mean')
        if agg_method not in valid_agg:
            issues.append(f"aggregation_method must be one of {valid_agg}")
        
        return len(issues) == 0, issues
    
    def analyze(self,
                data,
                bandwidth: Optional[float] = None,
                bandwidth_method: Optional[str] = None,
                adaptive: Optional[bool] = None,
                n_components: Optional[int] = None,
                kernel: Optional[str] = None,
                standardize: bool = True,
                block_size_km: Optional[float] = None,
                use_block_aggregation: Optional[bool] = None,
                aggregation_method: str = 'mean',
                **kwargs) -> AnalysisResult:
        """
        Perform GWPCA analysis with optional block aggregation.
        
        Args:
            data: Input data (P, A, F values)
            bandwidth: Spatial bandwidth (auto-selected if None)
            bandwidth_method: Method for bandwidth selection
            adaptive: Whether to use adaptive bandwidth
            n_components: Number of components to compute
            kernel: Spatial kernel type
            standardize: Whether to standardize data
            block_size_km: Size of aggregation blocks in kilometers
            use_block_aggregation: Whether to aggregate to blocks
            aggregation_method: Method for aggregating pixels to blocks
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
            'standardize': standardize,
            'block_size_km': block_size_km or params['block_size_km'],
            'use_block_aggregation': use_block_aggregation if use_block_aggregation is not None else params['use_block_aggregation'],
            'aggregation_method': aggregation_method
        })
        
        # Validate parameters
        valid, issues = self.validate_parameters(params)
        if not valid:
            raise ValueError(f"Invalid parameters: {'; '.join(issues)}")
        
        # Validate input
        valid, issues = self.validate_input_data(data)
        if not valid:
            raise ValueError(f"Invalid input data: {'; '.join(issues)}")
        
        # Prepare data
        self._update_progress(1, 7, "Preparing data")
        prepared_data, metadata = self.prepare_data(data, flatten=False, normalize=False)
        
        # Add n_pixels for compatibility
        if 'spatial_shape' in metadata:
            metadata['n_pixels'] = np.prod(metadata['spatial_shape'][-2:])  # Last 2 dims are spatial
        elif prepared_data.ndim >= 2:
            metadata['n_pixels'] = np.prod(prepared_data.shape[-2:])  # Last 2 dims are spatial
        
        # Apply block aggregation if requested
        if params['use_block_aggregation']:
            self._update_progress(2, 7, "Aggregating to blocks")
            block_data, block_metadata = self._aggregate_to_blocks(
                prepared_data, metadata, params
            )
            analysis_data = block_data
            analysis_metadata = block_metadata
            logger.info(f"Aggregated to {block_metadata['n_blocks']} blocks from {metadata['n_pixels']} pixels")
        else:
            analysis_data = prepared_data
            analysis_metadata = metadata
        
        # Get spatial structure
        if analysis_data.ndim == 3:
            n_bands, height, width = analysis_data.shape
            X = analysis_data.reshape(n_bands, -1).T
        else:
            height, width = analysis_data.shape
            X = analysis_data.reshape(-1, 1)
            n_bands = 1
        
        n_samples = height * width
        
        # Create coordinate arrays for analysis
        if params['use_block_aggregation']:
            coords = analysis_metadata['block_coords']
        else:
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            coords = np.column_stack([x_coords.ravel(), y_coords.ravel()])
        
        logger.info(f"Analyzing {n_samples} spatial units with {n_bands} features")
        
        # Standardize if requested
        if params['standardize']:
            self._update_progress(3, 7, "Standardizing data")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X
        
        # Select bandwidth if not provided
        if params['bandwidth'] is None:
            self._update_progress(4, 7, "Selecting optimal bandwidth")
            params['bandwidth'] = self._select_bandwidth(
                X_scaled, coords, params
            )
            logger.info(f"Selected bandwidth: {params['bandwidth']}")
        
        # Perform GWPCA
        self._update_progress(5, 7, "Computing local PCAs")
        local_results = self._compute_gwpca(
            X_scaled, coords, params
        )
        
        # Calculate statistics
        self._update_progress(6, 7, "Calculating statistics")
        statistics = self._calculate_statistics(
            X_scaled, local_results, params, analysis_metadata
        )
        
        # Create spatial outputs
        self._update_progress(7, 7, "Creating spatial outputs")
        spatial_outputs = self._create_spatial_outputs(
            local_results, height, width, analysis_metadata, params
        )
        
        # Create metadata
        analysis_meta = AnalysisMetadata(
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
            metadata=analysis_meta,
            statistics=statistics,
            spatial_output=spatial_outputs['local_r2_map'],
            additional_outputs=spatial_outputs
        )
        
        # Store in database if configured
        if self.save_results_enabled:
            self.store_in_database(result)
        
        return result
    
    def _aggregate_to_blocks(self, data: np.ndarray, metadata: Dict[str, Any],
                           params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Aggregate pixel-level data to blocks.
        
        Args:
            data: Original pixel data
            metadata: Original metadata
            params: Parameters including block size
            
        Returns:
            Tuple of (block_data, block_metadata)
        """
        block_size_km = params['block_size_km']
        pixel_size_km = self.pixel_size_km
        block_size_pixels = int(block_size_km / pixel_size_km)
        
        if data.ndim == 3:
            n_bands, height, width = data.shape
        else:
            height, width = data.shape
            n_bands = 1
            data = data.reshape(1, height, width)
        
        # Calculate block dimensions
        n_blocks_y = (height + block_size_pixels - 1) // block_size_pixels
        n_blocks_x = (width + block_size_pixels - 1) // block_size_pixels
        
        # Initialize block arrays
        block_data = np.zeros((n_bands, n_blocks_y, n_blocks_x))
        block_counts = np.zeros((n_blocks_y, n_blocks_x))
        
        # Aggregate data
        agg_func = {
            'mean': np.nanmean,
            'median': np.nanmedian,
            'sum': np.nansum
        }[params['aggregation_method']]
        
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):
                # Define pixel range for this block
                y_start = by * block_size_pixels
                y_end = min((by + 1) * block_size_pixels, height)
                x_start = bx * block_size_pixels
                x_end = min((bx + 1) * block_size_pixels, width)
                
                # Extract block data
                block_pixels = data[:, y_start:y_end, x_start:x_end]
                
                # Count valid pixels
                if n_bands > 1:
                    valid_mask = ~np.isnan(block_pixels[0])
                else:
                    valid_mask = ~np.isnan(block_pixels)
                    
                block_counts[by, bx] = np.sum(valid_mask)
                
                # Aggregate
                for band in range(n_bands):
                    band_data = block_pixels[band]
                    if np.any(~np.isnan(band_data)):
                        block_data[band, by, bx] = agg_func(band_data)
                    else:
                        block_data[band, by, bx] = np.nan
        
        # Create block coordinates (centers)
        block_coords = []
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):
                # Center of block in pixel coordinates
                center_y = (by + 0.5) * block_size_pixels
                center_x = (bx + 0.5) * block_size_pixels
                block_coords.append([center_x, center_y])
        
        block_coords = np.array(block_coords)
        
        # Update metadata
        block_metadata = metadata.copy()
        block_metadata.update({
            'block_size_km': block_size_km,
            'block_size_pixels': block_size_pixels,
            'n_blocks': n_blocks_y * n_blocks_x,
            'n_blocks_y': n_blocks_y,
            'n_blocks_x': n_blocks_x,
            'block_coords': block_coords,
            'block_counts': block_counts,
            'aggregation_method': params['aggregation_method'],
            'original_pixels': height * width,
            'n_pixels': n_blocks_y * n_blocks_x  # For compatibility
        })
        
        # Handle coordinates transformation if original had geographic coords
        if 'coords' in metadata and metadata['coords']:
            # Transform block centers to geographic coordinates
            orig_lat = metadata['coords'].get('lat', [])
            orig_lon = metadata['coords'].get('lon', [])
            
            if len(orig_lat) > 0 and len(orig_lon) > 0:
                # Calculate lat/lon for block centers
                lat_step = (orig_lat[-1] - orig_lat[0]) / (height - 1) if height > 1 else 0
                lon_step = (orig_lon[-1] - orig_lon[0]) / (width - 1) if width > 1 else 0
                
                block_lats = []
                block_lons = []
                
                for by in range(n_blocks_y):
                    lat_center = orig_lat[0] + (by + 0.5) * block_size_pixels * lat_step
                    block_lats.append(lat_center)
                
                for bx in range(n_blocks_x):
                    lon_center = orig_lon[0] + (bx + 0.5) * block_size_pixels * lon_step
                    block_lons.append(lon_center)
                
                block_metadata['coords'] = {
                    'lat': block_lats,
                    'lon': block_lons
                }
        
        return block_data, block_metadata
    
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
            # For blocks, use smaller search range
            if params['use_block_aggregation']:
                n_coords = len(coords)
                search_min = max(10, int(0.05 * n_coords))
                search_max = min(int(0.5 * n_coords), 100)
            else:
                search_min = 50
                search_max = 500
            
            bw = selector.search(
                criterion=params['bandwidth_method'],
                search_method='golden',
                fixed=False,
                min_bandwidth=search_min,
                max_bandwidth=search_max
            )
        else:
            # Search fixed bandwidth (distance)
            # For blocks, use larger distances
            if params['use_block_aggregation']:
                max_dist = np.max(np.sqrt(np.sum((coords[0] - coords) ** 2, axis=1)))
                search_min = 0.05 * max_dist
                search_max = 0.5 * max_dist
            else:
                search_min = 10
                search_max = 1000
            
            bw = selector.search(
                criterion=params['bandwidth_method'],
                search_method='golden',
                kernel=params['kernel'],
                fixed=True,
                min_bandwidth=search_min,
                max_bandwidth=search_max
            )
        
        return bw
    
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
        
        # Compute weights for each location using simple distance-based kernels
        def gaussian_weights(dist, bandwidth):
            """Simple Gaussian kernel."""
            return np.exp(-0.5 * (dist / bandwidth) ** 2)
        
        def bisquare_weights(dist, bandwidth):
            """Simple bisquare kernel."""
            normalized_dist = dist / bandwidth
            weights = np.where(normalized_dist < 1, (1 - normalized_dist**2)**2, 0)
            return weights
        
        # Select kernel function
        if params['kernel'] == 'gaussian':
            kernel_func = gaussian_weights
        elif params['kernel'] == 'bisquare':
            kernel_func = bisquare_weights
        else:  # exponential
            kernel_func = gaussian_weights  # Default fallback
        
        bandwidth = params['bandwidth']
        
        # Process each location
        for i in range(n_pixels):
            if i % max(1, n_pixels // 20) == 0:
                progress = int(5 + (i / n_pixels) * 1)
                self._update_progress(progress, 7, f"Processing unit {i}/{n_pixels}")
            
            # Calculate weights
            distances = np.sqrt(np.sum((coords - coords[i]) ** 2, axis=1))
            
            if params['adaptive']:
                # Adaptive: find k nearest neighbors
                sorted_indices = np.argsort(distances)
                if int(bandwidth) < len(distances):
                    bw_i = distances[sorted_indices[int(bandwidth)]]
                else:
                    bw_i = distances[sorted_indices[-1]]
                weights = kernel_func(distances, bw_i)
            else:
                # Fixed bandwidth
                weights = kernel_func(distances, bandwidth)
            
            # Weight threshold
            weight_threshold = 1e-5
            valid_weights = weights > weight_threshold
            
            if np.sum(valid_weights) < n_features + 1:
                # Not enough neighbors, use all with distance weighting
                weights = 1.0 / (1.0 + distances)
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
                            params: Dict[str, Any],
                            metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate GWPCA statistics."""
        # Global PCA for comparison
        pca_global = PCA(n_components=params['n_components'])
        pca_global.fit(X)
        
        # Local statistics
        local_r2 = local_results['local_r2']
        local_eigenvalues = local_results['local_eigenvalues']
        
        # Calculate variation in local structure
        loading_variation = np.std(local_results['local_loadings'], axis=0)
        
        stats = {
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
        
        # Add block-specific statistics if applicable
        if params['use_block_aggregation']:
            stats.update({
                'analysis_scale': f"{params['block_size_km']}km blocks",
                'n_blocks_analyzed': metadata['n_blocks'],
                'original_pixels': metadata['original_pixels'],
                'aggregation_method': params['aggregation_method'],
                'data_reduction_factor': metadata['original_pixels'] / metadata['n_blocks']
            })
        else:
            stats['analysis_scale'] = f"{self.pixel_size_km}km pixels"
        
        return stats
    
    def _create_spatial_outputs(self, local_results: Dict[str, np.ndarray],
                              height: int, width: int,
                              metadata: Dict[str, Any],
                              params: Dict[str, Any]) -> Dict[str, Any]:
        """Create spatial output arrays."""
        import xarray as xr
        
        # Create coordinate arrays
        if 'coords' in metadata and metadata['coords']:
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
            attrs={
                'description': 'Local R² from first principal component',
                'analysis_scale': f"{params['block_size_km']}km blocks" if params['use_block_aggregation'] else "pixels"
            }
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
        
        # Add block information if using aggregation
        if params['use_block_aggregation'] and 'block_counts' in metadata:
            outputs['block_pixel_counts'] = xr.DataArray(
                metadata['block_counts'],
                coords={'lat': lat_coords, 'lon': lon_coords},
                dims=['lat', 'lon'],
                name='block_pixel_counts',
                attrs={'description': 'Number of pixels aggregated in each block'}
            )
        
        return outputs