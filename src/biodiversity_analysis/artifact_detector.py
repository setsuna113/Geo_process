"""
Artifact Detection for Biodiversity Data
=========================================

Statistical methods to detect and quantify artifacts in raster data,
particularly striping patterns common in SDM outputs.
"""

import numpy as np
import rasterio
from scipy import stats, signal
from typing import Dict, List, Tuple, Optional
import warnings
from pathlib import Path

class ArtifactDetector:
    """
    Detect and quantify artifacts in biodiversity raster data.
    
    Focuses on:
    1. Striping patterns (horizontal/vertical)
    2. Checkerboard patterns
    3. Blockiness/artificial boundaries
    4. Unrealistic smoothness
    """
    
    def __init__(self):
        """Initialize the artifact detector"""
        self.results = {}
        
    def detect_striping(self, data: np.ndarray, mask: np.ndarray = None) -> Dict:
        """
        Detect horizontal and vertical striping patterns.
        
        Striping occurs when rows or columns have unusually similar values,
        creating artificial linear patterns.
        
        Parameters
        ----------
        data : np.ndarray
            2D array of raster data
        mask : np.ndarray
            Boolean mask for valid data
            
        Returns
        -------
        Dict
            Detection results with statistics
        """
        if mask is None:
            mask = ~np.isnan(data)
            
        results = {
            'horizontal_striping': {},
            'vertical_striping': {},
            'diagonal_striping': {}
        }
        
        # Horizontal striping (row-wise similarity)
        row_means = np.nanmean(data, axis=1)
        row_stds = np.nanstd(data, axis=1)
        
        # Calculate coefficient of variation for each row
        row_cv = np.divide(row_stds, row_means, 
                          out=np.zeros_like(row_stds), 
                          where=row_means!=0)
        
        # Low CV across rows indicates striping
        valid_row_cv = row_cv[~np.isnan(row_cv)]
        if len(valid_row_cv) > 0:
            # Autocorrelation of row means to detect periodic striping
            if len(row_means) > 10:
                row_autocorr = np.correlate(row_means - np.nanmean(row_means), 
                                           row_means - np.nanmean(row_means), 
                                           mode='same')
                row_autocorr = row_autocorr / np.max(row_autocorr)
                
                # Find peaks in autocorrelation (indicates periodicity)
                peaks, properties = signal.find_peaks(row_autocorr[len(row_autocorr)//2:], 
                                                     height=0.5)
                
                results['horizontal_striping'] = {
                    'mean_cv': float(np.mean(valid_row_cv)),
                    'std_cv': float(np.std(valid_row_cv)),
                    'min_cv': float(np.min(valid_row_cv)),
                    'periodic': len(peaks) > 0,
                    'period': int(peaks[0]) if len(peaks) > 0 else None,
                    'severity': self._classify_striping_severity(np.mean(valid_row_cv))
                }
        
        # Vertical striping (column-wise similarity)
        col_means = np.nanmean(data, axis=0)
        col_stds = np.nanstd(data, axis=0)
        
        col_cv = np.divide(col_stds, col_means, 
                          out=np.zeros_like(col_stds), 
                          where=col_means!=0)
        
        valid_col_cv = col_cv[~np.isnan(col_cv)]
        if len(valid_col_cv) > 0:
            # Autocorrelation of column means
            if len(col_means) > 10:
                col_autocorr = np.correlate(col_means - np.nanmean(col_means), 
                                           col_means - np.nanmean(col_means), 
                                           mode='same')
                col_autocorr = col_autocorr / np.max(col_autocorr)
                
                peaks, properties = signal.find_peaks(col_autocorr[len(col_autocorr)//2:], 
                                                     height=0.5)
                
                results['vertical_striping'] = {
                    'mean_cv': float(np.mean(valid_col_cv)),
                    'std_cv': float(np.std(valid_col_cv)),
                    'min_cv': float(np.min(valid_col_cv)),
                    'periodic': len(peaks) > 0,
                    'period': int(peaks[0]) if len(peaks) > 0 else None,
                    'severity': self._classify_striping_severity(np.mean(valid_col_cv))
                }
        
        # Diagonal striping (45-degree patterns)
        # Extract diagonals and check for consistency
        diag_cvs = []
        for offset in range(-data.shape[0] + 1, data.shape[1]):
            diag = np.diagonal(data, offset=offset)
            if len(diag) > 10:
                diag_mean = np.nanmean(diag)
                diag_std = np.nanstd(diag)
                if diag_mean > 0:
                    diag_cvs.append(diag_std / diag_mean)
                    
        if diag_cvs:
            results['diagonal_striping'] = {
                'mean_cv': float(np.mean(diag_cvs)),
                'severity': self._classify_striping_severity(np.mean(diag_cvs))
            }
            
        return results
    
    def calculate_morans_i(self, data: np.ndarray, mask: np.ndarray = None) -> Dict:
        """
        Calculate Moran's I spatial autocorrelation statistic.
        
        Moran's I ranges from -1 (perfect dispersion/checkerboard) to +1 (perfect clustering).
        Values near 0 indicate random spatial pattern.
        
        Parameters
        ----------
        data : np.ndarray
            2D array of raster data
        mask : np.ndarray
            Boolean mask for valid data
            
        Returns
        -------
        Dict
            Moran's I statistics and interpretation
        """
        if mask is None:
            mask = ~np.isnan(data)
            
        # Flatten and remove NaN values
        valid_mask = mask.flatten()
        values = data.flatten()[valid_mask]
        
        if len(values) < 10:
            return {'error': 'Insufficient valid data'}
            
        # Create spatial weights matrix (queen contiguity)
        n = len(values)
        
        # For computational efficiency, subsample if too large
        if n > 10000:
            sample_idx = np.random.choice(n, 10000, replace=False)
            values = values[sample_idx]
            n = 10000
            
        # Calculate deviations from mean
        mean_val = np.mean(values)
        deviations = values - mean_val
        
        # Simple Moran's I for raster (using rook contiguity)
        # This is a simplified version for demonstration
        rows, cols = data.shape
        moran_sum = 0
        weight_sum = 0
        
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if mask[i, j]:
                    center_val = data[i, j] - mean_val
                    
                    # Check 4 neighbors (rook contiguity)
                    neighbors = [
                        (i-1, j), (i+1, j),  # vertical
                        (i, j-1), (i, j+1)   # horizontal
                    ]
                    
                    for ni, nj in neighbors:
                        if mask[ni, nj]:
                            neighbor_val = data[ni, nj] - mean_val
                            moran_sum += center_val * neighbor_val
                            weight_sum += 1
                            
        if weight_sum == 0:
            return {'error': 'No valid neighbor pairs'}
            
        # Calculate Moran's I
        variance = np.var(values)
        if variance == 0:
            return {'error': 'Zero variance in data'}
            
        morans_i = (n / weight_sum) * (moran_sum / (n * variance))
        
        # Expected value under null hypothesis
        expected_i = -1 / (n - 1)
        
        # Classify the spatial pattern
        if morans_i > 0.3:
            pattern = "Strong clustering"
            interpretation = "Values are spatially clustered (similar values near each other)"
        elif morans_i > 0.1:
            pattern = "Moderate clustering"
            interpretation = "Some spatial clustering present"
        elif morans_i > -0.1:
            pattern = "Random"
            interpretation = "No clear spatial pattern"
        elif morans_i > -0.3:
            pattern = "Moderate dispersion"
            interpretation = "Some checkerboard-like pattern"
        else:
            pattern = "Strong dispersion"
            interpretation = "Strong checkerboard/striping pattern (artifact likely)"
            
        return {
            'morans_i': float(morans_i),
            'expected_i': float(expected_i),
            'pattern': pattern,
            'interpretation': interpretation,
            'artifact_likely': morans_i < -0.2 or morans_i > 0.8
        }
    
    def detect_blockiness(self, data: np.ndarray, mask: np.ndarray = None) -> Dict:
        """
        Detect artificial blockiness or grid-like patterns.
        
        Common in SDMs when environmental predictors have different resolutions.
        
        Parameters
        ----------
        data : np.ndarray
            2D array of raster data
        mask : np.ndarray
            Boolean mask for valid data
            
        Returns
        -------
        Dict
            Blockiness detection results
        """
        if mask is None:
            mask = ~np.isnan(data)
            
        # Calculate local variance in sliding windows
        window_sizes = [3, 5, 7]
        blockiness_scores = {}
        
        for window_size in window_sizes:
            local_vars = []
            
            for i in range(0, data.shape[0] - window_size, window_size):
                for j in range(0, data.shape[1] - window_size, window_size):
                    window = data[i:i+window_size, j:j+window_size]
                    window_mask = mask[i:i+window_size, j:j+window_size]
                    
                    if np.sum(window_mask) > window_size:
                        local_var = np.nanvar(window)
                        local_vars.append(local_var)
                        
            if local_vars:
                # Low variance within blocks indicates blockiness
                mean_local_var = np.mean(local_vars)
                global_var = np.nanvar(data[mask])
                
                if global_var > 0:
                    blockiness_ratio = mean_local_var / global_var
                else:
                    blockiness_ratio = 1.0
                    
                blockiness_scores[f'window_{window_size}'] = {
                    'ratio': float(blockiness_ratio),
                    'blocky': blockiness_ratio < 0.3
                }
                
        # Edge detection to find artificial boundaries
        # Sobel filter for edge detection
        from scipy import ndimage
        
        # Fill NaN with local mean for edge detection
        filled_data = data.copy()
        filled_data[~mask] = np.nanmean(data)
        
        sobel_h = ndimage.sobel(filled_data, axis=0)
        sobel_v = ndimage.sobel(filled_data, axis=1)
        edge_magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
        
        # Check for regular grid pattern in edges
        edge_threshold = np.percentile(edge_magnitude[mask], 90)
        strong_edges = edge_magnitude > edge_threshold
        
        # Detect if edges form a grid
        edge_row_sums = np.sum(strong_edges, axis=1)
        edge_col_sums = np.sum(strong_edges, axis=0)
        
        # Autocorrelation to detect periodicity in edges
        if len(edge_row_sums) > 20:
            row_edge_autocorr = np.correlate(edge_row_sums - np.mean(edge_row_sums),
                                            edge_row_sums - np.mean(edge_row_sums),
                                            mode='same')
            peaks, _ = signal.find_peaks(row_edge_autocorr[len(row_edge_autocorr)//2:],
                                        height=np.std(row_edge_autocorr))
            
            grid_detected = len(peaks) > 0 and peaks[0] < 50  # Grid with period < 50 pixels
        else:
            grid_detected = False
            
        return {
            'blockiness_scores': blockiness_scores,
            'grid_pattern_detected': grid_detected,
            'edge_regularity': float(np.std(edge_row_sums)) if len(edge_row_sums) > 0 else None,
            'interpretation': self._interpret_blockiness(blockiness_scores, grid_detected)
        }
    
    def assess_smoothness(self, data: np.ndarray, mask: np.ndarray = None) -> Dict:
        """
        Assess if data is unrealistically smooth (common in SDMs).
        
        Natural species distributions should show some roughness/patchiness.
        
        Parameters
        ----------
        data : np.ndarray
            2D array of raster data
        mask : np.ndarray
            Boolean mask for valid data
            
        Returns
        -------
        Dict
            Smoothness assessment results
        """
        if mask is None:
            mask = ~np.isnan(data)
            
        # Calculate roughness using different methods
        results = {}
        
        # 1. Local gradient magnitude
        grad_y, grad_x = np.gradient(data)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        valid_gradients = gradient_magnitude[mask]
        if len(valid_gradients) > 0:
            results['mean_gradient'] = float(np.nanmean(valid_gradients))
            results['std_gradient'] = float(np.nanstd(valid_gradients))
            
            # Low gradient indicates smoothness
            results['smooth_score'] = 1.0 / (1.0 + results['mean_gradient'])
            
        # 2. Fractal dimension (roughness measure)
        # Using box-counting method (simplified)
        def fractal_dimension(Z, threshold=0.9):
            # Only for 2D arrays
            assert(len(Z.shape) == 2)
            
            # Box sizes to use
            scales = np.logspace(0.01, 1, num=10, base=2, dtype=int)
            scales = np.unique(scales)
            
            Ns = []
            for scale in scales:
                H, W = Z.shape
                grids = []
                
                for i in range(0, H, scale):
                    for j in range(0, W, scale):
                        box = Z[i:i+scale, j:j+scale]
                        if np.any(mask[i:i+scale, j:j+scale]):
                            grids.append(np.nanstd(box))
                            
                Ns.append(len([g for g in grids if g > threshold]))
                
            # Linear fit in log-log space
            coeffs = np.polyfit(np.log(scales), np.log(Ns + 1), 1)
            
            return -coeffs[0]
        
        try:
            fd = fractal_dimension(data)
            results['fractal_dimension'] = float(fd)
            # Higher fractal dimension = more complex/rough
            # Lower fractal dimension = smoother
            results['complexity'] = 'smooth' if fd < 1.5 else 'rough' if fd > 2.5 else 'moderate'
        except:
            results['fractal_dimension'] = None
            
        # 3. Coefficient of variation in local neighborhoods
        local_cvs = []
        window_size = 5
        
        for i in range(0, data.shape[0] - window_size, window_size):
            for j in range(0, data.shape[1] - window_size, window_size):
                window = data[i:i+window_size, j:j+window_size]
                window_mask = mask[i:i+window_size, j:j+window_size]
                
                if np.sum(window_mask) > window_size:
                    mean_val = np.nanmean(window)
                    std_val = np.nanstd(window)
                    if mean_val > 0:
                        local_cvs.append(std_val / mean_val)
                        
        if local_cvs:
            results['mean_local_cv'] = float(np.mean(local_cvs))
            results['overly_smooth'] = results['mean_local_cv'] < 0.1
            
        # Interpretation
        if results.get('smooth_score', 0) > 0.8 and results.get('mean_local_cv', 1) < 0.1:
            results['interpretation'] = "Unrealistically smooth - likely SDM artifact"
        elif results.get('smooth_score', 0) > 0.6:
            results['interpretation'] = "Moderately smooth - possible SDM smoothing"
        else:
            results['interpretation'] = "Natural roughness/patchiness present"
            
        return results
    
    def _classify_striping_severity(self, cv: float) -> str:
        """Classify striping severity based on coefficient of variation"""
        if cv < 0.05:
            return "Severe striping"
        elif cv < 0.1:
            return "Moderate striping"
        elif cv < 0.2:
            return "Mild striping"
        else:
            return "No significant striping"
            
    def _interpret_blockiness(self, scores: Dict, grid_detected: bool) -> str:
        """Interpret blockiness scores"""
        if grid_detected:
            return "Clear grid pattern detected - likely artifact from predictor resolution"
            
        blocky_windows = sum(1 for k, v in scores.items() if v.get('blocky', False))
        
        if blocky_windows >= 2:
            return "Significant blockiness detected - possible resolution mismatch"
        elif blocky_windows == 1:
            return "Some blockiness present"
        else:
            return "No significant blockiness"
            
    def full_artifact_analysis(self, file_path: str) -> Dict:
        """
        Perform complete artifact analysis on a raster file.
        
        Parameters
        ----------
        file_path : str
            Path to raster file
            
        Returns
        -------
        Dict
            Complete artifact analysis results
        """
        with rasterio.open(file_path) as src:
            data = src.read(1)
            
            # Create mask for valid data
            if src.nodata is not None:
                mask = (data != src.nodata)
            else:
                mask = ~np.isnan(data)
                
        results = {
            'file': Path(file_path).name,
            'striping': self.detect_striping(data, mask),
            'spatial_autocorrelation': self.calculate_morans_i(data, mask),
            'blockiness': self.detect_blockiness(data, mask),
            'smoothness': self.assess_smoothness(data, mask)
        }
        
        # Overall artifact assessment
        artifact_score = 0
        artifact_types = []
        
        # Check striping
        if results['striping']['horizontal_striping'].get('severity') in ['Severe striping', 'Moderate striping']:
            artifact_score += 2
            artifact_types.append('horizontal striping')
            
        if results['striping']['vertical_striping'].get('severity') in ['Severe striping', 'Moderate striping']:
            artifact_score += 2
            artifact_types.append('vertical striping')
            
        # Check spatial autocorrelation
        if results['spatial_autocorrelation'].get('artifact_likely', False):
            artifact_score += 3
            artifact_types.append('abnormal spatial pattern')
            
        # Check blockiness
        if results['blockiness'].get('grid_pattern_detected', False):
            artifact_score += 3
            artifact_types.append('grid artifacts')
            
        # Check smoothness
        if results['smoothness'].get('overly_smooth', False):
            artifact_score += 2
            artifact_types.append('over-smoothing')
            
        # Overall assessment
        if artifact_score >= 5:
            overall = "High artifact presence - data quality questionable"
        elif artifact_score >= 3:
            overall = "Moderate artifacts - use with caution"
        elif artifact_score >= 1:
            overall = "Minor artifacts - generally acceptable"
        else:
            overall = "No significant artifacts detected"
            
        results['overall_assessment'] = {
            'artifact_score': artifact_score,
            'artifact_types': artifact_types,
            'interpretation': overall
        }
        
        return results