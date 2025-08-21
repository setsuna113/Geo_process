#!/usr/bin/env python3
"""
Nuisance Function Diagnostics for DML
"""

import numpy as np
from typing import Dict, Any
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class NuisanceDiagnostics:
    """Diagnose nuisance function estimation quality"""
    
    def __init__(self, estimator):
        self.estimator = estimator
        
    def check_nuisance_fit(self) -> Dict[str, Any]:
        """Check nuisance function fitting quality"""
        
        diagnostics = {
            'model_y_type': type(getattr(self.estimator, '_model_y', None)).__name__,
            'model_t_type': type(getattr(self.estimator, '_model_t', None)).__name__,
        }
        
        # Check if nuisance scores are available
        if hasattr(self.estimator, 'nuisance_scores_'):
            scores = self.estimator.nuisance_scores_
            diagnostics['nuisance_scores_available'] = True
            diagnostics['mean_score'] = float(np.mean(scores))
            diagnostics['std_score'] = float(np.std(scores))
        else:
            diagnostics['nuisance_scores_available'] = False
            
        # Check cross-validation scores if available
        if hasattr(self.estimator, 'cv_results_'):
            cv_results = self.estimator.cv_results_
            diagnostics['cv_scores'] = {
                'mean': float(np.mean(cv_results)),
                'std': float(np.std(cv_results))
            }
            
        return diagnostics
    
    def check_overlap(self, T: np.ndarray, W: np.ndarray) -> Dict[str, Any]:
        """Check overlap/common support"""
        
        # Simple overlap check based on treatment variation within confounder bins
        diagnostics = {}
        
        # Bin confounders and check treatment variation
        n_bins = min(10, len(T) // 100)
        
        if len(W.shape) > 1:
            # Use first principal component for binning
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            W_1d = pca.fit_transform(W).flatten()
        else:
            W_1d = W.flatten()
            
        bins = np.percentile(W_1d, np.linspace(0, 100, n_bins + 1))
        bin_indices = np.digitize(W_1d, bins) - 1
        
        treatment_variation = []
        for i in range(n_bins):
            bin_mask = bin_indices == i
            if np.sum(bin_mask) > 10:
                if len(T.shape) > 1:
                    var = np.mean([np.var(T[bin_mask, j]) for j in range(T.shape[1])])
                else:
                    var = np.var(T[bin_mask])
                treatment_variation.append(var)
                
        diagnostics['mean_treatment_variation'] = float(np.mean(treatment_variation))
        diagnostics['min_treatment_variation'] = float(np.min(treatment_variation))
        diagnostics['sufficient_overlap'] = diagnostics['min_treatment_variation'] > 0.01
        
        if not diagnostics['sufficient_overlap']:
            diagnostics['warning'] = "Insufficient treatment variation in some confounder regions"
            
        return diagnostics