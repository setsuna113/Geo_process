#!/usr/bin/env python3
"""
Residual Analysis for DML
Analyzes residuals from nuisance function estimation
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class ResidualAnalyzer:
    """Analyze residuals from DML estimation"""
    
    def __init__(self, estimator, Y: np.ndarray, T: np.ndarray, W: np.ndarray):
        """
        Initialize residual analyzer
        
        Args:
            estimator: Fitted DML estimator
            Y: Outcome array
            T: Treatment array
            W: Confounder array
        """
        self.estimator = estimator
        self.Y = Y
        self.T = T if len(T.shape) > 1 else T.reshape(-1, 1)
        self.W = W
        self.residuals = {}
        
    def compute_residuals(self) -> Dict[str, np.ndarray]:
        """Compute all residuals"""
        
        print("\nComputing residuals...")
        
        # Get predictions for outcome
        try:
            # For DML, residuals are computed during cross-fitting
            # We need to access the nuisance models
            
            # Outcome residuals (Y - E[Y|W])
            if hasattr(self.estimator, '_models_y'):
                # Access first stage models
                Y_pred = self._get_cross_fitted_predictions(self.estimator._models_y, self.W)
                self.residuals['outcome'] = self.Y - Y_pred
            else:
                # Approximate using final model
                Y_pred = self._predict_outcome(self.W)
                self.residuals['outcome'] = self.Y - Y_pred
                
            # Treatment residuals (T - E[T|W])
            if self.T.shape[1] == 1:
                if hasattr(self.estimator, '_models_t'):
                    T_pred = self._get_cross_fitted_predictions(self.estimator._models_t, self.W)
                    self.residuals['treatment'] = self.T.flatten() - T_pred
                else:
                    T_pred = self._predict_treatment(self.W)
                    self.residuals['treatment'] = self.T.flatten() - T_pred
            else:
                # Multiple treatments
                self.residuals['treatment'] = []
                for i in range(self.T.shape[1]):
                    T_i_pred = self._predict_treatment_i(self.W, i)
                    self.residuals[f'treatment_{i}'] = self.T[:, i] - T_i_pred
                    
        except Exception as e:
            print(f"Warning: Could not compute exact residuals: {e}")
            print("Using approximations...")
            
            # Simple residuals based on effects
            effects = self.estimator.const_marginal_effect(X=None)
            if len(effects.shape) == 2:
                effects = effects[0]
            
            # Approximate residuals
            T_effect = np.dot(self.T, effects) if len(self.T.shape) > 1 else self.T * effects[0]
            self.residuals['approximate'] = self.Y - T_effect
            
        print(f"Computed {len(self.residuals)} residual types")
        return self.residuals
    
    def analyze_residuals(self) -> Dict[str, Any]:
        """Comprehensive residual analysis"""
        
        if not self.residuals:
            self.compute_residuals()
            
        analysis = {}
        
        print("\n" + "="*50)
        print("RESIDUAL ANALYSIS")
        print("="*50)
        
        for name, resid in self.residuals.items():
            if isinstance(resid, list):
                continue
                
            print(f"\n{name.upper()} Residuals:")
            
            # Basic statistics
            stats_dict = {
                'mean': float(np.mean(resid)),
                'std': float(np.std(resid)),
                'min': float(np.min(resid)),
                'max': float(np.max(resid)),
                'median': float(np.median(resid)),
                'skewness': float(stats.skew(resid)),
                'kurtosis': float(stats.kurtosis(resid))
            }
            
            print(f"  Mean: {stats_dict['mean']:.6f}")
            print(f"  Std: {stats_dict['std']:.4f}")
            print(f"  Skewness: {stats_dict['skewness']:.4f}")
            print(f"  Kurtosis: {stats_dict['kurtosis']:.4f}")
            
            # Normality tests
            if len(resid) > 20:
                # Shapiro-Wilk test (for smaller samples)
                if len(resid) < 5000:
                    shapiro_stat, shapiro_p = stats.shapiro(resid)
                    stats_dict['shapiro_p'] = float(shapiro_p)
                    print(f"  Shapiro-Wilk p-value: {shapiro_p:.4f}")
                    
                # Jarque-Bera test
                jb_stat, jb_p = stats.jarque_bera(resid)
                stats_dict['jarque_bera_p'] = float(jb_p)
                print(f"  Jarque-Bera p-value: {jb_p:.4f}")
                
                # D'Agostino-Pearson test
                dp_stat, dp_p = stats.normaltest(resid)
                stats_dict['dagostino_p'] = float(dp_p)
                
                # Check for normality
                alpha = 0.05
                stats_dict['appears_normal'] = jb_p > alpha
                
                if not stats_dict['appears_normal']:
                    print(f"  ⚠ Residuals appear non-normal")
                    
            # Autocorrelation (if sequential data)
            try:
                from statsmodels.stats.stattools import durbin_watson
                dw_stat = durbin_watson(resid)
                stats_dict['durbin_watson'] = float(dw_stat)
                print(f"  Durbin-Watson: {dw_stat:.4f}")
                
                if dw_stat < 1.5 or dw_stat > 2.5:
                    print(f"  ⚠ Potential autocorrelation detected")
            except:
                pass
                
            # Heteroscedasticity check
            stats_dict['heteroscedasticity'] = self._check_heteroscedasticity(resid)
            if stats_dict['heteroscedasticity']['detected']:
                print(f"  ⚠ Heteroscedasticity detected")
                
            # Outlier detection
            outliers = self._detect_outliers(resid)
            stats_dict['outliers'] = outliers
            print(f"  Outliers (>3σ): {outliers['n_outliers']} ({outliers['pct_outliers']:.1f}%)")
            
            if outliers['pct_outliers'] > 5:
                print(f"  ⚠ High proportion of outliers")
                
            analysis[name] = stats_dict
            
        return analysis
    
    def _check_heteroscedasticity(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Check for heteroscedasticity in residuals"""
        
        # Bin residuals and check variance
        n_bins = min(10, len(residuals) // 100)
        
        # Sort by predicted values (approximation)
        sorted_idx = np.argsort(np.abs(residuals))
        bin_size = len(residuals) // n_bins
        
        bin_variances = []
        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < n_bins - 1 else len(residuals)
            bin_resid = residuals[sorted_idx[start:end]]
            bin_variances.append(np.var(bin_resid))
            
        # Coefficient of variation of variances
        cv = np.std(bin_variances) / np.mean(bin_variances)
        
        return {
            'cv_of_variances': float(cv),
            'detected': cv > 0.5,
            'bin_variances': [float(v) for v in bin_variances]
        }
    
    def _detect_outliers(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Detect outliers using multiple methods"""
        
        # Method 1: 3-sigma rule
        mean = np.mean(residuals)
        std = np.std(residuals)
        outliers_3sigma = np.abs(residuals - mean) > 3 * std
        
        # Method 2: IQR method
        q1, q3 = np.percentile(residuals, [25, 75])
        iqr = q3 - q1
        outliers_iqr = (residuals < q1 - 1.5 * iqr) | (residuals > q3 + 1.5 * iqr)
        
        # Method 3: Modified Z-score
        median = np.median(residuals)
        mad = np.median(np.abs(residuals - median))
        modified_z = 0.6745 * (residuals - median) / (mad + 1e-10)
        outliers_mz = np.abs(modified_z) > 3.5
        
        return {
            'n_outliers': int(np.sum(outliers_3sigma)),
            'pct_outliers': float(100 * np.mean(outliers_3sigma)),
            'n_outliers_iqr': int(np.sum(outliers_iqr)),
            'n_outliers_mad': int(np.sum(outliers_mz)),
            'outlier_indices': np.where(outliers_3sigma)[0].tolist()[:10]  # First 10
        }
    
    def _get_cross_fitted_predictions(self, models, W):
        """Get cross-fitted predictions (placeholder)"""
        # This would need access to the internal cross-fitting structure
        # For now, return zeros
        return np.zeros(len(W))
    
    def _predict_outcome(self, W):
        """Predict outcome given confounders"""
        # Simplified prediction
        return np.zeros(len(W))
    
    def _predict_treatment(self, W):
        """Predict treatment given confounders"""
        return np.zeros(len(W))
    
    def _predict_treatment_i(self, W, i):
        """Predict i-th treatment given confounders"""
        return np.zeros(len(W))
    
    def check_score_residuals(self) -> Dict[str, Any]:
        """Check orthogonal score residuals"""
        
        print("\n" + "="*50)
        print("SCORE RESIDUAL ANALYSIS")
        print("="*50)
        
        results = {}
        
        # Check if score is available
        if hasattr(self.estimator, 'score_'):
            score = self.estimator.score_
            
            print(f"Score shape: {score.shape}")
            print(f"Score mean: {np.mean(score):.6f}")
            print(f"Score std: {np.std(score):.4f}")
            
            # Score should have mean zero
            score_mean = np.mean(score)
            t_stat = score_mean / (np.std(score) / np.sqrt(len(score)))
            p_value = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))
            
            results['score_mean'] = float(score_mean)
            results['score_t_stat'] = float(t_stat)
            results['score_p_value'] = float(p_value)
            results['mean_zero_test'] = p_value > 0.05
            
            print(f"Mean zero test p-value: {p_value:.4f}")
            
            if not results['mean_zero_test']:
                print("⚠ Score mean significantly different from zero")
                results['warning'] = "Possible misspecification"
        else:
            print("Score not available")
            results['score_available'] = False
            
        return results
    
    def generate_report(self) -> str:
        """Generate text report of residual analysis"""
        
        analysis = self.analyze_residuals()
        score_analysis = self.check_score_residuals()
        
        report = []
        report.append("="*70)
        report.append("RESIDUAL ANALYSIS REPORT")
        report.append("="*70)
        
        # Summary statistics
        for name, stats in analysis.items():
            report.append(f"\n{name.upper()} Residuals:")
            report.append(f"  Mean: {stats['mean']:.6f}")
            report.append(f"  Std Dev: {stats['std']:.4f}")
            report.append(f"  Skewness: {stats['skewness']:.4f}")
            report.append(f"  Kurtosis: {stats['kurtosis']:.4f}")
            
            if 'appears_normal' in stats:
                report.append(f"  Appears Normal: {stats['appears_normal']}")
                
            if 'outliers' in stats:
                report.append(f"  Outliers: {stats['outliers']['n_outliers']} ({stats['outliers']['pct_outliers']:.1f}%)")
                
        # Score residuals
        if score_analysis:
            report.append("\nSCORE RESIDUALS:")
            if 'score_mean' in score_analysis:
                report.append(f"  Mean: {score_analysis['score_mean']:.6f}")
                report.append(f"  Mean Zero Test: {'Pass' if score_analysis['mean_zero_test'] else 'Fail'}")
                
        # Warnings
        warnings_list = []
        for name, stats in analysis.items():
            if 'appears_normal' in stats and not stats['appears_normal']:
                warnings_list.append(f"{name}: Non-normal residuals")
            if 'outliers' in stats and stats['outliers']['pct_outliers'] > 5:
                warnings_list.append(f"{name}: High outlier percentage")
                
        if warnings_list:
            report.append("\n⚠ WARNINGS:")
            for w in warnings_list:
                report.append(f"  - {w}")
                
        return "\n".join(report)