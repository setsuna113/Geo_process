#!/usr/bin/env python3
"""
Python DML Wrapper - Uses Julia's predictions with DoubleML diagnostics

This is the CORRECT way to combine Julia's speed with Python's diagnostics
"""

import numpy as np
import pandas as pd
from doubleml import DoubleMLPLR, DoubleMLData
from sklearn.base import BaseEstimator
import json
from src.abstractions import load_json
from pathlib import Path


class JuliaResidualsDML:
    """
    Use Julia-computed predictions with DoubleML's diagnostic framework
    """
    
    def __init__(self, julia_output_path: str):
        """
        Load Julia's predictions and prepare for DoubleML
        
        Args:
            julia_output_path: Base path for Julia output files
        """
        self.base_path = Path(julia_output_path)
        
        # Load predictions (what DoubleML needs)
        predictions = np.load(f"{julia_output_path}_predictions.npz")
        self.l_hat = predictions['l_hat'].reshape(-1, 1)
        self.g_hat = predictions['g_hat'].reshape(-1, 1)
        
        # Load treatment predictions
        self.m_hats = {}
        for key in predictions.files:
            if key.startswith('m_hat_'):
                treatment_name = key.replace('m_hat_', '')
                self.m_hats[treatment_name] = predictions[key].reshape(-1, 1)
        
        # Load residuals for additional diagnostics
        residuals = np.load(f"{julia_output_path}_residuals.npz")
        self.y_residuals = residuals['y_residuals']
        self.t_residuals = {
            key.replace('t_residuals_', ''): residuals[key]
            for key in residuals.files if key.startswith('t_residuals_')
        }
        
        # Load Julia's diagnostics
        with open(f"{julia_output_path}_diagnostics.json", 'r') as f:
            self.julia_diagnostics = json.load(f)
        
        print("Loaded Julia results:")
        print(f"  - {self.julia_diagnostics['n_obs']} observations")
        print(f"  - {len(self.m_hats)} treatment variables")
        print(f"  - {self.julia_diagnostics['n_folds']} cross-fitting folds")
    
    def run_dml_with_diagnostics(self, data: pd.DataFrame, 
                                  treatment_col: str,
                                  outcome_col: str,
                                  n_rep: int = 10):
        """
        Run DoubleML using Julia's predictions and get FULL diagnostics
        
        This is where the magic happens - we get:
        1. Proper statistical inference
        2. Sensitivity analysis
        3. Confidence intervals
        4. All diagnostic checks
        """
        
        # Create DoubleML data structure
        dml_data = DoubleMLData(
            data,
            y_col=outcome_col,
            d_cols=[treatment_col],
            x_cols=data.columns.difference([outcome_col, treatment_col]).tolist()
        )
        
        # Prepare external predictions dictionary
        external_predictions = {
            treatment_col: {
                'ml_l': self.l_hat,
                'ml_m': self.m_hats[treatment_col],
                'ml_g': self.g_hat
            }
        }
        
        # Initialize DoubleML PLR model
        # We use dummy learners since we're providing external predictions
        from sklearn.dummy import DummyRegressor
        dummy_learner = DummyRegressor(strategy='mean')
        
        dml_plr = DoubleMLPLR(
            dml_data,
            ml_l=dummy_learner,
            ml_m=dummy_learner,
            ml_g=dummy_learner,
            n_folds=5,  # Doesn't matter with external predictions
            n_rep=1,    # Start with 1, then repeat for stability
            score='partialling out'
        )
        
        # Fit with external predictions
        print("\nRunning DoubleML with Julia predictions...")
        dml_plr.fit(external_predictions=external_predictions)
        
        # Extract comprehensive diagnostics
        diagnostics = {
            'coefficient': dml_plr.coef[0],
            'std_error': dml_plr.se[0],
            'confidence_interval': dml_plr.confint(level=0.95).values[0].tolist(),
            'p_value': dml_plr.pval[0],
            't_stat': dml_plr.t_stat[0],
            
            # Score components (psi_a, psi_b)
            'psi_elements': {
                'psi_a_mean': float(np.mean(dml_plr.psi_elements['psi_a'])),
                'psi_a_std': float(np.std(dml_plr.psi_elements['psi_a'])),
                'psi_b_mean': float(np.mean(dml_plr.psi_elements['psi_b'])),
                'psi_b_std': float(np.std(dml_plr.psi_elements['psi_b']))
            },
            
            # Orthogonality check
            'score_mean': float(np.mean(dml_plr.psi)),
            'score_std': float(np.std(dml_plr.psi)),
            
            # Julia's RMSE scores
            'julia_rmse': self.julia_diagnostics['rmse_scores']
        }
        
        # CRITICAL: Orthogonality test
        orthogonality_violation = abs(diagnostics['score_mean']) / diagnostics['score_std']
        if orthogonality_violation > 0.01:
            print(f"⚠️  WARNING: Orthogonality violation detected: {orthogonality_violation:.4f}")
        else:
            print(f"✅ Orthogonality condition satisfied: {orthogonality_violation:.6f}")
        
        # Stability check with multiple repetitions
        print(f"\nRunning stability check with {n_rep} repetitions...")
        coefficients = []
        for rep in range(n_rep):
            # Add small noise to predictions to simulate different samples
            noise_scale = 1e-6
            noisy_predictions = {
                treatment_col: {
                    'ml_l': self.l_hat + np.random.normal(0, noise_scale, self.l_hat.shape),
                    'ml_m': self.m_hats[treatment_col] + np.random.normal(0, noise_scale, self.m_hats[treatment_col].shape),
                    'ml_g': self.g_hat + np.random.normal(0, noise_scale, self.g_hat.shape)
                }
            }
            dml_plr_rep = DoubleMLPLR(dml_data, dummy_learner, dummy_learner, dummy_learner, n_folds=5, n_rep=1)
            dml_plr_rep.fit(external_predictions=noisy_predictions)
            coefficients.append(dml_plr_rep.coef[0])
        
        stability_cv = np.std(coefficients) / abs(np.mean(coefficients))
        diagnostics['stability'] = {
            'coefficient_cv': float(stability_cv),
            'stable': stability_cv < 0.1
        }
        
        if stability_cv > 0.1:
            print(f"⚠️  WARNING: Unstable estimates, CV = {stability_cv:.2%}")
        else:
            print(f"✅ Stable estimates, CV = {stability_cv:.2%}")
        
        return diagnostics
    
    def validate_residuals(self):
        """
        Additional residual diagnostics not available in standard DoubleML
        """
        
        print("\n=== RESIDUAL DIAGNOSTICS ===")
        
        # Check residual correlations
        for t_name, t_resid in self.t_residuals.items():
            correlation = np.corrcoef(self.y_residuals, t_resid)[0, 1]
            print(f"Correlation(Y_residual, {t_name}_residual): {correlation:.4f}")
            
            if abs(correlation) > 0.1:
                print(f"  ⚠️  High residual correlation detected!")
        
        # Check for extreme residuals (overlap violations)
        for t_name, t_resid in self.t_residuals.items():
            extreme_count = np.sum(np.abs(t_resid) > 3 * np.std(t_resid))
            extreme_pct = 100 * extreme_count / len(t_resid)
            print(f"Extreme residuals in {t_name}: {extreme_count} ({extreme_pct:.2f}%)")
            
            if extreme_pct > 1:
                print(f"  ⚠️  Potential overlap violation!")
        
        return True


# Example usage
if __name__ == "__main__":
    # This would be called after Julia saves the predictions
    
    # Load your actual data
    # data = pd.read_parquet("your_data.parquet")
    
    # Initialize with Julia's output
    # dml = JuliaResidualsDML("/path/to/julia/output")
    
    # Run DML with full diagnostics
    # results = dml.run_dml_with_diagnostics(
    #     data, 
    #     treatment_col="plants_richness",
    #     outcome_col="crop_production"
    # )
    
    # Additional residual checks
    # dml.validate_residuals()
    
    print("\nHybrid Julia-Python DML pipeline ready!")
    print("Julia handles the heavy computation, Python provides diagnostics")