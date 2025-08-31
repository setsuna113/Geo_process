#!/usr/bin/env python3
"""
Generate mock Julia output with SHAP parameters in standard format.
Simulates what the updated Julia pipeline would produce.
"""

import json
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from pathlib import Path

def generate_mock_julia_output(output_dir="results/julia_mock_standard", n_samples=1000, n_features=10):
    """Generate mock Julia standard output with SHAP params."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating mock Julia standard output in {output_dir}")
    
    np.random.seed(42)
    
    # Feature names (similar to actual pipeline)
    feature_names = [
        "bio01_mean_annual_temp",
        "bio04_temp_seasonality", 
        "bio12_annual_precipitation",
        "bio15_precip_seasonality",
        "plants_richness",
        "vertebrates_richness",
        "elevation",
        "soil_ph",
        "ndvi",
        "human_footprint"
    ][:n_features]
    
    # Generate mock coefficients
    coefficients = np.random.randn(n_features + 1) * 0.5
    coefficients[0] = 2.5  # Intercept
    
    # Coefficient names
    coef_names = ["(Intercept)"] + feature_names
    
    # Generate mock variance-covariance matrix
    vcov = np.random.randn(n_features + 1, n_features + 1) * 0.01
    vcov = vcov @ vcov.T  # Make it positive definite
    std_errors = np.sqrt(np.diag(vcov))
    
    # Create model diagnostics for method1_full
    method1_full = {
        "coefficients": coefficients.tolist(),
        "coef_names": coef_names,
        "std_errors": std_errors.tolist(),
        "vcov": vcov.tolist(),
        "aic": 12345.6,
        "bic": 12456.7,
        "loglik": -6172.8,
        "deviance": 1234.5,
        "df_residual": n_samples - n_features - 1,
        "n_obs": n_samples,
        "theta": 2.5,
        "residuals_file": "method1_full_residuals.npz",
        # SHAP parameters - this is what we added
        "shap_params": {
            "coefficients": coefficients.tolist(),
            "coef_names": coef_names,
            "intercept_idx": 1,
            "feature_names": feature_names,
            "theta": 2.5,
            "link": "log",
            "family": "negative_binomial",
            "n_features": n_features
        }
    }
    
    # Create model diagnostics for method2_continuous (slightly different)
    coefficients2 = coefficients * 0.8
    method2_continuous = {
        "coefficients": coefficients2.tolist(),
        "coef_names": coef_names,
        "std_errors": (std_errors * 1.1).tolist(),
        "vcov": (vcov * 1.21).tolist(),
        "aic": 12350.2,
        "bic": 12461.3,
        "loglik": -6175.1,
        "deviance": 1238.9,
        "df_residual": n_samples - n_features - 1,
        "n_obs": n_samples,
        "theta": 2.3,
        "residuals_file": "method2_continuous_residuals.npz",
        "shap_params": {
            "coefficients": coefficients2.tolist(),
            "coef_names": coef_names,
            "intercept_idx": 1,
            "feature_names": feature_names,
            "theta": 2.3,
            "link": "log",
            "family": "negative_binomial",
            "n_features": n_features
        }
    }
    
    # Create full Julia results structure
    julia_results = {
        "models": {
            "method1_full": method1_full,
            "method2_continuous": method2_continuous,
            "method1_lr_test": {
                "lr_statistic": 45.6,
                "df": 2,
                "p_value": 0.0001
            }
        },
        "metadata": {
            "n_rows": n_samples,
            "n_cols": n_features + 200,  # Including MESFs
            "mesf_count": 200,
            "timestamp": "2024-08-31_00-00-00",
            "julia_version": "1.9.0"
        }
    }
    
    # Save as JSON (standard Julia output)
    json_file = output_dir / "julia_results_latest.json"
    with open(json_file, 'w') as f:
        json.dump(julia_results, f, indent=4)
    print(f"  ✓ Saved Julia results: {json_file}")
    
    # Generate background data for SHAP
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features) * np.array([10, 5, 100, 20, 50, 30, 500, 1, 0.2, 5][:n_features]),
        columns=feature_names
    )
    
    # Make some features more realistic
    if "bio01_mean_annual_temp" in X.columns:
        X["bio01_mean_annual_temp"] = np.clip(X["bio01_mean_annual_temp"], -10, 30)
    if "bio12_annual_precipitation" in X.columns:
        X["bio12_annual_precipitation"] = np.abs(X["bio12_annual_precipitation"]) + 200
    if "plants_richness" in X.columns:
        X["plants_richness"] = np.abs(X["plants_richness"].astype(int))
    if "vertebrates_richness" in X.columns:
        X["vertebrates_richness"] = np.abs(X["vertebrates_richness"].astype(int))
    
    # Save background data as Arrow
    background = X.sample(min(1000, len(X)), random_state=42)
    arrow_file = output_dir / "shap_background.arrow"
    feather.write_feather(background, arrow_file)
    print(f"  ✓ Saved background data: {arrow_file}")
    
    print(f"\n✓ Mock Julia standard output complete!")
    print(f"  Location: {output_dir}")
    
    return str(output_dir)

if __name__ == "__main__":
    output_dir = generate_mock_julia_output()
    print(f"\nNow run: python src/machine_learning/interpretation/shap_glm_analysis.py {output_dir}")