#!/usr/bin/env python3
"""
Generate mock Julia GLM export data for testing the SHAP pipeline.
Simulates what Julia's export_for_shap() function would produce.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

def generate_mock_data(output_dir="results/julia_mock", n_samples=1000, n_features=10):
    """Generate mock Julia GLM export for testing."""
    
    # Create output directories
    output_dir = Path(output_dir)
    shap_dir = output_dir / "shap_data"
    shap_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating mock Julia GLM export in {shap_dir}")
    
    # Set random seed for reproducibility
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
    
    # Generate mock coefficients (including intercept)
    coefficients = np.random.randn(n_features + 1) * 0.5
    coefficients[0] = 2.5  # Intercept
    
    # Coefficient names
    coef_names = ["(Intercept)"] + feature_names
    
    # Generate mock variance-covariance matrix
    vcov = np.random.randn(n_features + 1, n_features + 1) * 0.01
    vcov = vcov @ vcov.T  # Make it positive definite
    
    # Standard errors
    std_errors = np.sqrt(np.diag(vcov))
    
    # Create model export (what Julia would save)
    model_export = {
        "model_name": "method1_full",
        "coefficients": coefficients.tolist(),
        "coef_names": coef_names,
        "intercept": float(coefficients[0]),
        "theta": 2.5,  # Negative binomial dispersion
        "link": "log",
        "family": "negative_binomial",
        "n_features": n_features,
        "n_obs": n_samples,
        "vcov": vcov.tolist(),
        "std_errors": std_errors.tolist()
    }
    
    # Save model JSON
    model_file = shap_dir / "method1_full_model.json"
    with open(model_file, 'w') as f:
        json.dump(model_export, f, indent=4)
    print(f"  ✓ Saved model parameters: {model_file}")
    
    # Generate mock feature data
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features) * np.array([10, 5, 100, 20, 50, 30, 500, 1, 0.2, 5]),
        columns=feature_names
    )
    
    # Make some features more realistic
    X["bio01_mean_annual_temp"] = np.clip(X["bio01_mean_annual_temp"], -10, 30)
    X["bio12_annual_precipitation"] = np.abs(X["bio12_annual_precipitation"]) + 200
    X["plants_richness"] = np.abs(X["plants_richness"].astype(int))
    X["vertebrates_richness"] = np.abs(X["vertebrates_richness"].astype(int))
    X["elevation"] = np.clip(X["elevation"], 0, 5000)
    X["soil_ph"] = np.clip(X["soil_ph"], 4, 9)
    X["ndvi"] = np.clip(X["ndvi"], -0.2, 1.0)
    X["human_footprint"] = np.clip(X["human_footprint"], 0, 100)
    
    # Save background data (sample for SHAP)
    background = X.sample(min(1000, len(X)), random_state=42)
    background_file = shap_dir / "method1_full_background.csv"
    background.to_csv(background_file, index=False)
    print(f"  ✓ Saved background data: {background_file}")
    
    # Save full feature data
    features_file = shap_dir / "method1_full_features.csv"
    X.to_csv(features_file, index=False)
    print(f"  ✓ Saved feature data: {features_file}")
    
    # Create a second model for comparison
    model_export2 = model_export.copy()
    model_export2["model_name"] = "method2_continuous"
    model_export2["coefficients"] = (np.array(coefficients) * 0.8).tolist()  # Slightly different
    
    model_file2 = shap_dir / "method2_continuous_model.json"
    with open(model_file2, 'w') as f:
        json.dump(model_export2, f, indent=4)
    
    # Save data for second model
    background.to_csv(shap_dir / "method2_continuous_background.csv", index=False)
    X.to_csv(shap_dir / "method2_continuous_features.csv", index=False)
    print(f"  ✓ Saved second model: {model_file2}")
    
    # Save export summary
    summary = {
        "method1_full": str(model_file),
        "method2_continuous": str(model_file2)
    }
    
    summary_file = shap_dir / "shap_exports.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n✓ Mock Julia export complete!")
    print(f"  Models: method1_full, method2_continuous")
    print(f"  Location: {shap_dir}")
    
    return str(output_dir)

if __name__ == "__main__":
    output_dir = generate_mock_data()
    print(f"\nNow run: python src/machine_learning/interpretation/shap_glm_analysis.py {output_dir}")