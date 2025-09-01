#!/usr/bin/env python3
"""
SHAP analysis for Julia GLM models using LinearExplainer.
Reads model parameters from Julia export and computes SHAP values efficiently.
"""

import json
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# No class needed - just pass coefficients directly to SHAP


def load_julia_results(results_path: Path) -> Dict[str, Any]:
    """Load Julia GLM results from standard JSON output."""
    # Try to find the latest results file
    if results_path.is_file():
        json_file = results_path
    else:
        # Look for julia_results_latest.json in the directory
        json_file = results_path / "julia_results_latest.json"
        if not json_file.exists():
            # Try to find any julia_results_*.json file
            json_files = list(results_path.glob("julia_results_*.json"))
            if json_files:
                json_file = sorted(json_files)[-1]  # Get the most recent
            else:
                raise FileNotFoundError(f"No Julia results found in {results_path}")
    
    with open(json_file, 'r') as f:
        return json.load(f)


def load_background_data(output_dir: Path) -> pd.DataFrame:
    """Load background data for SHAP from unified analysis data."""
    # Look for unified analysis data first (new format)
    arrow_file = output_dir / "analysis_data.arrow"
    if arrow_file.exists():
        logger.info(f"Loading background data from {arrow_file}")
        import pyarrow.feather as feather
        return feather.read_feather(arrow_file)
    
    # Fallback to old SHAP-specific file for backward compatibility
    old_arrow = output_dir / "shap_background.arrow"
    if old_arrow.exists():
        logger.info(f"Loading background data from {old_arrow} (legacy)")
        import pyarrow.feather as feather
        return feather.read_feather(old_arrow)
    
    # Fallback to CSV if exists
    csv_file = output_dir / "shap_background.csv"
    if csv_file.exists():
        logger.info(f"Loading background data from {csv_file}")
        return pd.read_csv(csv_file)
    
    logger.warning(f"No background data found in {output_dir}")
    return None


def compute_shap_values(model_params: Dict[str, Any], 
                       X: pd.DataFrame,
                       background: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Compute SHAP values for GLM using LinearExplainer.
    
    Args:
        model_params: Julia model parameters with coefficients and feature names
        X: Feature data for SHAP computation  
        background: Background data for SHAP (if None, uses X)
        
    Returns:
        Dictionary containing SHAP values and metadata
    """
    # Extract what we actually need - no wrapper class nonsense
    coefficients = np.array(model_params['coefficients'])
    coef_names = model_params['coef_names']
    
    # Split intercept and feature coefficients
    intercept = coefficients[0]
    beta = coefficients[1:]
    feature_names = [name for name in coef_names if name != '(Intercept)']
    
    # Validate features - fail loudly on mismatch
    missing = set(feature_names) - set(X.columns)
    if missing:
        raise ValueError(f"Missing features in data: {missing}")
    
    # Order features to match model
    X_ordered = X[feature_names].values
    background_ordered = (background[feature_names].values if background is not None 
                         else X_ordered)
    
    logger.info(f"Computing SHAP for {len(X)} samples, {len(feature_names)} features")
    
    # LinearExplainer is exact for GLM - that's all we need
    explainer = shap.LinearExplainer(
        (beta, intercept),
        background_ordered,
        feature_names=feature_names
    )
    
    shap_values = explainer.shap_values(X_ordered)
    
    return {
        'shap_values': shap_values,
        'expected_value': explainer.expected_value,
        'feature_names': feature_names,
        'X': X_ordered,
        'model_params': model_params
    }


def save_shap_results(shap_results: Dict[str, Any], 
                      output_dir: Path,
                      model_name: str) -> None:
    """
    Save SHAP results as a single Arrow file with metadata.
    
    Args:
        shap_results: Dictionary from compute_shap_values
        output_dir: Directory to save results
        model_name: Name of the model
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import pyarrow as pa
    import pyarrow.feather as feather
    
    # Combine features and SHAP values in one DataFrame
    combined_df = pd.DataFrame()
    
    # Add feature columns
    feature_names = shap_results['feature_names']
    for i, fname in enumerate(feature_names):
        combined_df[fname] = shap_results['X'][:, i]
    
    # Add SHAP value columns with prefix
    for i, fname in enumerate(feature_names):
        combined_df[f'shap_{fname}'] = shap_results['shap_values'][:, i]
    
    # Create Arrow table with metadata
    table = pa.Table.from_pandas(combined_df)
    
    # Add metadata to the Arrow schema
    metadata = {
        'model_name': model_name,
        'baseline': str(float(shap_results['expected_value']) if np.isscalar(shap_results['expected_value']) else float(shap_results['expected_value'][0])),
        'n_samples': str(len(shap_results['shap_values'])),
        'n_features': str(len(feature_names)),
        'feature_names': json.dumps(feature_names),
        'model_family': shap_results['model_params'].get('family', 'unknown'),
        'model_link': shap_results['model_params'].get('link', 'unknown')
    }
    
    # Add metadata to table
    table = table.replace_schema_metadata(metadata)
    
    # Save single Arrow file
    arrow_file = output_dir / f"{model_name}_shap_results.arrow"
    feather.write_feather(table, arrow_file)
    logger.info(f"Saved all SHAP results to single Arrow file: {arrow_file}")
    
    # Still save NPZ for backward compatibility with Python users
    np_file = output_dir / f"{model_name}_shap.npz"
    np.savez_compressed(
        np_file,
        shap_values=shap_results['shap_values'],
        X=shap_results['X'],
        expected_value=shap_results['expected_value'],
        feature_names=shap_results['feature_names']
    )
    logger.info(f"Saved numpy arrays to {np_file}")


def process_julia_glm_models(julia_output_dir: Path,
                            shap_output_dir: Path,
                            models_to_process: Optional[list] = None) -> None:
    """
    Process Julia GLM models for SHAP analysis from standard output.
    
    Args:
        julia_output_dir: Directory containing Julia results JSON
        shap_output_dir: Directory to save SHAP results
        models_to_process: List of model names to process (if None, process all)
    """
    julia_output_dir = Path(julia_output_dir)
    
    # Load Julia results
    try:
        julia_results = load_julia_results(julia_output_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    # Load background data
    background = load_background_data(julia_output_dir)
    
    # Get models from results
    if "models" not in julia_results:
        logger.error("No models found in Julia results")
        return
    
    models = julia_results["models"]
    logger.info(f"Found {len(models)} model(s) in Julia results")
    
    # Process each model
    for model_name, model_data in models.items():
        # Skip non-model entries (like LRT tests)
        if not isinstance(model_data, dict) or "shap_params" not in model_data:
            logger.info(f"Skipping {model_name} (no SHAP parameters)")
            continue
        
        if models_to_process and model_name not in models_to_process:
            logger.info(f"Skipping {model_name} (not in processing list)")
            continue
        
        logger.info(f"\nProcessing model: {model_name}")
        
        # Get SHAP parameters from model diagnostics
        shap_params = model_data["shap_params"]
        
        # Use background data for X if we don't have separate feature data
        if background is not None:
            X = background
            logger.info(f"Using background data ({len(X)} samples) for SHAP")
        else:
            logger.error(f"No feature data available for {model_name}")
            continue
        
        # Ensure feature names match
        feature_names = shap_params.get("feature_names", [])
        if feature_names:
            # Filter X to only include features used in the model
            available_features = [f for f in feature_names if f in X.columns]
            if len(available_features) != len(feature_names):
                logger.warning(f"Some features missing: {set(feature_names) - set(available_features)}")
            X = X[available_features]
        
        # Compute SHAP values
        shap_results = compute_shap_values(
            shap_params, 
            X, 
            background[available_features] if background is not None else None
        )
        
        # Save results
        model_output_dir = Path(shap_output_dir) / model_name
        save_shap_results(shap_results, model_output_dir, model_name)
        
        logger.info(f"Completed SHAP analysis for {model_name}")


def main():
    """Main entry point for command-line usage."""
    import argparse
    import subprocess
    
    parser = argparse.ArgumentParser(description='Compute SHAP values for Julia GLM models')
    parser.add_argument('julia_output_dir', type=str,
                       help='Directory containing Julia model exports')
    parser.add_argument('--output_dir', type=str, default='results/shap',
                       help='Directory to save SHAP results (default: results/shap)')
    parser.add_argument('--models', nargs='+', default=None,
                       help='Specific models to process (default: all)')
    parser.add_argument('--run_shapviz', action='store_true',
                       help='Also run R shapviz visualizations')
    
    args = parser.parse_args()
    
    # Process SHAP
    process_julia_glm_models(
        Path(args.julia_output_dir),
        Path(args.output_dir),
        args.models
    )
    
    # Optionally run R shapviz
    if args.run_shapviz:
        try:
            subprocess.run([
                'Rscript', 
                'src/machine_learning/visualization/shapviz_plots.R',
                str(args.output_dir),
                str(Path(args.output_dir).parent / 'shapviz')
            ], check=True)
            logger.info("Shapviz visualizations complete")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Shapviz failed (R or shapviz package not available)")


if __name__ == '__main__':
    main()