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


class JuliaGLMPredictor:
    """Wrapper to create a predict function from Julia GLM parameters."""
    
    def __init__(self, model_params: Dict[str, Any]):
        """
        Initialize predictor from Julia model parameters.
        
        Args:
            model_params: Dictionary containing coefficients, coef_names, etc.
        """
        self.coefficients = np.array(model_params['coefficients'])
        self.coef_names = model_params['coef_names']
        self.intercept = self.coefficients[0]  # First coef is intercept
        self.beta = self.coefficients[1:]  # Rest are feature coefficients
        self.theta = model_params.get('theta', 1.0)
        self.link = model_params.get('link', 'log')
        self.family = model_params.get('family', 'negative_binomial')
        
        # Feature names (excluding intercept)
        self.feature_names = [name for name in self.coef_names if name != '(Intercept)']
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using Julia GLM parameters.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predicted values after applying inverse link function
        """
        # Linear predictor
        linear_pred = self.intercept + X @ self.beta
        
        # Apply inverse link function
        if self.link == 'log':
            return np.exp(linear_pred)
        elif self.link == 'identity':
            return linear_pred
        else:
            raise ValueError(f"Unsupported link function: {self.link}")
    
    def predict_linear(self, X: np.ndarray) -> np.ndarray:
        """
        Get linear predictor (before inverse link).
        This is what LinearExplainer needs for log-link models.
        """
        return self.intercept + X @ self.beta


def load_julia_model(model_path: Path) -> Dict[str, Any]:
    """Load Julia model parameters from JSON file."""
    with open(model_path, 'r') as f:
        return json.load(f)


def load_background_data(shap_dir: Path, model_name: str) -> pd.DataFrame:
    """Load background data for SHAP from CSV."""
    background_file = shap_dir / f"{model_name}_background.csv"
    if background_file.exists():
        logger.info(f"Loading background data from {background_file}")
        return pd.read_csv(background_file)
    else:
        logger.warning(f"No background data found at {background_file}")
        return None


def compute_shap_values(model_params: Dict[str, Any], 
                       X: pd.DataFrame,
                       background: Optional[pd.DataFrame] = None,
                       use_linear_explainer: bool = True) -> Dict[str, Any]:
    """
    Compute SHAP values using LinearExplainer (exact for GLM).
    
    Args:
        model_params: Julia model parameters
        X: Feature data for SHAP computation
        background: Background data for SHAP (if None, uses X)
        use_linear_explainer: If True, use LinearExplainer (exact), else KernelExplainer
        
    Returns:
        Dictionary containing SHAP values and metadata
    """
    predictor = JuliaGLMPredictor(model_params)
    
    # Ensure feature order matches model
    feature_names = predictor.feature_names
    X_ordered = X[feature_names].values
    
    if background is None:
        background = X
    background_ordered = background[feature_names].values
    
    logger.info(f"Computing SHAP values for {len(X)} samples with {len(feature_names)} features")
    
    if use_linear_explainer:
        # LinearExplainer is exact for linear models
        # For GLM with log link, we explain the linear predictor
        logger.info("Using LinearExplainer (exact computation)")
        
        # Create explainer with coefficients and intercept
        explainer = shap.LinearExplainer(
            (predictor.beta, predictor.intercept),
            background_ordered,
            feature_names=feature_names
        )
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X_ordered)
        
    else:
        # KernelExplainer for comparison (approximate, slower)
        logger.info("Using KernelExplainer (approximate computation)")
        
        explainer = shap.KernelExplainer(
            predictor.predict,
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
    Save SHAP results for visualization in Python and R.
    
    Args:
        shap_results: Dictionary from compute_shap_values
        output_dir: Directory to save results
        model_name: Name of the model
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save SHAP values as CSV for R shapviz
    shap_df = pd.DataFrame(
        shap_results['shap_values'],
        columns=shap_results['feature_names']
    )
    shap_csv = output_dir / f"{model_name}_shap_values.csv"
    shap_df.to_csv(shap_csv, index=False)
    logger.info(f"Saved SHAP values to {shap_csv}")
    
    # Save feature data
    feature_df = pd.DataFrame(
        shap_results['X'],
        columns=shap_results['feature_names']
    )
    feature_csv = output_dir / f"{model_name}_features.csv"
    feature_df.to_csv(feature_csv, index=False)
    logger.info(f"Saved feature data to {feature_csv}")
    
    # Save metadata for R
    metadata = {
        'model_name': model_name,
        'baseline': float(shap_results['expected_value']) if np.isscalar(shap_results['expected_value']) else float(shap_results['expected_value'][0]),
        'n_samples': len(shap_results['shap_values']),
        'n_features': len(shap_results['feature_names']),
        'feature_names': shap_results['feature_names'],
        'model_family': shap_results['model_params'].get('family', 'unknown'),
        'model_link': shap_results['model_params'].get('link', 'unknown')
    }
    
    metadata_json = output_dir / f"{model_name}_shap_metadata.json"
    with open(metadata_json, 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Saved metadata to {metadata_json}")
    
    # Save as numpy arrays for fast Python loading
    np_file = output_dir / f"{model_name}_shap.npz"
    np.savez_compressed(
        np_file,
        shap_values=shap_results['shap_values'],
        X=shap_results['X'],
        expected_value=shap_results['expected_value'],
        feature_names=shap_results['feature_names']
    )
    logger.info(f"Saved numpy arrays to {np_file}")


def create_basic_visualizations(shap_results: Dict[str, Any],
                               output_dir: Path,
                               model_name: str) -> None:
    """
    Create basic SHAP visualizations using matplotlib.
    
    Args:
        shap_results: Dictionary from compute_shap_values
        output_dir: Directory to save plots
        model_name: Name of the model
    """
    import matplotlib.pyplot as plt
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    shap_values = shap_results['shap_values']
    feature_names = shap_results['feature_names']
    X = shap_results['X']
    
    # 1. Feature importance (mean absolute SHAP)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    importance_order = np.argsort(mean_abs_shap)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_names)), 
             mean_abs_shap[importance_order],
             color='steelblue')
    plt.yticks(range(len(feature_names)), 
               [feature_names[i] for i in importance_order])
    plt.xlabel('Mean |SHAP value|')
    plt.title(f'Feature Importance - {model_name}')
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_importance.png", dpi=150)
    plt.close()
    logger.info(f"Saved importance plot to {output_dir / f'{model_name}_importance.png'}")
    
    # 2. Summary plot (if we have many samples)
    if len(shap_values) > 100:
        # Create a simple beeswarm-like plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for feat_idx in importance_order[:10]:  # Top 10 features
            y_pos = len(importance_order[:10]) - list(importance_order[:10]).index(feat_idx) - 1
            
            # Scatter with jitter
            y_jitter = np.random.normal(0, 0.1, len(shap_values))
            ax.scatter(shap_values[:, feat_idx], 
                      y_pos + y_jitter,
                      c=X[:, feat_idx],
                      cmap='coolwarm',
                      alpha=0.5,
                      s=5)
        
        ax.set_yticks(range(len(importance_order[:10])))
        ax.set_yticklabels([feature_names[i] for i in importance_order[:10][::-1]])
        ax.set_xlabel('SHAP value')
        ax.set_title(f'SHAP Summary - {model_name}')
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{model_name}_summary.png", dpi=150)
        plt.close()
        logger.info(f"Saved summary plot to {output_dir / f'{model_name}_summary.png'}")


def process_julia_glm_models(julia_output_dir: Path,
                            shap_output_dir: Path,
                            models_to_process: Optional[list] = None) -> None:
    """
    Process all Julia GLM models for SHAP analysis.
    
    Args:
        julia_output_dir: Directory containing Julia model exports
        shap_output_dir: Directory to save SHAP results
        models_to_process: List of model names to process (if None, process all)
    """
    julia_output_dir = Path(julia_output_dir)
    shap_dir = julia_output_dir / 'shap_data'
    
    if not shap_dir.exists():
        logger.error(f"SHAP data directory not found: {shap_dir}")
        return
    
    # Find all model JSON files
    model_files = list(shap_dir.glob('*_model.json'))
    
    if not model_files:
        logger.error(f"No model files found in {shap_dir}")
        return
    
    logger.info(f"Found {len(model_files)} model(s) to process")
    
    for model_file in model_files:
        model_name = model_file.stem.replace('_model', '')
        
        if models_to_process and model_name not in models_to_process:
            logger.info(f"Skipping {model_name} (not in processing list)")
            continue
        
        logger.info(f"\nProcessing model: {model_name}")
        
        # Load model parameters
        model_params = load_julia_model(model_file)
        
        # Load background data
        background = load_background_data(shap_dir, model_name)
        
        # Load feature data (try full data first, then background)
        feature_file = shap_dir / f"{model_name}_features.csv"
        if feature_file.exists():
            X = pd.read_csv(feature_file)
            logger.info(f"Loaded {len(X)} samples from feature file")
        elif background is not None:
            X = background
            logger.info(f"Using background data ({len(X)} samples) for SHAP")
        else:
            logger.error(f"No feature data found for {model_name}")
            continue
        
        # Compute SHAP values
        shap_results = compute_shap_values(
            model_params, 
            X, 
            background,
            use_linear_explainer=True  # Use exact LinearExplainer
        )
        
        # Save results
        model_output_dir = Path(shap_output_dir) / model_name
        save_shap_results(shap_results, model_output_dir, model_name)
        
        # Create basic visualizations
        create_basic_visualizations(shap_results, model_output_dir, model_name)
        
        logger.info(f"Completed SHAP analysis for {model_name}")


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute SHAP values for Julia GLM models')
    parser.add_argument('julia_output_dir', type=str,
                       help='Directory containing Julia model exports')
    parser.add_argument('--output_dir', type=str, default='results/shap',
                       help='Directory to save SHAP results (default: results/shap)')
    parser.add_argument('--models', nargs='+', default=None,
                       help='Specific models to process (default: all)')
    
    args = parser.parse_args()
    
    process_julia_glm_models(
        Path(args.julia_output_dir),
        Path(args.output_dir),
        args.models
    )


if __name__ == '__main__':
    main()