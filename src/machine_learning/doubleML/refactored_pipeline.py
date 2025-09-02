#!/usr/bin/env python3
"""Refactored Causal Effect Pipeline

Principle: Clear separation between homogeneous (LinearDML) and heterogeneous (CausalForestDML) effects
Author: Following Linus's KISS principle
"""

# Standard library imports
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

# Third-party imports
import numpy as np
import pandas as pd
import psutil
import yaml
import pyarrow.feather as feather
import pyarrow.parquet as pq
from sklearn.multioutput import MultiOutputRegressor
from econml.dml import CausalForestDML

# Local imports
from src.abstractions.ml_io import save_pipeline_results, load_pipeline_results
from src.machine_learning.doubleML.optimized_lgbm import OptimizedLGBMRegressor
from src.machine_learning.doubleML.linear_dml_with_residuals import LinearDMLWithResiduals as LinearDML


# ==================== Configuration Loading ====================

def load_config(config_path: str = None) -> Dict:
    """
    Load configuration from YAML file. NO DEFAULTS, NO FALLBACKS.
    
    Args:
        config_path: Path to config file. If None, looks for default location.
        
    Returns:
        Config dictionary
        
    Raises:
        FileNotFoundError: If config file not found
        ValueError: If required parameters are missing
    """
    if config_path is None:
        # Try default location
        config_path = Path(__file__).parent.parent.parent / "config/ml/econml_pipeline_config.yml"
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please create the config file or specify the path explicitly."
        )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required parameters exist
    required_params = [
        'output_dir',  # REQUIRED - no fallback
        'num_threads', 'cv_folds', 'random_state', 'alpha',
        'mesf_cache_dir', 'spatial_step',
        'outcome', 'treatments', 'effect_modifiers',
        'mc_iters', 'inference_homo', 'inference_hetero',
        'n_estimators', 'max_depth', 'min_samples_leaf',
        'objective', 'metric', 'learning_rate', 'num_iterations'
    ]
    
    missing = [p for p in required_params if p not in config]
    if missing:
        raise ValueError(
            f"Missing required configuration parameters: {missing}\n"
            f"Check your config file: {config_path}"
        )
    
    return config


# ==================== Data Preprocessing ====================

class DataPreprocessor:
    """
    Complete data preprocessing with MESF, standardization, and feature engineering
    """
    
    def __init__(self, cache_dir: str):
        """
        Args:
            cache_dir: MESF cache directory (REQUIRED - no default)
        """
        if not cache_dir:
            raise ValueError("cache_dir is required - no default provided")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data = None
        self.feature_info = {}
    
    def load_and_preprocess(self, 
                          data_path: str, 
                          spatial_step: int,
                          method: str = 'dml') -> Dict[str, np.ndarray]:
        """Load data and apply all preprocessing steps"""
        
        print(f"Loading data from: {data_path}")
        
        # Always use feather for Arrow format (3x faster)
        table = feather.read_table(data_path)
        
        # Convert to numpy dict immediately - all as float32
        data = {}
        n_rows = len(table)
        print(f"Loading {n_rows:,} rows, {len(table.column_names)} columns")
        
        for col in table.column_names:
            data[col] = table[col].to_numpy().astype(np.float32)
        
        # Spatial sampling
        if spatial_step > 1:
            for col in data:
                data[col] = data[col][::spatial_step]
            n_rows = len(data[list(data.keys())[0]])
            print(f"Sampled to {n_rows:,} rows (step={spatial_step})")
        
        # Add MESF spatial features
        data = self._add_mesf_features(data, spatial_step)
        
        # Apply preprocessing transformations
        data = self._preprocess_features(data, method)
        
        return data
    
    def _add_mesf_features(self, data: Dict[str, np.ndarray], spatial_step: int) -> Dict[str, np.ndarray]:
        """Load pre-computed MESF spatial features from Julia cache"""
        
        n_rows = len(data[list(data.keys())[0]])
        
        # Look for Julia-generated MESF cache
        # Try multiple patterns to find MESF files
        patterns = [
            f"mesf_n{n_rows}_k*.npy",           # Most flexible: mesf_n5183621_k200.npy
            f"mesf_n{n_rows}_k200.npy",         # Exact match
            f"mesf_step{spatial_step}_n*.npy",  # Old naming convention
        ]
        
        mesf_files = []
        for pattern in patterns:
            mesf_files = list(self.cache_dir.glob(pattern))
            if mesf_files:
                break
        
        if mesf_files:
            # Use the first matching file
            mesf_file = mesf_files[0]
            print(f"Loading cached MESF: {mesf_file}")
            mesf_array = np.load(mesf_file).astype(np.float32)
            
            if mesf_array.shape[0] != n_rows:
                raise ValueError(
                    f"MESF shape mismatch: cached has {mesf_array.shape[0]} rows, "
                    f"but data has {n_rows} rows. Please recompute MESF using Julia."
                )
            
            # Add MESF columns to dict
            n_mesf_cols = mesf_array.shape[1]
            for i in range(n_mesf_cols):
                data[f'mesf_{i+1}'] = mesf_array[:, i]
            print(f"Added {n_mesf_cols} MESF columns from cache")
        else:
            raise FileNotFoundError(
                f"No MESF cache found for {n_rows} samples in {self.cache_dir}\n"
                f"Searched patterns: {', '.join(patterns)}\n"
                f"Please run Julia MESF computation first:\n"
                f"  julia src/machine_learning/spatial/mesf.jl --data your_data.arrow --spatial-step {spatial_step}"
            )
        
        return data
    
    def _preprocess_features(self, data: Dict[str, np.ndarray], method: str) -> Dict[str, np.ndarray]:
        """Apply preprocessing transformations"""
        
        # Remove missing values
        n_rows = len(data[list(data.keys())[0]])
        valid_mask = np.ones(n_rows, dtype=bool)
        for col in data.values():
            valid_mask &= ~np.isnan(col)
        
        if not np.all(valid_mask):
            for key in data:
                data[key] = data[key][valid_mask]
            print(f"Removed {n_rows - np.sum(valid_mask)} rows with missing values")
        
        # Create fungi_richness (combine AM and ECM)
        if 'am_fungi_richness' in data and 'ecm_fungi_richness' in data:
            data['fungi_richness'] = data['am_fungi_richness'] + data['ecm_fungi_richness']
        
        # Standardize naming
        if 'plant_richness' in data:
            data['plants_richness'] = data['plant_richness']
        if 'animal_richness' in data:
            data['vertebrates_richness'] = data['animal_richness']
        elif 'terrestrial_richness' in data:
            data['vertebrates_richness'] = data['terrestrial_richness']
        
        # Standardize all continuous variables (2SD method)
        # Climate variables
        climate_vars = ['bio01', 'bio04', 'bio12', 'bio15']
        # Topographic variables  
        topo_vars = ['elevation', 'slope', 'aspect_cos', 'aspect_sin', 'tpi']
        # Soil variables
        soil_vars = ['soil_ph', 'soil_organic_carbon', 'sand_content_pct']
        # Water balance variables
        water_vars = ['pet_mm_month']
        
        # Combine all variables to standardize
        all_vars_to_std = climate_vars + topo_vars + soil_vars + water_vars
        
        for var in all_vars_to_std:
            if var in data:
                mean_val = np.mean(data[var])
                std_val = np.std(data[var])
                # Use 2SD standardization for interpretability
                data[f'{var}_std'] = (data[var] - mean_val) / (2 * std_val)
                print(f"  Standardized {var}: mean={mean_val:.2f}, 2*sd={2*std_val:.2f}")
        
        # Method-specific transformations
        if method in ['dml', 'method3']:
            # Keep raw counts for DML
            print("Keeping raw counts for DML")
            
            # Add interaction term
            if 'plants_richness' in data and 'vertebrates_richness' in data:
                data['plants_x_vertebrates'] = data['plants_richness'] * data['vertebrates_richness']
                print("Added interaction term: plants_x_vertebrates")
        else:
            # GLM methods need log transforms
            # Keep taxa as raw float values - no transformations for LightGBM
            # LightGBM handles raw values better than log-transformed values
            if 'fungi_richness' in data:
                data['fungi_richness'] = data['fungi_richness'].astype(np.float32)
            if 'plants_richness' in data:
                data['plants_richness'] = data['plants_richness'].astype(np.float32)
            if 'vertebrates_richness' in data:
                data['vertebrates_richness'] = data['vertebrates_richness'].astype(np.float32)
        
        n_final = len(data[list(data.keys())[0]])
        print(f"Final shape: ({n_final}, {len(data)} features)")
        return data
        
    def prepare_for_dml(self, 
                       data: Union[pd.DataFrame, Dict[str, np.ndarray]],
                       outcome: str,
                       treatments: List[str],
                       confounders: List[str],
                       effect_modifiers: Optional[List[str]] = None) -> Tuple:
        """
        Prepare data for DML estimation
        
        Returns:
            Tuple of (Y, T, X, W) where X is None for homogeneous case
        """
        # Convert DataFrame to dict if needed
        if isinstance(data, pd.DataFrame):
            data = {col: data[col].values.astype(np.float32) for col in data.columns}
        
        # Extract arrays
        Y = data[outcome]
        T = np.column_stack([data[t] for t in treatments]) if len(treatments) > 1 else data[treatments[0]].reshape(-1, 1)
        
        # Handle effect modifiers for heterogeneous case
        X = None
        if effect_modifiers:
            X = np.column_stack([data[m] for m in effect_modifiers])
            # Remove effect modifiers from confounders
            confounders = [c for c in confounders if c not in effect_modifiers]
        
        # Confounders
        W = np.column_stack([data[c] for c in confounders]) if confounders else None
        
        # Remove rows with NaN values
        mask = ~np.isnan(Y)
        if T.ndim > 1:
            mask &= ~np.any(np.isnan(T), axis=1)
        else:
            mask &= ~np.isnan(T).ravel()
        
        if X is not None:
            if X.ndim > 1:
                mask &= ~np.any(np.isnan(X), axis=1)
            else:
                mask &= ~np.isnan(X).ravel()
                
        if W is not None:
            if W.ndim > 1:
                mask &= ~np.any(np.isnan(W), axis=1)
            else:
                mask &= ~np.isnan(W).ravel()
        
        # Apply mask
        n_before = len(Y)
        Y = Y[mask]
        T = T[mask] if T.ndim > 1 else T[mask].reshape(-1, 1)
        if X is not None:
            X = X[mask] if X.ndim > 1 else X[mask].reshape(-1, 1)
        if W is not None:
            W = W[mask] if W.ndim > 1 else W[mask].reshape(-1, 1)
        
        n_after = len(Y)
        if n_before != n_after:
            print(f"Removed {n_before - n_after} rows with NaN values ({n_after} remaining)")
        
        # Store info for later reference
        self.feature_info = {
            'n_samples': len(Y),
            'outcome': outcome,
            'treatments': treatments,
            'confounders': confounders,
            'effect_modifiers': effect_modifiers
        }
        
        return Y, T, X, W


# ==================== Shared Utilities ====================

def create_nuisance_models(n_samples: int, n_treatments: int, num_threads: int, config: Dict):
    """
    Create LightGBM models for nuisance functions
    
    Args:
        n_samples: Number of samples in dataset
        n_treatments: Number of treatment variables
        num_threads: Number of threads for LightGBM (None = all physical cores)
    
    Returns:
        model_y: Model for outcome nuisance
        model_t: Model for treatment nuisance
    """
    model_y = OptimizedLGBMRegressor(
        config=config,
        dataset_size=n_samples,
        num_threads=num_threads
    )
    
    # Treatment model(s)
    if n_treatments > 1:
        # Multiple treatments need separate models
        model_t = MultiOutputRegressor(
            OptimizedLGBMRegressor(
                config=config,
                dataset_size=n_samples,
                num_threads=num_threads
            )
        )
    else:
        # Single treatment
        model_t = OptimizedLGBMRegressor(
            config=config,
            dataset_size=n_samples,
            num_threads=num_threads
        )
    
    return model_y, model_t


# ==================== Homogeneous Effects Estimator ====================

class HomogeneousEffectEstimator:
    """
    Estimates constant treatment effects using LinearDML
    No heterogeneity, no bullshit
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary (REQUIRED - no defaults)
        """
        if not config:
            raise ValueError("config is required - no defaults provided")
        
        self.n_folds = config['cv_folds']
        self.num_threads = config['num_threads']
        self.random_state = config['random_state']
        self.alpha = config['alpha']
        self.mc_iters = config['mc_iters']
        self.inference = config['inference_homo']
        self.config = config  # Store full config for nuisance models
        self.estimator = None
        self.results = None
        self.fit_time = None
        
    def fit(self, Y: np.ndarray, T: np.ndarray, W: np.ndarray,
            treatment_names: Optional[List[str]] = None) -> 'HomogeneousEffectEstimator':
        """
        Fit LinearDML for constant effects
        
        Args:
            Y: Outcome variable
            T: Treatment variables (can be multiple)
            W: Confounders
            treatment_names: Names for treatments (for results)
        """
        print("\n" + "="*70)
        print("FITTING HOMOGENEOUS EFFECTS MODEL (LinearDML)")
        print("="*70)
        
        # Store data for later use (e.g., saving/loading)
        self.last_Y = Y
        self.last_T = T
        self.last_W = W
        self.last_X = None  # Homogeneous doesn't use X
        
        n_samples = len(Y)
        n_treatments = T.shape[1] if T.ndim > 1 else 1
        n_confounders = W.shape[1] if W is not None else 0
        
        print(f"Data shape: {n_samples:,} samples")
        print(f"Treatments: {n_treatments}")
        print(f"Confounders: {n_confounders}")
        
        # Create standardized nuisance models with config
        model_y, model_t = create_nuisance_models(n_samples, n_treatments, self.num_threads, self.config)
        
        # Create and fit LinearDML
        self.estimator = LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=False,
            cv=self.n_folds,
            mc_iters=self.mc_iters,
            random_state=self.random_state
        )
        
        print(f"Fitting with {self.n_folds}-fold cross-validation...")
        start_time = time.time()
        
        self.estimator.fit(
            Y, T, 
            X=None,  # No effect modifiers for homogeneous
            W=W,
            inference=self.inference,
            cache_values=True
        )
        
        self.fit_time = time.time() - start_time
        print(f"Completed in {self.fit_time:.2f} seconds")
        
        # Extract results immediately
        self.results = self._extract_results(treatment_names)
        self._print_summary()
        
        return self
    
    def _extract_results(self, treatment_names: Optional[List[str]] = None) -> Dict:
        """Extract results from fitted LinearDML"""
        
        # Get inference object
        inference = self.estimator.intercept__inference()
        
        # Extract all statistics
        coef = self.estimator.intercept_
        stderr = inference.stderr
        pvalues = inference.pvalue(value=0)
        z_stats = inference.zstat(value=0)
        
        # Handle confidence intervals (can be tuple or array)
        conf_int = inference.conf_int(alpha=0.05)
        if isinstance(conf_int, tuple):
            ci_lower = conf_int[0]
            ci_upper = conf_int[1]
        else:
            ci_lower = conf_int[:, 0]
            ci_upper = conf_int[:, 1]
        
        # Format results
        n_treatments = len(coef)
        if treatment_names is None:
            treatment_names = [f'Treatment_{i+1}' for i in range(n_treatments)]
        
        treatments = {}
        for i, name in enumerate(treatment_names):
            treatments[name] = {
                'coefficient': float(coef[i]),
                'std_error': float(stderr[i]),
                'z_statistic': float(z_stats[i]),
                'p_value': float(pvalues[i]),
                'ci_95': [float(ci_lower[i]), float(ci_upper[i])],
                'significant': bool(pvalues[i] < 0.05)
            }
        
        return {
            'type': 'homogeneous',
            'method': 'LinearDML',
            'n_folds': self.n_folds,
            'fit_time': self.fit_time,
            'treatments': treatments,
            'coefficients': coef.tolist(),
            'std_errors': stderr.tolist()
        }
    
    def _print_summary(self):
        """Print summary of results"""
        print("\nTREATMENT EFFECTS (Homogeneous):")
        print("-" * 50)
        
        for name, stats in self.results['treatments'].items():
            print(f"\n{name}:")
            print(f"  Coefficient: {stats['coefficient']:.6f}")
            print(f"  Std Error: {stats['std_error']:.6f}")
            print(f"  Z-stat: {stats['z_statistic']:.2f}")
            print(f"  P-value: {stats['p_value']:.4f}")
            print(f"  95% CI: [{stats['ci_95'][0]:.6f}, {stats['ci_95'][1]:.6f}]")
            if stats['significant']:
                print(f"  ✓ Statistically significant")


# ==================== Heterogeneous Effects Estimator ====================

class HeterogeneousEffectEstimator:
    """
    Estimates heterogeneous treatment effects using CausalForestDML
    Discovers non-linear patterns automatically
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary (REQUIRED - no defaults)
        """
        if not config:
            raise ValueError("config is required - no defaults provided")
        
        self.n_estimators = config['n_estimators']
        self.max_depth = config['max_depth']
        self.min_samples_leaf = config['min_samples_leaf']
        self.subforest_size = config['subforest_size']
        self.num_threads = config['num_threads']
        self.cv_folds = config['cv_folds']
        self.random_state = config['random_state']
        self.alpha = config['alpha']
        self.inference = config['inference_hetero']
        # New parameters for better heterogeneity
        self.max_samples = config.get('max_samples', 0.45)
        self.min_impurity_decrease = config.get('min_impurity_decrease', 0.0)
        self.min_balancedness_tol = config.get('min_balancedness_tol', 0.45)
        self.honest = config.get('honest', True)
        self.config = config  # Store full config for nuisance models
        self.estimator = None
        self.results = None
        self.fit_time = None
        
    def fit(self, Y: np.ndarray, T: np.ndarray, X: np.ndarray, W: np.ndarray,
            treatment_names: Optional[List[str]] = None,
            modifier_names: Optional[List[str]] = None) -> 'HeterogeneousEffectEstimator':
        """
        Fit CausalForestDML for heterogeneous effects
        
        Args:
            Y: Outcome variable
            T: Treatment variables
            X: Effect modifiers (REQUIRED!)
            W: Confounders
            treatment_names: Names for treatments
            modifier_names: Names for effect modifiers
        """
        if X is None or (X.ndim == 1 and len(X) == 0) or (X.ndim == 2 and X.shape[1] == 0):
            raise ValueError("Effect modifiers X cannot be None/empty for heterogeneous effects!")
        
        print("\n" + "="*70)
        print("FITTING HETEROGENEOUS EFFECTS MODEL (CausalForestDML)")
        print("="*70)
        
        # Store data for later use (e.g., saving/loading)
        self.last_Y = Y
        self.last_T = T
        self.last_X = X
        self.last_W = W
        
        n_samples = len(Y)
        n_treatments = T.shape[1] if T.ndim > 1 else 1
        n_modifiers = X.shape[1] if X.ndim > 1 else 1
        n_confounders = W.shape[1] if W is not None else 0
        
        print(f"Data shape: {n_samples:,} samples")
        print(f"Treatments: {n_treatments}")
        print(f"Effect modifiers: {n_modifiers}")
        print(f"Confounders: {n_confounders}")
        
        # Use configured min_samples_leaf directly - no adjustment needed
        actual_min_samples_leaf = self.min_samples_leaf
        
        # LOG CRITICAL PARAMETERS
        print("\n" + "="*70)
        print("CAUSALFOREST PARAMETERS (DEBUG)")
        print("="*70)
        print(f"min_samples_leaf configured: {self.min_samples_leaf}")
        print(f"min_samples_leaf actual: {actual_min_samples_leaf}")
        print(f"n_estimators: {self.n_estimators}")
        print(f"max_depth: {self.max_depth}")
        print(f"max_samples: {self.max_samples}")
        print(f"honest: {self.honest}")
        print(f"subforest_size: {self.subforest_size}")
        print(f"num_threads: {self.num_threads}")
        print("="*70 + "\n")
        
        # Create standardized nuisance models with config
        model_y, model_t = create_nuisance_models(n_samples, n_treatments, self.num_threads, self.config)
        
        # Create and fit CausalForestDML
        # Ensure n_estimators is divisible by subforest_size
        actual_n_estimators = self.n_estimators
        if actual_n_estimators % self.subforest_size != 0:
            actual_n_estimators = (actual_n_estimators // self.subforest_size + 1) * self.subforest_size
            print(f"Adjusted n_estimators from {self.n_estimators} to {actual_n_estimators} (divisible by {self.subforest_size})")
        
        self.estimator = CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            n_estimators=actual_n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=actual_min_samples_leaf,
            max_samples=self.max_samples,
            # min_impurity_decrease=self.min_impurity_decrease,  # REMOVED: Causes zeros with larger datasets
            min_balancedness_tol=self.min_balancedness_tol,
            honest=self.honest,
            discrete_treatment=False,
            cv=self.cv_folds,
            random_state=self.random_state,
            n_jobs=self.num_threads,
            inference=self.inference  # Pass inference directly, 'blb' is valid
        )
        
        print(f"Fitting CausalForest with {actual_n_estimators} trees...")
        start_time = time.time()
        
        self.estimator.fit(
            Y, T,
            X=X,  # Effect modifiers REQUIRED
            W=W,
            cache_values=True
            # Note: inference is already set in the constructor, don't pass it here
        )
        
        self.fit_time = time.time() - start_time
        print(f"Completed in {self.fit_time:.2f} seconds")
        
        # Extract results
        self.results = self._extract_results(treatment_names, modifier_names)
        self._print_summary()
        
        return self
    
    def _extract_results(self, 
                        treatment_names: Optional[List[str]] = None,
                        modifier_names: Optional[List[str]] = None) -> Dict:
        """Extract results from fitted CausalForestDML"""
        
        # Get feature importances
        importances = self.estimator.feature_importances_
        
        n_modifiers = len(importances)
        if modifier_names is None:
            modifier_names = [f'Modifier_{i+1}' for i in range(n_modifiers)]
        
        # Create feature importance dict
        feature_importance = {
            name: float(imp) for name, imp in zip(modifier_names, importances)
        }
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        
        return {
            'type': 'heterogeneous',
            'method': 'CausalForestDML',
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'fit_time': self.fit_time,
            'feature_importances': feature_importance,
            'sorted_features': sorted_features
        }
    
    def _print_summary(self):
        """Print summary of results"""
        print("\nFEATURE IMPORTANCES (Heterogeneity drivers):")
        print("-" * 50)
        
        for feat, imp in self.results['sorted_features']:
            print(f"{feat:20s}: {imp:.4f}")
    
    def predict_effects(self, X_new: np.ndarray) -> np.ndarray:
        """Predict treatment effects for new data points"""
        if self.estimator is None:
            raise ValueError("Model not fitted yet!")
        return self.estimator.effect(X_new)
    
    def predict_interval(self, X_new: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Predict treatment effects with confidence intervals"""
        if self.estimator is None:
            raise ValueError("Model not fitted yet!")
        return self.estimator.effect_interval(X_new, alpha=alpha)


# ==================== Unified Pipeline Interface ====================

class CausalEffectPipeline:
    """
    High-level interface for causal effect estimation
    Clear separation: use estimate_homogeneous OR estimate_heterogeneous
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary (REQUIRED - no defaults)
        """
        if not config:
            raise ValueError("config is required - no defaults provided")
        
        self.config = config
        self.preprocessor = DataPreprocessor(cache_dir=config['mesf_cache_dir'])
        self.estimator = None
        self.last_results = None
        self.data = None
    
    def load_data(self, data_path: str, spatial_step: Optional[int] = None) -> 'CausalEffectPipeline':
        """
        Load and preprocess data using full preprocessing pipeline
        
        Args:
            data_path: Path to parquet file
            spatial_step: Spatial sampling step (1 = no sampling)
        """
        print("="*70)
        print("LOADING AND PREPROCESSING DATA")
        print("="*70)
        
        # Use spatial_step from config if not provided
        if spatial_step is None:
            spatial_step = self.config['spatial_step']
        
        self.data = self.preprocessor.load_and_preprocess(
            data_path, spatial_step, method='dml'
        )
        
        return self
    
    def fit_homogeneous(self) -> Dict:
        """
        Fit homogeneous effects model using 3 SEPARATE single-treatment models.
        
        This fits:
        1. fungi ~ plants (controlling for W)
        2. plants ~ vertebrates (controlling for W)
        3. vertebrates ~ fungi (controlling for W)
        
        Returns residuals for all three taxa for RDA analysis.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print("="*70)
        print("HOMOGENEOUS EFFECTS: 3 SINGLE-TREATMENT MODELS")
        print("="*70)
        
        # Define the three taxa
        taxa = ['fungi_richness', 'plants_richness', 'vertebrates_richness']
        
        # Define confounders from config
        climate = self.config.get('climate_confounders', [])
        topo = self.config.get('topographic_confounders', [])
        soil = self.config.get('soil_confounders', [])
        water = self.config.get('water_balance_confounders', [])
        mesf_cols = [col for col in self.data.keys() if col.startswith('mesf_')]
        
        confounders = climate + topo + soil + water + mesf_cols
        confounders = [c for c in confounders if c in self.data]
        
        W = np.column_stack([self.data[c] for c in confounders]) if confounders else None
        
        # Store results from all 3 models
        all_residuals = {}
        all_coefficients = {}
        all_models = {}
        
        # Define the 3 pairwise relationships
        pairs = [
            ('fungi_richness', 'plants_richness'),
            ('plants_richness', 'vertebrates_richness'),
            ('vertebrates_richness', 'fungi_richness')
        ]
        
        for outcome, treatment in pairs:
            print(f"\nModel: {outcome} ~ {treatment}")
            print("-"*40)
            
            Y = self.data[outcome]
            T = self.data[treatment].reshape(-1, 1)  # Single treatment!
            
            # Fit LinearDML with single treatment
            estimator = HomogeneousEffectEstimator(config=self.config)
            estimator.fit(Y, T, W, treatment_names=[treatment])
            
            # Extract residuals using our extended LinearDML
            Y_res, T_res = estimator.estimator.get_residuals()
            
            # Store everything
            all_residuals[outcome] = Y_res
            all_coefficients[f"{outcome}~{treatment}"] = estimator.estimator.intercept_[0]
            all_models[f"{outcome}~{treatment}"] = estimator
            
            print(f"  Coefficient: {estimator.estimator.intercept_[0]:.4f}")
            print(f"  Residuals shape: {Y_res.shape}")
        
        # Combine residuals into matrix for RDA
        residual_matrix = np.column_stack([
            all_residuals['fungi_richness'],
            all_residuals['plants_richness'],
            all_residuals['vertebrates_richness']
        ])
        
        print("\n" + "="*70)
        print("RESIDUAL MATRIX FOR RDA")
        print("="*70)
        print(f"Shape: {residual_matrix.shape}")
        print(f"Taxa columns: {taxa}")
        
        # Extract coordinates and environmental variables for R analysis
        additional_cols = {}
        
        # Get coordinates (check for different naming conventions)
        # self.data is a dict with numpy arrays
        if 'longitude' in self.data and 'latitude' in self.data:
            additional_cols['longitude'] = self.data['longitude']
            additional_cols['latitude'] = self.data['latitude']
        elif 'x' in self.data and 'y' in self.data:
            additional_cols['longitude'] = self.data['x']
            additional_cols['latitude'] = self.data['y']
        
        # Get environmental variables if present
        env_vars = ['temperature', 'precipitation', 'elevation', 'soil_pH']
        for var in env_vars:
            if var in self.data:
                additional_cols[var] = self.data[var]
        
        # Store results with proper structure for saving
        self.last_results = {
            'residual_matrix': residual_matrix,
            'coefficients': all_coefficients,
            'models': all_models,  # Keep DML models for inference/predictions
            'taxa': taxa,  # Need this for save function
            'additional_data': additional_cols  # Coordinates and env vars for R
        }
        
        # Save results using unified save function
        output_dir = self.config['output_dir']  # Required - no fallback
        saved_paths = save_pipeline_results(
            mode='homogeneous',
            results=self.last_results,
            output_dir=output_dir,
            config=self.config
        )
        print(f"\nResults saved to: {output_dir}/homogeneous/")
        
        return self.last_results
    
    
    def fit_heterogeneous(self, 
                         effect_modifiers: Optional[List[str]] = None) -> Dict:
        """
        Fit heterogeneous effects model using preprocessed data
        
        Args:
            effect_modifiers: Variables that modify treatment effects (default: ['bio01_std', 'bio04_std'])
            n_estimators: Number of trees
            max_depth: Maximum tree depth
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Use effect modifiers from config if not provided
        if effect_modifiers is None:
            effect_modifiers = self.config['effect_modifiers']
        
        # Use variable definitions from config
        outcome = self.config['outcome']
        treatments = self.config['treatments']
        
        # Define confounders from config (excluding effect modifiers)
        climate = self.config.get('climate_confounders', [])
        topo = self.config.get('topographic_confounders', [])
        soil = self.config.get('soil_confounders', [])
        water = self.config.get('water_balance_confounders', [])
        mesf_cols = [col for col in self.data.keys() if col.startswith('mesf_')]
        all_confounders = climate + topo + soil + water + mesf_cols
        
        # Remove effect modifiers from confounders
        confounders = [c for c in all_confounders if c not in effect_modifiers and c in self.data]
        
        # Extract arrays
        Y = self.data[outcome]
        T = np.column_stack([self.data[t] for t in treatments])
        X = np.column_stack([self.data[m] for m in effect_modifiers])
        W = np.column_stack([self.data[c] for c in confounders])
        
        # Fit heterogeneous model with config
        self.estimator = HeterogeneousEffectEstimator(config=self.config)
        self.estimator.fit(Y, T, X, W,
                          treatment_names=treatments,
                          modifier_names=effect_modifiers)
        
        # Combine results with model for saving
        self.last_results = self.estimator.results.copy()
        self.last_results['models'] = {'causal_forest': self.estimator}
        
        # Save results using unified save function
        output_dir = self.config['output_dir']  # Required - no fallback
        saved_paths = save_pipeline_results(
            mode='heterogeneous',
            results=self.last_results,
            output_dir=output_dir,
            config=self.config
        )
        print(f"\nResults saved to: {output_dir}/heterogeneous/")
        
        return self.last_results
        
    def estimate_homogeneous_effects(self,
                                    data: Union[pd.DataFrame, Dict[str, np.ndarray]],
                                    outcome: str,
                                    treatments: List[str],
                                    confounders: List[str],
                                    n_folds: int = 5) -> Dict:
        """
        Estimate constant treatment effects using LinearDML
        
        Args:
            data: Input data (DataFrame or dict of arrays)
            outcome: Name of outcome variable
            treatments: List of treatment variable names
            confounders: List of confounder variable names
            n_folds: Number of cross-validation folds
        """
        # Prepare data
        Y, T, X, W = self.preprocessor.prepare_for_dml(
            data, outcome, treatments, confounders, effect_modifiers=None
        )
        
        # Fit homogeneous model
        self.estimator = HomogeneousEffectEstimator(n_folds=n_folds)
        self.estimator.fit(Y, T, W, treatment_names=treatments)
        
        self.last_results = self.estimator.results
        return self.last_results
    
    def estimate_heterogeneous_effects(self,
                                      data: Union[pd.DataFrame, Dict[str, np.ndarray]],
                                      outcome: str,
                                      treatments: List[str],
                                      effect_modifiers: List[str],
                                      confounders: List[str],
                                      n_estimators: int = 100,
                                      max_depth: int = 10) -> Dict:
        """
        Estimate heterogeneous treatment effects using CausalForestDML
        
        Args:
            data: Input data
            outcome: Name of outcome variable
            treatments: List of treatment variable names
            effect_modifiers: Variables that modify treatment effects (REQUIRED!)
            confounders: List of confounder variable names
            n_estimators: Number of trees in the forest
            max_depth: Maximum tree depth
        """
        if not effect_modifiers:
            raise ValueError("effect_modifiers cannot be empty! Use estimate_homogeneous_effects for constant effects.")
        
        # Prepare data
        Y, T, X, W = self.preprocessor.prepare_for_dml(
            data, outcome, treatments, confounders, effect_modifiers
        )
        
        # Fit heterogeneous model
        self.estimator = HeterogeneousEffectEstimator(
            n_estimators=n_estimators,
            max_depth=max_depth
        )
        self.estimator.fit(Y, T, X, W, 
                          treatment_names=treatments,
                          modifier_names=effect_modifiers)
        
        self.last_results = self.estimator.results
        return self.last_results
    
