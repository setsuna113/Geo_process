#!/usr/bin/env python3
"""
PROPER EconML Usage Examples

This shows the CORRECT way to use EconML's built-in inference,
without reimplementing anything.
"""

import numpy as np
from econml.dml import LinearDML, SparseLinearDML
from econml.inference import BootstrapInference
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb


def example_basic_dml_with_inference():
    """
    Example 1: Basic DML with built-in statsmodels inference
    """
    print("="*70)
    print("EXAMPLE 1: Basic LinearDML with Statsmodels Inference")
    print("="*70)
    
    # Generate example data
    np.random.seed(42)
    n = 1000
    p = 5
    
    W = np.random.normal(0, 1, (n, p))  # Confounders
    T = np.random.normal(W[:, 0], 1, (n, 2))  # 2 treatments
    Y = T[:, 0] * 2 + T[:, 1] * 3 + W[:, 0] + np.random.normal(0, 1, n)
    
    # Fit with statsmodels inference (default)
    est = LinearDML(
        model_y=RandomForestRegressor(n_estimators=10, random_state=42),
        model_t=RandomForestRegressor(n_estimators=10, random_state=42),
        cv=3
    )
    
    # CORRECT: Fit with inference
    est.fit(Y, T, X=None, W=W, inference='statsmodels')
    
    # CORRECT: Get summary table - THIS HAS EVERYTHING!
    print("\nSummary Table:")
    print(est.summary())
    
    # CORRECT: Extract specific inference results
    inference = est.intercept__inference()  # Double underscore is CORRECT!
    
    print("\nExtracted Results:")
    print(f"Coefficients: {est.intercept_}")
    print(f"Standard Errors: {inference.stderr}")
    print(f"P-values: {inference.pvalue(value=0)}")  # Test H0: effect = 0
    print(f"95% CI: {inference.conf_int(alpha=0.05)}")


def example_bootstrap_inference():
    """
    Example 2: Using EconML's BootstrapInference
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: LinearDML with Bootstrap Inference")
    print("="*70)
    
    # Generate data
    np.random.seed(42)
    n = 500
    W = np.random.normal(0, 1, (n, 3))
    T = np.random.normal(W[:, 0], 1, n)
    Y = T * 1.5 + W[:, 0] + np.random.normal(0, 1, n)
    
    # CORRECT: Use EconML's BootstrapInference
    est = LinearDML(
        model_y=lgb.LGBMRegressor(n_estimators=10, verbose=-1),
        model_t=lgb.LGBMRegressor(n_estimators=10, verbose=-1),
        cv=2
    )
    
    # Fit with bootstrap
    est.fit(Y, T, X=None, W=W,
            inference=BootstrapInference(n_bootstrap_samples=100, n_jobs=-1))
    
    # Get results
    inference = est.intercept__inference()
    
    print(f"\nBootstrap Results:")
    print(f"Coefficient: {est.intercept_:.4f}")
    print(f"Bootstrap SE: {inference.stderr:.4f}")
    print(f"95% CI: {inference.conf_int(alpha=0.05)}")
    print(f"P-value: {inference.pvalue(value=0):.4f}")


def example_heterogeneous_effects():
    """
    Example 3: Heterogeneous treatment effects
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Heterogeneous Effects with X")
    print("="*70)
    
    # Generate data with heterogeneous effects
    np.random.seed(42)
    n = 1000
    X = np.random.normal(0, 1, (n, 2))  # Effect modifiers
    W = np.random.normal(0, 1, (n, 3))  # Confounders
    T = np.random.normal(W[:, 0], 1, n)
    
    # Heterogeneous effect: stronger for high X[:, 0]
    effect = 1 + 2 * X[:, 0]
    Y = T * effect + W[:, 0] + np.random.normal(0, 1, n)
    
    # Fit
    est = LinearDML(
        model_y=RandomForestRegressor(n_estimators=10, random_state=42),
        model_t=RandomForestRegressor(n_estimators=10, random_state=42),
        cv=3
    )
    
    est.fit(Y, T, X=X, W=W, inference='statsmodels')
    
    # CORRECT: For heterogeneous effects, use effect_inference
    X_test = np.array([[0, 0], [1, 0], [-1, 0]])  # Test points
    effect_inf = est.effect_inference(X_test)
    
    print("\nHeterogeneous Effects at Test Points:")
    print(effect_inf.summary_frame(alpha=0.05, value=0, decimals=3))
    
    # Population summary
    print("\nPopulation Summary:")
    print(effect_inf.population_summary(alpha=0.05))


def example_sparse_high_dim():
    """
    Example 4: High-dimensional X with automatic debiased lasso
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: SparseLinearDML for High-Dimensional X")
    print("="*70)
    
    # High-dimensional X
    np.random.seed(42)
    n = 500
    p = 100  # High-dimensional!
    
    X = np.random.normal(0, 1, (n, p))
    W = np.random.normal(0, 1, (n, 5))
    T = np.random.normal(W[:, 0], 1, n)
    
    # Only first 3 X's matter
    effect = 1 + X[:, 0] * 0.5 + X[:, 1] * 0.3 + X[:, 2] * 0.2
    Y = T * effect + W[:, 0] + np.random.normal(0, 1, n)
    
    # CORRECT: Use SparseLinearDML for high-dim
    sparse_est = SparseLinearDML(
        model_y=lgb.LGBMRegressor(n_estimators=50, verbose=-1),
        model_t=lgb.LGBMRegressor(n_estimators=50, verbose=-1),
        cv=3
    )
    
    # Automatic debiased lasso inference!
    sparse_est.fit(Y, T, X=X, W=W)
    
    print("\nSparse DML Summary:")
    print(sparse_est.summary())
    
    # Check which coefficients are non-zero
    coef = sparse_est.coef_
    non_zero = np.where(np.abs(coef) > 0.01)[0]
    print(f"\nNon-zero coefficients: {len(non_zero)}/{p}")
    print(f"Indices: {non_zero[:10]}...")  # First 10


def main():
    """Run all examples"""
    
    print("\n" + "="*70)
    print("CORRECT ECONML USAGE DEMONSTRATIONS")
    print("="*70)
    print("\nKey Points:")
    print("1. Use est.summary() - it has everything")
    print("2. Use double underscore: intercept__inference()")
    print("3. Use BootstrapInference class, not custom bootstrap")
    print("4. Use SparseLinearDML for high-dimensional problems")
    print("5. Use effect_inference() when X is not None")
    print("\n")
    
    # Run examples
    example_basic_dml_with_inference()
    example_bootstrap_inference()
    example_heterogeneous_effects()
    example_sparse_high_dim()
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("EconML provides everything needed for proper inference.")
    print("Don't reimplement - use the built-in methods correctly!")


if __name__ == "__main__":
    main()