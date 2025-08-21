#!/usr/bin/env python3
"""Debug EconML to understand the API"""

import numpy as np
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb

# Create simple synthetic data
np.random.seed(42)
n = 1000
n_treatments = 3
n_confounders = 5

W = np.random.randn(n, n_confounders)  # Confounders
T = np.random.randn(n, n_treatments)   # Multiple treatments
Y = T[:, 0] * 2 + T[:, 1] * 1.5 + T[:, 2] * 0.5 + W[:, 0] + np.random.randn(n) * 0.5

print(f"Data shapes: Y={Y.shape}, T={T.shape}, W={W.shape}")

# Create models
model_y = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
model_t = MultiOutputRegressor(
    lgb.LGBMRegressor(n_estimators=10, num_leaves=10, verbosity=-1)
)

# Fit LinearDML
dml = LinearDML(
    model_y=model_y,
    model_t=model_t,
    discrete_treatment=False,
    cv=2,
    random_state=42
)

print("\nFitting DML...")
dml.fit(Y, T, X=None, W=W)

print("\nExploring attributes:")
print(f"Has coef_: {hasattr(dml, 'coef_')}")
print(f"Has intercept_: {hasattr(dml, 'intercept_')}")

if hasattr(dml, 'coef_'):
    print(f"coef_ shape: {dml.coef_.shape}")
    print(f"coef_ values: {dml.coef_}")

if hasattr(dml, 'intercept_'):
    print(f"intercept_ shape: {dml.intercept_.shape}")
    print(f"intercept_ values: {dml.intercept_}")

# Try inference
print("\nTrying inference methods:")
try:
    inf = dml.intercept__inference()
    print(f"intercept__inference worked!")
    print(f"  point_estimate: {inf.point_estimate}")
    print(f"  stderr: {inf.stderr}")
except Exception as e:
    print(f"intercept__inference failed: {e}")

# Try effect inference
print("\nTrying effect inference:")
try:
    effect = dml.effect(T0=np.zeros((1, n_treatments)), T1=np.ones((1, n_treatments)))
    print(f"effect shape: {effect.shape}")
    print(f"effect values: {effect}")
except Exception as e:
    print(f"effect failed: {e}")

# Try const_marginal_effect
print("\nTrying const_marginal_effect:")
try:
    cme = dml.const_marginal_effect(X=None)
    print(f"const_marginal_effect shape: {cme.shape}")
    print(f"const_marginal_effect values: {cme}")
except Exception as e:
    print(f"const_marginal_effect failed: {e}")

print("\n" + "="*50)
print("Summary:")
print(f"The model expects X=None for constant effects")
print(f"Coefficients represent marginal effects of each treatment")
print(f"Use intercept_ for the constant treatment effect")