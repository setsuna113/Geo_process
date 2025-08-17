"""Machine learning methods for biodiversity hypothesis testing.

Provides three methods for testing biodiversity hypotheses:
- Method 1: H0-H1 testing (Model 1 vs Model 2)
- Method 2: H2 testing (Model 3a continuous, Model 3b categorical)
- Method 3: H0-H1 causal testing (Double Machine Learning)

REFACTORED VERSION 3.0:
- Method 1: Tests H0-H1 only (no interaction)
- Method 2: Tests H2 with both continuous and categorical approaches
- Method 3: Provides causal inference for H0-H1
- Correct separation of hypothesis testing concerns
"""

# Import refactored versions
from .method1_continuous_refactored import Method1H0H1GLM
from .method2_categorical_refactored import Method2H2Testing
from .method3_dml_julia import Method3DML  # Use Julia-compatible version

__all__ = [
    'Method1H0H1GLM',
    'Method2H2Testing',
    'Method3DML'
]

# Version info
__version__ = '3.0.0'  # Correct hypothesis testing structure