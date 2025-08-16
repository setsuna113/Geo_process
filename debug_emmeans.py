#!/usr/bin/env python
"""Debug script to test R emmeans with minimal example."""

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up pandas converter (new way)
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
pandas_numpy_converter = pandas2ri.converter

# Load R packages
ro.r('library(emmeans)')

def test_simple_emmeans():
    """Test with a simple example to ensure R emmeans works."""
    
    logger.info("Creating simple test data...")
    
    # Create simple test data
    n = 100
    np.random.seed(42)
    
    data = pd.DataFrame({
        'y': np.random.poisson(5, n),
        'x1': np.random.normal(0, 1, n),
        'x2': np.random.normal(0, 1, n),
        'group': np.random.choice(['A', 'B'], n)
    })
    
    # Convert to R
    with localconverter(pandas_numpy_converter):
        r_data = ro.conversion.py2rpy(data)
    
    # Fit simple model
    logger.info("Fitting simple GLM...")
    model = ro.r('glm(y ~ x1 + x2 + group, data=data, family=poisson())')
    ro.globalenv['data'] = r_data
    model = ro.r('glm(y ~ x1 + x2 + group, data=data, family=poisson())')
    
    # Extract components
    coef = ro.r('coef(model)')
    vcov = ro.r('vcov(model)')
    
    logger.info(f"Coefficients shape: {len(coef)}")
    logger.info(f"VCov shape: {vcov.dim}")
    
    # Create reference grid
    logger.info("Creating reference grid...")
    ref_grid = ro.r('''
        emmeans::qdrg(
            formula = ~ x1 + x2 + group,
            data = data,
            coef = coef(model),
            vcov = vcov(model),
            df = df.residual(model),
            family = poisson(),
            link = "log"
        )
    ''')
    
    # Compute emmeans
    logger.info("Computing emmeans...")
    emmeans_result = ro.r('emmeans::emmeans(ref_grid, ~ group)')
    
    # Get summary
    summary = ro.r('summary(emmeans_result)')
    logger.info("Success! EMMs computed:")
    print(summary)
    
    return True

def test_actual_dimensions():
    """Test with actual dimensions from the pipeline."""
    
    logger.info("\nTesting with actual pipeline dimensions...")
    
    # These are the actual dimensions from Method 2
    n_coef = 21  # 21 coefficients including intercept
    n_obs = 9910  # Sample size for emmeans
    
    # Create mock data matching actual structure
    np.random.seed(42)
    
    # Create coefficient names matching actual model
    coef_names = ['Intercept'] + [f'coef_{i}' for i in range(20)]
    
    # Create coefficients
    coefficients = np.random.randn(n_coef)
    
    # Create VCov matrix (21x21)
    vcov_matrix = np.random.randn(n_coef, n_coef)
    vcov_matrix = vcov_matrix @ vcov_matrix.T  # Make it positive definite
    
    logger.info(f"Coefficients: {len(coefficients)}")
    logger.info(f"VCov matrix: {vcov_matrix.shape}")
    logger.info(f"Coefficient names: {len(coef_names)}")
    
    # Check dimensions match
    assert len(coefficients) == n_coef
    assert vcov_matrix.shape == (n_coef, n_coef)
    assert len(coef_names) == n_coef
    
    # Convert to R
    r_coef = ro.FloatVector(coefficients)
    r_coef.names = ro.StrVector(coef_names)
    
    # Convert vcov to R matrix
    vcov_list = vcov_matrix.flatten().tolist()
    r_vcov = ro.r.matrix(ro.FloatVector(vcov_list), nrow=n_coef, ncol=n_coef, byrow=True)
    
    # Set names
    ro.r.assign("vcov_matrix", r_vcov)
    ro.r.assign("coef_names", ro.StrVector(coef_names))
    r_vcov = ro.r("rownames(vcov_matrix) <- coef_names; colnames(vcov_matrix) <- coef_names; vcov_matrix")
    
    # Check in R
    ro.r.assign("test_coef", r_coef)
    ro.r.assign("test_vcov", r_vcov)
    
    logger.info("Checking dimensions in R...")
    r_coef_len = ro.r('length(test_coef)')[0]
    r_vcov_dim = ro.r('dim(test_vcov)')
    
    logger.info(f"R coefficient length: {r_coef_len}")
    logger.info(f"R vcov dimensions: {r_vcov_dim[0]} x {r_vcov_dim[1]}")
    
    # Check names match
    r_coef_names = ro.r('names(test_coef)')
    r_vcov_rownames = ro.r('rownames(test_vcov)')
    r_vcov_colnames = ro.r('colnames(test_vcov)')
    
    logger.info(f"Coefficient names match: {all(r_coef_names == coef_names)}")
    logger.info(f"VCov row names match: {all(r_vcov_rownames == coef_names)}")
    logger.info(f"VCov col names match: {all(r_vcov_colnames == coef_names)}")
    
    return True

if __name__ == "__main__":
    try:
        # Test simple case first
        test_simple_emmeans()
        
        # Test with actual dimensions
        test_actual_dimensions()
        
        logger.info("\n✅ All tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()