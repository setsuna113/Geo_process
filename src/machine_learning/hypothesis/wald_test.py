"""Wald test implementation for GLM models."""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, Union, List
import logging

logger = logging.getLogger(__name__)


class WaldTest:
    """Wald test for hypothesis testing in GLM models.
    
    The Wald test evaluates the significance of coefficients by:
    W = (beta - beta_0) / SE(beta) ~ N(0,1)
    
    For multiple coefficients:
    W = (beta - beta_0)' * Cov(beta)^(-1) * (beta - beta_0) ~ chi2(df)
    """
    
    @staticmethod
    def test_single_coefficient(
        coefficient: float,
        standard_error: float,
        null_value: float = 0.0
    ) -> Dict[str, Any]:
        """Test a single coefficient against a null hypothesis.
        
        Args:
            coefficient: Estimated coefficient value
            standard_error: Standard error of the coefficient
            null_value: Null hypothesis value (default: 0)
            
        Returns:
            Dictionary with test statistics and p-value
        """
        if standard_error <= 0:
            raise ValueError("Standard error must be positive")
        
        # Compute Wald statistic
        wald_statistic = (coefficient - null_value) / standard_error
        
        # Two-tailed p-value from normal distribution
        p_value = 2 * (1 - stats.norm.cdf(abs(wald_statistic)))
        
        return {
            'coefficient': coefficient,
            'standard_error': standard_error,
            'wald_statistic': wald_statistic,
            'p_value': p_value,
            'null_value': null_value,
            'significant_at_0.05': p_value < 0.05,
            'test_type': 'wald_single'
        }
    
    @staticmethod
    def test_multiple_coefficients(
        coefficients: Union[List[float], np.ndarray],
        covariance_matrix: np.ndarray,
        null_values: Optional[Union[List[float], np.ndarray]] = None
    ) -> Dict[str, Any]:
        """Test multiple coefficients jointly using multivariate Wald test.
        
        Args:
            coefficients: Vector of coefficient estimates
            covariance_matrix: Covariance matrix of coefficients
            null_values: Null hypothesis values (default: zeros)
            
        Returns:
            Dictionary with test statistics and p-value
        """
        coefficients = np.array(coefficients)
        k = len(coefficients)
        
        if null_values is None:
            null_values = np.zeros(k)
        else:
            null_values = np.array(null_values)
        
        if covariance_matrix.shape != (k, k):
            raise ValueError(f"Covariance matrix shape {covariance_matrix.shape} doesn't match coefficients length {k}")
        
        # Check if covariance matrix is positive definite
        try:
            cov_inv = np.linalg.inv(covariance_matrix)
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix is singular, using pseudo-inverse")
            cov_inv = np.linalg.pinv(covariance_matrix)
        
        # Compute Wald statistic: W = (beta - beta_0)' * Cov^(-1) * (beta - beta_0)
        diff = coefficients - null_values
        wald_statistic = float(diff.T @ cov_inv @ diff)
        
        # Chi-squared p-value with k degrees of freedom
        p_value = 1 - stats.chi2.cdf(wald_statistic, df=k)
        
        return {
            'coefficients': coefficients.tolist(),
            'null_values': null_values.tolist(),
            'wald_statistic': wald_statistic,
            'degrees_of_freedom': k,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'test_type': 'wald_multiple'
        }
    
    @staticmethod
    def test_linear_restriction(
        coefficients: Union[List[float], np.ndarray],
        covariance_matrix: np.ndarray,
        restriction_matrix: np.ndarray,
        restriction_values: Optional[Union[List[float], np.ndarray]] = None
    ) -> Dict[str, Any]:
        """Test linear restrictions on coefficients: R*beta = q.
        
        Args:
            coefficients: Vector of coefficient estimates
            covariance_matrix: Covariance matrix of coefficients
            restriction_matrix: Matrix R defining linear restrictions
            restriction_values: Vector q of restriction values (default: zeros)
            
        Returns:
            Dictionary with test statistics and p-value
        """
        coefficients = np.array(coefficients)
        R = np.array(restriction_matrix)
        
        if restriction_values is None:
            q = np.zeros(R.shape[0])
        else:
            q = np.array(restriction_values)
        
        # Number of restrictions
        r = R.shape[0]
        
        # Check dimensions
        if R.shape[1] != len(coefficients):
            raise ValueError(f"Restriction matrix columns {R.shape[1]} must match coefficients length {len(coefficients)}")
        
        if len(q) != r:
            raise ValueError(f"Restriction values length {len(q)} must match number of restrictions {r}")
        
        # Compute R*beta - q
        diff = R @ coefficients - q
        
        # Compute variance of R*beta: R * Cov * R'
        var_Rbeta = R @ covariance_matrix @ R.T
        
        # Check if variance matrix is positive definite
        try:
            var_inv = np.linalg.inv(var_Rbeta)
        except np.linalg.LinAlgError:
            logger.warning("Variance matrix is singular, using pseudo-inverse")
            var_inv = np.linalg.pinv(var_Rbeta)
        
        # Compute Wald statistic
        wald_statistic = float(diff.T @ var_inv @ diff)
        
        # Chi-squared p-value with r degrees of freedom
        p_value = 1 - stats.chi2.cdf(wald_statistic, df=r)
        
        return {
            'coefficients': coefficients.tolist(),
            'restriction_matrix': R.tolist(),
            'restriction_values': q.tolist(),
            'wald_statistic': wald_statistic,
            'degrees_of_freedom': r,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'test_type': 'wald_linear_restriction'
        }
    
    @staticmethod
    def test_coefficients_from_model(
        model: Any,
        coefficient_names: Union[str, List[str]],
        null_values: Optional[Union[float, List[float]]] = None,
        joint_test: bool = False
    ) -> Dict[str, Any]:
        """Convenience method to test coefficients directly from a model object.
        
        Args:
            model: Model object with get_coefficient() method
            coefficient_names: Name(s) of coefficient(s) to test
            null_values: Null hypothesis value(s) (default: 0)
            joint_test: If True and multiple coefficients, perform joint test
            
        Returns:
            Dictionary with test results
        """
        if isinstance(coefficient_names, str):
            coefficient_names = [coefficient_names]
        
        if null_values is None:
            null_values = [0.0] * len(coefficient_names)
        elif not isinstance(null_values, list):
            null_values = [null_values]
        
        if len(coefficient_names) == 1:
            # Single coefficient test
            coef_info = model.get_coefficient(coefficient_names[0])
            if coef_info is None:
                raise ValueError(f"Coefficient '{coefficient_names[0]}' not found in model")
            
            return WaldTest.test_single_coefficient(
                coefficient=coef_info['coefficient'],
                standard_error=coef_info['std_error'],
                null_value=null_values[0]
            )
        
        elif joint_test:
            # Multiple coefficients joint test
            coefficients = []
            for name in coefficient_names:
                coef_info = model.get_coefficient(name)
                if coef_info is None:
                    raise ValueError(f"Coefficient '{name}' not found in model")
                coefficients.append(coef_info['coefficient'])
            
            # Get covariance matrix (would need model support)
            if hasattr(model, 'get_covariance_matrix'):
                cov_matrix = model.get_covariance_matrix(coefficient_names)
                return WaldTest.test_multiple_coefficients(
                    coefficients=coefficients,
                    covariance_matrix=cov_matrix,
                    null_values=null_values
                )
            else:
                logger.warning("Model doesn't support covariance matrix extraction, performing individual tests")
                joint_test = False
        
        if not joint_test:
            # Individual tests for each coefficient
            results = {}
            for name, null_val in zip(coefficient_names, null_values):
                coef_info = model.get_coefficient(name)
                if coef_info is None:
                    logger.warning(f"Coefficient '{name}' not found in model")
                    continue
                
                results[name] = WaldTest.test_single_coefficient(
                    coefficient=coef_info['coefficient'],
                    standard_error=coef_info['std_error'],
                    null_value=null_val
                )
            
            return {
                'individual_tests': results,
                'test_type': 'wald_individual'
            }