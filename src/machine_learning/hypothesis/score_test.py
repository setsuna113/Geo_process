"""Score Test (Lagrange Multiplier Test) implementation for GLM models."""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, Union, List
import logging

logger = logging.getLogger(__name__)


class ScoreTest:
    """Score Test (Lagrange Multiplier Test) for hypothesis testing in GLM models.
    
    The Score test evaluates hypotheses using only the null model, making it
    computationally efficient. It tests whether the score (gradient) of the 
    log-likelihood at the null hypothesis is significantly different from zero.
    
    Score statistic: S = U'(θ₀) * I(θ₀)^(-1) * U(θ₀) ~ χ²(df)
    where U is the score vector and I is the Fisher information matrix.
    """
    
    @staticmethod
    def test_coefficient_addition(
        null_model: Any,
        score_vector: Optional[np.ndarray] = None,
        information_matrix: Optional[np.ndarray] = None,
        df: int = 1
    ) -> Dict[str, Any]:
        """Test whether adding coefficients would improve the model.
        
        This is the classic score test for testing restrictions on parameters
        without fitting the full model.
        
        Args:
            null_model: Fitted null (restricted) model
            score_vector: Score vector at null hypothesis (if not available from model)
            information_matrix: Fisher information matrix (if not available from model)
            df: Degrees of freedom (number of parameters being tested)
            
        Returns:
            Dictionary with test statistics and p-value
        """
        # Try to get score and information from model
        if score_vector is None:
            if hasattr(null_model, 'get_score'):
                score_vector = null_model.get_score()
            elif hasattr(null_model, 'score'):
                score_vector = null_model.score
            else:
                raise ValueError("Score vector not provided and not available from model")
        
        if information_matrix is None:
            if hasattr(null_model, 'get_information_matrix'):
                information_matrix = null_model.get_information_matrix()
            elif hasattr(null_model, 'information_matrix'):
                information_matrix = null_model.information_matrix
            elif hasattr(null_model, 'get_hessian'):
                # Information matrix is negative expected Hessian
                information_matrix = -null_model.get_hessian()
            else:
                raise ValueError("Information matrix not provided and not available from model")
        
        # Ensure arrays
        score_vector = np.array(score_vector)
        information_matrix = np.array(information_matrix)
        
        # Check dimensions
        if len(score_vector.shape) == 1:
            score_vector = score_vector.reshape(-1, 1)
        
        # Compute score statistic: S = U' * I^(-1) * U
        try:
            info_inv = np.linalg.inv(information_matrix)
        except np.linalg.LinAlgError:
            logger.warning("Information matrix is singular, using pseudo-inverse")
            info_inv = np.linalg.pinv(information_matrix)
        
        score_statistic = float(score_vector.T @ info_inv @ score_vector)
        
        # Chi-squared p-value
        p_value = 1 - stats.chi2.cdf(score_statistic, df=df)
        
        return {
            'score_statistic': score_statistic,
            'degrees_of_freedom': df,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'test_type': 'score_test'
        }
    
    @staticmethod
    def test_from_gradients(
        null_model: Any,
        data: Any,
        additional_predictors: List[str],
        compute_gradient_func: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Score test using gradient computation for additional predictors.
        
        This version computes the score for adding new predictors to a model
        without actually fitting the expanded model.
        
        Args:
            null_model: Fitted null model
            data: Data containing additional predictors
            additional_predictors: Names of predictors to test
            compute_gradient_func: Function to compute gradient (if model doesn't provide)
            
        Returns:
            Dictionary with test results
        """
        df = len(additional_predictors)
        
        # Get residuals from null model
        if hasattr(null_model, 'get_residuals'):
            residuals = null_model.get_residuals()
        elif hasattr(null_model, 'residuals'):
            residuals = null_model.residuals
        else:
            if hasattr(null_model, 'predict'):
                predictions = null_model.predict(data)
                if hasattr(data, 'get_response'):
                    y = data.get_response()
                else:
                    raise ValueError("Cannot compute residuals")
                residuals = y - predictions
            else:
                raise ValueError("Cannot compute residuals from model")
        
        # Compute score for additional predictors
        # For GLM: score = X' * (y - μ) / φ
        # where φ is dispersion parameter
        
        # Get design matrix for additional predictors
        if hasattr(data, '__getitem__'):
            X_add = np.column_stack([data[col] for col in additional_predictors])
        else:
            raise ValueError("Cannot extract additional predictors from data")
        
        # Get dispersion parameter
        if hasattr(null_model, 'get_dispersion'):
            dispersion = null_model.get_dispersion()
        elif hasattr(null_model, 'dispersion'):
            dispersion = null_model.dispersion
        else:
            dispersion = 1.0  # Assume 1 for Poisson/Binomial
            logger.info("Dispersion parameter not available, assuming 1.0")
        
        # Compute score vector
        score_vector = X_add.T @ residuals / dispersion
        
        # Estimate information matrix for additional parameters
        # For GLM: I = X' * W * X / φ
        # where W is weight matrix (depends on variance function)
        
        if hasattr(null_model, 'get_weights'):
            weights = null_model.get_weights()
        else:
            # Assume constant weights
            weights = np.ones(len(residuals))
            logger.info("Weights not available, assuming constant")
        
        # Information matrix for additional parameters
        W = np.diag(weights)
        info_matrix = X_add.T @ W @ X_add / dispersion
        
        # Compute score statistic
        try:
            info_inv = np.linalg.inv(info_matrix)
        except np.linalg.LinAlgError:
            logger.warning("Information matrix is singular, using pseudo-inverse")
            info_inv = np.linalg.pinv(info_matrix)
        
        score_statistic = float(score_vector.T @ info_inv @ score_vector)
        
        # Chi-squared p-value
        p_value = 1 - stats.chi2.cdf(score_statistic, df=df)
        
        return {
            'score_statistic': score_statistic,
            'degrees_of_freedom': df,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'tested_predictors': additional_predictors,
            'test_type': 'score_test_gradients'
        }
    
    @staticmethod
    def test_interaction(
        null_model: Any,
        var1: str,
        var2: str,
        data: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Test for interaction effect using score test.
        
        Tests whether adding an interaction term var1:var2 would
        significantly improve the model fit.
        
        Args:
            null_model: Model without interaction
            var1: First variable in interaction
            var2: Second variable in interaction
            data: Data containing the variables
            
        Returns:
            Dictionary with test results
        """
        if data is None:
            if hasattr(null_model, 'data'):
                data = null_model.data
            else:
                raise ValueError("Data must be provided or available from model")
        
        # Create interaction term
        if hasattr(data, '__getitem__'):
            interaction = data[var1] * data[var2]
        else:
            raise ValueError("Cannot compute interaction from data")
        
        # Get residuals and weights from null model
        if hasattr(null_model, 'get_working_residuals'):
            residuals = null_model.get_working_residuals()
        elif hasattr(null_model, 'residuals'):
            residuals = null_model.residuals
        else:
            raise ValueError("Cannot get residuals from model")
        
        # Compute score for interaction term
        score = np.sum(interaction * residuals)
        
        # Compute information (variance of score)
        if hasattr(null_model, 'get_weights'):
            weights = null_model.get_weights()
        else:
            weights = np.ones(len(residuals))
        
        information = np.sum(weights * interaction**2)
        
        # Score statistic
        score_statistic = score**2 / information
        
        # Chi-squared test with 1 df
        p_value = 1 - stats.chi2.cdf(score_statistic, df=1)
        
        return {
            'interaction': f"{var1}:{var2}",
            'score_statistic': score_statistic,
            'degrees_of_freedom': 1,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'test_type': 'score_test_interaction'
        }
    
    @staticmethod
    def compare_with_other_tests(
        null_model: Any,
        full_model: Any,
        score_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare score test results with Wald and LRT for validation.
        
        All three tests (Score, Wald, LRT) are asymptotically equivalent.
        This function helps verify consistency.
        
        Args:
            null_model: Null (restricted) model
            full_model: Full (unrestricted) model
            score_result: Results from score test
            
        Returns:
            Dictionary comparing all three tests
        """
        from .wald_test import WaldTest
        from .lrt_test import LikelihoodRatioTest
        
        comparison = {
            'score_test': score_result,
            'test_type': 'three_test_comparison'
        }
        
        # Try to get Wald test
        try:
            # This would need the specific coefficient names
            logger.info("Wald test comparison requires specific coefficient names")
        except Exception as e:
            logger.debug(f"Could not perform Wald test: {e}")
        
        # Try to get LRT
        try:
            lrt_result = LikelihoodRatioTest.test_nested_models(
                null_model=null_model,
                full_model=full_model,
                df_difference=score_result.get('degrees_of_freedom', 1)
            )
            comparison['lrt_test'] = lrt_result
            
            # Check consistency
            score_p = score_result['p_value']
            lrt_p = lrt_result['p_value']
            
            if abs(score_p - lrt_p) > 0.1:
                comparison['warning'] = "Score and LRT p-values differ substantially"
            else:
                comparison['consistency'] = "Score and LRT results are consistent"
                
        except Exception as e:
            logger.debug(f"Could not perform LRT: {e}")
        
        return comparison
    
    @staticmethod
    def test_from_model_statistics(
        deviance_null: float,
        deviance_residual: float,
        df_null: int,
        df_residual: int,
        n_obs: int
    ) -> Dict[str, Any]:
        """Approximate score test using model deviance statistics.
        
        This is useful when the model doesn't provide score/information
        but does provide deviance statistics.
        
        Args:
            deviance_null: Null deviance
            deviance_residual: Residual deviance
            df_null: Null model degrees of freedom
            df_residual: Residual degrees of freedom
            n_obs: Number of observations
            
        Returns:
            Dictionary with approximate score test
        """
        # Approximate score statistic from deviance reduction
        # This uses the relationship between score and deviance
        deviance_diff = deviance_null - deviance_residual
        df_diff = df_null - df_residual
        
        # For large samples, deviance difference approximates score statistic
        score_statistic = deviance_diff
        
        # Chi-squared test
        p_value = 1 - stats.chi2.cdf(score_statistic, df=df_diff)
        
        return {
            'score_statistic': score_statistic,
            'degrees_of_freedom': df_diff,
            'p_value': p_value,
            'significant_at_0.05': p_value < 0.05,
            'note': 'Approximated from deviance statistics',
            'deviance_reduction': deviance_diff,
            'test_type': 'score_test_approximate'
        }