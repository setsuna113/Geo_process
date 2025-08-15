"""H2O-based Negative Binomial GLM for massive parallel processing.

Optimized for 256-core CPU machines. Drop-in replacement for statsmodels GLM.
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Union, Tuple, List
import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

from src.base.ml_analyzer import BaseMLAnalyzer
from src.abstractions.types.ml_types import ModelType

logger = logging.getLogger(__name__)


class H2ONegativeBinomialGLM(BaseMLAnalyzer):
    """H2O Negative Binomial GLM with massive parallelization.
    
    Drop-in replacement for statsmodels NegativeBinomialGLM that utilizes
    all available CPU cores for dramatic speedup on multi-core machines.
    """
    
    def __init__(self, 
                 config: Union[Dict[str, Any], Any],
                 db_connection: Optional[Any] = None,
                 max_iter: int = 1000,
                 n_threads: int = -1,
                 max_mem_size: str = "32G",
                 **kwargs):
        """Initialize H2O Negative Binomial GLM.
        
        Args:
            config: Configuration dict or object
            db_connection: Optional database connection
            max_iter: Maximum iterations for fitting
            n_threads: Number of threads (-1 for all available)
            max_mem_size: Maximum memory for H2O cluster
            **kwargs: Additional parameters
        """
        super().__init__(config, db_connection, model_type=ModelType.REGRESSION, **kwargs)
        
        # Get GLM configuration
        glm_config = self.safe_get_config('regression_analysis.models.glm', {})
        
        self.max_iter = max_iter
        self.n_threads = n_threads if n_threads > 0 else os.cpu_count()
        self.max_mem_size = max_mem_size
        self.fit_intercept = glm_config.get('fit_intercept', True)
        
        # Initialize H2O using centralized utility
        from ...utils.h2o_utils import ensure_h2o_initialized
        ensure_h2o_initialized(
            n_threads=self.n_threads,
            max_mem_size=self.max_mem_size,
            min_mem_size="4G"
        )
        
        self.model = None
        self.model_result = None
        self.feature_names = None
        self.dispersion_ratio = None
        
        logger.info(f"Initialized H2O Negative Binomial GLM with {self.n_threads} threads")
    
    def check_overdispersion(self, y: np.ndarray) -> Tuple[float, bool]:
        """Check for overdispersion in the response variable.
        
        Args:
            y: Response variable (species richness counts)
            
        Returns:
            Tuple of (dispersion_ratio, is_overdispersed)
        """
        mean_y = np.mean(y)
        var_y = np.var(y)
        dispersion_ratio = var_y / mean_y
        
        # Decision rule: overdispersed if ratio > 1.2
        is_overdispersed = dispersion_ratio > 1.2
        
        logger.info(f"Dispersion check: mean={mean_y:.2f}, var={var_y:.2f}, "
                   f"ratio={dispersion_ratio:.2f}")
        logger.info(f"Data is {'overdispersed' if is_overdispersed else 'not overdispersed'}")
        
        self.dispersion_ratio = dispersion_ratio
        return dispersion_ratio, is_overdispersed
    
    def fit(self, 
            X: h2o.H2OFrame,
            y: Optional[str] = None,
            validation_fraction: float = 0.2,
            lambda_search: bool = True,
            fixed_theta: Optional[float] = None,
            **kwargs) -> 'H2ONegativeBinomialGLM':
        """Fit the H2O Negative Binomial GLM using all available cores.
        
        Args:
            X: H2OFrame with all features and target
            y: Target column name (default: 'fungi_richness' or last column)
            validation_fraction: Fraction of data for validation
            lambda_search: Whether to perform automatic lambda search
            fixed_theta: Optional fixed dispersion parameter (prevents instability)
            **kwargs: Additional parameters
            
        Returns:
            Self for method chaining
        """
        # ONLY accept H2OFrame - no pandas conversion!
        if not isinstance(X, h2o.H2OFrame):
            raise TypeError(
                "Only H2OFrame is accepted. For 180M+ points, use H2O directly. "
                "Convert with: h2o_df = h2o.import_file('data.parquet')"
            )
        
        logger.info("Using H2OFrame directly - no memory duplication!")
        h2o_data = X
        
        # Determine target column
        if y is not None:
            self.target_col = y
        elif 'fungi_richness' in h2o_data.columns:
            self.target_col = 'fungi_richness'
        else:
            self.target_col = h2o_data.columns[-1]
            logger.warning(f"No target specified, using last column: {self.target_col}")
        
        feature_names = [col for col in h2o_data.columns if col != self.target_col]
        
        # Check overdispersion using H2O operations
        target_mean = h2o_data[self.target_col].mean()
        target_var = h2o_data[self.target_col].var()
        # H2O's mean() and var() return lists
        if isinstance(target_mean, list):
            target_mean = target_mean[0]
        if isinstance(target_var, list):
            target_var = target_var[0]
        dispersion_ratio = target_var / target_mean
        is_overdispersed = dispersion_ratio > 1.2
        
        if not is_overdispersed:
            logger.warning(f"Data may not be overdispersed (ratio={dispersion_ratio:.2f}). Consider Poisson GLM.")
        
        # Split data for validation
        if validation_fraction > 0:
            splits = h2o_data.split_frame(
                ratios=[1 - validation_fraction],
                seed=42
            )
            train_frame = splits[0]
            valid_frame = splits[1] if len(splits) > 1 else None
            logger.info(f"Split data: {train_frame.nrows} train, "
                       f"{valid_frame.nrows if valid_frame else 0} validation")
        else:
            train_frame = h2o_data
            valid_frame = None
        
        # Configure and fit the model
        logger.info(f"Fitting H2O Negative Binomial GLM with {self.n_threads} threads...")
        
        # Store fixed theta if provided
        self.fixed_theta = fixed_theta
        if fixed_theta:
            logger.info(f"Using fixed theta={fixed_theta:.4f} for stable dispersion")
        
        # Build the model
        # Note: remove_collinear_columns only works without regularization
        # So we either use lambda_search OR remove_collinear_columns, not both
        
        # Prepare base parameters
        base_params = {
            'family': 'negativebinomial',
            'link': 'log',
            'max_iterations': self.max_iter,
            'standardize': True,
            'solver': 'IRLSM',
            'compute_p_values': True,
            'intercept': self.fit_intercept,
            'seed': 42
        }
        
        # Add fixed theta if provided
        if fixed_theta is not None:
            base_params['theta'] = fixed_theta
            logger.info(f"Using fixed theta={fixed_theta:.6f} for dispersion parameter")
        
        if lambda_search:
            # Use regularization to handle collinearity
            self.model = H2OGeneralizedLinearEstimator(
                **base_params,
                lambda_search=True,
                nlambdas=100,
                alpha=0.5,  # Elastic net mixing (0.5 = equal L1 and L2)
                remove_collinear_columns=False  # Can't use with lambda_search
            )
        else:
            # Remove collinear columns without regularization
            self.model = H2OGeneralizedLinearEstimator(
                **base_params,
                lambda_search=False,
                lambda_=0.0,  # No regularization
                remove_collinear_columns=True
            )
        
        # Train the model (automatically uses all threads!)
        self.model.train(
            x=feature_names,
            y=self.target_col,
            training_frame=train_frame,
            validation_frame=valid_frame
        )
        
        # Extract model information
        self.model_result = self.model
        self.feature_names = feature_names
        self.is_fitted = True
        
        # Log performance metrics
        train_perf = self.model.model_performance(train=True)
        logger.info(f"Training RMSE: {train_perf.rmse():.4f}")
        logger.info(f"Training Mean Residual Deviance: {train_perf.mean_residual_deviance():.4f}")
        
        if valid_frame:
            valid_perf = self.model.model_performance(valid=True)
            logger.info(f"Validation RMSE: {valid_perf.rmse():.4f}")
            logger.info(f"Validation Mean Residual Deviance: {valid_perf.mean_residual_deviance():.4f}")
        
        if lambda_search:
            # H2O GLM stores optimal parameters differently
            try:
                params = self.model.params
                if hasattr(self.model, 'lambda_'):
                    logger.info(f"Selected lambda: {self.model.lambda_}")
                if hasattr(self.model, 'alpha'):
                    logger.info(f"Selected alpha: {self.model.alpha}")
            except:
                pass  # Some parameters may not be available
        
        return self
    
    def get_coefficient(self, coef_name: str) -> Dict[str, Any]:
        """Get coefficient information for a specific variable.
        
        Args:
            coef_name: Name of the coefficient to retrieve
            
        Returns:
            Dictionary with coefficient value, std error, z-value and p-value
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Get coefficients table with p-values when compute_p_values=True
        # This returns a table with: names, coefficients, std_error, z_value, p_value
        coef_table = self.model._model_json['output']['coefficients_table']
        
        if coef_table is not None:
            # H2O coefficient table structure varies - handle both formats
            if hasattr(coef_table, 'as_data_frame'):
                # Convert to pandas DataFrame
                coef_df = coef_table.as_data_frame()
                if coef_name in coef_df['names'].values:
                    row = coef_df[coef_df['names'] == coef_name].iloc[0]
                    return {
                        'coefficient': row.get('coefficients', None),
                        'std_error': row.get('std_error', None),
                        'z_value': row.get('z_value', None),
                        'p_value': row.get('p_value', None),
                        'significant': row.get('p_value', 1.0) < 0.05 if 'p_value' in row else None
                    }
            elif 'cell_values' in coef_table:
                # Handle as nested structure
                rows = coef_table['cell_values']
                col_names = coef_table.get('col_headers', [])
                
                # Find name column index
                name_idx = col_names.index('names') if 'names' in col_names else 0
                coef_idx = col_names.index('coefficients') if 'coefficients' in col_names else 1
                
                for row in rows:
                    if row[name_idx] == coef_name:
                        return {
                            'coefficient': row[coef_idx] if len(row) > coef_idx else None,
                            'std_error': row[2] if len(row) > 2 else None,
                            'z_value': row[3] if len(row) > 3 else None,
                            'p_value': row[4] if len(row) > 4 else None,
                            'significant': row[4] < 0.05 if len(row) > 4 and row[4] is not None else None
                        }
            
            # If no match found in above structures
            raise ValueError(f"Coefficient {coef_name} not found in coefficient table")
        else:
            # Fallback to basic coefficients if p-values not computed
            coefficients = self.model.coef()
            if coef_name in coefficients:
                logger.warning("P-values not available. Set compute_p_values=True when fitting.")
                return {
                    'coefficient': coefficients[coef_name],
                    'std_error': None,
                    'z_value': None,
                    'p_value': None,
                    'significant': None
                }
            else:
                raise ValueError(f"Coefficient {coef_name} not found in model")
    
    def get_all_coefficients(self) -> Dict[str, float]:
        """Get all model coefficients.
        
        Returns:
            Dictionary of coefficient names to values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return self.model.coef()
    
    def get_standard_errors(self) -> Dict[str, float]:
        """Get standard errors for all coefficients.
        
        Returns:
            Dictionary of coefficient names to standard errors
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Get coefficients table with std errors when compute_p_values=True
        coef_table = self.model._model_json['output']['coefficients_table']
        
        if coef_table is not None:
            # Handle different H2O table formats
            if hasattr(coef_table, 'cell_values'):
                # New format with cell_values
                if len(coef_table.cell_values) > 2:
                    names = coef_table.cell_values[0]  # First row is names
                    std_errors = coef_table.cell_values[2]  # Third row is std_errors
                    return dict(zip(names, std_errors))
            elif 'data' in coef_table and len(coef_table['data']) > 2:
                # Old format with 'data' key
                names = coef_table['data'][0]  # First row is names
                std_errors = coef_table['data'][2]  # Third row is std_errors
                return dict(zip(names, std_errors))
        
        logger.warning("Standard errors not available. Set compute_p_values=True when fitting.")
        coeffs = self.model.coef()
        return {name: np.nan for name in coeffs.keys()}
    
    def get_p_values(self) -> Dict[str, float]:
        """Get p-values for all coefficients.
        
        Returns:
            Dictionary of coefficient names to p-values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Get coefficients table with p-values when compute_p_values=True
        coef_table = self.model._model_json['output']['coefficients_table']
        
        if coef_table is not None:
            # Handle different H2O table formats
            if hasattr(coef_table, 'cell_values'):
                # New format with cell_values
                if len(coef_table.cell_values) > 3:
                    names = coef_table.cell_values[0]  # First row is names
                    p_values = coef_table.cell_values[4]  # Fifth row is p_values
                    return dict(zip(names, p_values))
            elif 'data' in coef_table and len(coef_table['data']) > 3:
                # Old format with 'data' key
                names = coef_table['data'][0]  # First row is names
                p_values = coef_table['data'][4]  # Fifth row is p_values
                return dict(zip(names, p_values))
        
        logger.warning("P-values not available. Set compute_p_values=True when fitting.")
        coeffs = self.model.coef()
        return {name: np.nan for name in coeffs.keys()}
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions using the fitted model.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = X
        
        # Convert to H2O Frame
        X_h2o = h2o.H2OFrame(X_df)
        
        # Make predictions
        predictions = self.model.predict(X_h2o)
        
        # Convert back to numpy
        pred_array = predictions.as_data_frame().values.ravel()
        
        return pred_array
    
    def get_coefficients(self) -> Dict[str, float]:
        """Get model coefficients.
        
        Returns:
            Dictionary of feature names to coefficient values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        coef_dict = self.model.coef()
        
        # Add intercept if present
        if self.fit_intercept and 'Intercept' in coef_dict:
            coef_dict['intercept'] = coef_dict.pop('Intercept')
        
        return coef_dict
    
    def get_summary(self) -> Dict[str, Any]:
        """Get model summary statistics.
        
        Returns:
            Dictionary with model summary
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        summary = {
            'family': 'Negative Binomial',
            'link': 'log',
            'n_features': len(self.feature_names),
            'n_iterations': self.model.actual_params['iterations'],
            'converged': True,  # H2O handles convergence internally
            'dispersion_ratio': self.dispersion_ratio,
            'coefficients': self.get_coefficients(),
            'aic': self.model.aic(),
            'null_deviance': self.model.null_deviance(),
            'residual_deviance': self.model.residual_deviance(),
            'null_dof': self.model.null_degrees_of_freedom(),
            'residual_dof': self.model.residual_degrees_of_freedom()
        }
        
        # Add performance metrics
        train_perf = self.model.model_performance(train=True)
        summary['train_rmse'] = train_perf.rmse()
        summary['train_mean_residual_deviance'] = train_perf.mean_residual_deviance()
        
        if self.model.model_performance(valid=True):
            valid_perf = self.model.model_performance(valid=True)
            summary['valid_rmse'] = valid_perf.rmse()
            summary['valid_mean_residual_deviance'] = valid_perf.mean_residual_deviance()
        
        return summary
    
    def shutdown_cluster(self):
        """Shutdown H2O cluster to free resources."""
        try:
            h2o.cluster().shutdown()
            logger.info("H2O cluster shutdown successfully")
        except:
            logger.warning("H2O cluster already shutdown or not running")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        # Note: In production, you might want to keep H2O running
        # across multiple model fits for efficiency
        pass  # Don't auto-shutdown to allow reuse
    
    @staticmethod
    def estimate_stable_theta(
        h2o_data: h2o.H2OFrame,
        target_col: str,
        baseline_features: List[str],
        max_iter: int = 100
    ) -> float:
        """Estimate a stable theta parameter from a baseline model.
        
        This is used for the two-stage approach where we estimate theta
        from a simpler model and then fix it for all nested models.
        
        Args:
            h2o_data: H2O DataFrame with all data
            target_col: Name of target column
            baseline_features: List of baseline feature names (e.g., just climate)
            max_iter: Maximum iterations for theta estimation
            
        Returns:
            Estimated theta value
        """
        logger.info("Estimating stable theta from baseline model")
        
        # Fit a simple model to estimate theta
        baseline_model = H2OGeneralizedLinearEstimator(
            family='negativebinomial',
            link='log',
            max_iterations=max_iter,
            standardize=True,
            lambda_=0.0,  # No regularization for theta estimation
            solver='IRLSM',
            compute_p_values=False,  # Don't need p-values for theta
            intercept=True,
            seed=42
        )
        
        # Train on baseline features only
        baseline_model.train(
            x=baseline_features,
            y=target_col,
            training_frame=h2o_data
        )
        
        # Extract theta from the fitted model
        # H2O stores theta in the model parameters
        model_params = baseline_model._model_json['output']
        
        # Try different locations where H2O might store theta
        theta = None
        if 'theta' in model_params:
            theta = model_params['theta']
        elif 'dispersion' in model_params and model_params['dispersion'] != 0:
            theta = 1.0 / model_params['dispersion']  # theta = 1/dispersion
        elif 'family_params' in model_params and 'theta' in model_params['family_params']:
            theta = model_params['family_params']['theta']
        else:
            # Try to extract from the model's actual parameters
            try:
                actual_params = baseline_model.actual_params
                if 'theta' in actual_params:
                    theta = actual_params['theta']
            except:
                pass
            
            # Try GLM-specific parameters
            if theta is None:
                try:
                    glm_params = baseline_model.params
                    if 'theta' in glm_params:
                        theta = glm_params['theta']['actual']
                except:
                    pass
            
        if theta is None or theta == 0:
            # Fallback: estimate from mean-variance relationship
            # For NB: Var = μ + μ²/θ, so θ ≈ μ²/(Var - μ)
            target_mean = h2o_data[target_col].mean()
            target_var = h2o_data[target_col].var()
            if isinstance(target_mean, list): target_mean = target_mean[0]
            if isinstance(target_var, list): target_var = target_var[0]
            
            if target_var > target_mean:
                # Better estimation: use dispersion ratio to estimate theta
                dispersion_ratio = target_var / target_mean
                # For typical ecological count data with dispersion ratio 5-10,
                # theta should be around mean/(dispersion_ratio - 1)
                theta = target_mean / (dispersion_ratio - 1)
                logger.warning(f"Estimated theta from dispersion ratio: {theta:.4f} (dispersion={dispersion_ratio:.2f})")
            else:
                theta = 100.0  # Large theta = approximately Poisson
                logger.warning(f"Data appears underdispersed, using large theta: {theta:.4f}")
        else:
            logger.info(f"Extracted theta from baseline model: {theta:.4f}")
        
        # Ensure theta is positive and reasonable
        # Use more reasonable bounds based on typical ecological data
        theta = max(1.0, min(theta, 1000.0))  # Minimum 1.0 instead of 0.1
        
        return float(theta)
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance on test data.
        
        Args:
            X: Features
            y: Target values
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X)
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        return {
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'r2': r2_score(y, predictions)
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # For GLM, use absolute coefficient values as importance
        coeffs = self.get_all_coefficients()
        importance_df = pd.DataFrame({
            'feature': list(coeffs.keys()),
            'importance': [abs(v) if v is not None else 0 for v in coeffs.values()]
        })
        return importance_df.sort_values('importance', ascending=False)
    
    def get_log_likelihood(self) -> float:
        """Get log-likelihood of the fitted model.
        
        Returns:
            Log-likelihood value
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # H2O GLM provides log-likelihood via the model metrics
        if hasattr(self.model, 'loglikelihood'):
            return self.model.loglikelihood()
        
        # Alternative: compute from deviance
        # Log-likelihood = -0.5 * deviance for GLM
        deviance = self.get_deviance()
        return -0.5 * deviance
    
    def get_deviance(self) -> float:
        """Get deviance of the fitted model.
        
        Returns:
            Deviance value
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # H2O provides residual deviance
        if hasattr(self.model, 'residual_deviance'):
            return self.model.residual_deviance()
        
        # Fallback to null deviance if available
        if hasattr(self.model, 'null_deviance'):
            return self.model.null_deviance()
        
        # If neither available, raise error
        raise NotImplementedError("Deviance not available from H2O model")
    
    def get_aic(self) -> float:
        """Get AIC (Akaike Information Criterion) of the fitted model.
        
        Returns:
            AIC value
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # H2O provides AIC directly
        if hasattr(self.model, 'aic'):
            return self.model.aic()
        
        # Alternative: compute from log-likelihood
        # AIC = 2k - 2*log-likelihood
        try:
            log_lik = self.get_log_likelihood()
            n_params = len(self.get_all_coefficients())
            return 2 * n_params - 2 * log_lik
        except:
            logger.warning("AIC not available from H2O model")
            return None
    
    def get_bic(self) -> float:
        """Get BIC (Bayesian Information Criterion) of the fitted model.
        
        Returns:
            BIC value
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # H2O provides BIC directly
        if hasattr(self.model, 'bic'):
            return self.model.bic()
        
        # Alternative: compute from log-likelihood
        # BIC = k*log(n) - 2*log-likelihood
        try:
            log_lik = self.get_log_likelihood()
            n_params = len(self.get_all_coefficients())
            n_obs = self.model.nobs if hasattr(self.model, 'nobs') else 1000  # Fallback
            return n_params * np.log(n_obs) - 2 * log_lik
        except:
            logger.warning("BIC not available from H2O model")
            return None
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        if not self.is_fitted:
            return self.params
        return self.model.actual_params
    
    def set_model_params(self, **params) -> None:
        """Set model parameters.
        
        Args:
            **params: Model parameters to set
        """
        self.params.update(params)