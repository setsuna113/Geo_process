"""Negative Binomial GLM implementation.

Based on regression_analysis_decisions.md lines 25-35 and 438-443.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Union, Tuple
import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial
from scipy import stats

from ...base.ml_analyzer import BaseMLAnalyzer
from ...abstractions.types.ml_types import ModelType

logger = logging.getLogger(__name__)


class NegativeBinomialGLM(BaseMLAnalyzer):
    """Negative Binomial GLM for overdispersed count data.
    
    Implements the Negative Binomial GLM as specified in regression_analysis_decisions.md.
    Handles overdispersed species richness data.
    """
    
    def __init__(self, 
                 config: Union[Dict[str, Any], Any],
                 db_connection: Optional[Any] = None,
                 alpha: float = 1.0,
                 **kwargs):
        """Initialize Negative Binomial GLM.
        
        Args:
            config: Configuration dict or object
            db_connection: Optional database connection
            alpha: Dispersion parameter (theta). Default 1.0, will be estimated.
            **kwargs: Additional parameters
        """
        super().__init__(config, db_connection, model_type=ModelType.REGRESSION, **kwargs)
        
        # Get GLM configuration
        glm_config = self.safe_get_config('regression_analysis.models.glm', {})
        
        self.alpha = alpha  # Initial dispersion parameter
        self.max_iter = glm_config.get('max_iter', 1000)
        self.fit_intercept = glm_config.get('fit_intercept', True)
        
        self.model = None
        self.model_result = None
        self.theta = None  # Estimated dispersion parameter
        
        logger.info(f"Initialized Negative Binomial GLM with max_iter={self.max_iter}")
    
    def check_overdispersion(self, y: np.ndarray) -> Tuple[float, bool]:
        """Check for overdispersion in the response variable.
        
        Implements dispersion ratio check from regression_analysis_decisions.md lines 16-24.
        
        Args:
            y: Response variable (species richness counts)
            
        Returns:
            Tuple of (dispersion_ratio, is_overdispersed)
        """
        mean_y = np.mean(y)
        var_y = np.var(y)
        dispersion_ratio = var_y / mean_y
        
        # Decision rule from lines 21-23
        is_overdispersed = dispersion_ratio > 1.2
        
        logger.info(f"Dispersion check: mean={mean_y:.2f}, var={var_y:.2f}, "
                   f"ratio={dispersion_ratio:.2f}")
        logger.info(f"Data is {'overdispersed' if is_overdispersed else 'not overdispersed'}")
        
        return dispersion_ratio, is_overdispersed
    
    def fit(self, 
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series],
            formula: Optional[str] = None,
            **kwargs) -> 'NegativeBinomialGLM':
        """Fit the Negative Binomial GLM.
        
        Args:
            X: Feature matrix
            y: Target values (species richness counts)
            formula: Optional R-style formula
            **kwargs: Additional parameters
            
        Returns:
            Self for method chaining
        """
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            feature_names = X.columns.tolist()
        else:
            X_array = X
            feature_names = [f'X{i}' for i in range(X.shape[1])]
        
        y_array = y.values if isinstance(y, pd.Series) else y
        
        # Check overdispersion
        dispersion_ratio, is_overdispersed = self.check_overdispersion(y_array)
        
        if not is_overdispersed:
            logger.warning("Data may not be overdispersed. Consider using Poisson GLM instead.")
        
        # Add intercept if configured
        if self.fit_intercept:
            X_array = sm.add_constant(X_array)
            feature_names = ['const'] + feature_names
        
        # Create and fit the model
        logger.info(f"Fitting Negative Binomial GLM with {X_array.shape[1]} features")
        
        try:
            # Create the model
            self.model = sm.GLM(
                y_array, 
                X_array,
                family=NegativeBinomial(alpha=self.alpha)
            )
            
            # Fit the model
            self.model_result = self.model.fit(maxiter=self.max_iter)
            
            # Extract theta (dispersion parameter)
            self.theta = self.model_result.scale
            
            logger.info(f"Model converged: {self.model_result.converged}")
            logger.info(f"Estimated theta (dispersion): {self.theta:.4f}")
            logger.info(f"AIC: {self.model_result.aic:.2f}")
            logger.info(f"Log-likelihood: {self.model_result.llf:.2f}")
            
            # Store feature names for later use
            self.feature_names = feature_names
            self.is_fitted = True
            
        except Exception as e:
            logger.error(f"Failed to fit Negative Binomial GLM: {e}")
            raise
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted counts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to numpy if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Add intercept if configured
        if self.fit_intercept:
            X_array = sm.add_constant(X_array)
        
        # Make predictions
        predictions = self.model_result.predict(X_array)
        
        return predictions
    
    def get_coefficients(self) -> pd.DataFrame:
        """Get model coefficients with statistics.
        
        Returns:
            DataFrame with coefficients, standard errors, p-values, etc.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        summary = pd.DataFrame({
            'coefficient': self.model_result.params,
            'std_error': self.model_result.bse,
            'z_value': self.model_result.tvalues,
            'p_value': self.model_result.pvalues,
            'ci_lower': self.model_result.conf_int()[0],
            'ci_upper': self.model_result.conf_int()[1]
        })
        
        summary.index = self.feature_names
        
        # Add exponentiated coefficients (multiplicative effects)
        summary['exp_coef'] = np.exp(summary['coefficient'])
        
        return summary
    
    def likelihood_ratio_test(self, null_model: 'NegativeBinomialGLM') -> Dict[str, float]:
        """Perform likelihood ratio test against null model.
        
        Implements LRT from regression_analysis_decisions.md lines 388-393.
        
        Args:
            null_model: Fitted null model (usually without biodiversity predictors)
            
        Returns:
            Dictionary with test statistic, p-value, and degrees of freedom
        """
        if not self.is_fitted or not null_model.is_fitted:
            raise ValueError("Both models must be fitted")
        
        # Calculate LRT statistic
        lrt_stat = 2 * (self.model_result.llf - null_model.model_result.llf)
        
        # Degrees of freedom = difference in number of parameters
        df = len(self.model_result.params) - len(null_model.model_result.params)
        
        # Calculate p-value
        p_value = 1 - stats.chi2.cdf(lrt_stat, df)
        
        logger.info(f"Likelihood Ratio Test: χ²={lrt_stat:.2f}, df={df}, p={p_value:.4f}")
        
        return {
            'lrt_statistic': lrt_stat,
            'p_value': p_value,
            'degrees_of_freedom': df,
            'null_llf': null_model.model_result.llf,
            'full_llf': self.model_result.llf
        }
    
    def aic_comparison(self, other_model: 'NegativeBinomialGLM') -> Dict[str, float]:
        """Compare models using AIC.
        
        Implements AIC comparison from regression_analysis_decisions.md lines 395-406.
        
        Args:
            other_model: Another fitted model
            
        Returns:
            Dictionary with AIC values and delta AIC
        """
        if not self.is_fitted or not other_model.is_fitted:
            raise ValueError("Both models must be fitted")
        
        aic_self = self.model_result.aic
        aic_other = other_model.model_result.aic
        delta_aic = aic_other - aic_self  # Positive if self is better
        
        # Interpretation from lines 402-405
        if delta_aic > 10:
            interpretation = "Essentially no support for other model"
        elif delta_aic > 4:
            interpretation = "Strong support for this model"
        elif delta_aic > 2:
            interpretation = "Substantial support for this model"
        else:
            interpretation = "Models are comparable"
        
        logger.info(f"AIC comparison: This={aic_self:.2f}, Other={aic_other:.2f}, "
                   f"ΔAIC={delta_aic:.2f}")
        logger.info(f"Interpretation: {interpretation}")
        
        return {
            'aic_this': aic_self,
            'aic_other': aic_other,
            'delta_aic': delta_aic,
            'interpretation': interpretation
        }
    
    def get_model_summary(self) -> str:
        """Get model summary similar to R's summary() output."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return str(self.model_result.summary())
    
    def evaluate(self, 
                 X: Union[np.ndarray, pd.DataFrame],
                 y: Union[np.ndarray, pd.Series],
                 metrics: Optional[list] = None) -> Dict[str, float]:
        """Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True values
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of metric values
        """
        y_pred = self.predict(X)
        y_true = y.values if isinstance(y, pd.Series) else y
        
        results = {}
        
        # Pseudo R-squared (McFadden's)
        null_llf = stats.nbinom.logpmf(y_true, 1, 0.5).sum()  # Null model log-likelihood
        results['pseudo_r2'] = 1 - (self.model_result.llf / null_llf)
        
        # Deviance
        results['deviance'] = self.model_result.deviance
        
        # AIC and BIC
        results['aic'] = self.model_result.aic
        results['bic'] = self.model_result.bic
        
        # RMSE
        results['rmse'] = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # MAE
        results['mae'] = np.mean(np.abs(y_true - y_pred))
        
        return results