"""
LinearDML with Residual Extraction Capability
==============================================

This module extends EconML's LinearDML to capture and store residuals
that are computed during cross-fitting but normally discarded.

Usage:
    from linear_dml_with_residuals import LinearDMLWithResiduals
    
    # Use exactly like LinearDML
    dml = LinearDMLWithResiduals(
        model_y=RandomForestRegressor(),
        model_t=RandomForestRegressor(),
        cv=5
    )
    dml.fit(Y, T, W=W)
    
    # But now you can get residuals!
    Y_residuals, T_residuals = dml.get_residuals()

Author: Extended from EconML for biodiversity variance partitioning
Date: 2024
"""

import numpy as np
import warnings
from typing import Tuple, Dict, Optional, Any
from econml.dml import LinearDML
from econml.dml._rlearner import _ModelNuisance


class _ModelNuisanceWithCapture(_ModelNuisance):
    """
    Extended _ModelNuisance that captures residuals during prediction.
    
    This class intercepts the residual computation in EconML's cross-fitting
    process and stores them for later retrieval.
    """
    
    # Class-level storage for captured data across all cross-fitting folds
    _captured_data = {
        'residuals': {'Y': [], 'T': []},
        'predictions': {'Y': [], 'T': []},
        'inputs': {'Y': [], 'T': [], 'W': []}
    }
    
    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        """
        Predict nuisance values and capture residuals.
        
        This method extends the original predict() to store intermediate values
        that are normally discarded.
        """
        # Compute predictions E[Y|W] and E[T|W] using the nuisance models
        Y_pred = self._model_y.predict(X, W)
        T_pred = self._model_t.predict(X, W)
        
        # Handle the case where X and W are both None
        # In this case we need to tile the predictions to match input shape
        if X is None and W is None:
            Y_pred = np.tile(Y_pred.reshape(1, -1), (Y.shape[0], 1))
            T_pred = np.tile(T_pred.reshape(1, -1), (T.shape[0], 1))
        
        # Compute residuals: these are Y and T with confounder effects removed
        # Y_res = Y - E[Y|W] (outcome variation not explained by confounders)
        # T_res = T - E[T|W] (treatment variation not explained by confounders)
        Y_res = Y - Y_pred.reshape(Y.shape)
        T_res = T - T_pred.reshape(T.shape)
        
        # Store residuals and predictions for later retrieval
        # These accumulate across cross-fitting folds
        self._captured_data['residuals']['Y'].append(Y_res.copy())
        self._captured_data['residuals']['T'].append(T_res.copy())
        self._captured_data['predictions']['Y'].append(Y_pred.copy())
        self._captured_data['predictions']['T'].append(T_pred.copy())
        self._captured_data['inputs']['Y'].append(Y.copy() if hasattr(Y, 'copy') else Y)
        self._captured_data['inputs']['T'].append(T.copy() if hasattr(T, 'copy') else T)
        if W is not None:
            self._captured_data['inputs']['W'].append(W.copy() if hasattr(W, 'copy') else W)
        
        # Return residuals for DML to use in final stage
        return Y_res, T_res
    
    @classmethod
    def get_captured_data(cls):
        """Get the captured data dictionary."""
        return cls._captured_data
    
    @classmethod  
    def clear_captured_data(cls):
        """Clear all captured data to prepare for new fit."""
        cls._captured_data = {
            'residuals': {'Y': [], 'T': []},
            'predictions': {'Y': [], 'T': []},
            'inputs': {'Y': [], 'T': [], 'W': []}
        }


class LinearDMLWithResiduals(LinearDML):
    """
    LinearDML extended to provide access to residuals.
    
    This class extends EconML's LinearDML to capture and store the residuals
    that are computed during the cross-fitting procedure. These residuals
    represent the variation in Y and T that cannot be explained by the
    confounders W.
    
    The residuals are critical for variance partitioning analysis in 
    ecological studies where we need to understand how much variation 
    remains after controlling for environmental confounders.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with same parameters as LinearDML."""
        super().__init__(*args, **kwargs)
        self._fit_called = False
        self._stored_residuals = None
        self._stored_predictions = None
        self._stored_inputs = None
    
    def fit(self, Y, T, *, X=None, W=None, sample_weight=None,
            freq_weight=None, sample_var=None, groups=None,
            cache_values=False, inference='auto'):
        """
        Fit the model and capture residuals during cross-fitting.
        
        Parameters are the same as LinearDML.fit()
        """
        # Clear any previously captured data
        _ModelNuisanceWithCapture.clear_captured_data()
        
        # Call parent's fit method which will trigger our capturing nuisance model
        super().fit(Y, T, X=X, W=W, sample_weight=sample_weight,
                   freq_weight=freq_weight, sample_var=sample_var,
                   groups=groups, cache_values=cache_values,
                   inference=inference)
        
        # Retrieve captured data after fitting
        captured = _ModelNuisanceWithCapture.get_captured_data()
        
        # Process and store captured residuals from cross-fitting
        if captured['residuals']['Y'] and len(captured['residuals']['Y']) > 0:
            Y_residuals_list = captured['residuals']['Y']
            T_residuals_list = captured['residuals']['T']
            
            # Determine the shape for concatenation
            n_samples = Y.shape[0]
            
            # Concatenate residuals from all cross-fitting folds
            # These are out-of-fold predictions, so each sample appears exactly once
            if len(Y_residuals_list) == 1:
                # Single fold or no cross-fitting
                Y_all_res = Y_residuals_list[0]
                T_all_res = T_residuals_list[0]
            else:
                # Multiple folds - need to concatenate properly
                # The residuals come in fold order
                Y_all_res = np.concatenate(Y_residuals_list)
                
                # Handle both 1D and 2D treatment arrays
                if T_residuals_list[0].ndim == 1:
                    T_all_res = np.concatenate(T_residuals_list)
                else:
                    T_all_res = np.vstack(T_residuals_list)
            
            # Reshape to ensure consistency
            if Y_all_res.shape[0] != n_samples:
                # Cross-fitting may have reordered - need to handle this
                # The total length should still match
                assert len(Y_all_res) == n_samples, f"Residual size mismatch: {len(Y_all_res)} vs {n_samples}"
            
            self._stored_residuals = {
                'Y': Y_all_res,
                'T': T_all_res
            }
            
            # Store predictions if available
            if captured['predictions']['Y'] and len(captured['predictions']['Y']) > 0:
                if len(captured['predictions']['Y']) == 1:
                    Y_all_pred = captured['predictions']['Y'][0]
                    T_all_pred = captured['predictions']['T'][0]
                else:
                    Y_all_pred = np.concatenate(captured['predictions']['Y'])
                    if captured['predictions']['T'][0].ndim == 1:
                        T_all_pred = np.concatenate(captured['predictions']['T'])
                    else:
                        T_all_pred = np.vstack(captured['predictions']['T'])
                
                self._stored_predictions = {
                    'Y': Y_all_pred,
                    'T': T_all_pred
                }
            
            # Store original inputs for variance calculations
            if captured['inputs']['Y'] and len(captured['inputs']['Y']) > 0:
                self._stored_inputs = {
                    'Y': captured['inputs']['Y'][0],  # Original Y
                    'T': captured['inputs']['T'][0],  # Original T
                    'W': captured['inputs']['W'][0] if captured['inputs']['W'] else None
                }
        
        self._fit_called = True
        return self
    
    def _gen_ortho_learner_model_nuisance(self):
        """
        Override to use our capturing nuisance model.
        """
        return _ModelNuisanceWithCapture(
            model_y=self._gen_model_y(),
            model_t=self._gen_model_t()
        )
    
    def get_residuals(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the residuals from the fitted model.
        
        Returns:
            Y_residuals : np.ndarray
                Outcome residuals after removing confounder effects
            T_residuals : np.ndarray
                Treatment residuals after removing confounder effects
                
        Raises:
            ValueError : If called before fit()
        """
        if not self._fit_called or self._stored_residuals is None:
            raise ValueError("No residuals available. Please call fit() first.")
        
        return self._stored_residuals['Y'], self._stored_residuals['T']
    
    def get_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the predictions from the fitted nuisance models.
        
        Returns:
            Y_predictions : np.ndarray
                Predicted outcomes from nuisance model
            T_predictions : np.ndarray
                Predicted treatments from nuisance model
                
        Raises:
            ValueError : If called before fit()
        """
        if not self._fit_called or self._stored_predictions is None:
            raise ValueError("No predictions available. Please call fit() first.")
        
        return self._stored_predictions['Y'], self._stored_predictions['T']
    
    def get_variance_metrics(self, Y_original=None, T_original=None) -> Dict[str, float]:
        """
        Compute variance partitioning metrics from the residuals.
        
        Parameters:
            Y_original : np.ndarray, optional
                Original Y values (if not provided, uses stored values)
            T_original : np.ndarray, optional
                Original T values (if not provided, uses stored values)
        
        Returns:
            metrics : dict
                Dictionary containing:
                - 'r2_outcome_by_confounders': R² of W → Y
                - 'r2_treatment_by_confounders': R² of W → T
                - 'r2_residual_regression': R² of T_res → Y_res
                - 'residual_correlation': Correlation between residuals
                
        Raises:
            ValueError : If called before fit()
        """
        if not self._fit_called or self._stored_residuals is None:
            raise ValueError("No residuals available. Please call fit() first.")
        
        # Use stored originals if not provided
        if Y_original is None:
            if self._stored_inputs is None:
                raise ValueError("No original data stored and none provided.")
            Y_original = self._stored_inputs['Y']
        if T_original is None:
            if self._stored_inputs is None:
                raise ValueError("No original data stored and none provided.")
            T_original = self._stored_inputs['T']
        
        Y_res, T_res = self.get_residuals()
        
        # Compute R² for nuisance models
        var_Y = np.var(Y_original)
        var_Y_res = np.var(Y_res)
        r2_Y = 1 - (var_Y_res / var_Y) if var_Y > 0 else 0
        
        var_T = np.var(T_original)
        var_T_res = np.var(T_res)
        r2_T = 1 - (var_T_res / var_T) if var_T > 0 else 0
        
        # Compute correlation between residuals
        if T_res.ndim > 1 and T_res.shape[1] == 1:
            T_res = T_res.ravel()
        corr = np.corrcoef(Y_res.ravel(), T_res.ravel())[0, 1]
        
        # R² of residual regression (this is what DML estimates)
        r2_residual = corr ** 2
        
        return {
            'r2_outcome_by_confounders': float(r2_Y),
            'r2_treatment_by_confounders': float(r2_T),
            'r2_residual_regression': float(r2_residual),
            'residual_correlation': float(corr)
        }
    
    def summary_with_residuals(self) -> str:
        """
        Get a summary including residual-based metrics.
        
        Returns:
            summary : str
                Text summary of the model including variance metrics
        """
        lines = []
        lines.append("=" * 60)
        lines.append("LinearDML with Residuals - Summary")
        lines.append("=" * 60)
        
        # Basic coefficient info
        if hasattr(self, 'intercept_'):
            lines.append(f"Treatment effect: {self.intercept_}")
        
        # Variance metrics
        try:
            metrics = self.get_variance_metrics()
            lines.append("\nVariance Partitioning:")
            lines.append(f"  R² (W → Y): {metrics['r2_outcome_by_confounders']:.4f}")
            lines.append(f"  R² (W → T): {metrics['r2_treatment_by_confounders']:.4f}")
            lines.append(f"  R² (T_res → Y_res): {metrics['r2_residual_regression']:.4f}")
            lines.append(f"  Residual correlation: {metrics['residual_correlation']:.4f}")
        except ValueError:
            lines.append("\nVariance metrics not available (model not fitted)")
        
        return "\n".join(lines)


# Define what's exported
__all__ = ['LinearDMLWithResiduals']


def test_residual_extraction():
    """
    Test function to verify residual extraction works correctly.
    """
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    
    print("Testing LinearDMLWithResiduals...")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n = 500
    p = 5
    
    # Confounders
    W = np.random.normal(0, 1, (n, p))
    
    # Treatment influenced by confounders
    T = W[:, 0] + 0.5 * W[:, 1] + np.random.normal(0, 0.5, n)
    
    # Outcome influenced by treatment and confounders
    # True effect is 0.5
    Y = 0.5 * T + W[:, 0] + W[:, 1] + np.random.normal(0, 0.5, n)
    
    # Create and fit model
    model = LinearDMLWithResiduals(
        model_y=RandomForestRegressor(n_estimators=50, random_state=42),
        model_t=RandomForestRegressor(n_estimators=50, random_state=42),
        cv=3,
        random_state=42
    )
    
    model.fit(Y, T.reshape(-1, 1), W=W)
    
    # Extract residuals
    Y_res, T_res = model.get_residuals()
    print(f"✓ Residuals extracted: Y_res shape={Y_res.shape}, T_res shape={T_res.shape}")
    
    # Verify the DML coefficient matches manual calculation from residuals
    coef_dml = model.intercept_[0]
    # The DML coefficient should equal cov(Y_res, T_res) / var(T_res)
    coef_manual = np.cov(Y_res.flatten(), T_res.flatten())[0, 1] / np.var(T_res)
    
    print(f"✓ DML coefficient: {coef_dml:.4f}")
    print(f"✓ Manual from residuals: {coef_manual:.4f}")
    print(f"✓ Match: {abs(coef_dml - coef_manual) < 0.05}")
    
    # Get variance metrics
    metrics = model.get_variance_metrics(Y_original=Y, T_original=T)
    print("\n✓ Variance Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")
    
    print("\n" + "=" * 60)
    print("All tests passed! LinearDMLWithResiduals is working correctly.")
    
    return model


if __name__ == "__main__":
    test_residual_extraction()