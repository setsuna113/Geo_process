"""Permutation feature importance for robust feature evaluation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional, Union, Any
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PermutationImportance:
    """
    Calculate permutation feature importance for any model.
    
    This is more robust than model-specific importance as it measures
    actual impact on predictions.
    """
    
    def __init__(self, 
                 n_repeats: int = 10,
                 random_state: Optional[int] = 42,
                 scoring: Union[str, Callable] = 'r2'):
        """
        Initialize permutation importance calculator.
        
        Args:
            n_repeats: Number of times to permute each feature
            random_state: Random seed for reproducibility
            scoring: Scoring function ('r2', 'mse', or callable)
        """
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.scoring = self._get_scorer(scoring)
        self.importances_ = None
        self.baseline_score_ = None
        
    def _get_scorer(self, scoring: Union[str, Callable]) -> Callable:
        """Get scoring function."""
        if callable(scoring):
            return scoring
        elif scoring == 'r2':
            return r2_score
        elif scoring == 'mse':
            return lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
        else:
            raise ValueError(f"Unknown scoring method: {scoring}")
    
    def calculate(self, 
                  model,
                  X: pd.DataFrame,
                  y: Union[pd.Series, np.ndarray],
                  sample_weight: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate permutation importance for all features.
        
        Args:
            model: Fitted model with predict method
            X: Feature matrix
            y: True target values
            sample_weight: Optional sample weights
            
        Returns:
            Dict with importance scores and statistics
        """
        # Ensure we have a DataFrame with column names
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        # Calculate baseline score
        y_pred = model.predict(X)
        self.baseline_score_ = self.scoring(y, y_pred)
        
        logger.info(f"Baseline score: {self.baseline_score_:.4f}")
        
        # Initialize results
        importances = {}
        
        # Set random seed
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Calculate importance for each feature
        for col in tqdm(X.columns, desc="Calculating permutation importance"):
            scores = []
            
            for _ in range(self.n_repeats):
                # Create copy and permute feature
                X_permuted = X.copy()
                X_permuted[col] = np.random.permutation(X_permuted[col])
                
                # Calculate score with permuted feature
                y_pred_permuted = model.predict(X_permuted)
                score_permuted = self.scoring(y, y_pred_permuted)
                
                # Importance is decrease in performance
                importance = self.baseline_score_ - score_permuted
                scores.append(importance)
            
            # Store statistics
            importances[col] = {
                'importance_mean': np.mean(scores),
                'importance_std': np.std(scores),
                'importance_values': scores,
                'relative_importance': np.mean(scores) / abs(self.baseline_score_) 
                                     if self.baseline_score_ != 0 else 0
            }
        
        self.importances_ = importances
        return importances
    
    def get_feature_importance_df(self) -> pd.DataFrame:
        """Get importance scores as a DataFrame sorted by importance."""
        if self.importances_ is None:
            raise ValueError("Must call calculate() first")
        
        data = []
        for feature, stats in self.importances_.items():
            data.append({
                'feature': feature,
                'importance': stats['importance_mean'],
                'std': stats['importance_std'],
                'relative_importance': stats['relative_importance']
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('importance', ascending=False)
    
    def plot_importance(self, top_n: Optional[int] = None, figsize: tuple = (10, 6)):
        """Plot feature importance with error bars."""
        import matplotlib.pyplot as plt
        
        df = self.get_feature_importance_df()
        
        if top_n is not None:
            df = df.head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(df))
        ax.barh(y_pos, df['importance'], xerr=df['std'], 
                align='center', alpha=0.8, capsize=5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance (decrease in score when permuted)')
        ax.set_title('Permutation Feature Importance')
        
        # Add vertical line at 0
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        return fig, ax


class TargetPermutationTest:
    """
    Test if model performance is better than random by permuting target.
    
    This tests if the relationship between features and target is real.
    """
    
    def __init__(self, n_permutations: int = 100, random_state: Optional[int] = 42):
        self.n_permutations = n_permutations
        self.random_state = random_state
        
    def test(self, 
             model_class,
             X: pd.DataFrame,
             y: Union[pd.Series, np.ndarray],
             model_params: Optional[dict] = None,
             scoring: Union[str, Callable] = 'r2') -> Dict[str, Any]:
        """
        Test model against random permutations of target.
        
        Args:
            model_class: Class of model to instantiate
            X: Feature matrix
            y: Target values
            model_params: Parameters for model initialization
            scoring: Scoring method
            
        Returns:
            Dict with test results and p-value
        """
        if model_params is None:
            model_params = {}
        
        # Get scorer
        if scoring == 'r2':
            scorer = r2_score
        elif scoring == 'mse':
            scorer = lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
        else:
            scorer = scoring
        
        # Fit model on real data
        model = model_class(**model_params)
        model.fit(X, y)
        y_pred = model.predict(X)
        real_score = scorer(y, y_pred)
        
        # Permutation scores
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        permutation_scores = []
        for i in tqdm(range(self.n_permutations), desc="Target permutation test"):
            # Permute target
            y_permuted = np.random.permutation(y)
            
            # Fit model on permuted data
            model_perm = model_class(**model_params)
            model_perm.fit(X, y_permuted)
            y_pred_perm = model_perm.predict(X)
            
            # Calculate score
            perm_score = scorer(y_permuted, y_pred_perm)
            permutation_scores.append(perm_score)
        
        # Calculate p-value
        permutation_scores = np.array(permutation_scores)
        p_value = np.mean(permutation_scores >= real_score)
        
        return {
            'real_score': real_score,
            'permutation_scores': permutation_scores,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'mean_permutation_score': np.mean(permutation_scores),
            'std_permutation_score': np.std(permutation_scores),
            'percentile': stats.percentileofscore(permutation_scores, real_score)
        }