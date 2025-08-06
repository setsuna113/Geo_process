"""Method 3: Double Machine Learning (DML) for causal inference.

Based on regression_analysis_decisions.md lines 777-887.
Implements DML with cross-fitting for causal effect estimation.
"""

import numpy as np
import pandas as pd
import logging
import gc
from typing import Dict, Any, Optional, Union, Tuple, List
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from scipy import stats

from ...config_loader import MLRegressionConfig
from ...spatial import compute_mesf  # Now uses vectorized implementation
from ...utils import MemoryMonitor, MemoryContext  # Memory optimization utilities

logger = logging.getLogger(__name__)


class Method3DML:
    """Method 3: Double Machine Learning for causal inference.
    
    Implements DML as specified in regression_analysis_decisions.md lines 777-887.
    Uses LightGBM for nuisance function estimation and cross-fitting for
    robust causal effect estimation.
    """
    
    def __init__(self, config: Optional[MLRegressionConfig] = None):
        """Initialize Method 3 DML.
        
        Args:
            config: ML regression configuration
        """
        self.config = config or MLRegressionConfig()
        self.preprocessing_config = self.config.get_preprocessing_config()
        self.lightgbm_config = self.config.get_lightgbm_config()
        self.spatial_config = self.config.get_spatial_config()
        self.dml_config = self.config.get_dml_config()
        
        # DML-specific settings from lines 792-816
        self.n_folds = self.dml_config.get('n_folds', 5)
        self.n_repetitions = self.dml_config.get('n_repetitions', 1)
        
        # Storage for results
        self.data_processed = None
        self.causal_effects = {}
        self.nuisance_models = {}
        self.cross_fit_results = []
        
        logger.info(f"Initialized Method 3: DML with {self.n_folds}-fold cross-fitting")
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for DML analysis.
        
        Implements preprocessing from regression_analysis_decisions.md lines 199-277
        and DML-specific setup from lines 780-785.
        
        Args:
            data: Raw biodiversity data
        
        Returns:
            Processed DataFrame ready for DML
        """
        logger.info("Preparing data for Method 3 (DML)")
        
        # Apply complete case analysis
        initial_rows = len(data)
        data_clean = data.dropna(subset=[
            'fungi_richness', 'plants_richness', 'vertebrates_richness',
            'bio1', 'bio4', 'bio12', 'bio15', 'elevation',
            'longitude', 'latitude'
        ])
        
        rows_dropped = initial_rows - len(data_clean)
        logger.info(f"Complete case analysis: dropped {rows_dropped} rows "
                   f"({rows_dropped/initial_rows*100:.2f}%)")
        
        # Standardize climate variables (2SD method)
        climate_vars = ['bio1', 'bio4', 'bio12', 'bio15', 'elevation']
        
        for var in climate_vars:
            mean_val = data_clean[var].mean()
            sd_val = data_clean[var].std()
            data_clean[f'{var}_std'] = (data_clean[var] - mean_val) / (2 * sd_val)
        
        # Add quadratic elevation term (from regression_analysis_decisions.md lines 288, 301, 314)
        data_clean['elevation_std_sq'] = data_clean['elevation_std'] ** 2
        logger.info("Created quadratic elevation term: elevation_std_sq")
        
        # Log transform biotic predictors (lines 235-240)
        data_clean['plants_richness_log'] = np.log(data_clean['plants_richness'] + 1)
        data_clean['vertebrates_richness_log'] = np.log(data_clean['vertebrates_richness'] + 1)
        data_clean['fungi_richness_log'] = np.log(data_clean['fungi_richness'] + 1)
        
        # Compute spatial eigenvectors
        logger.info("Computing spatial eigenvectors...")
        coords = data_clean[['longitude', 'latitude']].values
        mesf_df = compute_mesf(coords, self.spatial_config)
        
        # Add MESF columns
        for col in mesf_df.columns:
            data_clean[col] = mesf_df[col].values
        
        logger.info(f"Added {len(mesf_df.columns)} spatial eigenvectors")
        
        # Create interaction for H2 testing (continuous version for DML)
        data_clean['plants_bio4_interaction'] = (
            data_clean['plants_richness_log'] * data_clean['bio4_std']
        )
        
        self.data_processed = data_clean
        return data_clean
    
    def _get_treatment_confounders_outcome(self, hypothesis: str) -> Tuple[str, List[str], str]:
        """Define treatment, confounders, and outcome for each hypothesis.
        
        Based on regression_analysis_decisions.md lines 782-785.
        
        Args:
            hypothesis: One of 'h0', 'h1', 'h2'
            
        Returns:
            Tuple of (treatment_var, confounder_list, outcome_var)
        """
        me_cols = [col for col in self.data_processed.columns if col.startswith('ME')]
        
        # Always include climate and spatial as confounders
        base_confounders = [
            'bio1_std', 'bio4_std', 'bio12_std', 'bio15_std', 
            'elevation_std', 'elevation_std_sq'  # Added quadratic term
        ] + me_cols
        
        if hypothesis == 'h0':
            # H0: Test independence of fungi from plants
            treatment = 'plants_richness_log'
            confounders = base_confounders + ['vertebrates_richness_log']
            outcome = 'fungi_richness_log'
            
        elif hypothesis == 'h1':
            # H1: Test biodiversity begets biodiversity
            treatment = 'plants_richness_log'
            confounders = base_confounders + ['vertebrates_richness_log']
            outcome = 'fungi_richness_log'
            
        elif hypothesis == 'h2':
            # H2: Test temperate mismatch (interaction effect)
            treatment = 'plants_bio4_interaction'
            confounders = base_confounders + [
                'plants_richness_log', 
                'vertebrates_richness_log'
            ]
            outcome = 'fungi_richness_log'
            
        else:
            raise ValueError(f"Unknown hypothesis: {hypothesis}")
        
        logger.info(f"DML setup for {hypothesis}:")
        logger.info(f"  Treatment: {treatment}")
        logger.info(f"  Outcome: {outcome}")
        logger.info(f"  Confounders: {len(confounders)} variables")
        
        return treatment, confounders, outcome
    
    def _fit_nuisance_functions(self, 
                               X_train: pd.DataFrame,
                               D_train: pd.Series,
                               Y_train: pd.Series) -> Tuple[Any, Any]:
        """Fit nuisance functions using LightGBM.
        
        Implements nuisance estimation from regression_analysis_decisions.md lines 797-816.
        
        Args:
            X_train: Confounders (training fold)
            D_train: Treatment (training fold)
            Y_train: Outcome (training fold)
            
        Returns:
            Tuple of (g_model, m_model) where:
                g_model: E[Y|X] model
                m_model: E[D|X] model
        """
        # Configure LightGBM parameters
        lgb_params = {
            'objective': self.lightgbm_config.get('objective', 'regression'),
            'num_leaves': self.lightgbm_config.get('num_leaves', 255),
            'learning_rate': self.lightgbm_config.get('learning_rate', 0.05),
            'max_depth': self.lightgbm_config.get('max_depth', 15),
            'min_data_in_leaf': self.lightgbm_config.get('min_data_in_leaf', 20),
            'feature_fraction': self.lightgbm_config.get('feature_fraction', 0.7),
            'bagging_fraction': self.lightgbm_config.get('bagging_fraction', 0.8),
            'bagging_freq': self.lightgbm_config.get('bagging_freq', 5),
            'lambda_l1': self.lightgbm_config.get('lambda_l1', 0.0),
            'lambda_l2': self.lightgbm_config.get('lambda_l2', 0.0),
            'verbose': -1,
            'random_state': 42
        }
        
        # Fit g(X) = E[Y|X]
        g_train_data = lgb.Dataset(X_train, label=Y_train)
        g_model = lgb.train(
            lgb_params,
            g_train_data,
            num_boost_round=self.lightgbm_config.get('num_iterations', 200),
            valid_sets=[g_train_data],
            callbacks=[lgb.log_evaluation(0)]
        )
        
        # Fit m(X) = E[D|X]
        m_train_data = lgb.Dataset(X_train, label=D_train)
        m_model = lgb.train(
            lgb_params,
            m_train_data,
            num_boost_round=self.lightgbm_config.get('num_iterations', 200),
            valid_sets=[m_train_data],
            callbacks=[lgb.log_evaluation(0)]
        )
        
        return g_model, m_model
    
    def _compute_orthogonalized_residuals(self,
                                         X: pd.DataFrame,
                                         D: pd.Series,
                                         Y: pd.Series,
                                         g_model: Any,
                                         m_model: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Compute orthogonalized residuals for DML.
        
        Implements residual calculation from regression_analysis_decisions.md lines 818-833.
        
        Args:
            X: Confounders
            D: Treatment
            Y: Outcome
            g_model: Fitted E[Y|X] model
            m_model: Fitted E[D|X] model
            
        Returns:
            Tuple of (V_residuals, U_residuals) where:
                V_residuals: Y - g(X)
                U_residuals: D - m(X)
        """
        # Predict nuisance functions
        g_pred = g_model.predict(X, num_iteration=g_model.best_iteration)
        m_pred = m_model.predict(X, num_iteration=m_model.best_iteration)
        
        # Compute residuals
        V = Y - g_pred  # Outcome residuals
        U = D - m_pred  # Treatment residuals
        
        return V, U
    
    def cross_fit_dml(self, hypothesis: str = 'h1') -> Dict[str, Any]:
        """Perform cross-fitted DML estimation.
        
        Implements DML algorithm from regression_analysis_decisions.md lines 835-864.
        
        Args:
            hypothesis: Which hypothesis to test ('h0', 'h1', 'h2')
            
        Returns:
            Dictionary with causal effect estimates and statistics
        """
        if self.data_processed is None:
            raise ValueError("Must call prepare_data() first")
        
        logger.info(f"Running cross-fitted DML for hypothesis {hypothesis.upper()}")
        
        # Get treatment, confounders, and outcome
        treatment, confounders, outcome = self._get_treatment_confounders_outcome(hypothesis)
        
        # Extract variables
        D = self.data_processed[treatment].values
        Y = self.data_processed[outcome].values
        X = self.data_processed[confounders]
        
        # Storage for cross-fit estimates
        theta_estimates = []
        fold_results = []
        
        # K-fold cross-fitting
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            with MemoryContext(f"Fold {fold_idx + 1}/{self.n_folds}"):
                logger.info(f"Processing fold {fold_idx + 1}/{self.n_folds}")
                
                # Split data (using views where possible)
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                D_train, D_test = D[train_idx], D[test_idx]
                Y_train, Y_test = Y[train_idx], Y[test_idx]
                
                # Fit nuisance functions on training fold
                g_model, m_model = self._fit_nuisance_functions(X_train, D_train, Y_train)
                
                # Compute residuals on test fold
                V_test, U_test = self._compute_orthogonalized_residuals(
                    X_test, D_test, Y_test, g_model, m_model
                )
                
                # Clean up training data
                del X_train, D_train, Y_train
                gc.collect()
                
                # Estimate causal effect via OLS on residuals (lines 841-843)
                # theta = E[V * U] / E[U^2]
                theta_fold = np.sum(V_test * U_test) / np.sum(U_test ** 2)
                
                # Store results
                theta_estimates.append(theta_fold)
                fold_results.append({
                    'fold': fold_idx,
                    'theta': theta_fold,
                    'n_train': len(train_idx),
                    'n_test': len(test_idx),
                    'V_variance': np.var(V_test),
                    'U_variance': np.var(U_test)
                })
                
                logger.info(f"  Fold {fold_idx + 1} estimate: θ={theta_fold:.4f}")
        
        # Aggregate estimates (lines 854-864)
        theta_final = np.mean(theta_estimates)
        theta_std = np.std(theta_estimates) / np.sqrt(self.n_folds)
        
        # Compute confidence intervals and p-value
        z_score = theta_final / theta_std
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        ci_lower = theta_final - 1.96 * theta_std
        ci_upper = theta_final + 1.96 * theta_std
        
        logger.info(f"\nDML Results for {hypothesis.upper()}:")
        logger.info(f"  Causal effect: {theta_final:.4f} ± {theta_std:.4f}")
        logger.info(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        logger.info(f"  Z-score: {z_score:.2f}, p-value: {p_value:.4f}")
        
        # Interpretation based on hypothesis
        interpretation = self._interpret_effect(hypothesis, theta_final, p_value)
        
        # Store results
        results = {
            'hypothesis': hypothesis,
            'treatment': treatment,
            'outcome': outcome,
            'causal_effect': theta_final,
            'standard_error': theta_std,
            'z_score': z_score,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'fold_estimates': theta_estimates,
            'fold_details': fold_results,
            'interpretation': interpretation,
            'significant': p_value < 0.05
        }
        
        self.causal_effects[hypothesis] = results
        return results
    
    def _interpret_effect(self, hypothesis: str, effect: float, p_value: float) -> str:
        """Interpret the causal effect based on hypothesis.
        
        Args:
            hypothesis: Which hypothesis ('h0', 'h1', 'h2')
            effect: Estimated causal effect
            p_value: P-value for the effect
            
        Returns:
            String interpretation
        """
        if hypothesis == 'h0':
            if p_value >= 0.05:
                return "H0 SUPPORTED: No causal effect of plants on fungi (independence)"
            else:
                if effect > 0:
                    return f"H0 REJECTED: Plants have positive causal effect on fungi (+{effect:.3f})"
                else:
                    return f"H0 REJECTED: Plants have negative causal effect on fungi ({effect:.3f})"
        
        elif hypothesis == 'h1':
            if p_value < 0.05 and effect > 0:
                return f"H1 SUPPORTED: Biodiversity begets biodiversity (causal effect: +{effect:.3f})"
            elif p_value < 0.05 and effect < 0:
                return f"H1 REJECTED: Negative causal effect ({effect:.3f})"
            else:
                return "H1 NOT SUPPORTED: No significant causal effect"
        
        elif hypothesis == 'h2':
            if p_value < 0.05 and effect < 0:
                return f"H2 SUPPORTED: Temperate regions show decoupling (interaction effect: {effect:.3f})"
            elif p_value < 0.05 and effect > 0:
                return f"H2 REJECTED: Temperate regions show stronger coupling (+{effect:.3f})"
            else:
                return "H2 NOT SUPPORTED: No significant temperate interaction"
        
        return "Unknown hypothesis"
    
    def run_robustness_checks(self) -> Dict[str, Any]:
        """Run robustness checks for DML estimates.
        
        Implements checks from regression_analysis_decisions.md lines 866-887.
        
        Returns:
            Dictionary with robustness check results
        """
        logger.info("Running DML robustness checks")
        
        robustness_results = {}
        
        # 1. Sensitivity to number of folds (lines 867-871)
        logger.info("Testing sensitivity to number of folds...")
        fold_sensitivity = []
        
        for n_folds in [3, 5, 10]:
            self.n_folds = n_folds
            h1_result = self.cross_fit_dml('h1')
            fold_sensitivity.append({
                'n_folds': n_folds,
                'effect': h1_result['causal_effect'],
                'se': h1_result['standard_error']
            })
        
        robustness_results['fold_sensitivity'] = fold_sensitivity
        
        # Reset to original
        self.n_folds = self.dml_config.get('n_folds', 5)
        
        # 2. Alternative nuisance specifications (lines 873-878)
        logger.info("Testing alternative nuisance specifications...")
        
        # Try with different LightGBM parameters
        original_params = self.lightgbm_config.copy()
        
        alt_specs = []
        
        # Simpler model
        self.lightgbm_config['num_leaves'] = 31
        self.lightgbm_config['max_depth'] = 5
        h1_simple = self.cross_fit_dml('h1')
        alt_specs.append({
            'specification': 'simple',
            'effect': h1_simple['causal_effect'],
            'se': h1_simple['standard_error']
        })
        
        # More complex model
        self.lightgbm_config['num_leaves'] = 511
        self.lightgbm_config['max_depth'] = -1
        h1_complex = self.cross_fit_dml('h1')
        alt_specs.append({
            'specification': 'complex',
            'effect': h1_complex['causal_effect'],
            'se': h1_complex['standard_error']
        })
        
        robustness_results['nuisance_specifications'] = alt_specs
        
        # Restore original
        self.lightgbm_config = original_params
        
        # 3. Bootstrap confidence intervals (lines 880-887)
        # OPTIMIZED: Using index-based bootstrap without copying data
        logger.info("Computing bootstrap confidence intervals (memory-optimized)...")
        
        n_bootstrap = 100
        bootstrap_effects = []
        
        # Store reference to original data (no copy)
        original_data = self.data_processed
        n = len(original_data)
        
        for b in range(n_bootstrap):
            if b % 20 == 0:
                logger.info(f"  Bootstrap iteration {b}/{n_bootstrap}")
            
            # Generate bootstrap indices (memory efficient)
            boot_idx = np.random.choice(n, n, replace=True)
            
            # Use view instead of copy - massive memory savings
            self.data_processed = original_data.iloc[boot_idx]  # View, not copy
            
            # Re-run DML on bootstrap sample
            h1_boot = self.cross_fit_dml('h1')
            bootstrap_effects.append(h1_boot['causal_effect'])
            
            # Explicit memory cleanup
            import gc
            gc.collect()
        
        # Restore original data reference
        self.data_processed = original_data
        
        # Compute bootstrap CI
        boot_ci_lower = np.percentile(bootstrap_effects, 2.5)
        boot_ci_upper = np.percentile(bootstrap_effects, 97.5)
        
        robustness_results['bootstrap'] = {
            'ci_lower': boot_ci_lower,
            'ci_upper': boot_ci_upper,
            'n_bootstrap': n_bootstrap,
            'effects': bootstrap_effects
        }
        
        logger.info(f"Bootstrap 95% CI: [{boot_ci_lower:.4f}, {boot_ci_upper:.4f}]")
        
        # Summary of robustness
        all_estimates = [fs['effect'] for fs in fold_sensitivity] + \
                       [as_['effect'] for as_ in alt_specs] + \
                       bootstrap_effects
        
        robustness_results['summary'] = {
            'min_effect': np.min(all_estimates),
            'max_effect': np.max(all_estimates),
            'range': np.max(all_estimates) - np.min(all_estimates),
            'robust': np.max(all_estimates) - np.min(all_estimates) < 0.1
        }
        
        if robustness_results['summary']['robust']:
            logger.info("✓ DML estimates appear robust to specification choices")
        else:
            logger.warning("⚠ DML estimates sensitive to specification choices")
        
        return robustness_results
    
    def test_all_hypotheses(self) -> Dict[str, Any]:
        """Test all three hypotheses using DML.
        
        Returns:
            Dictionary with results for all hypotheses
        """
        logger.info("Testing all hypotheses with DML")
        
        results = {}
        
        for hypothesis in ['h0', 'h1', 'h2']:
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing {hypothesis.upper()}")
            logger.info('='*50)
            
            results[hypothesis] = self.cross_fit_dml(hypothesis)
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("DML HYPOTHESIS TESTING SUMMARY")
        logger.info("="*60)
        
        for hyp, res in results.items():
            logger.info(f"\n{hyp.upper()}:")
            logger.info(f"  Effect: {res['causal_effect']:.4f} (p={res['p_value']:.4f})")
            logger.info(f"  {res['interpretation']}")
        
        return results
    
    def create_visualization(self, save_path: Optional[str] = None) -> Any:
        """Create visualization for DML causal effects.
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        from ...visualization import create_dml_effects_plot
        
        if not self.results:
            raise ValueError("Must run analysis first")
        
        # Create visualization
        fig = create_dml_effects_plot(
            self.results,
            save_path=save_path
        )
        
        logger.info("Created Method 3 (DML) visualization")
        return fig
    
    def summarize_results(self) -> Dict[str, Any]:
        """Summarize Method 3 DML results.
        
        Returns:
            Dictionary with complete Method 3 results
        """
        if not self.causal_effects:
            raise ValueError("Must run cross_fit_dml() first")
        
        summary = {
            'method': 'Method 3: Double Machine Learning (DML)',
            'n_folds': self.n_folds,
            'n_samples': len(self.data_processed),
            'hypotheses': {}
        }
        
        for hyp, results in self.causal_effects.items():
            summary['hypotheses'][hyp] = {
                'treatment': results['treatment'],
                'causal_effect': results['causal_effect'],
                'standard_error': results['standard_error'],
                'p_value': results['p_value'],
                'ci_95': [results['ci_lower'], results['ci_upper']],
                'significant': results['significant'],
                'interpretation': results['interpretation']
            }
        
        # Add comparison across hypotheses
        if 'h1' in self.causal_effects and 'h2' in self.causal_effects:
            h1_effect = self.causal_effects['h1']['causal_effect']
            h2_effect = self.causal_effects['h2']['causal_effect']
            
            summary['comparison'] = {
                'h1_vs_h2': {
                    'h1_effect': h1_effect,
                    'h2_effect': h2_effect,
                    'difference': h1_effect - h2_effect,
                    'interpretation': 'H2 shows decoupling' if h2_effect < 0 else 'No decoupling'
                }
            }
        
        return summary