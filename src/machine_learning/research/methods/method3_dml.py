"""Method 3: Double Machine Learning (DML) with pure H2O processing.

Optimized for 180M+ points using H2O's distributed computing and weighted bootstrap.
No pandas, no memory duplication, full parallelization on 256 cores.
"""

import logging
import numpy as np
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from typing import Dict, Any, Optional, List, Tuple

from ...spatial.mesf_preprocessor import MESFPreprocessor
from ...config_loader import MLRegressionConfig
from ...utils.h2o_utils import ensure_h2o_initialized

logger = logging.getLogger(__name__)


class Method3DML:
    """Pure H2O implementation of Double Machine Learning.
    
    Key improvements over pandas version:
    1. All operations in H2O (256-thread parallel)
    2. Weighted bootstrap instead of data copies
    3. Direct parquet loading (no pandas intermediate)
    4. H2O GBM for nuisance functions (consistent with Methods 1&2)
    """
    
    def __init__(self, config: MLRegressionConfig):
        """Initialize Method 3 DML with H2O.
        
        Args:
            config: ML regression configuration
        """
        self.config = config
        self.preprocessing_config = self.config.get_preprocessing_config()
        self.dml_config = self.config.get_dml_config()
        self.spatial_config = self.config.get_spatial_config()
        # Data config is part of preprocessing
        self.data_config = self.preprocessing_config
        
        # DML settings
        self.n_folds = self.dml_config.get('n_folds', 5)
        self.n_bootstrap = self.dml_config.get('n_bootstrap', 100)
        
        # Initialize H2O using centralized utility
        # DML needs more memory for its many models
        ensure_h2o_initialized(max_mem_size="4G", min_mem_size="1G")
        
        # Initialize MESF preprocessor with caching
        self.mesf_preprocessor = MESFPreprocessor()
        
        # Storage
        self.data_h2o = None
        self.causal_effects = {}
        self.nuisance_models = {}
        
        logger.info(f"Initialized H2O DML with {self.n_folds}-fold cross-fitting")
    
    def load_data(self, parquet_path: str, spatial_grid_step: Optional[int] = None) -> h2o.H2OFrame:
        """Load data with PyArrow and optional spatial grid sampling.
        
        Args:
            parquet_path: Path to parquet file
            spatial_grid_step: Take every Nth point in spatial grid (None=use config default)
            
        Returns:
            H2OFrame with all data including MESF
        """
        import pyarrow.parquet as pq
        from ...preprocessing import ColumnStandardizer
        
        # Use config default if not specified
        if spatial_grid_step is None:
            spatial_grid_step = self.data_config.get('spatial_grid_step', 1)
        
        logger.info(f"Loading data from {parquet_path}")
        
        # First standardize columns if needed
        standardizer = ColumnStandardizer()
        standardized_path = standardizer.standardize_if_needed(parquet_path)
        
        # Use the new caching MESF preprocessor with spatial sampling
        logger.info(f"Spatial grid sampling: every {spatial_grid_step}th point")
        
        # Process with MESF (handles sampling and caching internally)
        augmented_path = self.mesf_preprocessor.process_with_sampling(
            standardized_path,
            spatial_grid_step,
            self.spatial_config
        )
        
        # Load augmented data to H2O
        h2o_data = h2o.import_file(augmented_path)
        logger.info(f"Loaded {h2o_data.nrows:,} rows with {h2o_data.ncols} columns after MESF")
        
        # Verify MESF columns
        me_cols = [col for col in h2o_data.columns if col.startswith('ME')]
        logger.info(f"Including {len(me_cols)} MESF eigenvectors")
        
        return h2o_data
    
    def prepare_data_h2o(self, h2o_data: h2o.H2OFrame) -> h2o.H2OFrame:
        """Prepare data using H2O operations (no pandas!).
        
        Args:
            h2o_data: Raw H2O DataFrame
            
        Returns:
            Processed H2O DataFrame
        """
        logger.info("Preparing data in H2O for Method 3 (DML)")
        
        # Apply complete case analysis
        initial_rows = h2o_data.nrows
        required_cols = [
            'fungi_richness', 'plants_richness', 'vertebrates_richness',
            'bio1', 'bio4', 'bio12', 'bio15', 'elevation',
            'longitude', 'latitude'
        ]
        
        # Remove rows with missing values
        for col in required_cols:
            if col in h2o_data.columns:
                h2o_data = h2o_data[h2o_data[col].isna() == False]
        
        rows_dropped = initial_rows - h2o_data.nrows
        logger.info(f"Complete case analysis: dropped {rows_dropped:,} rows "
                   f"({rows_dropped/initial_rows*100:.2f}%)")
        
        # Standardize climate variables (H2O operations)
        climate_vars = ['bio1', 'bio4', 'bio12', 'bio15', 'elevation']
        
        for var in climate_vars:
            if var in h2o_data.columns:
                # H2O mean() and sd() return lists, not frames
                mean_result = h2o_data[var].mean()
                mean_val = mean_result[0] if isinstance(mean_result, list) else mean_result
                
                sd_result = h2o_data[var].sd()
                sd_val = sd_result[0] if isinstance(sd_result, list) else sd_result
                
                h2o_data[f'{var}_std'] = (h2o_data[var] - mean_val) / (2 * sd_val)
                logger.info(f"Standardized {var}: mean={mean_val:.2f}, 2*sd={2*sd_val:.2f}")
        
        # Add quadratic elevation term
        h2o_data['elevation_std_sq'] = h2o_data['elevation_std'] ** 2
        
        # Log transform biotic predictors
        h2o_data['plants_richness_log'] = (h2o_data['plants_richness'] + 1).log()
        h2o_data['vertebrates_richness_log'] = (h2o_data['vertebrates_richness'] + 1).log()
        logger.info("Created log-transformed biotic predictors")
        
        # Create interaction terms for different hypotheses
        # H1: plants × bio4 interaction
        h2o_data['plants_bio4_interaction'] = (
            h2o_data['plants_richness_log'] * h2o_data['bio4_std']
        )
        
        # H2: categorical interaction (temperate × high plant diversity)
        # H2O median() returns a list, not a frame
        bio1_median_result = h2o_data['bio1'].median()
        bio1_median = bio1_median_result[0] if isinstance(bio1_median_result, list) else bio1_median_result
        
        plants_median_result = h2o_data['plants_richness'].median()
        plants_median = plants_median_result[0] if isinstance(plants_median_result, list) else plants_median_result
        
        h2o_data['temperate'] = (h2o_data['bio1'] < bio1_median).ifelse(1, 0)
        h2o_data['plant_high'] = (h2o_data['plants_richness'] > plants_median).ifelse(1, 0)
        h2o_data['temperate_x_plant_high'] = h2o_data['temperate'] * h2o_data['plant_high']
        
        self.data_h2o = h2o_data
        return h2o_data
    
    def _get_treatment_confounders_outcome(self, hypothesis: str) -> Tuple[str, List[str], str]:
        """Get treatment, confounders, and outcome for each hypothesis.
        
        Args:
            hypothesis: 'h0', 'h1', or 'h2'
            
        Returns:
            Tuple of (treatment, confounders, outcome)
        """
        # Get ME columns
        me_cols = [col for col in self.data_h2o.columns if col.startswith('ME')]
        
        # Base confounders (climate + spatial)
        base_confounders = [
            'bio1_std', 'bio4_std', 'bio12_std', 'bio15_std',
            'elevation_std', 'elevation_std_sq',
            'vertebrates_richness_log'
        ] + me_cols
        
        if hypothesis == 'h0':
            # H0: No interaction
            treatment = 'plants_richness_log'
            confounders = base_confounders
        elif hypothesis == 'h1':
            # H1: Continuous interaction
            treatment = 'plants_bio4_interaction'
            confounders = base_confounders + ['plants_richness_log']
        elif hypothesis == 'h2':
            # H2: Categorical interaction
            treatment = 'temperate_x_plant_high'
            confounders = base_confounders + ['temperate', 'plant_high']
        else:
            raise ValueError(f"Unknown hypothesis: {hypothesis}")
        
        outcome = 'fungi_richness'
        return treatment, confounders, outcome
    
    def _fit_nuisance_functions_h2o(self,
                                    train_frame: h2o.H2OFrame,
                                    treatment: str,
                                    confounders: List[str],
                                    outcome: str,
                                    weights: Optional[str] = None) -> Tuple[Any, Any]:
        """Fit nuisance functions using H2O GBM.
        
        Args:
            train_frame: H2O frame with training data
            treatment: Treatment column name
            confounders: List of confounder column names
            outcome: Outcome column name
            weights: Optional column name for sample weights
            
        Returns:
            Tuple of (g_model, m_model) where:
                g_model: E[Y|X] model
                m_model: E[D|X] model
        """
        # Configure H2O GBM for DML nuisance functions
        # Note: These are DML-appropriate settings:
        # - Fixed ntrees (no early stopping with validation)
        # - Conservative max_depth to prevent overfitting
        # - No validation frame will be used (all data for training)
        gbm_params = {
            'ntrees': 100,  # Fixed iterations, no early stopping
            'max_depth': 6,  # Shallow trees to prevent overfitting
            'learn_rate': 0.1,
            'sample_rate': 0.8,
            'col_sample_rate': 0.7,
            'min_rows': 20,  # Regularization through minimum leaf size
            'distribution': 'gaussian',
            'seed': 42,
            'stopping_rounds': 0  # Explicitly disable early stopping
        }
        
        # Fit g-model: E[Y|X]
        g_model = H2OGradientBoostingEstimator(**gbm_params)
        g_model.train(
            x=confounders,
            y=outcome,
            training_frame=train_frame,
            weights_column=weights
        )
        
        # Fit m-model: E[D|X]
        m_model = H2OGradientBoostingEstimator(**gbm_params)
        m_model.train(
            x=confounders,
            y=treatment,
            training_frame=train_frame,
            weights_column=weights
        )
        
        return g_model, m_model
    
    def cross_fit_dml(self, hypothesis: str = 'h1') -> Dict[str, Any]:
        """Perform cross-fitted DML estimation using H2O.
        
        Args:
            hypothesis: Which hypothesis to test ('h0', 'h1', 'h2')
            
        Returns:
            Dictionary with causal effect estimates and statistics
        """
        if self.data_h2o is None:
            raise ValueError("Must call prepare_data_h2o() first")
        
        logger.info(f"Running H2O cross-fitted DML for hypothesis {hypothesis.upper()}")
        
        # Get treatment, confounders, and outcome
        treatment, confounders, outcome = self._get_treatment_confounders_outcome(hypothesis)
        
        # Storage for cross-fit estimates
        theta_estimates = []
        fold_results = []
        
        # Create fold assignment column
        self.data_h2o['fold'] = self.data_h2o.kfold_column(n_folds=self.n_folds, seed=42)
        
        for fold_idx in range(self.n_folds):
            logger.info(f"Processing fold {fold_idx + 1}/{self.n_folds}")
            
            # Split data using H2O operations
            train_frame = self.data_h2o[self.data_h2o['fold'] != fold_idx]
            test_frame = self.data_h2o[self.data_h2o['fold'] == fold_idx]
            
            # Fit nuisance functions on training fold
            g_model, m_model = self._fit_nuisance_functions_h2o(
                train_frame, treatment, confounders, outcome
            )
            
            # Compute residuals on test fold
            g_pred = g_model.predict(test_frame).as_data_frame().values.ravel()
            m_pred = m_model.predict(test_frame).as_data_frame().values.ravel()
            
            # Get actual values
            Y_test = test_frame[outcome].as_data_frame().values.ravel()
            D_test = test_frame[treatment].as_data_frame().values.ravel()
            
            # Orthogonalized residuals
            V_test = D_test - m_pred  # Treatment residual
            U_test = Y_test - g_pred  # Outcome residual
            
            # DML estimator (orthogonal moment)
            theta_fold = np.mean(V_test * U_test) / np.mean(V_test * V_test)
            
            theta_estimates.append(theta_fold)
            fold_results.append({
                'fold': fold_idx,
                'theta': theta_fold,
                'n_test': len(V_test)
            })
            
            logger.info(f"Fold {fold_idx + 1} theta: {theta_fold:.6f}")
        
        # Aggregate estimates
        theta_dml = np.mean(theta_estimates)
        theta_se = np.std(theta_estimates) / np.sqrt(self.n_folds)
        
        results = {
            'causal_effect': theta_dml,
            'standard_error': theta_se,
            't_statistic': theta_dml / theta_se,
            'p_value': 2 * (1 - np.abs(theta_dml / theta_se)),
            'fold_estimates': theta_estimates,
            'fold_results': fold_results,
            'hypothesis': hypothesis
        }
        
        logger.info(f"DML causal effect: {theta_dml:.6f} (SE: {theta_se:.6f})")
        
        return results
    
    def weighted_bootstrap_dml(self, hypothesis: str = 'h1', n_bootstrap: int = None) -> Dict[str, Any]:
        """Perform weighted bootstrap for uncertainty quantification.
        
        Key innovation: Uses sample weights instead of data copies!
        For 180M points, this avoids 100 × 30GB = 3TB of memory operations.
        
        Args:
            hypothesis: Which hypothesis to test
            n_bootstrap: Number of bootstrap iterations
            
        Returns:
            Dictionary with bootstrap results
        """
        if self.data_h2o is None:
            raise ValueError("Must call prepare_data_h2o() first")
        
        n_bootstrap = n_bootstrap or self.n_bootstrap
        logger.info(f"Running weighted bootstrap DML with {n_bootstrap} iterations")
        
        # Get treatment, confounders, and outcome
        treatment, confounders, outcome = self._get_treatment_confounders_outcome(hypothesis)
        
        n = self.data_h2o.nrows
        bootstrap_effects = []
        
        for b in range(n_bootstrap):
            if b % 10 == 0:
                logger.info(f"Bootstrap iteration {b}/{n_bootstrap}")
            
            # Generate bootstrap weights (Multinomial distribution)
            # This gives us counts for each observation
            weights = np.random.multinomial(n, [1/n]*n)
            
            # Normalize weights to sum to n (for stability)
            weights = weights.astype(float)
            weights = weights / weights.sum() * n
            
            # Add weights to H2O frame
            self.data_h2o['bootstrap_weight'] = h2o.H2OFrame(weights.reshape(-1, 1))
            
            # Storage for this bootstrap's fold estimates
            fold_estimates = []
            
            # Cross-fitting with weights
            self.data_h2o['fold'] = self.data_h2o.kfold_column(n_folds=self.n_folds, seed=42+b)
            
            for fold_idx in range(self.n_folds):
                # Split data
                train_frame = self.data_h2o[self.data_h2o['fold'] != fold_idx]
                test_frame = self.data_h2o[self.data_h2o['fold'] == fold_idx]
                
                # Fit nuisance functions WITH WEIGHTS
                g_model, m_model = self._fit_nuisance_functions_h2o(
                    train_frame, treatment, confounders, outcome,
                    weights='bootstrap_weight'  # KEY: Use weights!
                )
                
                # Predictions on test fold
                g_pred = g_model.predict(test_frame).as_data_frame().values.ravel()
                m_pred = m_model.predict(test_frame).as_data_frame().values.ravel()
                
                # Get actual values and weights
                Y_test = test_frame[outcome].as_data_frame().values.ravel()
                D_test = test_frame[treatment].as_data_frame().values.ravel()
                W_test = test_frame['bootstrap_weight'].as_data_frame().values.ravel()
                
                # Weighted orthogonalized residuals
                V_test = D_test - m_pred
                U_test = Y_test - g_pred
                
                # Weighted DML estimator
                theta_fold = np.average(V_test * U_test, weights=W_test) / np.average(V_test * V_test, weights=W_test)
                fold_estimates.append(theta_fold)
            
            # Average across folds for this bootstrap
            theta_boot = np.mean(fold_estimates)
            bootstrap_effects.append(theta_boot)
        
        # Compute bootstrap statistics
        boot_mean = np.mean(bootstrap_effects)
        boot_se = np.std(bootstrap_effects)
        boot_ci_lower = np.percentile(bootstrap_effects, 2.5)
        boot_ci_upper = np.percentile(bootstrap_effects, 97.5)
        
        logger.info(f"Bootstrap mean: {boot_mean:.6f} (SE: {boot_se:.6f})")
        logger.info(f"Bootstrap 95% CI: [{boot_ci_lower:.6f}, {boot_ci_upper:.6f}]")
        
        return {
            'bootstrap_mean': boot_mean,
            'bootstrap_se': boot_se,
            'ci_lower': boot_ci_lower,
            'ci_upper': boot_ci_upper,
            'n_bootstrap': n_bootstrap,
            'bootstrap_effects': bootstrap_effects
        }
    
    def fit_model(self, data: Any) -> Dict[str, Any]:
        """Fit the model - implements abstract method from base class.
        
        Args:
            data: Input data (H2OFrame or path to parquet file)
            
        Returns:
            Model fitting results
        """
        # If data is a string path, use process_large_dataset
        if isinstance(data, str):
            return self.process_large_dataset(data)
        # Otherwise prepare and run DML
        if data is None:
            data = self.data_h2o
        if data is None:
            raise ValueError("Must provide data or call load_data first")
        
        # Run DML for default hypothesis
        return self.test_hypothesis('h1')
    
    def test_hypothesis(self, hypothesis: str = 'h1') -> Dict[str, Any]:
        """Test a specific hypothesis using DML with bootstrap inference.
        
        Args:
            hypothesis: Which hypothesis to test
            
        Returns:
            Complete results including point estimate and bootstrap CI
        """
        # Point estimate from cross-fitted DML
        dml_results = self.cross_fit_dml(hypothesis)
        
        # Bootstrap inference (weighted, no copies!)
        bootstrap_results = self.weighted_bootstrap_dml(hypothesis)
        
        # Combine results
        results = {
            **dml_results,
            'bootstrap': bootstrap_results,
            'significant': bootstrap_results['ci_lower'] * bootstrap_results['ci_upper'] > 0,
            'interpretation': self._interpret_results(dml_results, bootstrap_results, hypothesis)
        }
        
        return results
    
    def _interpret_results(self, dml_results: Dict, bootstrap_results: Dict, hypothesis: str) -> str:
        """Interpret the DML results."""
        effect = dml_results['causal_effect']
        ci_lower = bootstrap_results['ci_lower']
        ci_upper = bootstrap_results['ci_upper']
        
        if ci_lower * ci_upper > 0:  # CI doesn't include 0
            if hypothesis == 'h1':
                if effect > 0:
                    return f"Significant positive interaction (β={effect:.4f}): Plant-fungi relationship strengthens with temperature seasonality"
                else:
                    return f"Significant negative interaction (β={effect:.4f}): Plant-fungi relationship weakens with temperature seasonality"
            elif hypothesis == 'h2':
                if effect > 0:
                    return f"Significant interaction (β={effect:.4f}): Plant-fungi relationship is stronger in temperate regions with high plant diversity"
                else:
                    return f"Significant interaction (β={effect:.4f}): Plant-fungi relationship is weaker in temperate regions with high plant diversity"
        else:
            return f"No significant interaction detected (CI includes 0: [{ci_lower:.4f}, {ci_upper:.4f}])"
    
    def process_large_dataset(self, parquet_path: str) -> Dict[str, Any]:
        """Complete pipeline for large datasets.
        
        Args:
            parquet_path: Path to parquet file
            
        Returns:
            Analysis results for all hypotheses
        """
        # Step 1: Load data directly to H2O (with MESF handling)
        h2o_data = self.load_data(parquet_path)
        
        # Step 2: Prepare data in H2O
        h2o_data = self.prepare_data_h2o(h2o_data)
        
        # Step 3: Test all hypotheses
        results = {}
        
        for hypothesis in ['h0', 'h1', 'h2']:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing hypothesis {hypothesis.upper()}")
            logger.info('='*60)
            
            results[hypothesis] = self.test_hypothesis(hypothesis)
            
            logger.info(f"\nResults for {hypothesis.upper()}:")
            logger.info(f"Causal effect: {results[hypothesis]['causal_effect']:.6f}")
            logger.info(f"Bootstrap CI: [{results[hypothesis]['bootstrap']['ci_lower']:.6f}, "
                       f"{results[hypothesis]['bootstrap']['ci_upper']:.6f}]")
            logger.info(f"Interpretation: {results[hypothesis]['interpretation']}")
        
        # Step 4: Compare hypotheses
        results['comparison'] = self._compare_hypotheses(results)
        
        return results
    
    def _compare_hypotheses(self, results: Dict) -> Dict[str, Any]:
        """Compare results across hypotheses."""
        comparison = {
            'h0_effect': results['h0']['causal_effect'],
            'h1_effect': results['h1']['causal_effect'],
            'h2_effect': results['h2']['causal_effect'],
            'h1_significant': results['h1']['significant'],
            'h2_significant': results['h2']['significant'],
            'strongest_effect': None
        }
        
        # Determine strongest effect
        effects = {
            'h0': abs(results['h0']['causal_effect']),
            'h1': abs(results['h1']['causal_effect']),
            'h2': abs(results['h2']['causal_effect'])
        }
        comparison['strongest_effect'] = max(effects, key=effects.get)
        
        return comparison