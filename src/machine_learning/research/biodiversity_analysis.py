"""High-level research workflow for biodiversity analysis."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ..models.formula_model import NestedFormulaModels, FormulaTransformer
from ..evaluation.statistical_tests import NestedModelComparison, ModelDiagnostics
from ..evaluation.permutation_importance import PermutationImportance
from ..interpretation.pdp import PartialDependencePlot
from ...config import config

logger = logging.getLogger(__name__)


class BiodiversityResearchPipeline:
    """
    Complete research pipeline for testing biodiversity hypotheses.
    
    Implements the nested model approach for testing:
    1. Non-proxy relationships between biodiversity components
    2. The temperate mismatch hypothesis
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize research pipeline."""
        self.output_dir = output_dir or Path('outputs/research_results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = None
        self.models = {}
        self.results = {}
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for analysis, adding climate proxies from spatial features.
        
        Assumes data has:
        - latitude, longitude
        - plant/animal/fungal richness columns
        - Already computed spatial features
        """
        # Create climate proxies from spatial features if not present
        if 'avg_temp' not in df.columns:
            if 'temperature' in df.columns:
                df['avg_temp'] = df['temperature']
            elif 'spatial_abs_latitude' in df.columns:
                # Use absolute latitude as inverse proxy for temperature
                # Higher latitude = cooler temperature
                df['avg_temp'] = 30 - (df['spatial_abs_latitude'] / 90 * 40)
            elif 'latitude' in df.columns:
                # Use latitude directly
                lat_abs = np.abs(df['latitude'])
                df['avg_temp'] = 30 - (lat_abs / 90 * 40)
        
        if 'seasonal_temp' not in df.columns:
            if 'spatial_abs_latitude' in df.columns:
                lat_abs = df['spatial_abs_latitude']
            elif 'latitude' in df.columns:
                lat_abs = np.abs(df['latitude'])
            else:
                lat_abs = None
            
            if lat_abs is not None:
                # Seasonality peaks in temperate zones (30-60 degrees)
                df['seasonal_temp'] = np.where(
                    lat_abs < 30,
                    lat_abs / 30 * 10,  # Low in tropics
                    np.where(
                        lat_abs < 60,
                        10 + (lat_abs - 30) / 30 * 5,  # High in temperate
                        15 - (lat_abs - 60) / 30 * 10  # Lower toward poles
                    )
                )
        
        # Use existing precipitation if available
        if 'avg_precip' not in df.columns and 'precipitation' in df.columns:
            df['avg_precip'] = df['precipitation']
        
        # Create short variable names for formulas
        if 'total_richness' in df.columns and 'F' not in df.columns:
            df['F'] = df['total_richness']
        if 'plants_richness' in df.columns and 'P' not in df.columns:
            df['P'] = df['plants_richness']
        if 'animals_richness' in df.columns and 'A' not in df.columns:
            df['A'] = df['animals_richness']
        
        self.data = df
        logger.info(f"Prepared data with {len(df)} samples")
        logger.info(f"Climate variables: avg_temp, seasonal_temp, avg_precip")
        
        return df
    
    def run_nested_models(self, formulas: List[str]) -> Dict[str, Any]:
        """
        Run the nested model analysis.
        
        Default formulas for biodiversity analysis:
        1. F ~ avg_temp + avg_precip + seasonal_temp
        2. F ~ avg_temp + avg_precip + seasonal_temp + P + A
        3. F ~ avg_temp + avg_precip + seasonal_temp + P + A + P:seasonal_temp
        """
        logger.info("Running nested model analysis")
        
        # Create nested models
        nested_models = NestedFormulaModels(formulas)
        
        # Transform data for each model
        datasets = nested_models.transform_all(self.data)
        
        # Fit models and collect results
        model_results = []
        
        for i, (X, y) in enumerate(datasets):
            logger.info(f"Fitting Model {i+1}: {formulas[i]}")
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit Ridge regression
            model = Ridge(alpha=1.0)
            model.fit(X_scaled, y)
            
            # Make predictions
            y_pred = model.predict(X_scaled)
            
            # Store results
            result = {
                'model_num': i + 1,
                'formula': formulas[i],
                'X': X,
                'X_scaled': X_scaled,
                'y_true': y.values,
                'y_pred': y_pred,
                'n_params': X.shape[1] + 1,  # features + intercept
                'model': model,
                'scaler': scaler,
                'feature_names': X.columns.tolist()
            }
            
            model_results.append(result)
            self.models[f'model_{i+1}'] = result
        
        # Compare models
        comparison = NestedModelComparison()
        comparisons = comparison.compare_sequential(model_results)
        
        # Create summary
        summary_df = comparison.summary_table()
        
        # Save results
        summary_df.to_csv(self.output_dir / 'model_comparison_summary.csv', index=False)
        
        self.results['nested_models'] = {
            'models': model_results,
            'comparisons': comparisons,
            'summary': summary_df
        }
        
        return self.results['nested_models']
    
    def run_permutation_importance(self, model_key: str = 'model_2') -> Dict[str, Any]:
        """
        Run permutation importance on specified model.
        
        Default is Model 2 to test if P and A have importance beyond climate.
        """
        logger.info(f"Running permutation importance on {model_key}")
        
        model_data = self.models[model_key]
        
        # Create wrapper for sklearn model
        class ModelWrapper:
            def __init__(self, model, scaler):
                self.model = model
                self.scaler = scaler
            
            def predict(self, X):
                X_scaled = self.scaler.transform(X)
                return self.model.predict(X_scaled)
        
        wrapped_model = ModelWrapper(model_data['model'], model_data['scaler'])
        
        # Run permutation importance
        perm_imp = PermutationImportance(n_repeats=10, random_state=42)
        importances = perm_imp.calculate(
            wrapped_model,
            model_data['X'],
            model_data['y_true']
        )
        
        # Get importance dataframe
        importance_df = perm_imp.get_feature_importance_df()
        
        # Save results
        importance_df.to_csv(
            self.output_dir / f'{model_key}_permutation_importance.csv',
            index=False
        )
        
        # Create plot
        fig, ax = perm_imp.plot_importance()
        fig.savefig(
            self.output_dir / f'{model_key}_permutation_importance.png',
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        
        self.results['permutation_importance'] = {
            'importances': importances,
            'importance_df': importance_df
        }
        
        return importances
    
    def run_interaction_analysis(self,
                                feature1: str = 'P',
                                feature2: str = 'seasonal_temp',
                                model_key: str = 'model_3') -> Dict[str, Any]:
        """
        Run 2D PDP analysis for interaction effects.
        
        This is key for testing the temperate mismatch hypothesis!
        """
        logger.info(f"Running interaction analysis: {feature1} × {feature2}")
        
        model_data = self.models[model_key]
        
        # Create wrapper
        class ModelWrapper:
            def __init__(self, model, scaler):
                self.model = model
                self.scaler = scaler
            
            def predict(self, X):
                if isinstance(X, pd.DataFrame):
                    X_scaled = self.scaler.transform(X)
                else:
                    X_scaled = self.scaler.transform(X)
                return self.model.predict(X_scaled)
        
        wrapped_model = ModelWrapper(model_data['model'], model_data['scaler'])
        
        # Create PDP analyzer
        pdp = PartialDependencePlot(wrapped_model)
        
        # Calculate 2D PDP
        pdp_2d = pdp.calculate_2d_pdp(
            model_data['X'],
            (feature1, feature2),
            grid_resolution=25
        )
        
        # Create plots
        # 1. Main 2D PDP
        fig1, ax1 = pdp.plot_2d_pdp(pdp_2d)
        fig1.savefig(
            self.output_dir / f'pdp_2d_{feature1}_{feature2}.png',
            dpi=300, bbox_inches='tight'
        )
        
        # 2. Interaction slices
        fig2, axes = pdp.plot_interaction_slices(pdp_2d, n_slices=5)
        fig2.savefig(
            self.output_dir / f'pdp_slices_{feature1}_{feature2}.png',
            dpi=300, bbox_inches='tight'
        )
        
        plt.close('all')
        
        self.results['interaction_analysis'] = {
            'pdp_2d': pdp_2d,
            'features': (feature1, feature2)
        }
        
        return pdp_2d
    
    def test_hypotheses(self) -> Dict[str, Any]:
        """
        Test the main hypotheses and provide interpretation.
        """
        results = {
            'hypothesis_1_non_proxy': {},
            'hypothesis_2_temperate_mismatch': {}
        }
        
        # Hypothesis 1: Non-proxy relationships
        if 'nested_models' in self.results:
            comparison_1_2 = self.results['nested_models']['comparisons'][0]
            results['hypothesis_1_non_proxy'] = {
                'supported': bool(comparison_1_2['significant']),  # Convert numpy bool
                'p_value': float(comparison_1_2['p_value']),  # Convert to float
                'r2_improvement': float(comparison_1_2['r2_improvement']),
                'interpretation': (
                    "Plant and animal richness significantly improve fungal richness prediction "
                    "beyond climate variables alone" if comparison_1_2['significant']
                    else "No significant improvement from adding biodiversity predictors"
                )
            }
        
        # Hypothesis 2: Temperate mismatch
        if len(self.results['nested_models']['comparisons']) > 1:
            comparison_2_3 = self.results['nested_models']['comparisons'][1]
            results['hypothesis_2_temperate_mismatch'] = {
                'supported': bool(comparison_2_3['significant']),
                'p_value': float(comparison_2_3['p_value']),
                'interpretation': (
                    "Significant interaction between plant richness and temperature seasonality "
                    "supports the temperate mismatch hypothesis" if comparison_2_3['significant']
                    else "No significant interaction effect found"
                )
            }
        
        # Save hypothesis test results
        import json
        with open(self.output_dir / 'hypothesis_tests.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Store in instance for report generation
        self.results['hypothesis_tests'] = results
        
        return results
    
    def generate_report(self) -> str:
        """Generate a text report of all analyses."""
        report = []
        report.append("=" * 60)
        report.append("BIODIVERSITY RESEARCH ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Model comparison
        if 'nested_models' in self.results:
            report.append("NESTED MODEL COMPARISON")
            report.append("-" * 30)
            report.append(str(self.results['nested_models']['summary']))
            report.append("")
        
        # Hypothesis tests
        if 'hypothesis_tests' in self.results:
            report.append("HYPOTHESIS TESTS")
            report.append("-" * 30)
            for hyp, result in self.results['hypothesis_tests'].items():
                report.append(f"\n{hyp}:")
                report.append(f"  Supported: {result.get('supported', 'N/A')}")
                report.append(f"  P-value: {result.get('p_value', 'N/A'):.4f}")
                report.append(f"  {result.get('interpretation', '')}")
        
        report_text = "\n".join(report)
        
        # Save report
        with open(self.output_dir / 'analysis_report.txt', 'w') as f:
            f.write(report_text)
        
        return report_text