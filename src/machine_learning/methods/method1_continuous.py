from src.abstractions import get_logger
"""Method 1: H0-H1 Testing with GLM (Julia-based).

Tests:
- H0: Response richness is independent of other biodiversity predictors
- H1: Biodiversity begets biodiversity (positive correlation exists)

Compares Model 1 (climate only) vs Model 2 (climate + biodiversity).
Accepts results from Julia GLM via orchestrator.
Response variable is configurable (fungi, plants, or vertebrates).
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.stats import chi2
from typing import Dict, Any, Optional, Union, Tuple, List
import json
from src.abstractions import save_json
import warnings
warnings.filterwarnings('ignore')

# Import hypothesis test modules and visualization wrappers
from .wald_method import WaldMethod
from ..evaluation.model_comparison import ModelComparison
from ..config_loader import MLRegressionConfig

logger = get_logger(__name__)

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


class Method1H0H1GLM:
    """Method 1: Tests H0 vs H1 using nested GLM models.
    
    Model 1 (H0): Climate + spatial only
    Model 2 (H1): Climate + spatial + biodiversity
    
    This class:
    1. Accepts fitted model results from Julia (via orchestrator)
    2. Runs hypothesis tests (LRT, Wald-SESOI, Wald-TOST)
    3. Generates visualizations and reports
    """
    
    def __init__(self, config: MLRegressionConfig) -> None:
        """Initialize Method 1.
        
        Args:
            config: ML regression configuration
        """
        self.config = config
        
        # Storage for model results from Julia
        self.model_null = None  # Model 1: Climate only
        self.model_full = None  # Model 2: Climate + biodiversity
        self.lr_test_result = None  # LRT result from Julia
        self.data_metadata = None  # Metadata about the data
        self.response_type = 'fungi'  # Default response type
        
        # Storage for hypothesis test results
        self.hypothesis_results = {}
        
        # Initialize wrapped components
        self.wald_method = WaldMethod()  # Uses default SESOI and TOST thresholds
        self.model_comparator = ModelComparison()
        
        logger.info("Method 1 H0-H1 GLM initialized with wrapped components")
    
    def set_julia_results(self, models: Dict[str, Any], lr_test: Dict[str, Any], 
                         response_type: str = 'fungi') -> None:
        """Set the Julia model results for analysis.
        
        Called by orchestrator to pass Julia results to this method.
        
        Args:
            models: Dictionary containing 'method1_null' and 'method1_full' model results
            lr_test: MANDATORY LRT result from Julia (THIS IS THE PRIMARY TEST!)
            response_type: Type of response variable ('fungi', 'plants', or 'vertebrates')
        """
        if not lr_test:
            raise ValueError("LRT result is MANDATORY. This is THE test for nested models!")
        
        self.model_null = models.get('method1_null')
        self.model_full = models.get('method1_full')
        self.lr_test_result = lr_test
        self.response_type = response_type
        
        # Extract metadata
        if self.model_full:
            self.data_metadata = {
                'n_samples': self.model_full.get('n_obs', 0),
                'n_features': len(self.model_full.get('coef_names', [])),
                'has_mesf': any('ME' in name for name in self.model_full.get('coef_names', []))
            }
        
        null_aic = self.model_null.get('aic', 'N/A') if self.model_null else 'N/A'
        full_aic = self.model_full.get('aic', 'N/A') if self.model_full else 'N/A'
        
        if isinstance(null_aic, (int, float)) and isinstance(full_aic, (int, float)):
            logger.info(
                f"Julia results set: null model AIC={null_aic:.2f}, "
                f"full model AIC={full_aic:.2f}")
        else:
            logger.info(f"Julia results set: null model AIC={null_aic}, full model AIC={full_aic}")
    
    def run_hypothesis_tests(self) -> Dict[str, Any]:
        """Run all hypothesis tests on the loaded models.
        
        Returns:
            Dictionary with all test results
        """
        if not self.model_null or not self.model_full:
            raise ValueError("Models not loaded. Call set_julia_results() first.")
        
        logger.info("\n" + "="*70)
        logger.info("METHOD 1: HYPOTHESIS TESTING")
        logger.info("="*70)
        
        results = {}
        
        # 1. Likelihood Ratio Test (MANDATORY - this is THE test!)
        # Fix LR statistic if it's actually the full model deviance
        lrt_result = self.lr_test_result.copy()
        
        # Check if lr_statistic is incorrectly the full model deviance
        if (self.model_null and self.model_full and 
            abs(lrt_result['lr_statistic'] - self.model_full.get('deviance', 0)) < 0.01):
            # Calculate the correct LR statistic as deviance difference
            null_dev = self.model_null.get('deviance', 0) 
            full_dev = self.model_full.get('deviance', 0)
            lrt_result['lr_statistic'] = null_dev - full_dev
            logger.warning(
                f"Fixed LR statistic: {lrt_result['lr_statistic']:.2f} "
                f"(was incorrectly full deviance)")
        
        results['likelihood_ratio_test'] = lrt_result
        logger.info(f"LRT (PRIMARY TEST): χ²={lrt_result.get('lr_statistic', 0):.2f}, "
                   f"p={lrt_result.get('p_value', 0):.4e}")
        
        # Score Test removed - redundant with LRT for nested models
        
        # 3. Wald Tests with SESOI and TOST
        wald_results = self._run_wald_tests()
        results['wald_tests'] = wald_results
        
        # 4. Model Comparison
        comparison = self._compare_models()
        results['model_comparison'] = comparison
        
        # 5. Overall Conclusion
        results['conclusion'] = self._determine_conclusion(results)
        
        # Store results for reporting
        self.hypothesis_results = results
        
        return results
    
    def _run_wald_tests(self) -> Dict[str, Any]:
        """Run Wald tests with SESOI and TOST for biodiversity coefficients.
        
        Returns:
            Dictionary with Wald test results
        """
        # Extract coefficient information
        coefs = np.array(self.model_full.get('coefficients', []))
        ses = np.array(self.model_full.get('std_errors', []))
        names = self.model_full.get('coef_names', [])
        vcov = np.array(self.model_full.get('vcov', []))
        
        # Use WaldMethod wrapper to run tests
        wald_results = self.wald_method.run_all_tests(
            coefficients=coefs,
            std_errors=ses,
            coef_names=names,
            vcov_matrix=vcov if vcov.size > 0 else None
        )
        
        return wald_results
    
    
    def _compare_models(self) -> Dict[str, Any]:
        """Compare null and full models using information criteria.
        
        Returns:
            Model comparison results from ModelComparison class
        """
        # Use the external ModelComparison class - no duplication!
        models = {
            'null': self.model_null,
            'full': self.model_full
        }
        
        comparison = self.model_comparator.compare_models(models)
        
        # Log the results using the correct field names
        best_aic = comparison.get('best_model', {}).get('by_aic', 'unknown')
        best_bic = comparison.get('best_model', {}).get('by_bic', 'unknown')
        
        logger.info(f"\nModel Comparison:")
        logger.info(f"  Best model (AIC): {best_aic}")
        logger.info(f"  Best model (BIC): {best_bic}")
        
        return comparison
    
    def _determine_conclusion(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Determine overall conclusion based on all tests.
        
        Args:
            results: All test results
            
        Returns:
            Conclusion dictionary
        """
        # Check LRT (p < 0.05 means reject H0, support H1)
        lrt_p = results.get('likelihood_ratio_test', {}).get('p_value', 1.0)
        lrt_supports_h1 = lrt_p < 0.05
        
        
        # Check Wald tests
        wald_supports_h1 = False
        if 'wald_tests' in results:
            # Check if summary indicates meaningful effects
            summary = results['wald_tests'].get('summary', {})
            if summary.get('n_meaningful', 0) > 0:
                wald_supports_h1 = True
            # Also check joint test
            elif 'joint_test' in results['wald_tests']:
                joint_p = results['wald_tests']['joint_test'].get('p_value', 1.0)
                wald_supports_h1 = joint_p < 0.05
        
        # Check model comparison using correct field structure
        comparison = results.get('model_comparison', {})
        best_model = comparison.get('best_model', {})
        aic_supports_h1 = best_model.get('by_aic') == 'full'
        bic_supports_h1 = best_model.get('by_bic') == 'full'
        
        # Overall decision (majority vote with emphasis on LRT and Wald)
        evidence_count = sum([lrt_supports_h1, wald_supports_h1, aic_supports_h1])
        supports_h1 = evidence_count >= 2  # At least 2 out of 3 support H1
        
        # Create hypothesis text based on response type
        response_name = self.response_type.capitalize()
        h1_text = f'H1: Biodiversity affects {response_name} richness'
        h0_text = f'H0: No biodiversity effect on {response_name}'
        
        conclusion = {
            'supports_h1': supports_h1,
            'hypothesis': h1_text if supports_h1 else h0_text,
            'evidence': {
                'lrt': lrt_supports_h1,
                'wald_sesoi': wald_supports_h1,
                'aic': aic_supports_h1,
                'bic': bic_supports_h1
            },
            'interpretation': self._get_interpretation(supports_h1, evidence_count)
        }
        
        logger.info(f"\n" + "="*40)
        logger.info(f"CONCLUSION: {conclusion['hypothesis']}")
        logger.info(f"Interpretation: {conclusion['interpretation']}")
        logger.info("="*40)
        
        return conclusion
    
    def _get_interpretation(self, supports_h1: bool, evidence_count: int) -> str:
        """Get detailed interpretation of results.
        
        Args:
            supports_h1: Whether H1 is supported
            evidence_count: Number of tests supporting H1
            
        Returns:
            Interpretation string
        """
        if supports_h1:
            if evidence_count >= 4:
                return "Strong evidence that biodiversity positively influences fungal richness"
            elif evidence_count == 3:
                return "Good evidence that biodiversity influences fungal richness"
            else:
                return "Moderate evidence for biodiversity effects on fungal richness"
        else:
            if evidence_count == 0:
                return "No evidence for biodiversity effects on fungal richness"
            else:
                return "Insufficient evidence to conclude biodiversity affects fungal richness"
    
    def generate_visualizations(self, output_dir: Optional[Path] = None) -> List[Path]:
        """Generate all visualizations for Method 1.
        
        Includes both hypothesis test visualizations AND model diagnostics.
        
        Args:
            output_dir: Directory to save visualizations
            
        Returns:
            List of paths to generated visualizations
        """
        if not self.hypothesis_results:
            logger.warning("No results to visualize. Run hypothesis tests first.")
            return []
        
        if output_dir is None:
            output_dir = Path("results/method1_analysis")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        # ========= HYPOTHESIS TEST VISUALIZATIONS =========
        # Call all visualization functions - let them handle their own errors
        
        # 1. Forest Plots - Now creates TWO files: sesoi and tost
        forest_paths = self._create_forest_plots(output_dir)
        generated_files.extend(forest_paths)
        
        # 2. Model Comparison - AIC/BIC comparison (ESSENTIAL) 
        comparison_path = self._create_model_comparison_viz(output_dir)
        if comparison_path:
            generated_files.append(comparison_path)
        
        # 3. LRT Visualization
        lrt_path = self._create_lrt_viz(output_dir)
        if lrt_path:
            generated_files.append(lrt_path)
        
        # Score Test removed - redundant with LRT
        
        # ========= MODEL DIAGNOSTIC PLOTS (NEW) =========
        # Add comprehensive model diagnostics
        from ..diagnostics.gvif import plot_gvif_only
        
        # Only generate diagnostics for full model (not null model)
        if self.model_full:
            # Create simplified GVIF-only diagnostic plot
            path = plot_gvif_only(
                model_results=self.model_full,
                output_dir=output_dir,
                filename="method1_gvif_diagnostics.png"
            )
            if path:
                generated_files.append(path)
        
        # Coefficient comparison plot removed - deprecated function
        
        logger.info(f"Generated {len(generated_files)} visualizations in {output_dir}")
        return generated_files
    
    def _create_forest_plots(self, output_dir: Path) -> List[Path]:
        """Create separate forest plots for TOST and SESOI.
        
        Now creates TWO separate files via the updated create_forest_plot method.
        
        Args:
            output_dir: Directory to save plots
            
        Returns:
            List of paths to saved plots
        """
        if 'wald_tests' not in self.hypothesis_results:
            return []
        
        result = self.wald_method.create_forest_plot(
            results=self.hypothesis_results['wald_tests'],
            output_dir=output_dir,
            filename="forest_plot_improved.png"  # Ignored - will create sesoi and tost files
        )
        
        # The method now returns a list of paths
        if isinstance(result, list):
            return result
        else:
            return []
    
    def _create_model_comparison_viz(self, output_dir: Path) -> Optional[Path]:
        """Create model comparison visualization using ModelComparison wrapper.
        
        Args:
            output_dir: Directory to save plot
            
        Returns:
            Path to saved plot or None
        """
        if not self.model_null or not self.model_full:
            return None
        
        models = {
            'null': self.model_null,
            'full': self.model_full
        }
        
        # Prepare test results for summary
        test_results = {}
        if 'likelihood_ratio_test' in self.hypothesis_results:
            test_results['lrt'] = self.hypothesis_results['likelihood_ratio_test']
        
        # Use the visualization module directly
        try:
            from ..visualization.model_comparison_plots import plot_model_comparison
            import matplotlib.pyplot as plt
            
            # Use the full model comparison results if available
            if 'model_comparison' in self.hypothesis_results:
                comparison_results = self.hypothesis_results['model_comparison']
            else:
                # Fallback to manual construction
                comparison_results = {
                    'models': {
                        'null': {
                            'aic': self.model_null.get('aic'),
                            'bic': self.model_null.get('bic'),
                            'loglik': self.model_null.get('loglik'),
                            'df': self.model_null.get('df')
                        },
                        'full': {
                            'aic': self.model_full.get('aic'),
                            'bic': self.model_full.get('bic'),
                            'loglik': self.model_full.get('loglik'),
                            'df': self.model_full.get('df')
                        }
                },
                'best_model': {
                    'aic': ('full' if self.model_full.get('aic', 0) < 
                            self.model_null.get('aic', 0) else 'null'),
                    'bic': ('full' if self.model_full.get('bic', 0) < 
                            self.model_null.get('bic', 0) else 'null')
                }
            }
            
            fig = plot_model_comparison(
                comparison_results=comparison_results,
                title="Method 1: Model Comparison (H0 vs H1)"
            )
            
            save_path = output_dir / "model_comparison.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Model comparison plot saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to create model comparison plot: {e}")
            return None
    
    def _create_lrt_viz(self, output_dir: Path) -> Optional[Path]:
        """Create LRT visualization.
        
        Shows chi-square distribution and deviance table.
        Keep for now - decide later if truly redundant.
        """
        if 'likelihood_ratio_test' not in self.hypothesis_results:
            return None
        
        try:
            from ..visualization.lrt_plots import visualize_lrt_results
            
            fig = visualize_lrt_results(
                lrt_result=self.hypothesis_results['likelihood_ratio_test'],
                model_names=["Climate Only (H0)", "Climate + Biodiversity (H1)"]
            )
            
            save_path = output_dir / "lrt_test.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"LRT visualization saved to {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Failed to create LRT visualization: {e}")
            return None
    
    
    def generate_report(self, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive report for Method 1.
        
        Args:
            output_dir: Directory to save report files
            
        Returns:
            Dictionary with report content
        """
        if not self.hypothesis_results:
            logger.warning("No results to report. Run hypothesis tests first.")
            return {}
        
        if output_dir is None:
            output_dir = Path("results/method1_analysis")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate visualizations
        viz_files = self.generate_visualizations(output_dir)
        
        # Create report structure
        report = {
            'method': 'Method 1: H0-H1 Hypothesis Testing',
            'timestamp': datetime.now().isoformat(),
            'hypotheses': {
                'H0': 'Fungi richness is independent of plant/vertebrate richness',
                'H1': 'Biodiversity begets biodiversity (positive correlation)'
            },
            'data_summary': {
                'n_observations': self.model_full.get('n_obs', 0),
                'n_features_null': self.model_null.get('n_parameters', 0),
                'n_features_full': self.model_full.get('n_parameters', 0),
                'has_spatial_filtering': self.data_metadata.get('has_mesf', False)
            },
            'test_results': self.hypothesis_results,
            'visualizations': [str(f) for f in viz_files],
            'conclusion': self.hypothesis_results.get('conclusion', {})
        }
        
        # Save structured results for programmatic access
        json_path = output_dir / "method1_results.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=self._json_serialize)
        logger.info(f"Structured results saved to {json_path}")
        
        # Create text summary (simple, useful)
        text_path = self._create_text_summary(report, output_dir)
        if text_path:
            report['text_summary'] = str(text_path)
        
        logger.info(f"Method 1 report generation complete. Files saved to {output_dir}")
        return report
    
    def _create_text_summary(self, report: Dict[str, Any], output_dir: Path) -> Optional[Path]:
        """Create professional reports using the reporting module.
        
        No more custom text formatting garbage.
        Uses the professional reporting functions instead.
        
        Args:
            report: Report data
            output_dir: Directory to save summary
            
        Returns:
            Path to combined report file or None
        """
        from ..reporting import (
            generate_wald_publication_report,
            generate_lrt_publication_report
        )
        
        try:
            reports = []
            methods_texts = []  # Collect methods descriptions
            latex_tables = []
            supplementary_tables = []  # For CSV export
            supplementary_info = {}  # Assumptions and guidelines
            
            # Generate professional reports for each test
            
            # 1. LRT Report
            if 'likelihood_ratio_test' in report['test_results']:
                lrt = report['test_results']['likelihood_ratio_test']
                
                # Generate professional report
                lrt_report = generate_lrt_publication_report(
                    test_results=lrt,
                    hypothesis="H0: Climate-only model vs H1: Climate + Biodiversity model"
                )
                
                # Extract all valuable content
                reports.append(lrt_report.get('main_text', ''))
                
                # Collect methods text
                if 'methods_text' in lrt_report:
                    methods_texts.append(("LRT", lrt_report['methods_text']))
                
                # Save comparison details for supplementary
                if 'comparison_details' in lrt_report:
                    supplementary_tables.append(
                        ("lrt_comparison", lrt_report['comparison_details']))
                
                if 'latex_table' in lrt_report:
                    latex_tables.append(("LRT", lrt_report['latex_table']))
            
            # 3. Wald Tests with SESOI and TOST
            if 'wald_tests' in report['test_results']:
                for test_name, wald_result in report['test_results']['wald_tests'].items():
                    # Determine SESOI and TOST parameters
                    sesoi_threshold = None
                    tost_bounds = None
                    
                    if 'sesoi' in test_name.lower():
                        sesoi_threshold = wald_result.get('sesoi_threshold', 0.1)
                    elif 'tost' in test_name.lower():
                        tost_bounds = wald_result.get('tost_bounds', (-0.1, 0.1))
                    
                    # Generate professional report
                    wald_report = generate_wald_publication_report(
                        test_results=wald_result,
                        test_name=test_name,
                        hypothesis="Biodiversity effects",
                        sesoi_threshold=sesoi_threshold,
                        tost_bounds=tost_bounds
                    )
                    
                    reports.append(wald_report.get('main_text', ''))
                    
                    # Collect methods text
                    if 'methods_text' in wald_report:
                        methods_texts.append((f"Wald {test_name}", wald_report['methods_text']))
                    
                    # Collect supplementary info (assumptions, guidelines)
                    if 'supplementary_info' in wald_report:
                        supplementary_info[f"wald_{test_name}"] = wald_report['supplementary_info']
                    
                    # Save results table for supplementary
                    if 'results_table' in wald_report and wald_report['results_table'] is not None:
                        supplementary_tables.append((f"wald_{test_name}", wald_report['results_table']))
                    
                    if 'latex_table' in wald_report:
                        latex_tables.append((test_name, wald_report['latex_table']))
            
            # Combine all reports (professional reports already contain formatting)
            full_report = "\n\n".join(reports)
            
            # METHODS SECTION
            if methods_texts:
                full_report += "\nMETHODS\n"
                full_report += "=" * 70 + "\n"
                # Deduplicate methods texts (some might be identical)
                seen_methods = set()
                for test_name, methods_text in methods_texts:
                    if methods_text not in seen_methods:
                        full_report += f"\n{test_name}:\n"
                        full_report += methods_text + "\n"
                        seen_methods.add(methods_text)
                full_report += "\n"
            
            # CONCLUSION - This is the key summary that synthesizes all tests
            if 'conclusion' in report:
                full_report += "\n\nCONCLUSION\n"
                full_report += "=" * 70 + "\n"
                conclusion = report['conclusion']
                
                # Final decision
                if 'hypothesis' in conclusion:
                    full_report += f"Decision: {conclusion['hypothesis']}\n"
                if 'interpretation' in conclusion:
                    full_report += f"Interpretation: {conclusion['interpretation']}\n"
                
                # Evidence summary from all tests
                if 'evidence' in conclusion:
                    full_report += "\nEvidence Summary:\n"
                    for test_name, supports_h1 in conclusion['evidence'].items():
                        full_report += f"  {test_name.upper()}: {'Supports H1' if supports_h1 else 'Supports H0'}\n"
            
            # Save combined text report
            text_path = output_dir / "method1_professional_report.txt"
            with open(text_path, 'w') as f:
                f.write(full_report)
            
            # Save LaTeX tables separately
            if latex_tables:
                latex_path = output_dir / "method1_latex_tables.tex"
                with open(latex_path, 'w') as f:
                    f.write("% LaTeX Tables for Method 1 Results\n\n")
                    for name, table in latex_tables:
                        f.write(f"% {name}\n")
                        f.write(table)
                        f.write("\n\n")
                logger.info(f"LaTeX tables saved to {latex_path}")
            
            # Save supplementary tables as CSV (NEW!)
            if supplementary_tables:
                supp_dir = output_dir / "supplementary"
                supp_dir.mkdir(exist_ok=True)
                
                for table_name, df in supplementary_tables:
                    if df is not None:
                        csv_path = supp_dir / f"{table_name}.csv"
                        df.to_csv(csv_path, index=False)
                        logger.info(f"Supplementary table saved to {csv_path}")
            
            # Save supplementary info as JSON (NEW!)
            if supplementary_info:
                supp_json_path = output_dir / "supplementary" / "assumptions_guidelines.json"
                supp_json_path.parent.mkdir(exist_ok=True)
                with open(supp_json_path, 'w') as f:
                    json.dump(supplementary_info, f, indent=2)
                logger.info(f"Supplementary info saved to {supp_json_path}")
            
            logger.info(f"Professional report saved to {text_path}")
            return text_path
            
        except Exception as e:
            logger.error(f"Failed to create professional report: {e}")
            return None
    
    def _json_serialize(self, obj):
        """Custom JSON serializer for numpy types."""
        import pandas as pd
        
        # Handle pandas DataFrames
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        # Handle numpy types
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle infinity and NaN
        elif isinstance(obj, float):
            if np.isinf(obj):
                return "Infinity" if obj > 0 else "-Infinity"
            elif np.isnan(obj):
                return "NaN"
            return obj
        return obj
    
    def get_biodiversity_coefficients(self) -> Dict[str, Any]:
        """Get biodiversity coefficients for reporting.
        
        Returns:
            Dictionary with coefficient information
        """
        if not self.hypothesis_results or 'wald_tests' not in self.hypothesis_results:
            return {}
        
        coefficients = {}
        for name, result in self.hypothesis_results['wald_tests'].items():
            if 'joint' not in name and 'coefficient' in result:
                coefficients[name] = {
                    'coefficient': result['coefficient'],
                    'std_error': result['standard_error'],
                    'p_value': result.get('sesoi_test', {}).get('p_value', np.nan)
                }
        
        return coefficients