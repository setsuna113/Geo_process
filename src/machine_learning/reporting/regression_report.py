"""Comprehensive report generation for regression analysis.

Implements reporting structure from regression_analysis_decisions.md lines 760-765.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class RegressionReportGenerator:
    """Generate comprehensive reports for regression analysis results."""
    
    def __init__(self, results: Dict[str, Any], visualizations: Dict[str, str]):
        """Initialize report generator.
        
        Args:
            results: Complete results from hypothesis testing
            visualizations: Paths to visualization files
        """
        self.results = results
        self.visualizations = visualizations
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def generate_full_report(self, output_dir: str) -> None:
        """Generate complete report with all outputs.
        
        Implements publication structure from regression_analysis_decisions.md lines 760-765.
        
        Args:
            output_dir: Directory to save report files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate all report components
        self._generate_summary_table(output_path)
        self._generate_detailed_results(output_path)
        self._generate_interpretation_report(output_path)
        self._generate_supplementary_materials(output_path)
        self._generate_latex_tables(output_path)
        
        logger.info(f"Complete report generated in {output_path}")
    
    def _generate_summary_table(self, output_path: Path) -> None:
        """Generate Table 1: Summary of hypothesis tests.
        
        Args:
            output_path: Directory to save table
        """
        summary_data = []
        
        # H0 results
        if 'h0' in self.results.get('synthesis', {}):
            h0 = self.results['synthesis']['h0']
            summary_data.append({
                'Hypothesis': 'H0: Plants → Fungi',
                'Method 1 p-value': self._get_pvalue('method1', 'h0'),
                'Method 2 p-value': 'N/A',
                'Method 3 p-value': self._get_pvalue('method3', 'h0'),
                'Consensus': h0.get('consensus', 'Not tested'),
                'Supported': '✓' if h0.get('supported') else '✗'
            })
        
        # H1 results
        if 'h1' in self.results.get('synthesis', {}):
            h1 = self.results['synthesis']['h1']
            summary_data.append({
                'Hypothesis': 'H1: Vertebrates → Fungi',
                'Method 1 p-value': 'N/A',
                'Method 2 p-value': 'N/A',
                'Method 3 p-value': self._get_pvalue('method3', 'h1'),
                'Consensus': h1.get('consensus', 'Not tested'),
                'Supported': '✓' if h1.get('supported') else '✗'
            })
        
        # H2 results
        if 'h2' in self.results.get('synthesis', {}):
            h2 = self.results['synthesis']['h2']
            summary_data.append({
                'Hypothesis': 'H2: Temperate Mismatch',
                'Method 1 p-value': self._format_pvalue(
                    self.results.get('method1', {}).get('interaction_term', {}).get('p_value')
                ),
                'Method 2 p-value': self._format_pvalue(
                    self.results.get('method2', {}).get('h2_contrast', {}).get('p_value')
                ),
                'Method 3 p-value': self._format_pvalue(
                    self.results.get('method3', {}).get('h2', {}).get('p_value')
                ),
                'Consensus': h2.get('consensus', 'Not tested'),
                'Supported': '✓' if h2.get('interpretation', '').startswith('STRONG') or 
                            h2.get('interpretation', '').startswith('MODERATE') else '✗'
            })
        
        # Save as CSV
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path / 'table1_hypothesis_summary.csv', index=False)
        
        # Save as formatted text
        with open(output_path / 'table1_hypothesis_summary.txt', 'w') as f:
            f.write("Table 1: Summary of Hypothesis Tests\n")
            f.write("="*80 + "\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")
            f.write("Note: ✓ = Supported, ✗ = Not supported\n")
    
    def _generate_detailed_results(self, output_path: Path) -> None:
        """Generate detailed results for each method.
        
        Args:
            output_path: Directory to save results
        """
        with open(output_path / 'detailed_results.txt', 'w') as f:
            f.write("DETAILED REGRESSION ANALYSIS RESULTS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {self.timestamp}\n\n")
            
            # Method 1 Results
            if 'method1' in self.results:
                f.write("\nMETHOD 1: CONTINUOUS INTERACTION GLM\n")
                f.write("-"*40 + "\n")
                self._write_method1_details(f, self.results['method1'])
            
            # Method 2 Results
            if 'method2' in self.results:
                f.write("\nMETHOD 2: CATEGORICAL GLM WITH EMMEANS\n")
                f.write("-"*40 + "\n")
                self._write_method2_details(f, self.results['method2'])
            
            # Method 3 Results
            if 'method3' in self.results:
                f.write("\nMETHOD 3: DOUBLE MACHINE LEARNING\n")
                f.write("-"*40 + "\n")
                self._write_method3_details(f, self.results['method3'])
    
    def _generate_interpretation_report(self, output_path: Path) -> None:
        """Generate interpretation report with conclusions.
        
        Implements interpretation guidelines from regression_analysis_decisions.md lines 674-684.
        
        Args:
            output_path: Directory to save report
        """
        with open(output_path / 'interpretation_report.md', 'w') as f:
            f.write("# Biodiversity Regression Analysis - Interpretation Report\n\n")
            f.write(f"*Generated: {self.timestamp}*\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            synthesis = self.results.get('synthesis', {})
            
            if 'h2' in synthesis:
                h2_result = synthesis['h2']
                f.write(f"### H2 Temperate Mismatch Hypothesis\n\n")
                f.write(f"**Consensus**: {h2_result.get('consensus', 'Not tested')}\n\n")
                f.write(f"**Interpretation**: {h2_result.get('interpretation', '')}\n\n")
                
                # Method-specific support
                f.write("**Method Support**:\n")
                f.write(f"- Method 1 (Continuous): {'Supported' if h2_result.get('method1_supported') else 'Not supported'}\n")
                f.write(f"- Method 2 (Categorical): {'Supported' if h2_result.get('method2_supported') else 'Not supported'}\n")
                f.write(f"- Method 3 (DML): {'Supported' if h2_result.get('method3_supported') else 'Not supported'}\n\n")
            
            # Detailed Interpretations
            f.write("## Detailed Method Interpretations\n\n")
            
            # Method 1 Interpretation
            if 'method1' in self.results:
                f.write("### Method 1: Continuous Interaction\n\n")
                interaction = self.results['method1'].get('interaction_term', {})
                f.write(f"- Interaction coefficient: {interaction.get('coefficient', 0):.4f}\n")
                f.write(f"- P-value: {interaction.get('p_value', 1):.4f}\n")
                f.write(f"- Interpretation: {interaction.get('interpretation', 'No interpretation available')}\n\n")
            
            # Method 2 Interpretation
            if 'method2' in self.results:
                f.write("### Method 2: Categorical Analysis\n\n")
                h2_contrast = self.results['method2'].get('h2_contrast', {})
                f.write(f"- Contrast estimate: {h2_contrast.get('estimate', 0):.4f}\n")
                f.write(f"- P-value: {h2_contrast.get('p_value', 1):.4f}\n")
                f.write(f"- Interpretation: {h2_contrast.get('interpretation', 'No interpretation available')}\n\n")
                
                # Add emmeans table
                emmeans = self.results['method2'].get('marginal_means', {})
                if emmeans:
                    f.write("**Estimated Marginal Means**:\n\n")
                    f.write("| Group | Fungi Richness (95% CI) |\n")
                    f.write("|-------|------------------------|\n")
                    for key, val in emmeans.items():
                        group_name = key.replace('_', ' ').title()
                        mean = val.get('mean', 0)
                        se = val.get('se', 0)
                        f.write(f"| {group_name} | {mean:.2f} ({mean-1.96*se:.2f}, {mean+1.96*se:.2f}) |\n")
                    f.write("\n")
            
            # Method 3 Interpretation
            if 'method3' in self.results:
                f.write("### Method 3: Causal Inference (DML)\n\n")
                for hyp in ['h0', 'h1', 'h2']:
                    if hyp in self.results['method3']:
                        result = self.results['method3'][hyp]
                        f.write(f"**{hyp.upper()}**:\n")
                        f.write(f"- Causal effect: {result.get('causal_effect', 0):.4f}\n")
                        f.write(f"- 95% CI: [{result.get('ci_lower', 0):.4f}, {result.get('ci_upper', 0):.4f}]\n")
                        f.write(f"- P-value: {result.get('p_value', 1):.4f}\n")
                        f.write(f"- Interpretation: {result.get('interpretation', 'No interpretation available')}\n\n")
            
            # Visualizations
            f.write("## Visualizations\n\n")
            for viz_name, viz_path in self.visualizations.items():
                f.write(f"- [{viz_name.replace('_', ' ').title()}]({viz_path})\n")
            
            # Conclusions
            f.write("\n## Key Conclusions\n\n")
            if 'conclusions' in synthesis:
                for i, conclusion in enumerate(synthesis['conclusions'], 1):
                    f.write(f"{i}. {conclusion}\n")
    
    def _generate_supplementary_materials(self, output_path: Path) -> None:
        """Generate supplementary materials.
        
        Args:
            output_path: Directory to save materials
        """
        supp_dir = output_path / 'supplementary'
        supp_dir.mkdir(exist_ok=True)
        
        # Save full model coefficients
        if 'method1' in self.results:
            self._save_model_coefficients(
                supp_dir / 'method1_coefficients.csv',
                self.results['method1']
            )
        
        if 'method2' in self.results:
            self._save_model_coefficients(
                supp_dir / 'method2_coefficients.csv',
                self.results['method2']
            )
        
        # Save diagnostics if available
        for method in ['method1', 'method2']:
            if method in self.results:
                diagnostics = self.results[method].get('diagnostics', {})
                if diagnostics:
                    with open(supp_dir / f'{method}_diagnostics.json', 'w') as f:
                        json.dump(diagnostics, f, indent=2, default=str)
    
    def _generate_latex_tables(self, output_path: Path) -> None:
        """Generate LaTeX tables for publication.
        
        Args:
            output_path: Directory to save LaTeX files
        """
        latex_dir = output_path / 'latex_tables'
        latex_dir.mkdir(exist_ok=True)
        
        # Generate main results table
        with open(latex_dir / 'table_main_results.tex', 'w') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Summary of Hypothesis Tests Across Three Methods}\n")
            f.write("\\label{tab:main_results}\n")
            f.write("\\begin{tabular}{lccccc}\n")
            f.write("\\hline\n")
            f.write("Hypothesis & Method 1 & Method 2 & Method 3 & Consensus & Supported \\\\\n")
            f.write("\\hline\n")
            
            # Add results rows
            synthesis = self.results.get('synthesis', {})
            
            # H2 row (most important)
            if 'h2' in synthesis:
                h2 = synthesis['h2']
                f.write("H2: Temperate Mismatch & ")
                f.write(self._format_pvalue_latex(
                    self.results.get('method1', {}).get('interaction_term', {}).get('p_value')
                ) + " & ")
                f.write(self._format_pvalue_latex(
                    self.results.get('method2', {}).get('h2_contrast', {}).get('p_value')
                ) + " & ")
                f.write(self._format_pvalue_latex(
                    self.results.get('method3', {}).get('h2', {}).get('p_value')
                ) + " & ")
                consensus_count = sum([
                    h2.get('method1_supported', False),
                    h2.get('method2_supported', False),
                    h2.get('method3_supported', False)
                ])
                f.write(f"{consensus_count}/3 & ")
                f.write("\\checkmark" if consensus_count >= 2 else "$\\times$")
                f.write(" \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
    
    def _get_pvalue(self, method: str, hypothesis: str) -> str:
        """Get formatted p-value for a specific method and hypothesis.
        
        Args:
            method: Method name
            hypothesis: Hypothesis name
            
        Returns:
            Formatted p-value string
        """
        if method in self.results:
            if hypothesis in self.results[method]:
                p_val = self.results[method][hypothesis].get('p_value', 1)
                return self._format_pvalue(p_val)
        return 'N/A'
    
    def _format_pvalue(self, p_value: Optional[float]) -> str:
        """Format p-value for display.
        
        Args:
            p_value: P-value to format
            
        Returns:
            Formatted string
        """
        if p_value is None:
            return 'N/A'
        if p_value < 0.001:
            return '<0.001'
        elif p_value < 0.01:
            return f'{p_value:.3f}'
        elif p_value < 0.05:
            return f'{p_value:.3f}*'
        else:
            return f'{p_value:.3f}'
    
    def _format_pvalue_latex(self, p_value: Optional[float]) -> str:
        """Format p-value for LaTeX.
        
        Args:
            p_value: P-value to format
            
        Returns:
            LaTeX formatted string
        """
        if p_value is None:
            return '--'
        if p_value < 0.001:
            return '$<$0.001***'
        elif p_value < 0.01:
            return f'{p_value:.3f}**'
        elif p_value < 0.05:
            return f'{p_value:.3f}*'
        else:
            return f'{p_value:.3f}'
    
    def _write_method1_details(self, f, results: Dict[str, Any]) -> None:
        """Write Method 1 detailed results.
        
        Args:
            f: File handle
            results: Method 1 results
        """
        f.write("\nModel Comparison:\n")
        models = results.get('models', {})
        for model_name in ['null', 'full', 'interaction']:
            if model_name in models:
                model = models[model_name]
                f.write(f"  {model_name.capitalize()} Model:\n")
                f.write(f"    AIC: {model.get('aic', 'N/A'):.2f}\n")
                f.write(f"    BIC: {model.get('bic', 'N/A'):.2f}\n")
                f.write(f"    Log-likelihood: {model.get('llf', 'N/A'):.2f}\n")
        
        f.write("\nInteraction Term:\n")
        interaction = results.get('interaction_term', {})
        f.write(f"  Coefficient: {interaction.get('coefficient', 0):.4f}\n")
        f.write(f"  Std Error: {interaction.get('std_error', 0):.4f}\n")
        f.write(f"  P-value: {interaction.get('p_value', 1):.4f}\n")
        f.write(f"  Interpretation: {interaction.get('interpretation', 'N/A')}\n")
    
    def _write_method2_details(self, f, results: Dict[str, Any]) -> None:
        """Write Method 2 detailed results.
        
        Args:
            f: File handle
            results: Method 2 results
        """
        f.write("\nMarginal Means:\n")
        emmeans = results.get('marginal_means', {})
        for group, stats in emmeans.items():
            f.write(f"  {group}:\n")
            f.write(f"    Mean: {stats.get('mean', 0):.3f}\n")
            f.write(f"    SE: {stats.get('se', 0):.3f}\n")
            f.write(f"    95% CI: [{stats.get('mean', 0) - 1.96*stats.get('se', 0):.3f}, "
                   f"{stats.get('mean', 0) + 1.96*stats.get('se', 0):.3f}]\n")
        
        f.write("\nH2 Contrast Test:\n")
        h2_contrast = results.get('h2_contrast', {})
        f.write(f"  Estimate: {h2_contrast.get('estimate', 0):.4f}\n")
        f.write(f"  SE: {h2_contrast.get('se', 0):.4f}\n")
        f.write(f"  P-value: {h2_contrast.get('p_value', 1):.4f}\n")
        f.write(f"  Interpretation: {h2_contrast.get('interpretation', 'N/A')}\n")
    
    def _write_method3_details(self, f, results: Dict[str, Any]) -> None:
        """Write Method 3 detailed results.
        
        Args:
            f: File handle
            results: Method 3 results
        """
        for hypothesis in ['h0', 'h1', 'h2']:
            if hypothesis in results:
                hyp_results = results[hypothesis]
                f.write(f"\n{hypothesis.upper()} Results:\n")
                f.write(f"  Causal Effect: {hyp_results.get('causal_effect', 0):.4f}\n")
                f.write(f"  95% CI: [{hyp_results.get('ci_lower', 0):.4f}, "
                       f"{hyp_results.get('ci_upper', 0):.4f}]\n")
                f.write(f"  P-value: {hyp_results.get('p_value', 1):.4f}\n")
                f.write(f"  Interpretation: {hyp_results.get('interpretation', 'N/A')}\n")
    
    def _save_model_coefficients(self, path: Path, results: Dict[str, Any]) -> None:
        """Save model coefficients to CSV.
        
        Args:
            path: Path to save CSV
            results: Method results containing coefficients
        """
        # This would need to extract coefficients from the model objects
        # For now, save what we have
        coeffs = []
        
        if 'interaction_term' in results:
            interaction = results['interaction_term']
            coeffs.append({
                'Variable': 'plants_bio4_interaction',
                'Coefficient': interaction.get('coefficient', 0),
                'Std_Error': interaction.get('std_error', 0),
                'P_Value': interaction.get('p_value', 1)
            })
        
        if coeffs:
            df = pd.DataFrame(coeffs)
            df.to_csv(path, index=False)