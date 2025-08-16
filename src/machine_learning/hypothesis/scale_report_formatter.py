"""
Scale Report Formatter for publication-ready tables and summaries.

This module provides formatting utilities for hypothesis test results with scale transformations.
It doesn't duplicate test logic - it just formats results from the enhanced hypothesis tests.
"""

import pandas as pd
from typing import Dict, Any, List, Optional


class ScaleReportFormatter:
    """Format hypothesis test results with scale transformations for publication."""
    
    @staticmethod
    def create_coefficient_table(
        test_results: Dict[str, Dict[str, Any]],
        link: str = 'log',
        format: str = 'markdown'
    ) -> str:
        """
        Create a formatted table from hypothesis test results.
        
        Args:
            test_results: Dictionary of test results from WaldTest.test_single_coefficient_with_scale
            link: Link function used ('log' or 'identity')
            format: Output format ('markdown', 'latex', 'html')
            
        Returns:
            Formatted table string
        """
        rows = []
        
        for var_name, results in test_results.items():
            # Extract from the enhanced test results
            sesoi_result = results.get('sesoi', {})
            tost_result = results.get('tost', {})
            summary = results.get('summary', {})
            
            row = {
                'Variable': var_name,
                'Coefficient': f"{summary.get('coefficient', 0):.4f}",
                'SE': f"{summary.get('se', 0):.4f}",
                'p-value': f"{summary.get('p_value', 1):.4f}"
            }
            
            # Add scale-specific columns if log link
            if link == 'log' and 'response_scale_interpretation' in summary:
                # Parse the response scale interpretation
                interp = summary['response_scale_interpretation']
                if '%' in interp:
                    # Extract percent change
                    import re
                    match = re.search(r'([+-]?\d+\.?\d*)%', interp)
                    if match:
                        row['% Change'] = match.group(1) + '%'
                    
                    # Extract CI if present
                    ci_match = re.search(r'\[([-\d.]+)%, ([-\d.]+)%\]', interp)
                    if ci_match:
                        row['95% CI (%)'] = f"[{ci_match.group(1)}, {ci_match.group(2)}]"
            
            # Add practical significance
            row['Practical'] = 'Yes' if summary.get('exceeds_sesoi', False) else 'No'
            
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Format based on requested output
        if format == 'markdown':
            return ScaleReportFormatter._to_markdown(df, link)
        elif format == 'latex':
            return ScaleReportFormatter._to_latex(df, link)
        elif format == 'html':
            return df.to_html(index=False)
        else:
            return str(df)
    
    @staticmethod
    def _to_markdown(df: pd.DataFrame, link: str) -> str:
        """Convert DataFrame to markdown table."""
        lines = []
        
        # Header
        columns = df.columns.tolist()
        lines.append('| ' + ' | '.join(columns) + ' |')
        lines.append('|' + '|'.join(['---' for _ in columns]) + '|')
        
        # Data rows
        for _, row in df.iterrows():
            lines.append('| ' + ' | '.join(str(row[col]) for col in columns) + ' |')
        
        # Footer note
        if link == 'log':
            lines.append("")
            lines.append("Note: Practical significance threshold = 10.5% change on response scale")
        
        return '\n'.join(lines)
    
    @staticmethod
    def _to_latex(df: pd.DataFrame, link: str) -> str:
        """Convert DataFrame to LaTeX table."""
        lines = ["\\begin{table}[htbp]"]
        lines.append("\\centering")
        lines.append("\\caption{Hypothesis Test Results with Scale Transformations}")
        
        n_cols = len(df.columns)
        col_align = 'l' + 'c' * (n_cols - 1)
        lines.append(f"\\begin{{tabular}}{{{col_align}}}")
        lines.append("\\toprule")
        
        # Header
        headers = ' & '.join(df.columns) + ' \\\\'
        lines.append(headers)
        lines.append("\\midrule")
        
        # Data rows
        for _, row in df.iterrows():
            row_str = ' & '.join(str(val) for val in row.values) + ' \\\\'
            lines.append(row_str)
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        
        if link == 'log':
            lines.append("\\begin{tablenotes}")
            lines.append("\\small")
            lines.append("\\item Note: Coefficients on log scale. % Change shows response scale effect.")
            lines.append("\\item Practical significance: Effect > 10.5\\% change.")
            lines.append("\\end{tablenotes}")
        
        lines.append("\\end{table}")
        
        return '\n'.join(lines)
    
    @staticmethod
    def summarize_hypothesis_test(
        h_name: str,
        test_results: Dict[str, Any],
        link: str = 'log'
    ) -> str:
        """
        Create a text summary of hypothesis test results.
        
        Args:
            h_name: Hypothesis name (e.g., 'H1', 'H2')
            test_results: Results from enhanced hypothesis tests
            link: Link function used
            
        Returns:
            Formatted text summary
        """
        lines = []
        lines.append(f"=== {h_name} Test Results ===")
        
        # Check for key results
        if 'p_value' in test_results:
            lines.append(f"Statistical significance: p = {test_results['p_value']:.4f}")
        
        if 'response_scale_interpretation' in test_results and link == 'log':
            lines.append(f"Effect size: {test_results['response_scale_interpretation']}")
        
        if 'practical_significance' in test_results:
            lines.append(f"Practical significance: {test_results['practical_significance']}")
        
        if 'is_equivalent' in test_results:
            equiv = "Yes" if test_results['is_equivalent'] else "No"
            lines.append(f"Equivalent to null (TOST): {equiv}")
        
        # Overall conclusion
        if 'conclusion' in test_results:
            lines.append(f"\nConclusion: {test_results['conclusion']}")
        elif 'practical_significance' in test_results:
            lines.append(f"\nConclusion: {test_results['practical_significance']}")
        
        return '\n'.join(lines)
    
    @staticmethod
    def create_comparison_table(
        method1_results: Dict[str, Any],
        method2_results: Dict[str, Any],
        format: str = 'markdown'
    ) -> str:
        """
        Create a comparison table between two methods.
        
        Args:
            method1_results: Results from Method 1
            method2_results: Results from Method 2
            format: Output format
            
        Returns:
            Formatted comparison table
        """
        comparison_data = []
        
        # Method 1 summary
        comparison_data.append({
            'Method': 'Method 1 (Continuous)',
            'Hypothesis': 'H0/H1',
            'Test Type': 'Biodiversity as continuous',
            'Primary Test': 'LRT + Wald',
            'Scale Handling': 'Log → Response',
            'SESOI/TOST': 'Yes'
        })
        
        # Method 2 summary
        comparison_data.append({
            'Method': 'Method 2 (Categorical)',
            'Hypothesis': 'H2',
            'Test Type': '2x2 Factorial',
            'Primary Test': 'Emmeans + Contrast',
            'Scale Handling': 'Log → Response',
            'SESOI/TOST': 'Yes'
        })
        
        df = pd.DataFrame(comparison_data)
        
        if format == 'markdown':
            return df.to_markdown(index=False)
        elif format == 'latex':
            return df.to_latex(index=False)
        else:
            return str(df)