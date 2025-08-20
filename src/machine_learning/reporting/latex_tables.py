"""LaTeX table generation for hypothesis testing results.

Extracted from various hypothesis test files.
Consolidates all LaTeX generation in one place.
"""

import pandas as pd
from typing import Dict, Any, Optional, List


def generate_wald_latex_table(df: pd.DataFrame, test_name: str = "Wald Test") -> str:
    """Generate LaTeX table code for Wald test results.
    
    Extracted from wald_test.py lines 699-739.
    
    Args:
        df: DataFrame with results
        test_name: Name of test for caption
        
    Returns:
        LaTeX table code
    """
    n_cols = len(df.columns)
    col_align = 'l' + 'c' * (n_cols - 1)
    
    latex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{test_name} results for regression coefficients}}
\\label{{tab:wald_test}}
\\begin{{tabular}}{{{col_align}}}
\\toprule
"""
    
    # Add headers
    headers = ' & '.join(df.columns) + ' \\\\'
    latex += headers + '\n\\midrule\n'
    
    # Add data rows
    for _, row in df.iterrows():
        row_str = ' & '.join(str(val) for val in row.values) + ' \\\\'
        latex += row_str + '\n'
    
    # Add footer
    latex += """\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\\small
\\item Note: SE = Standard Error. Wald χ² = (β/SE)².
\\item Significance levels: *** p < 0.001, ** p < 0.01, * p < 0.05.
\\end{tablenotes}
\\end{table}"""
    
    return latex


def generate_contrast_latex_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table code for contrast analysis.
    
    Extracted from contrast_analysis.py lines 778-813.
    
    Args:
        df: DataFrame with contrast results
        
    Returns:
        LaTeX table code suitable for ecology journals
    """
    latex = r"""\begin{table}[htbp]
\centering
\caption{Contrast Analysis Results for H2 Hypothesis}
\label{tab:contrast_analysis}
\begin{tabular}{lcccccc}
\toprule
"""
    
    # Add headers
    headers = ' & '.join(df.columns) + ' \\\\'
    latex += headers + '\n\\midrule\n'
    
    # Add data rows
    for _, row in df.iterrows():
        row_str = ' & '.join(str(val) for val in row.values) + ' \\\\'
        latex += row_str + '\n'
    
    # Add footer with notes
    latex += r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: Contrast tests the difference between high-plant temperate and high-plant non-temperate regions.
\item Standard errors computed using the delta method. Significance assessed at $\alpha = 0.05$.
\end{tablenotes}
\end{table}"""
    
    return latex


def generate_lrt_latex_table(
    lrt_result: Dict[str, Any] = None,
    null_deviance: Optional[float] = None,
    full_deviance: Optional[float] = None,
    lr_statistic: Optional[float] = None,
    df: Optional[int] = None,
    p_value: Optional[float] = None
) -> str:
    """Generate LaTeX table for LRT results.
    
    Accepts EITHER a dictionary (from Julia/Method1) OR individual parameters.
    
    Args:
        lrt_result: Dictionary with LRT results (preferred - from Julia)
        null_deviance: Null model deviance (if not using dict)
        full_deviance: Full model deviance (if not using dict)
        lr_statistic: LR test statistic (if not using dict)
        df: Degrees of freedom (if not using dict)
        p_value: P-value (if not using dict)
        
    Returns:
        LaTeX formatted table
    """
    # Extract from dictionary if provided
    if lrt_result:
        null_deviance = lrt_result.get('null_deviance', null_deviance)
        full_deviance = lrt_result.get('full_deviance', full_deviance)
        lr_statistic = lrt_result.get('lr_statistic', lr_statistic)
        df = lrt_result.get('df', df)
        p_value = lrt_result.get('p_value', p_value)
    # Format p-value
    if p_value < 0.001:
        p_str = "< 0.001"
    else:
        p_str = f"{p_value:.4f}"
    
    latex = r"""\begin{table}[htbp]
\centering
\caption{Likelihood Ratio Test Results}
\label{tab:lrt}
\begin{tabular}{lc}
\toprule
\textbf{Statistic} & \textbf{Value} \\
\midrule
""" + f"""Null Deviance & {null_deviance:.2f} \\\\
Full Deviance & {full_deviance:.2f} \\\\
LR Statistic (χ²) & {lr_statistic:.2f} \\\\
Degrees of Freedom & {df} \\\\
p-value & {p_str} \\\\
""" + r"""\bottomrule
\end{tabular}
\end{table}"""
    
    return latex


def generate_summary_table(
    test_results: Dict[str, Any],
    test_type: str = "hypothesis"
) -> pd.DataFrame:
    """Generate summary table from any test results.
    
    Args:
        test_results: Dictionary of test results
        test_type: Type of test for formatting
        
    Returns:
        DataFrame suitable for display or export
    """
    rows = []
    
    for name, result in test_results.items():
        if isinstance(result, dict) and 'p_value' in result:
            row = {
                'Test/Variable': name,
                'Statistic': result.get('statistic', result.get('z_statistic', 'N/A')),
                'p-value': result.get('p_value', 'N/A'),
                'Significant': 'Yes' if result.get('significant', False) else 'No'
            }
            
            # Add test-specific columns
            if 'coefficient' in result:
                row['Coefficient'] = result['coefficient']
            if 'se' in result or 'standard_error' in result:
                row['SE'] = result.get('se', result.get('standard_error'))
            if 'df' in result:
                row['DF'] = result['df']
                
            rows.append(row)
    
    return pd.DataFrame(rows)


def generate_score_test_latex_table(
    test_results: Dict[str, Any]
) -> str:
    """Generate LaTeX table for Score test results.
    
    Args:
        test_results: Score test results dictionary
        
    Returns:
        LaTeX formatted table
    """
    # Extract statistics
    score_stat = test_results.get("score_statistic", test_results.get("statistic", 0))
    p_value = test_results.get("p_value", 1)
    df = test_results.get("degrees_of_freedom", test_results.get("df", 1))
    
    # Format p-value
    if p_value < 0.001:
        p_str = "< 0.001***"
    elif p_value < 0.01:
        p_str = f"{p_value:.4f}**"
    elif p_value < 0.05:
        p_str = f"{p_value:.4f}*"
    else:
        p_str = f"{p_value:.4f}"
    
    latex = r"""\begin{table}[htbp]
\centering
\caption{Score Test (Lagrange Multiplier) Results}
\label{tab:score_test}
\begin{tabular}{lc}
\toprule
\textbf{Statistic} & \textbf{Value} \\
\midrule
""" + f"""Score Statistic (χ²) & {score_stat:.2f} \\\\
Degrees of Freedom & {df} \\\\
p-value & {p_str} \\\\
Decision & {"Reject H₀" if p_value < 0.05 else "Fail to reject H₀"} \\\\
""" + r"""\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: The Score test is asymptotically equivalent to Wald and LRT tests.
\item Significance: *** p < 0.001, ** p < 0.01, * p < 0.05
\end{tablenotes}
\end{table}"""
    
    return latex
