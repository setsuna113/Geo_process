#!/usr/bin/env python3
"""
Report Generator for DML Diagnostics
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class DiagnosticReport:
    """Generate comprehensive diagnostic reports"""
    
    def __init__(self, diagnostics: Dict[str, Any]):
        self.diagnostics = diagnostics
        self.timestamp = datetime.now()
        
    def generate_html_report(self, save_path: str) -> str:
        """Generate HTML report"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DML Diagnostic Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; border-bottom: 2px solid #ddd; }}
                .warning {{ background-color: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; }}
                .success {{ background-color: #d4edda; padding: 10px; border-left: 4px solid #28a745; }}
                .info {{ background-color: #d1ecf1; padding: 10px; border-left: 4px solid #17a2b8; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Double Machine Learning Diagnostic Report</h1>
            <p>Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            {self._generate_summary_section()}
            {self._generate_coefficients_section()}
            {self._generate_diagnostics_section()}
            {self._generate_warnings_section()}
        </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(html)
            
        print(f"HTML report saved to {save_path}")
        return html
    
    def _generate_summary_section(self) -> str:
        """Generate summary HTML section"""
        
        html = "<h2>Summary</h2>"
        
        if 'model_info' in self.diagnostics:
            mi = self.diagnostics['model_info']
            html += f"""
            <div class='info'>
                <strong>Model Type:</strong> {mi.get('estimator_type', 'Unknown')}<br>
                <strong>Number of Treatments:</strong> {mi.get('n_treatments', 'N/A')}<br>
                <strong>Cross-validation Folds:</strong> {mi.get('n_folds', 'N/A')}
            </div>
            """
            
        if 'sample_size' in self.diagnostics:
            ss = self.diagnostics['sample_size']
            html += f"""
            <div class='info'>
                <strong>Sample Size:</strong> {ss.get('n_observations', 'N/A'):,}<br>
                <strong>N per Parameter:</strong> {ss.get('n_per_parameter', 'N/A'):.0f}
            </div>
            """
            
        return html
    
    def _generate_coefficients_section(self) -> str:
        """Generate coefficients HTML section"""
        
        html = "<h2>Coefficient Diagnostics</h2>"
        
        if 'coefficient_diagnostics' not in self.diagnostics:
            return html + "<p>No coefficient diagnostics available</p>"
            
        cd = self.diagnostics['coefficient_diagnostics']
        
        if 'coefficients' in cd and 't_statistics' in cd:
            html += """
            <table>
                <tr><th>Treatment</th><th>Coefficient</th><th>Std Error</th><th>t-statistic</th></tr>
            """
            
            coefs = cd['coefficients']
            t_stats = cd.get('t_statistics', [None] * len(coefs))
            stderrs = cd.get('std_errors', [None] * len(coefs))
            
            for i in range(len(coefs)):
                html += f"""
                <tr>
                    <td>Treatment {i+1}</td>
                    <td>{coefs[i]:.6f}</td>
                    <td>{stderrs[i]:.6f if stderrs[i] else 'N/A'}</td>
                    <td>{t_stats[i]:.2f if t_stats[i] else 'N/A'}</td>
                </tr>
                """
            html += "</table>"
            
        return html
    
    def _generate_diagnostics_section(self) -> str:
        """Generate diagnostics HTML section"""
        
        html = "<h2>Diagnostic Tests</h2>"
        
        # Cross-fitting
        if 'cross_fitting' in self.diagnostics:
            cf = self.diagnostics['cross_fitting']
            status = "✓" if cf.get('cross_fitting_used', False) else "✗"
            html += f"""
            <div class='{"success" if cf.get("cross_fitting_used") else "warning"}'>
                <strong>Cross-fitting:</strong> {status} 
                {f"({cf.get('n_folds')} folds)" if cf.get('n_folds') else "Not used"}
            </div>
            """
            
        # Orthogonality
        if 'orthogonality' in self.diagnostics:
            orth = self.diagnostics['orthogonality']
            html += f"""
            <div class='info'>
                <strong>Neyman Orthogonality:</strong> {"✓" if orth.get('neyman_orthogonal') else "✗"}<br>
                <strong>Score Type:</strong> {orth.get('score_type', 'Unknown')}
            </div>
            """
            
        # Convergence
        if 'convergence' in self.diagnostics:
            conv = self.diagnostics['convergence']
            status = "✓" if conv.get('converged', False) else "✗"
            html += f"""
            <div class='{"success" if conv.get("converged") else "warning"}'>
                <strong>Convergence:</strong> {status}
            </div>
            """
            
        return html
    
    def _generate_warnings_section(self) -> str:
        """Generate warnings HTML section"""
        
        html = "<h2>Warnings</h2>"
        
        warnings = []
        for section, diag in self.diagnostics.items():
            if isinstance(diag, dict) and 'warning' in diag:
                warnings.append(f"{section}: {diag['warning']}")
                
        if warnings:
            html += "<div class='warning'><ul>"
            for w in warnings:
                html += f"<li>{w}</li>"
            html += "</ul></div>"
        else:
            html += "<div class='success'>✓ No warnings detected</div>"
            
        return html
    
    def save_json(self, save_path: str):
        """Save diagnostics as JSON"""
        
        with open(save_path, 'w') as f:
            json.dump(self.diagnostics, f, indent=2, default=str)
            
        print(f"JSON report saved to {save_path}")
        
    def save_csv(self, save_path: str):
        """Save key metrics as CSV"""
        
        # Flatten diagnostics for CSV
        rows = []
        
        for section, content in self.diagnostics.items():
            if isinstance(content, dict):
                for key, value in content.items():
                    if isinstance(value, (int, float, bool, str)):
                        rows.append({
                            'section': section,
                            'metric': key,
                            'value': value
                        })
                        
        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False)
        
        print(f"CSV report saved to {save_path}")