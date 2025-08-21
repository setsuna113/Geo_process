#!/usr/bin/env python3
"""
Visualization for DML Diagnostics
Creates diagnostic plots for DML models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class DiagnosticPlotter:
    """Create diagnostic visualizations for DML"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize plotter"""
        self.figsize = figsize
        
    def plot_residuals(self, residuals: Dict[str, np.ndarray], 
                      save_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive residual plots"""
        
        n_residuals = len(residuals)
        fig, axes = plt.subplots(n_residuals, 4, figsize=(16, 4*n_residuals))
        
        if n_residuals == 1:
            axes = axes.reshape(1, -1)
            
        for idx, (name, resid) in enumerate(residuals.items()):
            if isinstance(resid, list):
                continue
                
            # 1. Histogram with normal overlay
            ax = axes[idx, 0]
            ax.hist(resid, bins=50, density=True, alpha=0.7, edgecolor='black')
            
            # Overlay normal distribution
            mu, sigma = np.mean(resid), np.std(resid)
            x = np.linspace(resid.min(), resid.max(), 100)
            ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label='Normal')
            ax.set_title(f'{name} - Histogram')
            ax.set_xlabel('Residual Value')
            ax.set_ylabel('Density')
            ax.legend()
            
            # 2. Q-Q plot
            ax = axes[idx, 1]
            stats.probplot(resid, dist="norm", plot=ax)
            ax.set_title(f'{name} - Q-Q Plot')
            ax.grid(True, alpha=0.3)
            
            # 3. Residuals vs Index
            ax = axes[idx, 2]
            ax.scatter(range(len(resid)), resid, alpha=0.5, s=1)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.axhline(y=3*sigma, color='orange', linestyle='--', alpha=0.5, label='±3σ')
            ax.axhline(y=-3*sigma, color='orange', linestyle='--', alpha=0.5)
            ax.set_title(f'{name} - Residuals vs Index')
            ax.set_xlabel('Index')
            ax.set_ylabel('Residual')
            ax.legend()
            
            # 4. ACF plot (simplified)
            ax = axes[idx, 3]
            n_lags = min(40, len(resid) // 4)
            acf = [1.0]
            for lag in range(1, n_lags):
                acf.append(np.corrcoef(resid[:-lag], resid[lag:])[0, 1])
            ax.bar(range(n_lags), acf, alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.axhline(y=1.96/np.sqrt(len(resid)), color='r', linestyle='--', alpha=0.5)
            ax.axhline(y=-1.96/np.sqrt(len(resid)), color='r', linestyle='--', alpha=0.5)
            ax.set_title(f'{name} - ACF')
            ax.set_xlabel('Lag')
            ax.set_ylabel('Autocorrelation')
            
        plt.suptitle('Residual Diagnostics', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Residual plots saved to {save_path}")
            
        return fig
    
    def plot_coefficients(self, coef: np.ndarray, stderr: np.ndarray,
                         names: List[str], save_path: Optional[str] = None) -> plt.Figure:
        """Plot coefficient estimates with confidence intervals"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Calculate confidence intervals
        ci_lower = coef - 1.96 * stderr
        ci_upper = coef + 1.96 * stderr
        
        # 1. Coefficient plot
        y_pos = np.arange(len(names))
        ax1.errorbar(coef, y_pos, xerr=1.96*stderr, fmt='o', capsize=5, capthick=2)
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(names)
        ax1.set_xlabel('Coefficient Value')
        ax1.set_title('Treatment Effect Estimates')
        ax1.grid(True, alpha=0.3)
        
        # 2. T-statistics
        t_stats = coef / stderr
        colors = ['green' if abs(t) > 2 else 'orange' if abs(t) > 1.5 else 'red' 
                 for t in t_stats]
        ax2.barh(y_pos, t_stats, color=colors, alpha=0.7)
        ax2.axvline(x=2, color='green', linestyle='--', alpha=0.5, label='|t|=2')
        ax2.axvline(x=-2, color='green', linestyle='--', alpha=0.5)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(names)
        ax2.set_xlabel('t-statistic')
        ax2.set_title('Statistical Significance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Coefficient Diagnostics', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Coefficient plots saved to {save_path}")
            
        return fig
    
    def plot_diagnostic_summary(self, diagnostics: Dict[str, Any],
                               save_path: Optional[str] = None) -> plt.Figure:
        """Create summary diagnostic dashboard"""
        
        fig = plt.figure(figsize=(14, 10))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Model info box
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        info_text = "MODEL DIAGNOSTICS SUMMARY\n" + "="*50 + "\n"
        if 'model_info' in diagnostics:
            mi = diagnostics['model_info']
            info_text += f"Estimator: {mi.get('estimator_type', 'Unknown')}\n"
            info_text += f"N Treatments: {mi.get('n_treatments', 'N/A')}\n"
            info_text += f"CV Folds: {mi.get('n_folds', 'N/A')}\n"
            
        if 'sample_size' in diagnostics:
            ss = diagnostics['sample_size']
            info_text += f"Sample Size: {ss.get('n_observations', 'N/A'):,}\n"
            
        ax1.text(0.5, 0.5, info_text, transform=ax1.transAxes,
                fontsize=11, verticalalignment='center',
                horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Warnings panel
        ax2 = fig.add_subplot(gs[1, :])
        ax2.axis('off')
        
        warnings_text = "DIAGNOSTIC WARNINGS\n" + "-"*30 + "\n"
        warning_count = 0
        
        for section, diag in diagnostics.items():
            if isinstance(diag, dict) and 'warning' in diag:
                warnings_text += f"• {section}: {diag['warning']}\n"
                warning_count += 1
                
        if warning_count == 0:
            warnings_text += "✓ No warnings detected\n"
            
        color = 'red' if warning_count > 2 else 'orange' if warning_count > 0 else 'green'
        ax2.text(0.5, 0.5, warnings_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='center',
                horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
        
        # 3. Coefficient statistics
        if 'coefficient_diagnostics' in diagnostics:
            ax3 = fig.add_subplot(gs[2, 0])
            cd = diagnostics['coefficient_diagnostics']
            
            if 't_statistics' in cd:
                t_stats = cd['t_statistics']
                ax3.hist(t_stats, bins=20, edgecolor='black', alpha=0.7)
                ax3.axvline(x=2, color='green', linestyle='--', alpha=0.5, label='|t|=2')
                ax3.axvline(x=-2, color='green', linestyle='--', alpha=0.5)
                ax3.set_xlabel('t-statistic')
                ax3.set_ylabel('Frequency')
                ax3.set_title('t-statistics Distribution')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax3.axis('off')
                ax3.text(0.5, 0.5, 'No t-statistics available',
                        transform=ax3.transAxes,
                        horizontalalignment='center')
                
        # 4. Sample size adequacy
        if 'sample_size' in diagnostics:
            ax4 = fig.add_subplot(gs[2, 1])
            ss = diagnostics['sample_size']
            
            if 'n_per_parameter' in ss:
                n_per_param = ss['n_per_parameter']
                adequate = ss.get('adequate_sample_size', False)
                
                color = 'green' if adequate else 'red'
                ax4.bar(['N per Parameter'], [n_per_param], color=color, alpha=0.7)
                ax4.axhline(y=100, color='orange', linestyle='--', 
                           label='Recommended minimum')
                ax4.set_ylabel('Sample Size per Parameter')
                ax4.set_title('Sample Size Adequacy')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax4.axis('off')
                
        # 5. Convergence indicators
        ax5 = fig.add_subplot(gs[2, 2])
        
        convergence_data = []
        convergence_labels = []
        
        if 'convergence' in diagnostics:
            conv = diagnostics['convergence']
            if 'condition_number' in conv:
                cond = np.log10(conv['condition_number'])
                convergence_data.append(cond)
                convergence_labels.append(f'log10(Cond#): {cond:.1f}')
                
        if 'orthogonality' in diagnostics:
            orth = diagnostics['orthogonality']
            if 'score_mean' in orth:
                score_mean = abs(orth['score_mean']) * 1000  # Scale for visibility
                convergence_data.append(score_mean)
                convergence_labels.append(f'Score Mean (×1000): {score_mean:.2f}')
                
        if convergence_data:
            y_pos = np.arange(len(convergence_labels))
            colors = ['green' if v < 5 else 'orange' if v < 10 else 'red' 
                     for v in convergence_data]
            ax5.barh(y_pos, convergence_data, color=colors, alpha=0.7)
            ax5.set_yticks(y_pos)
            ax5.set_yticklabels(convergence_labels)
            ax5.set_xlabel('Value')
            ax5.set_title('Convergence Indicators')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.axis('off')
            
        plt.suptitle('DML Diagnostic Dashboard', fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Diagnostic summary saved to {save_path}")
            
        return fig