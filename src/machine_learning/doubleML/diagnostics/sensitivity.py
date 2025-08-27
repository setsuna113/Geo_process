#!/usr/bin/env python3
"""
Standalone Sensitivity Analysis for Single-Treatment EconML Models

This module provides sensitivity analysis ONLY for single-treatment models.
Uses the 3 actual EconML sensitivity methods:
1. robustness_value()
2. sensitivity_interval()
3. sensitivity_summary()
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple, Any, List
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class SingleTreatmentSensitivity:
    """
    Sensitivity analysis for single-treatment causal models.
    
    REQUIRES: A fitted single-treatment LinearDML model with inference.
    
    Example:
        # Assuming you have a fitted single-treatment model
        sensitivity = SingleTreatmentSensitivity(model)
        results = sensitivity.run_all_sensitivity_analysis()
        sensitivity.plot_sensitivity_results()
    """
    
    def __init__(self, model, treatment_name: str = "Treatment"):
        """
        Initialize with a SINGLE-treatment fitted model.
        
        Parameters
        ----------
        model : LinearDML
            Fitted single-treatment EconML model (with inference enabled)
        treatment_name : str
            Name of the treatment for labeling
        """
        if not hasattr(model, 'intercept_'):
            raise ValueError("Model must be fitted with inference enabled")
            
        # Check if single treatment
        if hasattr(model, '_d_t') and model._d_t and model._d_t[0] > 1:
            raise ValueError("This class only works with single-treatment models. "
                           "For multi-treatment, fit separate single-treatment models.")
        
        self.model = model
        self.treatment_name = treatment_name
        self.results = {}
        
    def run_all_sensitivity_analysis(self, 
                                    alpha: float = 0.05,
                                    confounding_values: Optional[List[float]] = None) -> Dict:
        """
        Run all 3 available EconML sensitivity analyses.
        
        Parameters
        ----------
        alpha : float
            Significance level for tests
        confounding_values : list of float
            Confounding strengths to test (default: [0.01, 0.05, 0.1, 0.2, 0.3])
            
        Returns
        -------
        Dict with all sensitivity results
        """
        print(f"\n{'='*70}")
        print(f"SENSITIVITY ANALYSIS FOR: {self.treatment_name}")
        print(f"{'='*70}")
        
        if confounding_values is None:
            confounding_values = [0.01, 0.05, 0.1, 0.2, 0.3]
        
        results = {}
        
        # 1. Robustness Value
        print("\n1. ROBUSTNESS VALUE (robustness_value):")
        print("-" * 50)
        try:
            rv = self.model.robustness_value(null_hypothesis=0, alpha=alpha)
            results['robustness_value'] = rv
            print(f"   RV = {rv:.4f}")
            print(f"   Interpretation: A confounder needs to explain {rv*100:.1f}% of residual")
            print(f"                  variation to invalidate the result")
            
            # Interpret strength
            if rv < 0.05:
                strength = "VERY FRAGILE - vulnerable to weak confounders"
            elif rv < 0.1:
                strength = "FRAGILE - vulnerable to modest confounders"
            elif rv < 0.3:
                strength = "MODERATE - requires moderate confounding"
            else:
                strength = "ROBUST - requires strong confounding"
            print(f"   Robustness: {strength}")
            
        except Exception as e:
            print(f"   Failed: {e}")
            results['robustness_value'] = None
        
        # 2. Sensitivity Intervals for different confounding levels
        print("\n2. SENSITIVITY INTERVALS (sensitivity_interval):")
        print("-" * 50)
        
        intervals = []
        for c in confounding_values:
            try:
                lb, ub = self.model.sensitivity_interval(
                    alpha=alpha, 
                    c_y=c,  # confounding in outcome
                    c_t=c,  # confounding in treatment
                    rho=1.0  # worst-case correlation
                )
                intervals.append({
                    'confounding': c,
                    'lower': lb,
                    'upper': ub,
                    'includes_zero': lb <= 0 <= ub
                })
                
                print(f"   c={c:.2f}: [{lb:.4f}, {ub:.4f}] {'(includes 0)' if lb <= 0 <= ub else '(excludes 0)'}")
                
            except Exception as e:
                print(f"   c={c:.2f}: Failed - {e}")
                
        results['sensitivity_intervals'] = intervals
        
        # 3. Sensitivity Summary (new method!)
        print("\n3. SENSITIVITY SUMMARY (sensitivity_summary):")
        print("-" * 50)
        
        try:
            # Try different confounding levels to get summaries
            summaries = []
            for c in [0.05, 0.1, 0.2]:
                summary = self.model.sensitivity_summary(
                    null_hypothesis=0,
                    alpha=alpha,
                    c_y=c,
                    c_t=c,
                    rho=1.0,
                    decimals=3
                )
                summaries.append({'confounding': c, 'summary': summary})
                
            results['sensitivity_summaries'] = summaries
            
            # Print the moderate confounding summary
            if len(summaries) > 1:
                print(f"   Summary at c=0.1:")
                print(f"   {summaries[1]['summary']}")
            
        except Exception as e:
            print(f"   Failed to get summary: {e}")
            results['sensitivity_summaries'] = None
        
        # 4. Original effect and confidence interval
        print("\n4. ORIGINAL ESTIMATE (no confounding assumed):")
        print("-" * 50)
        
        effect = float(self.model.intercept_)
        inference = self.model.intercept__inference()
        stderr = float(inference.stderr)
        pvalue = float(inference.pvalue())
        
        ci_lower, ci_upper = inference.conf_int(alpha=alpha)
        
        results['original'] = {
            'effect': effect,
            'stderr': stderr,
            'pvalue': pvalue,
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'significant': pvalue < alpha
        }
        
        print(f"   Effect: {effect:.4f} (SE: {stderr:.4f})")
        print(f"   95% CI: [{float(ci_lower):.4f}, {float(ci_upper):.4f}]")
        print(f"   P-value: {pvalue:.4f}")
        print(f"   Significant: {'Yes' if pvalue < alpha else 'No'}")
        
        # 5. Summary
        print("\n5. OVERALL ASSESSMENT:")
        print("-" * 50)
        
        # How much confounding before losing significance?
        critical_c = None
        for interval in intervals:
            if interval['includes_zero']:
                critical_c = interval['confounding']
                break
        
        if critical_c:
            print(f"   Effect becomes insignificant at c={critical_c:.2f}")
            print(f"   → Vulnerable to confounders explaining >{critical_c*100:.0f}% of variation")
        else:
            print(f"   Effect remains significant even at c={confounding_values[-1]:.2f}")
            print(f"   → Robust to strong confounding")
        
        results['critical_confounding'] = critical_c
        
        self.results = results
        return results
    
    def plot_sensitivity_results(self, save_path: Optional[str] = None):
        """
        Create publication-quality visualization of sensitivity analysis.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the figure (300 DPI)
        """
        if not self.results:
            print("No results to plot. Run run_all_sensitivity_analysis() first.")
            return
        
        import matplotlib.gridspec as gridspec
        
        # Set style for publication quality
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, hspace=0.3, wspace=0.3)
        
        # Color scheme
        primary_color = '#2E7D32'  # Green
        secondary_color = '#1565C0'  # Blue
        warning_color = '#F57C00'  # Orange
        critical_color = '#C62828'  # Red
        
        # Title
        fig.suptitle(f'Sensitivity Analysis: {self.treatment_name}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Main Sensitivity Intervals Plot (spans 2 columns)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        
        intervals = self.results.get('sensitivity_intervals', [])
        if intervals:
            c_values = np.array([x['confounding'] for x in intervals])
            lower = np.array([x['lower'] for x in intervals])
            upper = np.array([x['upper'] for x in intervals])
            
            # Original estimate
            orig = self.results['original']
            
            # Plot sensitivity bounds
            ax1.fill_between(c_values, lower, upper, alpha=0.3, color=primary_color, 
                            label='Sensitivity bounds')
            ax1.plot(c_values, lower, '--', color=primary_color, alpha=0.6, linewidth=1)
            ax1.plot(c_values, upper, '--', color=primary_color, alpha=0.6, linewidth=1)
            
            # Original confidence interval
            ax1.fill_between([0, c_values[-1]], 
                           [orig['ci_lower']]*2, [orig['ci_upper']]*2,
                           alpha=0.15, color=secondary_color, 
                           label='Original 95% CI')
            
            # Point estimate line
            ax1.axhline(orig['effect'], color='black', linewidth=2, 
                       label=f"Point estimate: {orig['effect']:.4f}")
            
            # Zero effect line
            ax1.axhline(0, color=critical_color, linestyle='-', linewidth=1.5, 
                       alpha=0.7, label='Null effect')
            
            # Mark critical confounding point
            if self.results['critical_confounding']:
                cc = self.results['critical_confounding']
                ax1.axvline(cc, color=warning_color, linestyle=':', linewidth=2,
                          label=f'Critical c = {cc:.2f}')
                
                # Add shaded region after critical point
                ax1.axvspan(cc, c_values[-1], alpha=0.1, color=critical_color)
                ax1.text(cc + 0.01, ax1.get_ylim()[0] * 0.9, 
                        'Effect not significant', rotation=90, 
                        fontsize=9, alpha=0.7)
            
            ax1.set_xlabel('Confounding Strength (c)', fontsize=12)
            ax1.set_ylabel('Treatment Effect', fontsize=12)
            ax1.set_title('Sensitivity of Effect Estimate to Unmeasured Confounding', 
                         fontsize=13, fontweight='bold')
            ax1.legend(loc='best', framealpha=0.9)
            ax1.grid(True, alpha=0.3)
            
            # Add annotations for interpretation
            y_pos = ax1.get_ylim()[1] * 0.85
            for c, label in [(0.05, 'Weak'), (0.1, 'Modest'), (0.2, 'Moderate'), (0.3, 'Strong')]:
                if c <= c_values[-1]:
                    ax1.text(c, y_pos, label, ha='center', fontsize=8, 
                            alpha=0.6, bbox=dict(boxstyle='round', 
                            facecolor='white', alpha=0.5))
        
        # 2. Robustness Value Gauge
        ax2 = fig.add_subplot(gs[0, 2])
        rv = self.results.get('robustness_value', 0)
        
        if rv:
            # Create gauge visualization
            theta = np.linspace(np.pi, 0, 100)
            r_inner = 0.7
            r_outer = 1.0
            
            # Define color zones
            zones = [
                (0, 0.05, critical_color, 'Very\nFragile'),
                (0.05, 0.1, warning_color, 'Fragile'),
                (0.1, 0.3, '#FFC107', 'Moderate'),
                (0.3, 1.0, primary_color, 'Robust')
            ]
            
            ax2.set_xlim(-1.2, 1.2)
            ax2.set_ylim(-0.2, 1.2)
            
            for start, end, color, label in zones:
                theta_zone = np.linspace(np.pi * (1 - start), np.pi * (1 - end), 50)
                x_outer = r_outer * np.cos(theta_zone)
                y_outer = r_outer * np.sin(theta_zone)
                x_inner = r_inner * np.cos(theta_zone)
                y_inner = r_inner * np.sin(theta_zone)
                
                vertices = list(zip(x_outer, y_outer)) + list(zip(x_inner[::-1], y_inner[::-1]))
                from matplotlib.patches import Polygon
                poly = Polygon(vertices, facecolor=color, alpha=0.6)
                ax2.add_patch(poly)
            
            # Add needle
            needle_angle = np.pi * (1 - rv)
            ax2.arrow(0, 0, 0.85 * np.cos(needle_angle), 0.85 * np.sin(needle_angle),
                     head_width=0.1, head_length=0.05, fc='black', ec='black', linewidth=2)
            
            # Add center dot
            circle = plt.Circle((0, 0), 0.05, color='black')
            ax2.add_artist(circle)
            
            # Add value text
            ax2.text(0, -0.1, f'RV = {rv:.4f}', ha='center', fontsize=12, fontweight='bold')
            
            # Interpretation
            if rv < 0.05:
                interp = "VERY FRAGILE"
                color = critical_color
            elif rv < 0.1:
                interp = "FRAGILE"
                color = warning_color
            elif rv < 0.3:
                interp = "MODERATE"
                color = '#FFC107'
            else:
                interp = "ROBUST"
                color = primary_color
                
            ax2.text(0, -0.3, interp, ha='center', fontsize=11, 
                    color=color, fontweight='bold')
            
            ax2.set_aspect('equal')
            ax2.axis('off')
            ax2.set_title('Robustness Value', fontsize=12, fontweight='bold')
        
        # 3. Effect Degradation Chart
        ax3 = fig.add_subplot(gs[1, 2])
        
        if intervals:
            # Calculate effect size relative to original
            midpoints = [(x['lower'] + x['upper'])/2 for x in intervals]
            relative_effect = np.array([0] + midpoints) / orig['effect'] * 100
            c_with_zero = np.array([0] + list(c_values))
            
            ax3.plot(c_with_zero, relative_effect, 'o-', linewidth=2, 
                    markersize=6, color=primary_color)
            ax3.axhline(100, color='gray', linestyle='--', alpha=0.5, 
                       label='Original effect')
            ax3.axhline(0, color=critical_color, linestyle='-', alpha=0.5,
                       label='Null effect')
            
            # Shade confidence region
            lower_rel = np.array([orig['ci_lower']] + list(lower)) / orig['effect'] * 100
            upper_rel = np.array([orig['ci_upper']] + list(upper)) / orig['effect'] * 100
            ax3.fill_between(c_with_zero, lower_rel, upper_rel, alpha=0.2, color=primary_color)
            
            ax3.set_xlabel('Confounding Strength (c)', fontsize=11)
            ax3.set_ylabel('Effect Size (% of original)', fontsize=11)
            ax3.set_title('Effect Degradation', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc='best', fontsize=9)
        
        # 4. Statistical Summary Table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # Create summary data
        summary_data = [
            ['Metric', 'Value', 'Interpretation'],
            ['Original Effect', f"{orig['effect']:.6f} ± {orig['stderr']:.6f}", 
             'Significant' if orig['pvalue'] < 0.05 else 'Not significant'],
            ['P-value', f"{orig['pvalue']:.4e}", 
             '***' if orig['pvalue'] < 0.001 else '**' if orig['pvalue'] < 0.01 else '*'],
            ['95% CI', f"[{orig['ci_lower']:.6f}, {orig['ci_upper']:.6f}]", 
             'Excludes zero' if orig['ci_lower'] > 0 or orig['ci_upper'] < 0 else 'Includes zero'],
            ['Robustness Value', f"{rv:.4f}" if rv else 'N/A',
             'Fragile' if rv and rv < 0.1 else 'Moderate' if rv and rv < 0.3 else 'Robust'],
            ['Critical Confounding', 
             f"{self.results['critical_confounding']:.2f}" if self.results['critical_confounding'] else '>0.50',
             f"Effect lost at {self.results['critical_confounding']*100:.0f}% confounding" if self.results['critical_confounding'] else 'Robust to strong confounding']
        ]
        
        # Create table
        col_widths = [0.25, 0.35, 0.4]
        table = ax4.table(cellText=summary_data, loc='center', 
                         colWidths=col_widths, cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)
        
        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('#E0E0E0')
            table[(0, i)].set_text_props(weight='bold')
            table[(0, i)].set_text_props(ha='center')
        
        # Style cells
        for i in range(1, len(summary_data)):
            table[(i, 0)].set_facecolor('#F5F5F5')
            table[(i, 0)].set_text_props(weight='bold')
            
            # Color code based on robustness
            if i == 4 and rv:  # Robustness value row
                if rv < 0.05:
                    table[(i, 1)].set_facecolor('#FFEBEE')  # Light red
                elif rv < 0.1:
                    table[(i, 1)].set_facecolor('#FFF3E0')  # Light orange
                elif rv < 0.3:
                    table[(i, 1)].set_facecolor('#FFFDE7')  # Light yellow
                else:
                    table[(i, 1)].set_facecolor('#E8F5E9')  # Light green
        
        ax4.set_title('Statistical Summary', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nEnhanced visualization saved to: {save_path}")
        
        plt.show()
        return fig


def analyze_single_treatment_from_file(model_path: str, 
                                      treatment_name: str = "Treatment") -> SingleTreatmentSensitivity:
    """
    Convenience function to load a model and run sensitivity analysis.
    
    Parameters
    ----------
    model_path : str
        Path to pickled single-treatment LinearDML model
    treatment_name : str
        Name of the treatment for labeling
        
    Returns
    -------
    SingleTreatmentSensitivity object with results
    """
    import pickle
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    sensitivity = SingleTreatmentSensitivity(model, treatment_name)
    sensitivity.run_all_sensitivity_analysis()
    
    return sensitivity


if __name__ == "__main__":
    # Example usage
    print("Single-Treatment Sensitivity Analysis Module")
    print("=" * 70)
    print("\nThis module uses EconML's 3 native sensitivity methods:")
    print("  1. robustness_value() - Confounding strength needed to invalidate")
    print("  2. sensitivity_interval() - Confidence bounds under confounding")
    print("  3. sensitivity_summary() - Formatted summary of sensitivity")
    print("\nExample usage:")
    print("  from diagnostics.sensitivity import SingleTreatmentSensitivity")
    print("  sensitivity = SingleTreatmentSensitivity(model, 'Plants Richness')")
    print("  results = sensitivity.run_all_sensitivity_analysis()")
    print("  sensitivity.plot_sensitivity_results()")