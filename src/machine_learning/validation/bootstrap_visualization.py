#!/usr/bin/env python3
"""
Bootstrap Results Visualization

Reads results from Julia bootstrap analysis and creates comprehensive visualizations
for sample size determination and effect size distributions.

LINUS: Simple, direct visualization. No fancy libraries, just matplotlib.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class BootstrapVisualizer:
    """Visualize bootstrap sampling results from Julia analysis."""
    
    def __init__(self, results_dir: str = "bootstrap_results"):
        """Initialize with results directory."""
        self.results_dir = Path(results_dir)
        self.results = {}
        self.raw_data = {}
        
    def load_results(self, model_name: str) -> Dict:
        """Load results for a specific model."""
        # Load everything from NPZ
        npz_path = self.results_dir / f"bootstrap_{model_name}.npz"
        self.raw_data[model_name] = np.load(npz_path)
        
        # Reconstruct results structure from NPZ
        raw = self.raw_data[model_name]
        results = {
            'model_name': model_name,
            'N': int(raw['N_total']),
            'optimal_n': int(raw['optimal_n']),
            'config': {
                'n_iterations': int(raw['n_iterations']),
                'effect_threshold': float(raw['effect_threshold']),
                'power_threshold': float(raw['power_threshold'])
            },
            'sample_results': {}
        }
        
        # Extract results for each sample size
        sample_sizes = raw['sample_sizes']
        for n in sample_sizes:
            # Calculate statistics from raw data if not stored
            plants_effects = raw[f'n{n}_plants_effects']
            vertebrates_effects = raw[f'n{n}_vertebrates_effects']
            
            results['sample_results'][str(n)] = {
                'plants_effect': {
                    'mean': float(raw.get(f'n{n}_plants_mean', np.mean(plants_effects))),
                    'std': float(raw.get(f'n{n}_plants_std', np.std(plants_effects))),
                    'var': np.var(plants_effects),
                    'ci_95': [np.percentile(plants_effects, 2.5), np.percentile(plants_effects, 97.5)],
                    'raw_values': plants_effects
                },
                'vertebrates_effect': {
                    'mean': np.mean(vertebrates_effects),
                    'std': np.std(vertebrates_effects),
                    'var': np.var(vertebrates_effects),
                    'raw_values': vertebrates_effects
                },
                'power': float(raw[f'n{n}_power']),
                'ecological_power': float(raw[f'n{n}_eco_power']),
                'r2_improvement': {
                    'mean': np.mean(raw[f'n{n}_r2_improvements'])
                },
                'p_values': raw[f'n{n}_p_values'],
                'r2_improvements': raw[f'n{n}_r2_improvements']
            }
            
            # Add interaction if present
            if f'n{n}_interaction_effects' in raw:
                interaction_effects = raw[f'n{n}_interaction_effects']
                results['sample_results'][str(n)]['interaction_effect'] = {
                    'mean': np.mean(interaction_effects),
                    'std': np.std(interaction_effects),
                    'raw_values': interaction_effects
                }
            else:
                results['sample_results'][str(n)]['interaction_effect'] = None
        
        self.results[model_name] = results
        
        print(f"Loaded results for {model_name}")
        print(f"Optimal sample size: {raw['optimal_n']:,}")
        
        return self.results[model_name]
    
    def plot_comprehensive_analysis(self, model_name: str, save: bool = True):
        """Create comprehensive 4-panel visualization."""
        if model_name not in self.results:
            self.load_results(model_name)
        
        results = self.results[model_name]
        raw = self.raw_data[model_name]
        
        # Extract data
        sample_sizes = raw['sample_sizes']
        optimal_n = raw['optimal_n']
        
        # Prepare data for plotting
        plants_means = []
        plants_vars = []
        plants_cis = []
        vertebrates_means = []
        vertebrates_vars = []
        powers = []
        eco_powers = []
        r2_means = []
        
        for n in sample_sizes:
            sr = results['sample_results'][str(n)]
            plants_means.append(sr['plants_effect']['mean'])
            plants_vars.append(sr['plants_effect']['var'])
            plants_cis.append(sr['plants_effect']['ci_95'])
            vertebrates_means.append(sr['vertebrates_effect']['mean'])
            vertebrates_vars.append(sr['vertebrates_effect']['var'])
            powers.append(sr['power'])
            eco_powers.append(sr['ecological_power'])
            r2_means.append(sr['r2_improvement']['mean'])
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Bootstrap Sample Size Analysis: {model_name}', fontsize=16, fontweight='bold')
        
        # Panel 1: Variance Stabilization
        ax1 = axes[0, 0]
        ax1.semilogx(sample_sizes, plants_vars, 'o-', linewidth=2, markersize=8, label='Plants effect')
        ax1.semilogx(sample_sizes, vertebrates_vars, 's-', linewidth=2, markersize=8, label='Vertebrates effect')
        ax1.axhline(y=plants_vars[-1] * 1.1, color='red', linestyle='--', alpha=0.5, label='10% threshold')
        ax1.axvline(x=optimal_n, color='black', linestyle=':', linewidth=2, label=f'Optimal n={optimal_n:,}')
        ax1.set_xlabel('Bootstrap Sample Size (log scale)', fontsize=12)
        ax1.set_ylabel('Variance of Effect Size', fontsize=12)
        ax1.set_title('A. Variance Stabilization', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Effect Size Stability with CI
        ax2 = axes[0, 1]
        plants_cis = np.array(plants_cis)
        plants_lower = plants_cis[:, 0]
        plants_upper = plants_cis[:, 1]
        
        ax2.semilogx(sample_sizes, plants_means, 'o-', linewidth=2, markersize=8, label='Plants effect')
        ax2.fill_between(sample_sizes, plants_lower, plants_upper, alpha=0.3)
        ax2.semilogx(sample_sizes, vertebrates_means, 's-', linewidth=2, markersize=8, label='Vertebrates effect')
        ax2.axvline(x=optimal_n, color='black', linestyle=':', linewidth=2, label=f'Optimal n={optimal_n:,}')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Bootstrap Sample Size (log scale)', fontsize=12)
        ax2.set_ylabel('Effect Size (Mean ± 95% CI)', fontsize=12)
        ax2.set_title('B. Effect Size Stability', fontsize=13, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Statistical Power
        ax3 = axes[1, 0]
        ax3.semilogx(sample_sizes, powers, 'o-', linewidth=2, markersize=8, label='Standard power (p<0.05)')
        ax3.semilogx(sample_sizes, eco_powers, 's-', linewidth=2, markersize=8, 
                     label=f'Ecological power (p<0.05 & R²>{results["config"]["effect_threshold"]})')
        ax3.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='80% threshold')
        ax3.axvline(x=optimal_n, color='black', linestyle=':', linewidth=2, label=f'Optimal n={optimal_n:,}')
        ax3.set_xlabel('Bootstrap Sample Size (log scale)', fontsize=12)
        ax3.set_ylabel('Power', fontsize=12)
        ax3.set_ylim([0, 1.05])
        ax3.set_title('C. Statistical Power Analysis', fontsize=13, fontweight='bold')
        ax3.legend(loc='lower right')
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Effect Size Distributions
        ax4 = axes[1, 1]
        
        # Select subset of sample sizes for clarity
        display_sizes = [sample_sizes[i] for i in [0, len(sample_sizes)//3, 2*len(sample_sizes)//3, -1]]
        violin_data = []
        labels = []
        
        for n in display_sizes:
            effects = raw[f'n{n}_plants_effects']
            violin_data.append(effects)
            labels.append(f'n={n:,}')
        
        parts = ax4.violinplot(violin_data, positions=range(len(display_sizes)), 
                               widths=0.7, showmeans=True, showmedians=True)
        
        # Color the violins
        for pc in parts['bodies']:
            pc.set_facecolor('steelblue')
            pc.set_alpha(0.7)
        
        ax4.set_xticks(range(len(display_sizes)))
        ax4.set_xticklabels(labels)
        ax4.set_xlabel('Sample Size', fontsize=12)
        ax4.set_ylabel('Plants Effect Size Distribution', fontsize=12)
        ax4.set_title('D. Effect Size Distributions', fontsize=13, fontweight='bold')
        ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            save_path = self.results_dir / f"bootstrap_{model_name}_analysis.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_convergence_diagnostic(self, model_name: str, save: bool = True):
        """Create convergence diagnostic plot."""
        if model_name not in self.results:
            self.load_results(model_name)
        
        results = self.results[model_name]
        raw = self.raw_data[model_name]
        
        sample_sizes = raw['sample_sizes']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Convergence Diagnostics: {model_name}', fontsize=14, fontweight='bold')
        
        # Plot 1: Relative variance change
        plants_vars = [results['sample_results'][str(n)]['plants_effect']['var'] for n in sample_sizes]
        relative_changes = []
        for i in range(1, len(plants_vars)):
            rel_change = abs(plants_vars[i] - plants_vars[i-1]) / plants_vars[i-1]
            relative_changes.append(rel_change * 100)
        
        ax1 = axes[0]
        ax1.semilogx(sample_sizes[1:], relative_changes, 'o-', linewidth=2, markersize=8)
        ax1.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% threshold')
        ax1.set_xlabel('Sample Size (log scale)', fontsize=12)
        ax1.set_ylabel('Relative Variance Change (%)', fontsize=12)
        ax1.set_title('Variance Convergence', fontsize=13)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Power trajectory
        ax2 = axes[1]
        powers = [results['sample_results'][str(n)]['power'] for n in sample_sizes]
        eco_powers = [results['sample_results'][str(n)]['ecological_power'] for n in sample_sizes]
        
        ax2.semilogx(sample_sizes, powers, 'o-', linewidth=2, markersize=8, label='Statistical')
        ax2.semilogx(sample_sizes, eco_powers, 's-', linewidth=2, markersize=8, label='Ecological')
        ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Sample Size (log scale)', fontsize=12)
        ax2.set_ylabel('Power', fontsize=12)
        ax2.set_title('Power Trajectory', fontsize=13)
        ax2.set_ylim([0, 1.05])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: R² improvement distribution
        ax3 = axes[2]
        r2_data = []
        labels = []
        for i, n in enumerate(sample_sizes[::2]):  # Every other size for clarity
            r2_data.append(raw[f'n{n}_r2_improvements'] * 100)  # Convert to percentage
            labels.append(f'{n//1000}k' if n >= 1000 else str(n))
        
        bp = ax3.boxplot(r2_data, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax3.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5% threshold')
        ax3.set_xlabel('Sample Size', fontsize=12)
        ax3.set_ylabel('R² Improvement (%)', fontsize=12)
        ax3.set_title('Model Improvement Distribution', fontsize=13)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            save_path = self.results_dir / f"bootstrap_{model_name}_convergence.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        
        plt.show()
        
        return fig
    
    def create_summary_table(self, model_names: List[str]) -> pd.DataFrame:
        """Create summary table comparing all models."""
        summary_data = []
        
        for model in model_names:
            if model not in self.results:
                self.load_results(model)
            
            results = self.results[model]
            optimal_n = results['optimal_n']
            
            # Get results at optimal n
            optimal_results = results['sample_results'][str(optimal_n)]
            
            summary_data.append({
                'Model': model,
                'Optimal n': f"{optimal_n:,}",
                'Optimal n (% of data)': f"{100 * optimal_n / results['N']:.1f}%",
                'Plants Effect': f"{optimal_results['plants_effect']['mean']:.3f} ± {optimal_results['plants_effect']['std']:.3f}",
                'Vertebrates Effect': f"{optimal_results['vertebrates_effect']['mean']:.3f} ± {optimal_results['vertebrates_effect']['std']:.3f}",
                'Statistical Power': f"{optimal_results['power']:.1%}",
                'Ecological Power': f"{optimal_results['ecological_power']:.1%}",
                'Mean R² Improvement': f"{optimal_results['r2_improvement']['mean']:.2%}",
            })
            
            # Add interaction if present
            if optimal_results.get('interaction_effect') and optimal_results['interaction_effect']:
                summary_data[-1]['Interaction Effect'] = \
                    f"{optimal_results['interaction_effect']['mean']:.3f} ± {optimal_results['interaction_effect']['std']:.3f}"
        
        df = pd.DataFrame(summary_data)
        
        # Save as CSV
        csv_path = self.results_dir / "bootstrap_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved summary table to: {csv_path}")
        
        # Display
        print("\n" + "="*80)
        print("BOOTSTRAP ANALYSIS SUMMARY")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        return df
    
    def plot_effect_comparison(self, model_names: List[str], save: bool = True):
        """Compare effect sizes across models at their optimal sample sizes."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Effect Size Comparison at Optimal Sample Sizes', fontsize=14, fontweight='bold')
        
        # Detect which biodiversity predictors are available
        first_model = model_names[0]
        if first_model not in self.results:
            self.load_results(first_model)
        
        # Check for available effect columns
        optimal_n = self.results[first_model]['optimal_n']
        raw = self.raw_data[first_model]
        effect_cols = [col for col in raw.keys() if col.startswith(f'n{optimal_n}_') and col.endswith('_effects')]
        
        # Extract predictor names
        predictors = []
        for col in effect_cols:
            pred_name = col.replace(f'n{optimal_n}_', '').replace('_effects', '')
            if pred_name not in predictors:
                predictors.append(pred_name)
        
        # Default to plants and vertebrates if available, otherwise use first two
        if 'plants' in predictors and 'vertebrates' in predictors:
            pred1, pred2 = 'plants', 'vertebrates'
        elif 'fungi' in predictors and 'vertebrates' in predictors:
            pred1, pred2 = 'fungi', 'vertebrates'
        elif 'fungi' in predictors and 'plants' in predictors:
            pred1, pred2 = 'fungi', 'plants'
        elif len(predictors) >= 2:
            pred1, pred2 = predictors[0], predictors[1]
        else:
            print(f"Warning: Not enough predictors found. Available: {predictors}")
            return
        
        pred1_data = []
        pred2_data = []
        labels = []
        
        for model in model_names:
            if model not in self.results:
                self.load_results(model)
            
            optimal_n = self.results[model]['optimal_n']
            raw = self.raw_data[model]
            
            pred1_data.append(raw[f'n{optimal_n}_{pred1}_effects'])
            pred2_data.append(raw[f'n{optimal_n}_{pred2}_effects'])
            labels.append(f"{model}\n(n={optimal_n:,})")
        
        # First predictor effect comparison
        ax1 = axes[0]
        bp1 = ax1.boxplot(pred1_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp1['boxes'], sns.color_palette()):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('Effect Size', fontsize=12)
        ax1.set_title(f'{pred1.capitalize()} Richness Effect', fontsize=13)
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Second biodiversity predictor effect comparison
        ax2 = axes[1]
        bp2 = ax2.boxplot(vertebrates_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp2['boxes'], sns.color_palette()):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Effect Size', fontsize=12)
        ax2.set_title(f'{pred2.capitalize()} Richness Effect', fontsize=13)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            save_path = self.results_dir / "bootstrap_effect_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to: {save_path}")
        
        plt.show()
        
        return fig


def main():
    """Main execution for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize bootstrap sampling results')
    parser.add_argument('--results-dir', default='bootstrap_results', 
                       help='Directory containing bootstrap results')
    parser.add_argument('--models', nargs='+', 
                       default=['method1', 'method2_continuous', 'method2_categorical'],
                       help='Model names to analyze')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save plots')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    viz = BootstrapVisualizer(args.results_dir)
    
    # Create visualizations for each model
    for model in args.models:
        try:
            print(f"\nProcessing {model}...")
            viz.plot_comprehensive_analysis(model, save=not args.no_save)
            viz.plot_convergence_diagnostic(model, save=not args.no_save)
        except FileNotFoundError:
            print(f"Results not found for {model}, skipping...")
    
    # Create comparison if multiple models
    if len(args.models) > 1:
        viz.create_summary_table(args.models)
        viz.plot_effect_comparison(args.models, save=not args.no_save)


if __name__ == "__main__":
    main()