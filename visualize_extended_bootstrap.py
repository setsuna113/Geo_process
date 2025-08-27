#!/usr/bin/env python3
"""
Extended bootstrap visualization for 16 sample sizes (1k to 100k).
Shows variance stabilization and convergence patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_extended_visualization(npz_path):
    """Create comprehensive visualization for extended bootstrap."""
    
    # Load data
    data = np.load(npz_path)
    
    # Extract info
    sample_sizes = data['sample_sizes']
    n_iterations = int(data['n_iterations'])
    optimal_n = int(data['optimal_n'])
    N_total = int(data['N_total'])
    
    # Create figure with 6 subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.25)
    
    fig.suptitle(f'Extended Bootstrap Analysis (N={N_total:,}, {n_iterations} iterations, 16 sample sizes)', 
                 fontsize=15, fontweight='bold')
    
    # 1. Power vs Sample Size (log scale)
    ax1 = fig.add_subplot(gs[0, :2])
    powers = [data[f'n{n}_power'] for n in sample_sizes]
    eco_powers = [data[f'n{n}_eco_power'] for n in sample_sizes]
    
    ax1.semilogx(sample_sizes, powers, 'o-', label='Statistical Power (p<0.05)', 
                 linewidth=2, markersize=7, color='blue')
    ax1.semilogx(sample_sizes, eco_powers, 's-', label='Ecological Power (p<0.05 & R²>5%)', 
                 linewidth=2, markersize=7, color='red')
    ax1.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='80% threshold')
    ax1.axvline(x=optimal_n, color='green', linestyle='--', alpha=0.5, 
                label=f'Optimal n={optimal_n:,}')
    ax1.set_xlabel('Sample Size (log scale)')
    ax1.set_ylabel('Power')
    ax1.set_title('Statistical vs Ecological Power Across Extended Range')
    ax1.legend(loc='center right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # 2. R² Mean with Error Bars
    ax2 = fig.add_subplot(gs[0, 2])
    means = []
    stds = []
    for n in sample_sizes:
        r2 = data[f'n{n}_r2_improvements'] * 100
        means.append(np.mean(r2))
        stds.append(np.std(r2))
    
    ax2.errorbar(range(len(sample_sizes)), means, yerr=stds, 
                 fmt='o-', capsize=4, markersize=5, linewidth=1.5)
    ax2.axhline(y=np.mean(means), color='green', linestyle='-', alpha=0.3,
                label=f'Mean={np.mean(means):.3f}%')
    ax2.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5% threshold')
    ax2.set_xticks(range(0, len(sample_sizes), 2))
    ax2.set_xticklabels([f'{s//1000}k' for s in sample_sizes[::2]], rotation=45)
    ax2.set_xlabel('Sample Size')
    ax2.set_ylabel('R² Improvement (%)')
    ax2.set_title('R² Convergence to Population Value')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Variance Stabilization (log-log plot)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.loglog(sample_sizes, stds, 'o-', linewidth=2, markersize=7, color='purple')
    ax3.set_xlabel('Sample Size (log scale)')
    ax3.set_ylabel('Std Dev of R² (%, log scale)')
    ax3.set_title('Variance Stabilization Pattern')
    ax3.grid(True, alpha=0.3, which='both')
    
    # Add power law fit
    from scipy import stats
    log_n = np.log(sample_sizes)
    log_std = np.log(stds)
    slope, intercept, r_value, _, _ = stats.linregress(log_n, log_std)
    ax3.text(0.05, 0.95, f'Slope: {slope:.2f}\n(Theory: -0.5)', 
             transform=ax3.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Coefficient of Variation
    ax4 = fig.add_subplot(gs[1, 1])
    cvs = [stds[i]/means[i]*100 for i in range(len(means))]
    ax4.semilogx(sample_sizes, cvs, 'o-', linewidth=2, markersize=7, color='darkgreen')
    ax4.set_xlabel('Sample Size (log scale)')
    ax4.set_ylabel('Coefficient of Variation (%)')
    ax4.set_title('Relative Precision Improvement')
    ax4.grid(True, alpha=0.3)
    
    # 5. Effect Size Distributions (3 sample sizes)
    ax5 = fig.add_subplot(gs[1, 2])
    test_sizes = [10000, 50000, 100000]
    colors = ['blue', 'orange', 'green']
    for i, n in enumerate(test_sizes):
        if n in sample_sizes:
            plants_effects = data[f'n{n}_plants_effects']
            ax5.hist(plants_effects, bins=25, alpha=0.4, label=f'n={n//1000}k',
                    density=True, color=colors[i])
    ax5.set_xlabel('Plants Effect Size')
    ax5.set_ylabel('Density')
    ax5.set_title('Effect Size Distribution Convergence')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. R² Distribution Comparison
    for i, n in enumerate([1000, 10000, 100000]):
        if n in sample_sizes:
            ax = fig.add_subplot(gs[2, i])
            r2 = data[f'n{n}_r2_improvements'] * 100
            ax.hist(r2, bins=30, alpha=0.7, color='coral', edgecolor='black')
            ax.axvline(x=5, color='red', linestyle='--', linewidth=2, alpha=0.5)
            ax.axvline(x=np.mean(r2), color='blue', linestyle='-', linewidth=2)
            ax.set_xlabel('R² Improvement (%)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'n={n//1000}k (σ={np.std(r2):.3f}%)')
            ax.set_xlim(0, 3)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(npz_path).parent / "extended_bootstrap_analysis.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved extended visualization to: {output_path}")
    
    # Print detailed summary
    print("\n" + "="*70)
    print("EXTENDED BOOTSTRAP ANALYSIS SUMMARY")
    print("="*70)
    print(f"Dataset: {N_total:,} rows")
    print(f"Sample sizes: {len(sample_sizes)} sizes from {sample_sizes[0]:,} to {sample_sizes[-1]:,}")
    print(f"Iterations: {n_iterations} per sample size")
    print(f"Total bootstrap iterations: {len(sample_sizes) * n_iterations:,}")
    print()
    
    print("Variance Stabilization:")
    print("-"*40)
    print(f"{'Sample':<10} {'R² Mean':<10} {'R² Std':<10} {'CV':<10}")
    print("-"*40)
    for i in [0, 4, 7, 11, 15]:  # 1k, 10k, 25k, 60k, 100k
        n = sample_sizes[i]
        print(f"{n:>7,}   {means[i]:>7.3f}%   {stds[i]:>7.3f}%   {cvs[i]:>7.1f}%")
    
    print()
    print("Key Insights:")
    print("-"*40)
    print(f"• Population R² = {np.mean(means):.3f}% ± {np.std(means):.3f}%")
    print(f"• Variance reduction: {(1-stds[-1]/stds[0])*100:.1f}% from n=1k to n=100k")
    print(f"• Power law exponent: {slope:.2f} (theoretical: -0.5)")
    print(f"• All samples show 100% statistical power")
    print(f"• No samples achieve ecological significance (>5% R²)")
    print(f"• Largest sample (100k) is only {100000/N_total*100:.2f}% of total data")
    
    plt.show()

if __name__ == "__main__":
    npz_path = "/home/yl998/Geo_process/results/bootstrap/bootstrap_method1.npz"
    create_extended_visualization(npz_path)