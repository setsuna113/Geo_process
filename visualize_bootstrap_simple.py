#!/usr/bin/env python3
"""
Simple bootstrap visualization for our actual NPZ format.
Creates 4-panel analysis plot showing bootstrap results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_and_visualize(npz_path):
    """Load NPZ and create comprehensive visualization."""
    
    # Load data
    data = np.load(npz_path)
    
    # Extract basic info
    sample_sizes = data['sample_sizes']
    n_iterations = int(data['n_iterations'])
    optimal_n = int(data['optimal_n'])
    N_total = int(data['N_total'])
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Bootstrap Analysis Results (N={N_total:,}, {n_iterations} iterations)', 
                 fontsize=14, fontweight='bold')
    
    # 1. Power vs Sample Size
    ax = axes[0, 0]
    powers = [data[f'n{n}_power'] for n in sample_sizes]
    eco_powers = [data[f'n{n}_eco_power'] for n in sample_sizes]
    
    ax.plot(sample_sizes, powers, 'o-', label='Statistical Power (p<0.05)', 
            linewidth=2, markersize=8)
    ax.plot(sample_sizes, eco_powers, 's-', label='Ecological Power (p<0.05 & R²>5%)', 
            linewidth=2, markersize=8)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='80% threshold')
    ax.axvline(x=optimal_n, color='green', linestyle='--', alpha=0.5, 
               label=f'Optimal n={optimal_n:,}')
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Power')
    ax.set_title('Statistical vs Ecological Power')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # 2. Effect Size Distributions (for n=10000)
    ax = axes[0, 1]
    test_n = 10000
    plants_effects = data[f'n{test_n}_plants_effects']
    vertebrates_effects = data[f'n{test_n}_vertebrates_effects']
    
    ax.hist(plants_effects, bins=30, alpha=0.5, label='Plants', density=True)
    ax.hist(vertebrates_effects, bins=30, alpha=0.5, label='Vertebrates', density=True)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Effect Size (log scale)')
    ax.set_ylabel('Density')
    ax.set_title(f'Effect Size Distributions (n={test_n:,})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Variance Stabilization
    ax = axes[1, 0]
    plants_stds = [data[f'n{n}_plants_std'] for n in sample_sizes]
    
    ax.plot(sample_sizes, plants_stds, 'o-', linewidth=2, markersize=8, 
            color='forestgreen')
    ax.axvline(x=optimal_n, color='green', linestyle='--', alpha=0.5,
               label=f'Optimal n={optimal_n:,}')
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Variance Stabilization (Plants Effect)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # 4. R² Improvement Distribution
    ax = axes[1, 1]
    r2_improvements = data[f'n{test_n}_r2_improvements'] * 100  # Convert to percentage
    
    ax.hist(r2_improvements, bins=30, alpha=0.7, color='coral', edgecolor='black')
    ax.axvline(x=5, color='red', linestyle='--', linewidth=2, 
               label='5% threshold')
    ax.set_xlabel('R² Improvement (%)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'R² Improvement Distribution (n={test_n:,})')
    
    # Add text box with statistics
    mean_r2 = np.mean(r2_improvements)
    std_r2 = np.std(r2_improvements)
    above_5 = np.sum(r2_improvements > 5) / len(r2_improvements) * 100
    
    textstr = f'Mean: {mean_r2:.2f}%\nStd: {std_r2:.2f}%\n>5%: {above_5:.1f}%'
    ax.text(0.7, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(npz_path).parent / f"{Path(npz_path).stem}_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("BOOTSTRAP ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total dataset size: {N_total:,} rows")
    print(f"Sample sizes tested: {list(sample_sizes)}")
    print(f"Iterations per sample: {n_iterations}")
    print(f"Optimal sample size: {optimal_n:,}")
    print("\nKey Findings:")
    print(f"• Statistical power at n=1000: {data['n1000_power']:.1%}")
    print(f"• Ecological power at n=25000: {data['n25000_eco_power']:.1%}")
    print(f"• Mean R² improvement: {mean_r2:.2f}%")
    print(f"• Ecological significance achieved: {'No' if above_5 == 0 else 'Yes'}")
    
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        npz_path = sys.argv[1]
    else:
        npz_path = "results/bootstrap/bootstrap_method1.npz"
    
    load_and_visualize(npz_path)