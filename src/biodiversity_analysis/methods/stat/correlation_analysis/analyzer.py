#!/usr/bin/env python
"""
Correlation Explorer: Comprehensive correlation analysis for biodiversity richness data.
Implements multiple correlation methods with clear explanations of assumptions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='sklearn')

# Import unified data utilities
from src.biodiversity_analysis.data_utils import prepare_for_correlation_analysis


class CorrelationExplorer:
    """Explore correlations between biodiversity datasets using multiple methods"""
    
    def __init__(self, output_dir: str = "outputs/biodiversity_analysis/correlations"):
        """Initialize correlation explorer"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_and_prepare_data(self, datasets: Dict[str, str], sample_factor: int = 1) -> Tuple[pd.DataFrame, Dict]:
        """Load datasets and prepare for correlation analysis using unified utilities"""
        print("Loading and preparing data...")
        
        # Use unified data preparation function
        flattened_data, metadata = prepare_for_correlation_analysis(
            datasets, sample_factor=sample_factor, min_value=0.0
        )
        
        # Convert to DataFrame
        df_clean = pd.DataFrame(flattened_data)
        
        print(f"Data shape: {df_clean.shape}")
        print(f"Samples with valid data: {len(df_clean):,}")
        print(f"Coverage: {metadata['coverage_percent']:.1f}%")
        
        return df_clean, metadata
    
    def explain_correlation_methods(self) -> None:
        """Create educational figure explaining different correlation methods"""
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle("Understanding Correlation Methods for Biodiversity Analysis", 
                     fontsize=16, fontweight='bold')
        
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.25)
        
        # 1. Pearson Correlation
        ax1 = fig.add_subplot(gs[0, :])
        self._explain_pearson(ax1)
        
        # 2. Spearman Correlation
        ax2 = fig.add_subplot(gs[1, :])
        self._explain_spearman(ax2)
        
        # 3. Kendall Correlation
        ax3 = fig.add_subplot(gs[2, :])
        self._explain_kendall(ax3)
        
        # 4. Multiple Correlation Methods
        ax4 = fig.add_subplot(gs[3, :])
        self._explain_multiple_correlation(ax4)
        
        plt.savefig(self.output_dir / "correlation_methods_explained.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved: correlation_methods_explained.png")
    
    def _explain_pearson(self, ax: plt.Axes) -> None:
        """Explain Pearson correlation"""
        ax.axis('off')
        
        text = """
PEARSON CORRELATION (r)
========================
WHAT IT MEASURES: Linear relationship between variables
FORMULA: r = Σ[(xi - x̄)(yi - ȳ)] / sqrt[Σ(xi - x̄)² × Σ(yi - ȳ)²]
RANGE: -1 to +1 (negative to positive linear relationship)

ASSUMPTIONS:
✓ Variables are continuous
✓ Linear relationship exists
✓ Data is normally distributed (for significance testing)
✓ Homoscedasticity (constant variance)
✓ No significant outliers

WHEN TO USE:
• Comparing variables with similar distributions
• When you expect linear relationships
• Example: Plant richness vs Vertebrate richness (both observed counts)

LIMITATIONS:
✗ Sensitive to outliers
✗ Only detects linear relationships (misses curves)
✗ Assumes equal intervals in data
✗ May be misleading for SDM vs observed data (different scales)

BIODIVERSITY CONTEXT:
Good for: Comparing datasets from same method (observed vs observed, SDM vs SDM)
Poor for: Comparing different data types (SDM predictions vs observed counts)
"""
        ax.text(0.02, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
    
    def _explain_spearman(self, ax: plt.Axes) -> None:
        """Explain Spearman correlation"""
        ax.axis('off')
        
        text = """
SPEARMAN CORRELATION (ρ or rho)
================================
WHAT IT MEASURES: Monotonic relationship (ranks) between variables
FORMULA: Same as Pearson but applied to ranks, not raw values
RANGE: -1 to +1 (perfect negative to positive monotonic relationship)

ASSUMPTIONS:
✓ Variables are ordinal, interval, or ratio
✓ Monotonic relationship exists (not necessarily linear)
✓ NO assumption about data distribution
✓ Paired observations

WHEN TO USE:
• Non-normally distributed data
• Presence of outliers
• Comparing different data types (SDM vs observed)
• When relationship might be non-linear but monotonic

LIMITATIONS:
✗ Less powerful than Pearson for truly linear relationships
✗ Still assumes monotonic relationship
✗ Tied ranks can affect results

BIODIVERSITY CONTEXT:
Good for: Comparing SDM predictions with observed counts (different scales)
Better than Pearson for: Data with different distributions or outliers
Example: AM Fungi SDM (0-100 scale) vs Plants observed (0-5000+ species)

WHY WE USED THIS: More robust for comparing different data types
"""
        ax.text(0.02, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
    
    def _explain_kendall(self, ax: plt.Axes) -> None:
        """Explain Kendall correlation"""
        ax.axis('off')
        
        text = """
KENDALL'S TAU (τ)
==================
WHAT IT MEASURES: Concordance between rankings (agreement in order)
FORMULA: τ = (concordant pairs - discordant pairs) / total pairs
RANGE: -1 to +1 (complete disagreement to complete agreement)

ASSUMPTIONS:
✓ Minimal - only requires ordinal data
✓ No distribution assumptions
✓ Robust to outliers
✓ Works with tied ranks

WHEN TO USE:
• Small sample sizes (more reliable than Spearman)
• Many tied ranks in data
• When you want most conservative estimate
• Ordinal data or categories

LIMITATIONS:
✗ Less statistically efficient (lower power)
✗ More computationally intensive
✗ Generally gives smaller values than Spearman

BIODIVERSITY CONTEXT:
Good for: Small regional studies, presence/absence data
Conservative choice: When unsure about data properties
Example: Comparing biodiversity hotspot rankings between methods

INTERPRETATION: |τ| < 0.3 = weak, 0.3-0.5 = moderate, > 0.5 = strong
"""
        ax.text(0.02, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
    
    def _explain_multiple_correlation(self, ax: plt.Axes) -> None:
        """Explain multiple correlation methods"""
        ax.axis('off')
        
        text = """
MULTIPLE CORRELATION METHODS (>2 variables)
============================================

1. PAIRWISE CORRELATIONS (what we currently use)
   • Calculate correlation between each pair
   • Results in correlation matrix
   • Simple but ignores joint relationships

2. PARTIAL CORRELATION
   • Correlation between X and Y controlling for Z
   • Formula: r(XY.Z) = [r(XY) - r(XZ)×r(YZ)] / sqrt[(1-r²(XZ))(1-r²(YZ))]
   • Shows "pure" relationship removing confounders
   • Example: Plants~Fungi controlling for climate

3. CANONICAL CORRELATION ANALYSIS (CCA)
   • Finds linear combinations that maximize correlation
   • Can handle two SETS of variables
   • Example: (Plants, Vertebrates) vs (AM Fungi, EcM Fungi)
   • Good for SDM vs Observed comparisons as groups

4. MULTIPLE CORRELATION COEFFICIENT (R)
   • Correlation between one variable and best linear combination of others
   • R² tells how much variance is explained by all others combined
   • Example: How well do Plants+AM+EcM predict Vertebrates?

RECOMMENDATION FOR 4 DATASETS:
• Use correlation matrix for overview (6 pairs)
• Calculate partial correlations to remove confounding
• Try CCA to compare observed (plants+vertebrates) vs SDM (fungi) as groups
• Report multiple R² to show predictability
"""
        ax.text(0.02, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
    
    def calculate_all_correlations(self, df: pd.DataFrame) -> Dict:
        """Calculate all types of correlations"""
        results = {}
        
        print("\n" + "="*60)
        print("CALCULATING ALL CORRELATION TYPES")
        print("="*60)
        
        # 1. Pearson correlations
        print("\n1. PEARSON CORRELATION (Linear relationships)")
        print("-" * 40)
        pearson_matrix = df.corr(method='pearson')
        results['pearson'] = pearson_matrix
        
        for col1 in df.columns:
            for col2 in df.columns:
                if col1 < col2:  # Upper triangle only
                    r = pearson_matrix.loc[col1, col2]
                    # Calculate p-value
                    _, p = stats.pearsonr(df[col1], df[col2])
                    print(f"  {col1} ~ {col2}: r = {r:.3f} (p = {p:.3e})")
        
        # 2. Spearman correlations
        print("\n2. SPEARMAN CORRELATION (Rank-based, robust)")
        print("-" * 40)
        spearman_matrix = df.corr(method='spearman')
        results['spearman'] = spearman_matrix
        
        for col1 in df.columns:
            for col2 in df.columns:
                if col1 < col2:
                    rho = spearman_matrix.loc[col1, col2]
                    _, p = stats.spearmanr(df[col1], df[col2])
                    print(f"  {col1} ~ {col2}: ρ = {rho:.3f} (p = {p:.3e})")
        
        # 3. Kendall correlations
        print("\n3. KENDALL'S TAU (Most conservative)")
        print("-" * 40)
        kendall_matrix = df.corr(method='kendall')
        results['kendall'] = kendall_matrix
        
        for col1 in df.columns:
            for col2 in df.columns:
                if col1 < col2:
                    tau = kendall_matrix.loc[col1, col2]
                    _, p = stats.kendalltau(df[col1], df[col2])
                    print(f"  {col1} ~ {col2}: τ = {tau:.3f} (p = {p:.3e})")
        
        # 4. Partial correlations (example: controlling for one variable)
        print("\n4. PARTIAL CORRELATIONS (Controlling for confounders)")
        print("-" * 40)
        
        # Example: Plants~AM_fungi controlling for animals
        if all(col in df.columns for col in ['plants', 'am_fungi', 'animals']):
            partial_r = self._partial_correlation(df, 'plants', 'am_fungi', 'animals')
            results['partial_example'] = partial_r
            print(f"  Plants ~ AM_fungi | animals: r = {partial_r:.3f}")
            
            # Compare with simple correlation
            simple_r = pearson_matrix.loc['plants', 'am_fungi']
            print(f"  (Simple correlation was: {simple_r:.3f})")
            print(f"  Difference: {abs(partial_r - simple_r):.3f}")
        
        # 5. Multiple correlation (one vs all others)
        print("\n5. MULTIPLE CORRELATION (One variable vs all others)")
        print("-" * 40)
        
        multiple_r = {}
        for target in df.columns:
            # Predict target from all others
            X = df.drop(columns=[target])
            y = df[target]
            
            # Standardize
            X_std = StandardScaler().fit_transform(X)
            y_std = StandardScaler().fit_transform(y.values.reshape(-1, 1)).flatten()
            
            # Calculate R²
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X_std, y_std)
            r_squared = model.score(X_std, y_std)
            multiple_r[target] = np.sqrt(r_squared)
            
            print(f"  {target} ~ all others: R = {np.sqrt(r_squared):.3f} (R² = {r_squared:.3f})")
        
        results['multiple_r'] = multiple_r
        
        # 6. Canonical correlation (if we have clear groups)
        print("\n6. CANONICAL CORRELATION ANALYSIS")
        print("-" * 40)
        
        # Group: Observed (plants, terrestrial) vs SDM (fungi)
        observed_cols = [col for col in df.columns if 'fungi' not in col]
        sdm_cols = [col for col in df.columns if 'fungi' in col]
        
        if len(observed_cols) >= 2 and len(sdm_cols) >= 2:
            X = df[observed_cols].values
            Y = df[sdm_cols].values
            
            # Standardize
            X = StandardScaler().fit_transform(X)
            Y = StandardScaler().fit_transform(Y)
            
            # Fit CCA
            cca = CCA(n_components=min(len(observed_cols), len(sdm_cols)))
            cca.fit(X, Y)
            
            # Get canonical correlations
            X_c, Y_c = cca.transform(X, Y)
            canonical_corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] 
                              for i in range(X_c.shape[1])]
            
            results['canonical'] = canonical_corrs
            
            print(f"  Observed group: {observed_cols}")
            print(f"  SDM group: {sdm_cols}")
            for i, corr in enumerate(canonical_corrs):
                print(f"  Canonical correlation {i+1}: {corr:.3f}")
        
        return results
    
    def _partial_correlation(self, df: pd.DataFrame, x: str, y: str, z: str) -> float:
        """Calculate partial correlation between x and y controlling for z"""
        # Simple correlations
        r_xy = df[x].corr(df[y])
        r_xz = df[x].corr(df[z])
        r_yz = df[y].corr(df[z])
        
        # Partial correlation formula
        numerator = r_xy - (r_xz * r_yz)
        denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    def visualize_correlation_comparison(self, results: Dict) -> None:
        """Create comprehensive visualization comparing all correlation methods"""
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle("Correlation Analysis Comparison: Multiple Methods", 
                     fontsize=16, fontweight='bold')
        
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Pearson correlation matrix
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(results['pearson'], annot=True, fmt='.2f', cmap='RdBu_r', 
                   vmin=-1, vmax=1, ax=ax1, cbar_kws={'label': 'Pearson r'})
        ax1.set_title("Pearson Correlation\n(Linear relationships)")
        
        # 2. Spearman correlation matrix
        ax2 = fig.add_subplot(gs[0, 1])
        sns.heatmap(results['spearman'], annot=True, fmt='.2f', cmap='RdBu_r',
                   vmin=-1, vmax=1, ax=ax2, cbar_kws={'label': 'Spearman ρ'})
        ax2.set_title("Spearman Correlation\n(Rank-based, robust)")
        
        # 3. Kendall correlation matrix
        ax3 = fig.add_subplot(gs[0, 2])
        sns.heatmap(results['kendall'], annot=True, fmt='.2f', cmap='RdBu_r',
                   vmin=-1, vmax=1, ax=ax3, cbar_kws={'label': 'Kendall τ'})
        ax3.set_title("Kendall's Tau\n(Most conservative)")
        
        # 4. Method comparison
        ax4 = fig.add_subplot(gs[1, :])
        self._plot_method_comparison(results, ax4)
        
        # 5. Multiple R values
        ax5 = fig.add_subplot(gs[2, 0])
        if 'multiple_r' in results:
            multiple_r = results['multiple_r']
            vars = list(multiple_r.keys())
            values = list(multiple_r.values())
            colors = ['green' if v > 0.7 else 'orange' if v > 0.4 else 'red' for v in values]
            
            bars = ax5.bar(range(len(vars)), values, color=colors)
            ax5.set_xticks(range(len(vars)))
            ax5.set_xticklabels(vars, rotation=45, ha='right')
            ax5.set_ylabel("Multiple R")
            ax5.set_title("Multiple Correlation\n(Each variable vs all others)")
            ax5.set_ylim(0, 1)
            ax5.axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='Strong')
            ax5.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Moderate')
            ax5.legend()
            
            # Add values on bars
            for bar, val in zip(bars, values):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2f}', ha='center', va='bottom')
        
        # 6. Canonical correlations
        ax6 = fig.add_subplot(gs[2, 1])
        if 'canonical' in results:
            canonical = results['canonical']
            ax6.bar(range(len(canonical)), canonical, color='purple')
            ax6.set_xlabel("Component")
            ax6.set_ylabel("Canonical Correlation")
            ax6.set_title("Canonical Correlation Analysis\n(Observed vs SDM groups)")
            ax6.set_ylim(0, 1)
            ax6.set_xticks(range(len(canonical)))
            ax6.set_xticklabels([f"CC{i+1}" for i in range(len(canonical))])
            
            for i, val in enumerate(canonical):
                ax6.text(i, val + 0.02, f'{val:.3f}', ha='center', va='bottom')
        
        # 7. Interpretation guide
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        interpretation = """
INTERPRETATION GUIDE
====================

CORRELATION STRENGTH:
• 0.0-0.3: Weak/None
• 0.3-0.5: Moderate  
• 0.5-0.7: Strong
• 0.7-1.0: Very Strong

KEY FINDINGS:
✓ Spearman often differs from
  Pearson for SDM vs observed
✓ Kendall gives most 
  conservative estimates
✓ Multiple R shows how well
  other variables predict each
✓ Canonical correlation shows
  group-level relationships

NEGATIVE CORRELATIONS:
⚠ EcM~Plants negative = 
  ECOLOGICAL IMPOSSIBILITY
  (indicates data problems)
"""
        ax7.text(0.05, 0.95, interpretation, transform=ax7.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))
        
        plt.savefig(self.output_dir / "correlation_comparison_all_methods.png",
                   dpi=150, bbox_inches='tight')
        plt.close()
        print("\nSaved: correlation_comparison_all_methods.png")
    
    def _plot_method_comparison(self, results: Dict, ax: plt.Axes) -> None:
        """Compare correlation values across methods"""
        # Extract pairwise correlations for comparison
        pearson = results['pearson']
        spearman = results['spearman']
        kendall = results['kendall']
        
        # Get pairs
        pairs = []
        pearson_vals = []
        spearman_vals = []
        kendall_vals = []
        
        for col1 in pearson.columns:
            for col2 in pearson.columns:
                if col1 < col2:
                    pair_name = f"{col1[:3]}~{col2[:3]}"
                    pairs.append(pair_name)
                    pearson_vals.append(pearson.loc[col1, col2])
                    spearman_vals.append(spearman.loc[col1, col2])
                    kendall_vals.append(kendall.loc[col1, col2])
        
        x = np.arange(len(pairs))
        width = 0.25
        
        ax.bar(x - width, pearson_vals, width, label='Pearson', color='blue', alpha=0.7)
        ax.bar(x, spearman_vals, width, label='Spearman', color='green', alpha=0.7)
        ax.bar(x + width, kendall_vals, width, label='Kendall', color='red', alpha=0.7)
        
        ax.set_xlabel('Dataset Pairs')
        ax.set_ylabel('Correlation Coefficient')
        ax.set_title('Correlation Method Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(pairs, rotation=45, ha='right')
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        
        # Highlight negative correlations
        for i, (p, s, k) in enumerate(zip(pearson_vals, spearman_vals, kendall_vals)):
            if p < 0 or s < 0 or k < 0:
                ax.axvspan(i-0.4, i+0.4, alpha=0.2, color='red')


def main():
    """Run comprehensive correlation analysis"""
    # Use StatisticalDataPaths for dataset configuration
    paths_config = StatisticalDataPaths()
    datasets = {}
    
    # Get all available datasets from configuration
    for name, dataset_config in paths_config.datasets.items():
        path = dataset_config.get_valid_path()
        if path is not None:
            datasets[name] = str(path)
        else:
            print(f"Warning: No valid path found for {name}")
    
    if not datasets:
        print("Error: No valid datasets found. Check configuration and file paths.")
        return
    
    # Check which files exist
    existing = {}
    for name, path in datasets.items():
        if Path(path).exists():
            existing[name] = path
        else:
            print(f"Warning: Dataset not found: {name} -> {path}")
    
    if len(existing) < 2:
        print("Need at least 2 datasets for correlation analysis")
        return
    
    # Initialize explorer
    explorer = CorrelationExplorer()
    
    # Create educational materials
    print("\nCreating correlation method explanations...")
    explorer.explain_correlation_methods()
    
    # Load and prepare data
    df, metadata = explorer.load_and_prepare_data(existing)
    
    # Calculate all correlations
    results = explorer.calculate_all_correlations(df)
    
    # Visualize comparisons
    explorer.visualize_correlation_comparison(results)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nOutputs saved to: outputs/biodiversity_analysis/correlations/")
    print("\nKey files:")
    print("  - correlation_methods_explained.png: Educational guide")
    print("  - correlation_comparison_all_methods.png: Results comparison")


if __name__ == "__main__":
    main()