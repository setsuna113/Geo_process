from src.abstractions import get_logger
"""DML Effect Interpretation with Publication-Quality Visualizations.

This module provides standalone interpretation methods for DML causal effects,
separate from hypothesis testing. All visualizations follow publication standards.

References:
- VanderWeele (2013): Three-way decomposition of causal effects
- Aguinis & Murphy (2022): Reporting interaction effects
- Forest plots: Publication standards from forestplot package
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import logging

# Set publication quality defaults
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

logger = get_logger(__name__)


class DMLEffectInterpreter:
    """Publication-quality interpretation of DML causal effects.
    
    This class provides pure interpretation methods (no hypothesis testing).
    All visualizations follow publication standards with proper effect sizes,
    confidence intervals, and decompositions.
    """
    
    def __init__(self, 
                 coefficients: Union[List[float], np.ndarray] = None,
                 std_errors: Union[List[float], np.ndarray] = None,
                 coef_names: List[str] = None,
                 n_obs: int = None,
                 summary_stats: Optional[Dict[str, Any]] = None,
                 julia_results: Optional[Dict[str, Any]] = None):
        """Initialize DML interpreter with results.
        
        Args:
            coefficients: DML theta estimates [θ₁, θ₂, θ₃] (optional if julia_results provided)
            std_errors: Standard errors for coefficients (optional if julia_results provided)
            coef_names: Names of coefficients (optional if julia_results provided)
            n_obs: Number of observations (optional if julia_results provided)
            summary_stats: Optional dict with means, quantiles, variances
            julia_results: Complete Julia DML results dict with pre-computed statistics
        """
        if julia_results is not None:
            # Use pre-computed Julia statistics
            self._load_from_julia(julia_results)
        else:
            # Legacy initialization
            if coefficients is None or std_errors is None:
                raise ValueError("Either provide julia_results or (coefficients, std_errors)")
            self.coefficients = np.array(coefficients)
            self.std_errors = np.array(std_errors)
            self.coef_names = coef_names or [f'coef_{i}' for i in range(len(coefficients))]
            self.n_obs = n_obs or 0
            self.summary_stats = summary_stats or {}
            
            # Calculate confidence intervals (95%) - only if not from Julia
            self.ci_lower = self.coefficients - 1.96 * self.std_errors
            self.ci_upper = self.coefficients + 1.96 * self.std_errors
            self.z_statistics = self.coefficients / self.std_errors
            self.p_values = 2 * (1 - stats.norm.cdf(np.abs(self.z_statistics)))
            
            # No pre-computed Julia stats available
            self.julia_stats = None
    
    def _load_from_julia(self, julia_results: Dict[str, Any]):
        """Load statistics directly from Julia results.
        
        Args:
            julia_results: Julia DML results dictionary
        """
        # Extract basic arrays - these are ALWAYS in correct order
        self.coefficients = np.array(julia_results['coefficients'])
        self.std_errors = np.array(julia_results['std_errors'])
        self.coef_names = julia_results.get('coef_names', julia_results.get('treatment_names', []))
        self.n_obs = julia_results['n_obs']
        self.vcov = np.array(julia_results.get('vcov', []))
        
        # Use pre-computed statistics from treatments dict if available
        if 'treatments' in julia_results and julia_results['treatments']:
            # Julia treatments dict keys might not match order of coef_names
            # Build ordered arrays matching coef_names order
            self.ci_lower = []
            self.ci_upper = []
            self.z_statistics = []
            self.p_values = []
            self.significant_flags = []
            
            for i, name in enumerate(self.coef_names):
                if name in julia_results['treatments']:
                    # Use Julia pre-computed values
                    treat_stats = julia_results['treatments'][name]
                    self.ci_lower.append(treat_stats['ci_95'][0])
                    self.ci_upper.append(treat_stats['ci_95'][1])
                    self.z_statistics.append(treat_stats['z_statistic'])
                    self.p_values.append(treat_stats['p_value'])
                    self.significant_flags.append(treat_stats['statistically_significant'])
                else:
                    # Fallback for missing entry (shouldn't happen)
                    coef = self.coefficients[i]
                    se = self.std_errors[i]
                    z = coef / se if se > 0 else 0
                    p = 2 * (1 - stats.norm.cdf(abs(z)))
                    self.ci_lower.append(coef - 1.96 * se)
                    self.ci_upper.append(coef + 1.96 * se)
                    self.z_statistics.append(z)
                    self.p_values.append(p)
                    self.significant_flags.append(p < 0.05)
            
            self.ci_lower = np.array(self.ci_lower)
            self.ci_upper = np.array(self.ci_upper)
            self.z_statistics = np.array(self.z_statistics)
            self.p_values = np.array(self.p_values)
            self.significant_flags = np.array(self.significant_flags)
        else:
            # No treatments dict - compute everything
            self.z_statistics = np.divide(self.coefficients, self.std_errors, 
                                         out=np.zeros_like(self.coefficients), 
                                         where=self.std_errors!=0)
            self.p_values = 2 * (1 - stats.norm.cdf(np.abs(self.z_statistics)))
            self.ci_lower = self.coefficients - 1.96 * self.std_errors
            self.ci_upper = self.coefficients + 1.96 * self.std_errors
            self.significant_flags = self.p_values < 0.05
        
        # Store model-level statistics
        self.model_stats = julia_results.get('model_stats', {})
        self.joint_test = self.model_stats.get('joint_test', None)
        self.condition_number = self.model_stats.get('condition_number', None)
        
        # Store BDML stats if available
        self.bdml_stats = julia_results.get('bdml_stats', None)
        
        # Store full Julia results for reference
        self.julia_stats = julia_results
        self.summary_stats = julia_results.get('summary_stats', {})
    
    def effect_size_plot(self, 
                        figsize: Tuple[float, float] = (8, 6),
                        colors: Optional[List[str]] = None,
                        title: Optional[str] = None,
                        save_path: Optional[Path] = None) -> plt.Figure:
        """Create publication-quality effect size bar plot.
        
        Shows magnitude of causal effects with confidence intervals.
        Follows publication standards with proper labeling and formatting.
        
        Args:
            figsize: Figure size in inches
            colors: Custom colors for bars (default: colorblind-friendly)
            title: Custom title (default: automatic)
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        # Default colorblind-friendly colors
        if colors is None:
            colors = ['#0173B2', '#DE8F05', '#CC78BC']  # Blue, Orange, Purple
        
        # Create figure with publication settings
        fig, ax = plt.subplots(figsize=figsize)
        
        # Positions for bars
        x_pos = np.arange(len(self.coefficients))
        
        # Create bars with confidence intervals
        bars = ax.bar(x_pos, self.coefficients, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add error bars using pre-computed CIs if available
        if hasattr(self, 'ci_lower') and hasattr(self, 'ci_upper'):
            # Use pre-computed confidence intervals
            yerr_lower = self.coefficients - self.ci_lower
            yerr_upper = self.ci_upper - self.coefficients
            yerr = [yerr_lower, yerr_upper]
        else:
            # Fallback to computing from SE
            yerr = 1.96 * self.std_errors
        
        ax.errorbar(x_pos, self.coefficients, 
                   yerr=yerr,
                   fmt='none', color='black', capsize=4, capthick=1.5, linewidth=1.5)
        
        # Customize axes
        ax.set_xticks(x_pos)
        ax.set_xticklabels(self._format_coef_names(), rotation=0, ha='center')
        ax.set_ylabel('Causal Effect Estimate', fontweight='semibold')
        ax.set_xlabel('Treatment Variables', fontweight='semibold')
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # Add value labels on bars
        for i, (bar, coef, se) in enumerate(zip(bars, self.coefficients, self.std_errors)):
            height = bar.get_height()
            # Position text above or below bar depending on sign
            y_pos = height + 1.5 * se if height > 0 else height - 1.5 * se
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{coef:.4f}',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=8, fontweight='bold')
        
        # Add title
        title = title or 'DML Causal Effect Estimates'
        ax.set_title(title, fontweight='bold', pad=20)
        
        # Add sample size annotation
        ax.text(0.02, 0.98, f'n = {self.n_obs:,}', 
               transform=ax.transAxes, fontsize=8,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Grid for easier reading
        ax.yaxis.grid(True, linestyle=':', alpha=0.3)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Effect size plot saved to {save_path}")
        
        return fig
    
    def simplified_effect_functions(self,
                                  figsize: Tuple[float, float] = (10, 4),
                                  save_path: Optional[Path] = None) -> plt.Figure:
        """Visualize simplified effect equations.
        
        Shows how effects combine mathematically:
        - Effect(plants) = θ₁ + θ₂ × vertebrates_mean
        - Effect(vertebrates) = θ₃ + θ₂ × plants_mean
        
        Args:
            figsize: Figure size in inches
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        # Get means from summary stats or use defaults
        plants_mean = self.summary_stats.get('means', {}).get('plants', 0.5)
        vertebrates_mean = self.summary_stats.get('means', {}).get('vertebrates', 0.5)
        
        # Calculate simplified effects
        plant_effect = self.coefficients[0] + self.coefficients[1] * vertebrates_mean
        vertebrate_effect = self.coefficients[2] + self.coefficients[1] * plants_mean
        
        # Create figure with two panels
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Panel 1: Effect equations as text
        ax1.axis('off')
        equations_text = (
            r'$\mathbf{Simplified\ Effect\ Functions}$' + '\n\n' +
            r'$\mathrm{Effect}_{plants} = \theta_1 + \theta_2 \times \overline{vertebrates}$' + '\n' +
            f'$= {self.coefficients[0]:.4f} + ({self.coefficients[1]:.6f}) \\times {vertebrates_mean:.2f}$\n' +
            f'$= {plant_effect:.4f}$\n\n' +
            r'$\mathrm{Effect}_{vertebrates} = \theta_3 + \theta_2 \times \overline{plants}$' + '\n' +
            f'$= {self.coefficients[2]:.4f} + ({self.coefficients[1]:.6f}) \\times {plants_mean:.2f}$\n' +
            f'$= {vertebrate_effect:.4f}$\n\n' +
            r'$\mathrm{Interaction}\ \mathrm{term}:\ \theta_2 = $' + f'{self.coefficients[1]:.6f}'
        )
        
        ax1.text(0.1, 0.5, equations_text, fontsize=11, 
                verticalalignment='center', family='monospace')
        
        # Panel 2: Visual representation
        effects = [plant_effect, vertebrate_effect, self.coefficients[1]]
        labels = ['Plants Effect\n(at mean vertebrates)', 
                 'Vertebrates Effect\n(at mean plants)', 
                 'Interaction']
        colors = ['#2E7D32', '#1565C0', '#E65100']  # Green, Blue, Orange
        
        bars = ax2.bar(range(3), effects, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_xticks(range(3))
        ax2.set_xticklabels(labels, fontsize=9)
        ax2.set_ylabel('Effect Size', fontweight='semibold')
        ax2.set_title('Decomposed Effects at Mean Values', fontweight='bold')
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # Add value labels
        for bar, effect in zip(bars, effects):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{effect:.5f}',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=8, fontweight='bold')
        
        # Remove spines
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.yaxis.grid(True, linestyle=':', alpha=0.3)
        ax2.set_axisbelow(True)
        
        plt.suptitle('DML Effect Function Decomposition', fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Effect functions plot saved to {save_path}")
        
        return fig
    
    def marginal_effects_plot(self,
                             representative_values: Optional[Dict[str, List[float]]] = None,
                             figsize: Tuple[float, float] = (10, 6),
                             save_path: Optional[Path] = None) -> plt.Figure:
        """Plot marginal effects at representative values.
        
        Shows how effects change at different levels of covariates.
        
        Args:
            representative_values: Dict with 'plants' and 'vertebrates' values
                                 (default: uses quantiles [0.1, 0.25, 0.5, 0.75, 0.9])
            figsize: Figure size in inches
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        # Default representative values from quantiles
        if representative_values is None:
            if 'quantiles' in self.summary_stats:
                quantiles = self.summary_stats['quantiles']
                representative_values = {
                    'plants': quantiles.get('plants', [0.1, 0.3, 0.5, 0.7, 0.9]),
                    'vertebrates': quantiles.get('vertebrates', [0.1, 0.3, 0.5, 0.7, 0.9])
                }
            else:
                # Use default values
                representative_values = {
                    'plants': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'vertebrates': [0.1, 0.3, 0.5, 0.7, 0.9]
                }
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Panel 1: Plant effect at different vertebrate levels
        vertebrate_levels = representative_values['vertebrates']
        plant_effects = [self.coefficients[0] + self.coefficients[1] * v for v in vertebrate_levels]
        
        ax1.plot(vertebrate_levels, plant_effects, 'o-', color='#2E7D32', 
                linewidth=2, markersize=8, markeredgecolor='black', markeredgewidth=0.5)
        ax1.set_xlabel('Vertebrate Richness Level', fontweight='semibold')
        ax1.set_ylabel('Marginal Effect of Plants', fontweight='semibold')
        ax1.set_title('Plant Effect Varies with Vertebrate Levels', fontweight='bold')
        ax1.grid(True, linestyle=':', alpha=0.3)
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        
        # Add annotations for key points
        for i, (v, effect) in enumerate(zip(vertebrate_levels, plant_effects)):
            if i == 0 or i == len(vertebrate_levels) - 1 or i == len(vertebrate_levels) // 2:
                ax1.annotate(f'{effect:.4f}', 
                           xy=(v, effect), 
                           xytext=(5, 5),
                           textcoords='offset points',
                           fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
        
        # Panel 2: Vertebrate effect at different plant levels
        plant_levels = representative_values['plants']
        vertebrate_effects = [self.coefficients[2] + self.coefficients[1] * p for p in plant_levels]
        
        ax2.plot(plant_levels, vertebrate_effects, 's-', color='#1565C0',
                linewidth=2, markersize=8, markeredgecolor='black', markeredgewidth=0.5)
        ax2.set_xlabel('Plant Richness Level', fontweight='semibold')
        ax2.set_ylabel('Marginal Effect of Vertebrates', fontweight='semibold')
        ax2.set_title('Vertebrate Effect Varies with Plant Levels', fontweight='bold')
        ax2.grid(True, linestyle=':', alpha=0.3)
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        
        # Add annotations
        for i, (p, effect) in enumerate(zip(plant_levels, vertebrate_effects)):
            if i == 0 or i == len(plant_levels) - 1 or i == len(plant_levels) // 2:
                ax2.annotate(f'{effect:.4f}',
                           xy=(p, effect),
                           xytext=(5, 5),
                           textcoords='offset points',
                           fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
        
        # Remove top and right spines
        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.suptitle('Marginal Effects at Representative Values', fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Marginal effects plot saved to {save_path}")
        
        return fig
    
    def interaction_decomposition(self,
                                 figsize: Tuple[float, float] = (12, 5),
                                 save_path: Optional[Path] = None) -> Tuple[plt.Figure, Dict[str, float]]:
        """Decompose and visualize interaction effects.
        
        Following VanderWeele's three-way decomposition:
        - Direct effects (main effects)
        - Interaction effect
        - Relative contributions
        
        Args:
            figsize: Figure size in inches
            save_path: Path to save figure
            
        Returns:
            Tuple of (Figure, decomposition statistics dict)
        """
        # Calculate variance explained by each component
        # Using simplified variance decomposition (requires covariance info)
        if 'variance_covariance' in self.summary_stats:
            vcov = self.summary_stats['variance_covariance']
            var_plants = vcov.get('plants', 1.0)
            var_vertebrates = vcov.get('vertebrates', 1.0)
            cov_pv = vcov.get('plants_vertebrates', 0.0)
        else:
            # Use default values if not provided
            var_plants = 1.0
            var_vertebrates = 1.0
            cov_pv = 0.2  # Assume small positive correlation
        
        # Calculate variance contributions (simplified)
        # Var(Y) ≈ θ₁²Var(P) + θ₃²Var(V) + θ₂²Var(P×V) + covariance terms
        var_main_plants = self.coefficients[0]**2 * var_plants
        var_main_vertebrates = self.coefficients[2]**2 * var_vertebrates
        var_interaction = self.coefficients[1]**2 * var_plants * var_vertebrates
        
        total_var = var_main_plants + var_main_vertebrates + var_interaction
        
        # Calculate percentages
        pct_plants = (var_main_plants / total_var) * 100
        pct_vertebrates = (var_main_vertebrates / total_var) * 100
        pct_interaction = (var_interaction / total_var) * 100
        
        # Create figure with three panels
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1])
        
        # Panel 1: Pie chart of variance decomposition
        ax1 = fig.add_subplot(gs[0])
        colors = ['#2E7D32', '#1565C0', '#E65100']
        wedges, texts, autotexts = ax1.pie(
            [pct_plants, pct_vertebrates, pct_interaction],
            labels=['Plants\nMain Effect', 'Vertebrates\nMain Effect', 'Interaction'],
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            explode=[0.05, 0.05, 0.1],  # Emphasize interaction
            textprops={'fontsize': 9}
        )
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_color('white')
        
        ax1.set_title('Variance Explained', fontweight='bold', fontsize=11)
        
        # Panel 2: Absolute effects
        ax2 = fig.add_subplot(gs[1])
        abs_effects = np.abs(self.coefficients)
        bars = ax2.bar(range(3), abs_effects, color=colors, alpha=0.7, 
                      edgecolor='black', linewidth=0.5)
        ax2.set_xticks(range(3))
        ax2.set_xticklabels(['Plants', 'Interaction', 'Vertebrates'], rotation=45, ha='right')
        ax2.set_ylabel('|Effect Size|', fontweight='semibold')
        ax2.set_title('Absolute Effect Magnitudes', fontweight='bold', fontsize=11)
        
        # Add value labels
        for bar, effect in zip(bars, abs_effects):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{effect:.5f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Panel 3: Interaction interpretation
        ax3 = fig.add_subplot(gs[2])
        ax3.axis('off')
        
        # Determine interaction interpretation
        if self.coefficients[1] < 0:
            interaction_type = "Antagonistic"
            interpretation = ("Negative interaction:\nPlants and vertebrates\n"
                            "compete for their\neffects on fungi")
        else:
            interaction_type = "Synergistic"
            interpretation = ("Positive interaction:\nPlants and vertebrates\n"
                            "enhance each other's\neffects on fungi")
        
        interpretation_text = (
            f"Interaction Type:\n{interaction_type}\n\n"
            f"Coefficient: {self.coefficients[1]:.6f}\n"
            f"(SE: {self.std_errors[1]:.6f})\n\n"
            f"{interpretation}\n\n"
            f"Explains {pct_interaction:.1f}% of\ntotal variance"
        )
        
        ax3.text(0.5, 0.5, interpretation_text, 
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))
        
        # Clean up axes
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.yaxis.grid(True, linestyle=':', alpha=0.3)
        ax2.set_axisbelow(True)
        
        plt.suptitle('Three-Way Effect Decomposition', fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Prepare statistics dictionary
        stats = {
            'variance_main_plants': var_main_plants,
            'variance_main_vertebrates': var_main_vertebrates,
            'variance_interaction': var_interaction,
            'percent_plants': pct_plants,
            'percent_vertebrates': pct_vertebrates,
            'percent_interaction': pct_interaction,
            'interaction_type': interaction_type
        }
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Interaction decomposition saved to {save_path}")
        
        return fig, stats
    
    def _format_coef_names(self) -> List[str]:
        """Format coefficient names for display."""
        formatted = []
        for name in self.coef_names:
            if 'interaction' in name.lower() or '×' in name or 'x' in name.lower():
                formatted.append('Plants × Vertebrates')
            elif 'plant' in name.lower():
                formatted.append('Plants')
            elif 'vertebrate' in name.lower():
                formatted.append('Vertebrates')
            else:
                formatted.append(name.replace('_', ' ').title())
        return formatted
    
    def create_publication_summary(self,
                                  output_dir: Path,
                                  prefix: str = 'dml') -> Dict[str, Path]:
        """Create all publication-quality visualizations.
        
        Args:
            output_dir: Directory to save figures
            prefix: Prefix for file names
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        
        # Create all visualizations
        logger.info("Creating DML interpretation visualizations...")
        
        # 1. Effect size plot
        fig1 = self.effect_size_plot()
        path1 = output_dir / f'{prefix}_effect_sizes.png'
        fig1.savefig(path1, dpi=300, bbox_inches='tight')
        paths['effect_sizes'] = path1
        plt.close(fig1)
        
        # 2. Simplified effect functions
        fig2 = self.simplified_effect_functions()
        path2 = output_dir / f'{prefix}_effect_functions.png'
        fig2.savefig(path2, dpi=300, bbox_inches='tight')
        paths['effect_functions'] = path2
        plt.close(fig2)
        
        # 3. Marginal effects
        fig3 = self.marginal_effects_plot()
        path3 = output_dir / f'{prefix}_marginal_effects.png'
        fig3.savefig(path3, dpi=300, bbox_inches='tight')
        paths['marginal_effects'] = path3
        plt.close(fig3)
        
        # 4. Interaction decomposition
        fig4, stats = self.interaction_decomposition()
        path4 = output_dir / f'{prefix}_interaction_decomposition.png'
        fig4.savefig(path4, dpi=300, bbox_inches='tight')
        paths['interaction_decomposition'] = path4
        plt.close(fig4)
        
        # Save decomposition statistics
        import json
        stats_path = output_dir / f'{prefix}_decomposition_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        paths['decomposition_stats'] = stats_path
        
        logger.info(f"All DML visualizations saved to {output_dir}")
        
        return paths