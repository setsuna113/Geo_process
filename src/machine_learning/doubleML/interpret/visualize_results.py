#!/usr/bin/env python3
"""
Orchestrator for comprehensive heterogeneous treatment effect visualizations

This script creates:
- SHAP visualizations (summary, beeswarm, waterfall, dependence plots)
- Subgroup analysis (decision trees, forest plots, TOC curves)
- Climate gradient visualizations (1D effect magnitudes, 2D climate spaces)
- Combined summary dashboard

The script automatically calls:
- plot_climate_gradients.py for 1D effect magnitude plots
- plot_climate_heatmap.py for 2D climate space heatmaps

Usage:
  python visualize_results.py --model MODEL.pkl --results RESULTS.json --output ./figures
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/home/yl998/Geo_process')

from interpret import shap_analysis, SubgroupAnalyzer
from src.machine_learning.doubleML.refactored_pipeline import HeterogeneousEffectEstimator, HomogeneousEffectEstimator


class HeterogeneousEffectsVisualizer:
    """
    Complete visualization suite for heterogeneous treatment effects
    """
    
    def __init__(self, model_path: str, results_path: Optional[str], data_path: str):
        """
        Initialize visualizer with saved model and optionally results
        
        Args:
            model_path: Path to saved .pkl model file
            results_path: Path to saved .json results file (OPTIONAL - can be None)
            data_path: Path to preprocessed data (.npz file) - REQUIRED
        """
        self.model_path = Path(model_path)
        self.results_path = Path(results_path) if results_path else None
        self.data_path = Path(data_path)
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Load model (required) and results (optional)
        self.model_wrapper = self._load_model()
        self.results = self._load_results() if self.results_path else {}
        
        # Extract configuration from model or results
        self._extract_config()
        
        # Initialize subgroup analyzer
        self.subgroup_analyzer = None
        
        print(f"✓ Loaded model from: {self.model_path}")
        if self.results_path:
            print(f"✓ Loaded results from: {self.results_path}")
        else:
            print(f"✓ No JSON results provided - using model information only")
        print(f"✓ Model type: {self.model_type}")
        print(f"✓ Heterogeneous: {self.is_heterogeneous}")
        if self.is_heterogeneous:
            print(f"✓ Effect modifiers: {self.effect_modifiers}")
    
    def _load_model(self) -> Any:
        """Load pickled model"""
        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)
            
        # Store the wrapper to extract metadata
        self.wrapper = model if hasattr(model, 'estimator') else None
            
        # If it's a wrapper, extract the actual estimator
        if hasattr(model, 'estimator'):
            print(f"  Extracted estimator from wrapper: {type(model.estimator).__name__}")
            self.model = model.estimator
            return model  # Return wrapper for metadata
        
        self.model = model
        return model
    
    def _load_results(self) -> Dict:
        """Load JSON results"""
        with open(self.results_path, 'r') as f:
            return json.load(f)
    
    def _extract_config(self):
        """Extract configuration from model and/or results"""
        # Try to get info from wrapper first, then results, then defaults
        if self.wrapper:
            # We have a wrapper - extract from it
            self.model_type = type(self.model).__name__
            self.is_heterogeneous = 'Forest' in self.model_type or 'Heterogeneous' in type(self.wrapper).__name__
            
            # Get effect modifiers from wrapper results or feature importances
            if hasattr(self.wrapper, 'results') and self.wrapper.results:
                feature_imps = self.wrapper.results.get('feature_importances', {})
                self.effect_modifiers = list(feature_imps.keys()) if feature_imps else ['bio01_std', 'bio04_std', 'bio12_std']
            elif hasattr(self.model, 'feature_importances_'):
                # Default names if we only have importances array
                n_features = len(self.model.feature_importances_)
                self.effect_modifiers = ['bio01_std', 'bio04_std', 'bio12_std'][:n_features]
            else:
                self.effect_modifiers = self.results.get('effect_modifiers', ['bio01_std', 'bio04_std', 'bio12_std'])
        else:
            # No wrapper - extract from model directly or results
            self.model_type = type(self.model).__name__
            self.is_heterogeneous = 'Forest' in self.model_type
            
            if self.results:
                # Use results if available
                self.is_heterogeneous = self.results.get('type') == 'heterogeneous' or \
                                       self.results.get('model_info', {}).get('heterogeneous', self.is_heterogeneous)
                self.effect_modifiers = self.results.get('effect_modifiers', [])
                if not self.effect_modifiers and 'feature_importances' in self.results:
                    self.effect_modifiers = list(self.results['feature_importances'].keys())
            else:
                # Default values
                self.effect_modifiers = ['bio01_std', 'bio04_std', 'bio12_std']
        
        # Treatment names are standard
        self.treatment_names = self.results.get('treatment_names', 
                                               ['plants_richness', 'vertebrates_richness', 'plants_x_vertebrates'])
        self.n_observations = self.results.get('n_observations', 0)
    
    def load_data(self, n_samples: Optional[int] = None, seed: int = 42) -> np.ndarray:
        """
        Load preprocessed data from .npz file
        
        Args:
            n_samples: Number of samples to use (None = use all)
            seed: Random seed for sampling
            
        Returns:
            Feature matrix X (effect modifiers)
        """
        # Load the npz file
        data = np.load(self.data_path)
        
        # Get X (effect modifiers)
        if 'X' not in data:
            raise KeyError(f"'X' (effect modifiers) not found in {self.data_path}")
        
        X = data['X']
        total_samples = len(X)
        
        print(f"✓ Loaded data from {self.data_path}")
        print(f"  Total samples available: {total_samples:,}")
        print(f"  Shape: {X.shape}")
        
        # Sample if requested
        if n_samples is not None and n_samples < total_samples:
            np.random.seed(seed)
            # Use systematic sampling for better coverage
            step = total_samples // n_samples
            indices = np.arange(0, total_samples, step)[:n_samples]
            X = X[indices]
            print(f"  Using {len(X):,} samples (every {step}th point)")
        else:
            print(f"  Using all {total_samples:,} samples")
        
        return X
    
    def create_all_visualizations(self, X: np.ndarray,
                                 output_dir: str = "./figures",
                                 format: str = 'png', dpi: int = 150):
        """
        Create all available visualizations
        
        Args:
            X: Feature matrix (effect modifiers) - REQUIRED
            output_dir: Directory to save figures
            format: Image format (png, pdf, svg)
            dpi: Resolution for raster formats
        """
        if not self.is_heterogeneous:
            print("✗ Model is not heterogeneous - visualizations require effect modifiers")
            return
        
        if X is None:
            raise ValueError("X (feature matrix) is required for visualizations")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*60}")
        
        # Initialize subgroup analyzer
        self.subgroup_analyzer = SubgroupAnalyzer(self.model, self.effect_modifiers, self.treatment_names)
        
        # Run subgroup analysis
        print("\n1. Running subgroup analysis...")
        subgroup_results = self.subgroup_analyzer.identify_subgroups(X, max_depth=2, min_samples_leaf=50)
        
        # SHAP Visualizations using new module
        print("\n2. Creating SHAP visualizations...")
        try:
            # Use the new shap module to create all SHAP visualizations
            shap_results = shap_analysis.create_shap_visualizations(
                model=self.model,
                X=X,
                effect_modifiers=self.effect_modifiers,
                treatment_names=self.treatment_names,
                output_dir=output_path,
                shap_sample_size=len(X),  # Use all available samples for SHAP
                verbose=True
            )
            print("   ✓ SHAP visualizations completed")
        except Exception as e:
            print(f"   ✗ SHAP visualizations failed: {e}")
        
        # Subgroup Visualizations
        print("\n3. Creating subgroup visualizations...")
        
        # Decision trees
        for treatment in self.treatment_names[:2]:  # Main treatments only
            try:
                fig = self.subgroup_analyzer.plot_tree(treatment, figsize=(12, 8))
                fig.savefig(output_path / f"tree_{treatment}.{format}", dpi=dpi, bbox_inches='tight')
                print(f"   ✓ Saved: tree_{treatment}.{format}")
                plt.close(fig)
            except Exception as e:
                print(f"   ✗ Tree for {treatment} failed: {e}")
        
        # Forest plot
        try:
            fig = self.subgroup_analyzer.plot_forest(X, figsize=(10, 12))
            fig.savefig(output_path / f"forest_plot.{format}", dpi=dpi, bbox_inches='tight')
            print(f"   ✓ Saved: forest_plot.{format}")
            plt.close(fig)
        except Exception as e:
            print(f"   ✗ Forest plot failed: {e}")
        
        # TOC curves
        for treatment in self.treatment_names[:2]:
            try:
                fig = self.subgroup_analyzer.plot_toc_curve(X, treatment)
                fig.savefig(output_path / f"toc_curve_{treatment}.{format}", dpi=dpi, bbox_inches='tight')
                print(f"   ✓ Saved: toc_curve_{treatment}.{format}")
                plt.close(fig)
            except Exception as e:
                print(f"   ✗ TOC curve for {treatment} failed: {e}")
        
        # Climate-specific visualizations
        print("\n4. Creating climate gradient visualizations...")
        
        # Import and run climate visualization tools
        try:
            # Import the standalone climate visualization functions
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, current_dir)
            
            from plot_climate_gradients import plot_effect_magnitudes
            from plot_climate_heatmap import plot_climate_slices_2d
            
            # Generate climate gradient plots for each treatment
            for treatment in self.treatment_names[:2]:  # Skip interaction term
                try:
                    # 1D Effect magnitudes along each climate gradient
                    fig = plot_effect_magnitudes(
                        self.model, X[:500], self.effect_modifiers, 
                        self.treatment_names, n_points=100
                    )
                    fig.savefig(output_path / f"effect_magnitudes_{treatment}.{format}", 
                               dpi=dpi, bbox_inches='tight')
                    print(f"   ✓ Saved: effect_magnitudes_{treatment}.{format}")
                    plt.close(fig)
                except Exception as e:
                    print(f"   ✗ Effect magnitudes for {treatment} failed: {e}")
                
                # 2D Climate space heatmaps (if we have 2+ modifiers)
                if len(self.effect_modifiers) >= 2:
                    try:
                        fig = plot_climate_slices_2d(
                            self.model, X[:500], self.effect_modifiers, 
                            treatment, n_grid=30
                        )
                        fig.savefig(output_path / f"climate_slices_2d_{treatment}.{format}", 
                                   dpi=dpi, bbox_inches='tight')
                        print(f"   ✓ Saved: climate_slices_2d_{treatment}.{format}")
                        plt.close(fig)
                    except Exception as e:
                        print(f"   ✗ Climate slices for {treatment} failed: {e}")
        except ImportError as e:
            print(f"   ✗ Could not import climate visualization tools: {e}")
            print("   - Ensure plot_climate_gradients.py and plot_climate_heatmap.py are in the same directory")
        except Exception as e:
            print(f"   ✗ Climate visualizations failed: {e}")
        
        # Combined summary figure
        print("\n5. Creating combined summary figure...")
        try:
            self._create_summary_figure(X[:500], output_path, format, dpi)
            print(f"   ✓ Saved: summary_combined.{format}")
        except Exception as e:
            print(f"   ✗ Combined summary failed: {e}")
        
        print(f"\n{'='*60}")
        print(f"✓ Visualizations saved to: {output_path}")
        print(f"\nGenerated visualization types:")
        print(f"  • SHAP analysis (summary, beeswarm, waterfall, dependence)")
        print(f"  • Subgroup analysis (trees, forest plots, TOC curves)")
        print(f"  • Climate gradients (1D effects, 2D heatmaps)")
        print(f"  • Combined summary dashboard")
        print(f"{'='*60}")
    
    def _create_summary_figure(self, X: np.ndarray, output_path: Path, 
                              format: str, dpi: int):
        """Create a combined summary figure"""
        fig = plt.figure(figsize=(14, 10))
        
        # Layout: 2x2 grid
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Effect distribution (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        effects_plants = self.model.const_marginal_effect(X)[:, 0]
        ax1.hist(effects_plants, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax1.set_xlabel('Treatment Effect')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Plants Effect Distribution')
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # 2. Heterogeneity scatter (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        if len(self.effect_modifiers) >= 1:
            effects_verts = self.model.const_marginal_effect(X)[:, 1]
            ax2.scatter(X[:, 0], effects_verts, alpha=0.5, c=effects_verts, cmap='viridis', s=20)
            ax2.set_xlabel(self.effect_modifiers[0])
            ax2.set_ylabel('Vertebrates Treatment Effect')
            ax2.set_title('Treatment Effect Heterogeneity')
            ax2.grid(True, alpha=0.3)
        
        # 3. Subgroup comparison (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])
        # Get subgroup stats
        subgroup_results = self.subgroup_analyzer.identify_subgroups(X[:200], max_depth=2)
        
        treatments = []
        n_subgroups = []
        for treatment, data in subgroup_results.items():
            if 'interaction' not in treatment.lower():
                treatments.append(treatment.replace('_', ' ').title())
                n_subgroups.append(data['n_subgroups'])
        
        ax3.bar(treatments, n_subgroups, color=['green', 'blue'], alpha=0.7)
        ax3.set_ylabel('Number of Subgroups')
        ax3.set_title('Identified Subgroups by Treatment')
        
        # 4. Results summary table (bottom right)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('tight')
        ax4.axis('off')
        
        # Create summary statistics table
        summary_data = []
        
        if 'effects_by_location' in self.results:
            for location, data in self.results['effects_by_location'].items():
                row = [location.replace('_', ' ').title()]
                for treatment in self.treatment_names[:2]:
                    if treatment in data['treatments']:
                        effect = data['treatments'][treatment]['coefficient']
                        p_val = data['treatments'][treatment]['p_value']
                        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                        row.append(f"{effect:.4f}{sig}")
                    else:
                        row.append("N/A")
                summary_data.append(row)
        
        if summary_data:
            table = ax4.table(cellText=summary_data,
                            colLabels=['Condition'] + [t.replace('_', ' ').title() for t in self.treatment_names[:2]],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax4.set_title('Treatment Effects by Condition (* p<0.05, ** p<0.01, *** p<0.001)', 
                         fontsize=12, pad=20)
        
        # Overall title
        fig.suptitle('Heterogeneous Treatment Effects - Summary Dashboard', 
                    fontsize=16, y=0.98)
        
        # Save
        fig.savefig(output_path / f"summary_combined.{format}", dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    
    def print_summary(self):
        """Print summary of loaded model and results"""
        print("\n" + "="*60)
        print("MODEL SUMMARY")
        print("="*60)
        
        print(f"\nModel Type: {self.model_type}")
        if self.n_observations:
            print(f"Observations: {self.n_observations:,}")
        
        # Get metadata from wrapper if available
        if self.wrapper:
            if hasattr(self.wrapper, 'cv_folds'):
                print(f"CV Folds: {self.wrapper.cv_folds}")
            if hasattr(self.wrapper, 'fit_time'):
                print(f"Fit Time: {self.wrapper.fit_time:.2f} seconds")
            if hasattr(self.wrapper, 'n_estimators'):
                print(f"N Estimators: {self.wrapper.n_estimators}")
        elif self.results:
            # Fall back to results if available
            if 'model_info' in self.results:
                print(f"CV Folds: {self.results['model_info'].get('cv_folds', 'N/A')}")
            if 'fit_time' in self.results or 'fit_time_seconds' in self.results:
                fit_time = self.results.get('fit_time', self.results.get('fit_time_seconds', 'N/A'))
                if fit_time != 'N/A':
                    print(f"Fit Time: {fit_time:.2f} seconds")
        
        if self.is_heterogeneous:
            print(f"\nHeterogeneous Effects Model")
            print(f"Effect Modifiers: {', '.join(self.effect_modifiers)}")
            
            # Print feature importances if available
            if hasattr(self.model, 'feature_importances_'):
                print("\nFeature Importances:")
                for mod, imp in zip(self.effect_modifiers, self.model.feature_importances_):
                    print(f"  {mod}: {imp:.4f}")
            elif self.wrapper and hasattr(self.wrapper, 'results') and self.wrapper.results:
                if 'feature_importances' in self.wrapper.results:
                    print("\nFeature Importances:")
                    for mod, imp in self.wrapper.results['feature_importances'].items():
                        print(f"  {mod}: {imp:.4f}")
            
            # Only show effects_by_location if it exists in results
            if self.results and 'effects_by_location' in self.results:
                print("\nEffects by Condition:")
                for location, data in self.results['effects_by_location'].items():
                    print(f"\n  {location}:")
                    print(f"    Conditions: {data['conditions']}")
                    for treatment, stats in data['treatments'].items():
                        print(f"    {treatment}: {stats['coefficient']:.6f} "
                              f"(p={stats['p_value']:.4f})")
        else:
            print(f"\nConstant Effects Model")
            if self.results and 'treatments' in self.results:
                print("\nTreatment Effects:")
                for treatment, stats in self.results['treatments'].items():
                    print(f"  {treatment}: {stats['coefficient']:.6f} "
                          f"(SE={stats['std_error']:.6f}, p={stats['p_value']:.4f})")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize EconML heterogeneous treatment effects')
    parser.add_argument('--model', type=str, required=True, help='Path to saved .pkl model file')
    parser.add_argument('--results', type=str, default=None, help='Path to saved .json results file (OPTIONAL)')
    parser.add_argument('--data', type=str, required=True, help='Path to preprocessed .npz data file (REQUIRED)')
    parser.add_argument('--output', type=str, default='./figures', help='Output directory for figures')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'pdf', 'svg'],
                       help='Image format')
    parser.add_argument('--dpi', type=int, default=150, help='Resolution for raster formats')
    parser.add_argument('--n-samples', type=int, default=10000, help='Number of samples to use (default: 10000)')
    
    args = parser.parse_args()
    
    # Create visualizer with optional results path
    visualizer = HeterogeneousEffectsVisualizer(args.model, args.results, args.data)
    
    # Print summary
    visualizer.print_summary()
    
    # Load actual data with sampling
    X = visualizer.load_data(n_samples=args.n_samples)
    
    # Generate visualizations with actual data
    visualizer.create_all_visualizations(
        X=X,
        output_dir=args.output,
        format=args.format,
        dpi=args.dpi
    )


if __name__ == "__main__":
    # If no arguments provided, show usage
    import sys
    if len(sys.argv) == 1:
        print("Usage: python visualize_results.py --model MODEL.pkl --results RESULTS.json --data DATA.npz [options]")
        print("\nRequired arguments:")
        print("  --model MODEL.pkl     Path to saved model file")
        print("  --results RESULTS.json Path to saved results file")
        print("  --data DATA.npz       Path to preprocessed data file")
        print("\nOptional arguments:")
        print("  --output DIR          Output directory (default: ./figures)")
        print("  --n-samples N         Number of samples to use (default: 10000)")
        print("  --format {png,pdf,svg} Image format (default: png)")
        print("  --dpi DPI            Resolution (default: 150)")
        sys.exit(1)
    else:
        main()