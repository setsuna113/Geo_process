#!/usr/bin/env python3
"""
Standalone diagnostics runner for saved EconML models
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Union, Optional
import sys
import os
import argparse

# Fix import paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

def load_model(model_path: Union[str, Path]):
    """Load saved model from disk"""
    
    model_path = Path(model_path)
    arrays_path = model_path.with_suffix('.npz')
    
    # Load model structure
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load arrays if they exist
    if arrays_path.exists():
        arrays = np.load(arrays_path)
        if 'nuisance_scores' in arrays:
            model.nuisance_scores_ = arrays['nuisance_scores']
        if 'cached_values' in arrays:
            model.cached_values_ = arrays['cached_values']
    
    return model


def run_diagnostics(model, data_path: Optional[str] = None):
    """
    Run diagnostics on loaded model
    
    Parameters
    ----------
    model : fitted EconML model
        The model to diagnose
    data_path : str, optional
        Path to test data for additional diagnostics
    
    NOTE: Sensitivity analysis has been moved to a standalone module.
    For single-treatment sensitivity analysis, use:
        from diagnostics.sensitivity import SingleTreatmentSensitivity
        
    Returns
    -------
    dict with diagnostic results
    """
    
    # Import diagnostics  
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from diagnostics import EconMLDiagnostics
    
    # If data provided, load it for additional diagnostics
    X_test, Y_test, T_test, W_test = None, None, None, None
    if data_path:
        # This would need implementation based on your data format
        print(f"Loading test data from: {data_path}")
        # Placeholder - implement based on your data format
        pass
    
    # Initialize diagnostics
    diag = EconMLDiagnostics(model, X_test, Y_test, T_test, W_test)
    
    # Run diagnostics (sensitivity analysis now in separate module)
    print("Running diagnostics...")
    
    results = diag.run_diagnostics()
    
    return results


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Run diagnostics on saved EconML model')
    parser.add_argument('--model', type=str, 
                       default='/scratch/yl998/econml_results/econml_model_latest.pkl',
                       help='Path to saved model (default: latest)')
    parser.add_argument('--data', type=str, default=None,
                       help='Optional path to test data for additional diagnostics')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for diagnostics (default: model_path.diagnostics.json)')
    # NOTE: Sensitivity analysis has been moved to diagnostics.sensitivity module
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--plot-dir', type=str, default=None,
                       help='Directory to save plots (default: same as output)')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from: {args.model}")
    try:
        model = load_model(args.model)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Run diagnostics
    print("Running diagnostics...")
    try:
        diagnostics = run_diagnostics(
            model, 
            args.data
        )
        print("Diagnostics completed")
    except Exception as e:
        print(f"Failed to run diagnostics: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save diagnostics
    output_path = args.output
    if not output_path:
        output_path = Path(args.model).with_suffix('.diagnostics.json')
    
    with open(output_path, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    
    print(f"Diagnostics saved to: {output_path}")
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating visualization plots...")
        try:
            from diagnostics.visualization import DiagnosticsVisualizer
            
            # Determine plot directory
            plot_dir = args.plot_dir
            if not plot_dir:
                plot_dir = Path(output_path).parent / "plots"
            
            # Create visualizer and generate plots
            viz = DiagnosticsVisualizer(diagnostics, output_dir=plot_dir)
            figures = viz.plot_all(save=True, show=False)
            
            # Also create summary figure
            summary_fig = viz.create_summary_figure()
            summary_fig.savefig(f"{plot_dir}/summary.png", dpi=300, bbox_inches='tight')
            
            print(f"Plots saved to: {plot_dir}")
            print(f"Generated {len(figures) + 1} plots:")
            for name in figures.keys():
                print(f"  - {name}.png")
            print(f"  - summary.png")
            
        except Exception as e:
            print(f"Failed to generate plots: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "="*50)
    print("DIAGNOSTICS SUMMARY")
    print("="*50)
    
    if 'score' in diagnostics and diagnostics['score'].get('fitted_score') is not None:
        print(f"Model score: {diagnostics['score']['fitted_score']:.4f}")
    
    if 'effects' in diagnostics:
        if 'mean_effect' in diagnostics['effects']:
            print(f"Mean treatment effect: {diagnostics['effects']['mean_effect']:.4f}")
        if 'std_effect' in diagnostics['effects']:
            print(f"Effect std dev: {diagnostics['effects']['std_effect']:.4f}")
    
    if 'nuisance' in diagnostics:
        print(f"Nuisance models checked: {len(diagnostics['nuisance'])}")
    
    print("="*50)


if __name__ == "__main__":
    main()