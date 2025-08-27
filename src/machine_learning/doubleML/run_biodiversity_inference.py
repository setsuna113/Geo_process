#!/usr/bin/env python3
"""
Run biodiversity inference analysis and save results
"""

import json
from pathlib import Path
import pandas as pd
from biodiversity_inference import BiodiversityInference


def main():
    """Run inference and save results"""
    
    # Input and output paths
    input_path = "/scratch/yl998/econml_results/econml_results_20250821_203021.json"
    output_dir = Path("/scratch/yl998/econml_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    print("Running biodiversity inference analysis...")
    bio_inf = BiodiversityInference(results_path=input_path)
    results = bio_inf.test_biodiversity_hypothesis()
    
    # Save results to JSON
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"biodiversity_inference_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("RESULTS SAVED")
    print(f"{'='*70}")
    print(f"Output file: {output_file}")
    
    # Create symlink to latest
    latest_link = output_dir / "biodiversity_inference_latest.json"
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(output_file.name)
    print(f"Latest link: {latest_link}")
    
    # Save summary report
    report_file = output_dir / f"biodiversity_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BIODIVERSITY INFERENCE REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"Data source: {input_path}\n\n")
        
        f.write("HYPOTHESIS TEST RESULTS\n")
        f.write("-"*70 + "\n")
        f.write("H0: Biodiversity does not provide additional information\n")
        f.write("H1: Biodiversity provides additional information\n\n")
        
        f.write(f"Decision: {results['decision']['decision']}\n")
        f.write(f"Evidence: {results['decision']['evidence']}\n")
        f.write(f"Recommendation: {results['decision']['recommendation']}\n\n")
        
        f.write("INDIVIDUAL EFFECTS\n")
        f.write("-"*70 + "\n")
        for name, stats in results['treatments'].items():
            f.write(f"\n{name}:\n")
            f.write(f"  Coefficient: {stats['coefficient']:.6f}\n")
            f.write(f"  P-value: {stats['p_value']:.6f}\n")
            f.write(f"  Significant: {'Yes' if stats['significant'] else 'No'}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write(f"Significant treatments: {results['significant_count']}/3\n")
        f.write(f"Joint p-value: {results['p_value_joint']:.6e}\n")
        f.write(f"Wald statistic: {results['wald_statistic']:.2f}\n")
    
    print(f"Report saved: {report_file}")
    
    return results


if __name__ == "__main__":
    results = main()