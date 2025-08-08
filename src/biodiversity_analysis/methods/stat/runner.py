#!/usr/bin/env python
"""
Unified Biodiversity Statistical Analysis Runner

This script provides a uniform interface to run biodiversity statistical analyses.

Usage:
    python runner.py                           # Run all analyses sequentially
    python runner.py --analysis coverage      # Run specific analysis
    python runner.py --analysis trust_maps,latitudinal  # Run multiple analyses
    python runner.py --clean                  # Clean all output files
    python runner.py --clean --analysis coverage        # Clean specific analysis outputs
    python runner.py --list                   # List available analyses
    python runner.py --validate               # Validate configuration and paths
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Optional

# Import robust import helpers
try:
    from ...utils.import_helpers import safe_import, safe_import_class
except ImportError:
    # Fallback for development environment
    project_root = Path(__file__).parent.parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.biodiversity_analysis.utils.import_helpers import safe_import, safe_import_class

# Import centralized configuration using robust imports
config_module = safe_import('src.config.stat')
StatisticalDataPaths = config_module.StatisticalDataPaths
StatisticalAnalysisConfig = config_module.StatisticalAnalysisConfig

# Import analysis modules dynamically to avoid import errors


class BiodiversityAnalysisRunner:
    """Unified runner for biodiversity statistical analyses"""
    
    def __init__(self):
        # Initialize configuration
        self.data_paths = StatisticalDataPaths()
        self.analysis_config = StatisticalAnalysisConfig()
        self.base_output_dir = self.data_paths.output_base_dir
        
        self.available_analyses = {
            'coverage': {
                'name': 'Coverage Analysis',
                'description': 'Data coverage and completeness analysis',
                'output_dir': 'coverage_enhanced',
                'runner': self._run_coverage_analysis
            },
            'trust_maps': {
                'name': 'Trust Maps',
                'description': 'Trust and reliability mapping',
                'output_dir': 'trust_maps',
                'runner': self._run_trust_maps_analysis
            },
            'latitudinal': {
                'name': 'Latitudinal Analysis', 
                'description': 'Biodiversity patterns across latitude',
                'output_dir': 'latitudinal',
                'runner': self._run_latitudinal_analysis
            },
            'global_maps': {
                'name': 'Global Maps',
                'description': 'Global biodiversity visualization',
                'output_dir': 'global_maps',
                'runner': self._run_global_maps_analysis
            },
            'correlation': {
                'name': 'Correlation Analysis',
                'description': 'Multi-dataset correlation studies',
                'output_dir': 'correlation',
                'runner': self._run_correlation_analysis
            },
            'artifacts': {
                'name': 'Artifact Detection',
                'description': 'Data artifact and anomaly detection',
                'output_dir': 'artifacts',
                'runner': self._run_artifact_analysis
            },
            'overlap': {
                'name': 'Overlap Analysis',
                'description': 'Dataset overlap and intersection analysis',
                'output_dir': 'overlap',
                'runner': self._run_overlap_analysis
            }
        }
    
    def list_analyses(self):
        """List all available analyses"""
        print("🔬 Available Biodiversity Statistical Analyses:")
        print("=" * 60)
        for key, info in self.available_analyses.items():
            print(f"  {key:15} - {info['name']}")
            print(f"{'':17} {info['description']}")
            print()
    
    def validate_configuration(self):
        """Validate all paths and configuration"""
        print("🔧 Validating Configuration...")
        print("=" * 50)
        
        # Validate paths
        validation_results = self.data_paths.validate_all_paths()
        
        valid_count = sum(1 for valid in validation_results.values() if valid)
        total_count = len(validation_results)
        
        print(f"Path Validation: {valid_count}/{total_count} valid")
        
        for name, is_valid in validation_results.items():
            status = "✅" if is_valid else "❌"
            print(f"  {name:15} {status}")
        
        print(f"\n{self.data_paths.get_summary()}")
        print(f"\n{self.analysis_config.get_summary()}")
        
        if valid_count < total_count:
            print("\n⚠️  Some paths are invalid. Set environment variables to override defaults.")
            print("Example:")
            print("  export BIODIV_SCRATCH_DIR=/your/data/directory")
            print("  export BIODIV_CLIMATE_FILE=/your/climate/file.parquet")
        
        return valid_count == total_count
    
    def clean_outputs(self, analyses: Optional[List[str]] = None):
        """Clean output files for specified analyses (or all if None)"""
        if analyses is None:
            analyses = list(self.available_analyses.keys())
        
        print(f"🧹 Cleaning outputs for: {', '.join(analyses)}")
        
        cleaned_count = 0
        for analysis in analyses:
            if analysis not in self.available_analyses:
                print(f"  ⚠️  Unknown analysis: {analysis}")
                continue
                
            output_dir = self.base_output_dir / self.available_analyses[analysis]['output_dir']
            if output_dir.exists():
                shutil.rmtree(output_dir)
                print(f"  ✅ Cleaned: {output_dir}")
                cleaned_count += 1
            else:
                print(f"  ℹ️  No outputs found: {output_dir}")
        
        print(f"\n🎯 Cleaned {cleaned_count} analysis output directories")
    
    def run_analyses(self, analyses: List[str]):
        """Run specified analyses"""
        print(f"🚀 Running biodiversity statistical analyses: {', '.join(analyses)}")
        print("=" * 80)
        
        results = {}
        for i, analysis in enumerate(analyses, 1):
            if analysis not in self.available_analyses:
                print(f"  ⚠️  Unknown analysis: {analysis}")
                results[analysis] = False
                continue
            
            info = self.available_analyses[analysis]
            print(f"\n[{i}/{len(analyses)}] Running {info['name']}...")
            print(f"Description: {info['description']}")
            
            try:
                # Ensure output directory exists
                output_dir = self.base_output_dir / info['output_dir']
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Run the analysis
                info['runner']()
                
                print(f"  ✅ {info['name']} completed successfully")
                results[analysis] = True
                
            except Exception as e:
                print(f"  ❌ {info['name']} failed: {str(e)}")
                results[analysis] = False
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        total = len(analyses)
        
        print(f"\n🎯 Analysis Summary: {successful}/{total} successful")
        
        if successful < total:
            print("\nFailed analyses:")
            for analysis, success in results.items():
                if not success:
                    print(f"  - {analysis}")
    
    def _run_coverage_analysis(self):
        """Run coverage analysis"""
        # Use robust import for coverage analyzer
        EnhancedCoverageAnalyzer = safe_import_class(
            'src.biodiversity_analysis.methods.stat.coverage_analysis.analyzer',
            'EnhancedCoverageAnalyzer',
            alternative_names=['CoverageAnalyzerEnhanced']
        )
        
        # Get climate data path from configuration
        climate_path = self.data_paths.get_climate_data_path()
        if not climate_path:
            raise FileNotFoundError("No valid climate data file found. Check configuration.")
        
        # Get analysis configuration
        config = self.analysis_config.get_analysis_config('coverage')
        
        analyzer = EnhancedCoverageAnalyzer(str(climate_path))
        analyzer.run_complete_analysis()
    
    def _run_trust_maps_analysis(self):
        """Run trust maps analysis"""
        TrustMapVisualizer = safe_import_class(
            'src.biodiversity_analysis.methods.stat.trust_maps.analyzer',
            'TrustMapVisualizer'
        )
        
        # Get dataset paths from configuration
        dataset_paths = self.data_paths.get_dataset_paths()
        if not dataset_paths:
            raise FileNotFoundError("No valid dataset files found. Check configuration.")
        
        # Convert Path objects to strings for compatibility
        datasets = {name: str(path) for name, path in dataset_paths.items()}
        
        # Get analysis configuration
        config = self.analysis_config.get_analysis_config('trust_maps')
        
        visualizer = TrustMapVisualizer(output_dir=self.base_output_dir / 'trust_maps')
        visualizer.create_trust_maps(datasets)
    
    def _run_latitudinal_analysis(self):
        """Run latitudinal analysis"""
        latitudinal_module = safe_import('src.biodiversity_analysis.methods.stat.latitudinal_analysis.analyzer')
        latitudinal_module.main()
    
    def _run_global_maps_analysis(self):
        """Run global maps analysis"""
        GlobalMapVisualizer = safe_import_class(
            'src.biodiversity_analysis.methods.stat.global_maps.analyzer',
            'GlobalMapVisualizer'
        )
        
        # Get dataset paths from configuration
        dataset_paths = self.data_paths.get_dataset_paths()
        if not dataset_paths:
            raise FileNotFoundError("No valid dataset files found. Check configuration.")
        
        # Get analysis configuration
        config = self.analysis_config.get_analysis_config('global_maps')
        
        visualizer = GlobalMapVisualizer(output_dir=self.base_output_dir / 'global_maps')
        for name, path in dataset_paths.items():
            visualizer.create_global_map(str(path), name)
    
    def _run_correlation_analysis(self):
        """Run correlation analysis"""
        CorrelationExplorer = safe_import_class(
            'src.biodiversity_analysis.methods.stat.correlation_analysis.analyzer',
            'CorrelationExplorer'
        )
        
        # Get dataset paths from configuration
        dataset_paths = self.data_paths.get_dataset_paths()
        if not dataset_paths:
            raise FileNotFoundError("No valid dataset files found. Check configuration.")
        
        # Convert Path objects to strings for compatibility
        datasets = {name: str(path) for name, path in dataset_paths.items()}
        
        # Get analysis configuration
        config = self.analysis_config.get_analysis_config('correlation')
        
        explorer = CorrelationExplorer(datasets, output_dir=self.base_output_dir / 'correlation')
        explorer.run_comprehensive_analysis()
    
    def _run_artifact_analysis(self):
        """Run artifact detection analysis"""
        ArtifactDetector = safe_import_class(
            'src.biodiversity_analysis.methods.stat.artifact_detection.analyzer',
            'ArtifactDetector'
        )
        
        # Get dataset paths from configuration
        dataset_paths = self.data_paths.get_dataset_paths()
        if not dataset_paths:
            raise FileNotFoundError("No valid dataset files found. Check configuration.")
        
        # Convert Path objects to strings for compatibility
        datasets = {name: str(path) for name, path in dataset_paths.items()}
        
        # Get analysis configuration
        config = self.analysis_config.get_analysis_config('artifacts')
        
        detector = ArtifactDetector(datasets, output_dir=self.base_output_dir / 'artifacts')
        detector.run_comprehensive_detection()
    
    def _run_overlap_analysis(self):
        """Run overlap analysis"""
        OverlapAnalyzer = safe_import_class(
            'src.biodiversity_analysis.methods.stat.overlap_analysis.analyzer',
            'OverlapAnalyzer'
        )
        
        # Get dataset paths from configuration
        dataset_paths = self.data_paths.get_dataset_paths()
        if not dataset_paths:
            raise FileNotFoundError("No valid dataset files found. Check configuration.")
        
        # Convert Path objects to strings for compatibility
        datasets = {name: str(path) for name, path in dataset_paths.items()}
        
        # Get analysis configuration
        config = self.analysis_config.get_analysis_config('overlap')
        
        analyzer = OverlapAnalyzer(datasets, output_dir=self.base_output_dir / 'overlap')
        analyzer.run_comprehensive_overlap_analysis()


def main():
    parser = argparse.ArgumentParser(
        description="Unified Biodiversity Statistical Analysis Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runner.py                              # Run all analyses
  python runner.py --analysis coverage         # Run coverage analysis only
  python runner.py --analysis trust_maps,latitudinal  # Run multiple analyses
  python runner.py --clean                     # Clean all outputs
  python runner.py --clean --analysis coverage # Clean coverage outputs only
  python runner.py --list                      # List available analyses
  python runner.py --validate                  # Validate configuration and paths

Environment Variables:
  BIODIV_SCRATCH_DIR       Base directory for biodiversity data
  BIODIV_CLIMATE_FILE      Climate data file path
  BIODIV_PLANTS_FILE       Plants richness dataset path
  BIODIV_MAX_WORKERS       Maximum parallel workers (default: auto)
  BIODIV_MAX_MEMORY_GB     Maximum memory usage in GB (default: 32)
        """
    )
    
    parser.add_argument(
        '--analysis', '--analyses',
        type=str,
        help='Comma-separated list of analyses to run (default: all)'
    )
    
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean output files instead of running analyses'
    )
    
    parser.add_argument(
        '--list',
        action='store_true', 
        help='List available analyses and exit'
    )
    
    parser.add_argument(
        '--validate', '--config',
        action='store_true',
        help='Validate configuration and paths, then exit'
    )
    
    args = parser.parse_args()
    
    runner = BiodiversityAnalysisRunner()
    
    # List analyses if requested
    if args.list:
        runner.list_analyses()
        return
    
    # Validate configuration if requested
    if args.validate:
        is_valid = runner.validate_configuration()
        sys.exit(0 if is_valid else 1)
    
    # Parse analysis selection
    if args.analysis:
        selected_analyses = [a.strip() for a in args.analysis.split(',')]
    else:
        selected_analyses = list(runner.available_analyses.keys())
    
    # Clean or run analyses
    if args.clean:
        runner.clean_outputs(selected_analyses)
    else:
        runner.run_analyses(selected_analyses)


if __name__ == "__main__":
    main()