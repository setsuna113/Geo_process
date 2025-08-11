#!/usr/bin/env python
"""
Statistical Analysis Runner for Biodiversity Data
==================================================

Simple, clean runner with single YAML configuration.
"""

import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import analyzers
from src.biodiversity_analysis.methods.stat.correlation_analysis.analyzer import CorrelationExplorer
from src.biodiversity_analysis.methods.stat.trust_maps.analyzer import TrustMapVisualizer
from src.biodiversity_analysis.methods.stat.overlap_analysis.analyzer import OverlapAnalyzer
from src.biodiversity_analysis.methods.stat.coverage_analysis.analyzer import CoverageAnalyzer
from src.biodiversity_analysis.methods.stat.latitudinal_analysis.analyzer import LatitudinalAnalyzer
from src.biodiversity_analysis.methods.stat.global_maps.analyzer import GlobalMapAnalyzer


class StatAnalysisRunner:
    """Simple runner with single YAML configuration."""
    
    def __init__(self, config_path: str = "src/config/stat/stat_methods.yml"):
        """Initialize with configuration file."""
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup output directory
        self.output_dir = Path(self.config['settings']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Map analyzer names to classes
        self.analyzer_classes = {
            'correlation': CorrelationExplorer,
            'trust_maps': TrustMapVisualizer,
            'overlap': OverlapAnalyzer,
            'coverage': CoverageAnalyzer,
            'latitudinal': LatitudinalAnalyzer,
            'global_maps': GlobalMapAnalyzer
        }
        
        # Log initialization
        self.logger.info(f"Initialized StatAnalysisRunner")
        self.logger.info(f"Config: {config_path}")
        self.logger.info(f"Output: {self.output_dir}")
        
        # Log enabled/disabled methods
        enabled = [name for name, cfg in self.config['methods'].items() if cfg.get('enabled', True)]
        disabled = [name for name, cfg in self.config['methods'].items() if not cfg.get('enabled', True)]
        
        self.logger.info(f"Enabled methods: {enabled}")
        if disabled:
            for method in disabled:
                reason = self.config['methods'][method].get('skip_reason', 'Disabled')
                self.logger.info(f"  ⏭️  {method}: {reason}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration."""
        config_file = Path(config_path)
        if not config_file.exists():
            # Try relative to project root
            config_file = Path(__file__).parent.parent / config_path
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """Setup logging from config."""
        log_level = getattr(logging, self.config['settings']['log_level'])
        log_file = self.config['settings']['log_file']
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_analysis(self, method: str, datasets: Optional[Dict] = None) -> bool:
        """
        Run a specific analysis method.
        
        Args:
            method: Name of method to run
            datasets: Optional dataset override
            
        Returns:
            Success status
        """
        # Check if method exists
        if method not in self.analyzer_classes:
            self.logger.error(f"Unknown method: {method}")
            return False
        
        # Check if method is enabled
        method_config = self.config['methods'].get(method, {})
        if not method_config.get('enabled', True):
            reason = method_config.get('skip_reason', 'Disabled')
            self.logger.info(f"⏭️  Skipping {method}: {reason}")
            return False
        
        # Use config datasets if not provided
        if datasets is None:
            datasets = {}
            for name, path in self.config['datasets'].items():
                if name != 'climate_parquet' and path and Path(path).exists():
                    datasets[name] = path
        
        if not datasets:
            self.logger.error("No valid datasets found")
            return False
        
        # Get sample factor from config
        sample_factor = method_config.get('sample_factor', 10)
        
        try:
            # Create analyzer
            analyzer_class = self.analyzer_classes[method]
            output_dir = self.output_dir / method
            analyzer = analyzer_class(str(output_dir))
            
            self.logger.info(f"🔬 Running {method} analysis")
            self.logger.info(f"   Datasets: {list(datasets.keys())}")
            self.logger.info(f"   Sample factor: {sample_factor}")
            
            # Run method-specific analysis
            if method == 'correlation':
                df_clean, metadata = analyzer.load_and_prepare_data(datasets, sample_factor)
                analyzer.explain_correlation_methods()
                results = analyzer.calculate_all_correlations(df_clean)
                analyzer.visualize_correlation_comparison(results)
            
            elif method == 'trust_maps':
                analyzer.create_trust_maps(datasets)
            
            elif method == 'overlap':
                aligned_data = analyzer.load_and_align_datasets(datasets, sample_factor)
                stats = analyzer.calculate_overlap_statistics()
                concordance = analyzer.calculate_concordance()
                hotspots = analyzer.identify_hotspots()
                analyzer.create_overlap_visualizations()
                report = analyzer.generate_overlap_report()
            
            elif method == 'coverage':
                analyzer.analyze_coverage(datasets, sample_factor=sample_factor)
            
            elif method == 'latitudinal':
                analyzer.analyze_latitudinal_patterns(datasets, sample_factor=sample_factor)
            
            elif method == 'global_maps':
                analyzer.create_global_maps(datasets, sample_factor=sample_factor)
            
            self.logger.info(f"✅ Completed {method}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed {method}: {e}")
            return False
    
    def run_all_enabled(self) -> Dict[str, bool]:
        """Run all enabled methods."""
        results = {}
        
        # Get enabled methods
        enabled = [name for name, cfg in self.config['methods'].items() 
                  if cfg.get('enabled', True)]
        
        self.logger.info(f"Running {len(enabled)} enabled methods")
        
        for method in enabled:
            self.logger.info("="*50)
            results[method] = self.run_analysis(method)
        
        # Summary
        self.logger.info("="*60)
        self.logger.info("SUMMARY:")
        for method, success in results.items():
            status = "✅" if success else "❌"
            self.logger.info(f"  {status} {method}")
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run statistical analyses for biodiversity data",
        epilog="""
Examples:
  # Run all enabled methods
  %(prog)s
  
  # Run specific method
  %(prog)s --method correlation
  
  # Use custom config
  %(prog)s --config my_config.yml
  
  # Override datasets
  %(prog)s --dataset plants:data/plants.tif --dataset animals:data/animals.tif
        """
    )
    
    parser.add_argument(
        '--config',
        default='src/config/stat/stat_methods.yml',
        help='Configuration file (default: src/config/stat/stat_methods.yml)'
    )
    
    parser.add_argument(
        '--method',
        choices=['correlation', 'trust_maps', 'overlap', 'coverage', 'latitudinal', 'global_maps'],
        help='Specific method to run (default: all enabled)'
    )
    
    parser.add_argument(
        '--dataset',
        action='append',
        help='Override dataset (format: name:path)'
    )
    
    parser.add_argument(
        '--show-config',
        action='store_true',
        help='Show configuration and exit'
    )
    
    args = parser.parse_args()
    
    # Initialize runner
    try:
        runner = StatAnalysisRunner(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    # Show config if requested
    if args.show_config:
        import json
        print("\nConfiguration:")
        print("="*60)
        print(yaml.dump(runner.config, default_flow_style=False))
        return 0
    
    # Parse dataset overrides
    datasets = None
    if args.dataset:
        datasets = {}
        for spec in args.dataset:
            if ':' in spec:
                name, path = spec.split(':', 1)
                datasets[name] = path
    
    # Run analysis
    if args.method:
        success = runner.run_analysis(args.method, datasets)
        return 0 if success else 1
    else:
        results = runner.run_all_enabled()
        total = len(results)
        successful = sum(results.values())
        
        if successful == total:
            runner.logger.info(f"\n🎉 All {total} methods completed successfully!")
            return 0
        else:
            runner.logger.warning(f"\n⚠️  {total - successful}/{total} methods failed")
            return 1


if __name__ == "__main__":
    sys.exit(main())