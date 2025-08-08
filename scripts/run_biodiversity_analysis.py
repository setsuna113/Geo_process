#!/usr/bin/env python
"""
Unified Biodiversity Analysis Runner
====================================

Production-ready script to run all biodiversity analyses with clean interface.
Eliminates redundant logic and provides consistent data processing across all analyzers.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the refactored analyzers
from src.biodiversity_analysis.methods.stat.correlation_analysis.analyzer import CorrelationExplorer
from src.biodiversity_analysis.methods.stat.trust_maps.analyzer import TrustMapVisualizer
from src.biodiversity_analysis.methods.stat.overlap_analysis.analyzer import OverlapAnalyzer
from src.biodiversity_analysis.methods.stat.coverage_analysis.analyzer import CoverageAnalyzer
from src.biodiversity_analysis.methods.stat.latitudinal_analysis.analyzer import LatitudinalAnalyzer
from src.biodiversity_analysis.methods.stat.global_maps.analyzer import GlobalMapAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('biodiversity_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BiodiversityAnalysisRunner:
    """Unified runner for all biodiversity analyses with optimized data processing."""
    
    def __init__(self, output_dir: str = "outputs/biodiversity_analysis"):
        """Initialize analysis runner."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Available analyzers
        self.available_analyzers = {
            'correlation': CorrelationExplorer,
            'trust_maps': TrustMapVisualizer,
            'overlap': OverlapAnalyzer,
            'coverage': CoverageAnalyzer,
            'latitudinal': LatitudinalAnalyzer,
            'global_maps': GlobalMapAnalyzer
        }
        
        logger.info(f"Initialized BiodiversityAnalysisRunner with output: {self.output_dir}")
        logger.info(f"Available analyzers: {list(self.available_analyzers.keys())}")
    
    def run_analysis(self, 
                    analyzer_name: str,
                    datasets: Dict[str, str],
                    sample_factor: int = 1,
                    **kwargs) -> bool:
        """
        Run a specific analysis with unified data processing.
        
        Args:
            analyzer_name: Name of analyzer to run
            datasets: Dictionary of {name: file_path}
            sample_factor: Spatial sampling factor for large datasets
            **kwargs: Additional analyzer-specific parameters
            
        Returns:
            Success status
        """
        if analyzer_name not in self.available_analyzers:
            logger.error(f"Unknown analyzer: {analyzer_name}")
            return False
        
        try:
            # Create analyzer instance
            analyzer_class = self.available_analyzers[analyzer_name]
            analyzer_output_dir = self.output_dir / analyzer_name
            analyzer = analyzer_class(str(analyzer_output_dir))
            
            logger.info(f"Starting {analyzer_name} analysis...")
            logger.info(f"Datasets: {list(datasets.keys())}")
            logger.info(f"Sample factor: {sample_factor}")
            
            # Run analysis based on type
            if analyzer_name == 'correlation':
                self._run_correlation_analysis(analyzer, datasets, sample_factor, **kwargs)
            
            elif analyzer_name == 'trust_maps':
                self._run_trust_maps_analysis(analyzer, datasets, sample_factor, **kwargs)
            
            elif analyzer_name == 'overlap':
                self._run_overlap_analysis(analyzer, datasets, sample_factor, **kwargs)
            
            elif analyzer_name == 'coverage':
                self._run_coverage_analysis(analyzer, datasets, sample_factor, **kwargs)
            
            elif analyzer_name == 'latitudinal':
                self._run_latitudinal_analysis(analyzer, datasets, sample_factor, **kwargs)
            
            elif analyzer_name == 'global_maps':
                self._run_global_maps_analysis(analyzer, datasets, sample_factor, **kwargs)
            
            logger.info(f"✓ Completed {analyzer_name} analysis")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed {analyzer_name} analysis: {e}")
            return False
    
    def _run_correlation_analysis(self, analyzer, datasets, sample_factor, **kwargs):
        """Run correlation analysis with unified data processing."""
        # Load and prepare data using unified utilities
        df_clean, metadata = analyzer.load_and_prepare_data(datasets, sample_factor)
        
        # Generate correlation analysis
        analyzer.explain_correlation_methods()
        results = analyzer.calculate_all_correlations(df_clean)
        analyzer.visualize_correlation_comparison(results)
    
    def _run_trust_maps_analysis(self, analyzer, datasets, sample_factor, **kwargs):
        """Run trust maps analysis with unified data processing."""
        analyzer.create_trust_maps(datasets)
    
    def _run_overlap_analysis(self, analyzer, datasets, sample_factor, **kwargs):
        """Run overlap analysis with unified data processing."""
        aligned_data = analyzer.load_and_align_datasets(datasets, sample_factor)
        stats = analyzer.calculate_overlap_statistics()
        concordance = analyzer.calculate_concordance()
        hotspots = analyzer.identify_hotspots()
        analyzer.create_overlap_visualizations()
        report = analyzer.generate_overlap_report()
    
    def _run_coverage_analysis(self, analyzer, datasets, sample_factor, **kwargs):
        """Run coverage analysis with unified data processing."""
        analyzer.analyze_coverage(datasets, sample_factor=sample_factor)
    
    def _run_latitudinal_analysis(self, analyzer, datasets, sample_factor, **kwargs):
        """Run latitudinal analysis with unified data processing."""
        analyzer.analyze_latitudinal_patterns(datasets, sample_factor=sample_factor)
    
    def _run_global_maps_analysis(self, analyzer, datasets, sample_factor, **kwargs):
        """Run global maps analysis with unified data processing."""
        analyzer.create_global_maps(datasets, sample_factor=sample_factor)
    
    def run_multiple_analyses(self, 
                             analyzer_names: List[str],
                             datasets: Dict[str, str],
                             sample_factor: int = 1,
                             **kwargs) -> Dict[str, bool]:
        """
        Run multiple analyses in sequence.
        
        Args:
            analyzer_names: List of analyzer names to run
            datasets: Dictionary of datasets
            sample_factor: Spatial sampling factor
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of {analyzer_name: success_status}
        """
        results = {}
        
        logger.info(f"Running {len(analyzer_names)} analyses in sequence")
        
        for analyzer_name in analyzer_names:
            logger.info(f"{'='*50}")
            success = self.run_analysis(analyzer_name, datasets, sample_factor, **kwargs)
            results[analyzer_name] = success
        
        logger.info(f"{'='*50}")
        logger.info("ANALYSIS SUMMARY:")
        for analyzer_name, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            logger.info(f"  {analyzer_name}: {status}")
        
        return results
    
    def run_all_analyses(self, 
                        datasets: Dict[str, str],
                        sample_factor: int = 1,
                        **kwargs) -> Dict[str, bool]:
        """
        Run all available analyses.
        
        Args:
            datasets: Dictionary of datasets
            sample_factor: Spatial sampling factor
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of results
        """
        return self.run_multiple_analyses(
            list(self.available_analyzers.keys()),
            datasets,
            sample_factor,
            **kwargs
        )


def create_sample_datasets() -> Dict[str, str]:
    """Create sample dataset configuration for testing."""
    # These are example paths - replace with actual data paths
    return {
        'plants': 'data/daru-plants-richness.tif',
        'animals': 'data/iucn-animals-richness.tif',
        'fungi_gbif': 'data/gbif-fungi-richness.tif',
        'fungi_predicted': 'data/predicted-fungi-richness.tif'
    }


def main():
    """Main entry point for biodiversity analysis."""
    parser = argparse.ArgumentParser(
        description="Run biodiversity analyses with unified data processing",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--analyzers', 
        nargs='*', 
        default=['all'],
        choices=['all'] + list(BiodiversityAnalysisRunner('temp').available_analyzers.keys()),
        help='Analyzers to run (default: all)'
    )
    
    parser.add_argument(
        '--datasets',
        nargs='*',
        help='Dataset files in format name:path (e.g., plants:data/plants.tif)'
    )
    
    parser.add_argument(
        '--sample-factor',
        type=int,
        default=1,
        help='Spatial sampling factor for large datasets (default: 1 = no sampling)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='outputs/biodiversity_analysis',
        help='Output directory (default: outputs/biodiversity_analysis)'
    )
    
    parser.add_argument(
        '--config-file',
        help='JSON/YAML config file with dataset paths'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Parse datasets
    if args.config_file:
        # Load from config file
        import json
        import yaml
        config_path = Path(args.config_file)
        
        if config_path.suffix.lower() == '.json':
            with open(config_path) as f:
                config = json.load(f)
        else:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        
        datasets = config.get('datasets', {})
    
    elif args.datasets:
        # Parse from command line
        datasets = {}
        for dataset_spec in args.datasets:
            if ':' in dataset_spec:
                name, path = dataset_spec.split(':', 1)
                datasets[name] = path
            else:
                logger.error(f"Invalid dataset format: {dataset_spec}. Use name:path")
                sys.exit(1)
    
    else:
        # Use sample datasets
        logger.warning("No datasets specified, using sample configuration")
        datasets = create_sample_datasets()
    
    # Validate dataset paths
    valid_datasets = {}
    for name, path in datasets.items():
        if Path(path).exists():
            valid_datasets[name] = path
            logger.info(f"Found dataset: {name} -> {path}")
        else:
            logger.warning(f"Dataset not found: {name} -> {path}")
    
    if not valid_datasets:
        logger.error("No valid datasets found!")
        sys.exit(1)
    
    # Initialize runner
    runner = BiodiversityAnalysisRunner(args.output_dir)
    
    # Determine which analyses to run
    if 'all' in args.analyzers:
        analyzer_names = list(runner.available_analyzers.keys())
    else:
        analyzer_names = args.analyzers
    
    # Run analyses
    logger.info(f"Starting biodiversity analysis with {len(valid_datasets)} datasets")
    logger.info(f"Sample factor: {args.sample_factor}")
    logger.info(f"Output directory: {args.output_dir}")
    
    results = runner.run_multiple_analyses(
        analyzer_names,
        valid_datasets,
        args.sample_factor
    )
    
    # Final summary
    successful = sum(results.values())
    total = len(results)
    logger.info(f"FINAL RESULT: {successful}/{total} analyses completed successfully")
    
    if successful < total:
        sys.exit(1)


if __name__ == "__main__":
    main()