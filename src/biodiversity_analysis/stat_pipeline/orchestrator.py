"""Orchestrator for statistical analysis pipeline.

This module coordinates data loading and method execution for statistical
analysis of biodiversity data.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

from src.biodiversity_analysis.stat_pipeline.utils import (
    OptimizedStatLoader, 
    load_parquet_for_correlation,
    load_parquet_for_spatial
)
from src.biodiversity_analysis.stat_pipeline.raster_loader import (
    StatRasterLoader,
    load_rasters_for_trust_maps,
    load_rasters_for_global_maps
)

logger = logging.getLogger(__name__)


class StatPipelineOrchestrator:
    """Orchestrates statistical analysis pipeline with optimized data loading."""
    
    def __init__(self, config: Dict):
        """Initialize orchestrator with configuration.
        
        Args:
            config: Configuration dictionary with paths and settings
        """
        self.config = config
        self.parquet_loader = OptimizedStatLoader()
        self.raster_loader = StatRasterLoader()
        
        # Extract paths from config
        self.parquet_path = config.get('parquet_path')
        self.raster_paths = config.get('raster_paths', {})
        self.output_dir = Path(config.get('output_dir', 'outputs/stat_analysis'))
        
        # Analysis settings
        self.variant = config.get('variant', '4_features')
        self.sample_factor = config.get('sample_factor', 5)
        
        # Methods to run
        self.enabled_methods = config.get('methods', [
            'correlation', 'trust_maps', 'overlap', 
            'coverage', 'latitudinal', 'global_maps'
        ])
        
        logger.info(f"Initialized StatPipelineOrchestrator with variant: {self.variant}")
    
    def run_analysis(self):
        """Run all enabled statistical analysis methods."""
        logger.info("=" * 60)
        logger.info(f"Starting statistical analysis pipeline")
        logger.info(f"Variant: {self.variant}")
        logger.info(f"Enabled methods: {self.enabled_methods}")
        logger.info("=" * 60)
        
        # Prepare output directories
        self._prepare_output_dirs()
        
        # Load data once for all methods
        data = self._load_all_data()
        
        # Run each enabled method
        results = {}
        for method in self.enabled_methods:
            if method in self.get_available_methods():
                logger.info(f"\nRunning {method} analysis...")
                try:
                    result = self._run_method(method, data)
                    results[method] = result
                    logger.info(f"✓ {method} analysis completed")
                except Exception as e:
                    logger.error(f"✗ {method} analysis failed: {e}")
                    results[method] = {'status': 'failed', 'error': str(e)}
            else:
                logger.warning(f"Unknown method: {method}")
        
        # Generate summary report
        self._generate_summary(results)
        
        logger.info("\n" + "=" * 60)
        logger.info("Statistical analysis pipeline completed")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("=" * 60)
        
        return results
    
    def _load_all_data(self) -> Dict:
        """Load all required data formats once."""
        data = {}
        
        # Load parquet data if available
        if self.parquet_path and Path(self.parquet_path).exists():
            logger.info(f"Loading parquet data from: {self.parquet_path}")
            
            # Load as DataFrame for correlation/coverage
            data['dataframe'] = self.parquet_loader.load_parquet_for_stats(
                self.parquet_path, 
                self.variant, 
                'dataframe'
            )
            
            # Load as arrays for spatial methods if needed
            if any(m in self.enabled_methods for m in ['overlap', 'trust_maps']):
                data['arrays'] = self.parquet_loader.load_parquet_for_stats(
                    self.parquet_path,
                    self.variant,
                    'arrays'
                )
        
        # Load raster data if available and needed
        if self.raster_paths and any(m in self.enabled_methods for m in ['trust_maps', 'global_maps', 'latitudinal']):
            logger.info(f"Loading raster data for {len(self.raster_paths)} datasets")
            
            # Load aligned rasters for trust maps/overlap
            if 'trust_maps' in self.enabled_methods or 'overlap' in self.enabled_methods:
                data['rasters_aligned'] = self.raster_loader.load(
                    self.raster_paths,
                    self.variant,
                    self.sample_factor,
                    align=True
                )
            
            # Load with coordinates for global maps/latitudinal
            if 'global_maps' in self.enabled_methods or 'latitudinal' in self.enabled_methods:
                data['rasters_with_coords'] = []
                for name, path in self.raster_paths.items():
                    result = self.raster_loader.load_with_coordinates(path, self.sample_factor)
                    result['name'] = name
                    data['rasters_with_coords'].append(result)
        
        return data
    
    def _run_method(self, method: str, data: Dict) -> Dict:
        """Run a specific analysis method.
        
        Args:
            method: Method name
            data: Loaded data dictionary
            
        Returns:
            Method results
        """
        output_dir = self.output_dir / self.variant / method
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if method == 'correlation':
            return self._run_correlation(data.get('dataframe'), output_dir)
        elif method == 'trust_maps':
            return self._run_trust_maps(data.get('rasters_aligned', data.get('arrays')), output_dir)
        elif method == 'overlap':
            return self._run_overlap(data.get('rasters_aligned', data.get('arrays')), output_dir)
        elif method == 'coverage':
            return self._run_coverage(data, output_dir)
        elif method == 'latitudinal':
            return self._run_latitudinal(data.get('rasters_with_coords'), output_dir)
        elif method == 'global_maps':
            return self._run_global_maps(data.get('rasters_with_coords'), output_dir)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _run_correlation(self, df: pd.DataFrame, output_dir: Path) -> Dict:
        """Run correlation analysis."""
        from src.biodiversity_analysis.methods.stat.correlation_analysis.analyzer import CorrelationExplorer
        
        analyzer = CorrelationExplorer(str(output_dir))
        
        # Convert DataFrame to format expected by analyzer
        # The analyzer expects to do its own loading, so we need to adapt
        # For now, save to temp parquet and let it load
        # TODO: Refactor analyzer to accept DataFrame directly
        
        temp_path = output_dir / 'temp_data.parquet'
        df.to_parquet(temp_path)
        
        # Create fake dataset paths dict
        dataset_paths = {'temp': str(temp_path)}
        
        # Run analysis
        analyzer.run_correlation_analysis(dataset_paths)
        
        # Clean up temp file
        temp_path.unlink()
        
        return {'status': 'completed', 'output_dir': str(output_dir)}
    
    def _run_trust_maps(self, data: Union[Dict[str, np.ndarray], None], output_dir: Path) -> Dict:
        """Run trust maps analysis."""
        from src.biodiversity_analysis.methods.stat.trust_maps.analyzer import TrustMapVisualizer
        
        if data is None:
            raise ValueError("No data available for trust maps analysis")
        
        analyzer = TrustMapVisualizer(str(output_dir))
        
        # The analyzer expects file paths, but we have arrays
        # Need to adapt by creating the trust maps directly
        # TODO: Refactor analyzer to accept arrays directly
        
        # For now, create temporary paths dict
        temp_paths = {}
        for name in data.keys():
            temp_paths[name] = f"memory://{name}"  # Fake path
        
        # Monkey-patch the load method to use our data
        original_load = analyzer.load_and_align_data
        analyzer.load_and_align_data = lambda paths, sample_factor=10: (data, np.ones_like(list(data.values())[0], dtype=bool))
        
        analyzer.create_trust_maps(temp_paths)
        
        # Restore original method
        analyzer.load_and_align_data = original_load
        
        return {'status': 'completed', 'output_dir': str(output_dir)}
    
    def _run_overlap(self, data: Union[Dict[str, np.ndarray], None], output_dir: Path) -> Dict:
        """Run overlap analysis."""
        from src.biodiversity_analysis.methods.stat.overlap_analysis.analyzer import OverlapAnalyzer
        
        if data is None:
            raise ValueError("No data available for overlap analysis")
        
        analyzer = OverlapAnalyzer(str(output_dir))
        
        # Set aligned data directly
        analyzer.aligned_data = data
        
        # Run analysis methods
        analyzer.calculate_coverage_statistics()
        analyzer.calculate_concordance()
        analyzer.identify_hotspots()
        analyzer.create_overlap_visualizations()
        report = analyzer.generate_overlap_report()
        
        # Save report
        report_path = output_dir / 'overlap_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        return {'status': 'completed', 'output_dir': str(output_dir)}
    
    def _run_coverage(self, data: Dict, output_dir: Path) -> Dict:
        """Run coverage analysis."""
        from src.biodiversity_analysis.methods.stat.coverage_analysis.analyzer import CoverageAnalyzer
        
        # Coverage needs both parquet (for land mask) and rasters
        analyzer = CoverageAnalyzer(
            climate_parquet_path=self.parquet_path,
            output_dir=str(output_dir)
        )
        
        # Create dataset paths for analyzer
        dataset_paths = self.raster_paths if self.raster_paths else {'dataframe': 'memory://dataframe'}
        
        analyzer.analyze_coverage(dataset_paths, sample_factor=self.sample_factor)
        
        return {'status': 'completed', 'output_dir': str(output_dir)}
    
    def _run_latitudinal(self, data: List[Dict], output_dir: Path) -> Dict:
        """Run latitudinal analysis."""
        from src.biodiversity_analysis.methods.stat.latitudinal_analysis.analyzer import LatitudinalAnalyzer
        
        if not data:
            raise ValueError("No data available for latitudinal analysis")
        
        analyzer = LatitudinalAnalyzer(str(output_dir))
        
        # Convert our data format to what analyzer expects
        dataset_paths = {}
        for item in data:
            name = item['name']
            # Create temporary path
            dataset_paths[name] = f"memory://{name}"
        
        # Monkey-patch the _analyze_single_dataset method
        original_analyze = analyzer._analyze_single_dataset
        
        def analyze_with_data(raster_path, name, sample_factor=5):
            # Find our data
            for item in data:
                if item['name'] == name:
                    # Return in expected format
                    return {
                        'name': name,
                        'latitudes': item['lats'],
                        'means': np.nanmean(item['data'], axis=1),  # Mean across longitude
                        'stds': np.nanstd(item['data'], axis=1),
                        'counts': np.sum(~np.isnan(item['data']), axis=1)
                    }
            return original_analyze(raster_path, name, sample_factor)
        
        analyzer._analyze_single_dataset = analyze_with_data
        
        analyzer.analyze_latitudinal_patterns(dataset_paths, sample_factor=self.sample_factor)
        
        # Restore original method
        analyzer._analyze_single_dataset = original_analyze
        
        return {'status': 'completed', 'output_dir': str(output_dir)}
    
    def _run_global_maps(self, data: List[Dict], output_dir: Path) -> Dict:
        """Run global maps analysis."""
        from src.biodiversity_analysis.methods.stat.global_maps.analyzer import GlobalMapAnalyzer
        
        if not data:
            raise ValueError("No data available for global maps analysis")
        
        analyzer = GlobalMapAnalyzer(str(output_dir))
        
        # Convert to expected format
        datasets = {}
        for item in data:
            datasets[item['name']] = f"memory://{item['name']}"
        
        # Monkey-patch load_raster_data
        original_load = analyzer.load_raster_data
        
        def load_with_data(raster_path):
            # Extract name from fake path
            name = raster_path.replace("memory://", "")
            for item in data:
                if item['name'] == name:
                    return item['data'], {
                        'bounds': item['bounds'],
                        'transform': item['transform'],
                        'lon': item['lon_grid'],
                        'lat': item['lat_grid']
                    }
            return original_load(raster_path)
        
        analyzer.load_raster_data = load_with_data
        
        analyzer.create_global_maps(datasets)
        
        # Restore original
        analyzer.load_raster_data = original_load
        
        return {'status': 'completed', 'output_dir': str(output_dir)}
    
    def _prepare_output_dirs(self):
        """Create output directory structure."""
        for method in self.enabled_methods:
            method_dir = self.output_dir / self.variant / method
            method_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_summary(self, results: Dict):
        """Generate summary report of all analyses."""
        summary_path = self.output_dir / self.variant / 'analysis_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("STATISTICAL ANALYSIS PIPELINE SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Configuration:\n")
            f.write(f"  Variant: {self.variant}\n")
            f.write(f"  Sample Factor: {self.sample_factor}\n")
            f.write(f"  Output Directory: {self.output_dir}\n\n")
            
            f.write("Results:\n")
            for method, result in results.items():
                status = result.get('status', 'unknown')
                f.write(f"  {method}: {status}\n")
                if status == 'failed':
                    f.write(f"    Error: {result.get('error', 'Unknown error')}\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        logger.info(f"Summary report saved to: {summary_path}")
    
    @staticmethod
    def get_available_methods() -> List[str]:
        """Get list of available analysis methods."""
        return [
            'correlation',
            'trust_maps', 
            'overlap',
            'coverage',
            'latitudinal',
            'global_maps'
        ]


def run_stat_pipeline(config_path: str):
    """Run statistical analysis pipeline from configuration file.
    
    Args:
        config_path: Path to YAML configuration file
    """
    import yaml
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create and run orchestrator
    orchestrator = StatPipelineOrchestrator(config)
    return orchestrator.run_analysis()