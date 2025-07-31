"""
Biodiversity SOM Analysis Module

Simple, clean implementation following real VLRSOM research patterns.

Key components:
1. SimpleVLRSOM - Real VLRSOM training (QE primary, TE monitoring)
2. BiodiversitySpatialSplitter - Spatial validation (handles autocorrelation)
3. ManhattanSOM - Optimal distance metric for species data
4. BiodiversitySOMAnalyzer - Complete analysis pipeline
5. SOMAnalysisReporter - Clean reporting
"""

# Core SOM components
from .manhattan_som import ManhattanSOM, ManhattanSOMWrapper, create_manhattan_som

# Simple VLRSOM implementation (following real research)
from .simple_vlrsom import SimpleVLRSOM, VLRSOMResult, create_simple_vlrsom

# Spatial validation (handles autocorrelation)
from .spatial_validation import (
    BiodiversitySpatialSplitter, 
    SpatialDataSplit, 
    SpatialSplitStrategy,
    create_spatial_splitter
)

# Complete analysis pipeline
from .biodiversity_som_analyzer import BiodiversitySOMAnalyzer, create_biodiversity_som_analyzer

# Clean reporting
from .som_reporter_clean import SOMAnalysisReporter, create_som_reporter

# Convenience function for complete analysis
def analyze_biodiversity_som(parquet_path, config=None, output_dir=None):
    """
    Convenience function for complete biodiversity SOM analysis.
    
    Args:
        parquet_path: Path to parquet file with biodiversity data
        config: Configuration dict or object (optional)
        output_dir: Output directory for reports (optional)
        
    Returns:
        Tuple of (AnalysisResult, report_files)
    """
    # Use default config if none provided
    if config is None:
        config = {
            'som_analysis': {
                'default_grid_size': [10, 10],
                'iterations': 2000,
                'sigma': 1.0,
                'learning_rate': 0.5,
                'vlrsom': {
                    'qe_threshold': 1e-6,
                    'te_threshold': 0.05,
                    'patience': 50
                },
                'validation': {
                    'enabled': True,
                    'spatial_split_strategy': 'latitudinal',
                    'train_ratio': 0.7,
                    'validation_ratio': 0.15,
                    'test_ratio': 0.15
                }
            }
        }
    
    # Create analyzer and run analysis
    analyzer = create_biodiversity_som_analyzer(config)
    result = analyzer.analyze(parquet_path)
    
    # Generate reports
    reporter = create_som_reporter(output_dir)
    report_files = reporter.generate_report(result)
    reporter.print_summary(result)
    
    return result, report_files


__all__ = [
    # Core SOM
    'ManhattanSOM',
    'ManhattanSOMWrapper', 
    'create_manhattan_som',
    
    # VLRSOM
    'SimpleVLRSOM',
    'VLRSOMResult',
    'create_simple_vlrsom',
    
    # Spatial validation
    'BiodiversitySpatialSplitter',
    'SpatialDataSplit',
    'SpatialSplitStrategy',
    'create_spatial_splitter',
    
    # Complete analyzer
    'BiodiversitySOMAnalyzer',
    'create_biodiversity_som_analyzer',
    
    # Reporting
    'SOMAnalysisReporter',
    'create_som_reporter',
    
    # Convenience function
    'analyze_biodiversity_som'
]