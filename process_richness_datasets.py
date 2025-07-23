#!/usr/bin/env python3
"""
Process and analyze richness datasets using the spatial analysis framework.

This script:
1. Loads the two richness datasets into the database catalog
2. Merges them into a single multi-band dataset
3. Conducts SOM analysis
4. Stores results in outputs/
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.config import Config
from src.database.connection import DatabaseManager
from src.database.setup import setup_database
from src.raster.catalog import RasterCatalog
from src.processors.data_preparation.raster_merger import RasterMerger
from src.spatial_analysis.som.som_trainer import SOMAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command-line arguments for memory and processing control."""
    parser = argparse.ArgumentParser(
        description="Process richness datasets with memory-aware spatial analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Memory control options
    parser.add_argument('--max-samples', type=int, default=500000,
                        help='Maximum samples for in-memory processing')
    parser.add_argument('--sampling-strategy', choices=['random', 'stratified', 'grid'],
                        default='stratified', help='Subsampling strategy for large datasets')
    parser.add_argument('--memory-limit', type=float, default=8.0,
                        help='Memory limit in GB')
    parser.add_argument('--batch-processing', action='store_true',
                        help='Use batch processing for very large datasets')
    parser.add_argument('--chunk-size', type=int, default=50000,
                        help='Chunk size for batch processing')
    
    # SOM parameters
    parser.add_argument('--som-grid-size', type=int, nargs=2, default=[10, 10],
                        help='SOM grid dimensions [width height]')
    parser.add_argument('--som-iterations', type=int, default=1000,
                        help='Number of SOM training iterations')
    
    # Processing options
    parser.add_argument('--skip-merge', action='store_true',
                        help='Skip raster merging step (use existing merged data)')
    parser.add_argument('--output-dir', type=str, default='outputs/spatial_analysis',
                        help='Output directory for results')
    parser.add_argument('--spatial-subset', type=float, nargs=4, metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'),
                        help='Process only spatial subset (longitude/latitude bounds)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.memory_limit <= 0:
        parser.error("--memory-limit must be positive")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.som_grid_size[0] <= 0 or args.som_grid_size[1] <= 0:
        parser.error("--som-grid-size dimensions must be positive")
    if args.som_iterations <= 0:
        parser.error("--som-iterations must be positive")
    
    return args

def main():
    """Main processing pipeline."""
    args = parse_arguments()
    
    logger.info("🚀 Starting richness dataset processing pipeline")
    logger.info(f"Memory limit: {args.memory_limit} GB")
    logger.info(f"Max samples: {args.max_samples:,}")
    logger.info(f"Sampling strategy: {args.sampling_strategy}")
    
    # Initialize configuration and database
    config = Config()
    
    # Ensure config object has proper structure
    if not hasattr(config, 'config'):
        config.config = {}
    
    # Update config with command-line arguments using safer nested updates
    processing_config = config.config.setdefault('processing', {})
    subsampling_config = processing_config.setdefault('subsampling', {})
    subsampling_config.update({
        'enabled': True,
        'max_samples': args.max_samples,
        'strategy': args.sampling_strategy,
        'memory_limit_gb': args.memory_limit
    })
    
    # Configure spatial analysis memory limits
    spatial_analysis_config = config.config.setdefault('spatial_analysis', {})
    spatial_analysis_config.update({
        'batch_size': args.chunk_size,
        'memory_limit_mb': int(args.memory_limit * 1024),  # Convert GB to MB
        'normalize_data': True,
        'save_results': True
    })
    
    # Configure data preparation memory limits
    data_prep_config = config.config.setdefault('data_preparation', {})
    data_prep_config.update({
        'memory_limit_gb': args.memory_limit,
        'chunk_processing': args.batch_processing,
        'chunk_size': args.chunk_size,
        'use_memory_mapping': args.memory_limit <= 4.0  # Use memory mapping for low memory systems
    })
    
    som_config = config.config.setdefault('som_analysis', {})
    som_config.update({
        'max_pixels_in_memory': min(args.max_samples, 1000000),
        'use_memory_mapping': args.memory_limit <= 16.0,  # Use memory mapping for systems with <= 16GB
        'batch_training': {
            'enabled': args.batch_processing,
            'batch_size': args.chunk_size
        }
    })
    
    db = DatabaseManager()
    
    # Ensure database is ready
    logger.info("Checking database setup...")
    try:
        if not db.test_connection():
            logger.error("Database connection failed")
            return False
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return False
    
    # Define dataset paths
    data_dir = Path("data/richness_maps")
    daru_path = data_dir / "daru-plants-richness.tif"
    iucn_path = data_dir / "iucn-terrestrial-richness.tif"
    
    # Verify files exist
    if not daru_path.exists():
        logger.error(f"Plants dataset not found: {daru_path}")
        return False
    
    if not iucn_path.exists():
        logger.error(f"Terrestrial dataset not found: {iucn_path}")
        return False
    
    logger.info(f"✅ Found datasets:\n  - Plants: {daru_path}\n  - Terrestrial: {iucn_path}")
    
    try:
        # Initialize components
        catalog = RasterCatalog(db, config)
        merger = RasterMerger(config, db)
        som_analyzer = SOMAnalyzer(config, db)
        
        # Step 1: Load datasets into catalog
        logger.info("📥 Loading datasets into catalog...")
        
        # Add datasets to catalog if not already present
        plants_entry = None
        terrestrial_entry = None
        
        try:
            plants_entry = catalog.get_raster("daru-plants-richness")
        except:
            pass
        
        if plants_entry is None:
            logger.info("Adding plants dataset to catalog...")
            plants_entry = catalog.add_raster(
                daru_path, 
                dataset_type="plants",
                validate=True
            )
            logger.info(f"✅ Added plants dataset: {plants_entry.name}")
        else:
            logger.info("✅ Plants dataset already in catalog")
        
        try:
            terrestrial_entry = catalog.get_raster("iucn-terrestrial-richness")
        except:
            pass
        
        if terrestrial_entry is None:
            logger.info("Adding terrestrial dataset to catalog...")
            terrestrial_entry = catalog.add_raster(
                iucn_path,
                dataset_type="terrestrial", 
                validate=True
            )
            logger.info(f"✅ Added terrestrial dataset: {terrestrial_entry.name}")
        else:
            logger.info("✅ Terrestrial dataset already in catalog")
        
        # Step 2: Merge datasets into multi-band format
        logger.info("🔗 Merging datasets into multi-band format...")
        
        merged_result = merger.merge_custom_rasters(
            raster_names={
                'plants_richness': plants_entry.name,
                'terrestrial_richness': terrestrial_entry.name
            },
            band_names=['plants_richness', 'terrestrial_richness']
        )
        
        merged_data = merged_result['data']
        logger.info(f"✅ Merged data shape: {dict(merged_data.sizes)}")
        logger.info(f"   Bands: {list(merged_data.data_vars)}")
        
        # Step 3: Conduct SOM analysis
        logger.info("🧠 Running SOM analysis...")
        
        # Configure SOM parameters
        som_params = {
            'grid_size': [8, 8],  # 8x8 SOM grid
            'iterations': 1000,
            'sigma': 1.5,
            'learning_rate': 0.5,
            'neighborhood_function': 'gaussian',
            'random_seed': 42
        }
        
        logger.info(f"SOM parameters: {som_params}")
        
        # Run SOM analysis
        som_result = som_analyzer.analyze(
            data=merged_data,
            **som_params
        )
        
        # Step 4: Save results
        logger.info("💾 Saving results...")
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f"outputs/spatial_analysis/Richness_SOM_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save using the analyzer's built-in save functionality
        saved_path = som_analyzer.save_results(som_result, f"Richness_SOM_{timestamp}")
        
        logger.info(f"✅ Results saved to: {saved_path}")
        
        # Print summary statistics
        stats = som_result.statistics
        logger.info("📊 Analysis Summary:")
        logger.info(f"   Grid size: {stats['grid_size']}")
        logger.info(f"   Number of clusters: {stats['n_clusters']}")
        logger.info(f"   Quantization error: {stats['quantization_error']:.4f}")
        logger.info(f"   Topographic error: {stats['topographic_error']:.4f}")
        logger.info(f"   Empty neurons: {stats['empty_neurons']}")
        logger.info(f"   Cluster balance: {stats['cluster_balance']:.4f}")
        
        # Log cluster statistics
        cluster_stats = stats['cluster_statistics']
        logger.info("   Cluster distribution:")
        for cluster_id, cluster_info in cluster_stats.items():
            logger.info(f"     Cluster {cluster_id}: {cluster_info['count']} pixels ({cluster_info['percentage']:.1f}%)")
        
        logger.info("🎉 Processing pipeline completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)