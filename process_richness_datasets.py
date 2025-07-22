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

def main():
    """Main processing pipeline."""
    logger.info("🚀 Starting richness dataset processing pipeline")
    
    # Initialize configuration and database
    config = Config()
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