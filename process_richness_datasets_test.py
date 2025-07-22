#!/usr/bin/env python3
"""
Test version of richness dataset processing with limited data size.

This script processes only a small subset of the raster data to verify
the pipeline works before running on the full datasets.
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
from src.processors.data_preparation.raster_alignment import RasterAligner, AlignmentConfig
from src.spatial_analysis.som.som_trainer import SOMAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_aligned_test_subsets(daru_path: Path, iucn_path: Path, test_daru_path: Path, test_iucn_path: Path, max_size: int = 100):
    """
    Create aligned test subsets using the robust RasterAligner.
    
    Args:
        daru_path: Original DARU plants raster
        iucn_path: Original IUCN terrestrial raster
        test_daru_path: Output path for DARU subset
        test_iucn_path: Output path for IUCN subset
        max_size: Maximum dimension size (pixels)
    """
    logger.info(f"Creating aligned test subsets using RasterAligner...")
    
    # Configure alignment
    config = AlignmentConfig(
        resolution_tolerance=1e-6,
        bounds_tolerance=1e-4
    )
    
    aligner = RasterAligner(config)
    
    # Create test subset directory
    subset_dir = test_daru_path.parent
    subset_dir.mkdir(parents=True, exist_ok=True)
    
    # Create aligned subsets
    raster_paths = [daru_path, iucn_path]
    
    # Use the aligner's create_aligned_subsets method
    subset_paths = aligner.create_aligned_subsets(
        raster_paths=raster_paths,
        output_dir=subset_dir,
        subset_size=max_size
    )
    
    # Map outputs to expected paths
    daru_subset = subset_paths[str(daru_path)]
    iucn_subset = subset_paths[str(iucn_path)]
    
    # Rename to expected names
    daru_subset.rename(test_daru_path)
    iucn_subset.rename(test_iucn_path)
    
    logger.info(f"✅ Created perfectly aligned subsets using RasterAligner")
    logger.info(f"   DARU test: {test_daru_path}")
    logger.info(f"   IUCN test: {test_iucn_path}")
    
    # Verify alignment
    report = aligner.analyze_alignment([test_daru_path, test_iucn_path])
    if report.aligned:
        logger.info("✅ Alignment verified: subsets are perfectly aligned")
    else:
        logger.warning(f"⚠️ Alignment issues remain: {len(report.issues)} issues")
        for issue in report.issues:
            logger.warning(f"   {issue.type}: {issue.description}")
    
    return test_daru_path, test_iucn_path

def main():
    """Main testing pipeline."""
    logger.info("🧪 Starting TESTING version of richness dataset processing")
    
    # Initialize configuration and database
    config = Config()
    db = DatabaseManager()
    
    # Check database connection
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
    
    # Create test data directory
    test_data_dir = Path("data/test_subsets")
    test_daru_path = test_data_dir / "test_daru-plants-richness.tif"
    test_iucn_path = test_data_dir / "test_iucn-terrestrial-richness.tif"
    
    # Verify original files exist
    if not daru_path.exists():
        logger.error(f"Plants dataset not found: {daru_path}")
        return False
    
    if not iucn_path.exists():
        logger.error(f"Terrestrial dataset not found: {iucn_path}")
        return False
    
    try:
        # Step 0: Create aligned test subsets (100x100 pixels each)
        logger.info("🔬 Creating aligned test data subsets...")
        
        # Always recreate subsets to ensure alignment
        if test_daru_path.exists():
            test_daru_path.unlink()  # Remove existing file
        if test_iucn_path.exists():
            test_iucn_path.unlink()  # Remove existing file
            
        create_aligned_test_subsets(daru_path, iucn_path, test_daru_path, test_iucn_path, max_size=100)
        
        # Initialize components
        catalog = RasterCatalog(db, config)
        merger = RasterMerger(config, db)
        som_analyzer = SOMAnalyzer(config, db)
        
        # Step 1: Load test datasets into catalog
        logger.info("📥 Loading TEST datasets into catalog...")
        
        # Check if test datasets already in catalog
        plants_entry = None
        terrestrial_entry = None
        
        try:
            plants_entry = catalog.get_raster("test_daru-plants-richness")
        except:
            pass
        
        if plants_entry is None:
            logger.info("Adding TEST plants dataset to catalog...")
            plants_entry = catalog.add_raster(
                test_daru_path, 
                dataset_type="plants",
                validate=True
            )
            logger.info(f"✅ Added TEST plants dataset: {plants_entry.name}")
        else:
            logger.info("✅ TEST plants dataset already in catalog")
        
        try:
            terrestrial_entry = catalog.get_raster("test_iucn-terrestrial-richness")
        except:
            pass
        
        if terrestrial_entry is None:
            logger.info("Adding TEST terrestrial dataset to catalog...")
            terrestrial_entry = catalog.add_raster(
                test_iucn_path,
                dataset_type="terrestrial", 
                validate=True
            )
            logger.info(f"✅ Added TEST terrestrial dataset: {terrestrial_entry.name}")
        else:
            logger.info("✅ TEST terrestrial dataset already in catalog")
        
        # Step 2: Merge test datasets
        logger.info("🔗 Merging TEST datasets into multi-band format...")
        
        merged_result = merger.merge_custom_rasters(
            raster_names={
                'plants_richness': plants_entry.name,
                'terrestrial_richness': terrestrial_entry.name
            },
            band_names=['plants_richness', 'terrestrial_richness']
        )
        
        merged_data = merged_result['data']
        logger.info(f"✅ Merged TEST data shape: {dict(merged_data.sizes)}")
        logger.info(f"   Bands: {list(merged_data.data_vars)}")
        
        # Step 3: Conduct SOM analysis on test data
        logger.info("🧠 Running SOM analysis on TEST data...")
        
        # Use smaller SOM grid for test data
        som_params = {
            'grid_size': [4, 4],  # 4x4 SOM grid (smaller for test)
            'iterations': 100,    # Fewer iterations for faster testing
            'sigma': 1.0,
            'learning_rate': 0.5,
            'neighborhood_function': 'gaussian',
            'random_seed': 42
        }
        
        logger.info(f"TEST SOM parameters: {som_params}")
        
        # Run SOM analysis
        som_result = som_analyzer.analyze(
            data=merged_data,
            **som_params
        )
        
        # Step 4: Save test results
        logger.info("💾 Saving TEST results...")
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_path = som_analyzer.save_results(som_result, f"TEST_Richness_SOM_{timestamp}")
        
        logger.info(f"✅ TEST results saved to: {saved_path}")
        
        # Print summary statistics
        stats = som_result.statistics
        logger.info("📊 TEST Analysis Summary:")
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
        
        logger.info("🎉 TEST processing pipeline completed successfully!")
        logger.info("📈 Pipeline validation complete - ready for full dataset!")
        return True
        
    except Exception as e:
        logger.error(f"❌ TEST processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n" + "="*60)
        print("🎉 TEST SUCCESSFUL! Pipeline is working correctly.")
        print("📋 Next step: Run full dataset with process_richness_datasets.py")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ TEST FAILED! Check errors above.")
        print("="*60)
    sys.exit(0 if success else 1)