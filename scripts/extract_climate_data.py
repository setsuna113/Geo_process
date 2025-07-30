#!/usr/bin/env python3
"""
Standalone Climate Data Extraction Script

Ultra-standalone script for extracting WorldClim bioclimatic variables
using Google Earth Engine. Integrates with existing pipeline coordinate
system and logging infrastructure.

Usage:
    python scripts/extract_climate_data.py --output data/climate/
    python scripts/extract_climate_data.py --bounds -10,-10,10,10 --output data/climate_test/
    python scripts/extract_climate_data.py --resume --output data/climate/
"""

import sys
import argparse
import os
from pathlib import Path
from typing import Tuple, List, Optional
import json
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import GEE module components
from src.climate_gee import GEEClimateExtractor, CoordinateGenerator, ParquetConverter
from src.climate_gee.auth import setup_gee_auth
from src.climate_gee.coordinate_generator import load_config_resolution, get_processing_bounds


def setup_logging(log_level: str = 'INFO') -> any:
    """Setup logging using project's infrastructure."""
    try:
        from src.infrastructure.logging import setup_simple_logging, get_logger
        setup_simple_logging(log_level)
        return get_logger(__name__)
    except ImportError:
        # Fallback to basic logging if project logging not available
        import logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)


def validate_file_path(file_path: str, description: str) -> Path:
    """Validate file path for security (prevent path traversal attacks)."""
    try:
        path = Path(file_path).resolve()
        
        # Check for path traversal attempts
        if '..' in str(path) or str(path).startswith('/'):
            # Allow absolute paths only if they don't contain traversal
            if not str(path).startswith('/') and '..' in str(path):
                raise ValueError(f"Path traversal not allowed in {description}")
        
        # Ensure the path exists for input files
        if description.lower().startswith('input') and not path.exists():
            raise FileNotFoundError(f"{description} file not found: {path}")
            
        return path
        
    except Exception as e:
        raise ValueError(f"Invalid {description} path: {e}")


def parse_bounds(bounds_str: str) -> Tuple[float, float, float, float]:
    """Parse bounds string to tuple."""
    try:
        parts = bounds_str.split(',')
        if len(parts) != 4:
            raise ValueError("Bounds must have 4 values: min_x,min_y,max_x,max_y")
        
        bounds = tuple(float(x.strip()) for x in parts)
        
        # Validate geographic bounds
        min_x, min_y, max_x, max_y = bounds
        if not (-180 <= min_x <= 180) or not (-180 <= max_x <= 180):
            raise ValueError("Longitude must be between -180 and 180")
        if not (-90 <= min_y <= 90) or not (-90 <= max_y <= 90):
            raise ValueError("Latitude must be between -90 and 90")
        if min_x >= max_x or min_y >= max_y:
            raise ValueError("Invalid bounds: min values must be less than max values")
            
        return bounds
        
    except Exception as e:
        raise ValueError(f"Invalid bounds format: {e}")


def create_checkpoint_file(output_dir: Path, config: dict) -> Path:
    """Create checkpoint file for resume capability."""
    checkpoint_path = output_dir / 'extraction_checkpoint.json'
    
    checkpoint_data = {
        'created_at': time.time(),
        'config': config,
        'status': 'started',
        'chunks_completed': 0,
        'total_chunks': None,
        'last_chunk_id': None
    }
    
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    return checkpoint_path


def load_checkpoint(checkpoint_path: Path) -> Optional[dict]:
    """Load checkpoint data if available."""
    if not checkpoint_path.exists():
        return None
    
    try:
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def update_checkpoint(checkpoint_path: Path, **updates):
    """Update checkpoint file with new data."""
    try:
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
        else:
            checkpoint = {}
        
        checkpoint.update(updates)
        checkpoint['updated_at'] = time.time()
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to update checkpoint: {e}")


def main():
    """Main extraction script."""
    parser = argparse.ArgumentParser(
        description='Extract WorldClim climate data using Google Earth Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction using config bounds
  python scripts/extract_climate_data.py --output data/climate/
  
  # Extract specific region
  python scripts/extract_climate_data.py --bounds -10,-10,10,10 --output data/climate_test/
  
  # Resume interrupted extraction
  python scripts/extract_climate_data.py --resume --output data/climate/
  
  # Test with small area
  python scripts/extract_climate_data.py --test --output data/climate_test/
  
  # Use service account authentication
  python scripts/extract_climate_data.py --service-account /path/to/key.json --output data/climate/
        """)
    
    parser.add_argument('--output', required=True, type=str,
                       help='Output directory for climate data')
    
    parser.add_argument('--bounds', type=str,
                       help='Geographic bounds as min_x,min_y,max_x,max_y (overrides config)')
    
    parser.add_argument('--config', type=str, default='config.yml',
                       help='Path to config.yml file (default: config.yml)')
    
    parser.add_argument('--chunk-size', type=int, default=5000,
                       help='Points per GEE request (max 5000, default: 5000)')
    
    parser.add_argument('--variables', nargs='+', 
                       choices=['bio01', 'bio04', 'bio12'],
                       default=['bio01', 'bio04', 'bio12'],
                       help='Climate variables to extract')
    
    parser.add_argument('--service-account', type=str,
                       help='Path to GEE service account JSON key file')
    
    parser.add_argument('--project-id', type=str,
                       help='Google Earth Engine project ID')
    
    parser.add_argument('--resume', action='store_true',
                       help='Resume interrupted extraction')
    
    parser.add_argument('--test', action='store_true',
                       help='Run test extraction on small area')
    
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    parser.add_argument('--experiment-id', type=str,
                       help='Experiment ID for output filenames')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting GEE climate data extraction")
    
    # Validate and create output directory
    try:
        output_dir = validate_file_path(args.output, "output directory")
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Invalid output directory: {e}")
        return 1
    
    try:
        # Check for resume
        checkpoint_path = output_dir / 'extraction_checkpoint.json'
        checkpoint = load_checkpoint(checkpoint_path) if args.resume else None
        
        if checkpoint:
            logger.info(f"Resuming extraction from checkpoint")
            config_data = checkpoint.get('config', {})
            bounds = tuple(config_data.get('bounds', [-180, -90, 180, 90]))
            variables = config_data.get('variables', args.variables)
            chunk_size = config_data.get('chunk_size', args.chunk_size)
        else:
            # Configure extraction parameters
            if args.test:
                logger.info("Running test extraction")
                bounds = (-1, -1, 1, 1)  # Small test area
                variables = ['bio01']  # Just temperature for testing
            elif args.bounds:
                bounds = parse_bounds(args.bounds)
                variables = args.variables
            else:
                # Load bounds from config
                bounds = get_processing_bounds(args.config, 'global')
                variables = args.variables
            
            chunk_size = min(args.chunk_size, 5000)  # GEE limit
            
            # Create new checkpoint
            config_data = {
                'bounds': bounds,
                'variables': variables,
                'chunk_size': chunk_size,
                'config_path': args.config,
                'service_account': args.service_account,
                'project_id': args.project_id
            }
            checkpoint_path = create_checkpoint_file(output_dir, config_data)
        
        logger.info(f"Extraction parameters:")
        logger.info(f"  Bounds: {bounds}")
        logger.info(f"  Variables: {variables}")
        logger.info(f"  Chunk size: {chunk_size}")
        logger.info(f"  Output: {output_dir}")
        
        # Validate file paths
        validated_service_account = None
        if args.service_account:
            try:
                validated_service_account = str(validate_file_path(args.service_account, "input service account key"))
            except Exception as e:
                logger.error(f"Invalid service account key path: {e}")
                return 1
        
        validated_config = args.config
        if args.config != 'config.yml':  # Only validate if not default
            try:
                validated_config = str(validate_file_path(args.config, "input config"))
            except Exception as e:
                logger.error(f"Invalid config file path: {e}")
                return 1
        
        # Setup GEE authentication
        logger.info("Setting up Google Earth Engine authentication...")
        auth = setup_gee_auth(
            service_account_key=validated_service_account,
            project_id=args.project_id,
            logger=logger
        )
        
        if not auth:
            logger.error("Failed to authenticate with Google Earth Engine")
            return 1
        
        # Test GEE connection
        if not auth.test_connection():
            logger.error("GEE connection test failed")
            return 1
        
        # Create coordinate generator
        logger.info("Creating coordinate generator...")
        resolution = load_config_resolution(validated_config)
        coord_gen = CoordinateGenerator(target_resolution=resolution, logger=logger)
        
        # Create GEE extractor
        logger.info("Creating GEE climate extractor...")
        extractor = GEEClimateExtractor(
            authenticator=auth,
            coordinate_generator=coord_gen,
            chunk_size=chunk_size,
            logger=logger
        )
        
        # Run extraction
        logger.info("Starting climate data extraction...")
        start_time = time.time()
        
        update_checkpoint(checkpoint_path, status='extracting', started_at=start_time)
        
        climate_data = extractor.extract_climate_data(
            bounds=bounds,
            variables=variables,
            output_dir=str(output_dir)
        )
        
        extraction_duration = time.time() - start_time
        
        if climate_data.empty:
            logger.error("No climate data extracted")
            update_checkpoint(checkpoint_path, status='failed', error='No data extracted')
            return 1
        
        logger.info(f"Extraction completed in {extraction_duration:.2f}s")
        
        # Convert to parquet
        logger.info("Converting to parquet format...")
        converter = ParquetConverter(logger=logger)
        
        parquet_path = output_dir / 'climate_data.parquet'
        export_stats = converter.convert_to_parquet(
            climate_data=climate_data,
            output_path=parquet_path,
            experiment_id=args.experiment_id,
            validate_schema=True,
            include_metadata=True
        )
        
        # Update checkpoint with success
        update_checkpoint(
            checkpoint_path, 
            status='completed',
            extraction_duration=extraction_duration,
            total_points=len(climate_data),
            output_file=str(export_stats['output_path']),
            export_stats=export_stats
        )
        
        # Generate summary report
        logger.info("="*60)
        logger.info("EXTRACTION SUMMARY")
        logger.info("="*60)
        logger.info(f"Status: SUCCESS")
        logger.info(f"Bounds: {bounds}")
        logger.info(f"Variables: {', '.join(variables)}")
        logger.info(f"Total points: {len(climate_data):,}")
        logger.info(f"Output file: {export_stats['output_path']}")
        logger.info(f"File size: {export_stats['file_size_mb']:.2f} MB")
        logger.info(f"Extraction time: {extraction_duration:.2f}s")
        logger.info(f"Conversion time: {export_stats['conversion_duration']:.2f}s")
        logger.info("="*60)
        
        # Validate output
        validation_results = converter.validate_parquet_compatibility(export_stats['output_path'])
        if validation_results['valid']:
            logger.info("Output validation: PASSED")
        else:
            logger.warning(f"Output validation issues: {validation_results['issues']}")
        
        logger.info("Climate data extraction completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Extraction interrupted by user")
        update_checkpoint(checkpoint_path, status='interrupted')
        return 1
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        update_checkpoint(checkpoint_path, status='failed', error=str(e))
        return 1


if __name__ == '__main__':
    sys.exit(main())