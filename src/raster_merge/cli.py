#!/usr/bin/env python
from src.abstractions import get_logger
"""
Command-line interface for raster merging.

Simple, explicit, no magic.
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import List
import yaml

from .config import RasterConfig
from .merger_raster.vrt_merger import VRTMerger
from .pipeline import RasterPipeline


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Industrial-grade raster merging using rioxarray",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic merge
  python -m raster_merge input1.tif input2.tif -o output_dir/
  
  # With specific options
  python -m raster_merge *.tif -o output/ --method mean --compress zstd
  
  # Using config file
  python -m raster_merge --config config.yaml
  
  # Large files are handled automatically by VRTMerger
  python -m raster_merge huge1.tif huge2.tif -o output/
        """
    )
    
    # Input/output arguments
    parser.add_argument(
        'inputs', 
        nargs='*',
        help='Input raster files (GeoTIFF)'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        default=Path('./output'),
        help='Output directory (default: ./output)'
    )
    
    parser.add_argument(
        '--output-name',
        default='merged_raster',
        help='Output filename without extension (default: merged_raster)'
    )
    
    # Processing options
    parser.add_argument(
        '--resolution',
        choices=['equal', 'average', 'lowest', 'highest'],
        default='average',
        help='Resolution strategy for merging (default: average)'
    )
    
    parser.add_argument(
        '--method',
        choices=['first', 'last', 'min', 'max', 'mean'],
        default='first',
        help='Method for handling overlaps (default: first)'
    )
    
    parser.add_argument(
        '--resampling',
        choices=['nearest', 'bilinear', 'cubic', 'average'],
        default='bilinear',
        help='Resampling method (default: bilinear)'
    )
    
    # Memory optimization
    parser.add_argument(
        '--chunked',
        action='store_true',
        help='Use chunked processing for large files'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=2048,
        help='Chunk size in pixels (default: 2048)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    
    parser.add_argument(
        '--no-dask',
        action='store_true',
        help='Disable dask for lazy loading'
    )
    
    # Output options
    parser.add_argument(
        '--compress',
        choices=['zstd', 'deflate', 'lzw', 'none'],
        default='zstd',
        help='Compression method (default: zstd)'
    )
    
    parser.add_argument(
        '--no-geotiff',
        action='store_true',
        help='Skip GeoTIFF export'
    )
    
    parser.add_argument(
        '--no-arrow',
        action='store_true',
        help='Skip Arrow/Parquet export'
    )
    
    parser.add_argument(
        '--no-overviews',
        action='store_true',
        help='Skip building overviews'
    )
    
    # Config file
    parser.add_argument(
        '--config',
        type=Path,
        help='Load configuration from YAML/JSON file'
    )
    
    # Utility
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration without processing'
    )
    
    return parser.parse_args()


def load_config_file(config_path: Path) -> dict:
    """Load configuration from YAML or JSON file."""
    with open(config_path) as f:
        if config_path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif config_path.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unknown config format: {config_path.suffix}")


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.log_level)
    logger = get_logger(__name__)
    
    try:
        # Build configuration
        if args.config:
            # Load from file
            logger.info(f"Loading config from: {args.config}")
            config_dict = load_config_file(args.config)
            config = RasterConfig.from_dict(config_dict)
        else:
            # Build from command line arguments
            if not args.inputs:
                logger.error("No input files specified")
                sys.exit(1)
            
            # Convert compression option
            compress = None if args.compress == 'none' else args.compress
            
            # Map resampling string to enum
            from rasterio.enums import Resampling
            resampling_map = {
                'nearest': Resampling.nearest,
                'bilinear': Resampling.bilinear,
                'cubic': Resampling.cubic,
                'average': Resampling.average
            }
            
            config = RasterConfig(
                input_paths=args.inputs,
                output_dir=args.output_dir,
                output_name=args.output_name,
                resolution_strategy=args.resolution,
                overlap_method=args.method,
                resampling=resampling_map[args.resampling],
                chunk_size=args.chunk_size,
                use_dask=not args.no_dask,
                n_workers=args.workers,
                export_geotiff=not args.no_geotiff,
                export_arrow=not args.no_arrow,
                compress=compress,
                build_overviews=not args.no_overviews
            )
        
        # Show configuration in dry run
        if args.dry_run:
            logger.info("Configuration (dry run):")
            for key, value in config.to_dict().items():
                logger.info(f"  {key}: {value}")
            return
        
        # Process based on mode
        if args.chunked:
            # Note: --chunked flag is ignored, VRTMerger handles large files efficiently
            logger.info("Note: VRTMerger automatically handles large files efficiently")
            logger.info("The --chunked flag is deprecated and will be removed")
            
        # Use VRTMerger for all cases (handles both small and large files)
        if hasattr(config, 'input_paths') and config.input_paths:
            logger.info(f"Merging {len(config.input_paths)} rasters using VRTMerger")
            merger = VRTMerger(config)
            output_path = merger.merge()
            logger.info(f"Merged output: {output_path}")
            
            # Additional processing if needed
            if config.export_arrow:
                # Use pipeline for additional exports
                pipeline = RasterPipeline()
                pipeline.config = config
                pipeline.set_inputs(config.input_paths)
                pipeline.export(output_path, format="parquet")
                result = pipeline.run()
                if result.metadata.get('exported_paths'):
                    for path in result.metadata['exported_paths']:
                        if str(path).endswith('.parquet'):
                            logger.info(f"Arrow output: {path}")
        else:
            # Fallback for other configurations
            logger.warning("No input paths specified in config")
        
        logger.info("Processing complete")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()