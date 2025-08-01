"""Command-line interface for rasterio resampler."""

import argparse
import sys
import os
import json
import logging
import daemon
import daemon.pidfile
from typing import Optional
from pathlib import Path

from .config import ResamplingConfig
from .resampler import RasterioResampler

logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Resample raster files using rasterio with memory management"
    )
    
    # Input/output arguments
    parser.add_argument('input_path', help='Path to input raster file')
    parser.add_argument('-o', '--output', help='Output path (auto-generated if not specified)')
    parser.add_argument('-d', '--output-dir', default='./resampled',
                       help='Output directory for resampled files (default: ./resampled)')
    
    # Resolution and projection
    parser.add_argument('-r', '--resolution', type=float, default=0.166744,
                       help='Target resolution in degrees (default: 0.166744, ~18.5km)')
    parser.add_argument('-c', '--crs', default='EPSG:4326',
                       help='Target CRS (default: EPSG:4326)')
    
    # Resampling method
    parser.add_argument('-m', '--method', default='average',
                       choices=['nearest', 'bilinear', 'cubic', 'cubic_spline', 
                               'lanczos', 'average', 'mode', 'max', 'min', 
                               'med', 'q1', 'q3', 'sum'],
                       help='Resampling method (default: average)')
    
    # Memory and CPU management
    parser.add_argument('--memory-limit', type=float,
                       help='Memory limit in GB (auto-detected if not specified)')
    parser.add_argument('--max-workers', type=int,
                       help='Maximum parallel workers (auto-detected if not specified)')
    parser.add_argument('--window-size', type=int, default=2048,
                       help='Window size for processing (default: 2048)')
    
    # Progress and debugging
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                       help='Save progress every N windows (default: 10)')
    parser.add_argument('--progress-file', default='resampling_progress.json',
                       help='Progress file for resume capability')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--log-file', help='Log to file')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Log level (default: INFO)')
    
    # Output options
    parser.add_argument('--compress', default='lzw',
                       choices=['none', 'lzw', 'deflate', 'packbits'],
                       help='Compression method (default: lzw)')
    parser.add_argument('--no-tiled', action='store_true',
                       help='Disable tiled output')
    parser.add_argument('--blocksize', type=int, default=512,
                       help='Block size for tiled output (default: 512)')
    
    # Background/daemon mode
    parser.add_argument('--daemon', action='store_true',
                       help='Run as background daemon')
    parser.add_argument('--pid-file', help='PID file for daemon mode')
    
    # Validation
    parser.add_argument('--validate', action='store_true',
                       help='Validate output after resampling')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without processing')
    
    args = parser.parse_args()
    
    # Create configuration
    config_dict = {
        'target_resolution': args.resolution,
        'target_crs': args.crs,
        'resampling_method': args.method,
        'memory_limit_gb': args.memory_limit,
        'max_workers': args.max_workers,
        'window_size': args.window_size,
        'checkpoint_interval': args.checkpoint_interval,
        'progress_file': args.progress_file,
        'debug': args.debug,
        'log_level': args.log_level,
        'log_file': args.log_file,
        'output_dir': args.output_dir,
        'compress': 'none' if args.compress == 'none' else args.compress,
        'tiled': not args.no_tiled,
        'blockxsize': args.blocksize,
        'blockysize': args.blocksize,
        'daemon': args.daemon,
        'pid_file': args.pid_file
    }
    
    config = ResamplingConfig.from_dict(config_dict)
    
    # Validate input
    if not os.path.exists(args.input_path):
        print(f"Error: Input file not found: {args.input_path}")
        sys.exit(1)
    
    # Dry run
    if args.dry_run:
        print("Dry run - configuration:")
        print(json.dumps(config.__dict__, indent=2, default=str))
        print(f"\nWould resample: {args.input_path}")
        print(f"Output: {args.output or 'auto-generated'}")
        sys.exit(0)
    
    # Run in daemon mode if requested
    if args.daemon:
        run_as_daemon(args.input_path, args.output, config, args.validate)
    else:
        run_resampling(args.input_path, args.output, config, args.validate)


def run_resampling(input_path: str, output_path: Optional[str], 
                   config: ResamplingConfig, validate: bool = False):
    """Run the resampling process."""
    try:
        # Create resampler
        resampler = RasterioResampler(config)
        
        # Perform resampling
        output_file = resampler.resample(input_path, output_path)
        
        print(f"\nResampling completed successfully!")
        print(f"Output: {output_file}")
        
        # Validate if requested
        if validate:
            print("\nValidating output...")
            validation = resampler.validate_output(output_file, input_path)
            
            if validation['valid']:
                print("✓ Validation passed")
            else:
                print("✗ Validation failed")
            
            if validation['warnings']:
                print("\nWarnings:")
                for warning in validation['warnings']:
                    print(f"  - {warning}")
            
            print("\nValidation info:")
            print(json.dumps(validation['info'], indent=2))
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}")
        if config.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_as_daemon(input_path: str, output_path: Optional[str], 
                  config: ResamplingConfig, validate: bool = False):
    """Run resampling as a background daemon.
    
    Uses atomic file operations to prevent PID file collisions.
    """
    import tempfile
    import fcntl
    
    # Set up directories
    pid_dir = Path.home() / '.biodiversity' / 'rasterio_resampler'
    pid_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path.home() / '.biodiversity' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary PID file atomically
    temp_fd, temp_pid_file = tempfile.mkstemp(
        suffix='.pid', 
        prefix='resampler_temp_', 
        dir=str(pid_dir)
    )
    
    try:
        # Try to acquire exclusive lock
        fcntl.flock(temp_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        
        # Generate final PID file name after acquiring lock
        import uuid
        import time
        daemon_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Set up paths
        pid_file = config.pid_file or str(pid_dir / f"resampler_{daemon_id}.pid")
        config.log_file = config.log_file or str(log_dir / f"resampler_{daemon_id}.log")
        
        print(f"Starting daemon process...")
        print(f"PID file: {pid_file}")
        print(f"Log file: {config.log_file}")
        
        # Close and remove temp file before daemon context
        os.close(temp_fd)
        os.unlink(temp_pid_file)
        
        # Create daemon context with final PID file
        with daemon.DaemonContext(
            pidfile=daemon.pidfile.PIDLockFile(pid_file),
            working_directory=os.getcwd(),
            umask=0o002,
            stdout=open(config.log_file, 'w+'),
            stderr=open(config.log_file, 'w+')
        ):
            # Run resampling in daemon
            run_resampling(input_path, output_path, config, validate)
            
    except IOError:
        os.close(temp_fd)
        os.unlink(temp_pid_file)
        print("Error: Another resampler daemon is starting. Please try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()