#!/usr/bin/env python3
"""
Raster data management CLI tool.

This tool provides command-line operations for managing raster data sources,
processing tiles, and managing resampling cache.
"""

import click
import logging
from pathlib import Path
from typing import Optional
import json
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.raster.manager import raster_manager
from src.database.schema import schema
from src.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def cli(verbose):
    """Raster data management CLI."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--name', '-n', help='Name for the raster source')
@click.option('--dataset', help='Source dataset name')
@click.option('--variable', help='Variable name')
@click.option('--units', help='Units of measurement')
@click.option('--description', help='Description of the raster')
@click.option('--auto-tile', is_flag=True, default=True, help='Automatically create tiles')
def register(file_path, name, dataset, variable, units, description, auto_tile):
    """Register a new raster data source."""
    try:
        file_path = Path(file_path)
        
        # Prepare metadata
        metadata = {}
        if dataset:
            metadata['source_dataset'] = dataset
        if variable:
            metadata['variable_name'] = variable
        if units:
            metadata['units'] = units
        if description:
            metadata['description'] = description
        
        # Register raster
        raster_id = raster_manager.register_raster_source(
            file_path=file_path,
            name=name,
            metadata=metadata if metadata else None
        )
        
        click.echo(f"✅ Registered raster source: {raster_id}")
        
        # Optionally create tiles
        if auto_tile:
            click.echo("Creating tiles...")
            tile_count = raster_manager.create_raster_tiles(raster_id)
            click.echo(f"✅ Created {tile_count} tiles")
            
    except Exception as e:
        click.echo(f"❌ Failed to register raster: {e}", err=True)
        raise click.Abort()

@cli.command()
@click.option('--status', help='Filter by processing status')
@click.option('--active-only/--all', default=True, help='Show only active rasters')
def list(status, active_only):
    """List registered raster sources."""
    try:
        sources = schema.get_raster_sources(
            active_only=active_only,
            processing_status=status
        )
        
        if not sources:
            click.echo("No raster sources found.")
            return
        
        # Display table header
        click.echo(f"{'Name':<30} {'Status':<15} {'Size (MB)':<12} {'Tiles':<8} {'Type':<10}")
        click.echo("-" * 80)
        
        # Get processing status for tile counts
        processing_status = {
            ps['raster_id']: ps for ps in raster_manager.get_processing_status()
        }
        
        for source in sources:
            raster_id = source['id']
            ps = processing_status.get(raster_id, {})
            
            click.echo(
                f"{source['name']:<30} "
                f"{source['processing_status']:<15} "
                f"{source.get('file_size_mb', 0):<12.1f} "
                f"{ps.get('total_tiles', 0):<8} "
                f"{source['data_type']:<10}"
            )
            
    except Exception as e:
        click.echo(f"❌ Failed to list rasters: {e}", err=True)
        raise click.Abort()

@cli.command()
@click.argument('raster_id')
@click.option('--force', is_flag=True, help='Force re-tiling even if already tiled')
def tile(raster_id, force):
    """Create tiles for a raster source."""
    try:
        click.echo(f"Creating tiles for raster {raster_id}...")
        tile_count = raster_manager.create_raster_tiles(raster_id, force_retile=force)
        click.echo(f"✅ Created {tile_count} tiles")
        
    except Exception as e:
        click.echo(f"❌ Failed to create tiles: {e}", err=True)
        raise click.Abort()

@cli.command()
@click.argument('raster_id')
@click.argument('grid_id')
@click.option('--method', default='bilinear', 
              type=click.Choice(['nearest', 'bilinear', 'cubic', 'average']),
              help='Resampling method')
@click.option('--band', default=1, type=int, help='Band number to resample')
@click.option('--cell-ids', help='Comma-separated list of specific cell IDs')
@click.option('--no-cache', is_flag=True, help='Skip cache lookup and storage')
@click.option('--output', '-o', type=click.Path(), help='Save results to JSON file')
def resample(raster_id, grid_id, method, band, cell_ids, no_cache, output):
    """Resample raster to grid cells."""
    try:
        # Parse cell IDs if provided
        cell_id_list = None
        if cell_ids:
            cell_id_list = [cid.strip() for cid in cell_ids.split(',')]
        
        click.echo(f"Resampling raster {raster_id} to grid {grid_id}...")
        click.echo(f"Method: {method}, Band: {band}, Use cache: {not no_cache}")
        
        results = raster_manager.resample_to_grid(
            raster_id=raster_id,
            grid_id=grid_id,
            cell_ids=cell_id_list,
            method=method,
            band_number=band,
            use_cache=not no_cache
        )
        
        click.echo(f"✅ Resampled {len(results)} cells")
        
        # Display sample results
        if results:
            click.echo("\nSample results:")
            for i, (cell_id, value) in enumerate(list(results.items())[:5]):
                click.echo(f"  {cell_id}: {value:.3f}")
            if len(results) > 5:
                click.echo(f"  ... and {len(results) - 5} more")
        
        # Save to file if requested
        if output:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
            click.echo(f"Results saved to {output}")
            
    except Exception as e:
        click.echo(f"❌ Failed to resample: {e}", err=True)
        raise click.Abort()

@cli.command()
@click.option('--raster-id', help='Show status for specific raster')
def status(raster_id):
    """Show raster processing status."""
    try:
        status_list = raster_manager.get_processing_status(raster_id)
        
        if not status_list:
            click.echo("No raster processing status found.")
            return
        
        for status_info in status_list:
            click.echo(f"\n📊 Raster: {status_info['raster_name']} ({status_info['raster_id']})")
            click.echo(f"   Status: {status_info['processing_status']}")
            click.echo(f"   File size: {status_info.get('file_size_mb', 0):.1f} MB")
            click.echo(f"   Total tiles: {status_info.get('total_tiles', 0)}")
            click.echo(f"   Completed: {status_info.get('completed_tiles', 0)}")
            click.echo(f"   Failed: {status_info.get('failed_tiles', 0)}")
            click.echo(f"   Pending: {status_info.get('pending_tiles', 0)}")
            
            completion = status_info.get('completion_percent', 0)
            if completion is not None:
                click.echo(f"   Progress: {completion:.1f}%")
            
    except Exception as e:
        click.echo(f"❌ Failed to get status: {e}", err=True)
        raise click.Abort()

@cli.command()
@click.option('--raster-id', help='Show cache stats for specific raster')
@click.option('--grid-id', help='Show cache stats for specific grid')
def cache_stats(raster_id, grid_id):
    """Show resampling cache statistics."""
    try:
        stats = raster_manager.get_cache_statistics(raster_id, grid_id)
        
        if not stats:
            click.echo("No cache statistics found.")
            return
        
        click.echo("🗄️  Cache Statistics:")
        click.echo(f"{'Raster':<25} {'Grid':<20} {'Method':<12} {'Cached':<8} {'Avg Conf':<10} {'Avg Access':<12}")
        click.echo("-" * 95)
        
        for stat in stats:
            click.echo(
                f"{stat['raster_name']:<25} "
                f"{stat['grid_name']:<20} "
                f"{stat['method']:<12} "
                f"{stat['cached_cells']:<8} "
                f"{stat.get('avg_confidence', 0):<10.3f} "
                f"{stat.get('avg_access_count', 0):<12.1f}"
            )
            
    except Exception as e:
        click.echo(f"❌ Failed to get cache stats: {e}", err=True)
        raise click.Abort()

@cli.command()
@click.option('--days-old', default=30, type=int, help='Clean entries older than N days')
@click.option('--min-access', default=1, type=int, help='Clean entries with fewer than N accesses')
@click.option('--dry-run', is_flag=True, help='Show what would be cleaned without doing it')
def cache_cleanup(days_old, min_access, dry_run):
    """Clean up old resampling cache entries."""
    try:
        if dry_run:
            click.echo(f"🔍 Dry run: Would clean cache entries older than {days_old} days with < {min_access} accesses")
            # In a real implementation, you'd query the database to show what would be deleted
            click.echo("(Dry run mode - no actual cleanup performed)")
        else:
            click.echo(f"🧹 Cleaning cache entries older than {days_old} days with < {min_access} accesses...")
            deleted_count = raster_manager.cleanup_cache(days_old, min_access)
            click.echo(f"✅ Cleaned up {deleted_count} cache entries")
            
    except Exception as e:
        click.echo(f"❌ Failed to cleanup cache: {e}", err=True)
        raise click.Abort()

@cli.command()
@click.option('--queue-type', 
              type=click.Choice(['raster_tiling', 'resampling', 'validation']),
              help='Show tasks for specific queue type')
def queue_status(queue_type):
    """Show processing queue status."""
    try:
        queue_summary = schema.get_processing_queue_summary()
        
        if not queue_summary:
            click.echo("No processing queue information found.")
            return
        
        # Filter by queue type if specified
        if queue_type:
            queue_summary = [qs for qs in queue_summary if qs['queue_type'] == queue_type]
        
        click.echo("⚡ Processing Queue Status:")
        click.echo(f"{'Queue Type':<20} {'Status':<12} {'Tasks':<8} {'Avg Priority':<12} {'Retried':<8}")
        click.echo("-" * 70)
        
        for qs in queue_summary:
            click.echo(
                f"{qs['queue_type']:<20} "
                f"{qs['status']:<12} "
                f"{qs['task_count']:<8} "
                f"{qs.get('avg_priority', 0):<12.1f} "
                f"{qs.get('retried_tasks', 0):<8}"
            )
            
    except Exception as e:
        click.echo(f"❌ Failed to get queue status: {e}", err=True)
        raise click.Abort()

@cli.command()
@click.argument('queue_type', type=click.Choice(['raster_tiling', 'resampling', 'validation']))
@click.option('--worker-id', default='cli-worker', help='Worker ID for processing')
@click.option('--max-tasks', default=1, type=int, help='Maximum number of tasks to process')
def process_tasks(queue_type, worker_id, max_tasks):
    """Process tasks from the queue."""
    try:
        processed = 0
        failed = 0
        
        click.echo(f"🚀 Processing {queue_type} tasks with worker {worker_id}")
        
        for i in range(max_tasks):
            success = raster_manager.process_queue_task(queue_type, worker_id)
            
            if success:
                processed += 1
                click.echo(f"  ✅ Processed task {i + 1}")
            else:
                failed += 1
                click.echo(f"  ❌ Failed to process task {i + 1} (or no tasks available)")
                break
        
        click.echo(f"\n📈 Summary: {processed} processed, {failed} failed")
        
    except Exception as e:
        click.echo(f"❌ Failed to process tasks: {e}", err=True)
        raise click.Abort()

@cli.command()
def config_info():
    """Show raster processing configuration."""
    try:
        raster_config = config.raster_processing
        
        click.echo("⚙️  Raster Processing Configuration:")
        click.echo(f"  Tile size: {raster_config['tile_size']} pixels")
        click.echo(f"  Cache TTL: {raster_config['cache_ttl_days']} days")
        click.echo(f"  Memory limit: {raster_config['memory_limit_mb']} MB")
        click.echo(f"  Parallel workers: {raster_config['parallel_workers']}")
        click.echo(f"  Chunk size: {raster_config['lazy_loading']['chunk_size_mb']} MB")
        click.echo(f"  Prefetch tiles: {raster_config['lazy_loading']['prefetch_tiles']}")
        click.echo(f"  Default resampling: {raster_config['resampling_methods']['default']}")
        click.echo(f"  Supported formats: {', '.join(raster_config['supported_formats'])}")
        
    except Exception as e:
        click.echo(f"❌ Failed to show config: {e}", err=True)
        raise click.Abort()

if __name__ == '__main__':
    cli()
