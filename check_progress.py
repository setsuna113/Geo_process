#!/usr/bin/env python3
"""Quick script to check analysis progress from database."""

import psycopg2
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path

def load_config():
    """Load configuration from config.yml."""
    config_path = Path(__file__).parent / 'config.yml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def get_db_connection():
    """Get database connection using config.yml settings."""
    config = load_config()
    db_config = config.get('database', {})
    
    return psycopg2.connect(
        host=db_config.get('host', 'localhost'),
        port=db_config.get('port', 5432),
        database=db_config.get('database', 'geo_analysis'),
        user=db_config.get('user', 'postgres'),
        password=db_config.get('password', '')
    )

def main():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        print("=" * 60)
        print("ANALYSIS PROGRESS CHECK")
        print("=" * 60)
        
        # Check experiments
        print("\n📊 EXPERIMENTS:")
        cursor.execute("""
            SELECT id, name, status, created_at, completed_at, 
                   EXTRACT(EPOCH FROM (COALESCE(completed_at, NOW()) - created_at))/3600 as hours_elapsed
            FROM experiments 
            ORDER BY created_at DESC 
            LIMIT 10
        """)
        experiments = cursor.fetchall()
        
        if experiments:
            for exp in experiments:
                status_icon = "✅" if exp[2] == 'completed' else "🔄" if exp[2] == 'running' else "❌"
                print(f"  {status_icon} {exp[1]} ({exp[2]}) - {exp[5]:.1f}h")
        else:
            print("  No experiments found")
        
        # Check processing jobs
        print("\n⚙️ PROCESSING JOBS:")
        cursor.execute("""
            SELECT job_name, job_type, status, progress_percent, 
                   started_at, completed_at,
                   EXTRACT(EPOCH FROM (COALESCE(completed_at, NOW()) - COALESCE(started_at, created_at)))/3600 as hours_elapsed
            FROM processing_jobs 
            ORDER BY created_at DESC 
            LIMIT 20
        """)
        jobs = cursor.fetchall()
        
        if jobs:
            for job in jobs:
                status_icon = "✅" if job[2] == 'completed' else "🔄" if job[2] == 'running' else "❌"
                progress = job[3] or 0
                elapsed = job[6] or 0
                print(f"  {status_icon} {job[0]} ({job[1]}) - {progress:.1f}% - {elapsed:.1f}h")
        else:
            print("  No processing jobs found")
        
        # Check raster sources
        print("\n🗺️ RASTER SOURCES:")
        cursor.execute("""
            SELECT source_name, processing_status, COUNT(*) as count
            FROM raster_sources 
            GROUP BY source_name, processing_status
            ORDER BY source_name, processing_status
        """)
        raster_sources = cursor.fetchall()
        
        if raster_sources:
            for source in raster_sources:
                status = source[1] or 'pending'
                print(f"  📁 {source[0]}: {status} ({source[2]} files)")
        else:
            print("  No raster sources found")
        
        # Check resampling cache (indicates resampling progress)
        print("\n🔄 RESAMPLING CACHE:")
        cursor.execute("""
            SELECT COUNT(*) as cached_tiles, 
                   COUNT(DISTINCT source_raster_id) as unique_rasters,
                   MAX(created_at) as last_cached
            FROM resampling_cache
        """)
        cache_info = cursor.fetchone()
        
        if cache_info and cache_info[0] > 0:
            print(f"  📦 {cache_info[0]} cached tiles from {cache_info[1]} rasters")
            print(f"  🕒 Last cached: {cache_info[2]}")
        else:
            print("  No resampling cache entries")
        
        # Check grid cells (indicates grid processing)
        print("\n🌐 GRID PROCESSING:")
        cursor.execute("""
            SELECT gd.grid_name, gd.resolution_km, COUNT(gc.id) as cell_count
            FROM grid_definitions gd
            LEFT JOIN grid_cells gc ON gd.id = gc.grid_definition_id
            GROUP BY gd.grid_name, gd.resolution_km
            ORDER BY gd.grid_name, gd.resolution_km
        """)
        grids = cursor.fetchall()
        
        if grids:
            for grid in grids:
                print(f"  🌐 {grid[0]} ({grid[1]}km): {grid[2]} cells")
        else:
            print("  No grids found")
        
        # Check features (indicates analysis progress)
        print("\n📈 FEATURES:")
        cursor.execute("""
            SELECT feature_name, COUNT(*) as count, MAX(created_at) as last_created
            FROM features 
            GROUP BY feature_name
            ORDER BY feature_name
        """)
        features = cursor.fetchall()
        
        if features:
            for feature in features:
                print(f"  📊 {feature[0]}: {feature[1]} entries (last: {feature[2]})")
        else:
            print("  No features computed yet")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure PostgreSQL is running and database exists")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()