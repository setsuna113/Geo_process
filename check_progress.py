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
        
        # First, check what tables exist
        print("\n📋 DATABASE TABLES:")
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name
        """)
        tables = cursor.fetchall()
        table_names = [t[0] for t in tables]
        print(f"  Available tables: {', '.join(table_names)}")
        
        # Check experiments table structure
        if 'experiments' in table_names:
            print("\n📊 EXPERIMENTS:")
            cursor.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'experiments' AND table_schema = 'public'
                ORDER BY ordinal_position
            """)
            exp_columns = [c[0] for c in cursor.fetchall()]
            print(f"  Columns: {', '.join(exp_columns)}")
            
            # Build query based on available columns
            select_cols = ['id', 'name']
            if 'status' in exp_columns:
                select_cols.append('status')
            if 'created_by' in exp_columns:
                select_cols.append('created_by')
            
            cursor.execute(f"SELECT {', '.join(select_cols)} FROM experiments LIMIT 10")
            experiments = cursor.fetchall()
            
            if experiments:
                for exp in experiments:
                    status = exp[2] if len(exp) > 2 else 'unknown'
                    status_icon = "✅" if status == 'completed' else "🔄" if status == 'running' else "❓"
                    print(f"  {status_icon} {exp[1]} ({status})")
            else:
                print("  No experiments found")
        else:
            print("\n📊 EXPERIMENTS: Table not found")
        
        # Check processing jobs
        if 'processing_jobs' in table_names:
            print("\n⚙️ PROCESSING JOBS:")
            cursor.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'processing_jobs' AND table_schema = 'public'
                ORDER BY ordinal_position
            """)
            job_columns = [c[0] for c in cursor.fetchall()]
            print(f"  Columns: {', '.join(job_columns)}")
            
            # Build safe query
            select_cols = []
            if 'job_name' in job_columns:
                select_cols.append('job_name')
            if 'job_type' in job_columns:
                select_cols.append('job_type')
            if 'status' in job_columns:
                select_cols.append('status')
            if 'progress_percent' in job_columns:
                select_cols.append('progress_percent')
            
            if select_cols:
                cursor.execute(f"SELECT {', '.join(select_cols)} FROM processing_jobs LIMIT 20")
                jobs = cursor.fetchall()
                
                if jobs:
                    for job in jobs:
                        job_name = job[0] if len(job) > 0 else 'unknown'
                        job_type = job[1] if len(job) > 1 else 'unknown'
                        status = job[2] if len(job) > 2 else 'unknown'
                        progress = job[3] if len(job) > 3 else 0
                        status_icon = "✅" if status == 'completed' else "🔄" if status == 'running' else "❓"
                        print(f"  {status_icon} {job_name} ({job_type}) - {progress or 0:.1f}%")
                else:
                    print("  No processing jobs found")
            else:
                print("  No recognizable columns in processing_jobs")
        else:
            print("\n⚙️ PROCESSING JOBS: Table not found")
        
        # Check each table safely
        for table_name in ['raster_sources', 'resampling_cache', 'grid_definitions', 'grid_cells', 'features']:
            if table_name in table_names:
                print(f"\n📊 {table_name.upper().replace('_', ' ')}:")
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    print(f"  📈 {count} entries")
                    
                    # Get recent entries if table has data
                    if count > 0:
                        cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                        sample_rows = cursor.fetchall()
                        if sample_rows:
                            print(f"  📋 Sample entries: {len(sample_rows)} rows")
                        
                        # Special handling for some tables
                        if table_name == 'resampling_cache':
                            cursor.execute("SELECT COUNT(DISTINCT source_raster_id) FROM resampling_cache")
                            unique_rasters = cursor.fetchone()[0]
                            print(f"  🗺️ Covers {unique_rasters} unique rasters")
                            
                        elif table_name == 'raster_sources':
                            cursor.execute("SELECT DISTINCT processing_status FROM raster_sources WHERE processing_status IS NOT NULL")
                            statuses = cursor.fetchall()
                            if statuses:
                                status_list = [s[0] for s in statuses]
                                print(f"  🔄 Statuses: {', '.join(status_list)}")
                                
                except Exception as e:
                    print(f"  ❌ Error querying {table_name}: {e}")
            else:
                print(f"\n📊 {table_name.upper().replace('_', ' ')}: Table not found")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure PostgreSQL is running and database exists")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()