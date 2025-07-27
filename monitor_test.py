#!/usr/bin/env python3
"""Monitor test pipeline execution and identify problems."""
import time
import subprocess
import psycopg2
from pathlib import Path

def run_cmd(cmd):
    """Run command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

def check_database():
    """Check database state."""
    conn = psycopg2.connect(
        host="localhost", port=51051, database="geo_cluster_db", 
        user="jason", password="123456"
    )
    cur = conn.cursor()
    
    # Check experiment status
    cur.execute("SELECT name, status FROM experiments ORDER BY started_at DESC LIMIT 1")
    exp = cur.fetchone()
    print(f"  Experiment: {exp[0] if exp else 'None'}")
    print(f"  Status: {exp[1] if exp else 'None'}")
    
    # Check table counts
    tables = ['raster_sources', 'processing_jobs', 'grid_cells', 'features']
    for table in tables:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        if count > 0:
            print(f"  {table}: {count} rows")
    
    conn.close()

def monitor_pipeline(process_name, duration=300):
    """Monitor pipeline for specified duration."""
    print(f"Monitoring {process_name} for {duration} seconds...")
    
    start_time = time.time()
    check_interval = 30
    
    while time.time() - start_time < duration:
        elapsed = int(time.time() - start_time)
        print(f"\n[{elapsed}s] Status Check:")
        
        # Process status
        status = run_cmd(f"python scripts/process_manager.py status | grep {process_name}")
        if status:
            print(f"  Process: {status}")
        else:
            print("  Process: NOT RUNNING")
            break
        
        # Database state
        check_database()
        
        # Output files
        output_dir = Path("/home/yl998/dev/geo/outputs")
        nc_files = list(output_dir.glob("*/merged_dataset.nc"))
        csv_files = list(output_dir.glob("*/*.csv"))
        if nc_files or csv_files:
            print(f"  Output files: {len(nc_files)} NC, {len(csv_files)} CSV")
        
        # Logs (last error)
        log_file = Path(f"/home/yl998/.biodiversity/logs/{process_name}.log")
        if log_file.exists():
            errors = run_cmd(f"grep -i 'error\\|failed' {log_file} | tail -3")
            if errors:
                print(f"  Recent errors:\n    {errors[:200]}")
        
        time.sleep(check_interval)
    
    print("\nFinal status:")
    run_cmd(f"python scripts/process_manager.py status")

if __name__ == "__main__":
    import sys
    process_name = sys.argv[1] if len(sys.argv) > 1 else "subset_test"
    monitor_pipeline(process_name)