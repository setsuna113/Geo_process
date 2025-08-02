#!/usr/bin/env python3
"""Monitor the fixed SOM production run with all improvements."""

import time
import subprocess
import os
from datetime import datetime

LOG_FILE = "logs/som_parallel_production.log"
PID = 37785  # The running process

def get_process_stats():
    """Get CPU and memory stats for the process."""
    try:
        result = subprocess.run(
            ['ps', '-p', str(PID), '-o', 'pid,comm,%cpu,%mem,etime,stat'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                return lines[1].strip()
    except:
        pass
    return None

def get_latest_progress():
    """Extract latest progress from log file."""
    if not os.path.exists(LOG_FILE):
        return "Log file not found"
    
    try:
        # Get last 100 lines and search for key information
        result = subprocess.run(
            ['tail', '-100', LOG_FILE],
            capture_output=True, text=True
        )
        
        lines = result.stdout.strip().split('\n')
        
        # Find latest epoch info
        epoch_info = None
        cv_fold = None
        qe_info = None
        chunk_progress = None
        
        for line in reversed(lines):
            if 'Starting epoch' in line and not epoch_info:
                epoch_info = line.split(' - ')[-1]
            elif 'CV fold' in line and 'Processing' in line and not cv_fold:
                cv_fold = line.split(' - ')[-1]
            elif 'QE:' in line and not qe_info:
                qe_info = line.split(' - ')[-1]
            elif 'out of' in line and 'elapsed' in line and not chunk_progress:
                # Extract chunk progress from joblib output
                chunk_progress = line.strip()
        
        return {
            'cv_fold': cv_fold,
            'epoch': epoch_info,
            'qe': qe_info,
            'chunks': chunk_progress
        }
    except:
        return {}

def monitor():
    """Main monitoring loop."""
    print("🔍 Monitoring Fixed SOM Production Run")
    print("=" * 60)
    print(f"Process PID: {PID}")
    print(f"Log file: {LOG_FILE}")
    print("=" * 60)
    
    while True:
        # Get process stats
        stats = get_process_stats()
        if not stats:
            print("\n❌ Process no longer running!")
            break
        
        # Get progress
        progress = get_latest_progress()
        
        # Clear screen and display
        os.system('clear')
        print(f"📊 SOM Production Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        print(f"Process: {stats}")
        print("-" * 60)
        
        if isinstance(progress, dict):
            if progress.get('cv_fold'):
                print(f"CV Fold: {progress['cv_fold']}")
            if progress.get('epoch'):
                print(f"Epoch: {progress['epoch']}")
            if progress.get('qe'):
                print(f"QE: {progress['qe']}")
            if progress.get('chunks'):
                print(f"Chunks: {progress['chunks']}")
        else:
            print(f"Progress: {progress}")
        
        print("\n💡 All fixes applied:")
        print("  ✅ Sampling for geographic coherence (20k samples)")
        print("  ✅ Optimized predict method (vectorized)")
        print("  ✅ Proper resource cleanup between CV folds")
        print("  ✅ Parallel processing with 32 workers")
        
        print("\nPress Ctrl+C to stop monitoring...")
        
        time.sleep(5)

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")