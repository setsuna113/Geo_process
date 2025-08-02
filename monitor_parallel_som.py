#!/usr/bin/env python3
"""Monitor parallel SOM training with CPU utilization tracking."""

import time
import psutil
import os
import sys
from datetime import datetime
from pathlib import Path

# Configuration
LOG_FILE = "/home/yl998/dev/geo/logs/som_parallel_production.log"
UPDATE_INTERVAL = 5  # seconds

def get_cpu_usage():
    """Get CPU usage percentage."""
    return psutil.cpu_percent(interval=1, percpu=True)

def get_process_info(pid):
    """Get process information."""
    try:
        proc = psutil.Process(pid)
        return {
            'cpu_percent': proc.cpu_percent(interval=0.1),
            'memory_percent': proc.memory_percent(),
            'num_threads': proc.num_threads(),
            'status': proc.status()
        }
    except:
        return None

def parse_log_for_progress(log_path):
    """Parse log file for training progress."""
    if not os.path.exists(log_path):
        return None
    
    info = {
        'epoch': None,
        'qe': None,
        'learning_rate': None,
        'phase': 'Starting',
        'chunks_processed': 0,
        'total_chunks': 0,
        'workers': None
    }
    
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
            
        for line in reversed(lines):
            if 'Epoch' in line and 'QE=' in line:
                # Extract epoch and QE
                if 'Epoch' in line:
                    epoch_part = line.split('Epoch')[1].split()[0]
                    info['epoch'] = int(epoch_part.strip(':'))
                if 'QE=' in line:
                    qe_part = line.split('QE=')[1].split()[0]
                    info['qe'] = float(qe_part)
                if 'LR=' in line:
                    lr_part = line.split('LR=')[1].split()[0]
                    info['learning_rate'] = float(lr_part)
                break
            elif 'Processing chunk' in line:
                # Extract chunk progress
                chunk_info = line.split('chunk')[1].split(':')[0].strip()
                current, total = chunk_info.split('/')
                info['chunks_processed'] = int(current)
                info['total_chunks'] = int(total)
            elif 'parallel with' in line and 'workers' in line:
                # Extract worker count
                workers = line.split('with')[1].split('workers')[0].strip()
                info['workers'] = int(workers)
            elif 'PARALLEL batch update' in line:
                info['phase'] = 'Parallel Processing'
            elif 'Training completed' in line:
                info['phase'] = 'Completed'
            elif 'Converged' in line:
                info['phase'] = 'Converged'
                
    except Exception as e:
        pass
    
    return info

def monitor():
    """Main monitoring loop."""
    print("🔍 SOM Parallel Training Monitor")
    print("=" * 80)
    print(f"Log file: {LOG_FILE}")
    print(f"Update interval: {UPDATE_INTERVAL}s")
    print("=" * 80)
    
    # Find python processes running SOM
    som_pids = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python' or proc.info['name'] == 'python3':
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'som' in cmdline.lower() or 'analysis' in cmdline.lower():
                    som_pids.append(proc.info['pid'])
        except:
            pass
    
    print(f"Monitoring PIDs: {som_pids}")
    print("")
    
    while True:
        os.system('clear')
        print(f"🔍 SOM Parallel Training Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Parse log for progress
        progress = parse_log_for_progress(LOG_FILE)
        
        if progress:
            print(f"📊 Training Progress:")
            print(f"   Phase: {progress['phase']}")
            if progress['epoch']:
                print(f"   Epoch: {progress['epoch']}")
            if progress['qe']:
                print(f"   QE: {progress['qe']:.6f}")
            if progress['learning_rate']:
                print(f"   Learning Rate: {progress['learning_rate']:.6f}")
            if progress['workers']:
                print(f"   Parallel Workers: {progress['workers']}")
            if progress['total_chunks'] > 0:
                chunk_percent = (progress['chunks_processed'] / progress['total_chunks']) * 100
                print(f"   Chunk Progress: {progress['chunks_processed']}/{progress['total_chunks']} ({chunk_percent:.1f}%)")
            print("")
        
        # CPU usage
        cpu_usage = get_cpu_usage()
        avg_cpu = sum(cpu_usage) / len(cpu_usage)
        active_cores = sum(1 for usage in cpu_usage if usage > 5.0)
        
        print(f"💻 System Resources:")
        print(f"   Average CPU: {avg_cpu:.1f}%")
        print(f"   Active Cores: {active_cores}/{len(cpu_usage)}")
        print(f"   Memory: {psutil.virtual_memory().percent:.1f}%")
        print("")
        
        # Show top CPU cores
        print(f"🔥 Top 10 CPU Cores:")
        sorted_cores = sorted(enumerate(cpu_usage), key=lambda x: x[1], reverse=True)[:10]
        for core_id, usage in sorted_cores:
            bar = '█' * int(usage / 5)
            print(f"   Core {core_id:3d}: {usage:5.1f}% {bar}")
        print("")
        
        # Process info
        if som_pids:
            print(f"📈 Process Information:")
            for pid in som_pids:
                info = get_process_info(pid)
                if info:
                    print(f"   PID {pid}: CPU={info['cpu_percent']:.1f}%, "
                          f"Mem={info['memory_percent']:.1f}%, "
                          f"Threads={info['num_threads']}, "
                          f"Status={info['status']}")
        
        print("\nPress Ctrl+C to exit")
        
        time.sleep(UPDATE_INTERVAL)

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        sys.exit(0)