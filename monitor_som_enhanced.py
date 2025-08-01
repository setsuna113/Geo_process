#!/usr/bin/env python3
"""Enhanced SOM monitor that reads progress files for detailed tracking."""

import os
import json
import time
import psutil
import glob
from datetime import datetime
from pathlib import Path


def find_latest_progress_file():
    """Find the most recent SOM progress file."""
    progress_files = glob.glob("outputs/analysis_results/som/**/som_progress_*.json", recursive=True)
    if not progress_files:
        return None
    return max(progress_files, key=os.path.getmtime)


def read_progress_file(filepath):
    """Read progress data from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds is None:
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_process_stats():
    """Get stats for any running SOM process."""
    for proc in psutil.process_iter(['pid', 'cmdline', 'cpu_percent', 'memory_info']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'run_analysis.py' in cmdline and 'som' in cmdline:
                return {
                    'pid': proc.info['pid'],
                    'cpu': proc.cpu_percent(interval=0.5),
                    'memory_gb': proc.memory_info().rss / 1024**3
                }
        except:
            continue
    return None


def monitor_loop():
    """Main monitoring loop."""
    print("🔍 Enhanced SOM Monitor")
    print("=" * 70)
    
    last_update = None
    
    while True:
        try:
            # Clear screen
            os.system('clear')
            
            # Header
            print(f"🔍 Enhanced SOM Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)
            
            # Find progress file
            progress_file = find_latest_progress_file()
            if not progress_file:
                print("❌ No SOM progress file found")
                print("\nWaiting for SOM analysis to start...")
                time.sleep(5)
                continue
            
            # Read progress
            progress = read_progress_file(progress_file)
            if not progress:
                print("❌ Could not read progress file")
                time.sleep(5)
                continue
            
            # Process stats
            proc_stats = get_process_stats()
            
            print(f"\n📊 Experiment: {progress.get('experiment', 'Unknown')}")
            print(f"📁 Progress file: {os.path.basename(progress_file)}")
            
            # Status
            status = progress.get('status', 'unknown')
            if status == 'running':
                print(f"\n✅ Status: Running")
            elif status == 'completed':
                print(f"\n✅ Status: Completed")
            elif status == 'failed':
                print(f"\n❌ Status: Failed")
            else:
                print(f"\n❓ Status: {status}")
            
            # Phase info
            phase = progress.get('current_phase', 'unknown')
            cv_fold = progress.get('cv_fold')
            if cv_fold:
                print(f"📍 Phase: {phase} (Fold {cv_fold}/5)")
            else:
                print(f"📍 Phase: {phase}")
            
            # Progress bar
            prog_pct = progress.get('progress_percent', 0)
            epoch = progress.get('epoch', 0)
            max_epochs = progress.get('max_epochs', 0)
            
            if max_epochs > 0:
                bar_width = 40
                filled = int(bar_width * prog_pct / 100)
                bar = '█' * filled + '░' * (bar_width - filled)
                print(f"\n📈 Epoch Progress: [{bar}] {prog_pct:.1f}%")
                print(f"   Epoch: {epoch}/{max_epochs}")
            
            # Metrics
            qe = progress.get('quantization_error')
            lr = progress.get('learning_rate')
            radius = progress.get('radius')
            
            if qe is not None:
                print(f"\n📊 Training Metrics:")
                print(f"   Quantization Error: {qe:.6f}")
                print(f"   Learning Rate: {lr:.4f}")
                print(f"   Radius: {radius:.2f}")
            
            # Timing
            elapsed = progress.get('elapsed_seconds', 0)
            remaining = progress.get('estimated_remaining')
            
            print(f"\n⏱️  Timing:")
            print(f"   Elapsed: {format_time(elapsed)}")
            if remaining:
                print(f"   Estimated remaining: {format_time(remaining)}")
                total = elapsed + remaining
                print(f"   Estimated total: {format_time(total)}")
            
            # System resources
            if proc_stats:
                print(f"\n💻 System Resources:")
                print(f"   PID: {proc_stats['pid']}")
                print(f"   CPU: {proc_stats['cpu']:.1f}%")
                print(f"   Memory: {proc_stats['memory_gb']:.2f} GB")
            
            # Last update
            last_update_str = progress.get('last_update', '')
            if last_update_str:
                try:
                    last_dt = datetime.fromisoformat(last_update_str)
                    ago = (datetime.now() - last_dt).total_seconds()
                    print(f"\n🕐 Last update: {int(ago)}s ago")
                    
                    if ago > 300:  # 5 minutes
                        print("   ⚠️  WARNING: No update for >5 minutes")
                except:
                    pass
            
            # Check if new update
            if last_update != progress.get('last_update'):
                last_update = progress.get('last_update')
                print("\n🔄 New update detected!")
            
            print("\n" + "-" * 70)
            print("Press Ctrl+C to stop monitoring")
            
            time.sleep(2)  # Update every 2 seconds
            
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    monitor_loop()