#!/usr/bin/env python3
"""Monitor running SOM analysis with detailed progress tracking."""

import os
import time
import psutil
import subprocess
from datetime import datetime

def get_process_info(pid):
    """Get detailed process information."""
    try:
        p = psutil.Process(pid)
        return {
            'cpu_percent': p.cpu_percent(interval=1),
            'memory_gb': p.memory_info().rss / 1024**3,
            'status': p.status(),
            'runtime': time.time() - p.create_time(),
            'num_threads': p.num_threads()
        }
    except:
        return None

def find_som_process():
    """Find running SOM process."""
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if 'run_analysis.py' in cmdline and 'som' in cmdline:
                return proc.info['pid']
        except:
            continue
    return None

def check_latest_log():
    """Get latest progress from log files."""
    try:
        # Check for latest log file
        result = subprocess.run(
            "find logs -name '*.log' -mmin -5 | xargs ls -t | head -1",
            shell=True, capture_output=True, text=True
        )
        if result.stdout.strip():
            log_file = result.stdout.strip()
            # Get last few meaningful lines
            result = subprocess.run(
                f"tail -20 {log_file} | grep -E 'fold|epoch|Converged|QE|Processing|ERROR'",
                shell=True, capture_output=True, text=True
            )
            return result.stdout.strip()
    except:
        pass
    return ""

def monitor_loop():
    """Main monitoring loop."""
    print("🔍 SOM Analysis Monitor")
    print("=" * 60)
    
    pid = find_som_process()
    if not pid:
        print("❌ No SOM process found running")
        return
    
    print(f"✅ Found SOM process: PID {pid}")
    print("=" * 60)
    
    last_log_lines = set()
    stuck_counter = 0
    last_cpu_check = 0
    
    while True:
        try:
            # Get process info
            info = get_process_info(pid)
            if not info:
                print("\n❌ Process ended")
                break
            
            # Clear screen for clean display
            os.system('clear')
            
            # Header
            print(f"🔍 SOM Analysis Monitor - {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 60)
            
            # Process stats
            runtime_min = info['runtime'] / 60
            print(f"📊 Process Stats:")
            print(f"   Runtime: {runtime_min:.1f} minutes")
            print(f"   CPU: {info['cpu_percent']:.1f}%")
            print(f"   Memory: {info['memory_gb']:.2f} GB")
            print(f"   Threads: {info['num_threads']}")
            print(f"   Status: {info['status']}")
            
            # Check if stuck
            if info['cpu_percent'] < 10:
                stuck_counter += 1
                if stuck_counter > 5:
                    print(f"\n⚠️  WARNING: Low CPU usage for {stuck_counter} checks - possibly stuck!")
            else:
                stuck_counter = 0
            
            # Get latest log progress
            print(f"\n📝 Latest Progress:")
            print("-" * 60)
            log_lines = check_latest_log()
            if log_lines:
                # Show new lines
                new_lines = []
                for line in log_lines.split('\n'):
                    if line and line not in last_log_lines:
                        new_lines.append(line)
                        last_log_lines.add(line)
                
                # Keep only recent lines in memory
                if len(last_log_lines) > 100:
                    last_log_lines = set(list(last_log_lines)[-50:])
                
                if new_lines:
                    for line in new_lines[-10:]:  # Show last 10 new lines
                        # Highlight important lines
                        if 'ERROR' in line:
                            print(f"❌ {line}")
                        elif 'fold' in line or 'CV' in line:
                            print(f"🔄 {line}")
                        elif 'Converged' in line:
                            print(f"✅ {line}")
                        elif 'epoch' in line:
                            print(f"📈 {line}")
                        else:
                            print(f"   {line}")
                else:
                    print("   (no new progress)")
            
            # File activity check
            print(f"\n📁 Recent Output Files:")
            result = subprocess.run(
                "find outputs -name '*som*' -mmin -2 -type f | tail -5",
                shell=True, capture_output=True, text=True
            )
            if result.stdout.strip():
                for f in result.stdout.strip().split('\n'):
                    print(f"   {f}")
            else:
                print("   (no recent file activity)")
            
            time.sleep(5)  # Update every 5 seconds
            
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_loop()