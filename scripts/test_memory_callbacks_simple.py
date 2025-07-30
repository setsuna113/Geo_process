#\!/usr/bin/env python3
"""Simple test for memory pressure callbacks."""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.database.connection import DatabaseManager
from src.pipelines.monitors.memory_monitor import MemoryMonitor


def main():
    """Test memory monitor callbacks."""
    print("Testing Memory Monitor Callbacks")
    print("="*60)
    
    # Initialize
    db = DatabaseManager()
    monitor = MemoryMonitor(config, db)
    
    # Track callbacks
    callbacks_triggered = []
    
    def on_warning(usage):
        callbacks_triggered.append(f"Warning at {usage:.1f}%")
        print(f"✓ Warning callback: {usage:.1f}%")
    
    def on_critical(usage):
        callbacks_triggered.append(f"Critical at {usage:.1f}%")
        print(f"✓ Critical callback: {usage:.1f}%")
    
    # Register callbacks
    monitor.register_warning_callback(on_warning)
    monitor.register_critical_callback(on_critical)
    
    # Start monitoring
    print("Starting monitor...")
    monitor.start()
    
    # Wait a bit
    time.sleep(3)
    
    # Stop monitoring
    monitor.stop()
    
    print(f"\nCallbacks triggered: {len(callbacks_triggered)}")
    for cb in callbacks_triggered:
        print(f"  - {cb}")
    
    print(f"Peak memory: {monitor.peak_memory_gb:.2f} GB")
    print(f"Memory limit: {monitor.memory_limit_gb} GB")
    
    # Check current memory usage
    import psutil
    process = psutil.Process()
    current_gb = process.memory_info().rss / (1024**3)
    usage_pct = (current_gb / monitor.memory_limit_gb) * 100
    
    print(f"Current memory: {current_gb:.2f} GB ({usage_pct:.1f}%)")
    print(f"Warning threshold: {monitor.warning_threshold * 100:.0f}%")
    print(f"Critical threshold: {monitor.critical_threshold * 100:.0f}%")
    
    print("\n✅ Test completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
