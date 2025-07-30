#!/usr/bin/env python3
"""Simple test for memory pressure callbacks without actual memory allocation."""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.pipelines.monitors.memory_monitor import MemoryMonitor

def test_callbacks():
    """Test that callbacks are properly registered and can be triggered."""
    print("Testing memory pressure callbacks...")
    
    # Create memory monitor
    monitor = MemoryMonitor(config)
    
    # Track calls
    callbacks_triggered = {'warning': False, 'critical': False}
    
    def on_warning(usage):
        callbacks_triggered['warning'] = True
        print(f"✅ Warning callback triggered! Memory: {usage['process_rss_gb']:.2f}GB")
    
    def on_critical(usage):
        callbacks_triggered['critical'] = True
        print(f"✅ Critical callback triggered! Memory: {usage['process_rss_gb']:.2f}GB")
    
    # Register callbacks
    monitor.register_warning_callback(on_warning)
    monitor.register_critical_callback(on_critical)
    
    # Manually trigger callbacks with fake usage data
    fake_usage = {
        'process_rss_gb': config.get('pipeline.memory_limit_gb', 16) * 0.85,
        'system_percent': 85,
        'system_available_gb': 2.0
    }
    
    print(f"Triggering warning with fake usage: {fake_usage['process_rss_gb']:.2f}GB")
    monitor._trigger_warning(fake_usage)
    
    fake_usage['process_rss_gb'] = config.get('pipeline.memory_limit_gb', 16) * 0.95
    print(f"Triggering critical with fake usage: {fake_usage['process_rss_gb']:.2f}GB")
    monitor._trigger_critical(fake_usage)
    
    # Check results
    print("\nResults:")
    print(f"Warning callback triggered: {callbacks_triggered['warning']}")
    print(f"Critical callback triggered: {callbacks_triggered['critical']}")
    
    success = callbacks_triggered['warning'] and callbacks_triggered['critical']
    print(f"\n{'✅ Test PASSED' if success else '❌ Test FAILED'}")
    
    return success

if __name__ == "__main__":
    success = test_callbacks()
    sys.exit(0 if success else 1)