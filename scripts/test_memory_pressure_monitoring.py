#!/usr/bin/env python3
"""Test memory pressure monitoring and adaptive behavior."""

import sys
import time
import numpy as np
import tempfile
from pathlib import Path
import psutil
import gc

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.database.connection import DatabaseManager
from src.pipelines.monitors.memory_monitor import MemoryMonitor
from src.pipelines.orchestrator import PipelineContext
from src.pipelines.stages.merge_stage import MergeStage
from src.pipelines.stages.export_stage import ExportStage


def allocate_memory(size_mb):
    """Allocate memory to simulate pressure."""
    # Each float64 is 8 bytes
    elements = (size_mb * 1024 * 1024) // 8
    return np.ones(elements, dtype=np.float64)


def test_memory_pressure_callbacks():
    """Test that memory pressure callbacks are triggered."""
    print("\n" + "="*60)
    print("Testing Memory Pressure Callbacks")
    print("="*60)
    
    # Initialize components
    db = DatabaseManager()
    memory_monitor = MemoryMonitor(config, db)
    
    # Track callback invocations
    warning_triggered = False
    critical_triggered = False
    
    def on_warning(usage):
        nonlocal warning_triggered
        warning_triggered = True
        print(f"✓ Warning callback triggered at {usage:.1f}%")
    
    def on_critical(usage):
        nonlocal critical_triggered
        critical_triggered = True
        print(f"✓ Critical callback triggered at {usage:.1f}%")
    
    # Register callbacks
    memory_monitor.register_warning_callback(on_warning)
    memory_monitor.register_critical_callback(on_critical)
    
    # Start monitoring
    memory_monitor.start()
    
    # Get baseline memory
    process = psutil.Process()
    baseline_memory = process.memory_info().rss / (1024 * 1024 * 1024)  # GB
    total_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
    
    print(f"System memory: {total_memory:.1f} GB")
    print(f"Baseline process memory: {baseline_memory:.1f} GB")
    print(f"Memory limit: {memory_monitor.memory_limit_gb} GB")
    
    # Allocate memory to trigger warnings
    allocations = []
    try:
        # Calculate how much to allocate to trigger warning (80%)
        target_warning = memory_monitor.memory_limit_gb * 0.85
        to_allocate = int((target_warning - baseline_memory) * 1024)  # MB
        
        if to_allocate > 0:
            print(f"\nAllocating {to_allocate} MB to trigger warning...")
            allocations.append(allocate_memory(to_allocate))
            time.sleep(2)  # Give monitor time to detect
        
        # Calculate how much more to allocate to trigger critical (90%)
        if warning_triggered and not critical_triggered:
            target_critical = memory_monitor.memory_limit_gb * 0.92
            additional = int((target_critical - target_warning) * 1024)  # MB
            
            if additional > 0:
                print(f"Allocating additional {additional} MB to trigger critical...")
                allocations.append(allocate_memory(additional))
                time.sleep(2)  # Give monitor time to detect
    
    except MemoryError:
        print("Memory allocation failed (expected)")
    
    # Check results
    memory_monitor.stop()
    
    print(f"\nPeak memory usage: {memory_monitor.peak_memory_gb:.2f} GB")
    print(f"Warning triggered: {'✅' if warning_triggered else '❌'}")
    print(f"Critical triggered: {'✅' if critical_triggered else '❌'}")
    
    # Cleanup
    allocations.clear()
    gc.collect()
    
    return warning_triggered


def test_adaptive_merge_behavior():
    """Test adaptive behavior in merge stage under memory pressure."""
    print("\n" + "="*60)
    print("Testing Adaptive Merge Behavior")
    print("="*60)
    
    # Create test context with memory monitor
    db = DatabaseManager()
    memory_monitor = MemoryMonitor(config, db)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        context = PipelineContext(
            config=config,
            db=db,
            experiment_id="test_adaptive",
            checkpoint_dir=Path(tmpdir) / "checkpoints",
            output_dir=Path(tmpdir) / "output",
            memory_monitor=memory_monitor
        )
        
        # Start monitoring
        memory_monitor.start()
        
        # Track configuration changes
        initial_chunk_size = config.get('merge.streaming_chunk_size', 5000)
        initial_streaming = config.get('merge.enable_streaming', False)
        
        print(f"Initial chunk size: {initial_chunk_size}")
        print(f"Initial streaming mode: {initial_streaming}")
        
        # Create merge stage
        merge_stage = MergeStage(config, db)
        
        # Setup happens in execute, but we can test the callback setup
        merge_stage._setup_memory_callbacks(context)
        
        # Simulate memory pressure
        print("\nSimulating memory pressure...")
        allocations = []
        
        try:
            # Trigger warning
            target_warning = memory_monitor.memory_limit_gb * 0.85
            process = psutil.Process()
            current_gb = process.memory_info().rss / (1024 * 1024 * 1024)
            to_allocate = int((target_warning - current_gb) * 1024)
            
            if to_allocate > 0:
                allocations.append(allocate_memory(to_allocate))
                time.sleep(2)
                
                # Check if chunk size reduced
                new_chunk_size = config.get('merge.streaming_chunk_size', 5000)
                print(f"Chunk size after warning: {new_chunk_size}")
                
            # Trigger critical
            target_critical = memory_monitor.memory_limit_gb * 0.92
            additional = int((target_critical - target_warning) * 1024)
            
            if additional > 0:
                allocations.append(allocate_memory(additional))
                time.sleep(2)
                
                # Check if streaming enabled
                streaming_enabled = config.get('merge.enable_streaming', False)
                print(f"Streaming mode after critical: {streaming_enabled}")
                
        except MemoryError:
            print("Memory allocation failed (expected)")
        
        # Stop monitoring
        memory_monitor.stop()
        
        # Cleanup
        allocations.clear()
        gc.collect()
        
        # Reset config
        config.set('merge.streaming_chunk_size', initial_chunk_size)
        config.set('merge.enable_streaming', initial_streaming)


def main():
    """Run memory pressure tests."""
    print("Starting Memory Pressure Tests")
    print("="*60)
    
    # Test 1: Memory pressure callbacks
    callbacks_work = test_memory_pressure_callbacks()
    
    # Test 2: Adaptive behavior
    test_adaptive_merge_behavior()
    
    print("\n" + "="*60)
    print("Test Summary:")
    print(f"Memory Pressure Callbacks: {'✅ PASSED' if callbacks_work else '❌ FAILED'}")
    print("="*60)
    
    return 0 if callbacks_work else 1


if __name__ == "__main__":
    sys.exit(main())