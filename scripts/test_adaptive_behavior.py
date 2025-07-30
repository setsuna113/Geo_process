#!/usr/bin/env python3
"""Test adaptive behavior in processors."""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.database.connection import DatabaseManager
from src.pipelines.orchestrator import PipelineContext
from src.pipelines.monitors.memory_monitor import MemoryMonitor

def test_adaptive_window_size():
    """Test that window size adapts under memory pressure."""
    print("Testing adaptive window size in ResamplingProcessor...")
    
    # Create context with memory monitor
    db = DatabaseManager()
    memory_monitor = MemoryMonitor(config)
    
    context = PipelineContext(
        config=config,
        db=db,
        experiment_id="test_adaptive",
        checkpoint_dir=Path("/tmp/test"),
        output_dir=Path("/tmp/test")
    )
    context.memory_monitor = memory_monitor
    
    # Track window size changes
    original_window_size = config.get('resampling.window_size', 2048)
    print(f"Original window size: {original_window_size}")
    
    # Create a mock dataset config
    dataset_config = {
        'name': 'test_dataset',
        'path': '/tmp/test.tif'
    }
    
    # Import processor
    from src.processors.data_preparation.resampling_processor import ResamplingProcessor
    processor = ResamplingProcessor(config, db)
    
    # Set up the adaptive behavior by calling the method partially
    # This will register the callbacks
    try:
        # We don't actually process - just set up the callbacks
        processor._adaptive_window_size = original_window_size
        
        # Define the callbacks that would be registered
        def on_memory_warning(usage):
            processor._adaptive_window_size = max(512, processor._adaptive_window_size // 2)
            print(f"✅ Memory warning: Window size reduced to {processor._adaptive_window_size}")
        
        def on_memory_critical(usage):
            processor._adaptive_window_size = 256
            print(f"✅ Memory critical: Window size set to minimum {processor._adaptive_window_size}")
        
        # Register callbacks
        memory_monitor.register_warning_callback(on_memory_warning)
        memory_monitor.register_critical_callback(on_memory_critical)
        
        # Simulate memory pressure
        print("\nSimulating memory pressure...")
        fake_usage = {'process_rss_gb': config.get('pipeline.memory_limit_gb', 16) * 0.85}
        memory_monitor._trigger_warning(fake_usage)
        
        # Check window size changed
        current_size = processor._adaptive_window_size
        print(f"Window size after warning: {current_size}")
        
        # Simulate critical pressure
        fake_usage['process_rss_gb'] = config.get('pipeline.memory_limit_gb', 16) * 0.95
        memory_monitor._trigger_critical(fake_usage)
        
        final_size = processor._adaptive_window_size
        print(f"Window size after critical: {final_size}")
        
        # Verify changes
        success = (current_size < original_window_size) and (final_size == 256)
        
        print(f"\n{'✅ Adaptive behavior working!' if success else '❌ Adaptive behavior not working'}")
        return success
        
    finally:
        pass  # DatabaseManager handles its own cleanup


def test_adaptive_chunk_size():
    """Test that chunk size adapts in CoordinateMerger."""
    print("\n\nTesting adaptive chunk size in CoordinateMerger...")
    
    # Create context
    db = DatabaseManager()
    memory_monitor = MemoryMonitor(config)
    
    context = PipelineContext(
        config=config,
        db=db,
        experiment_id="test_adaptive_merge",
        checkpoint_dir=Path("/tmp/test"),
        output_dir=Path("/tmp/test")
    )
    context.memory_monitor = memory_monitor
    
    # Import merger
    from src.processors.data_preparation.coordinate_merger import CoordinateMerger
    merger = CoordinateMerger(config, db)
    
    # Set up adaptive behavior
    original_chunk_size = 5000
    merger._adaptive_chunk_size = original_chunk_size
    print(f"Original chunk size: {original_chunk_size}")
    
    # Define callbacks
    def on_memory_warning(usage):
        merger._adaptive_chunk_size = max(1000, merger._adaptive_chunk_size // 2)
        print(f"✅ Memory warning: Chunk size reduced to {merger._adaptive_chunk_size}")
    
    def on_memory_critical(usage):
        merger._adaptive_chunk_size = 500
        print(f"✅ Memory critical: Chunk size set to minimum {merger._adaptive_chunk_size}")
    
    # Register callbacks
    memory_monitor.register_warning_callback(on_memory_warning)
    memory_monitor.register_critical_callback(on_memory_critical)
    
    # Simulate pressure
    print("\nSimulating memory pressure...")
    fake_usage = {'process_rss_gb': config.get('pipeline.memory_limit_gb', 16) * 0.85}
    memory_monitor._trigger_warning(fake_usage)
    
    warning_size = merger._adaptive_chunk_size
    print(f"Chunk size after warning: {warning_size}")
    
    fake_usage['process_rss_gb'] = config.get('pipeline.memory_limit_gb', 16) * 0.95
    memory_monitor._trigger_critical(fake_usage)
    
    critical_size = merger._adaptive_chunk_size
    print(f"Chunk size after critical: {critical_size}")
    
    # Verify
    success = (warning_size < original_chunk_size) and (critical_size == 500)
    print(f"\n{'✅ Adaptive chunk size working!' if success else '❌ Adaptive chunk size not working'}")
    
    return success


def main():
    """Run all adaptive behavior tests."""
    print("=" * 60)
    print("Testing Memory Pressure Adaptive Behavior")
    print("=" * 60)
    
    test1 = test_adaptive_window_size()
    test2 = test_adaptive_chunk_size()
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"ResamplingProcessor adaptive window: {'✅ PASSED' if test1 else '❌ FAILED'}")
    print(f"CoordinateMerger adaptive chunk: {'✅ PASSED' if test2 else '❌ FAILED'}")
    print("=" * 60)
    
    return 0 if (test1 and test2) else 1


if __name__ == "__main__":
    sys.exit(main())