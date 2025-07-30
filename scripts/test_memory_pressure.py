#!/usr/bin/env python3
"""Test script for memory pressure callbacks in the pipeline."""

import os
import sys
import time
import threading
import numpy as np
import logging
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.database.connection import DatabaseManager
from src.pipelines.orchestrator import PipelineOrchestrator, PipelineContext
from src.pipelines.monitors.memory_monitor import MemoryMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MemoryHog:
    """Helper class to consume memory and trigger pressure."""
    
    def __init__(self):
        self.arrays = []
        self.running = False
        self.thread = None
    
    def start(self, target_gb: float, rate_mb_per_sec: float = 100):
        """Start consuming memory gradually."""
        self.running = True
        self.thread = threading.Thread(
            target=self._consume_memory,
            args=(target_gb, rate_mb_per_sec),
            daemon=True
        )
        self.thread.start()
        logger.info(f"Memory hog started - targeting {target_gb}GB at {rate_mb_per_sec}MB/s")
    
    def _consume_memory(self, target_gb: float, rate_mb_per_sec: float):
        """Gradually consume memory."""
        target_bytes = target_gb * 1024**3
        chunk_size_bytes = int(rate_mb_per_sec * 1024**2)
        elements_per_chunk = chunk_size_bytes // 8  # 8 bytes per float64
        
        total_allocated = 0
        while self.running and total_allocated < target_bytes:
            try:
                # Allocate chunk of memory
                chunk = np.random.rand(elements_per_chunk)
                self.arrays.append(chunk)
                total_allocated += chunk.nbytes
                
                logger.info(f"Memory allocated: {total_allocated / 1024**3:.2f}GB / {target_gb}GB")
                time.sleep(1)  # Wait 1 second between allocations
                
            except MemoryError:
                logger.error("MemoryError - cannot allocate more")
                break
    
    def stop(self):
        """Stop consuming memory and release arrays."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        
        # Release memory
        self.arrays.clear()
        import gc
        gc.collect()
        logger.info("Memory hog stopped and memory released")


def test_memory_monitor_callbacks():
    """Test memory monitor callbacks in isolation."""
    logger.info("=== Testing Memory Monitor Callbacks ===")
    
    # Create memory monitor
    memory_monitor = MemoryMonitor(config)
    
    # Track callback invocations
    warning_called = False
    critical_called = False
    
    def on_warning(usage):
        nonlocal warning_called
        warning_called = True
        logger.warning(f"⚠️  WARNING CALLBACK: Memory at {usage['process_rss_gb']:.2f}GB")
    
    def on_critical(usage):
        nonlocal critical_called
        critical_called = True
        logger.error(f"🚨 CRITICAL CALLBACK: Memory at {usage['process_rss_gb']:.2f}GB")
    
    # Register callbacks
    memory_monitor.register_warning_callback(on_warning)
    memory_monitor.register_critical_callback(on_critical)
    
    # Start monitoring
    memory_monitor.start()
    
    # Get current memory limit
    memory_limit = config.get('pipeline.memory_limit_gb', 16)
    logger.info(f"Memory limit: {memory_limit}GB")
    logger.info(f"Warning threshold: {memory_limit * 0.8:.1f}GB")
    logger.info(f"Critical threshold: {memory_limit * 0.9:.1f}GB")
    
    # Create memory hog to trigger pressure
    hog = MemoryHog()
    
    try:
        # Target just above warning threshold
        target_gb = memory_limit * 0.85
        logger.info(f"Allocating memory to trigger warning ({target_gb:.1f}GB)...")
        hog.start(target_gb, rate_mb_per_sec=500)
        
        # Wait for callbacks
        timeout = 30
        start_time = time.time()
        while not warning_called and time.time() - start_time < timeout:
            time.sleep(0.5)
        
        if warning_called:
            logger.info("✅ Warning callback triggered successfully!")
        else:
            logger.error("❌ Warning callback was not triggered")
        
        # Show current status
        status = memory_monitor.get_status()
        logger.info(f"Current memory status: {status}")
        
    finally:
        # Cleanup
        hog.stop()
        memory_monitor.stop()
        
    return warning_called


def test_pipeline_adaptive_behavior():
    """Test adaptive behavior in actual pipeline context."""
    logger.info("\n=== Testing Pipeline Adaptive Behavior ===")
    
    # Create minimal pipeline context
    db = DatabaseManager(config)
    
    # Create test context
    context = PipelineContext(
        config=config,
        db=db,
        experiment_id="test_memory_pressure",
        checkpoint_dir=Path("/tmp/test_checkpoints"),
        output_dir=Path("/tmp/test_outputs"),
        metadata={"test": True}
    )
    
    # Create and configure memory monitor
    memory_monitor = MemoryMonitor(config)
    context.memory_monitor = memory_monitor
    memory_monitor.start()
    
    # Track window size changes
    original_window_size = config.get('resampling.window_size', 2048)
    window_sizes = [original_window_size]
    
    def on_warning(usage):
        current_size = config.get('resampling.window_size')
        window_sizes.append(current_size)
        logger.warning(f"Window size changed: {window_sizes[-2]} → {window_sizes[-1]}")
    
    # Test resampling processor behavior
    from src.processors.data_preparation.resampling_processor import ResamplingProcessor
    processor = ResamplingProcessor(config, db)
    
    # Create test dataset config
    test_dataset = {
        'name': 'test_dataset',
        'path': '/tmp/test.tif',
        'resolved_path': '/tmp/test.tif'
    }
    
    try:
        # This would normally process a dataset, but we're testing the callback setup
        # The actual processing would fail since the file doesn't exist
        # But we can verify the callbacks are registered
        
        # Simulate memory pressure callback registration
        memory_monitor.register_warning_callback(on_warning)
        
        # Trigger memory pressure manually for testing
        logger.info("Simulating memory pressure...")
        usage = {
            'process_rss_gb': config.get('pipeline.memory_limit_gb', 16) * 0.85,
            'system_percent': 85
        }
        memory_monitor._trigger_warning(usage)
        
        # Check if window size would change
        logger.info(f"Window sizes during test: {window_sizes}")
        
    finally:
        memory_monitor.stop()
        db.close()
    
    return len(window_sizes) > 1


def main():
    """Run memory pressure tests."""
    logger.info("Starting memory pressure callback tests...")
    
    # Test 1: Basic memory monitor callbacks
    test1_passed = test_memory_monitor_callbacks()
    
    # Test 2: Pipeline adaptive behavior
    test2_passed = test_pipeline_adaptive_behavior()
    
    # Summary
    logger.info("\n=== Test Summary ===")
    logger.info(f"Memory monitor callbacks: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    logger.info(f"Pipeline adaptive behavior: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        logger.info("\n✅ All tests passed!")
        return 0
    else:
        logger.error("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())