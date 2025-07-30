#!/usr/bin/env python3
"""Test script to verify signal handler pause/resume functionality."""

import sys
import os
import time
import signal
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.config import Config
from src.database.connection import DatabaseManager
from src.core.signal_handler_enhanced import EnhancedSignalHandler
from src.pipelines.orchestrator import PipelineOrchestrator

def test_signal_handler():
    """Test signal handler pause/resume functionality."""
    print("🧪 Testing Signal Handler Integration")
    print("=" * 50)
    
    # Initialize components
    config = Config()
    db = DatabaseManager()
    
    # Create signal handler
    signal_handler = EnhancedSignalHandler()
    
    # Create orchestrator with signal handler
    orchestrator = PipelineOrchestrator(config, db, signal_handler)
    
    # Register a simple test
    pause_called = False
    resume_called = False
    
    def test_pause():
        nonlocal pause_called
        pause_called = True
        print("✅ Pause callback triggered!")
    
    def test_resume():
        nonlocal resume_called
        resume_called = True
        print("✅ Resume callback triggered!")
    
    # Register callbacks
    signal_handler.register_pause_callback(test_pause)
    signal_handler.register_resume_callback(test_resume)
    
    print("\n📋 Test Instructions:")
    print(f"1. Process PID: {os.getpid()}")
    print("2. In another terminal, run:")
    print(f"   kill -USR1 {os.getpid()}  # To pause")
    print(f"   kill -USR2 {os.getpid()}  # To resume")
    print("3. Press Ctrl+C to exit\n")
    
    print("⏳ Waiting for signals...")
    
    try:
        while True:
            time.sleep(1)
            
            if pause_called:
                print("  Pipeline would be paused now")
                pause_called = False
                
            if resume_called:
                print("  Pipeline would be resumed now")
                resume_called = False
                
    except KeyboardInterrupt:
        print("\n\n🏁 Test completed!")
        print(f"Pause was called: {pause_called or 'Yes'}")
        print(f"Resume was called: {resume_called or 'Yes'}")

if __name__ == "__main__":
    test_signal_handler()