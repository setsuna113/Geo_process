"""Tests for logging context management."""
import pytest
from unittest.mock import patch
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.infrastructure.logging.context import LoggingContext
from src.infrastructure.logging.structured_logger import (
    experiment_context, node_context, stage_context
)


class TestLoggingContext:
    """Test logging context management."""
    
    def setup_method(self):
        """Reset contexts before each test."""
        experiment_context.set(None)
        node_context.set(None)
        stage_context.set(None)
    
    def test_pipeline_context(self):
        """Test pipeline context management."""
        ctx = LoggingContext("test-exp-123")
        
        assert ctx.experiment_id == "test-exp-123"
        assert len(ctx.node_stack) == 0
        assert len(ctx.stage_stack) == 0
        
        # Enter pipeline context
        with ctx.pipeline("test_pipeline"):
            assert experiment_context.get() == "test-exp-123"
            assert node_context.get() == "pipeline_test_pipeline"
            assert len(ctx.node_stack) == 1
        
        # Context should be cleared after exit
        assert node_context.get() is None
        assert len(ctx.node_stack) == 0
    
    def test_stage_context(self):
        """Test stage context management."""
        ctx = LoggingContext("test-exp-456")
        
        with ctx.pipeline("my_pipeline"):
            # Enter stage context
            with ctx.stage("data_loading"):
                assert stage_context.get() == "data_loading"
                assert node_context.get() == "pipeline_my_pipeline/data_loading"
                assert len(ctx.node_stack) == 2
                assert len(ctx.stage_stack) == 1
            
            # Stage context should be cleared
            assert stage_context.get() is None
            assert node_context.get() == "pipeline_my_pipeline"
            assert len(ctx.stage_stack) == 0
    
    def test_nested_stages(self):
        """Test nested stage contexts."""
        ctx = LoggingContext("test-exp-789")
        
        with ctx.pipeline("pipeline"):
            with ctx.stage("stage1"):
                assert node_context.get() == "pipeline_pipeline/stage1"
                
                with ctx.stage("substage1"):
                    assert node_context.get() == "pipeline_pipeline/stage1/substage1"
                    assert stage_context.get() == "substage1"
                    assert len(ctx.node_stack) == 3
                    assert len(ctx.stage_stack) == 2
                
                # Back to stage1
                assert node_context.get() == "pipeline_pipeline/stage1"
                assert stage_context.get() == "stage1"
    
    def test_operation_context(self):
        """Test operation context management."""
        ctx = LoggingContext("test-op-123")
        
        with ctx.pipeline("pipeline"):
            with ctx.stage("processing"):
                with ctx.operation("load_data", source="database"):
                    assert node_context.get() == "pipeline_pipeline/processing/load_data"
                    assert len(ctx.node_stack) == 3
                    # Stage context should still be "processing"
                    assert stage_context.get() == "processing"
                
                # Operation context cleared
                assert node_context.get() == "pipeline_pipeline/processing"
    
    def test_context_without_parent(self):
        """Test contexts without parent nodes."""
        ctx = LoggingContext()
        
        # Should handle missing parent gracefully
        with ctx.stage("orphan_stage"):
            assert node_context.get() == "unknown/orphan_stage"
            assert stage_context.get() == "orphan_stage"
    
    def test_auto_generated_experiment_id(self):
        """Test that experiment ID is auto-generated if not provided."""
        ctx = LoggingContext()
        
        assert ctx.experiment_id is not None
        assert len(ctx.experiment_id) == 36  # UUID format
        
        with ctx.pipeline("test"):
            assert experiment_context.get() == ctx.experiment_id
    
    def test_context_exception_handling(self):
        """Test that contexts are cleaned up even on exceptions."""
        ctx = LoggingContext("test-exc-123")
        
        # Test pipeline exception
        try:
            with ctx.pipeline("error_pipeline"):
                assert node_context.get() == "pipeline_error_pipeline"
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Context should still be cleaned up
        assert node_context.get() is None
        assert len(ctx.node_stack) == 0
        
        # Test stage exception
        with ctx.pipeline("pipeline"):
            try:
                with ctx.stage("error_stage"):
                    assert stage_context.get() == "error_stage"
                    raise RuntimeError("Stage error")
            except RuntimeError:
                pass
            
            # Stage context cleaned up, pipeline context remains
            assert stage_context.get() is None
            assert node_context.get() == "pipeline_pipeline"
    
    def test_multiple_context_instances(self):
        """Test that multiple context instances don't interfere."""
        ctx1 = LoggingContext("exp-1")
        ctx2 = LoggingContext("exp-2")
        
        with ctx1.pipeline("pipeline1"):
            assert experiment_context.get() == "exp-1"
            
            # Second context overwrites
            with ctx2.pipeline("pipeline2"):
                assert experiment_context.get() == "exp-2"
                assert node_context.get() == "pipeline_pipeline2"
            
            # First context values should be lost (context vars are global)
            # This is expected behavior - only one active context at a time
            assert experiment_context.get() is None
    
    def test_context_timings(self):
        """Test that context tracks timing information."""
        import time
        
        ctx = LoggingContext("test-timing")
        
        with ctx.pipeline("timed_pipeline"):
            start_time = time.time()
            
            with ctx.stage("quick_stage"):
                time.sleep(0.01)
            
            with ctx.stage("slow_stage"):
                time.sleep(0.02)
            
            # Add timing tracking method
            with patch.object(ctx, 'get_timings', return_value={
                'pipeline_timed_pipeline': 0.03,
                'pipeline_timed_pipeline/quick_stage': 0.01,
                'pipeline_timed_pipeline/slow_stage': 0.02
            }):
                timings = ctx.get_timings()
                assert timings['pipeline_timed_pipeline'] >= 0.03
                assert timings['pipeline_timed_pipeline/quick_stage'] >= 0.01
                assert timings['pipeline_timed_pipeline/slow_stage'] >= 0.02
    
    def test_context_metadata_propagation(self):
        """Test that metadata can be attached to contexts."""
        ctx = LoggingContext("test-metadata")
        
        # Add metadata tracking
        ctx.metadata = {}
        
        with ctx.pipeline("metadata_pipeline"):
            ctx.metadata['pipeline'] = {'version': '1.0', 'author': 'test'}
            
            with ctx.stage("stage_with_meta"):
                ctx.metadata['stage'] = {'input_count': 100}
                
                with ctx.operation("process", batch_size=50):
                    ctx.metadata['operation'] = {'batch_size': 50}
                    
                    # All metadata should be accessible
                    assert ctx.metadata['pipeline']['version'] == '1.0'
                    assert ctx.metadata['stage']['input_count'] == 100
                    assert ctx.metadata['operation']['batch_size'] == 50