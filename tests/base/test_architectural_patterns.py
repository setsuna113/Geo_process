"""Architectural regression tests for new patterns and improvements."""

import pytest
import threading
import time
import signal
import queue
from unittest.mock import Mock, patch, MagicMock

from src.base.processor import (
    BaseProcessor, ProcessorConfig, ProcessorDependencies, 
    ProcessorBuilder, ProcessingResult
)
from src.core.signal_handler import SignalHandler, SignalEvent
from src.database.schema import DatabaseSchema


class MockProcessor(BaseProcessor):
    """Mock processor for testing."""
    
    def process_single(self, item):
        """Mock process single item."""
        return f"processed_{item}"
    
    def validate_input(self, item):
        """Mock validate input."""
        return True, None


class TestProcessorConfig:
    """Test ProcessorConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ProcessorConfig()
        assert config.batch_size == 1000
        assert config.enable_progress is True
        assert config.enable_checkpoints is True
        assert config.supports_chunking is True
    
    def test_from_config_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            'processors': {
                'batch_size': 2000,
                'memory_limit_mb': 4096
            },
            'processors.TestProcessor': {
                'batch_size': 3000
            }
        }
        
        config = ProcessorConfig.from_config(config_dict, "TestProcessor")
        assert config.batch_size == 3000  # Processor-specific override
        assert config.memory_limit_mb == 4096  # Global setting
    
    def test_config_inheritance(self):
        """Test that processor-specific config overrides global config."""
        config_dict = {
            'processors': {
                'enable_progress': False,
                'batch_size': 1000
            },
            'processors.SpecialProcessor': {
                'enable_progress': True
            }
        }
        
        config = ProcessorConfig.from_config(config_dict, "SpecialProcessor")
        assert config.enable_progress is True  # Overridden
        assert config.batch_size == 1000  # Inherited


class TestProcessorDependencies:
    """Test ProcessorDependencies container."""
    
    def test_default_dependencies(self):
        """Test default dependency creation."""
        deps = ProcessorDependencies.create_default()
        assert deps.signal_handler is not None
        assert deps.config is not None
    
    def test_builder_methods(self):
        """Test builder pattern methods."""
        mock_handler = Mock()
        mock_manager = Mock()
        
        deps = (ProcessorDependencies()
                .with_signal_handler(mock_handler)
                .with_progress_manager(mock_manager))
        
        assert deps.signal_handler is mock_handler
        assert deps.progress_manager is mock_manager
    
    def test_dependency_injection(self):
        """Test that dependencies are properly injected."""
        mock_schema = Mock()
        deps = ProcessorDependencies().with_database_schema(mock_schema)
        
        processor = MockProcessor(dependencies=deps)
        assert processor.dependencies.database_schema is mock_schema
        assert processor._database_schema is mock_schema


class TestProcessorBuilder:
    """Test ProcessorBuilder pattern."""
    
    def test_builder_pattern(self):
        """Test complete builder pattern usage."""
        config = ProcessorConfig(batch_size=2000)
        deps = ProcessorDependencies.create_default()
        
        processor = (ProcessorBuilder(MockProcessor)
                     .with_config(config)
                     .with_dependencies(deps)
                     .with_batch_size(3000)  # Should override config
                     .with_memory_limit(4096)
                     .enable_progress(False)
                     .with_custom_options(test_param="test_value")
                     .build())
        
        assert processor.batch_size == 3000
        assert processor.processor_config.memory_limit_mb == 4096
        assert processor.enable_progress is False
    
    def test_builder_without_config(self):
        """Test builder with minimal configuration."""
        processor = ProcessorBuilder(MockProcessor).build()
        assert isinstance(processor, MockProcessor)
        assert processor.batch_size == 1000  # Default
    
    def test_builder_fluent_interface(self):
        """Test that builder methods return self for chaining."""
        builder = ProcessorBuilder(MockProcessor)
        
        result = builder.with_batch_size(500)
        assert result is builder
        
        result = builder.enable_checkpoints(False)
        assert result is builder


class TestBackwardCompatibility:
    """Test backward compatibility with legacy constructor."""
    
    def test_legacy_constructor_parameters(self):
        """Test that legacy constructor parameters still work."""
        processor = MockProcessor(
            batch_size=1500,
            enable_progress=False,
            checkpoint_interval=50,
            timeout_seconds=300
        )
        
        assert processor.batch_size == 1500
        assert processor.enable_progress is False
        assert processor.checkpoint_interval == 50
        assert processor.timeout_seconds == 300
    
    def test_mixed_legacy_and_new_parameters(self):
        """Test mixing legacy parameters with new dependency injection."""
        mock_handler = Mock()
        config = ProcessorConfig(batch_size=2000)
        
        processor = MockProcessor(
            processor_config=config,
            signal_handler=mock_handler,
            # Legacy parameter should be ignored in favor of config
            batch_size=1000  
        )
        
        assert processor.batch_size == 2000  # From config
        assert processor._signal_handler is mock_handler


class TestMemoryManagement:
    """Test memory management and cleanup."""
    
    def test_context_manager(self):
        """Test context manager interface."""
        cleanup_called = False
        
        class TestProcessor(MockProcessor):
            def cleanup(self):
                nonlocal cleanup_called
                cleanup_called = True
                super().cleanup()
        
        with TestProcessor() as processor:
            assert isinstance(processor, TestProcessor)
        
        assert cleanup_called
    
    def test_cleanup_method(self):
        """Test comprehensive cleanup."""
        processor = MockProcessor()
        
        # Set up some state to clean
        processor._is_processing = True
        processor._progress_callback = Mock()
        processor._timeout_timer = Mock()
        
        processor.cleanup()
        
        assert processor._progress_callback is None
        assert processor._timeout_timer is None
        assert processor._should_stop.is_set()
    
    def test_destructor_cleanup(self):
        """Test that destructor calls cleanup."""
        with patch.object(MockProcessor, 'cleanup') as mock_cleanup:
            processor = MockProcessor()
            del processor
            mock_cleanup.assert_called_once()


class TestSignalHandlerThreadSafety:
    """Test signal handler thread safety improvements."""
    
    def test_deferred_signal_processing(self):
        """Test that signals are processed in background thread."""
        handler = SignalHandler()
        
        # Verify background thread is running
        assert handler._processor_running.is_set()
        assert handler._processor_thread is not None
        assert handler._processor_thread.is_alive()
        
        # Test signal queue
        event = SignalEvent(signal_num=signal.SIGTERM, timestamp=time.time())
        handler._signal_queue.put_nowait(event)
        
        # Give time for processing
        time.sleep(0.1)
        
        # Cleanup
        handler.shutdown()
    
    def test_atomic_state_management(self):
        """Test atomic state flags."""
        handler = SignalHandler()
        
        # Test shutdown state
        assert not handler.is_shutdown_requested()
        handler._shutdown_in_progress.set()
        assert handler.is_shutdown_requested()
        
        # Test pause state  
        assert not handler.is_pause_requested()
        handler._pause_requested.set()
        assert handler.is_pause_requested()
        
        handler.shutdown()
    
    def test_signal_context_minimal_operations(self):
        """Test that signal handlers do minimal work."""
        handler = SignalHandler()
        
        # Mock the queue to verify signal context behavior
        with patch.object(handler._signal_queue, 'put_nowait') as mock_put:
            handler._handle_shutdown_signal(signal.SIGTERM)
            mock_put.assert_called_once()
            
            # Verify the signal event
            call_args = mock_put.call_args[0][0]
            assert isinstance(call_args, SignalEvent)
            assert call_args.signal_num == signal.SIGTERM
        
        handler.shutdown()
    
    def test_signal_processing_loop(self):
        """Test signal processing background loop."""
        handler = SignalHandler()
        
        # Add test handler
        test_handler_called = False
        def test_handler(sig):
            nonlocal test_handler_called
            test_handler_called = True
        
        handler.register_handler("test", test_handler)
        
        # Create and queue a signal event
        event = SignalEvent(signal_num=signal.SIGUSR1, timestamp=time.time())
        handler._signal_queue.put_nowait(event)
        
        # Wait for processing
        time.sleep(0.2)
        
        assert test_handler_called
        
        handler.shutdown()


class TestDatabaseSchemaArchitecture:
    """Test database schema facade architecture."""
    
    def test_schema_delegation(self):
        """Test that schema methods properly delegate to monolithic implementation."""
        with patch('src.database.schema.monolithic_module') as mock_module:
            mock_schema = Mock()
            mock_module.DatabaseSchema.return_value = mock_schema
            
            facade = DatabaseSchema()
            
            # Test delegation
            facade.store_species_range("test_data")
            mock_schema.store_species_range.assert_called_once_with("test_data")
            
            facade.get_experiments(status="active")
            mock_schema.get_experiments.assert_called_once_with(status="active")
    
    def test_lazy_loading_monolithic_schema(self):
        """Test that monolithic schema is loaded lazily."""
        with patch('src.database.schema.monolithic_module') as mock_module:
            mock_schema = Mock()
            mock_module.DatabaseSchema.return_value = mock_schema
            
            facade = DatabaseSchema()
            
            # Should not be loaded yet
            assert not hasattr(facade, '_monolithic_schema')
            
            # First call should load it
            facade._get_monolithic_schema()
            assert hasattr(facade, '_monolithic_schema')
            assert facade._monolithic_schema is mock_schema
            
            # Second call should reuse existing
            result = facade._get_monolithic_schema()
            assert result is mock_schema
            # Should only create one instance
            mock_module.DatabaseSchema.assert_called_once()
    
    def test_all_methods_implemented(self):
        """Test that all expected methods are implemented in facade."""
        facade = DatabaseSchema(Mock())
        
        # Critical methods that must be implemented
        required_methods = [
            'store_species_range', 'store_features_batch', 'store_climate_data_batch',
            'create_experiment', 'update_experiment_status', 'get_experiments',
            'store_resampled_dataset', 'get_resampled_datasets',
            'create_schema', 'drop_schema'
        ]
        
        for method_name in required_methods:
            assert hasattr(facade, method_name), f"Missing method: {method_name}"
            method = getattr(facade, method_name)
            assert callable(method), f"Method not callable: {method_name}"


class TestArchitecturalRegressions:
    """Test for architectural regression prevention."""
    
    def test_no_circular_imports(self):
        """Test that critical modules can be imported without circular dependencies."""
        # These imports should not cause circular import errors
        from src.base.processor import BaseProcessor
        from src.core.signal_handler import SignalHandler  
        from src.database.schema import DatabaseSchema
        
        # Test instantiation
        processor = MockProcessor()
        handler = SignalHandler()
        schema = DatabaseSchema(Mock())
        
        # Cleanup
        processor.cleanup()
        handler.shutdown()
    
    def test_dependency_injection_violations(self):
        """Test that base layer doesn't import from higher layers."""
        import src.base.processor as processor_module
        
        # Base layer should not import from core (except explicitly allowed)
        # This is a compile-time check
        source_lines = []
        with open(processor_module.__file__, 'r') as f:
            source_lines = f.readlines()
        
        # Check for forbidden imports (adjust based on actual architecture)
        forbidden_patterns = [
            'from src.pipelines import',
            'from src.spatial_analysis import',
            'from src.raster_data import'
        ]
        
        for line in source_lines:
            for pattern in forbidden_patterns:
                assert pattern not in line, f"Architectural violation: {line.strip()}"
    
    def test_memory_leak_prevention(self):
        """Test that processors don't leak memory through reference cycles."""
        processor = MockProcessor()
        
        # Create potential reference cycle
        callback = Mock()
        callback.processor = processor
        processor._test_callback = callback
        
        # Cleanup should break cycles
        processor.cleanup()
        
        # Verify callback is cleared (this would be processor-specific)
        assert not hasattr(processor, '_test_callback') or processor._test_callback is None
    
    def test_signal_handler_thread_cleanup(self):
        """Test that signal handler properly cleans up threads."""
        handler = SignalHandler()
        
        # Verify thread is running
        assert handler._processor_thread.is_alive()
        thread_id = handler._processor_thread.ident
        
        # Shutdown should clean up thread
        handler.shutdown()
        
        # Give time for thread to stop
        time.sleep(0.5)
        
        assert not handler._processor_thread.is_alive()


if __name__ == "__main__":
    pytest.main([__file__])