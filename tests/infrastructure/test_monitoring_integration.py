"""Integration tests for unified monitoring and logging system."""
import pytest
import time
import subprocess
import json
from pathlib import Path
import sys
from unittest.mock import Mock, patch
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.infrastructure.logging import get_logger, setup_logging, LoggingContext
from src.infrastructure.monitoring import UnifiedMonitor
from src.core.enhanced_progress_manager import get_enhanced_progress_manager
from src.pipelines.enhanced_context import EnhancedPipelineContext


class TestMonitoringIntegration:
    """Test integration of monitoring and logging components."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()
        config.get.side_effect = lambda key, default=None: {
            'logging.level': 'INFO',
            'logging.database.batch_size': 10,
            'logging.database.flush_interval': 0.5,
            'monitoring.enable_metrics': True,
            'monitoring.metrics_interval': 1.0
        }.get(key, default)
        return config
    
    @pytest.fixture
    def mock_db_manager(self):
        """Create mock database manager."""
        mock_db = Mock()
        mock_cursor = Mock()
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=None)
        mock_cursor.fetchone.return_value = None
        mock_cursor.fetchall.return_value = []
        mock_db.get_cursor.return_value = mock_cursor
        return mock_db
    
    def test_full_pipeline_monitoring(self, mock_config, mock_db_manager):
        """Test complete pipeline execution with monitoring."""
        # Setup logging
        setup_logging(
            config=mock_config,
            db_manager=mock_db_manager,
            console=False,
            database=True
        )
        
        logger = get_logger("test.pipeline")
        
        # Create enhanced pipeline context
        context = EnhancedPipelineContext(
            config=mock_config,
            db=mock_db_manager,
            experiment_id="test-pipeline-123",
            checkpoint_dir=Path("/tmp/checkpoints"),
            output_dir=Path("/tmp/outputs")
        )
        
        # Start monitoring
        context.start_monitoring()
        
        try:
            # Simulate pipeline execution
            with context.logging_context.pipeline("test_pipeline"):
                logger.info("Pipeline started")
                
                # Stage 1: Data loading
                with context.logging_context.stage("data_loading"):
                    logger.info("Loading data...")
                    
                    # Simulate progress
                    context.progress_manager.create_pipeline(
                        pipeline_id="pipeline/test",
                        total_phases=3
                    )
                    
                    # Record some metrics
                    context.record_metrics(
                        memory_mb=512,
                        files_loaded=100
                    )
                    
                    # Simulate some work
                    time.sleep(0.1)
                    
                    # Log performance
                    logger.log_performance("data_load", 0.1, files=100)
                
                # Stage 2: Processing
                with context.logging_context.stage("processing"):
                    logger.info("Processing data...")
                    
                    # Simulate an error
                    try:
                        raise ValueError("Simulated processing error")
                    except ValueError:
                        logger.error("Processing failed", exc_info=True)
                    
                    # Continue despite error
                    logger.warning("Continuing with partial data")
                
                # Stage 3: Output
                with context.logging_context.stage("output"):
                    logger.info("Writing results...")
                    context.record_metrics(
                        output_files=10,
                        total_size_mb=1024
                    )
                
                logger.info("Pipeline completed")
                
        finally:
            context.stop_monitoring()
        
        # Verify logging and monitoring worked
        assert context.experiment_id == "test-pipeline-123"
        assert context.monitor is not None
        assert context.progress_manager is not None
    
    def test_daemon_process_monitoring(self, mock_config, mock_db_manager, tmp_path):
        """Test monitoring of daemon processes."""
        from src.core.process_controller_enhanced import EnhancedProcessController
        
        # Create enhanced process controller
        controller = EnhancedProcessController(
            pid_dir=tmp_path / "pid",
            log_dir=tmp_path / "logs",
            experiment_id="test-daemon-exp"
        )
        
        # Python script that logs and crashes
        test_script = tmp_path / "test_daemon.py"
        test_script.write_text("""
import sys
import time
import logging

# This would be injected by the daemon wrapper
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

logger.info("Daemon started")
time.sleep(0.1)
logger.error("Simulating crash")
raise RuntimeError("Daemon crash test")
""")
        
        # Start as daemon
        with patch.object(controller, '_start_daemon_process') as mock_start:
            # Mock the daemon start to avoid actual forking in tests
            mock_start.return_value = 12345
            
            pid = controller.start_process(
                name="test_daemon",
                command=[sys.executable, str(test_script)],
                daemon_mode=True,
                auto_restart=False
            )
            
            assert pid == 12345
            assert mock_start.called
    
    def test_enhanced_signal_handling(self):
        """Test enhanced signal handler with logging."""
        from src.core.signal_handler_enhanced import EnhancedSignalHandler
        
        cleanup_called = False
        
        def cleanup():
            nonlocal cleanup_called
            cleanup_called = True
        
        # Create handler
        handler = EnhancedSignalHandler(cleanup_callback=cleanup)
        
        # Test uncaught exception handling
        def raise_error():
            raise ValueError("Test uncaught exception")
        
        # Replace sys.excepthook temporarily
        old_hook = sys.excepthook
        try:
            # Trigger exception handling
            try:
                raise_error()
            except ValueError:
                exc_info = sys.exc_info()
                handler._handle_uncaught_exception(*exc_info)
            
            # Cleanup should have been called
            assert cleanup_called
            assert handler._shutdown_requested
            
        finally:
            sys.excepthook = old_hook
    
    def test_monitor_cli_integration(self, mock_db_manager, tmp_path):
        """Test monitor CLI functionality."""
        from scripts.monitor import MonitorCLI
        
        # Create CLI instance
        with patch('src.database.connection.DatabaseManager', return_value=mock_db_manager):
            cli = MonitorCLI()
            
            # Mock experiment data
            mock_cursor = mock_db_manager.get_cursor().__enter__()
            mock_cursor.fetchone.return_value = {
                'id': 'test-exp',
                'name': 'Test Experiment',
                'status': 'running',
                'started_at': datetime.now(),
                'completed_at': None
            }
            
            # Mock progress data
            mock_cursor.fetchall.return_value = [
                {
                    'node_id': 'pipeline/test',
                    'node_name': 'Test Pipeline',
                    'status': 'running',
                    'progress_percent': 50.0,
                    'parent_id': None,
                    'node_level': 'pipeline'
                }
            ]
            
            # Test status command
            args = Mock(experiment='test-exp')
            
            # Should not raise
            cli.status(args)
            
            # Test logs command
            mock_cursor.fetchall.return_value = [
                {
                    'id': 'log-1',
                    'timestamp': datetime.now(),
                    'level': 'INFO',
                    'message': 'Test log',
                    'stage': 'testing',
                    'traceback': None
                }
            ]
            
            args = Mock(
                experiment='test-exp',
                level=None,
                search=None,
                since=None,
                limit=10,
                json=False,
                traceback=False
            )
            
            cli.logs(args)
    
    def test_progress_manager_integration(self, mock_db_manager):
        """Test enhanced progress manager with database backend."""
        manager = get_enhanced_progress_manager(
            experiment_id="test-progress",
            db_manager=mock_db_manager
        )
        
        # Create pipeline structure
        manager.create_pipeline(
            pipeline_id="test_pipeline",
            total_phases=3,
            metadata={'version': '1.0'}
        )
        
        # Create phases
        manager.create_phase(
            pipeline_id="test_pipeline",
            phase_id="phase1",
            phase_name="Data Loading",
            total_steps=2
        )
        
        # Update progress
        manager.update_progress(
            node_id="test_pipeline/phase1",
            completed_units=1,
            total_units=2,
            message="Loading file 1 of 2"
        )
        
        # Get progress tree
        tree = manager.get_progress_tree("test_pipeline")
        assert tree is not None
    
    def test_logging_decorators(self, mock_config, mock_db_manager):
        """Test logging decorators for automatic instrumentation."""
        from src.infrastructure.logging.decorators import log_operation, log_stage
        
        setup_logging(config=mock_config, db_manager=mock_db_manager)
        
        call_count = 0
        error_count = 0
        
        @log_operation("test_operation")
        def test_func(value):
            nonlocal call_count
            call_count += 1
            if value < 0:
                raise ValueError("Negative value")
            return value * 2
        
        # Test successful operation
        result = test_func(5)
        assert result == 10
        assert call_count == 1
        
        # Test operation with error
        with pytest.raises(ValueError):
            test_func(-1)
        assert call_count == 2
    
    def test_tmux_monitoring_setup(self, tmp_path):
        """Test tmux monitoring script setup."""
        # Create mock tmux script
        tmux_script = tmp_path / "run_monitored.sh"
        tmux_script.write_text("""#!/bin/bash
# Mock tmux monitoring script
echo "Setting up monitoring for experiment: $1"
""")
        tmux_script.chmod(0o755)
        
        # Test script execution
        result = subprocess.run(
            [str(tmux_script), "test-experiment"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Setting up monitoring for experiment: test-experiment" in result.stdout