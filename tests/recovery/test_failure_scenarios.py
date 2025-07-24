# tests/recovery/test_failure_scenarios.py
"""Tests for failure recovery and resilience scenarios."""

import pytest
import time
import signal
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import threading
import json
import os

from src.config.config import Config
from src.database.connection import DatabaseManager
from src.pipelines.unified_resampling.pipeline_orchestrator import UnifiedResamplingPipeline
from src.core.progress_manager import get_progress_manager
from src.core.checkpoint_manager import get_checkpoint_manager, CheckpointCorruptedError
from src.core.signal_handler import get_signal_handler
from src.core.process_controller import ProcessController
from src.processors.data_preparation.resampling_processor import ResampledDatasetInfo


@pytest.fixture
def test_config(tmp_path):
    """Create test configuration for recovery scenarios."""
    config = Config()
    
    config.settings.update({
        'checkpointing': {
            'enabled': True,
            'checkpoint_dir': str(tmp_path / 'checkpoints'),
            'compression': 'gzip',
            'validation': {
                'checksum_algorithm': 'sha256',
                'verify_on_load': True,
                'corruption_recovery': True
            }
        },
        'process_management': {
            'graceful_shutdown': {
                'timeout_seconds': 5,
                'save_checkpoint': True
            }
        },
        'resampling': {
            'target_resolution': 1.0,
            'engine': 'numpy'
        },
        'datasets': {
            'target_datasets': [
                {
                    'name': 'test-data',
                    'path': str(tmp_path / 'test.tif'),
                    'data_type': 'test',
                    'band_name': 'test',
                    'enabled': True
                }
            ]
        }
    })
    
    # Create directories
    Path(config.settings['checkpointing']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    
    return config


@pytest.fixture
def mock_db():
    """Create mock database for testing."""
    db = Mock(spec=DatabaseManager)
    db.get_connection.return_value.__enter__ = Mock()
    db.get_connection.return_value.__exit__ = Mock()
    return db


class TestFailureRecovery:
    """Test recovery from various failure scenarios."""
    
    def test_recovery_from_resampling_failure(self, test_config, mock_db):
        """Test recovery when failure occurs during resampling phase."""
        pipeline = UnifiedResamplingPipeline(test_config, mock_db)
        checkpoint_manager = get_checkpoint_manager()
        
        # Track processing attempts
        resampling_attempts = []
        
        def failing_resampling_phase(skip_existing):
            attempt = len(resampling_attempts) + 1
            resampling_attempts.append(attempt)
            
            # First attempt fails after processing one dataset
            if attempt == 1:
                # Simulate partial success
                partial_results = [
                    ResampledDatasetInfo(
                        name='test-dataset-1',
                        source_path=Path('/tmp/test1.tif'),
                        target_resolution=1.0,
                        target_crs='EPSG:4326',
                        bounds=(-10, -10, 10, 10),
                        shape=(20, 20),
                        data_type='test',
                        resampling_method='bilinear',
                        band_name='test1',
                        metadata={}
                    )
                ]
                
                # Save partial progress
                pipeline._checkpoint_data['resampled_datasets'] = partial_results
                pipeline._save_checkpoint('partial_resampling')
                
                # Then fail
                raise RuntimeError("Simulated resampling failure")
            
            # Second attempt succeeds
            return [
                ResampledDatasetInfo(
                    name='test-dataset-1',
                    source_path=Path('/tmp/test1.tif'),
                    target_resolution=1.0,
                    target_crs='EPSG:4326',
                    bounds=(-10, -10, 10, 10),
                    shape=(20, 20),
                    data_type='test',
                    resampling_method='bilinear',
                    band_name='test1',
                    metadata={}
                ),
                ResampledDatasetInfo(
                    name='test-dataset-2',
                    source_path=Path('/tmp/test2.tif'),
                    target_resolution=1.0,
                    target_crs='EPSG:4326',
                    bounds=(-10, -10, 10, 10),
                    shape=(20, 20),
                    data_type='test',
                    resampling_method='bilinear',
                    band_name='test2',
                    metadata={}
                )
            ]
        
        # First attempt - should fail
        with patch.object(pipeline, '_create_experiment', return_value='test-exp-123'):
            with patch.object(pipeline, '_run_resampling_phase', side_effect=failing_resampling_phase):
                with pytest.raises(RuntimeError, match="Simulated resampling failure"):
                    pipeline.run_complete_pipeline('test_recovery')
        
        assert len(resampling_attempts) == 1
        
        # Verify checkpoint was saved
        checkpoints = checkpoint_manager.list_checkpoints(
            processor_name='UnifiedResamplingPipeline',
            status='valid'
        )
        assert any('partial_resampling' in cp.checkpoint_id for cp in checkpoints)
        
        # Second attempt - resume from checkpoint
        pipeline2 = UnifiedResamplingPipeline(test_config, mock_db)
        
        # Find partial checkpoint
        partial_checkpoint = next(
            cp for cp in checkpoints 
            if 'partial_resampling' in cp.checkpoint_id
        )
        
        with patch.object(pipeline2, '_run_resampling_phase', side_effect=failing_resampling_phase):
            with patch.object(pipeline2, '_run_merging_phase', return_value=Mock()):
                with patch.object(pipeline2, '_run_analysis_phase', return_value=Mock()):
                    with patch.object(pipeline2, '_finalize_results', return_value={'success': True}):
                        # Resume from checkpoint
                        results = pipeline2.run_complete_pipeline(
                            'test_recovery_resume',
                            resume_from_checkpoint=partial_checkpoint.checkpoint_id
                        )
        
        # Verify recovery succeeded
        assert results['success'] is True
        assert len(resampling_attempts) == 2  # First failed, second succeeded
    
    def test_corrupted_checkpoint_handling(self, test_config, mock_db):
        """Test handling of corrupted checkpoint files."""
        checkpoint_manager = get_checkpoint_manager()
        
        # Save valid checkpoint
        checkpoint_data = {
            'experiment_id': 'test-123',
            'pipeline_id': 'pipeline-123',
            'completed_phases': ['resampling'],
            'phase_results': {'resampling': 'mock_data'}
        }
        
        checkpoint_id = 'test_corruption'
        checkpoint = checkpoint_manager.save_checkpoint(
            checkpoint_id,
            checkpoint_data,
            level='pipeline'
        )
        
        # Corrupt the checkpoint file
        checkpoint_info = checkpoint_manager.get_checkpoint_info(checkpoint_id)
        checkpoint_path = Path(checkpoint_info.file_path)
        
        # Write garbage to file
        with open(checkpoint_path, 'wb') as f:
            f.write(b'corrupted data')
        
        # Try to load corrupted checkpoint
        with pytest.raises(CheckpointCorruptedError):
            checkpoint_manager.load_checkpoint(checkpoint_id)
        
        # Verify checkpoint is marked as corrupted
        updated_info = checkpoint_manager.get_checkpoint_info(checkpoint_id)
        assert updated_info.status == 'corrupted'
        
        # Test recovery with fallback
        # Create fallback checkpoint
        fallback_data = {
            'experiment_id': 'test-123',
            'pipeline_id': 'pipeline-123',
            'completed_phases': [],  # Earlier state
            'phase_results': {}
        }
        
        fallback_id = 'test_fallback'
        checkpoint_manager.save_checkpoint(
            fallback_id,
            fallback_data,
            level='pipeline'
        )
        
        # Pipeline should be able to use fallback
        pipeline = UnifiedResamplingPipeline(test_config, mock_db)
        
        # Mock load to try corrupted first, then fallback
        original_load = checkpoint_manager.load_checkpoint
        load_attempts = []
        
        def mock_load(checkpoint_id):
            load_attempts.append(checkpoint_id)
            if checkpoint_id == 'test_corruption':
                raise CheckpointCorruptedError("Corrupted")
            return original_load(checkpoint_id)
        
        with patch.object(checkpoint_manager, 'load_checkpoint', side_effect=mock_load):
            with patch.object(checkpoint_manager, 'list_checkpoints') as mock_list:
                # Return both checkpoints, corrupted first
                mock_list.return_value = [
                    Mock(checkpoint_id='test_corruption', status='valid'),
                    Mock(checkpoint_id='test_fallback', status='valid')
                ]
                
                # Try to resume - should fallback
                with patch.object(pipeline, '_load_checkpoint') as mock_pipeline_load:
                    # Simulate the pipeline trying checkpoints
                    try:
                        checkpoint_manager.load_checkpoint('test_corruption')
                    except CheckpointCorruptedError:
                        # Fallback to next checkpoint
                        checkpoint_manager.load_checkpoint('test_fallback')
        
        # Verify fallback was used
        assert 'test_fallback' in load_attempts
    
    def test_partial_processing_resume(self, test_config, mock_db):
        """Test resuming from partial processing within a phase."""
        pipeline = UnifiedResamplingPipeline(test_config, mock_db)
        
        # Simulate partial dataset processing
        partial_checkpoint_data = {
            'experiment_id': 'test-partial-123',
            'pipeline_id': 'pipeline-partial',
            'current_phase': 'resampling',
            'completed_phases': [],
            'phase_results': {},
            'resampled_datasets': [
                {
                    'name': 'dataset-1',
                    'source_path': '/tmp/data1.tif',
                    'target_resolution': 1.0,
                    'target_crs': 'EPSG:4326',
                    'bounds': [-10, -10, 10, 10],
                    'shape': [20, 20],
                    'data_type': 'test',
                    'resampling_method': 'bilinear',
                    'band_name': 'test1',
                    'metadata': {}
                }
                # dataset-2 not yet processed
            ]
        }
        
        # Save partial checkpoint
        checkpoint_manager = get_checkpoint_manager()
        partial_checkpoint_id = 'partial_phase_checkpoint'
        checkpoint_manager.save_checkpoint(
            partial_checkpoint_id,
            partial_checkpoint_data,
            level='phase',
            metadata={'datasets_completed': 1, 'total_datasets': 2}
        )
        
        # Resume from partial checkpoint
        pipeline._load_checkpoint(partial_checkpoint_id)
        
        # Verify partial state was restored
        assert len(pipeline.resampled_datasets) == 1
        assert pipeline.resampled_datasets[0].name == 'dataset-1'
        assert pipeline.current_phase == 'resampling'
        
        # Mock remaining processing
        def continue_resampling(skip_existing):
            # Should only process remaining dataset
            existing = pipeline.resampled_datasets
            new_dataset = ResampledDatasetInfo(
                name='dataset-2',
                source_path=Path('/tmp/data2.tif'),
                target_resolution=1.0,
                target_crs='EPSG:4326',
                bounds=(-10, -10, 10, 10),
                shape=(20, 20),
                data_type='test',
                resampling_method='bilinear',
                band_name='test2',
                metadata={}
            )
            return existing + [new_dataset]
        
        with patch.object(pipeline, '_run_resampling_phase', side_effect=continue_resampling):
            with patch.object(pipeline, '_run_merging_phase', return_value=Mock()):
                with patch.object(pipeline, '_run_analysis_phase', return_value=Mock()):
                    with patch.object(pipeline, '_finalize_results', return_value={'complete': True}):
                        results = pipeline.run_complete_pipeline(
                            'test_partial_resume',
                            resume_from_checkpoint=partial_checkpoint_id
                        )
        
        # Verify completion
        assert results['complete'] is True
        assert len(pipeline.resampled_datasets) == 2
    
    def test_signal_based_process_control(self, test_config, mock_db):
        """Test signal-based pause, resume, and stop functionality."""
        process_controller = ProcessController()
        signal_handler = get_signal_handler()
        
        # Track process states
        process_states = []
        
        # Create mock process
        class MockProcess:
            def __init__(self):
                self.running = True
                self.paused = False
                self.checkpoint_saved = False
            
            def run(self):
                while self.running:
                    if signal_handler.is_paused:
                        if not self.paused:
                            process_states.append('paused')
                            self.paused = True
                    else:
                        if self.paused:
                            process_states.append('resumed')
                            self.paused = False
                        process_states.append('running')
                    
                    if signal_handler.should_shutdown:
                        process_states.append('stopping')
                        self.checkpoint_saved = True
                        self.running = False
                        break
                    
                    time.sleep(0.1)
        
        # Run process in thread
        mock_process = MockProcess()
        process_thread = threading.Thread(target=mock_process.run)
        process_thread.start()
        
        # Wait for process to start
        time.sleep(0.2)
        assert 'running' in process_states
        
        # Test pause
        signal_handler.handle_signal(signal.SIGUSR1, None)
        time.sleep(0.2)
        assert 'paused' in process_states
        
        # Test resume
        signal_handler.handle_signal(signal.SIGUSR2, None)
        time.sleep(0.2)
        assert 'resumed' in process_states
        
        # Test stop
        signal_handler.handle_signal(signal.SIGTERM, None)
        process_thread.join(timeout=2)
        
        assert 'stopping' in process_states
        assert mock_process.checkpoint_saved
    
    def test_memory_pressure_recovery(self, test_config, mock_db):
        """Test recovery from high memory pressure situations."""
        pipeline = UnifiedResamplingPipeline(test_config, mock_db)
        memory_manager = pipeline.memory_manager
        
        # Track memory responses
        memory_responses = []
        
        # Mock memory pressure scenarios
        def simulate_memory_pressure():
            # Simulate escalating memory pressure
            pressure_levels = ['normal', 'warning', 'high', 'critical']
            
            for level in pressure_levels:
                with patch.object(memory_manager, 'get_memory_pressure_level') as mock_pressure:
                    mock_pressure.return_value = Mock(value=level)
                    
                    # Get pressure response
                    response = memory_manager.get_pressure_response(level)
                    memory_responses.append({
                        'level': level,
                        'response': response
                    })
                    
                    # Critical pressure should trigger checkpoint and exit
                    if level == 'critical':
                        if response.get('checkpoint_and_exit'):
                            pipeline._save_checkpoint('critical_memory_checkpoint')
                            raise MemoryError("Critical memory pressure")
                
                time.sleep(0.1)
            
            return []  # Normal return if we don't hit critical
        
        # Run pipeline with memory pressure simulation
        with patch.object(pipeline, '_create_experiment', return_value='test-mem-123'):
            with patch.object(pipeline, '_run_resampling_phase', side_effect=simulate_memory_pressure):
                with pytest.raises(MemoryError, match="Critical memory pressure"):
                    pipeline.run_complete_pipeline('test_memory_pressure')
        
        # Verify appropriate responses at each level
        assert len(memory_responses) == 4
        
        # Check responses escalate appropriately
        normal_response = next(r for r in memory_responses if r['level'] == 'normal')
        assert not normal_response['response'].get('pause_processing')
        
        high_response = next(r for r in memory_responses if r['level'] == 'high')
        assert high_response['response'].get('force_gc')
        assert high_response['response'].get('checkpoint')
        
        critical_response = next(r for r in memory_responses if r['level'] == 'critical')
        assert critical_response['response'].get('checkpoint_and_exit')
        
        # Verify checkpoint was saved
        checkpoint_manager = get_checkpoint_manager()
        checkpoints = checkpoint_manager.list_checkpoints(
            processor_name='UnifiedResamplingPipeline'
        )
        assert any('critical_memory' in cp.checkpoint_id for cp in checkpoints)
    
    def test_concurrent_checkpoint_access(self, test_config):
        """Test handling concurrent checkpoint access and conflicts."""
        checkpoint_manager = get_checkpoint_manager()
        
        # Create test checkpoint
        checkpoint_id = 'concurrent_test'
        test_data = {'value': 0}
        
        checkpoint_manager.save_checkpoint(
            checkpoint_id,
            test_data,
            level='test'
        )
        
        # Simulate concurrent modifications
        results = []
        errors = []
        
        def modify_checkpoint(thread_id):
            try:
                # Load checkpoint
                data = checkpoint_manager.load_checkpoint(checkpoint_id)
                
                # Simulate processing time
                time.sleep(0.1)
                
                # Modify data
                data['value'] += 1
                data[f'thread_{thread_id}'] = True
                
                # Try to save (this might conflict)
                new_id = f"{checkpoint_id}_thread_{thread_id}"
                checkpoint_manager.save_checkpoint(
                    new_id,
                    data,
                    level='test'
                )
                
                results.append(thread_id)
                
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Run concurrent threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=modify_checkpoint, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # All threads should complete (no deadlocks)
        assert len(results) + len(errors) == 3
        
        # Verify individual checkpoints were created
        all_checkpoints = checkpoint_manager.list_checkpoints(level='test')
        thread_checkpoints = [
            cp for cp in all_checkpoints 
            if 'thread_' in cp.checkpoint_id
        ]
        assert len(thread_checkpoints) == len(results)
    
    def test_checkpoint_cleanup_during_recovery(self, test_config):
        """Test checkpoint cleanup doesn't interfere with recovery."""
        checkpoint_manager = get_checkpoint_manager()
        
        # Create multiple checkpoints simulating pipeline progression
        checkpoint_ids = []
        for i in range(5):
            checkpoint_data = {
                'experiment_id': 'cleanup-test',
                'progress': i * 20,
                'timestamp': time.time() - (5 - i) * 3600  # Older checkpoints
            }
            
            checkpoint_id = f'cleanup_test_{i}'
            checkpoint_manager.save_checkpoint(
                checkpoint_id,
                checkpoint_data,
                level='pipeline'
            )
            checkpoint_ids.append(checkpoint_id)
            
            time.sleep(0.1)  # Ensure different timestamps
        
        # Start recovery from middle checkpoint
        recovery_checkpoint = checkpoint_ids[2]
        recovery_data = checkpoint_manager.load_checkpoint(recovery_checkpoint)
        
        # Run cleanup while recovery is in progress
        # Should keep minimum checkpoints and not delete the one being used
        cleaned = checkpoint_manager.cleanup_old_checkpoints(
            days_old=0,  # Very aggressive cleanup
            keep_minimum={'pipeline': 2}  # Keep at least 2
        )
        
        # Verify recovery checkpoint still exists
        assert checkpoint_manager.validate_checkpoint(recovery_checkpoint)
        
        # Verify cleanup happened but kept minimum
        remaining = checkpoint_manager.list_checkpoints(
            processor_name='UnifiedResamplingPipeline',
            level='pipeline'
        )
        assert len(remaining) >= 2  # At least minimum kept
    
    def test_graceful_degradation(self, test_config, mock_db):
        """Test pipeline can continue with degraded functionality."""
        pipeline = UnifiedResamplingPipeline(test_config, mock_db)
        
        # Simulate various component failures
        failures = {
            'checkpoint_save': False,
            'progress_tracking': False,
            'memory_monitoring': False
        }
        
        # Mock component failures
        def failing_checkpoint_save(*args, **kwargs):
            failures['checkpoint_save'] = True
            raise IOError("Checkpoint save failed")
        
        def failing_progress_update(*args, **kwargs):
            failures['progress_tracking'] = True
            raise RuntimeError("Progress tracking failed")
        
        # Patch to simulate failures
        with patch.object(pipeline.checkpoint_manager, 'save_checkpoint', side_effect=failing_checkpoint_save):
            with patch.object(pipeline.progress_manager, 'update', side_effect=failing_progress_update):
                
                # Pipeline should still complete core functionality
                with patch.object(pipeline, '_create_experiment', return_value='test-degrade'):
                    with patch.object(pipeline, '_run_resampling_phase', return_value=[]):
                        with patch.object(pipeline, '_run_merging_phase', return_value=Mock()):
                            with patch.object(pipeline, '_run_analysis_phase', return_value=Mock()):
                                with patch.object(pipeline, '_finalize_results', return_value={'degraded': True}):
                                    
                                    # Should complete despite component failures
                                    results = pipeline.run_complete_pipeline('test_degraded')
        
        # Verify pipeline completed
        assert results['degraded'] is True
        
        # Verify failures were encountered but handled
        assert failures['checkpoint_save']
        assert failures['progress_tracking']


class TestProcessControllerRecovery:
    """Test process controller recovery scenarios."""
    
    def test_process_crash_recovery(self, test_config):
        """Test recovery when managed process crashes."""
        controller = ProcessController()
        
        # Create a process that will crash
        crash_file = test_config.paths['project_root'] / 'crash_test.py'
        crash_file.write_text("""
import time
import sys

# Simulate some work then crash
time.sleep(1)
sys.exit(1)  # Simulate crash
""")
        
        # Start process with auto-restart
        process_name = 'crash_test_process'
        try:
            pid = controller.start_process(
                name=process_name,
                command=['python', str(crash_file)],
                auto_restart=True,
                max_restarts=2
            )
            
            # Wait for process to crash and restart
            time.sleep(3)
            
            # Check process was restarted
            status = controller.get_process_status(process_name)
            assert status is not None
            assert status.restart_count > 0
            
        finally:
            # Cleanup
            controller.stop_process(process_name)
            crash_file.unlink()
    
    def test_orphaned_process_cleanup(self, test_config):
        """Test cleanup of orphaned processes."""
        controller = ProcessController()
        
        # Create a long-running process
        sleep_file = test_config.paths['project_root'] / 'sleep_test.py'
        sleep_file.write_text("""
import time
time.sleep(60)  # Long sleep
""")
        
        # Start process
        process_name = 'orphan_test'
        try:
            pid = controller.start_process(
                name=process_name,
                command=['python', str(sleep_file)]
            )
            
            # Simulate controller crash by clearing its tracking
            controller._processes.clear()
            
            # Controller should detect orphaned process
            orphans = controller.detect_orphaned_processes()
            assert len(orphans) > 0
            
            # Clean up orphans
            controller.cleanup_orphaned_processes()
            
            # Verify process was terminated
            import psutil
            with pytest.raises(psutil.NoSuchProcess):
                psutil.Process(pid)
                
        finally:
            sleep_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])