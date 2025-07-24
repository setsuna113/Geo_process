# tests/integration/test_progress_checkpointing.py
"""Integration tests for progress monitoring and checkpointing functionality."""

import pytest
import time
import signal
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import xarray as xr
from datetime import datetime
import json
import threading

from src.config.config import Config
from src.database.connection import DatabaseManager
from src.database.schema import schema
from src.pipelines.unified_resampling.pipeline_orchestrator import UnifiedResamplingPipeline
from src.core.progress_manager import get_progress_manager, ProgressNode
from src.core.checkpoint_manager import get_checkpoint_manager
from src.core.signal_handler import get_signal_handler
from src.base.memory_manager import get_memory_manager
from src.processors.data_preparation.resampling_processor import ResampledDatasetInfo


@pytest.fixture
def test_config(tmp_path):
    """Create test configuration."""
    config = Config()
    
    # Override with test settings
    config.settings.update({
        'resampling': {
            'target_resolution': 0.5,
            'target_crs': 'EPSG:4326',
            'strategies': {'richness_data': 'sum'},
            'engine': 'numpy',
            'chunk_size': 100
        },
        'datasets': {
            'target_datasets': [
                {
                    'name': 'test-dataset-1',
                    'path': str(tmp_path / 'test1.tif'),
                    'data_type': 'richness_data',
                    'band_name': 'test1',
                    'enabled': True
                },
                {
                    'name': 'test-dataset-2',
                    'path': str(tmp_path / 'test2.tif'),
                    'data_type': 'richness_data',
                    'band_name': 'test2',
                    'enabled': True
                }
            ]
        },
        'som_analysis': {
            'default_grid_size': [4, 4],
            'iterations': 100
        },
        'checkpointing': {
            'enabled': True,
            'checkpoint_dir': str(tmp_path / 'checkpoints'),
            'intervals': {
                'time_based': 5,
                'item_based': 10
            }
        },
        'progress_monitoring': {
            'enabled': True,
            'reporting_interval_seconds': 0.1
        }
    })
    
    # Create checkpoint directory
    Path(config.settings['checkpointing']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    
    return config


@pytest.fixture
def mock_db():
    """Create mock database connection."""
    db = Mock(spec=DatabaseManager)
    db.get_connection.return_value.__enter__ = Mock()
    db.get_connection.return_value.__exit__ = Mock()
    return db


@pytest.fixture
def sample_resampled_data():
    """Create sample resampled dataset info."""
    return [
        ResampledDatasetInfo(
            name='test-dataset-1',
            source_path=Path('/tmp/test1.tif'),
            target_resolution=0.5,
            target_crs='EPSG:4326',
            bounds=(-10, -10, 10, 10),
            shape=(40, 40),
            data_type='richness_data',
            resampling_method='sum',
            band_name='test1',
            metadata={}
        ),
        ResampledDatasetInfo(
            name='test-dataset-2',
            source_path=Path('/tmp/test2.tif'),
            target_resolution=0.5,
            target_crs='EPSG:4326',
            bounds=(-10, -10, 10, 10),
            shape=(40, 40),
            data_type='richness_data',
            resampling_method='sum',
            band_name='test2',
            metadata={}
        )
    ]


@pytest.fixture
def sample_merged_data():
    """Create sample merged xarray dataset."""
    x = np.linspace(-10, 10, 40)
    y = np.linspace(-10, 10, 40)
    
    data1 = np.random.rand(40, 40)
    data2 = np.random.rand(40, 40)
    
    return xr.Dataset({
        'test1': xr.DataArray(data1, coords={'x': x, 'y': y}, dims=['y', 'x']),
        'test2': xr.DataArray(data2, coords={'x': x, 'y': y}, dims=['y', 'x'])
    })


class TestProgressCheckpointing:
    """Test progress monitoring and checkpointing integration."""
    
    def test_full_pipeline_with_progress(self, test_config, mock_db, sample_resampled_data, sample_merged_data):
        """Test complete pipeline execution with progress monitoring."""
        # Setup
        pipeline = UnifiedResamplingPipeline(test_config, mock_db)
        progress_manager = get_progress_manager()
        
        # Track progress events
        progress_events = []
        
        def progress_callback(node: ProgressNode):
            progress_events.append({
                'name': node.name,
                'progress': node.progress_percent,
                'status': node.status,
                'level': node.level
            })
        
        progress_manager.register_callback('any', progress_callback)
        
        # Mock pipeline methods
        with patch.object(pipeline, '_create_experiment', return_value='test-exp-123'):
            with patch.object(pipeline, '_run_resampling_phase', return_value=sample_resampled_data):
                with patch.object(pipeline, '_run_merging_phase', return_value=sample_merged_data):
                    with patch.object(pipeline, '_run_analysis_phase', return_value={'results': 'mock'}):
                        with patch.object(pipeline, '_finalize_results', return_value={'final': 'results'}):
                            # Run pipeline
                            results = pipeline.run_complete_pipeline(
                                experiment_name='test_experiment',
                                description='Test run with progress'
                            )
        
        # Verify results
        assert results['final'] == 'results'
        assert pipeline.experiment_id == 'test-exp-123'
        
        # Verify progress events
        assert len(progress_events) > 0
        
        # Check we got events for all phases
        phase_names = [e['name'] for e in progress_events if 'phase' in e['name']]
        assert any('resampling' in name for name in phase_names)
        assert any('merging' in name for name in phase_names)
        assert any('analysis' in name for name in phase_names)
        assert any('export' in name for name in phase_names)
        
        # Verify progress completion
        pipeline_events = [e for e in progress_events if e['level'] == 'pipeline']
        if pipeline_events:
            final_event = pipeline_events[-1]
            assert final_event['status'] == 'completed'
    
    def test_checkpoint_save_restore_each_phase(self, test_config, mock_db, sample_resampled_data, sample_merged_data):
        """Test checkpoint saving and restoration at each phase."""
        pipeline = UnifiedResamplingPipeline(test_config, mock_db)
        checkpoint_manager = get_checkpoint_manager()
        
        # Track saved checkpoints
        saved_checkpoints = []
        
        # Mock checkpoint save
        original_save = checkpoint_manager.save_checkpoint
        def mock_save_checkpoint(checkpoint_id, data, **kwargs):
            saved_checkpoints.append({
                'id': checkpoint_id,
                'data': data,
                'level': kwargs.get('level'),
                'parent_id': kwargs.get('parent_id')
            })
            return original_save(checkpoint_id, data, **kwargs)
        
        checkpoint_manager.save_checkpoint = mock_save_checkpoint
        
        # Mock pipeline methods
        with patch.object(pipeline, '_create_experiment', return_value='test-exp-123'):
            with patch.object(pipeline, '_run_resampling_phase', return_value=sample_resampled_data):
                with patch.object(pipeline, '_run_merging_phase', return_value=sample_merged_data):
                    with patch.object(pipeline, '_run_analysis_phase', return_value={'results': 'mock'}):
                        with patch.object(pipeline, '_finalize_results', return_value={'final': 'results'}):
                            # Run pipeline
                            pipeline.run_complete_pipeline(
                                experiment_name='test_checkpoint_save'
                            )
        
        # Verify checkpoints were saved for each phase
        assert len(saved_checkpoints) >= 4  # One for each phase
        
        phase_checkpoints = [cp for cp in saved_checkpoints if cp['level'] == 'pipeline']
        assert len(phase_checkpoints) >= 4
        
        # Verify checkpoint data contains required fields
        for checkpoint in phase_checkpoints:
            data = checkpoint['data']
            assert 'experiment_id' in data
            assert 'pipeline_id' in data
            assert 'completed_phases' in data
            assert 'phase_results' in data
        
        # Test restore from checkpoint
        last_checkpoint = phase_checkpoints[-1]
        
        # Create new pipeline and restore
        pipeline2 = UnifiedResamplingPipeline(test_config, mock_db)
        pipeline2._load_checkpoint(last_checkpoint['id'])
        
        # Verify state was restored
        assert pipeline2.experiment_id == 'test-exp-123'
        assert len(pipeline2._checkpoint_data['completed_phases']) > 0
    
    def test_signal_handling_graceful_shutdown(self, test_config, mock_db):
        """Test signal handling and graceful shutdown."""
        pipeline = UnifiedResamplingPipeline(test_config, mock_db)
        signal_handler = get_signal_handler()
        
        # Track if shutdown callback was called
        shutdown_called = threading.Event()
        checkpoint_saved = threading.Event()
        
        # Mock checkpoint save
        original_save = pipeline._save_checkpoint
        def mock_save_checkpoint(checkpoint_id):
            if 'shutdown' in checkpoint_id:
                checkpoint_saved.set()
            return original_save(checkpoint_id)
        
        pipeline._save_checkpoint = mock_save_checkpoint
        
        # Start pipeline in thread
        def run_pipeline():
            try:
                with patch.object(pipeline, '_create_experiment', return_value='test-exp-123'):
                    # Simulate long-running phase
                    def slow_phase(*args, **kwargs):
                        for i in range(10):
                            time.sleep(0.1)
                            if signal_handler.should_shutdown:
                                shutdown_called.set()
                                raise KeyboardInterrupt("Shutdown requested")
                        return []
                    
                    with patch.object(pipeline, '_run_resampling_phase', side_effect=slow_phase):
                        pipeline.run_complete_pipeline('test_shutdown')
            except KeyboardInterrupt:
                pass
        
        # Start pipeline
        pipeline_thread = threading.Thread(target=run_pipeline)
        pipeline_thread.start()
        
        # Give it time to start
        time.sleep(0.2)
        
        # Send shutdown signal
        signal_handler.handle_signal(signal.SIGTERM, None)
        
        # Wait for thread to finish
        pipeline_thread.join(timeout=5)
        
        # Verify shutdown was handled
        assert shutdown_called.is_set()
        assert checkpoint_saved.is_set()
    
    def test_resume_from_checkpoint_states(self, test_config, mock_db, sample_resampled_data, sample_merged_data):
        """Test resuming from various checkpoint states."""
        checkpoint_manager = get_checkpoint_manager()
        
        # Test cases for different resume points
        test_cases = [
            {
                'name': 'resume_after_resampling',
                'completed_phases': ['resampling'],
                'phase_results': {'resampling': sample_resampled_data},
                'expected_phases_to_run': ['merging', 'analysis', 'export']
            },
            {
                'name': 'resume_after_merging',
                'completed_phases': ['resampling', 'merging'],
                'phase_results': {
                    'resampling': sample_resampled_data,
                    'merging': sample_merged_data
                },
                'expected_phases_to_run': ['analysis', 'export']
            },
            {
                'name': 'resume_after_analysis',
                'completed_phases': ['resampling', 'merging', 'analysis'],
                'phase_results': {
                    'resampling': sample_resampled_data,
                    'merging': sample_merged_data,
                    'analysis': {'som_results': 'mock'}
                },
                'expected_phases_to_run': ['export']
            }
        ]
        
        for test_case in test_cases:
            # Create checkpoint
            checkpoint_data = {
                'experiment_id': f"test-exp-{test_case['name']}",
                'pipeline_id': f"pipeline-{test_case['name']}",
                'completed_phases': test_case['completed_phases'],
                'phase_results': test_case['phase_results'],
                'resampled_datasets': [
                    {
                        'name': d.name,
                        'source_path': str(d.source_path),
                        'target_resolution': d.target_resolution,
                        'target_crs': d.target_crs,
                        'bounds': d.bounds,
                        'shape': d.shape,
                        'data_type': d.data_type,
                        'resampling_method': d.resampling_method,
                        'band_name': d.band_name,
                        'metadata': d.metadata
                    }
                    for d in sample_resampled_data
                ]
            }
            
            checkpoint_id = f"checkpoint_{test_case['name']}"
            checkpoint_manager.save_checkpoint(
                checkpoint_id,
                checkpoint_data,
                level='pipeline'
            )
            
            # Create pipeline and resume
            pipeline = UnifiedResamplingPipeline(test_config, mock_db)
            
            # Track which phases run
            phases_run = []
            
            # Mock phase methods
            def make_phase_tracker(phase_name):
                def track_phase(*args, **kwargs):
                    phases_run.append(phase_name)
                    return test_case['phase_results'].get(phase_name, {'mock': 'result'})
                return track_phase
            
            with patch.object(pipeline, '_run_resampling_phase', side_effect=make_phase_tracker('resampling')):
                with patch.object(pipeline, '_run_merging_phase', side_effect=make_phase_tracker('merging')):
                    with patch.object(pipeline, '_run_analysis_phase', side_effect=make_phase_tracker('analysis')):
                        with patch.object(pipeline, '_finalize_results', side_effect=make_phase_tracker('export')):
                            # Resume from checkpoint
                            pipeline.run_complete_pipeline(
                                experiment_name=f"resume_{test_case['name']}",
                                resume_from_checkpoint=checkpoint_id
                            )
            
            # Verify only expected phases ran
            assert phases_run == test_case['expected_phases_to_run']
    
    def test_progress_hierarchical_tracking(self, test_config, mock_db):
        """Test hierarchical progress tracking (pipeline -> phase -> step -> substep)."""
        pipeline = UnifiedResamplingPipeline(test_config, mock_db)
        progress_manager = get_progress_manager()
        
        # Track all progress nodes
        all_nodes = {}
        
        def track_progress(node: ProgressNode):
            all_nodes[node.node_id] = {
                'name': node.name,
                'level': node.level,
                'parent_id': node.parent_id,
                'progress': node.progress_percent,
                'status': node.status
            }
        
        progress_manager.register_callback('any', track_progress)
        
        # Mock resampling phase with detailed progress
        def detailed_resampling_phase(skip_existing):
            # Simulate detailed progress reporting
            phase_id = f"{pipeline.pipeline_id}_resampling"
            step_id = f"{phase_id}_datasets"
            
            # Create step node
            progress_manager.create_step(
                step_id,
                phase_id,
                total_substeps=2,
                metadata={'datasets': 2}
            )
            progress_manager.start(step_id)
            
            # Process datasets
            for i in range(2):
                substep_id = f"{step_id}_dataset_{i}"
                progress_manager.create_substep(
                    substep_id,
                    step_id,
                    metadata={'dataset': f'test-{i}'}
                )
                progress_manager.start(substep_id)
                
                # Simulate processing
                for j in range(10):
                    progress_manager.update(substep_id, completed_units=j*10)
                    time.sleep(0.01)
                
                progress_manager.complete(substep_id)
                progress_manager.update(step_id, increment=1)
            
            progress_manager.complete(step_id)
            return []
        
        # Run with detailed progress
        with patch.object(pipeline, '_create_experiment', return_value='test-exp-123'):
            with patch.object(pipeline, '_run_resampling_phase', side_effect=detailed_resampling_phase):
                with patch.object(pipeline, '_run_merging_phase', return_value=None):
                    with patch.object(pipeline, '_run_analysis_phase', return_value=None):
                        with patch.object(pipeline, '_finalize_results', return_value={}):
                            pipeline.run_complete_pipeline('test_hierarchical')
        
        # Verify hierarchical structure
        pipeline_nodes = [n for n in all_nodes.values() if n['level'] == 'pipeline']
        phase_nodes = [n for n in all_nodes.values() if n['level'] == 'phase']
        step_nodes = [n for n in all_nodes.values() if n['level'] == 'step']
        substep_nodes = [n for n in all_nodes.values() if n['level'] == 'substep']
        
        assert len(pipeline_nodes) >= 1
        assert len(phase_nodes) >= 1
        assert len(step_nodes) >= 1
        assert len(substep_nodes) >= 2
        
        # Verify parent-child relationships
        for phase in phase_nodes:
            assert any(p['node_id'] == phase['parent_id'] for p in all_nodes.values())
        
        for step in step_nodes:
            assert any(p['node_id'] == step['parent_id'] for p in all_nodes.values())
    
    def test_memory_aware_checkpointing(self, test_config, mock_db):
        """Test checkpoint triggering based on memory usage."""
        pipeline = UnifiedResamplingPipeline(test_config, mock_db)
        memory_manager = get_memory_manager()
        checkpoint_manager = get_checkpoint_manager()
        
        # Track checkpoints
        memory_checkpoints = []
        
        original_save = checkpoint_manager.save_checkpoint
        def track_checkpoint(checkpoint_id, data, **kwargs):
            if 'memory' in checkpoint_id:
                memory_checkpoints.append(checkpoint_id)
            return original_save(checkpoint_id, data, **kwargs)
        
        checkpoint_manager.save_checkpoint = track_checkpoint
        
        # Simulate high memory pressure during processing
        def memory_intensive_phase(*args, **kwargs):
            # Simulate gradual memory increase
            for i in range(5):
                # Allocate some memory
                dummy_data = np.random.rand(1000, 1000)
                
                # Simulate high memory pressure
                with patch.object(memory_manager, 'get_memory_pressure_level') as mock_pressure:
                    mock_pressure.return_value.value = 'high' if i > 2 else 'normal'
                    
                    # This should trigger checkpoint on high pressure
                    if mock_pressure.return_value.value == 'high':
                        pipeline._save_checkpoint(f"memory_pressure_{i}")
                
                time.sleep(0.1)
            
            return []
        
        # Run pipeline with memory-intensive phase
        with patch.object(pipeline, '_create_experiment', return_value='test-exp-123'):
            with patch.object(pipeline, '_run_resampling_phase', side_effect=memory_intensive_phase):
                with patch.object(pipeline, '_run_merging_phase', return_value=None):
                    with patch.object(pipeline, '_run_analysis_phase', return_value=None):
                        with patch.object(pipeline, '_finalize_results', return_value={}):
                            pipeline.run_complete_pipeline('test_memory_checkpoint')
        
        # Verify memory-triggered checkpoints were created
        assert len(memory_checkpoints) > 0
        assert any('memory_pressure' in cp for cp in memory_checkpoints)


class TestProgressCallbacks:
    """Test progress callback functionality."""
    
    def test_multiple_callback_registration(self):
        """Test registering multiple callbacks for different levels."""
        progress_manager = get_progress_manager()
        
        # Track callbacks
        pipeline_events = []
        phase_events = []
        any_events = []
        
        # Register callbacks
        progress_manager.register_callback('pipeline', lambda n: pipeline_events.append(n.level))
        progress_manager.register_callback('phase', lambda n: phase_events.append(n.level))
        progress_manager.register_callback('any', lambda n: any_events.append(n.level))
        
        # Create nodes at different levels
        progress_manager.create_pipeline('test_pipeline', 2)
        progress_manager.create_phase('test_phase', 'test_pipeline', 10)
        progress_manager.create_step('test_step', 'test_phase', 5)
        
        # Update progress
        progress_manager.update('test_pipeline', completed_units=1)
        progress_manager.update('test_phase', completed_units=5)
        progress_manager.update('test_step', completed_units=2)
        
        # Verify callbacks were called appropriately
        assert 'pipeline' in pipeline_events
        assert 'phase' in phase_events
        assert all(level in any_events for level in ['pipeline', 'phase', 'step'])
    
    def test_callback_error_handling(self):
        """Test that callback errors don't break progress tracking."""
        progress_manager = get_progress_manager()
        
        # Register callback that raises error
        def bad_callback(node):
            raise ValueError("Callback error")
        
        progress_manager.register_callback('any', bad_callback)
        
        # Also register good callback
        good_events = []
        progress_manager.register_callback('any', lambda n: good_events.append(n.name))
        
        # Create and update node - should not raise
        progress_manager.create_pipeline('test_pipeline', 10)
        progress_manager.update('test_pipeline', completed_units=5)
        
        # Good callback should still work
        assert 'test_pipeline' in good_events


class TestCheckpointValidation:
    """Test checkpoint validation and corruption handling."""
    
    def test_checkpoint_validation(self, test_config):
        """Test checkpoint validation functionality."""
        checkpoint_manager = get_checkpoint_manager()
        
        # Save valid checkpoint
        test_data = {
            'experiment_id': 'test-123',
            'completed_phases': ['phase1', 'phase2'],
            'data': [1, 2, 3, 4, 5]
        }
        
        checkpoint_id = 'test_validation_checkpoint'
        saved_checkpoint = checkpoint_manager.save_checkpoint(
            checkpoint_id,
            test_data,
            level='pipeline'
        )
        
        # Validate checkpoint
        is_valid = checkpoint_manager.validate_checkpoint(checkpoint_id)
        assert is_valid
        
        # Get checkpoint info
        info = checkpoint_manager.get_checkpoint_info(checkpoint_id)
        assert info is not None
        assert info.status == 'valid'
        assert info.validation_checksum is not None
    
    def test_checkpoint_compression(self, test_config):
        """Test checkpoint compression functionality."""
        checkpoint_manager = get_checkpoint_manager()
        
        # Create large data for compression test
        large_data = {
            'array': np.random.rand(1000, 1000).tolist(),
            'metadata': {'size': 'large'}
        }
        
        # Save with compression
        checkpoint_id = 'test_compression'
        checkpoint = checkpoint_manager.save_checkpoint(
            checkpoint_id,
            large_data,
            level='pipeline',
            compress=True
        )
        
        # Load and verify
        loaded_data = checkpoint_manager.load_checkpoint(checkpoint_id)
        assert loaded_data['metadata']['size'] == 'large'
        assert len(loaded_data['array']) == 1000
        
        # Check file size is reasonable (compressed)
        info = checkpoint_manager.get_checkpoint_info(checkpoint_id)
        assert info.compression_type is not None
        # Compressed size should be less than uncompressed
        assert info.file_size_bytes < 1000 * 1000 * 8  # Less than raw array size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])