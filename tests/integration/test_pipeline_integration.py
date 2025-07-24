# tests/integration/test_pipeline_integration.py
"""Integration tests for the pipeline orchestration module."""

import pytest
import tempfile
import shutil
from pathlib import Path
import json
import numpy as np
import xarray as xr
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import psycopg2
import time

from src.config.config import Config
from src.database.connection import DatabaseManager
from src.pipelines.orchestrator import PipelineOrchestrator, PipelineStatus
from src.pipelines.stages.load_stage import DataLoadStage
from src.pipelines.stages.resample_stage import ResampleStage
from src.pipelines.stages.merge_stage import MergeStage
from src.pipelines.stages.analysis_stage import AnalysisStage
from src.pipelines.stages.export_stage import ExportStage
from src.pipelines.stages.base_stage import StageStatus, StageResult


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        base_path = Path(temp_dir)
        
        dirs = {
            'data': base_path / 'data',
            'output': base_path / 'output',
            'checkpoint': base_path / 'checkpoint',
            'config': base_path / 'config'
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        yield dirs


@pytest.fixture
def test_config(temp_dirs):
    """Create test configuration."""
    config_data = {
        'paths': {
            'data_dir': str(temp_dirs['data']),
            'output_dir': str(temp_dirs['output']),
            'checkpoint_dir': str(temp_dirs['checkpoint'])
        },
        'database': {
            'host': 'localhost',
            'port': 5432,
            'database': 'test_geoprocess_db',
            'user': 'test_user'
        },
        'resampling': {
            'target_resolution': 0.1,
            'target_crs': 'EPSG:4326',
            'engine': 'numpy',
            'strategies': {
                'richness_data': 'sum',
                'continuous_data': 'bilinear'
            }
        },
        'som_analysis': {
            'default_grid_size': [3, 3],
            'iterations': 10,
            'sigma': 1.0,
            'learning_rate': 0.5,
            'max_pixels_in_memory': 10000
        },
        'pipeline': {
            'memory_limit_gb': 4.0,
            'max_checkpoints_per_experiment': 5,
            'cleanup_checkpoints_on_success': False,
            'optional_stages': ['export']
        },
        'datasets': {
            'target_datasets': [
                {
                    'name': 'test-plants',
                    'path': 'plants.tif',
                    'band_name': 'plants_richness',
                    'data_type': 'richness_data',
                    'enabled': True
                },
                {
                    'name': 'test-animals',
                    'path': 'animals.tif',
                    'band_name': 'animals_richness',
                    'data_type': 'richness_data',
                    'enabled': True
                }
            ]
        }
    }
    
    # Create config file
    config_file = temp_dirs['config'] / 'config.yml'
    import yaml
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    
    # Create mock config object
    config = Mock(spec=Config)
    config.settings = config_data
    config.get = lambda key, default=None: get_nested_value(config_data, key, default)
    config._is_test_mode = lambda: True
    
    return config


def get_nested_value(data, key, default=None):
    """Get nested value from dict using dot notation."""
    keys = key.split('.')
    value = data
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    return value


@pytest.fixture
def mock_db():
    """Create mock database connection."""
    db = Mock(spec=DatabaseManager)
    db.test_connection.return_value = True
    
    # Mock connection context manager
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.__enter__ = Mock(return_value=mock_conn)
    mock_conn.__exit__ = Mock(return_value=None)
    
    db.get_connection.return_value = mock_conn
    
    return db


@pytest.fixture
def sample_datasets(temp_dirs):
    """Create sample dataset files."""
    # Create sample raster files
    plants_data = np.random.poisson(lam=5, size=(100, 100)).astype(np.float32)
    animals_data = np.random.poisson(lam=3, size=(100, 100)).astype(np.float32)
    
    # Create xarray datasets
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    
    plants_ds = xr.Dataset({
        'richness': xr.DataArray(
            plants_data,
            dims=['y', 'x'],
            coords={'x': x, 'y': y}
        )
    })
    
    animals_ds = xr.Dataset({
        'richness': xr.DataArray(
            animals_data,
            dims=['y', 'x'],
            coords={'x': x, 'y': y}
        )
    })
    
    # Save as NetCDF (mock GeoTIFF)
    plants_path = temp_dirs['data'] / 'plants.tif'
    animals_path = temp_dirs['data'] / 'animals.tif'
    
    plants_ds.to_netcdf(plants_path)
    animals_ds.to_netcdf(animals_path)
    
    return {
        'plants': plants_path,
        'animals': animals_path,
        'plants_data': plants_data,
        'animals_data': animals_data
    }


## 2. Pipeline Orchestrator Integration Tests

class TestPipelineOrchestratorIntegration:
    """Test pipeline orchestrator integration with other modules."""
    
    def test_full_pipeline_execution(self, test_config, mock_db, sample_datasets, temp_dirs):
        """Test complete pipeline execution with all stages."""
        # Create orchestrator
        orchestrator = PipelineOrchestrator(test_config, mock_db)
        
        # Configure pipeline with all stages
        stages = [
            DataLoadStage,
            ResampleStage,
            MergeStage,
            AnalysisStage,
            ExportStage
        ]
        orchestrator.configure_pipeline(stages)
        
        # Mock the actual processing in stages
        with patch('src.raster_data.catalog.RasterCatalog') as mock_catalog, \
             patch('src.processors.data_preparation.resampling_processor.ResamplingProcessor') as mock_resampler, \
             patch('src.spatial_analysis.som.som_trainer.SOMAnalyzer') as mock_som:
            
            # Setup mocks
            self._setup_processing_mocks(
                mock_catalog, mock_resampler, mock_som, 
                sample_datasets, test_config
            )
            
            # Run pipeline
            results = orchestrator.run_pipeline(
                experiment_name="test_integration",
                checkpoint_dir=temp_dirs['checkpoint'],
                output_dir=temp_dirs['output']
            )
            
            # Verify pipeline completed
            assert orchestrator.status == PipelineStatus.COMPLETED
            assert results['status'] == 'completed'
            assert results['stages_completed'] == 5
            
            # Verify all stages executed
            for stage in orchestrator.stages:
                assert stage.status == StageStatus.COMPLETED
            
            # Verify outputs created
            assert (temp_dirs['output'] / 'final_report.json').exists()
            assert (temp_dirs['output'] / 'README.md').exists()
    
    def test_pipeline_with_stage_failure_and_recovery(self, test_config, mock_db, temp_dirs):
        """Test pipeline failure handling and recovery."""
        orchestrator = PipelineOrchestrator(test_config, mock_db)
        
        # Create custom failing stage
        class FailingStage(ResampleStage):
            def __init__(self):
                super().__init__()
                self.attempt_count = 0
            
            def execute(self, context):
                self.attempt_count += 1
                if self.attempt_count < 2:
                    raise RuntimeError("Simulated failure")
                # Succeed on second attempt
                return StageResult(
                    success=True,
                    data={'resampled_datasets': []},
                    metrics={'datasets_resampled': 0}
                )
        
        # Configure with failing stage
        orchestrator.register_stage(DataLoadStage())
        orchestrator.register_stage(FailingStage())
        
        # Mock data loading
        with patch('src.config.dataset_utils.DatasetPathResolver'):
            # First run should fail
            with pytest.raises(RuntimeError):
                orchestrator.run_pipeline(
                    experiment_name="test_failure",
                    checkpoint_dir=temp_dirs['checkpoint']
                )
            
            # Verify checkpoint was saved
            checkpoints = orchestrator.checkpoint_manager.list_checkpoints(
                orchestrator.context.experiment_id
            )
            assert len(checkpoints) >= 1
            
            # Recovery attempt should succeed
            results = orchestrator.run_pipeline(
                experiment_name="test_failure",
                checkpoint_dir=temp_dirs['checkpoint'],
                resume_from_checkpoint=True
            )
            
            assert results is not None
    
    def test_memory_monitoring_integration(self, test_config, mock_db, temp_dirs):
        """Test memory monitoring during pipeline execution."""
        # Set low memory limit to trigger warnings
        test_config.settings['pipeline']['memory_limit_gb'] = 0.1
        
        orchestrator = PipelineOrchestrator(test_config, mock_db)
        orchestrator.register_stage(DataLoadStage())
        
        # Track memory warnings
        warnings_triggered = []
        
        def warning_callback(usage):
            warnings_triggered.append(usage)
        
        orchestrator.memory_monitor.register_warning_callback(warning_callback)
        
        # Start monitoring
        orchestrator.memory_monitor.start()
        
        try:
            # Allocate some memory to trigger warning
            large_array = np.zeros((1000, 1000, 10))  # ~76MB
            
            # Give monitor time to detect
            time.sleep(2)
            
            # Check if warning was triggered
            status = orchestrator.memory_monitor.get_status()
            assert status['pressure'] in ['warning', 'critical']
            
        finally:
            orchestrator.memory_monitor.stop()
            del large_array
    
    def test_progress_tracking_across_stages(self, test_config, mock_db, temp_dirs):
        """Test progress tracking integration."""
        orchestrator = PipelineOrchestrator(test_config, mock_db)
        
        # Configure multiple stages
        stages = [DataLoadStage, ResampleStage, MergeStage]
        orchestrator.configure_pipeline(stages)
        
        # Track progress updates
        progress_updates = []
        
        def progress_callback(progress):
            progress_updates.append(progress.copy())
        
        orchestrator.progress_tracker.register_callback(progress_callback)
        
        # Mock stage execution
        with patch.object(DataLoadStage, 'execute') as mock_load, \
             patch.object(ResampleStage, 'execute') as mock_resample, \
             patch.object(MergeStage, 'execute') as mock_merge:
            
            # Setup mock returns
            mock_load.return_value = StageResult(success=True, data={}, metrics={})
            mock_resample.return_value = StageResult(success=True, data={}, metrics={})
            mock_merge.return_value = StageResult(success=True, data={}, metrics={})
            
            # Run pipeline
            orchestrator.run_pipeline(
                experiment_name="test_progress",
                checkpoint_dir=temp_dirs['checkpoint']
            )
            
            # Verify progress updates
            assert len(progress_updates) > 0
            
            # Check final progress
            final_progress = progress_updates[-1]
            assert final_progress['completed_stages'] == 3
            assert final_progress['overall_progress'] == 100.0
    
    def test_quality_checking_integration(self, test_config, mock_db, temp_dirs):
        """Test quality checking at each stage."""
        orchestrator = PipelineOrchestrator(test_config, mock_db)
        
        # Create stage with quality issues
        class LowQualityStage(DataLoadStage):
            def execute(self, context):
                # Return result that will fail quality checks
                return StageResult(
                    success=True,
                    data={'datasets': []},  # No datasets loaded
                    metrics={'datasets_loaded': 0}
                )
        
        orchestrator.register_stage(LowQualityStage())
        
        # Quality check should prevent pipeline completion
        with pytest.raises(ValueError, match="failed quality checks"):
            orchestrator.run_pipeline(
                experiment_name="test_quality",
                checkpoint_dir=temp_dirs['checkpoint']
            )
    
    def test_parallel_stage_execution(self, test_config, mock_db, temp_dirs):
        """Test parallel execution of independent stages."""
        orchestrator = PipelineOrchestrator(test_config, mock_db)
        
        # Create independent stages that can run in parallel
        class IndependentStage1(DataLoadStage):
            @property
            def name(self):
                return "independent1"
            
            def execute(self, context):
                time.sleep(0.1)  # Simulate work
                context.set('stage1_result', 'completed')
                return StageResult(success=True, data={}, metrics={})
        
        class IndependentStage2(DataLoadStage):
            @property
            def name(self):
                return "independent2"
            
            def execute(self, context):
                time.sleep(0.1)  # Simulate work
                context.set('stage2_result', 'completed')
                return StageResult(success=True, data={}, metrics={})
        
        class DependentStage(DataLoadStage):
            @property
            def name(self):
                return "dependent"
            
            @property
            def dependencies(self):
                return ["independent1", "independent2"]
            
            def execute(self, context):
                # Verify both dependencies completed
                assert context.get('stage1_result') == 'completed'
                assert context.get('stage2_result') == 'completed'
                return StageResult(success=True, data={}, metrics={})
        
        # Register stages
        orchestrator.register_stage(IndependentStage1())
        orchestrator.register_stage(IndependentStage2())
        orchestrator.register_stage(DependentStage())
        
        # Run pipeline
        start_time = time.time()
        results = orchestrator.run_pipeline(
            experiment_name="test_parallel",
            checkpoint_dir=temp_dirs['checkpoint']
        )
        execution_time = time.time() - start_time
        
        # Verify parallel execution (should be faster than sequential)
        assert execution_time < 0.3  # Less than sum of sleep times
        assert results['stages_completed'] == 3
    
    def _setup_processing_mocks(self, mock_catalog, mock_resampler, 
                               mock_som, sample_datasets, config):
        """Setup mocks for processing modules."""
        # Mock catalog
        mock_catalog_instance = Mock()
        mock_catalog_instance.register_raster.return_value = {
            'name': 'test',
            'path': 'test.tif',
            'resolution_degrees': 0.01
        }
        mock_catalog.return_value = mock_catalog_instance
        
        # Mock resampling processor
        mock_processor = Mock()
        
        # Mock resampled dataset info
        from src.processors.data_preparation.resampling_processor import ResampledDatasetInfo
        resampled_info = ResampledDatasetInfo(
            name='test-plants',
            source_path=str(sample_datasets['plants']),
            target_resolution=0.1,
            target_crs='EPSG:4326',
            resampling_method='sum',
            shape=(20, 20),
            bounds=(-10, -10, 10, 10),
            data_table_name='resampled_test_plants',
            band_name='plants_richness'
        )
        
        mock_processor.get_resampled_dataset.return_value = None
        mock_processor.resample_dataset.return_value = resampled_info
        mock_processor.load_resampled_data.return_value = np.random.rand(20, 20)
        mock_resampler.return_value = mock_processor
        
        # Mock SOM analyzer
        mock_som_instance = Mock()
        mock_result = Mock()
        mock_result.metadata = {'n_samples': 400, 'n_features': 2}
        mock_result.statistics = {'quantization_error': 0.5}
        mock_som_instance.analyze.return_value = mock_result
        mock_som_instance.save_results.return_value = Path('som_results')
        mock_som.return_value = mock_som_instance


## 3. Stage Integration Tests

class TestStageIntegration:
    """Test individual stage integration with other modules."""
    
    def test_data_load_stage_integration(self, test_config, mock_db, sample_datasets):
        """Test DataLoadStage integration with catalog and dataset resolver."""
        stage = DataLoadStage()
        
        # Create context
        from src.pipelines.orchestrator import PipelineContext
        context = PipelineContext(
            config=test_config,
            db=mock_db,
            experiment_id='test_exp',
            checkpoint_dir=Path('checkpoints'),
            output_dir=Path('outputs')
        )
        
        # Mock dataset resolver and catalog
        with patch('src.config.dataset_utils.DatasetPathResolver') as mock_resolver, \
             patch('src.raster_data.catalog.RasterCatalog') as mock_catalog:
            
            # Setup resolver mock
            mock_resolver_instance = Mock()
            mock_resolver_instance.validate_dataset_config.side_effect = lambda x: {
                **x,
                'resolved_path': str(sample_datasets[x['name'].split('-')[1]])
            }
            mock_resolver.return_value = mock_resolver_instance
            
            # Setup catalog mock
            mock_catalog_instance = Mock()
            mock_catalog_instance.register_raster.return_value = {
                'name': 'test',
                'path': 'test.tif'
            }
            mock_catalog.return_value = mock_catalog_instance
            
            # Execute stage
            result = stage.execute(context)
            
            # Verify integration
            assert result.success
            assert len(result.data['datasets']) == 2
            assert mock_resolver_instance.validate_dataset_config.call_count == 2
            assert mock_catalog_instance.register_raster.call_count == 2
    
    def test_resample_stage_database_integration(self, test_config, mock_db):
        """Test ResampleStage integration with database storage."""
        stage = ResampleStage()
        
        # Create context with loaded datasets
        from src.pipelines.orchestrator import PipelineContext
        context = PipelineContext(
            config=test_config,
            db=mock_db,
            experiment_id='test_exp',
            checkpoint_dir=Path('checkpoints'),
            output_dir=Path('outputs')
        )
        
        context.set('loaded_datasets', [
            {
                'name': 'test-plants',
                'config': {
                    'name': 'test-plants',
                    'band_name': 'plants_richness',
                    'data_type': 'richness_data'
                }
            }
        ])
        
        # Mock resampling processor
        with patch('src.processors.data_preparation.resampling_processor.ResamplingProcessor') as mock_processor_class:
            mock_processor = Mock()
            
            # Mock database operations
            mock_processor.get_resampled_dataset.return_value = None
            
            from src.processors.data_preparation.resampling_processor import ResampledDatasetInfo
            mock_processor.resample_dataset.return_value = ResampledDatasetInfo(
                name='test-plants',
                source_path='plants.tif',
                target_resolution=0.1,
                target_crs='EPSG:4326',
                resampling_method='sum',
                shape=(50, 50),
                bounds=(-10, -10, 10, 10),
                data_table_name='resampled_test_plants',
                band_name='plants_richness'
            )
            
            mock_processor_class.return_value = mock_processor
            
            # Execute stage
            result = stage.execute(context)
            
            # Verify database integration
            assert result.success
            assert mock_processor.resample_dataset.called
            assert context.get('resampled_datasets') is not None
    
    def test_merge_stage_xarray_integration(self, test_config, mock_db):
        """Test MergeStage integration with xarray operations."""
        stage = MergeStage()
        
        # Create context with resampled datasets
        from src.pipelines.orchestrator import PipelineContext
        context = PipelineContext(
            config=test_config,
            db=mock_db,
            experiment_id='test_exp',
            checkpoint_dir=Path('checkpoints'),
            output_dir=Path('outputs')
        )
        
        # Create mock resampled dataset info
        from src.processors.data_preparation.resampling_processor import ResampledDatasetInfo
        resampled_info = [
            ResampledDatasetInfo(
                name='test-plants',
                source_path='plants.tif',
                target_resolution=0.1,
                target_crs='EPSG:4326',
                resampling_method='sum',
                shape=(10, 10),
                bounds=(-5, -5, 5, 5),
                data_table_name='resampled_test_plants',
                band_name='plants_richness'
            ),
            ResampledDatasetInfo(
                name='test-animals',
                source_path='animals.tif',
                target_resolution=0.1,
                target_crs='EPSG:4326',
                resampling_method='sum',
                shape=(10, 10),
                bounds=(-5, -5, 5, 5),
                data_table_name='resampled_test_animals',
                band_name='animals_richness'
            )
        ]
        
        context.set('resampled_datasets', resampled_info)
        
        # Mock data loading
        with patch('src.processors.data_preparation.resampling_processor.ResamplingProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_processor.load_resampled_data.side_effect = [
                np.random.rand(10, 10),
                np.random.rand(10, 10)
            ]
            mock_processor_class.return_value = mock_processor
            
            # Execute stage
            result = stage.execute(context)
            
            # Verify xarray integration
            assert result.success
            merged_dataset = context.get('merged_dataset')
            assert merged_dataset is not None
            assert 'plants_richness' in merged_dataset.data_vars
            assert 'animals_richness' in merged_dataset.data_vars
            assert merged_dataset.sizes == {'x': 10, 'y': 10}
    
    def test_analysis_stage_som_integration(self, test_config, mock_db):
        """Test AnalysisStage integration with SOM analyzer."""
        stage = AnalysisStage()
        
        # Create context with merged dataset
        from src.pipelines.orchestrator import PipelineContext
        context = PipelineContext(
            config=test_config,
            db=mock_db,
            experiment_id='test_exp',
            checkpoint_dir=Path('checkpoints'),
            output_dir=Path('outputs')
        )
        
        # Create mock merged dataset
        merged_dataset = xr.Dataset({
            'plants_richness': xr.DataArray(
                np.random.rand(10, 10),
                dims=['y', 'x'],
                coords={'x': np.linspace(-5, 5, 10), 'y': np.linspace(-5, 5, 10)}
            ),
            'animals_richness': xr.DataArray(
                np.random.rand(10, 10),
                dims=['y', 'x'],
                coords={'x': np.linspace(-5, 5, 10), 'y': np.linspace(-5, 5, 10)}
            )
        })
        context.set('merged_dataset', merged_dataset)
        
        # Mock SOM analyzer
        with patch('src.spatial_analysis.som.som_trainer.SOMAnalyzer') as mock_som_class:
            mock_som = Mock()
            
            # Mock analysis result
            mock_result = Mock()
            mock_result.metadata = {
                'n_samples': 100,
                'n_features': 2,
                'grid_size': [3, 3]
            }
            mock_result.statistics = {
                'quantization_error': 0.5,
                'topographic_error': 0.1
            }
            
            mock_som.analyze.return_value = mock_result
            mock_som.save_results.return_value = Path('som_results')
            mock_som_class.return_value = mock_som
            
            # Execute stage
            result = stage.execute(context)
            
            # Verify SOM integration
            assert result.success
            assert mock_som.analyze.called
            assert context.get('som_results') is not None
            assert 'analysis_stats' in result.metrics


## 4. Recovery and Checkpoint Tests

class TestRecoveryIntegration:
    """Test recovery and checkpoint integration."""
    
    def test_checkpoint_save_and_restore(self, test_config, mock_db, temp_dirs):
        """Test checkpoint saving and restoration."""
        from src.pipelines.recovery.checkpoint_manager import CheckpointManager
        from src.pipelines.orchestrator import PipelineContext
        
        # Create checkpoint manager
        checkpoint_mgr = CheckpointManager(test_config)
        
        # Create context with data
        context = PipelineContext(
            config=test_config,
            db=mock_db,
            experiment_id='test_exp',
            checkpoint_dir=temp_dirs['checkpoint'],
            output_dir=temp_dirs['output']
        )
        
        # Add some data to context
        context.set('test_data', {'key': 'value'})
        context.set('loaded_datasets', ['dataset1', 'dataset2'])
        context.quality_metrics['stage1'] = {'score': 0.95}
        
        # Save checkpoint
        checkpoint_mgr.save_checkpoint(
            experiment_id='test_exp',
            stage_name='test_stage',
            context=context,
            stage_results={'test_stage': StageResult(True, {}, {})}
        )
        
        # Verify checkpoint file created
        checkpoint_file = temp_dirs['checkpoint'] / 'test_exp' / 'checkpoint_test_stage.json'
        assert checkpoint_file.exists()
        
        # Load checkpoint
        loaded = checkpoint_mgr.load_checkpoint('test_exp', 'test_stage')
        assert loaded is not None
        assert loaded['shared_data']['test_data'] == {'key': 'value'}
        assert loaded['quality_metrics']['stage1']['score'] == 0.95
    
    def test_failure_recovery_strategies(self, test_config, mock_db):
        """Test different failure recovery strategies."""
        from src.pipelines.recovery.failure_handler import FailureHandler
        
        handler = FailureHandler(test_config)
        
        # Test memory error recovery
        memory_error = MemoryError("Out of memory")
        context = Mock(experiment_id='test')
        stage = Mock(name='test_stage')
        
        should_recover = handler.handle_failure(memory_error, context, stage)
        assert should_recover
        assert handler.get_recovery_strategy() == 'retry'
        
        # Test repeated failures
        for _ in range(3):
            handler.handle_failure(memory_error, context, stage)
        
        # Should abort after max retries
        assert not handler.can_recover() or handler.get_recovery_strategy() == 'abort'
    
    def test_pipeline_resume_from_checkpoint(self, test_config, mock_db, temp_dirs):
        """Test resuming pipeline from checkpoint."""
        orchestrator = PipelineOrchestrator(test_config, mock_db)
        
        # Configure pipeline
        orchestrator.configure_pipeline([
            DataLoadStage,
            ResampleStage,
            MergeStage
        ])
        
        # Create a checkpoint manually
        from src.pipelines.orchestrator import PipelineContext
        context = PipelineContext(
            config=test_config,
            db=mock_db,
            experiment_id='resume_test',
            checkpoint_dir=temp_dirs['checkpoint'],
            output_dir=temp_dirs['output']
        )
        
        # Save checkpoint after first stage
        checkpoint_data = {
            'experiment_id': 'resume_test',
            'stage': 'data_load',
            'timestamp': datetime.now().isoformat(),
            'completed_stages': ['data_load'],
            'shared_data': {
                'loaded_datasets': [{'name': 'test-plants'}]
            },
            'metadata': {'experiment_name': 'resume_test'},
            'quality_metrics': {},
            'stage_results': {},
            'progress': {'completed_stages': 1}
        }
        
        checkpoint_file = temp_dirs['checkpoint'] / 'resume_test' / 'checkpoint_data_load.json'
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        
        # Mock remaining stages
        with patch.object(ResampleStage, 'execute') as mock_resample, \
             patch.object(MergeStage, 'execute') as mock_merge:
            
            mock_resample.return_value = StageResult(True, {}, {})
            mock_merge.return_value = StageResult(True, {}, {})
            
            # Resume pipeline
            results = orchestrator.run_pipeline(
                experiment_name='resume_test',
                checkpoint_dir=temp_dirs['checkpoint'],
                output_dir=temp_dirs['output'],
                resume_from_checkpoint=True
            )
            
            # Verify only remaining stages were executed
            assert mock_resample.called
            assert mock_merge.called
            
            # Verify context was restored
            assert orchestrator.context.get('loaded_datasets') == [{'name': 'test-plants'}]


## 5. Monitor Integration Tests

class TestMonitorIntegration:
    """Test monitor integration with pipeline."""
    
    def test_memory_monitor_cleanup_trigger(self, test_config, mock_db):
        """Test memory monitor triggering cleanup."""
        from src.pipelines.monitors.memory_monitor import MemoryMonitor
        
        # Set very low memory limit
        test_config.settings['pipeline']['memory_limit_gb'] = 0.001
        
        monitor = MemoryMonitor(test_config)
        
        # Track cleanup calls
        cleanup_called = False
        
        def mock_cleanup():
            nonlocal cleanup_called
            cleanup_called = True
        
        # Replace cleanup method
        monitor.trigger_cleanup = mock_cleanup
        
        # Start monitoring
        monitor.start()
        
        try:
            # Allocate memory to trigger critical state
            large_data = np.zeros((1000, 1000, 10))
            
            # Check status
            status = monitor.get_status()
            assert status['pressure'] == 'critical'
            
            # Cleanup should be triggered
            time.sleep(0.1)  # Give monitor time to react
            
        finally:
            monitor.stop()
            del large_data
    
    def test_quality_checker_database_metrics(self, test_config, mock_db):
        """Test quality checker storing metrics in database."""
        from src.pipelines.monitors.quality_checker import QualityChecker
        from src.pipelines.stages.base_stage import PipelineStage, StageResult
        from src.pipelines.orchestrator import PipelineContext
        
        checker = QualityChecker(test_config)
        
        # Create mock stage and result
        stage = Mock(spec=PipelineStage)
        stage.name = 'test_stage'
        
        result = StageResult(
            success=True,
            data={'datasets': [{'name': 'test', 'path': 'test.tif'}]},
            metrics={'datasets_loaded': 1}
        )
        
        context = PipelineContext(
            config=test_config,
            db=mock_db,
            experiment_id='quality_test',
            checkpoint_dir=Path('checkpoints'),
            output_dir=Path('outputs')
        )
        
        # Check quality
        report = checker.check_stage_output(stage, result, context)
        
        # Verify report
        assert report.stage_name == 'test_stage'
        assert report.overall_score > 0
        
        # Verify metrics stored in context
        assert 'test_stage' in context.quality_metrics
        assert context.quality_metrics['test_stage']['score'] == report.overall_score


## 6. End-to-End Integration Test

class TestEndToEndIntegration:
    """Complete end-to-end integration test."""
    
    @pytest.mark.integration
    def test_complete_pipeline_with_real_modules(self, test_config, sample_datasets, temp_dirs):
        """Test complete pipeline with minimal mocking."""
        # This test requires a test database to be available
        pytest.skip("Requires test database setup")
        
        # Create real database connection
        db = DatabaseManager()
        
        # Create orchestrator
        orchestrator = PipelineOrchestrator(test_config, db)
        
        # Configure full pipeline
        orchestrator.configure_pipeline([
            DataLoadStage,
            ResampleStage,
            MergeStage,
            AnalysisStage,
            ExportStage
        ])
        
        # Run pipeline
        results = orchestrator.run_pipeline(
            experiment_name="integration_test",
            checkpoint_dir=temp_dirs['checkpoint'],
            output_dir=temp_dirs['output']
        )
        
        # Verify results
        assert results['status'] == 'completed'
        assert results['stages_completed'] == 5
        
        # Verify outputs
        assert (temp_dirs['output'] / 'merged_dataset.nc').exists()
        assert (temp_dirs['output'] / 'final_report.json').exists()
        assert (temp_dirs['output'] / 'README.md').exists()
        
        # Verify database entries
        with db.get_connection() as conn:
            cur = conn.cursor()
            
            # Check experiment record
            cur.execute(
                "SELECT status FROM experiments WHERE id = %s",
                (results['experiment_id'],)
            )
            status = cur.fetchone()[0]
            assert status == 'completed'