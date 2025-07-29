"""Test validation monitoring and reporting in pipeline orchestrator."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile

from src.pipelines.orchestrator import PipelineOrchestrator, PipelineStatus
from src.pipelines.stages.base_stage import StageResult, StageStatus
from src.abstractions.interfaces.validator import ValidationResult, ValidationIssue, ValidationSeverity


class MockStage:
    """Mock pipeline stage for testing."""
    
    def __init__(self, name, validation_metrics=None):
        self.name = name
        self.status = StageStatus.PENDING
        self.dependencies = []
        self.memory_requirements = "1GB"
        self.disk_requirements = "10GB"
        self.supports_chunking = False
        self.validation_metrics = validation_metrics or {}
        
    def validate(self):
        """Mock validation."""
        return True, []
    
    def execute(self, context):
        """Mock execution with validation results."""
        self.status = StageStatus.RUNNING
        
        # Create mock result with validation metrics
        result = StageResult(
            success=True,
            data={'test': 'data'},
            metrics={
                'validation_checks': self.validation_metrics.get('checks', 5),
                'validation_errors': self.validation_metrics.get('errors', 0),
                'validation_warnings': self.validation_metrics.get('warnings', 1),
                'validation_failures': self.validation_metrics.get('failures', 0)
            }
        )
        
        self.status = StageStatus.COMPLETED
        return result
    
    def set_processing_config(self, config):
        """Mock method."""
        pass


class TestOrchestratorValidationMonitoring:
    """Test validation monitoring in pipeline orchestrator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock()
        self.db = Mock()
        
        # Mock config values
        self.config.get.return_value = {}
        
        # Create orchestrator with mocked components
        with patch('src.pipelines.orchestrator.MemoryMonitor'):
            with patch('src.pipelines.orchestrator.ProgressTracker'):
                with patch('src.pipelines.orchestrator.QualityChecker'):
                    with patch('src.pipelines.orchestrator.get_checkpoint_manager'):
                        with patch('src.pipelines.orchestrator.FailureHandler'):
                            self.orchestrator = PipelineOrchestrator(
                                self.config, self.db
                            )
    
    def test_validation_tracking_initialization(self):
        """Test validation tracking is properly initialized."""
        assert hasattr(self.orchestrator, 'validation_results')
        assert hasattr(self.orchestrator, 'validation_metrics')
        assert isinstance(self.orchestrator.validation_results, dict)
        assert isinstance(self.orchestrator.validation_metrics, dict)
    
    def test_stage_validation_tracking(self):
        """Test validation results are tracked for each stage."""
        # Create stage with validation results
        stage = MockStage('test_stage', {
            'checks': 10,
            'errors': 2,
            'warnings': 3,
            'failures': 1
        })
        
        # Add stage to orchestrator
        self.orchestrator.add_stage(stage)
        
        # Execute stage
        result = stage.execute(None)
        
        # Track validation results
        self.orchestrator._track_validation_results(stage, result)
        
        # Check tracking
        assert 'test_stage' in self.orchestrator.validation_metrics
        metrics = self.orchestrator.validation_metrics['test_stage']
        assert metrics['total_checks'] == 10
        assert metrics['total_errors'] == 2
        assert metrics['total_warnings'] == 3
        assert metrics['failed_checks'] == 1
    
    def test_validation_summary_generation(self):
        """Test generation of validation summary across stages."""
        # Create multiple stages with different validation results
        stages = [
            MockStage('stage1', {'checks': 5, 'errors': 0, 'warnings': 1, 'failures': 0}),
            MockStage('stage2', {'checks': 8, 'errors': 2, 'warnings': 2, 'failures': 1}),
            MockStage('stage3', {'checks': 3, 'errors': 1, 'warnings': 0, 'failures': 1})
        ]
        
        for stage in stages:
            self.orchestrator.add_stage(stage)
            result = stage.execute(None)
            self.orchestrator._track_validation_results(stage, result)
        
        # Get summary
        summary = self.orchestrator.get_validation_summary()
        
        # Check overall metrics
        assert summary['overall_metrics']['total_checks'] == 16  # 5+8+3
        assert summary['overall_metrics']['total_errors'] == 3   # 0+2+1
        assert summary['overall_metrics']['total_warnings'] == 3  # 1+2+0
        assert summary['overall_metrics']['failed_checks'] == 2   # 0+1+1
        
        # Check success rate
        expected_success_rate = (14/16) * 100  # 14 successful out of 16
        assert abs(summary['success_rate_percent'] - expected_success_rate) < 0.1
        
        # Check flags
        assert summary['has_errors'] == True
        assert summary['has_warnings'] == True
        assert summary['all_stages_passed'] == False
    
    def test_validation_results_from_context(self):
        """Test extraction of validation results from pipeline context."""
        # Create stage and context
        stage = MockStage('context_stage')
        
        # Create mock context with validation results
        mock_context = Mock()
        mock_context.shared_data = {
            'context_stage_validation_results': [
                {
                    'stage': 'context_stage',
                    'result': Mock(
                        is_valid=True,
                        error_count=0,
                        warning_count=2
                    )
                }
            ]
        }
        
        # Set up orchestrator context
        self.orchestrator.context = mock_context
        
        # Execute stage
        result = StageResult(success=True, data={}, metrics={})
        
        # Track validation
        self.orchestrator._track_validation_results(stage, result)
        
        # Check results were extracted from context
        assert 'context_stage' in self.orchestrator.validation_results
        assert len(self.orchestrator.validation_results['context_stage']) == 1
    
    def test_validation_summary_reporting(self):
        """Test final validation summary reporting."""
        # Set up validation metrics
        self.orchestrator.validation_metrics = {
            'resample': {'total_checks': 10, 'total_errors': 0, 'total_warnings': 2, 'failed_checks': 0},
            'merge': {'total_checks': 5, 'total_errors': 1, 'total_warnings': 1, 'failed_checks': 1}
        }
        
        # Capture log output
        with patch('src.pipelines.orchestrator.logger') as mock_logger:
            self.orchestrator._report_final_validation_summary()
            
            # Check summary was logged
            log_calls = [str(call) for call in mock_logger.info.call_args_list]
            
            assert any("PIPELINE VALIDATION SUMMARY" in call for call in log_calls)
            assert any("Total validation checks: 15" in call for call in log_calls)
            assert any("Failed checks: 1" in call for call in log_calls)
            assert any("Total errors: 1" in call for call in log_calls)
            assert any("Total warnings: 3" in call for call in log_calls)
            
            # Check warning about failed checks
            warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
            assert any("validation checks failed" in call for call in warning_calls)
    
    def test_validation_in_full_pipeline_execution(self):
        """Test validation tracking in full pipeline execution."""
        # Create test stages
        stages = [
            MockStage('load', {'checks': 3, 'errors': 0, 'warnings': 0, 'failures': 0}),
            MockStage('resample', {'checks': 5, 'errors': 0, 'warnings': 1, 'failures': 0}),
            MockStage('merge', {'checks': 4, 'errors': 0, 'warnings': 2, 'failures': 0})
        ]
        
        # Set up dependencies
        stages[1].dependencies = ['load']
        stages[2].dependencies = ['resample']
        
        for stage in stages:
            self.orchestrator.add_stage(stage)
        
        # Mock database schema
        with patch('src.database.schema.schema'):
            # Create pipeline context
            with tempfile.TemporaryDirectory() as temp_dir:
                self.orchestrator.setup_pipeline(
                    experiment_name="test_validation",
                    output_dir=Path(temp_dir)
                )
                
                # Mock checkpoint manager
                self.orchestrator.checkpoint_manager.save = Mock(return_value="checkpoint_123")
                
                # Mock signal handler
                self.orchestrator.signal_handler.is_pause_requested = Mock(return_value=False)
                self.orchestrator._pause_requested = False
                self.orchestrator._stop_requested = False
                
                # Execute pipeline stages
                completed_stages = set()
                for stage in stages:
                    if stage.name not in completed_stages:
                        # Execute with proper context
                        result = self.orchestrator._execute_stage(stage, completed_stages)
                        completed_stages.add(stage.name)
                
                # Get final validation summary
                summary = self.orchestrator.get_validation_summary()
                
                # Check all stages were tracked
                assert len(self.orchestrator.validation_metrics) == 3
                assert summary['overall_metrics']['total_checks'] == 12  # 3+5+4
                assert summary['overall_metrics']['total_warnings'] == 3  # 0+1+2
                assert summary['all_stages_passed'] == True
    
    def test_validation_context_storage(self):
        """Test validation summary is stored in context."""
        # Set up mock context
        mock_context = Mock()
        mock_context.set = Mock()
        self.orchestrator.context = mock_context
        
        # Set up some validation metrics
        self.orchestrator.validation_metrics = {
            'test': {'total_checks': 5, 'total_errors': 0, 'total_warnings': 1, 'failed_checks': 0}
        }
        
        # Report summary
        self.orchestrator._report_final_validation_summary()
        
        # Check summary was stored in context
        mock_context.set.assert_called()
        call_args = mock_context.set.call_args
        assert call_args[0][0] == 'validation_summary'
        
        summary = call_args[0][1]
        assert 'overall_metrics' in summary
        assert 'success_rate_percent' in summary
        assert 'stage_breakdown' in summary


class TestValidationErrorHandling:
    """Test error handling in validation monitoring."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = Mock()
        self.db = Mock()
        self.config.get.return_value = {}
        
        with patch('src.pipelines.orchestrator.MemoryMonitor'):
            with patch('src.pipelines.orchestrator.ProgressTracker'):
                with patch('src.pipelines.orchestrator.QualityChecker'):
                    with patch('src.pipelines.orchestrator.get_checkpoint_manager'):
                        with patch('src.pipelines.orchestrator.FailureHandler'):
                            self.orchestrator = PipelineOrchestrator(
                                self.config, self.db
                            )
    
    def test_missing_validation_data_handling(self):
        """Test handling when validation data is missing."""
        stage = MockStage('no_validation')
        
        # Result with no validation metrics
        result = StageResult(success=True, data={}, metrics={})
        
        # Should handle gracefully
        self.orchestrator._track_validation_results(stage, result)
        
        # Check empty metrics initialized
        assert 'no_validation' in self.orchestrator.validation_metrics
        metrics = self.orchestrator.validation_metrics['no_validation']
        assert metrics['total_checks'] == 0
        assert metrics['total_errors'] == 0
    
    def test_malformed_validation_result_handling(self):
        """Test handling of malformed validation results."""
        stage = MockStage('malformed')
        
        # Result with malformed validation data
        result = StageResult(
            success=True,
            data={
                'validation_results': [
                    {
                        'stage': 'malformed',
                        'result': "not_a_validation_result"  # Wrong type
                    }
                ]
            }
        )
        
        # Should handle without crashing
        self.orchestrator._track_validation_results(stage, result)
        
        # Check graceful handling
        assert 'malformed' in self.orchestrator.validation_results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])