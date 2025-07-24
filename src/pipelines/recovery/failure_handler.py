# src/pipelines/recovery/failure_handler.py
"""Failure handling and recovery strategies."""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY = "retry"
    SKIP = "skip"
    RESTART = "restart"
    ABORT = "abort"


class FailureHandler:
    """Handle pipeline failures and determine recovery strategies."""
    
    def __init__(self, config):
        self.config = config
        self.max_retries = config.get('pipeline.max_retries', 3)
        self.retry_delays = config.get('pipeline.retry_delays', [5, 30, 60])  # seconds
        
        # Failure history
        self.failures: List[Dict[str, Any]] = []
        self.retry_counts: Dict[str, int] = {}
    
    def handle_failure(self, error: Exception, context, stage) -> bool:
        """
        Handle a pipeline failure.
        
        Returns:
            bool: Whether recovery should be attempted
        """
        failure_info = {
            'timestamp': datetime.now(),
            'stage': stage.name if stage else 'unknown',
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': {
                'experiment_id': context.experiment_id if context else None
            }
        }
        
        self.failures.append(failure_info)
        
        # Log failure
        logger.error(f"Pipeline failure in stage '{failure_info['stage']}': {error}")
        
        # Determine recovery strategy
        strategy = self._determine_recovery_strategy(error, stage)
        
        if strategy == RecoveryStrategy.ABORT:
            return False
        
        # Update retry count
        if stage:
            self.retry_counts[stage.name] = self.retry_counts.get(stage.name, 0) + 1
        
        return True
    
    def _determine_recovery_strategy(self, error: Exception, stage) -> RecoveryStrategy:
        """Determine appropriate recovery strategy based on error type."""
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Memory errors - might recover after cleanup
        if error_type == 'MemoryError' or 'memory' in error_str:
            logger.info("Memory error detected - will attempt recovery after cleanup")
            return RecoveryStrategy.RETRY
        
        # Temporary failures - network, database connection
        if any(keyword in error_str for keyword in ['connection', 'timeout', 'temporary']):
            logger.info("Temporary failure detected - will retry")
            return RecoveryStrategy.RETRY
        
        # Data quality issues - might skip stage
        if 'quality' in error_str or 'validation' in error_str:
            logger.info("Quality issue detected - considering skip strategy")
            if self._can_skip_stage(stage):
                return RecoveryStrategy.SKIP
        
        # Check retry limit
        if stage and self.retry_counts.get(stage.name, 0) >= self.max_retries:
            logger.warning(f"Max retries ({self.max_retries}) reached for stage '{stage.name}'")
            return RecoveryStrategy.ABORT
        
        # Default to retry for unknown errors
        return RecoveryStrategy.RETRY
    
    def _can_skip_stage(self, stage) -> bool:
        """Check if stage can be skipped."""
        # Define which stages are optional
        optional_stages = self.config.get('pipeline.optional_stages', [])
        
        if stage and stage.name in optional_stages:
            return True
        
        # Don't skip critical stages
        critical_stages = ['data_load', 'resample', 'merge']
        if stage and stage.name in critical_stages:
            return False
        
        return False
    
    def get_recovery_strategy(self) -> str:
        """Get recommended recovery strategy."""
        if not self.failures:
            return RecoveryStrategy.RETRY.value
        
        latest_failure = self.failures[-1]
        
        # Check failure patterns
        if self._has_repeated_failures():
            logger.warning("Repeated failures detected - recommending abort")
            return RecoveryStrategy.ABORT.value
        
        # Based on last error
        if 'memory' in latest_failure['error_message'].lower():
            return RecoveryStrategy.RETRY.value
        
        return RecoveryStrategy.RETRY.value
    
    def _has_repeated_failures(self) -> bool:
        """Check if there are repeated failures."""
        if len(self.failures) < 3:
            return False
        
        # Check if last 3 failures are similar
        recent_failures = self.failures[-3:]
        error_types = [f['error_type'] for f in recent_failures]
        
        # All same error type
        if len(set(error_types)) == 1:
            return True
        
        # Same stage failing repeatedly
        stages = [f['stage'] for f in recent_failures]
        if len(set(stages)) == 1:
            return True
        
        return False
    
    def can_recover(self) -> bool:
        """Check if recovery is possible."""
        # Check total failure count
        if len(self.failures) > 10:
            logger.error("Too many failures - recovery not recommended")
            return False
        
        # Check for critical errors
        critical_errors = ['CorruptionError', 'DataLossError', 'SystemError']
        for failure in self.failures:
            if failure['error_type'] in critical_errors:
                logger.error(f"Critical error detected: {failure['error_type']}")
                return False
        
        return True
    
    def get_retry_delay(self, stage_name: str) -> int:
        """Get retry delay for stage."""
        retry_count = self.retry_counts.get(stage_name, 0)
        
        if retry_count < len(self.retry_delays):
            return self.retry_delays[retry_count]
        
        # Default to last delay
        return self.retry_delays[-1]
    
    def get_failure_report(self) -> Dict[str, Any]:
        """Get comprehensive failure report."""
        if not self.failures:
            return {'status': 'no_failures'}
        
        # Aggregate by stage
        stage_failures = {}
        for failure in self.failures:
            stage = failure['stage']
            if stage not in stage_failures:
                stage_failures[stage] = []
            stage_failures[stage].append(failure)
        
        # Aggregate by error type
        error_types = {}
        for failure in self.failures:
            error_type = failure['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_failures': len(self.failures),
            'stages_affected': list(stage_failures.keys()),
            'error_types': error_types,
            'retry_counts': self.retry_counts,
            'latest_failure': self.failures[-1] if self.failures else None,
            'recovery_possible': self.can_recover()
        }