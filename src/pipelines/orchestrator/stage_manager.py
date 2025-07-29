# src/pipelines/orchestrator/stage_manager.py
"""Stage execution management for pipeline orchestration."""

import logging
from typing import Dict, Any, List, Optional, Type, Tuple
from dataclasses import dataclass
import threading
import time

from ..stages.base_stage import PipelineStage, StageStatus, StageResult
from ..stages.base_stage import ProcessingConfig

logger = logging.getLogger(__name__)


class StageManager:
    """Manages stage registration, validation, and execution coordination."""
    
    def __init__(self, context=None):
        """Initialize stage manager."""
        self.context = context
        self.stages: List[PipelineStage] = []
        self.stage_registry: Dict[str, PipelineStage] = {}
        self._execution_lock = threading.Lock()
        
    def register_stage(self, stage: PipelineStage):
        """Register a pipeline stage."""
        if stage.stage_name in self.stage_registry:
            raise ValueError(f"Stage '{stage.stage_name}' already registered")
            
        self.stages.append(stage)
        self.stage_registry[stage.stage_name] = stage
        logger.info(f"Registered stage: {stage.stage_name}")
        
    def configure_pipeline(self, stages: List[Type[PipelineStage]]):
        """Configure pipeline with stage classes."""
        for stage_class in stages:
            stage = stage_class()
            self.register_stage(stage)
            
    def validate_pipeline(self) -> Tuple[bool, List[str]]:
        """Validate pipeline configuration."""
        errors = []
        
        if not self.stages:
            errors.append("No stages configured")
            return False, errors
            
        # Check dependencies
        for stage in self.stages:
            for dep in stage.dependencies:
                if dep not in self.stage_registry:
                    errors.append(f"Stage '{stage.stage_name}' depends on missing stage '{dep}'")
                    
        # Check for circular dependencies
        for stage in self.stages:
            if self._has_circular_dependency(stage):
                errors.append(f"Circular dependency detected involving stage '{stage.stage_name}'")
                
        return len(errors) == 0, errors
        
    def get_execution_order(self) -> List[PipelineStage]:
        """Get stages in execution order based on dependencies."""
        ordered_stages = []
        completed = set()
        
        def can_execute(stage: PipelineStage) -> bool:
            return all(dep in completed for dep in stage.dependencies)
            
        remaining = self.stages.copy()
        
        while remaining:
            ready = [s for s in remaining if can_execute(s)]
            
            if not ready:
                # Circular dependency or missing dependency
                remaining_names = [s.stage_name for s in remaining]
                raise RuntimeError(f"Cannot resolve execution order for stages: {remaining_names}")
                
            # Add ready stages
            for stage in ready:
                ordered_stages.append(stage)
                completed.add(stage.stage_name)
                remaining.remove(stage)
                
        return ordered_stages
        
    def execute_stage(self, stage: PipelineStage, **kwargs) -> StageResult:
        """Execute a single stage with proper error handling."""
        with self._execution_lock:
            logger.info(f"Executing stage: {stage.stage_name}")
            
            try:
                # Pre-stage checks
                self._pre_stage_checks(stage)
                
                # Check if stage can be skipped
                skip_result = self._check_stage_skip(stage)
                if skip_result:
                    return skip_result
                    
                # Execute the stage
                stage.status = StageStatus.RUNNING
                result = stage.execute(self.context, **kwargs)
                
                # Post-stage processing
                processed_result = self._post_stage_processing(stage, result)
                
                stage.status = StageStatus.COMPLETED
                logger.info(f"Stage {stage.stage_name} completed successfully")
                
                return processed_result
                
            except Exception as e:
                stage.status = StageStatus.FAILED
                logger.error(f"Stage {stage.stage_name} failed: {e}")
                
                error_result = StageResult(
                    stage_name=stage.stage_name,
                    success=False,
                    error_message=str(e),
                    metadata={'execution_time': time.time() - stage.start_time if hasattr(stage, 'start_time') else 0}
                )
                return error_result
                
    def execute_parallel_stages(self, stages: List[PipelineStage], **kwargs) -> Dict[str, StageResult]:
        """Execute multiple independent stages in parallel."""
        import concurrent.futures
        
        results = {}
        
        # Only run stages that have no unmet dependencies
        ready_stages = []
        for stage in stages:
            if all(dep_stage.status == StageStatus.COMPLETED 
                   for dep_name in stage.dependencies 
                   for dep_stage in self.stages 
                   if dep_stage.stage_name == dep_name):
                ready_stages.append(stage)
                
        if not ready_stages:
            logger.warning("No stages ready for parallel execution")
            return results
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(ready_stages)) as executor:
            # Submit all stages
            future_to_stage = {
                executor.submit(self.execute_stage, stage, **kwargs): stage 
                for stage in ready_stages
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_stage):
                stage = future_to_stage[future]
                try:
                    result = future.result()
                    results[stage.stage_name] = result
                except Exception as e:
                    logger.error(f"Parallel stage {stage.stage_name} failed: {e}")
                    results[stage.stage_name] = StageResult(
                        stage_name=stage.stage_name,
                        success=False,
                        error_message=str(e)
                    )
                    
        return results
        
    def _pre_stage_checks(self, stage: PipelineStage):
        """Perform pre-execution checks."""
        # Check dependencies are met
        for dep_name in stage.dependencies:
            dep_stage = self.stage_registry.get(dep_name)
            if not dep_stage or dep_stage.status != StageStatus.COMPLETED:
                raise RuntimeError(f"Dependency '{dep_name}' not completed for stage '{stage.stage_name}'")
                
        # Mark stage as starting
        stage.start_time = time.time()
        stage.status = StageStatus.RUNNING
        
    def _post_stage_processing(self, stage: PipelineStage, result: StageResult) -> StageResult:
        """Process stage results after execution."""
        # Add timing information
        if hasattr(stage, 'start_time'):
            execution_time = time.time() - stage.start_time
            if result.metadata is None:
                result.metadata = {}
            result.metadata['execution_time'] = execution_time
            
        # Store result in context if needed
        if self.context:
            self.context.set(f"stage_result_{stage.stage_name}", result)
            
        return result
        
    def _check_stage_skip(self, stage: PipelineStage) -> Optional[StageResult]:
        """Check if stage should be skipped."""
        # Check if stage has skip conditions
        if hasattr(stage, 'should_skip') and stage.should_skip(self.context):
            logger.info(f"Skipping stage {stage.stage_name} - conditions not met")
            stage.status = StageStatus.SKIPPED
            
            return StageResult(
                stage_name=stage.stage_name,
                success=True,
                message="Stage skipped - conditions not met",
                metadata={'skipped': True}
            )
            
        return None
        
    def _has_circular_dependency(self, start_stage: PipelineStage) -> bool:
        """Check for circular dependencies starting from a stage."""
        visited = set()
        rec_stack = set()
        
        def visit(stage_name: str) -> bool:
            if stage_name in rec_stack:
                return True  # Circular dependency found
            if stage_name in visited:
                return False
                
            visited.add(stage_name)
            rec_stack.add(stage_name)
            
            stage = self.stage_registry.get(stage_name)
            if stage:
                for dep in stage.dependencies:
                    if visit(dep):
                        return True
                        
            rec_stack.remove(stage_name)
            return False
            
        return visit(start_stage.stage_name)
        
    def get_stage_by_name(self, name: str) -> Optional[PipelineStage]:
        """Get stage by name."""
        return self.stage_registry.get(name)
        
    def get_completed_stages(self) -> List[str]:
        """Get list of completed stage names."""
        return [stage.stage_name for stage in self.stages 
                if stage.status == StageStatus.COMPLETED]
                
    def reset_pipeline(self):
        """Reset all stages to initial state."""
        for stage in self.stages:
            stage.status = StageStatus.PENDING
            if hasattr(stage, 'start_time'):
                delattr(stage, 'start_time')
                
        logger.info("Pipeline reset - all stages back to pending")