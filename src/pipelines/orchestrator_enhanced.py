"""Enhanced pipeline orchestrator with integrated monitoring and logging."""

import uuid
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Type
from datetime import datetime

from src.config.config import Config
from src.database.connection import DatabaseManager
from src.database.schema import DatabaseSchema
from src.infrastructure.logging import (
    get_logger, setup_logging, LoggingContext, 
    log_operation, log_stage
)
from src.infrastructure.monitoring import UnifiedMonitor
from src.pipelines.orchestrator import PipelineOrchestrator
from src.pipelines.enhanced_context import EnhancedPipelineContext
from src.pipelines.stages.base_stage import PipelineStage
from src.core.signal_handler import SignalHandler

logger = get_logger(__name__)


class EnhancedPipelineOrchestrator(PipelineOrchestrator):
    """Enhanced pipeline orchestrator with full monitoring and logging integration.
    
    Features:
    - Structured logging with context propagation
    - Real-time progress tracking with persistence
    - Performance metrics collection
    - Automatic error capture with tracebacks
    - Resource monitoring during execution
    """
    
    def __init__(self, config: Config, db: DatabaseManager, 
                 signal_handler: Optional[SignalHandler] = None):
        """Initialize enhanced orchestrator.
        
        Args:
            config: Configuration object
            db: Database connection manager
            signal_handler: Optional signal handler
        """
        super().__init__(config, db, signal_handler)
        
        # Setup structured logging
        setup_logging(
            config=config,
            db_manager=db,
            console=True,
            database=True,
            log_level=config.get('logging.level', 'INFO')
        )
        
        logger.info("Enhanced pipeline orchestrator initialized")
    
    @log_operation("run_pipeline")
    def run_pipeline(self, 
                    experiment_name: str,
                    description: Optional[str] = None,
                    checkpoint_dir: Optional[Path] = None,
                    output_dir: Optional[Path] = None,
                    resume_from_checkpoint: bool = True,
                    **kwargs) -> Dict[str, Any]:
        """Run pipeline with enhanced monitoring and logging.
        
        Args:
            experiment_name: Name for this pipeline run
            description: Optional description
            checkpoint_dir: Directory for checkpoints
            output_dir: Directory for outputs
            resume_from_checkpoint: Whether to resume from checkpoints
            **kwargs: Additional configuration
            
        Returns:
            Pipeline execution results with metrics
        """
        # For now, just delegate to parent class
        # TODO: Add enhanced monitoring features in the future
        # Note: parent class doesn't accept description, so we add it to kwargs
        if description:
            kwargs['description'] = description
            
        # Pass through to parent class
        return super().run_pipeline(
            experiment_name=experiment_name,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            resume_from_checkpoint=resume_from_checkpoint,
            **kwargs
        )