"""Workflow State Manager

Manages persistent state for the GEE workflow to enable resumption.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
from enum import Enum

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class WorkflowStep(Enum):
    """Workflow steps"""
    NOT_STARTED = "not_started"
    PREREQUISITES_VALIDATED = "prerequisites_validated"
    AUTHENTICATED = "authenticated"
    ASSETS_UPLOADED = "assets_uploaded"
    PROCESSING_COMPLETE = "processing_complete"
    EXPORT_COMPLETE = "export_complete"
    DOWNLOAD_COMPLETE = "download_complete"
    WORKFLOW_COMPLETE = "workflow_complete"


class WorkflowState:
    """Manages workflow state persistence"""
    
    def __init__(self, state_file: Optional[Path] = None):
        """
        Initialize state manager.
        
        Args:
            state_file: Path to state file
        """
        if state_file is None:
            state_file = Path.home() / '.biodiversity' / 'gee_workflow_state.json'
        
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        
        # Load or initialize state
        self.state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """Load state from file or create new"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                self.logger.info(f"Loaded workflow state from {self.state_file}")
                return state
            except Exception as e:
                self.logger.warning(f"Failed to load state: {e}")
        
        # Create new state
        return {
            'current_step': WorkflowStep.NOT_STARTED.value,
            'started_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'completed_steps': [],
            'metadata': {}
        }
    
    def save(self) -> None:
        """Save state to file"""
        self.state['updated_at'] = datetime.now().isoformat()
        
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
            self.logger.debug("Saved workflow state")
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    @property
    def current_step(self) -> WorkflowStep:
        """Get current workflow step"""
        return WorkflowStep(self.state['current_step'])
    
    def update_step(self, step: WorkflowStep) -> None:
        """Update current step"""
        self.state['current_step'] = step.value
        
        # Add to completed steps
        if step.value not in self.state['completed_steps']:
            self.state['completed_steps'].append(step.value)
        
        self.save()
        self.logger.info(f"Workflow step updated: {step.value}")
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value"""
        self.state['metadata'][key] = value
        self.save()
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value"""
        return self.state['metadata'].get(key, default)
    
    def is_step_complete(self, step: WorkflowStep) -> bool:
        """Check if a step is complete"""
        return step.value in self.state['completed_steps']
    
    def reset(self) -> None:
        """Reset workflow state"""
        self.state = {
            'current_step': WorkflowStep.NOT_STARTED.value,
            'started_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'completed_steps': [],
            'metadata': {}
        }
        self.save()
        self.logger.info("Workflow state reset")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get workflow summary"""
        return {
            'current_step': self.current_step.value,
            'completed_steps': len(self.state['completed_steps']),
            'total_steps': len(WorkflowStep),
            'started_at': self.state['started_at'],
            'updated_at': self.state['updated_at'],
            'is_complete': self.current_step == WorkflowStep.WORKFLOW_COMPLETE
        }
    
    @classmethod
    def load_or_create(cls, state_file: Optional[Path] = None) -> 'WorkflowState':
        """Load existing state or create new"""
        return cls(state_file)