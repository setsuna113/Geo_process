# src/pipelines/recovery/checkpoint_manager.py
"""Checkpoint management for pipeline recovery."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage pipeline checkpoints for recovery."""
    
    def __init__(self, config):
        self.config = config
        self.checkpoint_dir = Path(config.get('paths.checkpoint_dir', 'checkpoints'))
        self.max_checkpoints = config.get('pipeline.max_checkpoints_per_experiment', 10)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, experiment_id: str, stage_name: str,
                       context, stage_results: Dict[str, Any]):
        """Save checkpoint after stage completion."""
        checkpoint_data = {
            'experiment_id': experiment_id,
            'stage': stage_name,
            'timestamp': datetime.now().isoformat(),
            'completed_stages': self._get_completed_stages(context),
            'shared_data': self._serialize_shared_data(context.shared_data),
            'metadata': context.metadata,
            'quality_metrics': context.quality_metrics,
            'stage_results': self._serialize_stage_results(stage_results),
            'progress': {
                'completed_stages': len(self._get_completed_stages(context)),
                'stages': {}  # To be filled by progress tracker
            }
        }
        
        # Create checkpoint file
        checkpoint_path = self._get_checkpoint_path(experiment_id, stage_name)
        
        try:
            # Write to temporary file first
            temp_path = checkpoint_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            
            # Atomic rename
            temp_path.rename(checkpoint_path)
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints(experiment_id)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()
    
    def load_latest(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load latest checkpoint for experiment."""
        checkpoints = self._list_checkpoints(experiment_id)
        
        if not checkpoints:
            return None
        
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
        
        latest = checkpoints[0]
        checkpoint_path = self._get_checkpoint_path(experiment_id, latest['stage'])
        
        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Loaded checkpoint from stage: {latest['stage']}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def load_checkpoint(self, experiment_id: str, stage_name: str) -> Optional[Dict[str, Any]]:
        """Load specific checkpoint."""
        checkpoint_path = self._get_checkpoint_path(experiment_id, stage_name)
        
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def list_checkpoints(self, experiment_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for experiment."""
        return self._list_checkpoints(experiment_id)
    
    def cleanup_checkpoints(self, experiment_id: str):
        """Clean up all checkpoints for experiment."""
        exp_dir = self.checkpoint_dir / experiment_id
        
        if exp_dir.exists():
            try:
                shutil.rmtree(exp_dir)
                logger.info(f"Cleaned up checkpoints for experiment: {experiment_id}")
            except Exception as e:
                logger.error(f"Failed to cleanup checkpoints: {e}")
    
    def _get_checkpoint_path(self, experiment_id: str, stage_name: str) -> Path:
        """Get checkpoint file path."""
        exp_dir = self.checkpoint_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir / f"checkpoint_{stage_name}.json"
    
    def _list_checkpoints(self, experiment_id: str) -> List[Dict[str, Any]]:
        """List checkpoints for experiment."""
        exp_dir = self.checkpoint_dir / experiment_id
        
        if not exp_dir.exists():
            return []
        
        checkpoints = []
        for checkpoint_file in exp_dir.glob("checkpoint_*.json"):
            try:
                # Extract stage name from filename
                stage_name = checkpoint_file.stem.replace("checkpoint_", "")
                
                # Get file stats
                stat = checkpoint_file.stat()
                
                checkpoints.append({
                    'stage': stage_name,
                    'path': str(checkpoint_file),
                    'timestamp': datetime.fromtimestamp(stat.st_mtime),
                    'size_kb': stat.st_size / 1024
                })
            except Exception as e:
                logger.warning(f"Failed to read checkpoint file {checkpoint_file}: {e}")
        
        return checkpoints
    
    def _cleanup_old_checkpoints(self, experiment_id: str):
        """Remove old checkpoints keeping only recent ones."""
        checkpoints = self._list_checkpoints(experiment_id)
        
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by timestamp (oldest first)
        checkpoints.sort(key=lambda x: x['timestamp'])
        
        # Remove oldest checkpoints
        to_remove = len(checkpoints) - self.max_checkpoints
        for checkpoint in checkpoints[:to_remove]:
            try:
                Path(checkpoint['path']).unlink()
                logger.debug(f"Removed old checkpoint: {checkpoint['path']}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint: {e}")
    
    def _get_completed_stages(self, context) -> List[str]:
        """Get list of completed stages from context."""
        # This would be implemented based on your stage tracking
        completed = []
        # Placeholder - would check actual stage status
        return completed
    
    def _serialize_shared_data(self, shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize shared data for checkpointing."""
        # Handle special types that can't be JSON serialized
        serialized = {}
        
        for key, value in shared_data.items():
            if key in ['merged_dataset', 'som_results']:
                # Store reference instead of actual data
                serialized[key] = f"<{type(value).__name__} object>"
            elif hasattr(value, 'to_dict'):
                serialized[key] = value.to_dict()
            else:
                serialized[key] = value
        
        return serialized
    
    def _serialize_stage_results(self, stage_results: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize stage results for checkpointing."""
        serialized = {}
        
        for stage_name, result in stage_results.items():
            if hasattr(result, 'to_dict'):
                serialized[stage_name] = result.to_dict()
            elif hasattr(result, '__dict__'):
                serialized[stage_name] = {
                    'success': getattr(result, 'success', True),
                    'metrics': getattr(result, 'metrics', {}),
                    'warnings': getattr(result, 'warnings', [])
                }
            else:
                serialized[stage_name] = str(result)
        
        return serialized