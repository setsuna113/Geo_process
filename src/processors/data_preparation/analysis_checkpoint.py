"""Simple file-based checkpointing system for standalone analysis."""

import pickle
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AnalysisCheckpoint:
    """Simple checkpoint data structure for analysis."""
    checkpoint_id: str
    experiment_name: str
    analysis_method: str
    timestamp: str
    chunk_idx: int
    total_chunks: int
    progress_percent: float
    
    # Method-specific state
    analysis_state: Dict[str, Any]
    
    # Metadata
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisCheckpoint':
        """Create checkpoint from dictionary."""
        return cls(**data)


class SimpleAnalysisCheckpointer:
    """Simple file-based checkpointing for analysis processes."""
    
    def __init__(self, checkpoint_dir: Path, experiment_name: str, analysis_method: str):
        """Initialize checkpointer.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            experiment_name: Name of the experiment
            analysis_method: Analysis method (som, gwpca, maxp_regions)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        self.analysis_method = analysis_method
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint file naming
        self.checkpoint_prefix = f"{experiment_name}_{analysis_method}"
        
        logger.info(f"Initialized checkpointer for {experiment_name} ({analysis_method})")
        logger.debug(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def save_checkpoint(
        self,
        chunk_idx: int,
        total_chunks: int,
        analysis_state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save analysis checkpoint.
        
        Args:
            chunk_idx: Current chunk index
            total_chunks: Total number of chunks
            analysis_state: Method-specific analysis state
            metadata: Additional metadata
            
        Returns:
            Path to saved checkpoint file
        """
        # Calculate progress
        progress_percent = (chunk_idx / total_chunks) * 100
        
        # Generate checkpoint ID
        timestamp = datetime.now().isoformat()
        checkpoint_id = self._generate_checkpoint_id(chunk_idx, timestamp)
        
        # Create checkpoint object
        checkpoint = AnalysisCheckpoint(
            checkpoint_id=checkpoint_id,
            experiment_name=self.experiment_name,
            analysis_method=self.analysis_method,
            timestamp=timestamp,
            chunk_idx=chunk_idx,
            total_chunks=total_chunks,
            progress_percent=progress_percent,
            analysis_state=analysis_state,
            metadata=metadata or {}
        )
        
        # Save checkpoint
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)
        
        try:
            # Save binary checkpoint (for complex objects)
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Also save JSON metadata for easy inspection
            metadata_path = checkpoint_path.with_suffix('.json')
            checkpoint_dict = checkpoint.to_dict()
            
            # Remove complex objects from JSON (keep only metadata)
            json_safe_dict = {
                'checkpoint_id': checkpoint_dict['checkpoint_id'],
                'experiment_name': checkpoint_dict['experiment_name'],
                'analysis_method': checkpoint_dict['analysis_method'],
                'timestamp': checkpoint_dict['timestamp'],
                'chunk_idx': checkpoint_dict['chunk_idx'],
                'total_chunks': checkpoint_dict['total_chunks'],
                'progress_percent': checkpoint_dict['progress_percent'],
                'metadata': checkpoint_dict['metadata']
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(json_safe_dict, f, indent=2)
            
            logger.info(f"Saved checkpoint {checkpoint_id} at {progress_percent:.1f}% progress")
            logger.debug(f"Checkpoint saved to: {checkpoint_path}")
            
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_latest_checkpoint(self) -> Optional[AnalysisCheckpoint]:
        """Load the most recent checkpoint.
        
        Returns:
            Latest checkpoint or None if no checkpoints exist
        """
        checkpoint_files = list(self.checkpoint_dir.glob(f"{self.checkpoint_prefix}_*.pkl"))
        
        if not checkpoint_files:
            logger.info("No checkpoints found")
            return None
        
        # Sort by modification time (most recent first)
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest_checkpoint = checkpoint_files[0]
        
        try:
            with open(latest_checkpoint, 'rb') as f:
                checkpoint = pickle.load(f)
            
            logger.info(f"Loaded checkpoint {checkpoint.checkpoint_id} "
                       f"(chunk {checkpoint.chunk_idx}/{checkpoint.total_chunks})")
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {latest_checkpoint}: {e}")
            return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata.
        
        Returns:
            List of checkpoint metadata dictionaries
        """
        checkpoint_files = list(self.checkpoint_dir.glob(f"{self.checkpoint_prefix}_*.json"))
        checkpoints = []
        
        for checkpoint_file in sorted(checkpoint_files):
            try:
                with open(checkpoint_file, 'r') as f:
                    metadata = json.load(f)
                    
                # Add file info
                stat = checkpoint_file.stat()
                metadata['file_size_mb'] = stat.st_size / 1024 / 1024
                metadata['file_path'] = str(checkpoint_file)
                
                checkpoints.append(metadata)
                
            except Exception as e:
                logger.warning(f"Failed to read checkpoint metadata {checkpoint_file}: {e}")
        
        return checkpoints
    
    def cleanup_old_checkpoints(self, keep_last: int = 5):
        """Clean up old checkpoints, keeping only the most recent ones.
        
        Args:
            keep_last: Number of recent checkpoints to keep
        """
        checkpoint_files = list(self.checkpoint_dir.glob(f"{self.checkpoint_prefix}_*.pkl"))
        
        if len(checkpoint_files) <= keep_last:
            logger.debug(f"Only {len(checkpoint_files)} checkpoints found, no cleanup needed")
            return
        
        # Sort by modification time (oldest first)
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime)
        
        # Remove old checkpoints
        to_remove = checkpoint_files[:-keep_last]
        removed_count = 0
        
        for checkpoint_file in to_remove:
            try:
                # Remove both .pkl and .json files
                checkpoint_file.unlink()
                json_file = checkpoint_file.with_suffix('.json')
                if json_file.exists():
                    json_file.unlink()
                
                removed_count += 1
                logger.debug(f"Removed old checkpoint: {checkpoint_file.name}")
                
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint_file}: {e}")
        
        logger.info(f"Cleaned up {removed_count} old checkpoints, kept {keep_last} most recent")
    
    def _generate_checkpoint_id(self, chunk_idx: int, timestamp: str) -> str:
        """Generate unique checkpoint ID."""
        # Create hash from experiment details
        content = f"{self.experiment_name}_{self.analysis_method}_{chunk_idx}_{timestamp}"
        hash_obj = hashlib.md5(content.encode())
        short_hash = hash_obj.hexdigest()[:8]
        
        return f"{self.checkpoint_prefix}_chunk_{chunk_idx:04d}_{short_hash}"
    
    def _get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get full path for checkpoint file."""
        return self.checkpoint_dir / f"{checkpoint_id}.pkl"
    
    def resume_from_checkpoint(self, checkpoint: AnalysisCheckpoint) -> Dict[str, Any]:
        """Extract resume information from checkpoint.
        
        Args:
            checkpoint: Checkpoint to resume from
            
        Returns:
            Dictionary with resume information
        """
        resume_info = {
            'start_chunk': checkpoint.chunk_idx + 1,  # Resume from next chunk
            'total_chunks': checkpoint.total_chunks,
            'analysis_state': checkpoint.analysis_state,
            'metadata': checkpoint.metadata
        }
        
        logger.info(f"Resuming from checkpoint {checkpoint.checkpoint_id}")
        logger.info(f"Will resume from chunk {resume_info['start_chunk']}/{resume_info['total_chunks']}")
        
        return resume_info


def create_checkpointer(
    output_dir: Path,
    experiment_name: str,
    analysis_method: str
) -> SimpleAnalysisCheckpointer:
    """Factory function to create analysis checkpointer.
    
    Args:
        output_dir: Output directory for the experiment
        experiment_name: Name of the experiment
        analysis_method: Analysis method being used
        
    Returns:
        Configured checkpointer instance
    """
    checkpoint_dir = output_dir / 'checkpoints'
    return SimpleAnalysisCheckpointer(
        checkpoint_dir=checkpoint_dir,
        experiment_name=experiment_name,
        analysis_method=analysis_method
    )