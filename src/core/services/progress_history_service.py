"""Progress history recording service."""

import json
import threading
import time
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class ProgressHistoryService:
    """Service for recording progress history to files."""
    
    def __init__(self, history_file: Optional[Path] = None):
        self._history_file = history_file
        self._history_lock = threading.RLock()
        self._history_buffer = []
        self._buffer_size = 100  # Buffer entries before writing
    
    def set_history_file(self, history_file: Path) -> None:
        """Set the history file path."""
        with self._history_lock:
            self._history_file = history_file
            
            # Ensure parent directory exists
            if self._history_file:
                self._history_file.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Progress history will be saved to: {self._history_file}")
    
    def record_event(self, event_type: str, node_id: str, 
                    progress_data: Dict[str, Any]) -> None:
        """Record a progress event to history."""
        if not self._history_file:
            return
        
        timestamp = datetime.now().isoformat()
        
        history_entry = {
            'timestamp': timestamp,
            'event_type': event_type,
            'node_id': node_id,
            'progress_data': progress_data
        }
        
        with self._history_lock:
            self._history_buffer.append(history_entry)
            
            # Flush buffer if it's full
            if len(self._history_buffer) >= self._buffer_size:
                self._flush_history()
    
    def flush_history(self) -> None:
        """Manually flush history buffer to file."""
        with self._history_lock:
            self._flush_history()
    
    def _flush_history(self) -> None:
        """Internal method to flush history buffer."""
        if not self._history_file or not self._history_buffer:
            return
        
        try:
            # Append to history file
            with open(self._history_file, 'a') as f:
                for entry in self._history_buffer:
                    f.write(json.dumps(entry) + '\n')
            
            logger.debug(f"Flushed {len(self._history_buffer)} history entries")
            self._history_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to write progress history: {e}")
    
    def read_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Read progress history from file."""
        if not self._history_file or not self._history_file.exists():
            return []
        
        try:
            entries = []
            with open(self._history_file, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line.strip())
                        entries.append(entry)
                        
                        if limit and len(entries) >= limit:
                            break
            
            return entries
            
        except Exception as e:
            logger.error(f"Failed to read progress history: {e}")
            return []
    
    def clear_history(self) -> None:
        """Clear progress history file."""
        with self._history_lock:
            # Flush any remaining buffer
            self._flush_history()
            
            # Clear the file
            if self._history_file and self._history_file.exists():
                try:
                    self._history_file.unlink()
                    logger.info("Progress history cleared")
                except Exception as e:
                    logger.error(f"Failed to clear progress history: {e}")
    
    def get_history_stats(self) -> Dict[str, Any]:
        """Get statistics about progress history."""
        if not self._history_file or not self._history_file.exists():
            return {
                'file_exists': False,
                'total_entries': 0,
                'file_size_mb': 0.0,
                'buffer_size': len(self._history_buffer)
            }
        
        try:
            file_size = self._history_file.stat().st_size
            
            # Count entries (approximate)
            entry_count = 0
            with open(self._history_file, 'r') as f:
                for line in f:
                    if line.strip():
                        entry_count += 1
            
            return {
                'file_exists': True,
                'file_path': str(self._history_file),
                'total_entries': entry_count,
                'file_size_mb': file_size / (1024 * 1024),
                'buffer_size': len(self._history_buffer)
            }
            
        except Exception as e:
            logger.error(f"Failed to get history stats: {e}")
            return {
                'file_exists': True,
                'error': str(e),
                'buffer_size': len(self._history_buffer)
            }
    
    def __del__(self):
        """Cleanup when service is destroyed."""
        try:
            self._flush_history()
        except Exception as e:
            logger.error(f"Error flushing history on cleanup: {e}")