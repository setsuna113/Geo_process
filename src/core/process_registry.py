# src/core/process_registry.py
"""
Persistent process registry for daemon tracking.
Stores process information in JSON files for cross-invocation access.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
import fcntl  # For file locking
import psutil

@dataclass
class ProcessRecord:
    """Persistent process information."""
    name: str
    pid: int
    command: List[str]
    start_time: float
    status: str
    log_file: str
    daemon_mode: bool
    auto_restart: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

class ProcessRegistry:
    """
    File-based process registry with atomic operations.
    """
    
    def __init__(self, registry_dir: Path):
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._lock_file = self.registry_dir / ".registry.lock"
        
    def _acquire_lock(self, timeout=5.0):
        """Acquire exclusive lock for registry operations."""
        lock_fd = os.open(str(self._lock_file), os.O_CREAT | os.O_RDWR)
        start_time = time.time()
        
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return lock_fd
            except BlockingIOError:
                if time.time() - start_time > timeout:
                    os.close(lock_fd)
                    raise TimeoutError("Could not acquire registry lock")
                time.sleep(0.1)
    
    def _release_lock(self, lock_fd):
        """Release registry lock."""
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    
    def register_process(self, record: ProcessRecord):
        """Register a new process."""
        lock_fd = self._acquire_lock()
        try:
            record_file = self.registry_dir / f"{record.name}.json"
            
            # Write atomically
            temp_file = record_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(record.to_dict(), f, indent=2)
            
            # Atomic rename
            temp_file.rename(record_file)
            
        finally:
            self._release_lock(lock_fd)
    
    def get_process(self, name: str) -> Optional[ProcessRecord]:
        """Get process information by name."""
        record_file = self.registry_dir / f"{name}.json"
        
        if not record_file.exists():
            return None
        
        try:
            with open(record_file, 'r') as f:
                data = json.load(f)
                return ProcessRecord.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None
    
    def list_processes(self) -> List[ProcessRecord]:
        """List all registered processes."""
        processes = []
        
        for record_file in self.registry_dir.glob("*.json"):
            if record_file.name == ".registry.lock":
                continue
                
            try:
                with open(record_file, 'r') as f:
                    data = json.load(f)
                    processes.append(ProcessRecord.from_dict(data))
            except (json.JSONDecodeError, KeyError):
                continue
        
        return processes
    
    def update_process_status(self, name: str, status: str, **kwargs):
        """Update process status and optional fields."""
        lock_fd = self._acquire_lock()
        try:
            record = self.get_process(name)
            if not record:
                return False
            
            record.status = status
            for key, value in kwargs.items():
                if hasattr(record, key):
                    setattr(record, key, value)
            
            # Update directly without re-acquiring lock
            record_file = self.registry_dir / f"{record.name}.json"
            temp_file = record_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(record.to_dict(), f, indent=2)
            temp_file.rename(record_file)
            
            return True
            
        finally:
            self._release_lock(lock_fd)
    
    def remove_process(self, name: str):
        """Remove process from registry."""
        lock_fd = self._acquire_lock()
        try:
            record_file = self.registry_dir / f"{name}.json"
            if record_file.exists():
                record_file.unlink()
                
        finally:
            self._release_lock(lock_fd)
    
    def cleanup_stale_processes(self):
        """Remove records for processes that no longer exist."""
        for record in self.list_processes():
            try:
                if not psutil.pid_exists(record.pid):
                    self.remove_process(record.name)
                else:
                    # Verify it's actually our process
                    proc = psutil.Process(record.pid)
                    if proc.create_time() > record.start_time + 1:
                        # Different process with same PID
                        self.remove_process(record.name)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.remove_process(record.name)