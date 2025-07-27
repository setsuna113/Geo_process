# Daemon Process Tracking System Repair Checklist

## ⚠️ CRITICAL SYSTEM INVARIANTS - READ FIRST

### Process Management Rules (NEVER VIOLATE)
1. **Process Registry Persistence**:
   - Process information MUST be persisted, not just stored in memory
   - Each ProcessController instance MUST be able to read/write shared state
   - **NEVER** rely on in-memory dictionaries for cross-process communication

2. **PID File Management**:
   - **ALWAYS** write PID files atomically (write to temp, then rename)
   - **NEVER** trust a PID without verifying the process exists
   - **ALWAYS** clean up stale PID files on startup

3. **Daemon Communication**:
   - **NEVER** assume daemon output goes to original stdout/stderr
   - **ALWAYS** redirect daemon output to log files
   - **NEVER** use print() for critical logging in daemons

4. **State Synchronization**:
   - **ALWAYS** use file-based or database-based state for daemon tracking
   - **NEVER** use threading primitives across process boundaries
   - **ALWAYS** handle race conditions in PID file operations

## Executive Summary

The current daemon process tracking system fails because it stores process information in memory, making it impossible for different invocations of the process manager to track running daemons. This checklist provides a systematic approach to implementing persistent process tracking without breaking existing functionality.

## Success Metrics

### Primary Success Criteria
1. ✅ `process_manager.py status geo` correctly shows running daemon status
2. ✅ `process_manager.py logs geo -f` successfully tails daemon logs
3. ✅ `process_manager.py stop geo` properly terminates daemon process
4. ✅ Process survives system restarts (PID files persist)
5. ✅ Multiple daemons can run concurrently without conflicts

### Secondary Success Criteria
1. ✅ Automatic cleanup of stale PID files
2. ✅ Proper handling of daemon crashes
3. ✅ Resource monitoring works across invocations
4. ✅ Log rotation continues functioning
5. ✅ Auto-restart feature works with persistent state

## Phase 1: Assessment and Backup (30 minutes)

### Step 1.1: Create Safety Backup
```bash
# Create backup of current system
cd ~/dev/geo
git add -A
git commit -m "Pre-daemon-repair checkpoint"
git branch daemon-repair-backup

# Backup critical files
cp -r src/core/process_controller.py src/core/process_controller.py.backup
cp -r scripts/process_manager.py scripts/process_manager.py.backup
```

### Step 1.2: Document Current State
```bash
# Check for any running processes
ps aux | grep -E "geo|biodiversity" | grep -v grep > current_processes.txt

# List all PID files
ls -la ~/.biodiversity/pid/ > current_pid_files.txt

# Check log files
ls -la ~/.biodiversity/logs/ > current_log_files.txt
```

### Step 1.3: Clean Stale Resources
```bash
# Remove stale PID files (verify processes don't exist first!)
for pid_file in ~/.biodiversity/pid/*.pid; do
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if ! ps -p "$pid" > /dev/null 2>&1; then
            echo "Removing stale PID file: $pid_file"
            rm "$pid_file"
        fi
    fi
done
```

## Phase 2: Core Infrastructure Updates (2 hours)

### Step 2.1: Create Process Registry Module
**File**: `src/core/process_registry.py`

```python
# NEW FILE - Persistent process registry
"""
Persistent process registry for daemon tracking.
Stores process information in JSON files for cross-invocation access.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
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
            
            self.register_process(record)
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
```

### Step 2.2: Update ProcessController
**File**: `src/core/process_controller.py`

Key changes needed:
1. Add ProcessRegistry integration
2. Update all methods to use persistent registry
3. Ensure backward compatibility

**Critical sections to modify**:

```python
# At top of file, add import:
from .process_registry import ProcessRegistry, ProcessRecord

# In __init__ method, add:
self.registry = ProcessRegistry(self.pid_dir / "registry")

# In start_process method, after line 148:
# Create persistent record
record = ProcessRecord(
    name=name,
    pid=pid,
    command=command,
    start_time=time.time(),
    status="running",
    log_file=str(log_file),
    daemon_mode=daemon_mode,
    auto_restart=auto_restart
)
self.registry.register_process(record)

# In get_process_status method, replace with:
def get_process_status(self, name: str) -> ProcessInfo:
    """Get process status information."""
    # Check persistent registry first
    record = self.registry.get_process(name)
    if not record:
        raise ValueError(f"Unknown process: {name}")
    
    # Convert to ProcessInfo for compatibility
    info = ProcessInfo(
        pid=record.pid,
        name=record.name,
        command=record.command,
        start_time=record.start_time,
        status=record.status,
        log_file=record.log_file,
        metadata=record.metadata
    )
    
    # Update real-time metrics
    self._update_process_info(name, info)
    return info

# In list_processes method, replace with:
def list_processes(self) -> List[ProcessInfo]:
    """List all managed processes."""
    # Get from persistent registry
    records = self.registry.list_processes()
    
    # Convert and update metrics
    processes = []
    for record in records:
        info = ProcessInfo(
            pid=record.pid,
            name=record.name,
            command=record.command,
            start_time=record.start_time,
            status=record.status,
            log_file=record.log_file,
            metadata=record.metadata
        )
        self._update_process_info(record.name, info)
        processes.append(info)
    
    return processes
```

### Step 2.3: Update Process Manager CLI
**File**: `scripts/process_manager.py`

No major changes needed, but add cleanup on startup:

```python
# In PipelineCLI.__init__, add:
# Clean up stale processes on startup
self.process_controller.registry.cleanup_stale_processes()
```

## Phase 3: Testing Protocol (1 hour)

### Step 3.1: Unit Testing
Create test file: `tests/test_process_registry.py`

```python
import pytest
import tempfile
from pathlib import Path
from src.core.process_registry import ProcessRegistry, ProcessRecord

def test_registry_persistence():
    """Test that registry survives across instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # First instance
        registry1 = ProcessRegistry(Path(tmpdir))
        record = ProcessRecord(
            name="test_process",
            pid=12345,
            command=["python", "test.py"],
            start_time=time.time(),
            status="running",
            log_file="/tmp/test.log",
            daemon_mode=True,
            auto_restart=True
        )
        registry1.register_process(record)
        
        # Second instance
        registry2 = ProcessRegistry(Path(tmpdir))
        retrieved = registry2.get_process("test_process")
        
        assert retrieved is not None
        assert retrieved.pid == 12345
        assert retrieved.name == "test_process"
```

### Step 3.2: Integration Testing

```bash
# Test sequence for daemon tracking
cd ~/dev/geo

# 1. Start a test daemon
./run_pipeline.sh --daemon --process-name "test_daemon"

# 2. In a new terminal, check status
./scripts/production/run_unified_resampling.sh --process-name test_daemon --signal status

# 3. Check logs
python scripts/process_manager.py logs test_daemon -f

# 4. Stop daemon
./scripts/production/run_unified_resampling.sh --process-name test_daemon --signal stop

# 5. Verify cleanup
python scripts/process_manager.py status
```

### Step 3.3: Failure Mode Testing

```bash
# Test stale PID cleanup
# 1. Create fake PID file
echo "99999" > ~/.biodiversity/pid/fake_process.pid

# 2. Create fake registry entry
cat > ~/.biodiversity/pid/registry/fake_process.json << EOF
{
  "name": "fake_process",
  "pid": 99999,
  "command": ["python", "fake.py"],
  "start_time": 1234567890,
  "status": "running",
  "log_file": "/tmp/fake.log",
  "daemon_mode": true,
  "auto_restart": false
}
EOF

# 3. Run cleanup
python -c "
from src.core.process_controller import ProcessController
controller = ProcessController()
controller.registry.cleanup_stale_processes()
"

# 4. Verify cleanup
ls ~/.biodiversity/pid/registry/
```

## Phase 4: Rollout Strategy (30 minutes)

### Step 4.1: Gradual Migration
1. Deploy registry module first (no breaking changes)
2. Update ProcessController with fallback to in-memory
3. Test with new daemons while old ones still run
4. Migrate existing daemons one by one

### Step 4.2: Rollback Plan
```bash
# If issues arise, quick rollback:
cd ~/dev/geo
git checkout daemon-repair-backup
cp src/core/process_controller.py.backup src/core/process_controller.py
cp scripts/process_manager.py.backup scripts/process_manager.py

# Clean up new registry files
rm -rf ~/.biodiversity/pid/registry/
```

### Step 4.3: Monitoring
Create monitoring script: `scripts/monitor_daemons.sh`

```bash
#!/bin/bash
# Monitor daemon health

echo "=== Daemon Process Monitor ==="
echo "Time: $(date)"
echo

# Check registry
echo "Registered Processes:"
ls -la ~/.biodiversity/pid/registry/*.json 2>/dev/null || echo "No processes registered"

# Check actual processes
echo -e "\nRunning Processes:"
ps aux | grep -E "biodiversity|geo" | grep -v grep

# Check PID files vs registry
echo -e "\nPID File Consistency:"
for pid_file in ~/.biodiversity/pid/*.pid; do
    if [ -f "$pid_file" ]; then
        name=$(basename "$pid_file" .pid)
        pid=$(cat "$pid_file")
        if [ -f ~/.biodiversity/pid/registry/${name}.json ]; then
            echo "✅ $name: PID file and registry match"
        else
            echo "❌ $name: PID file exists but not in registry"
        fi
    fi
done
```

## Phase 5: Documentation Updates (30 minutes)

### Step 5.1: Update CLAUDE.md
Add section on daemon management:

```markdown
### Daemon Process Management
The system uses a persistent process registry for tracking daemons:
- Registry location: `~/.biodiversity/pid/registry/`
- Each process has a JSON file with full metadata
- Automatic cleanup of stale processes on startup
- File locking ensures atomic operations

#### Common Commands:
- Start daemon: `./run_pipeline.sh --daemon --process-name "name"`
- Check status: `python scripts/process_manager.py status name`
- View logs: `python scripts/process_manager.py logs name -f`
- Stop daemon: `python scripts/process_manager.py stop name`
```

### Step 5.2: Create Troubleshooting Guide
File: `docs/daemon_troubleshooting.md`

## Risk Mitigation

### Potential Issues and Solutions

1. **File Lock Contention**
   - Risk: Multiple processes trying to update registry
   - Solution: Timeout and retry logic implemented

2. **Disk Space Issues**
   - Risk: Registry files accumulate
   - Solution: Automatic cleanup of old entries

3. **Permission Problems**
   - Risk: Different users running daemons
   - Solution: User-specific registry directories

4. **Network File Systems**
   - Risk: File locking may not work on NFS
   - Solution: Detect NFS and use alternative locking

## Validation Checklist

After implementation, verify:

- [ ] Daemon starts successfully
- [ ] Status command shows running daemon
- [ ] Logs command tails output correctly
- [ ] Stop command terminates daemon
- [ ] Registry files created in correct location
- [ ] Stale process cleanup works
- [ ] Multiple daemons can run concurrently
- [ ] System survives restart
- [ ] No regression in non-daemon mode
- [ ] Performance impact < 100ms per operation

## Emergency Contacts

If critical issues arise:
1. Check `~/.biodiversity/logs/` for error details
2. Run `scripts/monitor_daemons.sh` for system state
3. Use rollback procedure if necessary
4. Document issue in `daemon_issues.log`