# src/core/process_controller.py
"""Python-based process management for the biodiversity pipeline."""

import os
import sys
import time
import signal
import logging
import threading
import subprocess
import psutil
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
import json
# Manual daemon implementation - no external dependencies needed
from .process_registry import ProcessRegistry, ProcessRecord

logger = logging.getLogger(__name__)


@dataclass
class ProcessInfo:
    """Information about a managed process."""
    pid: int
    name: str
    command: List[str]
    start_time: float
    status: str  # running, paused, stopped, crashed
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    log_file: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProcessController:
    """
    Python-based process management with daemon support.
    
    Features:
    - Daemon mode with PID file
    - Process health monitoring
    - Auto-restart on crash
    - Log rotation
    - Pause/resume support
    """
    
    def __init__(self, 
                 pid_dir: Optional[Path] = None,
                 log_dir: Optional[Path] = None,
                 max_log_size_mb: float = 100,
                 max_log_files: int = 5):
        """
        Initialize process controller.
        
        Args:
            pid_dir: Directory for PID files
            log_dir: Directory for log files
            max_log_size_mb: Maximum log file size before rotation
            max_log_files: Maximum number of rotated log files
        """
        import tempfile
        
        # Set default directories based on environment
        is_test_mode = os.getenv('FORCE_TEST_MODE', '').lower() in ('true', '1', 'yes') or 'pytest' in sys.modules
        
        if pid_dir is None:
            if is_test_mode:
                pid_dir = Path(tempfile.gettempdir()) / "biodiversity_test" / "pid"
            else:
                pid_dir = Path.home() / ".biodiversity" / "pid"
        
        if log_dir is None:
            if is_test_mode:
                log_dir = Path(tempfile.gettempdir()) / "biodiversity_test" / "logs"
            else:
                log_dir = Path.home() / ".biodiversity" / "logs"
        
        self.pid_dir = pid_dir
        self.log_dir = log_dir
        self.max_log_size_mb = max_log_size_mb
        self.max_log_files = max_log_files
        
        # Create directories
        self.pid_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Process registry - both in-memory and persistent
        self._processes: Dict[str, ProcessInfo] = {}
        self._process_lock = threading.RLock()
        self.registry = ProcessRegistry(self.pid_dir / "registry")
        
        # Monitoring
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._monitor_interval = 5.0  # seconds
        
        # Auto-restart configuration
        self._auto_restart: Dict[str, bool] = {}
        self._restart_delay = 5.0  # seconds
        self._max_restart_attempts = 3
        self._restart_attempts: Dict[str, int] = {}
        
        # Clean up stale processes on startup
        self.registry.cleanup_stale_processes()
    
    def start_process(self,
                     name: str,
                     command: List[str],
                     daemon_mode: bool = False,
                     auto_restart: bool = True,
                     max_restarts: Optional[int] = None,
                     env: Optional[Dict[str, str]] = None,
                     working_dir: Optional[Path] = None) -> int:
        """
        Start a managed process.
        
        Args:
            name: Process name
            command: Command and arguments
            daemon_mode: Run as daemon
            auto_restart: Auto-restart on crash
            env: Environment variables
            working_dir: Working directory
            
        Returns:
            Process ID
        """
        with self._process_lock:
            # Check if already running
            if name in self._processes and self._is_process_running(name):
                raise ValueError(f"Process '{name}' is already running")
            
            # Setup logging
            log_file = self.log_dir / f"{name}.log"
            
            if daemon_mode:
                # Start as daemon
                pid = self._start_daemon_process(name, command, log_file, env, working_dir)
            else:
                # Start as regular process
                pid = self._start_regular_process(name, command, log_file, env, working_dir)
            
            # Register process in memory
            process_info = ProcessInfo(
                pid=pid,
                name=name,
                command=command,
                start_time=time.time(),
                status="running",
                log_file=str(log_file)
            )
            self._processes[name] = process_info
            
            # Create persistent record
            record = ProcessRecord(
                name=name,
                pid=pid,
                command=command,
                start_time=process_info.start_time,
                status="running",
                log_file=str(log_file),
                daemon_mode=daemon_mode,
                auto_restart=auto_restart
            )
            self.registry.register_process(record)
            
            self._auto_restart[name] = auto_restart
            self._restart_attempts[name] = 0
            
            # Start monitoring if not running
            if not self._monitor_thread or not self._monitor_thread.is_alive():
                self._start_monitoring()
            
            logger.info(f"Started process '{name}' with PID {pid}")
            return pid
    
    def stop_process(self, name: str, timeout: float = 30.0) -> bool:
        """
        Stop a managed process gracefully.
        
        Args:
            name: Process name
            timeout: Timeout for graceful shutdown
            
        Returns:
            True if stopped successfully
        """
        with self._process_lock:
            # Check persistent registry first
            record = self.registry.get_process(name)
            if not record:
                raise ValueError(f"Unknown process: {name}")
            
            # Convert to ProcessInfo for compatibility
            process_info = ProcessInfo(
                pid=record.pid,
                name=record.name,
                command=record.command,
                start_time=record.start_time,
                status=record.status,
                log_file=record.log_file,
                metadata=record.metadata
            )
            
            # Update in-memory cache
            self._processes[name] = process_info
            
            try:
                process = psutil.Process(process_info.pid)
                
                # Send SIGTERM for graceful shutdown
                process.terminate()
                
                # Wait for process to stop
                try:
                    process.wait(timeout=timeout)
                except psutil.TimeoutExpired:
                    # Force kill if timeout
                    logger.warning(f"Process '{name}' did not stop gracefully, forcing kill")
                    process.kill()
                    process.wait(timeout=5.0)
                
                process_info.status = "stopped"
                self._auto_restart[name] = False  # Disable auto-restart
                
                # Update persistent registry
                self.registry.update_process_status(name, "stopped")
                
                logger.info(f"Stopped process '{name}'")
                return True
                
            except psutil.NoSuchProcess:
                process_info.status = "stopped"
                self.registry.update_process_status(name, "stopped")
                return True
            except Exception as e:
                logger.error(f"Failed to stop process '{name}': {e}")
                return False
    
    def pause_process(self, name: str) -> bool:
        """
        Pause a process (SIGUSR1).
        
        Args:
            name: Process name
            
        Returns:
            True if paused successfully
        """
        return self._send_signal(name, signal.SIGUSR1, "paused")
    
    def resume_process(self, name: str) -> bool:
        """
        Resume a paused process (SIGUSR2).
        
        Args:
            name: Process name
            
        Returns:
            True if resumed successfully
        """
        return self._send_signal(name, signal.SIGUSR2, "running")
    
    def get_process_status(self, name: str) -> ProcessInfo:
        """Get process status information."""
        with self._process_lock:
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
            
            # Update in-memory cache
            self._processes[name] = info
            
            # Update real-time metrics
            self._update_process_info(name)
            
            return self._processes[name]
    
    def list_processes(self) -> List[ProcessInfo]:
        """List all managed processes."""
        with self._process_lock:
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
                
                # Update in-memory cache
                self._processes[record.name] = info
                
                # Update real-time metrics
                self._update_process_info(record.name)
                processes.append(self._processes[record.name])
            
            return processes
    
    def detect_orphaned_processes(self) -> List[str]:
        """Detect orphaned processes that are no longer running."""
        orphaned = []
        with self._process_lock:
            for name, info in self._processes.items():
                try:
                    process = psutil.Process(info.pid)
                    if not process.is_running():
                        orphaned.append(name)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    orphaned.append(name)
        return orphaned
    
    def tail_log(self, name: str, lines: int = 100) -> List[str]:
        """
        Get last N lines from process log.
        
        Args:
            name: Process name
            lines: Number of lines to return
            
        Returns:
            Log lines
        """
        # Check persistent registry first
        record = self.registry.get_process(name)
        if not record:
            raise ValueError(f"Unknown process: {name}")
        
        if not record.log_file:
            return []
        
        log_file = Path(record.log_file)
        if not log_file.exists():
            return []
        
        # Use tail command for efficiency
        try:
            result = subprocess.run(
                ['tail', '-n', str(lines), str(log_file)],
                capture_output=True,
                text=True
            )
            return result.stdout.splitlines()
        except Exception as e:
            logger.error(f"Failed to tail log: {e}")
            return []
    
    def _start_daemon_process(self,
                            name: str,
                            command: List[str],
                            log_file: Path,
                            env: Optional[Dict[str, str]],
                            working_dir: Optional[Path]) -> int:
        """Start a process as daemon."""
        pid_file = self.pid_dir / f"{name}.pid"
        
        # Fork process to create daemon
        pid = os.fork()
        
        if pid == 0:
            # Child process - become daemon
            # Detach from parent environment
            os.setsid()
            
            # Second fork to ensure not a session leader
            pid2 = os.fork()
            if pid2 != 0:
                os._exit(0)
            
            # Change directory
            if working_dir:
                os.chdir(str(working_dir))
            else:
                os.chdir('/')
            
            # Set umask
            os.umask(0)
            
            # Close file descriptors
            import resource
            max_fd = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
            if max_fd == resource.RLIM_INFINITY:
                max_fd = 1024
            
            for fd in range(max_fd):
                try:
                    os.close(fd)
                except OSError:
                    pass
            
            # Redirect standard file descriptors
            with open(log_file, 'a') as log:
                os.dup2(log.fileno(), 0)  # stdin
                os.dup2(log.fileno(), 1)  # stdout  
                os.dup2(log.fileno(), 2)  # stderr
            
            # Update environment
            if env:
                os.environ.update(env)
            
            # Execute command and write PID
            try:
                process = subprocess.Popen(command)
                
                # Write PID to file
                with open(pid_file, 'w') as f:
                    f.write(str(process.pid))
                
                # Wait for process to complete
                process.wait()
                
            except Exception as e:
                logger.error(f"Daemon process failed: {e}")
            finally:
                # Clean up PID file
                try:
                    os.unlink(pid_file)
                except:
                    pass
                os._exit(0)
        
        else:
            # Parent process - wait for child to detach
            os.waitpid(pid, 0)
            
            # Give daemon time to write PID file
            max_wait = 3.0  # seconds
            wait_interval = 0.1
            waited = 0
            
            while waited < max_wait:
                try:
                    with open(pid_file, 'r') as f:
                        pid_str = f.read().strip()
                        if pid_str:
                            return int(pid_str)
                except (FileNotFoundError, ValueError):
                    pass
                
                time.sleep(wait_interval)
                waited += wait_interval
            
            # If we still can't read the PID, log but don't fail
            logger.warning(f"Could not read PID file for daemon {name}, using placeholder PID")
            # Return a placeholder PID and let monitoring handle it
            return -1
    
    def _start_regular_process(self,
                             name: str,
                             command: List[str],
                             log_file: Path,
                             env: Optional[Dict[str, str]],
                             working_dir: Optional[Path]) -> int:
        """Start a regular (non-daemon) process."""
        # Prepare environment
        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        
        # Start process
        with open(log_file, 'a') as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=process_env,
                cwd=working_dir
            )
        
        return process.pid
    
    def _send_signal(self, name: str, sig: signal.Signals, new_status: str) -> bool:
        """Send signal to process."""
        with self._process_lock:
            # Check persistent registry first
            record = self.registry.get_process(name)
            if not record:
                raise ValueError(f"Unknown process: {name}")
            
            # Convert to ProcessInfo for compatibility
            process_info = ProcessInfo(
                pid=record.pid,
                name=record.name,
                command=record.command,
                start_time=record.start_time,
                status=record.status,
                log_file=record.log_file,
                metadata=record.metadata
            )
            
            # Update in-memory cache
            self._processes[name] = process_info
            
            # Handle placeholder PID from daemon
            if process_info.pid == -1:
                pid_file = self.pid_dir / f"{name}.pid"
                try:
                    with open(pid_file, 'r') as f:
                        actual_pid = int(f.read().strip())
                        process_info.pid = actual_pid
                except (FileNotFoundError, ValueError):
                    logger.error(f"Cannot find PID for daemon process '{name}'")
                    return False
            
            try:
                process = psutil.Process(process_info.pid)
                process.send_signal(sig)
                process_info.status = new_status
                
                # Update persistent registry
                self.registry.update_process_status(name, new_status)
                
                logger.info(f"Sent {sig.name} to process '{name}'")
                return True
                
            except psutil.NoSuchProcess:
                logger.error(f"Process '{name}' not found")
                process_info.status = "stopped"
                self.registry.update_process_status(name, "stopped")
                return False
            except Exception as e:
                logger.error(f"Failed to send signal to '{name}': {e}")
                return False
    
    def _is_process_running(self, name: str) -> bool:
        """Check if process is running."""
        if name not in self._processes:
            return False
        
        process_info = self._processes[name]
        
        # Handle placeholder PID from daemon
        if process_info.pid == -1:
            # Try to read actual PID from file
            pid_file = self.pid_dir / f"{name}.pid"
            try:
                with open(pid_file, 'r') as f:
                    actual_pid = int(f.read().strip())
                    process_info.pid = actual_pid
            except (FileNotFoundError, ValueError):
                return False
        
        try:
            process = psutil.Process(process_info.pid)
            return process.is_running()
        except psutil.NoSuchProcess:
            return False
    
    def _update_process_info(self, name: str) -> None:
        """Update process information."""
        if name not in self._processes:
            return
        
        process_info = self._processes[name]
        
        # Handle placeholder PID from daemon
        if process_info.pid == -1:
            pid_file = self.pid_dir / f"{name}.pid"
            try:
                with open(pid_file, 'r') as f:
                    actual_pid = int(f.read().strip())
                    process_info.pid = actual_pid
            except (FileNotFoundError, ValueError):
                # PID file not ready yet, skip monitoring for now
                return
        
        try:
            process = psutil.Process(process_info.pid)
            
            # Update metrics
            process_info.cpu_percent = process.cpu_percent(interval=0.1)
            process_info.memory_mb = process.memory_info().rss / (1024 * 1024)
            
            # Check status
            if not process.is_running():
                process_info.status = "stopped"
            elif process.status() == psutil.STATUS_ZOMBIE:
                process_info.status = "zombie"
                
        except psutil.NoSuchProcess:
            process_info.status = "stopped"
    
    def _start_monitoring(self) -> None:
        """Start monitoring thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitor_thread.start()
    
    def _monitoring_loop(self) -> None:
        """Monitor process health and handle auto-restart."""
        while not self._stop_monitoring.is_set():
            try:
                with self._process_lock:
                    for name in list(self._processes.keys()):
                        self._update_process_info(name)
                        process_info = self._processes[name]
                        
                        # Check for crashed processes
                        if (process_info.status == "stopped" and 
                            self._auto_restart.get(name, False)):
                            
                            # Check restart attempts
                            attempts = self._restart_attempts.get(name, 0)
                            if attempts < self._max_restart_attempts:
                                logger.warning(f"Process '{name}' crashed, restarting...")
                                
                                # Wait before restart
                                time.sleep(self._restart_delay)
                                
                                try:
                                    # Restart process
                                    pid = self._start_regular_process(
                                        name,
                                        process_info.command,
                                        Path(process_info.log_file),
                                        None,
                                        None
                                    )
                                    
                                    process_info.pid = pid
                                    process_info.status = "running"
                                    process_info.start_time = time.time()
                                    self._restart_attempts[name] = attempts + 1
                                    
                                    logger.info(f"Restarted process '{name}' with PID {pid}")
                                    
                                except Exception as e:
                                    logger.error(f"Failed to restart process '{name}': {e}")
                            else:
                                logger.error(f"Process '{name}' exceeded restart attempts")
                                self._auto_restart[name] = False
                        
                        # Rotate logs if needed
                        if process_info.log_file:
                            self._check_log_rotation(process_info.log_file)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            
            # Wait for next check
            self._stop_monitoring.wait(self._monitor_interval)
    
    def _check_log_rotation(self, log_file: str) -> None:
        """Check and perform log rotation if needed."""
        log_path = Path(log_file)
        
        if not log_path.exists():
            return
        
        # Check size
        size_mb = log_path.stat().st_size / (1024 * 1024)
        if size_mb < self.max_log_size_mb:
            return
        
        # Rotate logs
        try:
            # Rename existing logs
            for i in range(self.max_log_files - 1, 0, -1):
                old_path = log_path.with_suffix(f'.log.{i}')
                new_path = log_path.with_suffix(f'.log.{i + 1}')
                if old_path.exists():
                    old_path.rename(new_path)
            
            # Rotate current log
            log_path.rename(log_path.with_suffix('.log.1'))
            
            # Create new empty log
            log_path.touch()
            
            logger.info(f"Rotated log file: {log_file}")
            
        except Exception as e:
            logger.error(f"Failed to rotate log {log_file}: {e}")
    
    def shutdown(self) -> None:
        """Shutdown process controller."""
        logger.info("Shutting down process controller")
        
        # Stop monitoring
        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        
        # Stop all processes
        with self._process_lock:
            for name in list(self._processes.keys()):
                try:
                    self.stop_process(name)
                except Exception as e:
                    logger.error(f"Failed to stop process '{name}': {e}")


# CLI interface
def main():
    """CLI interface for process controller."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Biodiversity Pipeline Process Controller")
    parser.add_argument('action', choices=['start', 'stop', 'pause', 'resume', 'status', 'list', 'tail'])
    parser.add_argument('--name', help='Process name')
    parser.add_argument('--command', nargs='+', help='Command to run')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    parser.add_argument('--no-restart', action='store_true', help='Disable auto-restart')
    parser.add_argument('--lines', type=int, default=100, help='Number of log lines to tail')
    
    args = parser.parse_args()
    
    # Initialize controller
    controller = ProcessController()
    
    try:
        if args.action == 'start':
            if not args.name or not args.command:
                parser.error("--name and --command required for start")
            
            pid = controller.start_process(
                args.name,
                args.command,
                daemon_mode=args.daemon,
                auto_restart=not args.no_restart
            )
            print(f"Started process '{args.name}' with PID {pid}")
            
        elif args.action == 'stop':
            if not args.name:
                parser.error("--name required for stop")
            
            if controller.stop_process(args.name):
                print(f"Stopped process '{args.name}'")
            else:
                print(f"Failed to stop process '{args.name}'")
                sys.exit(1)
                
        elif args.action == 'pause':
            if not args.name:
                parser.error("--name required for pause")
            
            if controller.pause_process(args.name):
                print(f"Paused process '{args.name}'")
            else:
                print(f"Failed to pause process '{args.name}'")
                sys.exit(1)
                
        elif args.action == 'resume':
            if not args.name:
                parser.error("--name required for resume")
            
            if controller.resume_process(args.name):
                print(f"Resumed process '{args.name}'")
            else:
                print(f"Failed to resume process '{args.name}'")
                sys.exit(1)
                
        elif args.action == 'status':
            if not args.name:
                parser.error("--name required for status")
            
            info = controller.get_process_status(args.name)
            print(f"Process: {info.name}")
            print(f"PID: {info.pid}")
            print(f"Status: {info.status}")
            print(f"CPU: {info.cpu_percent:.1f}%")
            print(f"Memory: {info.memory_mb:.1f} MB")
            print(f"Uptime: {time.time() - info.start_time:.0f} seconds")
            
        elif args.action == 'list':
            processes = controller.list_processes()
            if not processes:
                print("No managed processes")
            else:
                print(f"{'Name':<20} {'PID':<10} {'Status':<10} {'CPU%':<8} {'Memory(MB)':<12}")
                print("-" * 60)
                for info in processes:
                    print(f"{info.name:<20} {info.pid:<10} {info.status:<10} "
                          f"{info.cpu_percent:<8.1f} {info.memory_mb:<12.1f}")
                    
        elif args.action == 'tail':
            if not args.name:
                parser.error("--name required for tail")
            
            lines = controller.tail_log(args.name, args.lines)
            for line in lines:
                print(line)
                
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()