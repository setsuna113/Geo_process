# src/core/process_controller_enhanced.py
"""Enhanced process controller with structured logging integration."""

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

from .process_controller import ProcessController, ProcessInfo
from .process_registry import ProcessRegistry, ProcessRecord
from src.infrastructure.logging import get_logger, log_operation, experiment_context

logger = get_logger(__name__)


class EnhancedProcessController(ProcessController):
    """
    Enhanced process controller with structured logging and monitoring.
    
    Adds:
    - Structured logging for all operations
    - Automatic error capture with tracebacks
    - Context propagation to daemon processes
    - Performance metrics logging
    """
    
    def __init__(self, 
                 pid_dir: Optional[Path] = None,
                 log_dir: Optional[Path] = None,
                 max_log_size_mb: float = 100,
                 max_log_files: int = 5,
                 experiment_id: Optional[str] = None):
        """
        Initialize enhanced process controller.
        
        Args:
            pid_dir: Directory for PID files
            log_dir: Directory for log files
            max_log_size_mb: Maximum log file size before rotation
            max_log_files: Maximum number of rotated log files
            experiment_id: Current experiment ID for context
        """
        super().__init__(pid_dir, log_dir, max_log_size_mb, max_log_files)
        
        self.experiment_id = experiment_id
        if experiment_id:
            experiment_context.set(experiment_id)
            logger.info(f"Process controller initialized for experiment: {experiment_id}")
    
    @log_operation("start_process")
    def start_process(self,
                     name: str,
                     command: List[str],
                     daemon_mode: bool = False,
                     auto_restart: bool = True,
                     max_restarts: Optional[int] = None,
                     env: Optional[Dict[str, str]] = None,
                     working_dir: Optional[Path] = None) -> int:
        """
        Start a managed process with enhanced logging.
        
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
        logger.info(
            f"Starting process '{name}'",
            extra={
                'context': {
                    'process_name': name,
                    'daemon_mode': daemon_mode,
                    'auto_restart': auto_restart,
                    'command': ' '.join(command[:3]) + '...' if len(command) > 3 else ' '.join(command)
                }
            }
        )
        
        try:
            # Add experiment ID to environment
            if env is None:
                env = {}
            if self.experiment_id:
                env['EXPERIMENT_ID'] = self.experiment_id
            
            # Call parent implementation
            pid = super().start_process(
                name=name,
                command=command,
                daemon_mode=daemon_mode,
                auto_restart=auto_restart,
                max_restarts=max_restarts,
                env=env,
                working_dir=working_dir
            )
            
            logger.info(
                f"Successfully started process '{name}' with PID {pid}",
                extra={
                    'context': {
                        'process_name': name,
                        'pid': pid,
                        'log_file': str(self.log_dir / f"{name}.log")
                    }
                }
            )
            
            return pid
            
        except Exception as e:
            logger.error(
                f"Failed to start process '{name}'",
                exc_info=True,
                extra={
                    'context': {
                        'process_name': name,
                        'error_type': type(e).__name__
                    }
                }
            )
            raise
    
    @log_operation("stop_process")
    def stop_process(self, name: str, timeout: float = 30.0) -> bool:
        """Stop a managed process with logging."""
        logger.info(f"Stopping process '{name}' with timeout {timeout}s")
        
        try:
            result = super().stop_process(name, timeout)
            
            if result:
                logger.info(f"Successfully stopped process '{name}'")
            else:
                logger.warning(f"Failed to stop process '{name}' gracefully")
                
            return result
            
        except Exception as e:
            logger.error(
                f"Error stopping process '{name}'",
                exc_info=True,
                extra={'context': {'process_name': name}}
            )
            return False
    
    def _start_daemon_process(self,
                            name: str,
                            command: List[str],
                            log_file: Path,
                            env: Optional[Dict[str, str]],
                            working_dir: Optional[Path]) -> int:
        """Start a process as daemon with structured logging setup."""
        pid_file = self.pid_dir / f"{name}.pid"
        
        # Fork process to create daemon
        pid = os.fork()
        
        if pid == 0:
            # Child process - become daemon
            try:
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
                
                # Setup structured logging for daemon
                # This will be the first line in the daemon's log
                print(f"=== Daemon process starting at {datetime.utcnow().isoformat()} ===")
                print(f"Process: {name}")
                print(f"PID: {os.getpid()}")
                print(f"Experiment ID: {env.get('EXPERIMENT_ID', 'None')}")
                print(f"Command: {' '.join(command)}")
                print("=" * 60)
                
                # Add structured logging setup to command
                # Inject our logging setup before the actual command
                enhanced_command = [
                    sys.executable, "-c",
                    f"""
import sys
import os

# Setup structured logging for daemon
try:
    from src.infrastructure.logging import setup_daemon_logging
    setup_daemon_logging(
        '{name}',
        '{log_file}',
        '{env.get('EXPERIMENT_ID', '')}'
    )
except Exception as e:
    print(f"Failed to setup structured logging: {{e}}")

# Now execute the original command
{self._build_daemon_wrapper(command)}
"""
                ]
                
                # Execute enhanced command
                process = subprocess.Popen(enhanced_command)
                
                # Write PID to file
                with open(pid_file, 'w') as f:
                    f.write(str(process.pid))
                
                # Wait for process to complete
                exit_code = process.wait()
                
                # Log daemon exit
                print(f"=== Daemon process exiting at {datetime.utcnow().isoformat()} ===")
                print(f"Exit code: {exit_code}")
                
            except Exception as e:
                # Log any daemon startup errors
                print(f"DAEMON ERROR: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
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
            max_wait = 3.0
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
            
            # Log warning but continue
            logger.warning(f"Could not read PID file for daemon {name}")
            return -1
    
    def _build_daemon_wrapper(self, command: List[str]) -> str:
        """Build Python code to execute the original command."""
        if command[0] == sys.executable and len(command) > 2 and command[1] == "-c":
            # Python command, return the code directly
            return command[2]
        else:
            # Other command, use subprocess
            return f"""
import subprocess
subprocess.run({command!r})
"""
    
    def _monitoring_loop(self) -> None:
        """Enhanced monitoring loop with structured logging."""
        while not self._stop_monitoring.is_set():
            try:
                with self._process_lock:
                    for name in list(self._processes.keys()):
                        self._update_process_info(name)
                        process_info = self._processes[name]
                        
                        # Log resource usage periodically
                        if process_info.status == "running":
                            logger.debug(
                                f"Process '{name}' resource usage",
                                extra={
                                    'performance': {
                                        'process_name': name,
                                        'cpu_percent': process_info.cpu_percent,
                                        'memory_mb': process_info.memory_mb,
                                        'uptime_seconds': time.time() - process_info.start_time
                                    }
                                }
                            )
                        
                        # Check for crashed processes
                        if (process_info.status == "stopped" and 
                            self._auto_restart.get(name, False)):
                            
                            attempts = self._restart_attempts.get(name, 0)
                            
                            logger.error(
                                f"Process '{name}' crashed unexpectedly",
                                extra={
                                    'context': {
                                        'process_name': name,
                                        'pid': process_info.pid,
                                        'restart_attempts': attempts,
                                        'max_attempts': self._max_restart_attempts,
                                        'will_restart': attempts < self._max_restart_attempts
                                    }
                                }
                            )
                            
                            # Check restart attempts
                            if attempts < self._max_restart_attempts:
                                logger.info(f"Attempting to restart process '{name}' (attempt {attempts + 1})")
                                
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
                                    
                                    logger.info(
                                        f"Successfully restarted process '{name}'",
                                        extra={
                                            'context': {
                                                'process_name': name,
                                                'new_pid': pid,
                                                'attempt': attempts + 1
                                            }
                                        }
                                    )
                                    
                                except Exception as e:
                                    logger.error(
                                        f"Failed to restart process '{name}'",
                                        exc_info=True,
                                        extra={
                                            'context': {
                                                'process_name': name,
                                                'attempt': attempts + 1
                                            }
                                        }
                                    )
                            else:
                                logger.error(
                                    f"Process '{name}' exceeded restart attempts, giving up",
                                    extra={
                                        'context': {
                                            'process_name': name,
                                            'total_attempts': attempts
                                        }
                                    }
                                )
                                self._auto_restart[name] = False
                        
                        # Rotate logs if needed
                        if process_info.log_file:
                            self._check_log_rotation(process_info.log_file)
                
            except Exception as e:
                logger.error("Monitoring loop error", exc_info=True)
            
            # Wait for next check
            self._stop_monitoring.wait(self._monitor_interval)