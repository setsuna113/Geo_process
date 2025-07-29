"""Comprehensive cleanup service for fresh pipeline starts."""

import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import psutil
import signal

from src.config import config
from src.database.connection import DatabaseManager
from src.core.process_registry import ProcessRegistry
from src.checkpoints import get_checkpoint_manager

logger = logging.getLogger(__name__)


class CleanupService:
    """Service for comprehensive cleanup operations."""
    
    def __init__(self):
        self.config = config
        self.db = DatabaseManager()
        
        # Initialize process registry with proper directory
        registry_dir = Path.home() / '.biodiversity' / 'process_registry'
        self.process_registry = ProcessRegistry(registry_dir)
        self.checkpoint_manager = get_checkpoint_manager()
    
    def clean_all(self, experiment_name_pattern: str = None, force: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive cleanup for fresh pipeline starts.
        
        Args:
            experiment_name_pattern: Pattern to match experiment names (optional)
            force: Force cleanup even if processes are running
            
        Returns:
            Dictionary with cleanup results
        """
        results = {
            'processes_cleaned': 0,
            'checkpoints_cleaned': 0,
            'temp_files_cleaned': 0,
            'database_cleaned': False,
            'errors': []
        }
        
        logger.info("Starting comprehensive cleanup...")
        
        try:
            # 1. Clean up running processes
            process_results = self.cleanup_processes(experiment_name_pattern, force)
            results['processes_cleaned'] = process_results['stopped_count']
            if process_results['errors']:
                results['errors'].extend(process_results['errors'])
            
            # 2. Clean up checkpoints
            checkpoint_results = self.cleanup_checkpoints(experiment_name_pattern)
            results['checkpoints_cleaned'] = checkpoint_results['cleaned_count']
            if checkpoint_results['errors']:
                results['errors'].extend(checkpoint_results['errors'])
            
            # 3. Clean up temporary files
            temp_results = self.cleanup_temp_files()
            results['temp_files_cleaned'] = temp_results['cleaned_count']
            if temp_results['errors']:
                results['errors'].extend(temp_results['errors'])
            
            # 4. Clean up database (if specified)
            if experiment_name_pattern:
                db_results = self.cleanup_database_experiments(experiment_name_pattern)
                results['database_cleaned'] = db_results['success']
                if db_results['errors']:
                    results['errors'].extend(db_results['errors'])
            
            logger.info(f"Cleanup completed: {results}")
            return results
            
        except Exception as e:
            error_msg = f"Cleanup failed: {e}"
            logger.error(error_msg)
            results['errors'].append(error_msg)
            return results
    
    def cleanup_processes(self, name_pattern: str = None, force: bool = True) -> Dict[str, Any]:
        """Clean up running processes."""
        results = {
            'stopped_count': 0,
            'running_processes': [],
            'errors': []
        }
        
        try:
            # Get running processes from registry
            processes = self.process_registry.list_processes()
            
            for process_info in processes:
                process_name = process_info.get('name', 'unknown')
                
                # Skip if pattern specified and doesn't match
                if name_pattern and name_pattern not in process_name:
                    continue
                
                try:
                    pid = process_info.get('pid')
                    if pid and psutil.pid_exists(pid):
                        if force:
                            # Terminate process
                            try:
                                process = psutil.Process(pid)
                                process.terminate()
                                
                                # Wait for graceful termination
                                try:
                                    process.wait(timeout=10)
                                except psutil.TimeoutExpired:
                                    # Force kill if necessary
                                    process.kill()
                                
                                results['stopped_count'] += 1
                                logger.info(f"Stopped process: {process_name} (PID: {pid})")
                                
                            except psutil.NoSuchProcess:
                                # Process already stopped
                                pass
                            except Exception as e:
                                error_msg = f"Failed to stop process {process_name}: {e}"
                                results['errors'].append(error_msg)
                                logger.error(error_msg)
                        else:
                            results['running_processes'].append(process_info)
                    
                    # Clean up process registry entry
                    self.process_registry.cleanup_process(process_name)
                    
                except Exception as e:
                    error_msg = f"Error processing {process_name}: {e}"
                    results['errors'].append(error_msg)
                    logger.error(error_msg)
            
        except Exception as e:
            error_msg = f"Process cleanup failed: {e}"
            results['errors'].append(error_msg)
            logger.error(error_msg)
        
        return results
    
    def cleanup_checkpoints(self, experiment_pattern: str = None) -> Dict[str, Any]:
        """Clean up checkpoint files."""
        results = {
            'cleaned_count': 0,
            'errors': []
        }
        
        try:
            # Get checkpoint directories
            checkpoint_dirs = []
            
            if hasattr(self.checkpoint_manager, 'get_checkpoint_dirs'):
                checkpoint_dirs = self.checkpoint_manager.get_checkpoint_dirs()
            else:
                # Fallback to default checkpoint directory
                checkpoint_dir = Path(self.config.get('checkpoints.base_dir', 'checkpoint_outputs'))
                if checkpoint_dir.exists():
                    checkpoint_dirs = [checkpoint_dir]
            
            for checkpoint_dir in checkpoint_dirs:
                if not checkpoint_dir.exists():
                    continue
                
                # Clean checkpoint files
                for checkpoint_file in checkpoint_dir.rglob('*.checkpoint'):
                    try:
                        # Check if matches pattern
                        if experiment_pattern and experiment_pattern not in checkpoint_file.name:
                            continue
                        
                        checkpoint_file.unlink()
                        results['cleaned_count'] += 1
                        logger.debug(f"Removed checkpoint: {checkpoint_file}")
                        
                    except Exception as e:
                        error_msg = f"Failed to remove checkpoint {checkpoint_file}: {e}"
                        results['errors'].append(error_msg)
                        logger.error(error_msg)
                
                # Clean empty directories
                self._cleanup_empty_dirs(checkpoint_dir)
            
        except Exception as e:
            error_msg = f"Checkpoint cleanup failed: {e}"
            results['errors'].append(error_msg)
            logger.error(error_msg)
        
        return results
    
    def cleanup_temp_files(self) -> Dict[str, Any]:
        """Clean up temporary files."""
        results = {
            'cleaned_count': 0,
            'errors': []
        }
        
        temp_dirs = [
            Path('/tmp'),
            Path(self.config.get('processing.temp_dir', '/tmp/biodiversity')),
            Path(self.config.get('raster_processing.cache_dir', 'cache')),
        ]
        
        # Add any additional temp directories from config
        additional_temp = self.config.get('cleanup.temp_directories', [])
        temp_dirs.extend(Path(d) for d in additional_temp)
        
        for temp_dir in temp_dirs:
            if not temp_dir.exists():
                continue
            
            try:
                patterns = ['*.tmp', '*.temp', 'temp_*', 'biodiversity_*']
                
                for pattern in patterns:
                    for temp_file in temp_dir.glob(pattern):
                        try:
                            if temp_file.is_file():
                                # Check file age (only clean old temp files)
                                file_age = time.time() - temp_file.stat().st_mtime
                                max_age = self.config.get('cleanup.temp_file_max_age_hours', 24) * 3600
                                
                                if file_age > max_age:
                                    temp_file.unlink()
                                    results['cleaned_count'] += 1
                                    logger.debug(f"Removed temp file: {temp_file}")
                            
                            elif temp_file.is_dir():
                                # Remove empty temp directories
                                try:
                                    temp_file.rmdir()
                                    results['cleaned_count'] += 1
                                    logger.debug(f"Removed empty temp dir: {temp_file}")
                                except OSError:
                                    # Directory not empty, skip
                                    pass
                        
                        except Exception as e:
                            error_msg = f"Failed to remove temp file {temp_file}: {e}"
                            results['errors'].append(error_msg)
                            logger.error(error_msg)
                
            except Exception as e:
                error_msg = f"Temp cleanup failed for {temp_dir}: {e}"
                results['errors'].append(error_msg)
                logger.error(error_msg)
        
        return results
    
    def cleanup_database_experiments(self, experiment_pattern: str) -> Dict[str, Any]:
        """Clean up database experiments matching pattern."""
        results = {
            'success': False,
            'cleaned_experiments': 0,
            'errors': []
        }
        
        try:
            with self.db.get_cursor() as cursor:
                # Find matching experiments
                cursor.execute("""
                    SELECT id, name FROM experiments 
                    WHERE name LIKE %s
                """, (f"%{experiment_pattern}%",))
                
                experiments = cursor.fetchall()
                
                for exp in experiments:
                    try:
                        exp_id = exp['id']
                        exp_name = exp['name']
                        
                        # Delete related data (processing_jobs, etc.)
                        cursor.execute("DELETE FROM processing_jobs WHERE parent_experiment_id = %s", (exp_id,))
                        
                        # Delete experiment
                        cursor.execute("DELETE FROM experiments WHERE id = %s", (exp_id,))
                        
                        results['cleaned_experiments'] += 1
                        logger.info(f"Cleaned experiment: {exp_name}")
                        
                    except Exception as e:
                        error_msg = f"Failed to clean experiment {exp['name']}: {e}"
                        results['errors'].append(error_msg)
                        logger.error(error_msg)
                
                results['success'] = True
                
        except Exception as e:
            error_msg = f"Database cleanup failed: {e}"
            results['errors'].append(error_msg)
            logger.error(error_msg)
        
        return results
    
    def _cleanup_empty_dirs(self, directory: Path) -> None:
        """Recursively remove empty directories."""
        try:
            for item in directory.iterdir():
                if item.is_dir():
                    self._cleanup_empty_dirs(item)
                    try:
                        item.rmdir()  # Will only succeed if empty
                        logger.debug(f"Removed empty directory: {item}")
                    except OSError:
                        # Directory not empty, which is fine
                        pass
        except Exception as e:
            logger.debug(f"Error cleaning empty dirs in {directory}: {e}")
    
    def get_cleanup_status(self) -> Dict[str, Any]:
        """Get current cleanup status and recommendations."""
        status = {
            'running_processes': 0,
            'checkpoint_files': 0,
            'temp_files': 0,
            'experiments_count': 0,
            'recommendations': []
        }
        
        try:
            # Count running processes
            processes = self.process_registry.list_processes()
            status['running_processes'] = len([p for p in processes if p.get('status') == 'running'])
            
            # Count checkpoint files
            checkpoint_dir = Path(self.config.get('checkpoints.base_dir', 'checkpoints'))
            if checkpoint_dir.exists():
                status['checkpoint_files'] = len(list(checkpoint_dir.rglob('*.checkpoint')))
            
            # Count temp files (approximate)
            temp_dir = Path(self.config.get('processing.temp_dir', '/tmp/biodiversity'))
            if temp_dir.exists():
                status['temp_files'] = len(list(temp_dir.glob('*')))
            
            # Count experiments
            with self.db.get_cursor() as cursor:
                cursor.execute("SELECT COUNT(*) as count FROM experiments")
                status['experiments_count'] = cursor.fetchone()['count']
            
            # Generate recommendations
            if status['running_processes'] > 0:
                status['recommendations'].append("Stop running processes before starting new experiments")
            
            if status['checkpoint_files'] > 100:
                status['recommendations'].append("Consider cleaning old checkpoint files")
            
            if status['temp_files'] > 50:
                status['recommendations'].append("Clean temporary files to free disk space")
            
        except Exception as e:
            logger.error(f"Failed to get cleanup status: {e}")
        
        return status


# For backward compatibility
CleanupManager = CleanupService