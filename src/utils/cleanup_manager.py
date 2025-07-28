#!/usr/bin/env python3
"""Comprehensive cleanup manager for fresh pipeline starts."""

import logging
import os
import time
from pathlib import Path
from typing import List, Optional
import psutil
import signal

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import config
from src.database.connection import DatabaseManager
from src.core.process_registry import ProcessRegistry
from src.checkpoints import get_checkpoint_manager

logger = logging.getLogger(__name__)


class CleanupManager:
    """Manages comprehensive cleanup for fresh pipeline starts."""
    
    def __init__(self):
        self.config = config
        self.db = DatabaseManager()
        
        # Initialize process registry with proper directory
        registry_dir = Path.home() / '.biodiversity' / 'process_registry'
        self.process_registry = ProcessRegistry(registry_dir)
        self.checkpoint_manager = get_checkpoint_manager()
        
    def clean_all(self, experiment_name_pattern: str = None, force: bool = True) -> bool:
        """
        Perform comprehensive cleanup of all pipeline state.
        
        Args:
            experiment_name_pattern: Pattern to match experiment names (default: all)
            force: Force cleanup even if processes are running
            
        Returns:
            True if cleanup successful
        """
        logger.info("🧹 Starting comprehensive pipeline cleanup...")
        
        try:
            # 1. Stop running processes
            if force:
                self._stop_pipeline_processes()
            
            # 2. Clean process registry
            self._clean_process_registry()
            
            # 3. Clean database state
            self._clean_database_state(experiment_name_pattern)
            
            # 4. Clean checkpoints
            self._clean_checkpoints(experiment_name_pattern)
            
            # 5. Clean temporary files
            self._clean_temp_files()
            
            # 6. Clean log files (optional)
            self._clean_old_logs()
            
            logger.info("✅ Comprehensive cleanup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            return False
    
    def _stop_pipeline_processes(self) -> None:
        """Stop all running pipeline processes."""
        logger.info("🔄 Stopping running pipeline processes...")
        
        # Get all pipeline-related processes
        pipeline_patterns = [
            'process_manager.py',
            'run_pipeline.sh',
            'unified_resampling',
            'pipeline',
            'resampling_processor'
        ]
        
        killed_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if any(pattern in cmdline for pattern in pipeline_patterns):
                    pid = proc.info['pid']
                    if pid != os.getpid():  # Don't kill ourselves
                        logger.info(f"  Killing process {pid}: {proc.info['name']}")
                        proc.terminate()
                        killed_processes.append(pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Wait for graceful termination
        if killed_processes:
            time.sleep(2)
            
            # Force kill stubborn processes
            for pid in killed_processes:
                try:
                    proc = psutil.Process(pid)
                    if proc.is_running():
                        logger.warning(f"  Force killing stubborn process {pid}")
                        proc.kill()
                except psutil.NoSuchProcess:
                    pass
        
        # Clean tmux sessions
        self._clean_tmux_sessions()
    
    def _clean_tmux_sessions(self) -> None:
        """Clean up tmux sessions related to pipeline."""
        import subprocess
        
        try:
            # List tmux sessions
            result = subprocess.run(['tmux', 'ls'], capture_output=True, text=True)
            if result.returncode == 0:
                sessions = []
                for line in result.stdout.split('\n'):
                    if line.strip():
                        session_name = line.split(':')[0]
                        if any(pattern in session_name for pattern in 
                               ['pipeline', 'geo_', 'biodiversity', 'debug_', 'final_']):
                            sessions.append(session_name)
                
                for session in sessions:
                    logger.info(f"  Killing tmux session: {session}")
                    subprocess.run(['tmux', 'kill-session', '-t', session], 
                                 capture_output=True)
        except Exception as e:
            logger.debug(f"Tmux cleanup failed (may not be available): {e}")
    
    def _clean_process_registry(self) -> None:
        """Clean up process registry files."""
        logger.info("📝 Cleaning process registry...")
        
        try:
            # Clean stale processes first
            self.process_registry.cleanup_stale_processes()
            
            # Remove all registry files for fresh start
            registry_dir = Path(self.process_registry.registry_dir)
            if registry_dir.exists():
                for file in registry_dir.glob("*.json"):
                    logger.debug(f"  Removing registry file: {file}")
                    file.unlink(missing_ok=True)
                    
        except Exception as e:
            logger.warning(f"Process registry cleanup failed: {e}")
    
    def _clean_database_state(self, experiment_pattern: str = None) -> None:
        """Clean up database state for fresh start."""
        logger.info("🗄️  Cleaning database state...")
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    # Clean experiments
                    if experiment_pattern:
                        cur.execute("DELETE FROM experiments WHERE name LIKE %s", 
                                  (f"{experiment_pattern}%",))
                        deleted_exp = cur.rowcount
                        logger.info(f"  Deleted {deleted_exp} experiments matching '{experiment_pattern}'")
                    else:
                        # Clean all experiments older than 1 hour for safety
                        cur.execute("""
                            DELETE FROM experiments 
                            WHERE created_at < NOW() - INTERVAL '1 hour'
                        """)
                        deleted_exp = cur.rowcount
                        logger.info(f"  Deleted {deleted_exp} old experiments")
                    
                    # Clean orphaned resampled datasets
                    cur.execute("SELECT name, data_table_name FROM resampled_datasets")
                    datasets = cur.fetchall()
                    
                    for name, table_name in datasets:
                        if table_name:
                            # Check if table exists
                            cur.execute("""
                                SELECT EXISTS (
                                    SELECT FROM information_schema.tables 
                                    WHERE table_name = %s
                                )
                            """, (table_name,))
                            exists = cur.fetchone()[0]
                            
                            if not exists:
                                logger.info(f"  Cleaning orphaned dataset: {name}")
                                cur.execute("DELETE FROM resampled_datasets WHERE name = %s", (name,))
                    
                    # Optional: Clean all resampled data for fresh start
                    if experiment_pattern and 'fresh' in experiment_pattern.lower():
                        logger.info("  Fresh start requested - cleaning all resampled data")
                        cur.execute("SELECT data_table_name FROM resampled_datasets")
                        tables = [row[0] for row in cur.fetchall() if row[0]]
                        
                        cur.execute("TRUNCATE TABLE resampled_datasets")
                        
                        for table in tables:
                            try:
                                cur.execute(f"DROP TABLE IF EXISTS {table}")
                                logger.debug(f"    Dropped table: {table}")
                            except Exception as e:
                                logger.warning(f"    Failed to drop {table}: {e}")
                    
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Database cleanup failed: {e}")
            raise
    
    def _clean_checkpoints(self, experiment_pattern: str = None) -> None:
        """Clean up checkpoint files."""
        logger.info("💾 Cleaning checkpoints...")
        
        try:
            if experiment_pattern:
                # Clean specific experiment checkpoints
                checkpoint_dir = Path.home() / '.biodiversity' / 'checkpoints'
                if checkpoint_dir.exists():
                    pattern = f"{experiment_pattern}*"
                    for file in checkpoint_dir.glob(pattern):
                        logger.debug(f"  Removing checkpoint: {file}")
                        file.unlink(missing_ok=True)
            else:
                # Clean old checkpoints (older than 2 hours)
                self.checkpoint_manager.cleanup_old_checkpoints(
                    max_age_hours=2,
                    keep_latest=5
                )
                
        except Exception as e:
            logger.warning(f"Checkpoint cleanup failed: {e}")
    
    def _clean_temp_files(self) -> None:
        """Clean up temporary files."""
        logger.info("🗂️  Cleaning temporary files...")
        
        temp_patterns = [
            '/tmp/biodiversity_*',
            '/tmp/geo_*',
            '/tmp/raster_*',
            '/tmp/pipeline_*'
        ]
        
        import glob
        for pattern in temp_patterns:
            for file_path in glob.glob(pattern):
                try:
                    path = Path(file_path)
                    if path.exists():
                        if path.is_file():
                            path.unlink()
                        elif path.is_dir():
                            import shutil
                            shutil.rmtree(path)
                        logger.debug(f"  Removed temp: {file_path}")
                except Exception as e:
                    logger.debug(f"  Failed to remove {file_path}: {e}")
    
    def _clean_old_logs(self, max_age_hours: int = 24) -> None:
        """Clean up old log files."""
        logger.info(f"📋 Cleaning logs older than {max_age_hours} hours...")
        
        log_dir = Path.home() / '.biodiversity' / 'logs'
        if not log_dir.exists():
            return
        
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        for log_file in log_dir.glob("*.log"):
            try:
                if log_file.stat().st_mtime < cutoff_time:
                    logger.debug(f"  Removing old log: {log_file}")
                    log_file.unlink()
            except Exception as e:
                logger.debug(f"  Failed to remove log {log_file}: {e}")


def cleanup_for_fresh_start(experiment_name: str) -> bool:
    """
    Convenience function for fresh pipeline start.
    
    Args:
        experiment_name: Name of the experiment to clean
        
    Returns:
        True if cleanup successful
    """
    cleanup_manager = CleanupManager()
    return cleanup_manager.clean_all(experiment_name, force=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline cleanup utility")
    parser.add_argument("--experiment", help="Experiment name pattern to clean")
    parser.add_argument("--force", action="store_true", help="Force cleanup")
    parser.add_argument("--fresh", action="store_true", help="Complete fresh start")
    
    args = parser.parse_args()
    
    cleanup_manager = CleanupManager()
    pattern = args.experiment
    if args.fresh:
        pattern = f"{pattern}_fresh" if pattern else "fresh"
    
    success = cleanup_manager.clean_all(pattern, args.force)
    exit(0 if success else 1)