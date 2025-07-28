"""Stage skip control with safety checks."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class StageSkipController:
    """Controls whether pipeline stages can be safely skipped."""
    
    def __init__(self, config, db):
        self.config = config
        self.db = db
        
    def can_skip_stage(self, stage_name: str, experiment_name: str, 
                      force_fresh: bool = False) -> Tuple[bool, str]:
        """
        Determine if a stage can be safely skipped.
        
        Args:
            stage_name: Name of the stage
            experiment_name: Current experiment name
            force_fresh: Force fresh processing
            
        Returns:
            (can_skip, reason) tuple
        """
        # Never skip if force_fresh
        if force_fresh:
            return False, "Force fresh processing requested"
            
        # Check global config
        if not self.config.get('pipeline.allow_skip_stages', False):
            return False, "Stage skipping disabled in config"
            
        # Check stage-specific config
        stage_config = self.config.get(f'pipeline.stages.{stage_name}', {})
        if not stage_config.get('skip_if_exists', False):
            return False, f"Stage {stage_name} skip disabled"
            
        # Special handling for different stages
        if stage_name == 'data_load':
            return self._check_data_load_skip()
        elif stage_name == 'resample':
            return self._check_resample_skip()
        else:
            return False, f"Stage {stage_name} cannot be skipped"
    
    def _check_resample_skip(self) -> Tuple[bool, str]:
        """Check if resample stage can be skipped."""
        from src.database.schema import schema
        
        # Get existing resampled datasets
        existing = schema.get_resampled_datasets()
        if not existing:
            return False, "No existing resampled data found"
            
        # Check if we have all expected datasets
        expected_datasets = self.config.get('datasets.target_datasets', [])
        existing_names = {d['name'] for d in existing}
        expected_names = {d['name'] for d in expected_datasets if d.get('enabled', True)}
        
        if expected_names != existing_names:
            missing = expected_names - existing_names
            return False, f"Missing datasets: {missing}"
            
        # Check data freshness
        max_age_hours = self.config.get('pipeline.data_validation.max_age_hours', 24)
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        for dataset in existing:
            created_at = dataset['created_at']
            if created_at < cutoff_time:
                return False, f"Dataset {dataset['name']} is older than {max_age_hours} hours"
                
            # Check if source file is newer than processed data
            if self.config.get('pipeline.data_validation.check_source_timestamps', True):
                source_path = Path(dataset['source_path'])
                if source_path.exists():
                    source_mtime = datetime.fromtimestamp(source_path.stat().st_mtime)
                    if source_mtime > created_at:
                        return False, f"Source file {source_path} modified after processing"
        
        # Check data integrity
        for dataset in existing:
            table_name = dataset.get('data_table_name')
            if table_name:
                # Verify table exists and has data
                with self.db.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(f"""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables 
                                WHERE table_name = %s
                            )
                        """, (table_name,))
                        if not cur.fetchone()[0]:
                            return False, f"Data table {table_name} missing"
                            
                        cur.execute(f"SELECT COUNT(*) FROM {table_name} LIMIT 1")
                        if cur.fetchone()[0] == 0:
                            return False, f"Data table {table_name} is empty"
        
        return True, "All checks passed - safe to skip resample"
    
    def _check_data_load_skip(self) -> Tuple[bool, str]:
        """Check if data load stage can be skipped."""
        # For data_load, we mainly check if the files still exist
        datasets = self.config.get('datasets.target_datasets', [])
        
        for dataset in datasets:
            if not dataset.get('enabled', True):
                continue
                
            path = Path(dataset['path'])
            if not path.exists():
                return False, f"Source file {path} not found"
                
        return True, "All source files exist"
    
    def _check_db_status(self, dataset_name: str) -> str:
        """Check if dataset exists in DB and its status."""
        try:
            from src.database.schema import schema
            
            # Check passthrough table
            table_name = f"passthrough_{dataset_name.replace('-', '_')}"
            
            with self.db.get_cursor() as cursor:
                # Check if table exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_name = %s
                    )
                """, (table_name,))
                
                if not cursor.fetchone()[0]:
                    return "missing"
                
                # Check row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                
                if count == 0:
                    return "empty"
                
                # Check metadata
                datasets = schema.get_resampled_datasets({'name': dataset_name})
                if not datasets:
                    return "incomplete"  # Table exists but no metadata
                    
                return "complete"
                
        except Exception as e:
            logger.error(f"Error checking DB status: {e}")
            return "error"
    
    def can_skip_dataset_processing(self, dataset_name: str) -> Tuple[bool, str]:
        """Check if individual dataset processing can be skipped based on DB status."""
        db_status = self._check_db_status(dataset_name)
        
        if db_status == "complete":
            return True, f"Dataset {dataset_name} already complete in DB"
        elif db_status in ["error", "incomplete"]:
            return False, f"DB status {db_status}, cleaning and reprocessing needed"
        elif db_status == "empty":
            return False, f"Dataset {dataset_name} table is empty"
        elif db_status == "missing":
            return False, f"Dataset {dataset_name} not found in DB"
        else:
            return False, f"Unknown DB status: {db_status}"
    
    def cleanup_partial_data(self, dataset_name: str):
        """Clean up partial data for a dataset."""
        try:
            from src.database.schema import schema
            
            # Remove from resampled_datasets table
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    # Get table name before deletion
                    cur.execute("SELECT data_table_name FROM resampled_datasets WHERE name = %s", (dataset_name,))
                    result = cur.fetchone()
                    table_name = result[0] if result else None
                    
                    # Delete metadata
                    cur.execute("DELETE FROM resampled_datasets WHERE name = %s", (dataset_name,))
                    
                    # Drop data table if it exists
                    if table_name:
                        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
                        
                    conn.commit()
                    logger.info(f"Cleaned up partial data for {dataset_name}")
                    
        except Exception as e:
            logger.error(f"Failed to clean up partial data for {dataset_name}: {e}")
        
    def log_skip_decision(self, stage_name: str, can_skip: bool, reason: str):
        """Log the skip decision for audit trail."""
        if can_skip:
            logger.info(f"""
            ╔══════════════════════════════════════════════════════════╗
            ║ SKIPPING STAGE: {stage_name:<40} ║
            ║ Reason: {reason:<48} ║
            ║ Using existing data from database                        ║
            ╚══════════════════════════════════════════════════════════╝
            """)
        else:
            logger.info(f"Cannot skip {stage_name}: {reason}")