"""Experiment tracking operations for database schema."""

from typing import Dict, Any, List, Optional
import json
import logging
from datetime import datetime
from ..connection import db
from ..exceptions import handle_database_error, safe_fetch_id, DatabaseDuplicateError

logger = logging.getLogger(__name__)


class ExperimentTracking:
    """Experiment and job tracking database operations."""
    
    def create_experiment(self, name: str, description: str, config: Dict) -> str:
        """Create new experiment, handling duplicates gracefully."""
        with db.get_cursor() as cursor:
            try:
                cursor.execute("""
                    INSERT INTO experiments (name, description, config, created_by)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (name, description, json.dumps(config), config.get('created_by', 'system')))
                experiment_id = safe_fetch_id(cursor, "create_experiment")
                logger.info(f"✅ Created experiment '{name}': {experiment_id}")
                return experiment_id
            except DatabaseDuplicateError:
                # Handle duplicate experiment name by appending timestamp
                import time
                timestamp = int(time.time())
                new_name = f"{name}_{timestamp}"
                logger.warning(f"Experiment '{name}' exists, creating as '{new_name}'")
                cursor.execute("""
                    INSERT INTO experiments (name, description, config, created_by)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (new_name, description, json.dumps(config), config.get('created_by', 'system')))
                experiment_id = safe_fetch_id(cursor, "create_experiment_with_timestamp")
                logger.info(f"✅ Created experiment '{new_name}': {experiment_id}")
                return experiment_id
    
    @handle_database_error("update_experiment_status")
    def update_experiment_status(self, experiment_id: str, status: str, 
                                results: Optional[Dict] = None, 
                                error_message: Optional[str] = None):
        """Update experiment status."""
        with db.get_cursor() as cursor:
            completed_at = datetime.now() if status == 'completed' else None
            cursor.execute("""
                UPDATE experiments 
                SET status = %s, completed_at = %s, results = %s, error_message = %s
                WHERE id = %s
            """, (status, completed_at, json.dumps(results or {}), error_message, experiment_id))
    
    @handle_database_error("create_processing_job") 
    def create_processing_job(self, job_type: str, job_name: str, parameters: Dict,
                             parent_experiment_id: Optional[str] = None) -> str:
        """Create processing job."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO processing_jobs (job_type, job_name, parameters, parent_experiment_id)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (job_type, job_name, json.dumps(parameters), parent_experiment_id))
            job_id = safe_fetch_id(cursor, "create_processing_job")
            logger.info(f"✅ Created job '{job_name}' ({job_type}): {job_id}")
            return job_id
    
    def update_job_progress(self, job_id: str, progress_percent: float, 
                           status: Optional[str] = None, log_message: Optional[str] = None):
        """Update job progress."""
        with db.get_cursor() as cursor:
            updates = ["progress_percent = %s"]
            params: List[Any] = [progress_percent]
            
            if status:
                updates.append("status = %s")
                params.append(status)
                if status == 'running' and progress_percent == 0:
                    updates.append("started_at = CURRENT_TIMESTAMP")
                elif status in ['completed', 'failed']:
                    updates.append("completed_at = CURRENT_TIMESTAMP")
            
            if log_message:
                updates.append("log_messages = array_append(log_messages, %s)")
                params.append(log_message)
            
            params.append(job_id)
            
            cursor.execute(f"""
                UPDATE processing_jobs 
                SET {', '.join(updates)}
                WHERE id = %s
            """, params)

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment by ID."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM experiments WHERE id = %s
            """, (experiment_id,))
            return cursor.fetchone()
    
    def get_experiments(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get experiments with optional status filter."""
        with db.get_cursor() as cursor:
            query = "SELECT * FROM experiments"
            params = []
            
            if status:
                query += " WHERE status = %s"
                params.append(status)
            
            query += " ORDER BY created_at DESC"
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def get_processing_jobs(self, experiment_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get processing jobs, optionally filtered by experiment."""
        with db.get_cursor() as cursor:
            query = "SELECT * FROM processing_jobs"
            params = []
            
            if experiment_id:
                query += " WHERE parent_experiment_id = %s"
                params.append(experiment_id)
            
            query += " ORDER BY created_at DESC"
            cursor.execute(query, params)
            return cursor.fetchall()