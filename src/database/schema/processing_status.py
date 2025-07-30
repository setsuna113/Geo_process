"""Processing status tracking operations for database schema."""

from typing import Dict, Any, List, Optional
import json
import logging
from ..connection import db

logger = logging.getLogger(__name__)


class ProcessingStatus:
    """Processing status tracking database operations."""
    
    def create_processing_step(self, step_name: str, processor_name: str,
                             parent_job_id: str, total_items: int,
                             parameters: Optional[Dict] = None) -> str:
        """Create a processing step record."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO processing_steps 
                (step_name, processor_name, parent_job_id, total_items,
                 parameters, status)
                VALUES (%s, %s, %s, %s, %s, 'pending')
                RETURNING id
            """, (
                step_name, processor_name, parent_job_id, total_items,
                json.dumps(parameters or {})
            ))
            return cursor.fetchone()['id']
    
    def update_processing_step(self, step_id: str, processed_items: int,
                             failed_items: int = 0, status: Optional[str] = None,
                             error_messages: Optional[List[str]] = None,
                             checkpoint_id: Optional[str] = None):
        """Update processing step progress."""
        with db.get_cursor() as cursor:
            updates = ["processed_items = %s", "failed_items = %s"]
            params = [processed_items, failed_items]
            
            if status:
                updates.append("status = %s")
                params.append(status)
                if status == 'running' and processed_items == 0:
                    updates.append("started_at = CURRENT_TIMESTAMP")
                elif status in ['completed', 'failed']:
                    updates.append("completed_at = CURRENT_TIMESTAMP")
            
            if error_messages:
                updates.append("error_messages = array_cat(error_messages, %s)")
                params.append(error_messages)
            
            if checkpoint_id:
                updates.append("last_checkpoint_id = %s")
                params.append(checkpoint_id)
            
            params.append(step_id)
            
            cursor.execute(f"""
                UPDATE processing_steps 
                SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, params)
    
    def get_processing_steps(self, parent_job_id: str) -> List[Dict[str, Any]]:
        """Get all processing steps for a job."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM processing_steps 
                WHERE parent_job_id = %s
                ORDER BY created_at
            """, (parent_job_id,))
            return cursor.fetchall()
    
    def create_file_processing_status(self, file_path: str, file_type: str,
                                    file_size_bytes: int, processor_name: str,
                                    parent_job_id: Optional[str] = None) -> str:
        """Create file processing status record."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO file_processing_status 
                (file_path, file_type, file_size_bytes, processor_name,
                 parent_job_id, status)
                VALUES (%s, %s, %s, %s, %s, 'pending')
                RETURNING id
            """, (
                file_path, file_type, file_size_bytes, processor_name,
                parent_job_id
            ))
            return cursor.fetchone()['id']
    
    def update_file_processing_status(self, file_id: str, status: str,
                                    bytes_processed: Optional[int] = None,
                                    chunks_completed: Optional[int] = None,
                                    total_chunks: Optional[int] = None,
                                    checkpoint_id: Optional[str] = None,
                                    error_message: Optional[str] = None):
        """Update file processing status."""
        with db.get_cursor() as cursor:
            updates = ["status = %s"]
            params = [status]
            
            if bytes_processed is not None:
                updates.append("bytes_processed = %s")
                params.append(bytes_processed)
            
            if chunks_completed is not None:
                updates.append("chunks_completed = %s")
                params.append(chunks_completed)
            
            if total_chunks is not None:
                updates.append("total_chunks = %s")
                params.append(total_chunks)
            
            if checkpoint_id:
                updates.append("last_checkpoint_id = %s")
                params.append(checkpoint_id)
            
            if error_message:
                updates.append("error_message = %s")
                params.append(error_message)
            
            if status == 'processing' and chunks_completed == 0:
                updates.append("started_at = CURRENT_TIMESTAMP")
            elif status in ['completed', 'failed']:
                updates.append("completed_at = CURRENT_TIMESTAMP")
            
            params.append(file_id)
            
            cursor.execute(f"""
                UPDATE file_processing_status 
                SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, params)
    
    def get_file_processing_status(self, file_path: Optional[str] = None,
                                 parent_job_id: Optional[str] = None,
                                 status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get file processing status with optional filtering."""
        with db.get_cursor() as cursor:
            query = "SELECT * FROM file_processing_status WHERE 1=1"
            params = []
            
            if file_path:
                query += " AND file_path = %s"
                params.append(file_path)
            
            if parent_job_id:
                query += " AND parent_job_id = %s"
                params.append(parent_job_id)
            
            if status:
                query += " AND status = %s"
                params.append(status)
            
            query += " ORDER BY created_at DESC"
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def get_resumable_files(self, processor_name: str) -> List[Dict[str, Any]]:
        """Get files that can be resumed from checkpoints."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM file_processing_status 
                WHERE processor_name = %s 
                AND status IN ('processing', 'failed')
                AND last_checkpoint_id IS NOT NULL
                ORDER BY updated_at DESC
            """, (processor_name,))
            return cursor.fetchall()
    
    def get_processing_queue_summary(self) -> List[Dict[str, Any]]:
        """Get processing queue statistics."""
        with db.get_cursor() as cursor:
            cursor.execute("SELECT * FROM processing_queue_summary ORDER BY queue_type, status")
            return cursor.fetchall()