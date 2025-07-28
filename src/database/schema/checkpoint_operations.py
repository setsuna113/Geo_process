"""Checkpoint operations for database schema."""

from typing import Dict, Any, List, Optional
import json
import logging
from ..connection import db

logger = logging.getLogger(__name__)


class CheckpointOperations:
    """Checkpoint database operations."""
    
    def create_checkpoint(self, checkpoint_id: str, level: str, parent_id: Optional[str],
                         processor_name: str, data_summary: Dict[str, Any],
                         file_path: str, file_size_bytes: int,
                         compression_type: Optional[str] = None) -> str:
        """Create a checkpoint record."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO pipeline_checkpoints 
                (checkpoint_id, level, parent_id, processor_name, data_summary,
                 file_path, file_size_bytes, compression_type, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'created')
                RETURNING id
            """, (
                checkpoint_id, level, parent_id, processor_name, 
                json.dumps(data_summary), file_path, file_size_bytes, 
                compression_type
            ))
            return cursor.fetchone()['id']
    
    def update_checkpoint_status(self, checkpoint_id: str, status: str,
                               validation_checksum: Optional[str] = None,
                               validation_result: Optional[Dict] = None,
                               error_message: Optional[str] = None):
        """Update checkpoint status and validation info."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                UPDATE pipeline_checkpoints 
                SET status = %s, validation_checksum = %s, validation_result = %s,
                    error_message = %s, updated_at = CURRENT_TIMESTAMP
                WHERE checkpoint_id = %s
            """, (
                status, validation_checksum, json.dumps(validation_result or {}),
                error_message, checkpoint_id
            ))
    
    def get_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get checkpoint by ID."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM pipeline_checkpoints 
                WHERE checkpoint_id = %s
            """, (checkpoint_id,))
            return cursor.fetchone()
    
    def get_latest_checkpoint(self, processor_name: str, level: str) -> Optional[Dict[str, Any]]:
        """Get latest checkpoint for a processor at a specific level."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM pipeline_checkpoints 
                WHERE processor_name = %s AND level = %s AND status = 'valid'
                ORDER BY created_at DESC
                LIMIT 1
            """, (processor_name, level))
            return cursor.fetchone()
    
    def list_checkpoints(self, processor_name: Optional[str] = None,
                        level: Optional[str] = None,
                        parent_id: Optional[str] = None,
                        status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List checkpoints with optional filtering."""
        with db.get_cursor() as cursor:
            query = "SELECT * FROM pipeline_checkpoints WHERE 1=1"
            params = []
            
            if processor_name:
                query += " AND processor_name = %s"
                params.append(processor_name)
            if level:
                query += " AND level = %s"
                params.append(level)
            if parent_id:
                query += " AND parent_id = %s"
                params.append(parent_id)
            if status:
                query += " AND status = %s"
                params.append(status)
            
            query += " ORDER BY created_at DESC"
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def cleanup_old_checkpoints(self, days_old: int, keep_minimum: Dict[str, int]) -> int:
        """Clean up old checkpoints while preserving minimum counts per level."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                WITH ranked_checkpoints AS (
                    SELECT id, level, created_at,
                           ROW_NUMBER() OVER (PARTITION BY level ORDER BY created_at DESC) as rn
                    FROM pipeline_checkpoints
                    WHERE status = 'valid'
                )
                DELETE FROM pipeline_checkpoints
                WHERE id IN (
                    SELECT id FROM ranked_checkpoints
                    WHERE created_at < NOW() - INTERVAL '%s days'
                    AND rn > %s
                )
                RETURNING id
            """, (days_old, min(keep_minimum.values(), default=5)))
            
            return cursor.rowcount
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                DELETE FROM pipeline_checkpoints 
                WHERE checkpoint_id = %s
            """, (checkpoint_id,))
            return cursor.rowcount > 0
    
    def get_checkpoint_hierarchy(self, root_checkpoint_id: str) -> List[Dict[str, Any]]:
        """Get checkpoint hierarchy starting from a root checkpoint."""
        with db.get_cursor() as cursor:
            cursor.execute("""
                WITH RECURSIVE checkpoint_tree AS (
                    SELECT * FROM pipeline_checkpoints 
                    WHERE checkpoint_id = %s
                    UNION ALL
                    SELECT pc.* FROM pipeline_checkpoints pc
                    JOIN checkpoint_tree ct ON pc.parent_id = ct.checkpoint_id
                )
                SELECT * FROM checkpoint_tree ORDER BY created_at
            """, (root_checkpoint_id,))
            return cursor.fetchall()