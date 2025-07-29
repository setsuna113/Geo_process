"""Database backend for progress tracking."""

import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.base.monitoring.progress_backend import ProgressBackend
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class DatabaseProgressBackend(ProgressBackend):
    """PostgreSQL backend for progress tracking with persistence."""
    
    def __init__(self, db_manager):
        """Initialize database progress backend.
        
        Args:
            db_manager: DatabaseManager instance
        """
        self.db = db_manager
    
    def create_node(self, 
                   experiment_id: str,
                   node_id: str, 
                   parent_id: Optional[str],
                   level: str, 
                   name: str, 
                   total_units: int,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """Create a progress node in database."""
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO pipeline_progress 
                (experiment_id, node_id, parent_id, node_level, node_name, 
                 total_units, status, metadata, start_time)
                VALUES (%(exp_id)s, %(node_id)s, %(parent_id)s, %(level)s, 
                        %(name)s, %(total)s, 'pending', %(metadata)s::jsonb, 
                        CURRENT_TIMESTAMP)
                ON CONFLICT (experiment_id, node_id) DO UPDATE
                SET parent_id = EXCLUDED.parent_id,
                    node_name = EXCLUDED.node_name,
                    total_units = EXCLUDED.total_units,
                    metadata = COALESCE(pipeline_progress.metadata, '{}'::jsonb) || EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP
            """, {
                'exp_id': experiment_id,
                'node_id': node_id,
                'parent_id': parent_id,
                'level': level,
                'name': name,
                'total': total_units,
                'metadata': json.dumps(metadata or {})
            })
            
        logger.debug(f"Created progress node: {node_id}")
    
    def update_progress(self, 
                       experiment_id: str,
                       node_id: str, 
                       completed_units: int,
                       status: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update node progress in database."""
        with self.db.get_cursor() as cursor:
            # Calculate percentage
            cursor.execute("""
                UPDATE pipeline_progress
                SET completed_units = %(completed)s,
                    progress_percent = (%(completed)s::float / NULLIF(total_units, 0)) * 100,
                    status = %(status)s,
                    metadata = COALESCE(metadata, '{}'::jsonb) || %(metadata)s::jsonb,
                    updated_at = CURRENT_TIMESTAMP,
                    start_time = CASE 
                        WHEN start_time IS NULL AND %(status)s = 'running' 
                        THEN CURRENT_TIMESTAMP 
                        ELSE start_time 
                    END,
                    end_time = CASE 
                        WHEN %(status)s IN ('completed', 'failed', 'cancelled') 
                        THEN CURRENT_TIMESTAMP 
                        ELSE end_time 
                    END
                WHERE experiment_id = %(exp_id)s AND node_id = %(node_id)s
            """, {
                'exp_id': experiment_id,
                'node_id': node_id,
                'completed': completed_units,
                'status': status,
                'metadata': json.dumps(metadata or {})
            })
            
        logger.debug(f"Updated progress for {node_id}: {completed_units} units, status={status}")
    
    def get_node(self, experiment_id: str, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node information from database."""
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT 
                    node_id, parent_id, node_level, node_name, status,
                    progress_percent, completed_units, total_units,
                    start_time, end_time, metadata,
                    EXTRACT(EPOCH FROM (COALESCE(end_time, CURRENT_TIMESTAMP) - start_time)) as elapsed_seconds
                FROM pipeline_progress
                WHERE experiment_id = %(exp_id)s AND node_id = %(node_id)s
            """, {'exp_id': experiment_id, 'node_id': node_id})
            
            row = cursor.fetchone()
            if row:
                return self._row_to_dict(row)
            return None
    
    def get_children(self, experiment_id: str, parent_id: str) -> List[Dict[str, Any]]:
        """Get child nodes from database."""
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT 
                    node_id, parent_id, node_level, node_name, status,
                    progress_percent, completed_units, total_units,
                    start_time, end_time, metadata,
                    EXTRACT(EPOCH FROM (COALESCE(end_time, CURRENT_TIMESTAMP) - start_time)) as elapsed_seconds
                FROM pipeline_progress
                WHERE experiment_id = %(exp_id)s AND parent_id = %(parent_id)s
                ORDER BY node_id
            """, {'exp_id': experiment_id, 'parent_id': parent_id})
            
            return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def get_experiment_nodes(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get all nodes for an experiment."""
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT 
                    node_id, parent_id, node_level, node_name, status,
                    progress_percent, completed_units, total_units,
                    start_time, end_time, metadata,
                    EXTRACT(EPOCH FROM (COALESCE(end_time, CURRENT_TIMESTAMP) - start_time)) as elapsed_seconds
                FROM pipeline_progress
                WHERE experiment_id = %(exp_id)s
                ORDER BY node_level, node_id
            """, {'exp_id': experiment_id})
            
            return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def update_node_status(self, 
                          experiment_id: str,
                          node_id: str,
                          status: str,
                          end_time: Optional[datetime] = None) -> None:
        """Update node status and optionally end time."""
        with self.db.get_cursor() as cursor:
            if end_time:
                cursor.execute("""
                    UPDATE pipeline_progress
                    SET status = %(status)s,
                        end_time = %(end_time)s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE experiment_id = %(exp_id)s AND node_id = %(node_id)s
                """, {
                    'exp_id': experiment_id,
                    'node_id': node_id,
                    'status': status,
                    'end_time': end_time
                })
            else:
                cursor.execute("""
                    UPDATE pipeline_progress
                    SET status = %(status)s,
                        updated_at = CURRENT_TIMESTAMP,
                        end_time = CASE 
                            WHEN %(status)s IN ('completed', 'failed', 'cancelled') 
                            THEN CURRENT_TIMESTAMP 
                            ELSE end_time 
                        END
                    WHERE experiment_id = %(exp_id)s AND node_id = %(node_id)s
                """, {
                    'exp_id': experiment_id,
                    'node_id': node_id,
                    'status': status
                })
    
    def get_active_nodes(self, experiment_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get currently active (running) nodes."""
        with self.db.get_cursor() as cursor:
            if experiment_id:
                cursor.execute("""
                    SELECT 
                        node_id, parent_id, node_level, node_name, status,
                        progress_percent, completed_units, total_units,
                        start_time, end_time, metadata, experiment_id,
                        EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - start_time)) as elapsed_seconds
                    FROM pipeline_progress
                    WHERE experiment_id = %(exp_id)s AND status = 'running'
                    ORDER BY start_time DESC
                """, {'exp_id': experiment_id})
            else:
                cursor.execute("""
                    SELECT 
                        p.node_id, p.parent_id, p.node_level, p.node_name, p.status,
                        p.progress_percent, p.completed_units, p.total_units,
                        p.start_time, p.end_time, p.metadata, p.experiment_id,
                        e.name as experiment_name,
                        EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - p.start_time)) as elapsed_seconds
                    FROM pipeline_progress p
                    JOIN experiments e ON p.experiment_id = e.id
                    WHERE p.status = 'running'
                    ORDER BY p.start_time DESC
                """)
            
            return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def _row_to_dict(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Convert database row to progress node dict."""
        return {
            'node_id': row['node_id'],
            'parent_id': row['parent_id'],
            'level': row['node_level'],
            'name': row['node_name'],
            'status': row['status'],
            'progress_percent': float(row['progress_percent'] or 0),
            'completed_units': row['completed_units'],
            'total_units': row['total_units'],
            'start_time': row['start_time'],
            'end_time': row['end_time'],
            'elapsed_seconds': float(row.get('elapsed_seconds') or 0),
            'metadata': row['metadata'] or {},
            'experiment_id': row.get('experiment_id'),
            'experiment_name': row.get('experiment_name')
        }