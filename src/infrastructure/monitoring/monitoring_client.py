"""Client for querying monitoring data."""

from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import json

from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class MonitoringClient:
    """Client for querying monitoring data from database."""
    
    def __init__(self, db_manager):
        """Initialize monitoring client.
        
        Args:
            db_manager: DatabaseManager instance
        """
        self.db = db_manager
    
    def get_experiment_status(self, experiment: Union[str, str]) -> Dict[str, Any]:
        """Get comprehensive experiment status.
        
        Args:
            experiment: Experiment name or UUID
            
        Returns:
            Dict with experiment info, progress tree, and error count
        """
        # Get experiment info
        experiment_id = self._resolve_experiment_id(experiment)
        
        with self.db.get_cursor() as cursor:
            # Get experiment details
            cursor.execute("""
                SELECT id, name, status, started_at, completed_at, config
                FROM experiments
                WHERE id = %(exp_id)s
            """, {'exp_id': experiment_id})
            
            exp_row = cursor.fetchone()
            if not exp_row:
                raise ValueError(f"Experiment not found: {experiment}")
            
            # Get progress tree - direct query instead of function
            cursor.execute("""
                WITH RECURSIVE progress_tree AS (
                    -- Base case: root nodes
                    SELECT 
                        node_id,
                        parent_id,
                        0 as node_level,
                        node_name,
                        status,
                        progress_percent
                    FROM pipeline_progress
                    WHERE experiment_id = %(exp_id)s AND parent_id IS NULL
                    
                    UNION ALL
                    
                    -- Recursive case
                    SELECT 
                        p.node_id,
                        p.parent_id,
                        pt.node_level + 1,
                        p.node_name,
                        p.status,
                        p.progress_percent
                    FROM pipeline_progress p
                    INNER JOIN progress_tree pt ON p.parent_id = pt.node_id
                    WHERE p.experiment_id = %(exp_id)s
                )
                SELECT * FROM progress_tree
                ORDER BY node_level, node_name
            """, {'exp_id': experiment_id})
            progress_tree = cursor.fetchall()
            
            # Get error count
            cursor.execute("""
                SELECT COUNT(*) as error_count
                FROM pipeline_logs
                WHERE experiment_id = %(exp_id)s 
                AND level IN ('ERROR', 'CRITICAL')
            """, {'exp_id': experiment_id})
            
            error_count = cursor.fetchone()['error_count']
            
            return {
                'id': exp_row['id'],
                'name': exp_row['name'],
                'status': exp_row['status'],
                'started_at': exp_row['started_at'],
                'completed_at': exp_row['completed_at'],
                'config': exp_row['config'],
                'progress_tree': self._build_progress_tree(progress_tree),
                'error_count': error_count
            }
    
    def query_logs(self,
                   experiment_id: str,
                   level: Optional[str] = None,
                   search: Optional[str] = None,
                   start_time: Optional[Union[str, datetime]] = None,
                   after_id: Optional[str] = None,
                   limit: int = 100) -> List[Dict[str, Any]]:
        """Query logs with filters.
        
        Args:
            experiment_id: Experiment UUID
            level: Log level filter
            search: Text search in messages
            start_time: Time filter (datetime or string like "1h", "30m")
            after_id: Get logs after this ID (for pagination)
            limit: Maximum records
            
        Returns:
            List of log records
        """
        # Parse start time if string
        if isinstance(start_time, str):
            start_time = self._parse_time_delta(start_time)
        
        with self.db.get_cursor() as cursor:
            # Direct query instead of function
            query = """
                SELECT 
                    id, timestamp, level, node_id, message, context, traceback
                FROM pipeline_logs
                WHERE experiment_id = %(exp_id)s
            """
            params = {'exp_id': experiment_id}
            
            if level:
                query += " AND level = %(level)s"
                params['level'] = level
            
            if search:
                query += " AND message ILIKE %(search)s"
                params['search'] = f'%{search}%'
            
            if start_time:
                query += " AND timestamp >= %(start_time)s"
                params['start_time'] = start_time
            
            query += " ORDER BY timestamp DESC LIMIT %(limit)s"
            params['limit'] = limit
            
            cursor.execute(query, params)
            
            logs = []
            for row in cursor.fetchall():
                log = {
                    'id': row['id'],
                    'timestamp': row['timestamp'],
                    'level': row['level'],
                    'node_id': row['node_id'],
                    'message': row['message'],
                    'context': row['context'] or {},
                    'traceback': row['traceback']
                }
                
                # Extract stage from context
                if row['context']:
                    log['stage'] = row['context'].get('stage', 'unknown')
                else:
                    log['stage'] = 'unknown'
                
                logs.append(log)
            
            # Filter by after_id if provided
            if after_id:
                found = False
                filtered_logs = []
                for log in logs:
                    if found:
                        filtered_logs.append(log)
                    elif str(log['id']) == after_id:
                        found = True
                logs = filtered_logs
            
            return logs
    
    def get_metrics(self,
                   experiment_id: str,
                   node_id: Optional[str] = None,
                   metric_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get metrics for experiment.
        
        Args:
            experiment_id: Experiment UUID
            node_id: Optional node filter
            metric_type: Optional metric type filter
            
        Returns:
            List of metric records
        """
        query = """
            SELECT 
                id, node_id, timestamp,
                memory_mb, cpu_percent, disk_usage_mb, throughput_per_sec,
                custom_metrics
            FROM pipeline_metrics
            WHERE experiment_id = %(exp_id)s
        """
        
        params = {'exp_id': experiment_id}
        
        if node_id:
            query += " AND node_id = %(node_id)s"
            params['node_id'] = node_id
        
        query += " ORDER BY timestamp DESC LIMIT 1000"
        
        with self.db.get_cursor() as cursor:
            cursor.execute(query, params)
            
            metrics = []
            for row in cursor.fetchall():
                metric = {
                    'id': row['id'],
                    'node_id': row['node_id'],
                    'timestamp': row['timestamp']
                }
                
                # Add specific metric type or all
                if metric_type == 'memory':
                    metric['memory_mb'] = row['memory_mb']
                elif metric_type == 'cpu':
                    metric['cpu_percent'] = row['cpu_percent']
                elif metric_type == 'throughput':
                    metric['throughput_per_sec'] = row['throughput_per_sec']
                else:
                    # All metrics
                    if row['memory_mb'] is not None:
                        metric['memory_mb'] = row['memory_mb']
                    if row['cpu_percent'] is not None:
                        metric['cpu_percent'] = row['cpu_percent']
                    if row['disk_usage_mb'] is not None:
                        metric['disk_usage_mb'] = row['disk_usage_mb']
                    if row['throughput_per_sec'] is not None:
                        metric['throughput_per_sec'] = row['throughput_per_sec']
                    if row['custom_metrics']:
                        metric.update(row['custom_metrics'])
                
                metrics.append(metric)
            
            return metrics
    
    def get_error_summary(self, experiment_id: str) -> Dict[str, Any]:
        """Get error summary for experiment.
        
        Args:
            experiment_id: Experiment UUID
            
        Returns:
            Error summary with counts and recent errors
        """
        with self.db.get_cursor() as cursor:
            # Total error count
            cursor.execute("""
                SELECT COUNT(*) as total_errors
                FROM pipeline_logs
                WHERE experiment_id = %(exp_id)s 
                AND level IN ('ERROR', 'CRITICAL')
            """, {'exp_id': experiment_id})
            total_errors = cursor.fetchone()['total_errors']
            
            # Errors by level
            cursor.execute("""
                SELECT level, COUNT(*) as count
                FROM pipeline_logs
                WHERE experiment_id = %(exp_id)s 
                AND level IN ('ERROR', 'CRITICAL')
                GROUP BY level
            """, {'exp_id': experiment_id})
            by_level = {row['level']: row['count'] for row in cursor.fetchall()}
            
            # Errors by stage (from context)
            cursor.execute("""
                SELECT 
                    COALESCE(context->>'stage', 'unknown') as stage,
                    COUNT(*) as count
                FROM pipeline_logs
                WHERE experiment_id = %(exp_id)s 
                AND level IN ('ERROR', 'CRITICAL')
                GROUP BY COALESCE(context->>'stage', 'unknown')
            """, {'exp_id': experiment_id})
            by_stage = {row['stage']: row['count'] for row in cursor.fetchall()}
            
            # Recent errors
            cursor.execute("""
                SELECT 
                    timestamp, level, message, context, traceback,
                    COALESCE(context->>'stage', 'unknown') as stage
                FROM pipeline_logs
                WHERE experiment_id = %(exp_id)s 
                AND level IN ('ERROR', 'CRITICAL')
                ORDER BY timestamp DESC
                LIMIT 10
            """, {'exp_id': experiment_id})
            recent_errors = [dict(row) for row in cursor.fetchall()]
            
            return {
                'total_count': total_errors,
                'by_level': by_level,
                'by_stage': by_stage,
                'recent_errors': recent_errors
            }
    
    def _resolve_experiment_id(self, experiment: str) -> str:
        """Resolve experiment name to UUID.
        
        Args:
            experiment: Name or UUID
            
        Returns:
            Experiment UUID
        """
        # Check if already UUID format
        try:
            import uuid
            uuid.UUID(experiment)
            return experiment
        except ValueError:
            # Not a UUID, try to look up by name
            with self.db.get_cursor() as cursor:
                cursor.execute("""
                    SELECT id FROM experiments 
                    WHERE name = %(name)s 
                    ORDER BY started_at DESC 
                    LIMIT 1
                """, {'name': experiment})
                
                row = cursor.fetchone()
                if row:
                    return str(row['id'])
                else:
                    raise ValueError(f"Experiment not found: {experiment}")
    
    def _parse_time_delta(self, time_str: str) -> datetime:
        """Parse time delta string like '1h', '30m' to datetime.
        
        Args:
            time_str: Time string
            
        Returns:
            Datetime that far in the past
        """
        now = datetime.utcnow()
        
        if time_str.endswith('h'):
            hours = int(time_str[:-1])
            return now - timedelta(hours=hours)
        elif time_str.endswith('m'):
            minutes = int(time_str[:-1])
            return now - timedelta(minutes=minutes)
        elif time_str.endswith('d'):
            days = int(time_str[:-1])
            return now - timedelta(days=days)
        else:
            raise ValueError(f"Invalid time format: {time_str}. Use format like '1h', '30m', '7d'")
    
    def _build_progress_tree(self, rows: List[Dict]) -> List[Dict[str, Any]]:
        """Build hierarchical progress tree from flat rows.
        
        Args:
            rows: Flat list of progress nodes
            
        Returns:
            Hierarchical tree structure
        """
        # Build lookup map
        nodes = {}
        roots = []
        
        for row in rows:
            node = {
                'node_id': row['node_id'],
                'parent_id': row['parent_id'],
                'level': row['node_level'],
                'name': row['node_name'],
                'status': row['status'],
                'progress_percent': float(row['progress_percent'] or 0),
                'children': []
            }
            nodes[row['node_id']] = node
            
            if not row['parent_id']:
                roots.append(node)
        
        # Build tree
        for node_id, node in nodes.items():
            if node['parent_id'] and node['parent_id'] in nodes:
                nodes[node['parent_id']]['children'].append(node)
        
        return roots