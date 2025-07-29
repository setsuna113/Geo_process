"""Database backend for metrics collection."""

import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.base.monitoring.metrics_backend import MetricsBackend
from src.infrastructure.logging import get_logger

logger = get_logger(__name__)


class DatabaseMetricsBackend(MetricsBackend):
    """PostgreSQL backend for metrics storage."""
    
    def __init__(self, db_manager):
        """Initialize database metrics backend.
        
        Args:
            db_manager: DatabaseManager instance
        """
        self.db = db_manager
    
    def record_metrics(self,
                      experiment_id: str,
                      node_id: Optional[str],
                      metrics: Dict[str, float],
                      timestamp: Optional[datetime] = None) -> None:
        """Record performance metrics to database."""
        # Split standard and custom metrics
        standard_metrics = {}
        custom_metrics = {}
        
        standard_fields = {'memory_mb', 'cpu_percent', 'disk_usage_mb', 'throughput_per_sec'}
        
        for key, value in metrics.items():
            if key in standard_fields:
                standard_metrics[key] = value
            else:
                custom_metrics[key] = value
        
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO pipeline_metrics 
                (experiment_id, node_id, timestamp, memory_mb, cpu_percent, 
                 disk_usage_mb, throughput_per_sec, custom_metrics)
                VALUES (%(exp_id)s, %(node_id)s, %(timestamp)s, %(memory_mb)s, 
                        %(cpu_percent)s, %(disk_usage_mb)s, %(throughput_per_sec)s, 
                        %(custom_metrics)s::jsonb)
            """, {
                'exp_id': experiment_id,
                'node_id': node_id,
                'timestamp': timestamp or datetime.utcnow(),
                'memory_mb': standard_metrics.get('memory_mb'),
                'cpu_percent': standard_metrics.get('cpu_percent'),
                'disk_usage_mb': standard_metrics.get('disk_usage_mb'),
                'throughput_per_sec': standard_metrics.get('throughput_per_sec'),
                'custom_metrics': json.dumps(custom_metrics) if custom_metrics else None
            })
            
        logger.debug(f"Recorded metrics for experiment {experiment_id}, node {node_id}")
    
    def get_metrics(self,
                   experiment_id: str,
                   node_id: Optional[str] = None,
                   metric_names: Optional[List[str]] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   limit: int = 1000) -> List[Dict[str, Any]]:
        """Query metrics with filters."""
        query = """
            SELECT 
                id, experiment_id, node_id, timestamp,
                memory_mb, cpu_percent, disk_usage_mb, throughput_per_sec,
                custom_metrics
            FROM pipeline_metrics
            WHERE experiment_id = %(exp_id)s
        """
        
        params = {'exp_id': experiment_id}
        
        if node_id:
            query += " AND node_id = %(node_id)s"
            params['node_id'] = node_id
        
        if start_time:
            query += " AND timestamp >= %(start_time)s"
            params['start_time'] = start_time
            
        if end_time:
            query += " AND timestamp <= %(end_time)s"
            params['end_time'] = end_time
        
        query += " ORDER BY timestamp DESC LIMIT %(limit)s"
        params['limit'] = limit
        
        with self.db.get_cursor() as cursor:
            cursor.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                metrics = self._extract_metrics(row, metric_names)
                if metrics:  # Only include if has requested metrics
                    results.append({
                        'id': row['id'],
                        'experiment_id': row['experiment_id'],
                        'node_id': row['node_id'],
                        'timestamp': row['timestamp'],
                        'metrics': metrics
                    })
            
            return results
    
    def get_latest_metrics(self,
                          experiment_id: str,
                          node_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get most recent metrics."""
        query = """
            SELECT 
                id, experiment_id, node_id, timestamp,
                memory_mb, cpu_percent, disk_usage_mb, throughput_per_sec,
                custom_metrics
            FROM pipeline_metrics
            WHERE experiment_id = %(exp_id)s
        """
        
        params = {'exp_id': experiment_id}
        
        if node_id:
            query += " AND node_id = %(node_id)s"
            params['node_id'] = node_id
        
        query += " ORDER BY timestamp DESC LIMIT 1"
        
        with self.db.get_cursor() as cursor:
            cursor.execute(query, params)
            row = cursor.fetchone()
            
            if row:
                return {
                    'id': row['id'],
                    'experiment_id': row['experiment_id'],
                    'node_id': row['node_id'],
                    'timestamp': row['timestamp'],
                    'metrics': self._extract_metrics(row)
                }
            
            return None
    
    def get_metric_summary(self,
                          experiment_id: str,
                          metric_name: str,
                          node_id: Optional[str] = None) -> Dict[str, float]:
        """Get summary statistics for a metric."""
        # Check if it's a standard metric
        standard_fields = ['memory_mb', 'cpu_percent', 'disk_usage_mb', 'throughput_per_sec']
        
        if metric_name in standard_fields:
            query = f"""
                SELECT 
                    MIN({metric_name}) as min_val,
                    MAX({metric_name}) as max_val,
                    AVG({metric_name}) as avg_val,
                    COUNT({metric_name}) as count_val
                FROM pipeline_metrics
                WHERE experiment_id = %(exp_id)s
                AND {metric_name} IS NOT NULL
            """
        else:
            # Custom metric in JSONB
            query = """
                SELECT 
                    MIN((custom_metrics->>%(metric_name)s)::float) as min_val,
                    MAX((custom_metrics->>%(metric_name)s)::float) as max_val,
                    AVG((custom_metrics->>%(metric_name)s)::float) as avg_val,
                    COUNT(custom_metrics->>%(metric_name)s) as count_val
                FROM pipeline_metrics
                WHERE experiment_id = %(exp_id)s
                AND custom_metrics ? %(metric_name)s
            """
        
        params = {
            'exp_id': experiment_id,
            'metric_name': metric_name
        }
        
        if node_id:
            query += " AND node_id = %(node_id)s"
            params['node_id'] = node_id
        
        with self.db.get_cursor() as cursor:
            cursor.execute(query, params)
            row = cursor.fetchone()
            
            if row and row['count_val'] > 0:
                return {
                    'min': float(row['min_val'] or 0),
                    'max': float(row['max_val'] or 0),
                    'avg': float(row['avg_val'] or 0),
                    'count': int(row['count_val'])
                }
            
            return {'min': 0, 'max': 0, 'avg': 0, 'count': 0}
    
    def _extract_metrics(self, row: Dict[str, Any], 
                        metric_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Extract metrics from database row."""
        metrics = {}
        
        # Standard metrics
        standard_fields = {
            'memory_mb': row.get('memory_mb'),
            'cpu_percent': row.get('cpu_percent'),
            'disk_usage_mb': row.get('disk_usage_mb'),
            'throughput_per_sec': row.get('throughput_per_sec')
        }
        
        for name, value in standard_fields.items():
            if value is not None:
                if metric_names is None or name in metric_names:
                    metrics[name] = float(value)
        
        # Custom metrics
        if row.get('custom_metrics'):
            for name, value in row['custom_metrics'].items():
                if metric_names is None or name in metric_names:
                    try:
                        metrics[name] = float(value)
                    except (TypeError, ValueError):
                        # Skip non-numeric metrics
                        pass
        
        return metrics