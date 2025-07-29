"""Abstract base class for metrics collection backends."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime


class MetricsBackend(ABC):
    """Abstract backend for metrics storage.
    
    Defines the interface for storing and retrieving performance metrics.
    """
    
    @abstractmethod
    def record_metrics(self,
                      experiment_id: str,
                      node_id: Optional[str],
                      metrics: Dict[str, float],
                      timestamp: Optional[datetime] = None) -> None:
        """Record performance metrics.
        
        Args:
            experiment_id: Experiment UUID
            node_id: Optional node ID for context
            metrics: Dict of metric name to value
            timestamp: Optional timestamp (defaults to now)
        """
        pass
    
    @abstractmethod
    def get_metrics(self,
                   experiment_id: str,
                   node_id: Optional[str] = None,
                   metric_names: Optional[List[str]] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   limit: int = 1000) -> List[Dict[str, Any]]:
        """Query metrics with filters.
        
        Args:
            experiment_id: Experiment UUID
            node_id: Optional node filter
            metric_names: Optional list of metric names to filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum records to return
            
        Returns:
            List of metric records
        """
        pass
    
    @abstractmethod
    def get_latest_metrics(self,
                          experiment_id: str,
                          node_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get most recent metrics.
        
        Args:
            experiment_id: Experiment UUID
            node_id: Optional node filter
            
        Returns:
            Latest metric record or None
        """
        pass
    
    @abstractmethod
    def get_metric_summary(self,
                          experiment_id: str,
                          metric_name: str,
                          node_id: Optional[str] = None) -> Dict[str, float]:
        """Get summary statistics for a metric.
        
        Args:
            experiment_id: Experiment UUID
            metric_name: Name of metric to summarize
            node_id: Optional node filter
            
        Returns:
            Dict with min, max, avg, count
        """
        pass