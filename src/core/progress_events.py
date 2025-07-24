# src/core/progress_events.py
"""Progress event system for the biodiversity pipeline."""

import time
import threading
import logging
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import queue
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of progress events."""
    # File I/O events
    FILE_READ_START = "file_read_start"
    FILE_READ_PROGRESS = "file_read_progress" 
    FILE_READ_COMPLETE = "file_read_complete"
    FILE_WRITE_START = "file_write_start"
    FILE_WRITE_PROGRESS = "file_write_progress"
    FILE_WRITE_COMPLETE = "file_write_complete"
    
    # Processing events
    PROCESSING_START = "processing_start"
    PROCESSING_PROGRESS = "processing_progress"
    PROCESSING_COMPLETE = "processing_complete"
    PROCESSING_ERROR = "processing_error"
    
    # Memory events
    MEMORY_WARNING = "memory_warning"
    MEMORY_CRITICAL = "memory_critical"
    MEMORY_CLEANUP = "memory_cleanup"
    
    # Checkpoint events
    CHECKPOINT_SAVE = "checkpoint_save"
    CHECKPOINT_LOAD = "checkpoint_load"
    
    # System events
    SYSTEM_PAUSE = "system_pause"
    SYSTEM_RESUME = "system_resume"
    SYSTEM_CANCEL = "system_cancel"


@dataclass
class ProgressEvent:
    """Base class for all progress events."""
    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    source: str = ""  # Component that generated the event
    node_id: Optional[str] = None  # Related progress node
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'event_type': self.event_type.value,
            'timestamp': self.timestamp,
            'source': self.source,
            'node_id': self.node_id,
            'metadata': self.metadata
        }


@dataclass
class FileIOProgress(ProgressEvent):
    """File I/O progress event."""
    file_path: str = ""
    total_bytes: int = 0
    processed_bytes: int = 0
    operation: str = ""  # read, write, copy, etc.
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_bytes == 0:
            return 0.0
        return (self.processed_bytes / self.total_bytes) * 100.0


@dataclass
class ProcessingProgress(ProgressEvent):
    """Processing progress event."""
    operation_name: str = ""
    total_items: int = 0
    processed_items: int = 0
    current_item: Optional[str] = None
    items_per_second: float = 0.0
    estimated_remaining_seconds: Optional[float] = None
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100.0


@dataclass
class MemoryProgress(ProgressEvent):
    """Memory-related progress event."""
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    memory_percent: float = 0.0
    pressure_level: str = "normal"
    action_taken: Optional[str] = None


class EventFilter(ABC):
    """Abstract base class for event filters."""
    
    @abstractmethod
    def should_pass(self, event: ProgressEvent) -> bool:
        """Check if event should pass through filter."""
        pass


class TypeFilter(EventFilter):
    """Filter events by type."""
    
    def __init__(self, allowed_types: List[EventType]):
        self.allowed_types = set(allowed_types)
    
    def should_pass(self, event: ProgressEvent) -> bool:
        return event.event_type in self.allowed_types


class SourceFilter(EventFilter):
    """Filter events by source."""
    
    def __init__(self, allowed_sources: List[str]):
        self.allowed_sources = set(allowed_sources)
    
    def should_pass(self, event: ProgressEvent) -> bool:
        return event.source in self.allowed_sources


class RateLimitFilter(EventFilter):
    """Rate limit events."""
    
    def __init__(self, max_events_per_second: float = 10.0):
        self.max_events_per_second = max_events_per_second
        self.min_interval = 1.0 / max_events_per_second
        self.last_event_time: Dict[str, float] = {}
    
    def should_pass(self, event: ProgressEvent) -> bool:
        key = f"{event.event_type.value}:{event.source}"
        now = time.time()
        
        last_time = self.last_event_time.get(key, 0)
        if now - last_time >= self.min_interval:
            self.last_event_time[key] = now
            return True
        return False


class ProgressEventBus:
    """
    Central event bus for progress events.
    
    Features:
    - Asynchronous event processing
    - Event filtering and validation
    - Multiple subscriber support
    - Thread-safe operations
    """
    
    def __init__(self, max_queue_size: int = 10000):
        self.max_queue_size = max_queue_size
        self._event_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._subscribers: Dict[str, List[Callable]] = {}
        self._filters: List[EventFilter] = []
        self._subscriber_lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'events_published': 0,
            'events_delivered': 0,
            'events_filtered': 0,
            'events_dropped': 0
        }
        
        # Processing thread
        self._processing = False
        self._processing_thread: Optional[threading.Thread] = None
        
        # Start processing
        self.start()
    
    def publish(self, event: ProgressEvent) -> bool:
        """
        Publish an event to the bus.
        
        Args:
            event: Event to publish
            
        Returns:
            True if event was queued
        """
        # Validate event
        if not self._validate_event(event):
            return False
        
        # Apply filters
        for filter_obj in self._filters:
            if not filter_obj.should_pass(event):
                self._stats['events_filtered'] += 1
                return False
        
        # Queue event
        try:
            self._event_queue.put_nowait(event)
            self._stats['events_published'] += 1
            return True
        except queue.Full:
            self._stats['events_dropped'] += 1
            logger.warning(f"Event queue full, dropping event: {event.event_type}")
            return False
    
    def subscribe(self, 
                 subscriber_id: str,
                 callback: Callable[[ProgressEvent], None],
                 event_types: Optional[List[EventType]] = None) -> None:
        """
        Subscribe to events.
        
        Args:
            subscriber_id: Unique subscriber identifier
            callback: Function to call with events
            event_types: Specific event types to subscribe to (None for all)
        """
        with self._subscriber_lock:
            key = subscriber_id
            if event_types:
                # Create filtered callback
                allowed_types = set(event_types)
                
                def filtered_callback(event: ProgressEvent):
                    if event.event_type in allowed_types:
                        callback(event)
                
                self._subscribers[key] = self._subscribers.get(key, [])
                self._subscribers[key].append(filtered_callback)
            else:
                self._subscribers[key] = self._subscribers.get(key, [])
                self._subscribers[key].append(callback)
    
    def unsubscribe(self, subscriber_id: str) -> None:
        """Unsubscribe from events."""
        with self._subscriber_lock:
            self._subscribers.pop(subscriber_id, None)
    
    def add_filter(self, filter_obj: EventFilter) -> None:
        """Add an event filter."""
        self._filters.append(filter_obj)
    
    def start(self) -> None:
        """Start event processing."""
        if self._processing:
            return
        
        self._processing = True
        self._processing_thread = threading.Thread(
            target=self._process_events,
            daemon=True
        )
        self._processing_thread.start()
    
    def stop(self) -> None:
        """Stop event processing."""
        self._processing = False
        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)
    
    def get_statistics(self) -> Dict[str, int]:
        """Get event bus statistics."""
        return self._stats.copy()
    
    def _validate_event(self, event: ProgressEvent) -> bool:
        """Validate an event."""
        if not isinstance(event, ProgressEvent):
            logger.error("Invalid event type")
            return False
        
        if not event.source:
            logger.warning("Event missing source")
            return False
        
        return True
    
    def _process_events(self) -> None:
        """Process events from queue."""
        while self._processing:
            try:
                # Get event with timeout
                event = self._event_queue.get(timeout=1.0)
                
                # Deliver to subscribers
                self._deliver_event(event)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    def _deliver_event(self, event: ProgressEvent) -> None:
        """Deliver event to subscribers."""
        with self._subscriber_lock:
            for subscriber_callbacks in self._subscribers.values():
                for callback in subscriber_callbacks:
                    try:
                        callback(event)
                        self._stats['events_delivered'] += 1
                    except Exception as e:
                        logger.error(f"Error in event callback: {e}")


# Convenience functions for creating events
def create_file_progress(file_path: str, 
                        processed: int, 
                        total: int,
                        operation: str = "read",
                        source: str = "",
                        node_id: Optional[str] = None) -> FileIOProgress:
    """Create a file I/O progress event."""
    event_type = EventType.FILE_READ_PROGRESS if operation == "read" else EventType.FILE_WRITE_PROGRESS
    
    return FileIOProgress(
        event_type=event_type,
        source=source,
        node_id=node_id,
        file_path=file_path,
        processed_bytes=processed,
        total_bytes=total,
        operation=operation
    )


def create_processing_progress(operation_name: str,
                             processed: int,
                             total: int,
                             source: str = "",
                             node_id: Optional[str] = None,
                             **kwargs) -> ProcessingProgress:
    """Create a processing progress event."""
    return ProcessingProgress(
        event_type=EventType.PROCESSING_PROGRESS,
        source=source,
        node_id=node_id,
        operation_name=operation_name,
        processed_items=processed,
        total_items=total,
        **kwargs
    )


def create_memory_event(event_type: EventType,
                       memory_used_mb: float,
                       memory_available_mb: float,
                       pressure_level: str,
                       source: str = "",
                       action_taken: Optional[str] = None) -> MemoryProgress:
    """Create a memory-related event."""
    return MemoryProgress(
        event_type=event_type,
        source=source,
        memory_used_mb=memory_used_mb,
        memory_available_mb=memory_available_mb,
        memory_percent=(memory_used_mb / (memory_used_mb + memory_available_mb) * 100) if memory_available_mb > 0 else 100,
        pressure_level=pressure_level,
        action_taken=action_taken
    )


# Global event bus instance
_event_bus: Optional[ProgressEventBus] = None


def get_event_bus() -> ProgressEventBus:
    """Get the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = ProgressEventBus()
    return _event_bus