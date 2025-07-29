"""Database handler for structured logging with async batching."""

import logging
import json
import queue
import threading
import time
import sys
from typing import Optional, List, Dict, Any
from datetime import datetime
import atexit


class DatabaseLogHandler(logging.Handler):
    """Async database handler with batching for high-performance logging.
    
    Features:
    - Asynchronous writes to avoid blocking
    - Batching for efficient database operations
    - Automatic flush on batch size or time interval
    - Graceful shutdown with final flush
    - Queue overflow protection
    """
    
    def __init__(self, 
                 db_manager,
                 batch_size: int = 100,
                 flush_interval: float = 5.0,
                 max_queue_size: int = 10000):
        """Initialize database log handler.
        
        Args:
            db_manager: DatabaseManager instance
            batch_size: Number of records to batch before writing
            flush_interval: Maximum seconds between flushes
            max_queue_size: Maximum queue size before dropping logs
        """
        super().__init__()
        self.db = db_manager
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Thread-safe queue for log records
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.batch: List[logging.LogRecord] = []
        self._batch_lock = threading.Lock()
        
        # Statistics
        self._records_written = 0
        self._records_dropped = 0
        self._write_errors = 0
        
        # Background worker thread
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker,
            name="DatabaseLogHandler",
            daemon=True
        )
        self._worker_thread.start()
        
        # Register cleanup on exit
        atexit.register(self.close)
    
    def emit(self, record: logging.LogRecord):
        """Queue log record for database insertion.
        
        Args:
            record: LogRecord to process
        """
        try:
            # Don't block if queue is full
            self.queue.put_nowait(record)
        except queue.Full:
            self._records_dropped += 1
            # Fallback to stderr for critical logs
            if record.levelno >= logging.ERROR:
                sys.stderr.write(
                    f"[LOG QUEUE FULL] {record.levelname}: {record.getMessage()}\n"
                )
    
    def _worker(self):
        """Background worker to batch insert logs."""
        while not self._stop_event.is_set():
            try:
                # Collect batch with timeout
                self._collect_batch()
                
                # Flush if batch is ready
                if self._should_flush():
                    self._flush_batch()
                    
            except Exception as e:
                self._write_errors += 1
                sys.stderr.write(f"Database log handler error: {e}\n")
                # Don't let worker die on errors
                time.sleep(1)
    
    def _collect_batch(self):
        """Collect records into batch with timeout."""
        deadline = time.time() + self.flush_interval
        
        while len(self.batch) < self.batch_size and time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
                
            try:
                # Get record with timeout
                record = self.queue.get(timeout=min(remaining, 1.0))
                with self._batch_lock:
                    self.batch.append(record)
                self.queue.task_done()
            except queue.Empty:
                # Check if we should flush partial batch
                if self.batch and time.time() >= deadline:
                    break
    
    def _should_flush(self) -> bool:
        """Check if batch should be flushed."""
        with self._batch_lock:
            return len(self.batch) > 0
    
    def _flush_batch(self):
        """Insert batch of logs to database."""
        with self._batch_lock:
            if not self.batch:
                return
            
            # Copy batch and clear
            records_to_write = self.batch[:]
            self.batch.clear()
        
        try:
            # Prepare batch data
            values = []
            for record in records_to_write:
                # Extract structured data from record
                context = getattr(record, 'context', {})
                tb = getattr(record, 'traceback', None)
                perf = getattr(record, 'performance', {})
                
                # Build record dict
                log_data = {
                    'experiment_id': context.get('experiment_id'),
                    'job_id': context.get('job_id'),
                    'node_id': context.get('node_id'),
                    'timestamp': datetime.fromtimestamp(record.created),
                    'level': record.levelname,
                    'logger_name': record.name,
                    'message': record.getMessage(),
                    'context': json.dumps(context),
                    'traceback': tb,
                    'performance': json.dumps(perf) if perf else None
                }
                values.append(log_data)
            
            # Bulk insert with connection from pool
            with self.db.get_cursor() as cursor:
                cursor.executemany("""
                    INSERT INTO pipeline_logs 
                    (experiment_id, job_id, node_id, timestamp, level, 
                     logger_name, message, context, traceback, performance)
                    VALUES (%(experiment_id)s, %(job_id)s, %(node_id)s, 
                            %(timestamp)s, %(level)s, %(logger_name)s, 
                            %(message)s, %(context)s::jsonb, %(traceback)s, 
                            %(performance)s::jsonb)
                """, values)
            
            self._records_written += len(values)
            
        except Exception as e:
            self._write_errors += 1
            sys.stderr.write(
                f"Failed to write {len(records_to_write)} logs to database: {e}\n"
            )
            
            # Log critical errors to file as fallback
            self._write_fallback_log(records_to_write, e)
    
    def _write_fallback_log(self, records: List[logging.LogRecord], error: Exception):
        """Write logs to fallback file when database write fails.
        
        Args:
            records: Records that failed to write
            error: The error that occurred
        """
        try:
            from pathlib import Path
            fallback_dir = Path.home() / ".biodiversity" / "logs" / "fallback"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            
            fallback_file = fallback_dir / f"db_handler_{datetime.now():%Y%m%d}.log"
            
            with open(fallback_file, 'a') as f:
                f.write(f"\n=== Database write failed at {datetime.now()} ===\n")
                f.write(f"Error: {error}\n")
                f.write(f"Records:\n")
                
                for record in records:
                    f.write(json.dumps({
                        'timestamp': record.created,
                        'level': record.levelname,
                        'logger': record.name,
                        'message': record.getMessage(),
                        'context': getattr(record, 'context', {})
                    }) + '\n')
                    
        except Exception as fallback_error:
            sys.stderr.write(f"Fallback log write failed: {fallback_error}\n")
    
    def flush(self):
        """Force flush any pending logs."""
        # Signal worker to flush
        with self._batch_lock:
            if self.batch:
                self._flush_batch()
    
    def close(self):
        """Cleanup handler and flush remaining logs."""
        # Stop worker thread
        self._stop_event.set()
        
        # Give worker time to finish
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=10)
        
        # Final flush
        self.flush()
        
        # Log statistics
        if self._records_dropped > 0 or self._write_errors > 0:
            sys.stderr.write(
                f"DatabaseLogHandler stats: "
                f"written={self._records_written}, "
                f"dropped={self._records_dropped}, "
                f"errors={self._write_errors}\n"
            )
        
        super().close()
    
    def get_stats(self) -> Dict[str, int]:
        """Get handler statistics.
        
        Returns:
            Dict with records_written, records_dropped, write_errors
        """
        return {
            'records_written': self._records_written,
            'records_dropped': self._records_dropped,
            'write_errors': self._write_errors,
            'queue_size': self.queue.qsize(),
            'batch_size': len(self.batch)
        }