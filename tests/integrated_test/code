# tests/integration/validators.py
import psycopg2
from pathlib import Path
from typing import Dict, Any, List, Optional
import psutil
import gc

class SystemStateValidator:
    """Validate system state during and after tests."""
    
    def __init__(self, db_params: Dict[str, Any]):
        self.db_params = db_params
        self.initial_state = {}
        
    def capture_initial_state(self) -> None:
        """Capture system state before test."""
        self.initial_state = {
            'memory': psutil.Process().memory_info().rss,
            'open_files': len(psutil.Process().open_files()),
            'db_connections': self._count_db_connections(),
            'temp_files': self._count_temp_files()
        }
    
    def validate_final_state(self) -> List[str]:
        """Validate system state after test. Return list of issues."""
        issues = []
        
        # Check memory
        current_memory = psutil.Process().memory_info().rss
        memory_increase = current_memory - self.initial_state['memory']
        if memory_increase > 100 * 1024 * 1024:  # 100MB threshold
            issues.append(f"Memory leak detected: {memory_increase / 1024 / 1024:.1f}MB increase")
        
        # Check file handles
        current_files = len(psutil.Process().open_files())
        if current_files > self.initial_state['open_files']:
            issues.append(f"File handle leak: {current_files - self.initial_state['open_files']} unclosed files")
        
        # Check database connections
        current_connections = self._count_db_connections()
        if current_connections > self.initial_state['db_connections']:
            issues.append(f"DB connection leak: {current_connections - self.initial_state['db_connections']} extra connections")
        
        return issues
    
    def validate_database_consistency(self) -> List[str]:
        """Check database consistency."""
        issues = []
        conn = psycopg2.connect(**self.db_params)
        cur = conn.cursor()
        
        try:
            # Check for orphaned records
            cur.execute("""
                SELECT COUNT(*) FROM raster_tiles rt
                LEFT JOIN raster_sources rs ON rt.raster_id = rs.id
                WHERE rs.id IS NULL
            """)
            orphaned = cur.fetchone()[0]
            if orphaned > 0:
                issues.append(f"Found {orphaned} orphaned raster tiles")
            
            # Check cache consistency
            cur.execute("""
                SELECT COUNT(*) FROM resampling_cache rc
                WHERE rc.created_at < NOW() - INTERVAL '30 days'
            """)
            expired = cur.fetchone()[0]
            if expired > 0:
                issues.append(f"Found {expired} expired cache entries")
            
            # Check processing queue
            cur.execute("""
                SELECT COUNT(*) FROM processing_queue
                WHERE status = 'processing' AND updated_at < NOW() - INTERVAL '1 hour'
            """)
            stuck = cur.fetchone()[0]
            if stuck > 0:
                issues.append(f"Found {stuck} stuck processing tasks")
                
        finally:
            cur.close()
            conn.close()
            
        return issues
    
    def _count_db_connections(self) -> int:
        """Count active database connections."""
        conn = psycopg2.connect(**self.db_params)
        cur = conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM pg_stat_activity WHERE datname = current_database()")
            return cur.fetchone()[0]
        finally:
            cur.close()
            conn.close()
    
    def _count_temp_files(self) -> int:
        """Count temporary files."""
        temp_dir = Path("/tmp")
        return len(list(temp_dir.glob("test_*")))
    
    def force_cleanup(self) -> None:
        """Force cleanup of resources."""
        gc.collect()
        
        # Close any leaked file handles
        proc = psutil.Process()
        for f in proc.open_files():
            if 'test_' in f.path:
                try:
                    Path(f.path).unlink(missing_ok=True)
                except:
                    pass