#!/usr/bin/env python3
"""Clean up experiment records from database."""
import sys
sys.path.insert(0, '/home/yl998/dev/geo')

from src.database.connection import DatabaseManager
from src.config import config

def clean_experiments():
    """Remove all experiment records from database."""
    db = DatabaseManager()
    
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                # Get count before deletion
                cursor.execute("SELECT COUNT(*) FROM experiments")
                count = cursor.fetchone()[0]
                print(f"Found {count} experiments to delete")
                
                if count > 0:
                    # Delete all experiments (this will cascade to related tables)
                    cursor.execute("DELETE FROM experiments")
                    conn.commit()
                    print(f"✅ Deleted {count} experiments from database")
                else:
                    print("No experiments to delete")
                    
                # Also clean up orphaned resampling cache
                cursor.execute("DELETE FROM resampling_cache")
                cache_count = cursor.rowcount
                if cache_count > 0:
                    print(f"✅ Deleted {cache_count} resampling cache entries")
                    
                # Clean up orphaned raster passthrough data
                cursor.execute("DELETE FROM raster_passthrough_data")
                passthrough_count = cursor.rowcount
                if passthrough_count > 0:
                    print(f"✅ Deleted {passthrough_count} passthrough data entries")
                    
                conn.commit()
                
    except Exception as e:
        print(f"❌ Error cleaning experiments: {e}")
        return False
    finally:
        db.close()
    
    return True

if __name__ == "__main__":
    clean_experiments()