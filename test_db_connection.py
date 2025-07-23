#!/usr/bin/env python3
"""Test database connection in test mode."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Simulate test mode
sys.modules['pytest'] = type(sys)('pytest')

def test_db_connection():
    """Test database connection with test mode config."""
    try:
        from src.config.config import Config
        config = Config()
        
        print("🔍 Database config in test mode:")
        print(f"   Host: {config.get('database.host')}")
        print(f"   Port: {config.get('database.port')}")
        print(f"   Database: {config.get('database.database')}")
        print(f"   User: {config.get('database.user')}")
        
        # Test database connection (but don't fail if DB isn't running)
        try:
            from src.database.connection import DatabaseManager
            db = DatabaseManager()
            if db.test_connection():
                print("✅ Database connection successful!")
            else:
                print("⚠️  Database connection failed - make sure PostgreSQL is running on port 5432")
                print("   Start with: sudo systemctl start postgresql")
        except Exception as e:
            print(f"⚠️  Database connection error: {e}")
            print("   This is normal if PostgreSQL isn't running locally")
            print("   Start with: sudo systemctl start postgresql")
            
        print("✅ Config system working correctly in test mode!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_db_connection()