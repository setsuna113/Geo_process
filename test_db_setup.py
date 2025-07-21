#!/usr/bin/env python3
"""
Test the database setup process specifically.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_database_setup():
    """Test the database setup process step by step."""
    try:
        print("🔍 Testing database setup process...")
        
        # Import the setup function
        from src.database.setup import setup_database
        from src.database.connection import db
        
        print("✅ Successfully imported database modules")
        
        # Test connection first
        print("\n🔍 Testing database connection...")
        if db.test_connection():
            print("✅ Database connection test passed")
        else:
            print("❌ Database connection test failed")
            return False
        
        # Try to setup database
        print("\n🔧 Running database setup...")
        success = setup_database(reset=False)
        
        if success:
            print("✅ Database setup completed successfully")
            return True
        else:
            print("❌ Database setup failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during database setup test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_extensions():
    """Test if required database extensions are available."""
    try:
        print("\n🔍 Checking database extensions...")
        
        import psycopg2
        import os
        
        # Connect to database
        host = os.getenv('DB_HOST', 'localhost')
        port = int(os.getenv('DB_PORT', 5432))
        database = os.getenv('DB_NAME', 'geoprocess_db')
        user = os.getenv('DB_USER', os.getenv('USER'))
        password = os.getenv('DB_PASSWORD', '123456')
        
        conn = psycopg2.connect(
            host=host, port=port, database=database,
            user=user, password=password
        )
        
        cursor = conn.cursor()
        
        # Check for PostGIS
        cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'postgis');")
        result = cursor.fetchone()
        postgis_exists = result[0] if result else False
        
        # Check for UUID extension
        cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'uuid-ossp');")
        result = cursor.fetchone()
        uuid_exists = result[0] if result else False
        
        print(f"   PostGIS extension: {'✅ Available' if postgis_exists else '❌ Missing'}")
        print(f"   UUID extension: {'✅ Available' if uuid_exists else '❌ Missing'}")
        
        # Try to create extensions if missing
        if not postgis_exists:
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
                conn.commit()
                print("✅ PostGIS extension created")
            except Exception as e:
                print(f"❌ Failed to create PostGIS extension: {e}")
        
        if not uuid_exists:
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")
                conn.commit()
                print("✅ UUID extension created")
            except Exception as e:
                print(f"❌ Failed to create UUID extension: {e}")
        
        cursor.close()
        conn.close()
        
        return postgis_exists or uuid_exists
        
    except Exception as e:
        print(f"❌ Error checking extensions: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("DATABASE SETUP DIAGNOSTIC")
    print("=" * 60)
    
    # Check extensions first
    test_database_extensions()
    
    # Test setup process
    if test_database_setup():
        print("\n🎉 Database setup test PASSED!")
        sys.exit(0)
    else:
        print("\n💥 Database setup test FAILED!")
        sys.exit(1)
