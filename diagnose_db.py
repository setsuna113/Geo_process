#!/usr/bin/env python3
"""
Database connection test script to diagnose issues.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    import psycopg2
    print("✅ psycopg2 imported successfully")
except ImportError as e:
    print(f"❌ Failed to import psycopg2: {e}")
    sys.exit(1)

# Test direct connection with default settings
def test_direct_connection():
    """Test direct database connection."""
    try:
        print("\n🔍 Testing direct database connection...")
        
        # Get connection parameters
        host = os.getenv('DB_HOST', 'localhost')
        port = int(os.getenv('DB_PORT', 5432))
        database = os.getenv('DB_NAME', 'geoprocess_db')
        user = os.getenv('DB_USER', os.getenv('USER'))
        password = os.getenv('DB_PASSWORD', '123456')
        
        print(f"   Host: {host}:{port}")
        print(f"   Database: {database}")
        print(f"   User: {user}")
        print(f"   Password: {'*' * len(password) if password else 'None'}")
        
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            connect_timeout=10
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        result = cursor.fetchone()
        
        if result:
            print(f"✅ Connection successful!")
            print(f"   PostgreSQL version: {result[0]}")
        else:
            print(f"✅ Connection successful (no version info)")
        
        cursor.close()
        conn.close()
        return True
        
    except psycopg2.OperationalError as e:
        print(f"❌ Connection failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_database_exists():
    """Check if target database exists."""
    try:
        print("\n🔍 Checking if target database exists...")
        
        # Connect to postgres database to check if target exists
        host = os.getenv('DB_HOST', 'localhost')
        port = int(os.getenv('DB_PORT', 5432))
        user = os.getenv('DB_USER', os.getenv('USER'))
        password = os.getenv('DB_PASSWORD', '123456')
        target_db = os.getenv('DB_NAME', 'geoprocess_db')
        
        conn = psycopg2.connect(
            host=host,
            port=port,
            database='postgres',  # Connect to default postgres db
            user=user,
            password=password,
            connect_timeout=10
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT datname FROM pg_database WHERE datname = %s;", (target_db,))
        result = cursor.fetchone()
        
        if result:
            print(f"✅ Database '{target_db}' exists")
            return True
        else:
            print(f"❌ Database '{target_db}' does not exist")
            return False
            
    except Exception as e:
        print(f"❌ Failed to check database existence: {e}")
        return False

def create_database_if_missing():
    """Create the target database if it doesn't exist."""
    target_db = os.getenv('DB_NAME', 'geoprocess_db')
    
    try:
        print("\n🔧 Creating target database...")
        
        host = os.getenv('DB_HOST', 'localhost')
        port = int(os.getenv('DB_PORT', 5432))
        user = os.getenv('DB_USER', os.getenv('USER'))
        password = os.getenv('DB_PASSWORD', '123456')
        
        # Connect to postgres database
        conn = psycopg2.connect(
            host=host,
            port=port,
            database='postgres',
            user=user,
            password=password,
            connect_timeout=10
        )
        
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Create database
        cursor.execute(f'CREATE DATABASE "{target_db}";')
        print(f"✅ Database '{target_db}' created successfully")
        
        cursor.close()
        conn.close()
        return True
        
    except psycopg2.errors.DuplicateDatabase:
        print(f"ℹ️ Database '{target_db}' already exists")
        return True
    except Exception as e:
        print(f"❌ Failed to create database: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("DATABASE CONNECTION DIAGNOSTIC")
    print("=" * 60)
    
    # Check if we can connect to postgres database first
    if not test_database_exists():
        # Try to create the database
        if not create_database_if_missing():
            print("\n💥 Cannot create target database")
            sys.exit(1)
    
    # Test connection to target database
    if test_direct_connection():
        print("\n🎉 Database connection test PASSED!")
        sys.exit(0)
    else:
        print("\n💥 Database connection test FAILED!")
        sys.exit(1)
