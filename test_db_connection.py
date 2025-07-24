#!/usr/bin/env python3
"""Test database connection directly without global initialization."""

import psycopg2
import os

def test_connection_params():
    """Test different connection parameters to find what works."""
    
    # Parameters that work with psql
    test_configs = [
        {
            'name': 'Unix socket with different socket dir',
            'params': {
                'host': '/var/run/postgresql',
                'database': 'geoprocess_db',
                'user': 'jason',
            }
        },
        {
            'name': 'Unix socket with tmp',
            'params': {
                'host': '/tmp',
                'database': 'geoprocess_db',
                'user': 'jason',
            }
        },
        {
            'name': 'Local socket (no host)',
            'params': {
                'database': 'geoprocess_db',
                'user': 'jason',
            }
        },
        {
            'name': 'Localhost with user jason (no password)',
            'params': {
                'host': 'localhost',
                'port': 5432,
                'database': 'geoprocess_db',
                'user': 'jason',
            }
        }
    ]
    
    for config in test_configs:
        print(f"\n=== Testing: {config['name']} ===")
        try:
            conn = psycopg2.connect(**config['params'])
            cur = conn.cursor()
            
            # Test basic connection
            cur.execute("SELECT version();")
            pg_version = cur.fetchone()[0]
            print(f"✅ PostgreSQL connected: {pg_version.split(',')[0]}")
            
            # Test PostGIS
            cur.execute("SELECT PostGIS_Version();")
            postgis_version = cur.fetchone()[0]
            print(f"✅ PostGIS available: {postgis_version}")
            
            cur.close()
            conn.close()
            
            print(f"✅ SUCCESS: Connection works with {config['name']}")
            return config['params']
            
        except Exception as e:
            print(f"❌ FAILED: {config['name']} - {e}")
    
    return None

if __name__ == "__main__":
    print("🔍 Testing Database Connection Parameters")
    print("=" * 50)
    
    working_params = test_connection_params()
    
    if working_params:
        print(f"\n🎉 FOUND WORKING CONNECTION:")
        for key, value in working_params.items():
            print(f"   {key}: {value}")
    else:
        print(f"\n❌ NO WORKING CONNECTION FOUND")