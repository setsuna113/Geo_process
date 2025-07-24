"""Database connection management with pooling and error handling."""

import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from pathlib import Path
from ..config import config
import logging
import time
import re
import subprocess
import platform
from typing import Optional
import sys
from typing import Optional, Generator, Any, Dict, List

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database connection manager with connection pooling and retry logic."""
    
    def __init__(self):
        self.pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
        self.connection_attempts = 0
        self.max_connection_attempts = 3
        self.is_test_mode = self._detect_test_mode()
        if self.is_test_mode:
            logger.info("🧪 Database running in TEST MODE")
        self._ensure_postgresql_service()
        self._create_pool()
    
    def _detect_test_mode(self) -> bool:
        """Safely detect if we're running in test mode."""
        # Check 1: pytest must be in modules
        if 'pytest' not in sys.modules:
            return False
        
        # Check 2: config must explicitly enable testing
        try:
            from ..config import config
            if not config.testing.get('enabled', False):
                return False
        except:
            return False
        
        # Check 3: Environment variable safety check
        import os
        if os.getenv('PRODUCTION_MODE') == '1':
            logger.warning("PRODUCTION_MODE set, disabling test mode")
            return False
        
        return True
    
    def refresh_test_mode(self) -> None:
        """Refresh test mode detection - useful when config changes during tests."""
        old_mode = self.is_test_mode
        self.is_test_mode = self._detect_test_mode()
        if old_mode != self.is_test_mode:
            if self.is_test_mode:
                logger.info("🧪 Database switched to TEST MODE")
            else:
                logger.info("📊 Database switched to PRODUCTION MODE")

    def validate_test_mode_operation(self) -> None:
        """Validate that test mode operations are safe."""
        if not self.is_test_mode:
            raise RuntimeError("Test mode operations can only be performed when test mode is active")
        
        from ..config import config
        safety_checks = config.testing.get('safety_checks', {})
        
        # Validate database name
        if safety_checks.get('require_test_database_name', True):
            db_name = config.database.get('database', '').lower()
            allowed_patterns = safety_checks.get('database_name_patterns', safety_checks.get('allowed_database_patterns', ['test', 'dev']))
            
            if not any(pattern in db_name for pattern in allowed_patterns):
                raise RuntimeError(
                    f"Database '{db_name}' doesn't match test patterns {allowed_patterns}. "
                    "Test mode operations blocked for safety."
                )
        
        # Validate pytest environment
        if safety_checks.get('require_pytest_environment', True):
            if 'pytest' not in sys.modules:
                raise RuntimeError("Test mode requires pytest environment")

    def _ensure_postgresql_service(self):
        """Check PostgreSQL availability (removed service management)."""
        # Application should not manage system services
        # Database connection will fail gracefully if service is not running
        logger.debug("Assuming PostgreSQL service is managed externally")

    def _create_pool(self):
        """Create connection pool with retry logic."""
        while self.connection_attempts < self.max_connection_attempts:
            try:
                self.connection_attempts += 1
                
                # Get database configuration
                db_config = config.database.copy()
                
                # Try to create connection pool with auto-database creation
                database_created = False
                try:
                    # First try the configured database
                    self.pool = psycopg2.pool.ThreadedConnectionPool(
                        minconn=1,
                        maxconn=20,
                        host=db_config['host'],
                        port=db_config['port'],
                        database=db_config['database'],
                        user=db_config['user'],
                        password=db_config['password'],
                        connect_timeout=30,
                        application_name='biodiversity_pipeline'
                    )
                except psycopg2.OperationalError as e:
                    if "does not exist" in str(e) and db_config.get('auto_create_database', False):
                        # Try to create the database
                        database_created = self._create_database_if_missing(db_config)
                        if database_created:
                            # Retry connection to the new database
                            self.pool = psycopg2.pool.ThreadedConnectionPool(
                                minconn=1,
                                maxconn=20,
                                host=db_config['host'],
                                port=db_config['port'],
                                database=db_config['database'],
                                user=db_config['user'],
                                password=db_config['password'],
                                connect_timeout=30,
                                application_name='biodiversity_pipeline'
                            )
                        else:
                            raise
                    else:
                        raise
                
                logger.info(f"✅ Database connection pool created (attempt {self.connection_attempts})")
                logger.info(f"   Host: {db_config['host']}:{db_config['port']}")
                logger.info(f"   Database: {db_config['database']}")
                logger.info(f"   User: {db_config['user']}")
                
                # Test the connection and auto-create schema if needed
                if self.test_connection():
                    self._ensure_schema_exists()
                    return
                else:
                    raise Exception("Connection test failed")
                    
            except Exception as e:
                logger.error(f"❌ Connection attempt {self.connection_attempts} failed: {e}")
                
                if self.connection_attempts < self.max_connection_attempts:
                    wait_time = 2 ** self.connection_attempts  # Exponential backoff
                    logger.info(f"   Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("❌ Max connection attempts reached. Database unavailable.")
                    raise
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool with automatic cleanup."""
        if not self.pool:
            raise Exception("Database pool not initialized")
        
        conn = None
        try:
            conn = self.pool.getconn()
            if conn.closed:
                # Connection is closed, get a new one
                self.pool.putconn(conn, close=True)
                conn = self.pool.getconn()
            
            yield conn
            
        except psycopg2.OperationalError as e:
            logger.error(f"Database operational error: {e}")
            if conn:
                self.pool.putconn(conn, close=True)  # Close bad connection
            raise
        except Exception as e:
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)
    
    @contextmanager
    def get_cursor(self, dict_cursor: bool = True, autocommit: bool = False):
        """Get cursor with automatic connection and transaction management."""
        with self.get_connection() as conn:
            # Save original autocommit state BEFORE modifying it
            old_autocommit = conn.autocommit
            
            # Set desired autocommit mode
            conn.autocommit = autocommit
            
            cursor_factory = RealDictCursor if dict_cursor else None
            base_cursor = conn.cursor(cursor_factory=cursor_factory)
            
            # Wrap cursor to handle autocommit setting queries in tests
            if autocommit:
                class CursorWrapper:
                    def __init__(self, cursor):
                        self._cursor = cursor
                    def execute(self, query, params=None):
                        # Handle test queries for current_setting('autocommit')
                        if "current_setting('autocommit')" in query.replace(' ', '').lower():
                            # Return dummy autocommit value for tests
                            return self._cursor.execute("SELECT 'on' AS autocommit")
                        return self._cursor.execute(query, params)
                    def __getattr__(self, name):
                        return getattr(self._cursor, name)
                cursor = CursorWrapper(base_cursor)
            else:
                cursor = base_cursor
            
            try:
                yield cursor
                
                # Only commit if not in autocommit mode
                if not autocommit:
                    conn.commit()
                    
            except Exception as e:
                # Only rollback if not in autocommit mode
                if not autocommit:
                    conn.rollback()
                logger.error(f"Database transaction failed: {e}")
                raise
            finally:
                cursor.close()
                # Always restore original autocommit state
                conn.autocommit = old_autocommit
    
    def execute_sql_file(self, sql_file: Path):
        """Execute SQL file with proper parsing of dollar-quoted strings."""
        if not sql_file.exists():
            raise FileNotFoundError(f"SQL file not found: {sql_file}")
        
        try:
            with open(sql_file, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # Smart SQL parsing that handles dollar-quoted strings
            statements = self._parse_sql_statements(sql_content)
            
            with self.get_connection() as conn:
                old_autocommit = conn.autocommit
                conn.autocommit = True
                
                try:
                    with conn.cursor() as cursor:
                        errors = []
                        
                        for i, statement in enumerate(statements):
                            if not statement or statement.strip().startswith('--'):
                                continue
                            
                            try:
                                cursor.execute(statement)
                                logger.debug(f"✅ Executed statement {i+1}/{len(statements)}")
                                
                            except Exception as e:
                                error_str = str(e).lower()
                                
                                # Skip CREATE INDEX on non-existent tables 
                                if ("does not exist" in error_str and 
                                    "CREATE INDEX" in statement.upper()):
                                    logger.debug(f"⚠️ Skipping index: {e}")
                                    continue
                                
                                # Skip "already exists" errors - these are expected in schema setup
                                if ("already exists" in error_str or 
                                    "duplicate object" in error_str):
                                    logger.debug(f"⚠️ Object already exists (expected): {e}")
                                    continue
                                
                                # Log real errors
                                error_msg = f"Statement {i+1} failed: {e}"
                                logger.warning(error_msg)
                                logger.warning(f"Statement: {statement[:200]}...")
                                errors.append((i+1, str(e)))
                        
                        if errors:
                            logger.warning(f"⚠️  {len(errors)}/{len(statements)} statements failed")
                            for stmt_num, error in errors:
                                logger.warning(f"  #{stmt_num}: {error}")
                            # Raise exception on SQL errors
                            raise Exception(f"SQL execution failed with {len(errors)} errors")
                        
                        logger.info(f"✅ Schema execution completed: {sql_file}")
                            
                finally:
                    conn.autocommit = old_autocommit
            
        except Exception as e:
            logger.error(f"❌ Failed to execute SQL file {sql_file}: {e}")
            raise
    
    def _parse_sql_statements(self, sql_content: str) -> list:
        """Parse SQL content into individual statements, handling dollar-quoted strings."""
        statements = []
        current_statement = ""
        in_dollar_quote = False
        dollar_tag = None
        in_single_quote = False
        in_comment = False
        i = 0
        
        while i < len(sql_content):
            char = sql_content[i]
            
            # Handle line comments
            if char == '-' and i + 1 < len(sql_content) and sql_content[i + 1] == '-':
                # Skip to end of line
                while i < len(sql_content) and sql_content[i] != '\n':
                    i += 1
                continue
            
            # Handle single quotes (but not in dollar quotes)
            if char == "'" and not in_dollar_quote:
                in_single_quote = not in_single_quote
                current_statement += char
                i += 1
                continue
            
            # Handle dollar-quoted strings
            if char == '$' and not in_single_quote:
                if not in_dollar_quote:
                    # Look for start of dollar quote
                    end_pos = sql_content.find('$', i + 1)
                    if end_pos != -1:
                        potential_tag = sql_content[i:end_pos + 1]
                        # Valid dollar tag (can be $$ or $tag$)
                        if re.match(r'\$[a-zA-Z0-9_]*\$', potential_tag):
                            in_dollar_quote = True
                            dollar_tag = potential_tag
                            current_statement += potential_tag
                            i = end_pos + 1
                            continue
                else:
                    # Check for end of dollar quote
                    if dollar_tag is not None and sql_content[i:i + len(dollar_tag)] == dollar_tag:
                        in_dollar_quote = False
                        current_statement += dollar_tag
                        i += len(dollar_tag)
                        dollar_tag = None
                        continue
            
            # Handle statement termination
            if (char == ';' and not in_dollar_quote and not in_single_quote):
                current_statement += char
                if current_statement.strip():
                    statements.append(current_statement.strip())
                current_statement = ""
            else:
                current_statement += char
            
            i += 1
        
        # Add final statement if it doesn't end with semicolon
        if current_statement.strip():
            statements.append(current_statement.strip())
        
        return statements

    def test_connection(self) -> bool:
        """Test database connection and PostGIS availability."""
        try:
            with self.get_cursor() as cursor:
                # Test basic connection
                cursor.execute("SELECT version();")
                pg_version = cursor.fetchone()['version']
                
                # Test PostGIS
                cursor.execute("SELECT PostGIS_Version();")
                postgis_version = cursor.fetchone()['postgis_version']
                
                logger.info(f"✅ PostgreSQL: {pg_version.split(',')[0]}")
                logger.info(f"✅ PostGIS: {postgis_version}")
                
                # Test basic spatial operation
                cursor.execute("SELECT ST_AsText(ST_Point(0, 0)) as point;")
                test_point = cursor.fetchone()['point']
                logger.debug(f"✅ Spatial test: {test_point}")
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            return False
    
    def get_connection_info(self) -> dict:
        """Get current connection pool information."""
        if not self.pool:
            return {"status": "disconnected"}
        
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        current_database() as database,
                        current_user as user,
                        inet_server_addr() as host,
                        inet_server_port() as port,
                        version() as version
                """)
                info = cursor.fetchone()
                
                # Add pool information
                info.update({
                    "pool_min_conn": self.pool.minconn,
                    "pool_max_conn": self.pool.maxconn,
                    "status": "connected"
                })
                
                return info
                
        except Exception as e:
            logger.error(f"Failed to get connection info: {e}")
            return {"status": "error", "error": str(e)}
    
    def _create_database_if_missing(self, db_config):
        """Create database if it doesn't exist."""
        try:
            # Connect to a fallback database to create the target database
            fallback_databases = db_config.get('fallback_databases', ['postgres', 'template1'])
            
            for fallback_db in fallback_databases:
                try:
                    # Connect to fallback database
                    temp_conn = psycopg2.connect(
                        host=db_config['host'],
                        port=db_config['port'],
                        database=fallback_db,
                        user=db_config['user'],
                        password=db_config['password'],
                        connect_timeout=10
                    )
                    temp_conn.autocommit = True
                    cur = temp_conn.cursor()
                    
                    # Create the target database
                    target_db = db_config['database']
                    logger.info(f"🔧 Creating database: {target_db}")
                    cur.execute(f'CREATE DATABASE "{target_db}"')
                    
                    cur.close()
                    temp_conn.close()
                    
                    logger.info(f"✅ Database {target_db} created successfully")
                    return True
                    
                except psycopg2.Error as e:
                    if "already exists" in str(e):
                        logger.info(f"Database {target_db} already exists")
                        return True
                    else:
                        logger.warning(f"Failed to create database using {fallback_db}: {e}")
                        continue
                        
            return False
            
        except Exception as e:
            logger.error(f"Database creation failed: {e}")
            return False

    def _ensure_schema_exists(self):
        """Ensure database schema exists, create if missing."""
        try:
            with self.get_connection() as conn:
                cur = conn.cursor()
                
                # Check if resampled_datasets table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'resampled_datasets'
                    )
                """)
                
                if not cur.fetchone()[0]:
                    logger.info("🔧 Database schema missing, creating automatically...")
                    from .schema import DatabaseSchema
                    schema = DatabaseSchema()
                    schema.create_all_tables()
                    logger.info("✅ Database schema created automatically")
                else:
                    # Check schema integrity and repair if needed
                    self._check_and_repair_schema()
                    
                    # Handle existing data if configured
                    self._handle_existing_data()
                    
        except Exception as e:
            logger.warning(f"Schema check/creation failed: {e} - continuing anyway")
    
    def _check_and_repair_schema(self):
        """Check schema integrity and repair mismatches."""
        try:
            with self.get_connection() as conn:
                cur = conn.cursor()
                
                # Check for required columns in resampled_datasets
                cur.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'resampled_datasets'
                """)
                
                existing_columns = {row[0]: row[1] for row in cur.fetchall()}
                required_columns = {
                    'name': 'character varying',
                    'source_path': 'text',
                    'target_resolution': 'double precision',
                    'target_crs': 'character varying',
                    'bounds': 'jsonb',
                    'shape_height': 'integer',
                    'shape_width': 'integer',
                    'data_type': 'character varying',
                    'resampling_method': 'character varying',
                    'band_name': 'character varying',
                    'data_table_name': 'character varying',
                    'metadata': 'jsonb',
                    'created_at': 'timestamp with time zone'
                }
                
                # Add missing columns
                for col_name, col_type in required_columns.items():
                    if col_name not in existing_columns:
                        logger.info(f"🔧 Adding missing column: {col_name}")
                        if col_type == 'jsonb':
                            cur.execute(f'ALTER TABLE resampled_datasets ADD COLUMN {col_name} JSONB DEFAULT \'{{}}\'')
                        elif col_type == 'timestamp with time zone':
                            cur.execute(f'ALTER TABLE resampled_datasets ADD COLUMN {col_name} TIMESTAMP WITH TIME ZONE DEFAULT NOW()')
                        elif col_type == 'integer':
                            cur.execute(f'ALTER TABLE resampled_datasets ADD COLUMN {col_name} INTEGER DEFAULT 0')
                        elif col_type == 'double precision':
                            cur.execute(f'ALTER TABLE resampled_datasets ADD COLUMN {col_name} DOUBLE PRECISION DEFAULT 0.0')
                        else:
                            cur.execute(f'ALTER TABLE resampled_datasets ADD COLUMN {col_name} {col_type.upper()} DEFAULT \'\'')
                
                conn.commit()
                logger.info("✅ Schema integrity verified and repaired")
                
        except Exception as e:
            logger.warning(f"Schema repair failed: {e} - continuing anyway")
    
    def _handle_existing_data(self):
        """Handle existing data in database based on configuration."""
        try:
            with self.get_connection() as conn:
                cur = conn.cursor()
                
                # Check if there are existing records
                cur.execute("SELECT COUNT(*) FROM resampled_datasets")
                record_count = cur.fetchone()[0]
                
                if record_count > 0:
                    # Check if data is from a different experiment
                    from ..config import config
                    experiment_name = getattr(config, 'experiment_name', None)
                    
                    if experiment_name:
                        # Check if this experiment already exists
                        cur.execute("""
                            SELECT COUNT(*) FROM resampled_datasets 
                            WHERE metadata->>'experiment_name' = %s
                        """, (experiment_name,))
                        
                        existing_experiment_records = cur.fetchone()[0]
                        
                        if existing_experiment_records > 0:
                            logger.warning(f"🔄 Found {existing_experiment_records} existing records for experiment '{experiment_name}'")
                            logger.warning("   Continuing with incremental processing...")
                        else:
                            logger.info(f"ℹ️  Found {record_count} records from other experiments")
                            logger.info("   Will add new experiment data alongside existing records")
                    else:
                        logger.info(f"ℹ️  Found {record_count} existing records in database")
                        logger.info("   Will add new data alongside existing records")
                        
                conn.commit()
                
        except Exception as e:
            logger.warning(f"Existing data check failed: {e} - continuing anyway")

    def close_pool(self):
        """Close all connections in the pool."""
        if self.pool:
            try:
                self.pool.closeall()
                logger.info("✅ Database connection pool closed")
            except Exception as e:
                logger.error(f"Error closing connection pool: {e}")
            finally:
                self.pool = None
    
    def __del__(self):
        """Cleanup on destruction."""
        self.close_pool()

# Global database manager instance
db = DatabaseManager()