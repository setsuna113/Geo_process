#!/usr/bin/env python3
"""Validate that the monitoring and logging system is properly set up."""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def check_import(module_path, component_name):
    """Check if a module can be imported."""
    try:
        exec(f"from {module_path} import {component_name}")
        print(f"{GREEN}✓{RESET} {module_path}.{component_name}")
        return True
    except ImportError as e:
        print(f"{RED}✗{RESET} {module_path}.{component_name}: {e}")
        return False


def check_database_tables():
    """Check if monitoring tables exist."""
    print(f"\n{BLUE}Checking database tables...{RESET}")
    
    try:
        from src.database.connection import DatabaseManager
        from src.config import config
        
        db = DatabaseManager()
        
        tables_to_check = [
            'pipeline_logs',
            'pipeline_events', 
            'pipeline_progress',
            'pipeline_metrics'
        ]
        
        all_exist = True
        with db.get_cursor() as cursor:
            for table in tables_to_check:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = %s
                    )
                """, (table,))
                exists = cursor.fetchone()['exists']
                
                if exists:
                    print(f"{GREEN}✓{RESET} Table '{table}' exists")
                else:
                    print(f"{RED}✗{RESET} Table '{table}' missing")
                    all_exist = False
        
        return all_exist
        
    except Exception as e:
        print(f"{RED}✗{RESET} Database check failed: {e}")
        return False


def check_config():
    """Check configuration for monitoring settings."""
    print(f"\n{BLUE}Checking configuration...{RESET}")
    
    try:
        from src.config import config
        
        required_settings = [
            ('logging.level', 'INFO'),
            ('logging.database.batch_size', 100),
            ('monitoring.metrics_interval', 10.0)
        ]
        
        all_good = True
        for key, default in required_settings:
            value = config.get(key, default)
            if value:
                print(f"{GREEN}✓{RESET} {key} = {value}")
            else:
                print(f"{YELLOW}⚠{RESET} {key} not set (default: {default})")
        
        return all_good
        
    except Exception as e:
        print(f"{RED}✗{RESET} Config check failed: {e}")
        return False


def test_logging():
    """Test basic logging functionality."""
    print(f"\n{BLUE}Testing logging...{RESET}")
    
    try:
        from src.infrastructure.logging import get_logger, experiment_context
        
        # Set test context
        experiment_context.set("validation-test")
        
        logger = get_logger("test.validation")
        logger.info("Test log message")
        logger.log_performance("test_operation", 0.123, items=10)
        
        print(f"{GREEN}✓{RESET} Structured logging works")
        
        # Test error logging
        try:
            raise ValueError("Test error")
        except ValueError:
            logger.error("Test error capture", exc_info=True)
            print(f"{GREEN}✓{RESET} Error logging with traceback works")
        
        return True
        
    except Exception as e:
        print(f"{RED}✗{RESET} Logging test failed: {e}")
        return False
    finally:
        experiment_context.set(None)


def test_monitoring():
    """Test basic monitoring functionality."""
    print(f"\n{BLUE}Testing monitoring...{RESET}")
    
    try:
        from src.infrastructure.monitoring import UnifiedMonitor
        from src.config import config
        from src.database.connection import DatabaseManager
        
        db = DatabaseManager()
        monitor = UnifiedMonitor(config, db)
        
        # Start and stop monitoring
        monitor.start("validation-test", "validation-job")
        monitor.record_metrics(test_metric=42)
        monitor.stop()
        
        print(f"{GREEN}✓{RESET} Monitoring works")
        return True
        
    except Exception as e:
        print(f"{RED}✗{RESET} Monitoring test failed: {e}")
        return False


def check_scripts():
    """Check if required scripts exist and are executable."""
    print(f"\n{BLUE}Checking scripts...{RESET}")
    
    scripts_to_check = [
        'scripts/monitor.py',
        'scripts/test_monitoring.sh',
        'scripts/production/run_monitored_pipeline.sh',
        'run_monitored.sh'
    ]
    
    all_exist = True
    for script in scripts_to_check:
        script_path = project_root / script
        if script_path.exists():
            executable = os.access(script_path, os.X_OK)
            if executable or script.endswith('.py'):
                print(f"{GREEN}✓{RESET} {script}")
            else:
                print(f"{YELLOW}⚠{RESET} {script} (not executable)")
        else:
            print(f"{RED}✗{RESET} {script} (missing)")
            all_exist = False
    
    return all_exist


def main():
    """Run all validation checks."""
    print(f"{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Monitoring and Logging System Validation{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    # Track overall status
    all_passed = True
    
    # Check imports
    print(f"\n{BLUE}Checking component imports...{RESET}")
    components = [
        ("src.infrastructure.logging", "get_logger"),
        ("src.infrastructure.logging", "setup_logging"),
        ("src.infrastructure.logging.structured_logger", "StructuredLogger"),
        ("src.infrastructure.logging.handlers.database_handler", "DatabaseLogHandler"),
        ("src.infrastructure.logging.context", "LoggingContext"),
        ("src.infrastructure.monitoring", "UnifiedMonitor"),
        ("src.core.process_controller_enhanced", "EnhancedProcessController"),
        ("src.pipelines.orchestrator_enhanced", "EnhancedPipelineOrchestrator"),
        ("src.core.signal_handler_enhanced", "EnhancedSignalHandler"),
    ]
    
    for module, component in components:
        if not check_import(module, component):
            all_passed = False
    
    # Check database
    if not check_database_tables():
        all_passed = False
        print(f"\n{YELLOW}⚠ Run 'python scripts/migrate_monitoring_schema.py' to create tables{RESET}")
    
    # Check config
    if not check_config():
        all_passed = False
    
    # Check scripts
    if not check_scripts():
        all_passed = False
    
    # Test functionality
    if not test_logging():
        all_passed = False
    
    if not test_monitoring():
        all_passed = False
    
    # Summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    if all_passed:
        print(f"{GREEN}✅ All checks passed! The monitoring system is ready to use.{RESET}")
        print(f"\n{BLUE}Next steps:{RESET}")
        print("1. Run a test pipeline: ./run_monitored.sh --experiment-name test")
        print("2. Monitor it: python scripts/monitor.py watch test")
        print("3. Check logs: python scripts/monitor.py logs test")
    else:
        print(f"{RED}❌ Some checks failed. Please fix the issues above.{RESET}")
        print(f"\n{BLUE}Common fixes:{RESET}")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Run database migration: python scripts/migrate_monitoring_schema.py")
        print("3. Update config.yml with monitoring settings")
        print("4. Check file permissions on scripts")
    
    print(f"{BLUE}{'='*60}{RESET}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())