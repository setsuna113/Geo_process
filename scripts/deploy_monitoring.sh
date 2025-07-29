#!/bin/bash
# Deployment script for Unified Monitoring and Logging System

echo "🚀 Deploying Unified Monitoring and Logging System"
echo "================================================="

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;94m'
NC='\033[0m' # No Color

# Deployment mode
MODE="check"
FORCE=false
ROLLBACK=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --deploy)
            MODE="deploy"
            shift
            ;;
        --check)
            MODE="check"
            shift
            ;;
        --rollback)
            MODE="rollback"
            ROLLBACK=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --check      Check deployment readiness (default)"
            echo "  --deploy     Deploy the monitoring system"
            echo "  --rollback   Rollback to previous version"
            echo "  --force      Force deployment without confirmations"
            echo "  --help       Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Function to check Python environment
check_python_env() {
    echo -e "\n${BLUE}Checking Python environment...${NC}"
    
    if ! python --version &> /dev/null; then
        echo -e "${RED}✗ Python not found${NC}"
        return 1
    fi
    
    PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo -e "${GREEN}✓ Python ${PYTHON_VERSION} found${NC}"
    
    # Check required packages
    REQUIRED_PACKAGES=(
        "psycopg2"
        "psutil"
        "tabulate"
        "pytest"
    )
    
    MISSING_PACKAGES=()
    for package in "${REQUIRED_PACKAGES[@]}"; do
        if python -c "import $package" 2>/dev/null; then
            echo -e "${GREEN}✓ Package '$package' installed${NC}"
        else
            echo -e "${RED}✗ Package '$package' missing${NC}"
            MISSING_PACKAGES+=($package)
        fi
    done
    
    if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
        echo -e "\n${YELLOW}Missing packages: ${MISSING_PACKAGES[*]}${NC}"
        echo "Install with: pip install ${MISSING_PACKAGES[*]}"
        return 1
    fi
    
    return 0
}

# Function to check database connectivity
check_database() {
    echo -e "\n${BLUE}Checking database connectivity...${NC}"
    
    python - <<EOF
import sys
sys.path.insert(0, '$PROJECT_ROOT')
try:
    from src.database.connection import DatabaseManager
    db = DatabaseManager()
    with db.get_cursor() as cursor:
        cursor.execute("SELECT version()")
        version = cursor.fetchone()
        print("\033[32m✓ Database connected: PostgreSQL " + version['version'].split()[1] + "\033[0m")
except Exception as e:
    print(f"\033[31m✗ Database connection failed: {e}\033[0m")
    sys.exit(1)
EOF
    
    return $?
}

# Function to backup existing tables
backup_tables() {
    echo -e "\n${BLUE}Creating backup of existing data...${NC}"
    
    BACKUP_DIR="$PROJECT_ROOT/backups/monitoring_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    python - <<EOF
import sys
import json
from pathlib import Path
sys.path.insert(0, '$PROJECT_ROOT')

try:
    from src.database.connection import DatabaseManager
    db = DatabaseManager()
    
    tables = ['experiments', 'processing_jobs', 'pipeline_checkpoints']
    
    for table in tables:
        with db.get_cursor() as cursor:
            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                )
            """, (table,))
            
            if cursor.fetchone()['exists']:
                # Export table data
                cursor.execute(f"SELECT * FROM {table}")
                data = cursor.fetchall()
                
                # Save to JSON
                output_file = Path('$BACKUP_DIR') / f"{table}.json"
                with open(output_file, 'w') as f:
                    json.dump(data, f, default=str, indent=2)
                
                print(f"\033[32m✓ Backed up {table}: {len(data)} records\033[0m")
            else:
                print(f"\033[33m⚠ Table {table} not found, skipping\033[0m")
    
    print(f"\n\033[32m✓ Backup saved to: $BACKUP_DIR\033[0m")
    
except Exception as e:
    print(f"\033[31m✗ Backup failed: {e}\033[0m")
    sys.exit(1)
EOF
    
    return $?
}

# Function to deploy database schema
deploy_database_schema() {
    echo -e "\n${BLUE}Deploying database schema...${NC}"
    
    # Run migration script
    if python "$PROJECT_ROOT/scripts/migrate_monitoring_schema.py"; then
        echo -e "${GREEN}✓ Database schema deployed${NC}"
        return 0
    else
        echo -e "${RED}✗ Database schema deployment failed${NC}"
        return 1
    fi
}

# Function to update configuration
update_configuration() {
    echo -e "\n${BLUE}Updating configuration...${NC}"
    
    CONFIG_FILE="$PROJECT_ROOT/config.yml"
    BACKUP_CONFIG="$PROJECT_ROOT/config.yml.backup_$(date +%Y%m%d_%H%M%S)"
    
    if [ -f "$CONFIG_FILE" ]; then
        # Backup existing config
        cp "$CONFIG_FILE" "$BACKUP_CONFIG"
        echo -e "${GREEN}✓ Config backed up to: ${BACKUP_CONFIG}${NC}"
    fi
    
    # Check if monitoring config exists
    if grep -q "monitoring:" "$CONFIG_FILE" 2>/dev/null; then
        echo -e "${GREEN}✓ Monitoring configuration already present${NC}"
    else
        echo -e "${YELLOW}⚠ Adding monitoring configuration...${NC}"
        
        # Append monitoring config
        cat >> "$CONFIG_FILE" <<EOF

# Monitoring and Logging Configuration
logging:
  level: INFO
  database:
    enabled: true
    batch_size: 100
    flush_interval: 5.0
  file:
    enabled: true
    max_size_mb: 100
    backup_count: 5

monitoring:
  enabled: true
  metrics_interval: 10.0
  progress_update_interval: 5.0
  enable_resource_tracking: true
  
  # Performance tuning
  database:
    connection_pool_size: 10
    max_batch_size: 1000
    
  # Alerts (future feature)
  alerts:
    error_threshold: 10  # errors per minute
    memory_threshold_mb: 8192
    disk_threshold_percent: 90
EOF
        
        echo -e "${GREEN}✓ Monitoring configuration added${NC}"
    fi
    
    return 0
}

# Function to validate deployment
validate_deployment() {
    echo -e "\n${BLUE}Validating deployment...${NC}"
    
    # Run validation script
    if python "$PROJECT_ROOT/scripts/validate_monitoring_setup.py"; then
        echo -e "${GREEN}✓ Deployment validation passed${NC}"
        return 0
    else
        echo -e "${RED}✗ Deployment validation failed${NC}"
        return 1
    fi
}

# Function to run smoke tests
run_smoke_tests() {
    echo -e "\n${BLUE}Running smoke tests...${NC}"
    
    # Create test experiment
    python - <<EOF
import sys
import time
sys.path.insert(0, '$PROJECT_ROOT')

try:
    from src.infrastructure.logging import get_logger, setup_logging
    from src.infrastructure.monitoring import UnifiedMonitor
    from src.database.connection import DatabaseManager
    from src.config import config
    
    # Setup
    db = DatabaseManager()
    setup_logging(config, db, console=False, database=True)
    
    # Test logging
    logger = get_logger("deployment.test")
    logger.info("Deployment smoke test")
    logger.error("Test error", extra={'context': {'test': True}})
    
    # Test monitoring
    monitor = UnifiedMonitor(config, db)
    monitor.start("deploy-test", "smoke-test")
    monitor.record_metrics(test_metric=42)
    time.sleep(0.5)
    monitor.stop()
    
    # Verify data was written
    with db.get_cursor() as cursor:
        cursor.execute("""
            SELECT COUNT(*) as count FROM pipeline_logs 
            WHERE logger_name = 'deployment.test'
        """)
        log_count = cursor.fetchone()['count']
        
        if log_count > 0:
            print(f"\033[32m✓ Logging test passed: {log_count} logs written\033[0m")
        else:
            print("\033[31m✗ Logging test failed: No logs written\033[0m")
            sys.exit(1)
    
    print("\033[32m✓ All smoke tests passed\033[0m")
    
except Exception as e:
    print(f"\033[31m✗ Smoke test failed: {e}\033[0m")
    sys.exit(1)
EOF
    
    return $?
}

# Function to display deployment summary
show_deployment_summary() {
    echo -e "\n${BLUE}Deployment Summary${NC}"
    echo "=================="
    
    cat <<EOF

Monitoring System Components:
- Database tables: pipeline_logs, pipeline_events, pipeline_progress, pipeline_metrics
- Enhanced components: ProcessController, PipelineOrchestrator, SignalHandler
- CLI tool: scripts/monitor.py
- Run scripts: run_monitored.sh

Key Commands:
- Monitor status: python scripts/monitor.py status <experiment>
- View logs: python scripts/monitor.py logs <experiment>
- Watch live: python scripts/monitor.py watch <experiment>
- Run with monitoring: ./run_monitored.sh --experiment-name <name>

Configuration:
- Main config: config.yml
- Logs stored in: PostgreSQL database
- Log files in: logs/

Next Steps:
1. Test with a sample pipeline
2. Monitor resource usage
3. Set up alerts (if needed)
4. Train team on new tools

EOF
}

# Function to perform rollback
perform_rollback() {
    echo -e "\n${BLUE}Performing rollback...${NC}"
    
    # Find latest backup
    LATEST_BACKUP=$(ls -t "$PROJECT_ROOT/backups" 2>/dev/null | head -1)
    
    if [ -z "$LATEST_BACKUP" ]; then
        echo -e "${RED}✗ No backup found${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}Rolling back from backup: $LATEST_BACKUP${NC}"
    
    # Restore configuration
    CONFIG_BACKUPS=$(ls -t "$PROJECT_ROOT"/config.yml.backup_* 2>/dev/null | head -1)
    if [ -n "$CONFIG_BACKUPS" ]; then
        cp "$CONFIG_BACKUPS" "$PROJECT_ROOT/config.yml"
        echo -e "${GREEN}✓ Configuration restored${NC}"
    fi
    
    # Note: Database rollback would need more sophisticated handling
    echo -e "${YELLOW}Note: Database changes need manual rollback${NC}"
    echo "To remove monitoring tables:"
    echo "  DROP TABLE IF EXISTS pipeline_logs, pipeline_events, pipeline_progress, pipeline_metrics CASCADE;"
    
    return 0
}

# Main deployment flow
main() {
    case $MODE in
        check)
            echo -e "\n${BLUE}Checking deployment readiness...${NC}"
            
            check_python_env || exit 1
            check_database || exit 1
            
            echo -e "\n${GREEN}✅ System ready for deployment${NC}"
            echo "Run with --deploy to proceed"
            ;;
            
        deploy)
            echo -e "\n${BLUE}Starting deployment...${NC}"
            
            # Pre-flight checks
            check_python_env || exit 1
            check_database || exit 1
            
            if [ "$FORCE" != true ]; then
                echo -e "\n${YELLOW}This will deploy the monitoring system.${NC}"
                read -p "Continue? (y/N) " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    echo "Deployment cancelled"
                    exit 0
                fi
            fi
            
            # Deployment steps
            backup_tables || exit 1
            deploy_database_schema || exit 1
            update_configuration || exit 1
            validate_deployment || exit 1
            run_smoke_tests || exit 1
            
            echo -e "\n${GREEN}✅ Deployment completed successfully!${NC}"
            show_deployment_summary
            ;;
            
        rollback)
            echo -e "\n${BLUE}Starting rollback...${NC}"
            
            if [ "$FORCE" != true ]; then
                echo -e "\n${YELLOW}This will rollback the monitoring system.${NC}"
                read -p "Continue? (y/N) " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    echo "Rollback cancelled"
                    exit 0
                fi
            fi
            
            perform_rollback
            ;;
            
        *)
            echo -e "${RED}Invalid mode: $MODE${NC}"
            exit 1
            ;;
    esac
}

# Run main function
main