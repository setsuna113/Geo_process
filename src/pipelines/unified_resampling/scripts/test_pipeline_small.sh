#!/bin/bash
# Test script for unified resampling pipeline with small dataset samples
# Based on run_unified_resampling.sh but optimized for testing

echo "🧪 Unified Resampling Pipeline - Small Test Runner"
echo "================================================="

# Get script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$PROJECT_ROOT"

echo "📍 Project root: $PROJECT_ROOT"

# Check if we're in the right directory
if [ ! -f "src/pipelines/unified_resampling/scripts/run_unified_resampling.py" ]; then
    echo "❌ Error: Unified resampling script not found. Are you in the right directory?"
    exit 1
fi

echo ""
echo "🔍 Step 1: Checking system dependencies..."

# Check if tmux is available (optional for testing)
if command -v tmux &> /dev/null; then
    echo "✅ tmux is available"
    TMUX_AVAILABLE=true
else
    echo "⚠️ tmux not available - will run without monitoring"
    TMUX_AVAILABLE=false
fi

# Auto-detect Python environment (same logic as main script)
detect_python_env() {
    local possible_paths=(
        "$HOME/anaconda3/envs/geo_py311/bin/python"
        "$HOME/miniconda3/envs/geo_py311/bin/python" 
        "$HOME/conda/envs/geo_py311/bin/python"
        "/opt/anaconda3/envs/geo_py311/bin/python"
        "/opt/miniconda3/envs/geo_py311/bin/python"
    )
    
    # Try current conda environment if active
    if [ ! -z "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/bin/python" ]; then
        echo "$CONDA_PREFIX/bin/python"
        return 0
    fi
    
    # Try common paths
    for path in "${possible_paths[@]}"; do
        if [ -f "$path" ]; then
            echo "$path"
            return 0
        fi
    done
    
    # Fall back to system python if it has required packages
    if command -v python3 &> /dev/null; then
        if python3 -c "import sys; sys.path.insert(0, 'src'); from src.pipelines.unified_resampling import UnifiedResamplingPipeline" 2>/dev/null; then
            echo "python3"
            return 0
        fi
    fi
    
    return 1
}

PYTHON_ENV=$(detect_python_env)
if [ $? -ne 0 ] || [ -z "$PYTHON_ENV" ]; then
    echo "❌ Error: Python environment with required dependencies not found"
    echo "   Please ensure you have the geo_py311 environment or system python with dependencies"
    exit 1
else
    echo "✅ Python environment found: $PYTHON_ENV"
fi

echo ""
echo "🔍 Step 2: Checking test datasets..."

# Check if datasets exist (should be in data/richness_maps based on defaults.py)
DATA_DIR="data/richness_maps"
DARU_FILE="$DATA_DIR/daru-plants-richness.tif"
IUCN_FILE="$DATA_DIR/iucn-terrestrial-richness.tif"

if [ ! -f "$DARU_FILE" ]; then
    echo "❌ Error: Plants dataset not found: $DARU_FILE"
    exit 1
fi

if [ ! -f "$IUCN_FILE" ]; then
    echo "❌ Error: Terrestrial dataset not found: $IUCN_FILE"
    exit 1
fi

echo "✅ Test datasets found:"
echo "   - Plants: $DARU_FILE ($(du -h "$DARU_FILE" | cut -f1))"
echo "   - Terrestrial: $IUCN_FILE ($(du -h "$IUCN_FILE" | cut -f1))"

echo ""
echo "🔍 Step 3: Checking database connection..."

# Test database connection in test mode
$PYTHON_ENV -c "
import sys
sys.path.insert(0, 'src')
sys.modules['pytest'] = type(sys)('pytest')  # Force test mode
try:
    from src.database.connection import DatabaseManager
    db = DatabaseManager()
    if db.test_connection():
        print('✅ Database connection successful!')
        print('   Using test database configuration')
    else:
        print('❌ Database connection failed')
        exit(1)
except Exception as e:
    print(f'❌ Database error: {e}')
    print('   Make sure PostgreSQL is running: sudo systemctl start postgresql')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    echo "💡 Database Setup Instructions:"
    echo "   1. Start PostgreSQL: sudo systemctl start postgresql"
    echo "   2. Create database: sudo -u postgres createdb geoprocess_db"
    echo "   3. Grant permissions: sudo -u postgres psql -c \"GRANT ALL PRIVILEGES ON DATABASE geoprocess_db TO jason;\""
    exit 1
fi

echo ""
echo "🔍 Step 4: Running validation checks..."

# Run dry-run validation
echo "   Running configuration validation..."
$PYTHON_ENV src/pipelines/unified_resampling/scripts/run_unified_resampling.py --dry-run --validate-inputs

if [ $? -ne 0 ]; then
    echo "❌ Error: Pipeline validation failed"
    exit 1
fi

echo "   ✅ Pipeline validation passed"

echo ""
echo "⚙️ Step 5: Test configuration..."

# Small test parameters
TEST_ARGS="--target-resolution 0.2 --memory-limit 2 --max-samples 10000 --som-grid-size 3 3 --som-iterations 50 --experiment-name test_small_pipeline"

echo "   Test parameters:"
echo "     - Target resolution: 0.2° (very coarse for speed)"
echo "     - Memory limit: 2GB"
echo "     - Max samples: 10,000"
echo "     - SOM grid: 3x3"
echo "     - SOM iterations: 50"

echo ""
echo "🧹 Step 6: Pre-test cleanup..."

# Clean up any previous test outputs
if [ -d "outputs/unified_resampling" ]; then
    echo "   Cleaning previous test outputs..."
    rm -rf outputs/unified_resampling/test_*
fi

# Clean up test database entries (optional)
echo "   Cleaning test database entries..."
$PYTHON_ENV -c "
import sys
sys.path.insert(0, 'src')
sys.modules['pytest'] = type(sys)('pytest')  # Force test mode

try:
    from src.database.connection import DatabaseManager
    db = DatabaseManager()
    
    with db.get_connection() as conn:
        cur = conn.cursor()
        
        # Clean up test experiments
        cur.execute('DELETE FROM experiments WHERE name LIKE %s', ('test_%',))
        
        # Clean up test resampled datasets (if table exists)
        cur.execute('''
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'resampled_datasets'
            )
        ''')
        
        if cur.fetchone()[0]:
            cur.execute('DELETE FROM resampled_datasets WHERE name LIKE %s', ('test-%',))
            # Drop any test data tables
            cur.execute('''
                SELECT tablename FROM pg_tables 
                WHERE tablename LIKE 'resampled_test_%'
            ''')
            for (table,) in cur.fetchall():
                cur.execute(f'DROP TABLE IF EXISTS {table} CASCADE')
        
        conn.commit()
        print('✅ Test database cleanup completed')
        
except Exception as e:
    print(f'⚠️ Database cleanup warning: {e}')
" || echo "⚠️ Database cleanup had issues, continuing..."

echo ""
echo "🚀 Step 7: Running test pipeline..."

read -p "   Start the small test pipeline? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "📋 Test cancelled."
    exit 0
fi

echo ""
echo "🎬 Starting test pipeline..."
echo "Command: $PYTHON_ENV src/pipelines/unified_resampling/scripts/run_unified_resampling.py $TEST_ARGS"
echo ""

# Run the test
START_TIME=$(date +%s)
$PYTHON_ENV src/pipelines/unified_resampling/scripts/run_unified_resampling.py $TEST_ARGS

TEST_RESULT=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "📊 Test Results:"
echo "   Duration: ${DURATION}s"

if [ $TEST_RESULT -eq 0 ]; then
    echo "   Status: ✅ SUCCESS"
    
    echo ""
    echo "🔍 Step 8: Verifying outputs..."
    
    # Check outputs
    if [ -d "outputs/unified_resampling" ]; then
        echo "   Output directory contents:"
        ls -la outputs/unified_resampling/ | head -10
    fi
    
    # Check database
    echo "   Database entries created:"
    $PYTHON_ENV -c "
import sys
sys.path.insert(0, 'src')
sys.modules['pytest'] = type(sys)('pytest')

try:
    from src.database.connection import DatabaseManager
    db = DatabaseManager()
    
    with db.get_connection() as conn:
        cur = conn.cursor()
        
        # Check experiments
        cur.execute('SELECT COUNT(*) FROM experiments WHERE name LIKE %s', ('test_%',))
        exp_count = cur.fetchone()[0]
        print(f'     - Experiments: {exp_count}')
        
        # Check resampled datasets if table exists
        cur.execute('''
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'resampled_datasets'
            )
        ''')
        
        if cur.fetchone()[0]:
            cur.execute('SELECT COUNT(*) FROM resampled_datasets WHERE name LIKE %s', ('test-%',))
            dataset_count = cur.fetchone()[0]
            print(f'     - Resampled datasets: {dataset_count}')
        
except Exception as e:
    print(f'     Error checking database: {e}')
"
    
else
    echo "   Status: ❌ FAILED"
    echo "   Check the logs above for error details"
fi

echo ""
echo "🧹 Step 9: Post-test cleanup..."

read -p "   Clean up test data? (Y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo "   Cleaning up test outputs..."
    
    # Remove test output directories
    rm -rf outputs/unified_resampling/test_*
    
    # Clean up test database entries
    $PYTHON_ENV -c "
import sys
sys.path.insert(0, 'src')
sys.modules['pytest'] = type(sys)('pytest')

try:
    from src.database.connection import DatabaseManager
    db = DatabaseManager()
    
    with db.get_connection() as conn:
        cur = conn.cursor()
        
        # Clean up test experiments
        cur.execute('DELETE FROM experiments WHERE name LIKE %s', ('test_%',))
        exp_deleted = cur.rowcount
        
        # Clean up test resampled datasets
        cur.execute('''
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'resampled_datasets'
            )
        ''')
        
        if cur.fetchone()[0]:
            # Get test data table names to drop
            cur.execute('SELECT data_table_name FROM resampled_datasets WHERE name LIKE %s', ('test-%',))
            tables_to_drop = [row[0] for row in cur.fetchall() if row[0]]
            
            # Delete test dataset records
            cur.execute('DELETE FROM resampled_datasets WHERE name LIKE %s', ('test-%',))
            dataset_deleted = cur.rowcount
            
            # Drop test data tables
            for table in tables_to_drop:
                if table:
                    cur.execute(f'DROP TABLE IF EXISTS {table} CASCADE')
        else:
            dataset_deleted = 0
        
        conn.commit()
        print(f'   ✅ Cleaned up {exp_deleted} experiments and {dataset_deleted} datasets')
        
except Exception as e:
    print(f'   ⚠️ Cleanup warning: {e}')
"
    
    echo "   ✅ Cleanup completed"
else
    echo "   📁 Test data preserved for inspection"
fi

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo "🎉 Test pipeline completed successfully!"
    echo "   The unified resampling integration is working correctly."
else
    echo "💥 Test pipeline failed."
    echo "   Check the error messages above for debugging."
fi

echo ""
echo "📋 Test Summary:"
echo "   - Configuration: ✅ Working"
echo "   - Database: ✅ Connected"
echo "   - Validation: ✅ Passed"
if [ $TEST_RESULT -eq 0 ]; then
    echo "   - Pipeline: ✅ Success"
else
    echo "   - Pipeline: ❌ Failed"
fi

exit $TEST_RESULT