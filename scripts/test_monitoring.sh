#!/bin/bash
# Test runner for monitoring and logging system

echo "🧪 Testing Unified Monitoring and Logging System"
echo "=============================================="

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test categories
declare -A test_suites=(
    ["Structured Logging"]="tests/infrastructure/logging/test_structured_logger.py"
    ["Database Handler"]="tests/infrastructure/logging/test_database_handler.py"
    ["Logging Context"]="tests/infrastructure/logging/test_context.py"
    ["Unified Monitor"]="tests/infrastructure/monitoring/test_unified_monitor.py"
    ["Integration Tests"]="tests/infrastructure/test_monitoring_integration.py"
)

# Check if pytest is installed
if ! python -m pytest --version &> /dev/null; then
    echo -e "${RED}❌ Error: pytest not installed${NC}"
    echo "Install with: pip install pytest pytest-cov pytest-mock"
    exit 1
fi

# Parse arguments
VERBOSE=""
COVERAGE=""
SPECIFIC_TEST=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -c|--coverage)
            COVERAGE="--cov=src/infrastructure --cov-report=html --cov-report=term"
            shift
            ;;
        -t|--test)
            SPECIFIC_TEST="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose    Verbose test output"
            echo "  -c, --coverage   Generate coverage report"
            echo "  -t, --test NAME  Run specific test suite"
            echo "  -h, --help       Show this help"
            echo ""
            echo "Available test suites:"
            for suite in "${!test_suites[@]}"; do
                echo "  - $suite"
            done
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to run a test suite
run_test_suite() {
    local name=$1
    local path=$2
    
    echo ""
    echo -e "${YELLOW}Running: $name${NC}"
    echo "----------------------------------------"
    
    if python -m pytest $path $VERBOSE $COVERAGE -x; then
        echo -e "${GREEN}✅ $name: PASSED${NC}"
        return 0
    else
        echo -e "${RED}❌ $name: FAILED${NC}"
        return 1
    fi
}

# Run tests
FAILED=0

if [ -n "$SPECIFIC_TEST" ]; then
    # Run specific test suite
    if [ -n "${test_suites[$SPECIFIC_TEST]}" ]; then
        run_test_suite "$SPECIFIC_TEST" "${test_suites[$SPECIFIC_TEST]}"
        FAILED=$?
    else
        echo -e "${RED}Unknown test suite: $SPECIFIC_TEST${NC}"
        echo "Available suites:"
        for suite in "${!test_suites[@]}"; do
            echo "  - $suite"
        done
        exit 1
    fi
else
    # Run all test suites
    for suite in "${!test_suites[@]}"; do
        run_test_suite "$suite" "${test_suites[$suite]}"
        if [ $? -ne 0 ]; then
            FAILED=1
        fi
    done
fi

# Summary
echo ""
echo "=============================================="
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    
    if [ -n "$COVERAGE" ]; then
        echo ""
        echo "Coverage report generated in: htmlcov/index.html"
        echo "Open with: open htmlcov/index.html"
    fi
else
    echo -e "${RED}❌ Some tests failed${NC}"
fi

echo ""
echo "💡 Tips:"
echo "  - Run with -v for verbose output"
echo "  - Run with -c for coverage report"
echo "  - Run with -t \"Test Name\" for specific suite"

exit $FAILED