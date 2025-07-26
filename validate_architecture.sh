#!/bin/bash
# validate_architecture.sh - Architecture Validation Script

echo "=== Architecture Validation ==="

# Check for circular dependencies
echo -n "Circular dependencies: "
if command -v pycycle &> /dev/null; then
    python -m pycycle src/ --here . 2>/dev/null | wc -l
else
    echo "pycycle not available - skipping"
fi

# Check for old module references
echo -n "References to old raster module: "
OLD_RASTER_REFS=$(grep -r "src\.raster[^_]" src/ --include="*.py" 2>/dev/null | wc -l)
echo "$OLD_RASTER_REFS"

echo -n "References to old raster_data module: "
OLD_RASTER_DATA_REFS=$(grep -r "src\.raster_data" src/ --include="*.py" 2>/dev/null | wc -l)
echo "$OLD_RASTER_DATA_REFS"

# Check foundation layer purity
echo -n "Foundation layer violations: "
FOUNDATION_VIOLATIONS=$(grep -r "from src\." src/foundations/ --include="*.py" 2>/dev/null | grep -v "from src.foundations" | wc -l)
echo "$FOUNDATION_VIOLATIONS"

# Check for backup files
echo -n "Backup files remaining: "
BACKUP_FILES=$(find src/ -name "*.bak" -o -name "*.backup" -o -name "*~" | wc -l)
echo "$BACKUP_FILES"

# Check config import pattern consistency
echo -n "Inconsistent config imports: "
CONFIG_CLASS_IMPORTS=$(grep -r "from src.config import Config" src/ --include="*.py" 2>/dev/null | wc -l)
echo "$CONFIG_CLASS_IMPORTS (should be 0)"

# Check for test mode database usage in production
echo -n "Production test_mode usage: "
TEST_MODE_USAGE=$(grep -r "test_mode=True" src/ --include="*.py" --exclude-dir=tests 2>/dev/null | wc -l)
echo "$TEST_MODE_USAGE (should be 0)"

# Summary
echo -e "\n=== Summary ==="
TOTAL_ISSUES=$((OLD_RASTER_REFS + OLD_RASTER_DATA_REFS + FOUNDATION_VIOLATIONS + BACKUP_FILES + CONFIG_CLASS_IMPORTS + TEST_MODE_USAGE))

if [ $TOTAL_ISSUES -eq 0 ]; then
    echo "✅ Architecture validation PASSED - No issues found"
    exit 0
else
    echo "❌ Architecture validation FAILED - $TOTAL_ISSUES issues found"
    echo "Please review and fix the issues above"
    exit 1
fi