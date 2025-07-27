#!/bin/bash
# Monitor daemon health and registry consistency

echo "=== Daemon Process Monitor ==="
echo "Time: $(date)"
echo

# Check registry
echo "Registered Processes:"
if ls ~/.biodiversity/pid/registry/*.json 2>/dev/null | grep -v '\.registry\.lock' >/dev/null; then
    for registry_file in ~/.biodiversity/pid/registry/*.json; do
        if [ -f "$registry_file" ]; then
            name=$(basename "$registry_file" .json)
            pid=$(jq -r '.pid' "$registry_file" 2>/dev/null || echo "unknown")
            status=$(jq -r '.status' "$registry_file" 2>/dev/null || echo "unknown")
            echo "  $name: PID=$pid, Status=$status"
        fi
    done
else
    echo "  No processes registered"
fi

# Check actual processes
echo -e "\nRunning Pipeline Processes:"
if ps aux | grep -E "biodiversity|geo|python.*process_manager|python.*orchestrator" | grep -v grep | grep -v monitor_daemons >/dev/null; then
    ps aux | grep -E "biodiversity|geo|python.*process_manager|python.*orchestrator" | grep -v grep | grep -v monitor_daemons | while read line; do
        echo "  $line"
    done
else
    echo "  No pipeline processes running"
fi

# Check PID files vs registry
echo -e "\nPID File Consistency:"
consistency_ok=true
for pid_file in ~/.biodiversity/pid/*.pid; do
    if [ -f "$pid_file" ]; then
        name=$(basename "$pid_file" .pid)
        pid=$(cat "$pid_file")
        if [ -f ~/.biodiversity/pid/registry/${name}.json ]; then
            registry_pid=$(jq -r '.pid' ~/.biodiversity/pid/registry/${name}.json 2>/dev/null)
            if [ "$pid" = "$registry_pid" ]; then
                echo "  ✅ $name: PID file and registry match ($pid)"
            else
                echo "  ❌ $name: PID mismatch - file: $pid, registry: $registry_pid"
                consistency_ok=false
            fi
        else
            echo "  ⚠️  $name: PID file exists but not in registry"
            consistency_ok=false
        fi
    fi
done

# Check for registry entries without PID files
for registry_file in ~/.biodiversity/pid/registry/*.json; do
    if [ -f "$registry_file" ] && [[ ! "$registry_file" =~ \.registry\.lock ]]; then
        name=$(basename "$registry_file" .json)
        if [ ! -f ~/.biodiversity/pid/${name}.pid ]; then
            echo "  ⚠️  $name: Registry entry exists but no PID file"
            consistency_ok=false
        fi
    fi
done

if [ "$consistency_ok" = true ] && [ ! -f ~/.biodiversity/pid/*.pid ] 2>/dev/null; then
    echo "  ✅ All consistent (no processes running)"
elif [ "$consistency_ok" = true ]; then
    echo "  ✅ All PID files and registry entries are consistent"
fi

# Resource usage summary
echo -e "\nResource Usage:"
echo "  CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)% used"
echo "  Memory: $(free | grep Mem | awk '{printf "%.1f%% used", $3/$2 * 100.0}')"
echo "  Disk (home): $(df -h ~ | tail -1 | awk '{print $5}') used"

# Log file sizes
echo -e "\nLog Directory Status:"
if ls ~/.biodiversity/logs/*.log >/dev/null 2>&1; then
    total_size=$(du -sh ~/.biodiversity/logs/ 2>/dev/null | cut -f1)
    log_count=$(ls ~/.biodiversity/logs/*.log 2>/dev/null | wc -l)
    echo "  Total logs: $log_count files, $total_size"
    
    # Show largest log files
    echo "  Largest logs:"
    du -h ~/.biodiversity/logs/*.log 2>/dev/null | sort -hr | head -3 | while read size file; do
        echo "    $(basename "$file"): $size"
    done
else
    echo "  No log files found"
fi

echo -e "\n=== Monitor Complete ==="