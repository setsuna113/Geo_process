#!/bin/bash
# Monitor CSV export progress

echo "=== CSV Export Monitor ==="
echo "Started at: $(date)"

while true; do
    # Check process
    if ps aux | grep -q "[e]fficient_csv_export"; then
        echo -e "\n[$(date +%H:%M:%S)] Export process running"
        ps aux | grep "[e]fficient_csv_export" | awk '{print "  CPU:", $3"%, MEM:", $4"%, Time:", $10}'
        
        # Check log
        if [ -s csv_export_full.log ]; then
            echo "  Log output:"
            tail -5 csv_export_full.log | sed 's/^/    /'
        else
            echo "  Loading data..."
        fi
        
        # Check output files
        latest_csv=$(find outputs -name "*.csv" -mmin -10 2>/dev/null | tail -1)
        if [ ! -z "$latest_csv" ]; then
            size=$(ls -lh "$latest_csv" | awk '{print $5}')
            echo "  Output file: $latest_csv ($size)"
        fi
    else
        echo -e "\n[$(date +%H:%M:%S)] Export completed or stopped"
        if [ -s csv_export_full.log ]; then
            echo "Final log:"
            tail -20 csv_export_full.log
        fi
        break
    fi
    
    sleep 10
done