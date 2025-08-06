#!/bin/bash

echo "🔄 Monitoring GEE uploads for production_6bands_aggregation..."

# Wait for uploads to complete
while true; do
    # Check both upload tasks
    task1=$(/home/jason/anaconda3/envs/geo/bin/earthengine --service_account_file .gee/service-account-key.json --project triple-carrier-441622-t1 task info B24GIYVF4UYPA6IR3F74CLBB 2>/dev/null | grep State | awk '{print $2}')
    task2=$(/home/jason/anaconda3/envs/geo/bin/earthengine --service_account_file .gee/service-account-key.json --project triple-carrier-441622-t1 task info OJCF7JK6O53QF642ON7S65RT 2>/dev/null | grep State | awk '{print $2}')
    
    echo "[$(date +%H:%M:%S)] Upload status: AM_fungi_cv=$task1, ECM_fungi_cv=$task2"
    
    if [[ "$task1" == "COMPLETED" && "$task2" == "COMPLETED" ]]; then
        echo "✅ Both uploads completed successfully!"
        break
    elif [[ "$task1" == "FAILED" || "$task2" == "FAILED" ]]; then
        echo "❌ Upload failed! Status: AM=$task1, ECM=$task2"
        exit 1
    fi
    
    sleep 30
done

echo ""
echo "🚀 Continuing workflow with 6 biodiversity bands (11 total bands)..."
echo "   - animal_richness (max aggregation)"
echo "   - plant_richness (max aggregation)"
echo "   - am_fungi_richness (bilinear)"
echo "   - am_fungi_coeff_var (RMS aggregation)"
echo "   - ecm_fungi_richness (bilinear)"
echo "   - ecm_fungi_coeff_var (RMS aggregation)"
echo "   + 4 climate bands + 1 elevation"
echo ""

# Continue workflow
/home/jason/anaconda3/envs/geo/bin/python scripts/run_gee_workflow.py \
    --config config/gee_config.yml \
    --experiment-name "production_6bands_aggregation" \
    --resume

echo "✅ Workflow complete!"