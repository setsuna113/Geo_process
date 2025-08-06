#!/bin/bash

echo "🔄 Monitoring GEE uploads and continuing workflow..."

# Wait for uploads to complete
while true; do
    # Check both upload tasks
    task1=$(/home/jason/anaconda3/envs/geo/bin/earthengine --service_account_file .gee/service-account-key.json --project triple-carrier-441622-t1 task info HDFQHU4UJT453LXHN72BGXRO 2>/dev/null | grep State | awk '{print $2}')
    task2=$(/home/jason/anaconda3/envs/geo/bin/earthengine --service_account_file .gee/service-account-key.json --project triple-carrier-441622-t1 task info KQR7QGVXG3GLEZ4XWPOXC33D 2>/dev/null | grep State | awk '{print $2}')
    
    echo "Upload status: AM_fungi_cv=$task1, ECM_fungi_cv=$task2"
    
    if [[ "$task1" != "RUNNING" && "$task2" != "RUNNING" ]]; then
        echo "✅ Uploads completed!"
        break
    fi
    
    sleep 30
done

# Continue workflow
echo "🚀 Continuing workflow with 6 biodiversity bands..."
/home/jason/anaconda3/envs/geo/bin/python scripts/run_gee_workflow.py \
    --config config/gee_config.yml \
    --experiment-name "production_6bands_aggregation" \
    --resume

echo "✅ Workflow complete!"