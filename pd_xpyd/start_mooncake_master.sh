#!/bin/bash

BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/pd_env.sh

pkill -f mooncake_master
sleep 5s

# Define commands as arrays
mooncake_args=(
    -port 50001
    -max_threads 64
    -metrics_port 9004
    -eviction_high_watermark_ratio 0.8
    -eviction_ratio 0.2
    --enable_http_metadata_server=true
    --http_metadata_server_host=0.0.0.0
    --http_metadata_server_port=2379
)

# Check if XPYD_LOG is set
if [ -n "$XPYD_LOG" ]; then
    timestamp=$(date +"%Y%m%d_%H%M%S")

    # Run mooncake_master with logging
    MOON_LOG="$XPYD_LOG/mooncake_master_${timestamp}.log"
    echo "Starting mooncake_master, logging to $MOON_LOG..."
    mooncake_master "${mooncake_args[@]}" > "$MOON_LOG" 2>&1 &
else
    # Run without logging
    echo "XPYD_LOG not set, running without logging..."
    "${MOONCAKE_CMD[@]}" > /dev/null 2>&1 &
fi

