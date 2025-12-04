#!/bin/bash

# --- Configuration ---
MODEL="meta-llama/Llama-3.1-8B-Instruct"
lmcache_config_file="lmcache_config.yaml"

if [ "$DEBUG" = "1" ]; then
    # If DEBUG is set, enable DEBUG logging for LMCache.
    export LMCACHE_LOG_LEVEL=DEBUG
    echo "DEBUG environment variable detected. Setting LMCache log level to DEBUG."
else
    export LMCACHE_LOG_LEVEL=INFO
fi

# --- VLLM Service Execution ---
export PT_HPU_RECIPE_CACHE_CONFIG='/graph_cache/',True,16384

VLLM_USE_V1=0 \
VLLM_SKIP_WARMUP=true \
PT_HPU_GPU_MIGRATION=1 \
VLLM_DELAYED_SAMPLING=0 \
LMCACHE_CONFIG_FILE=$lmcache_config_file \
LMCACHE_USE_EXPERIMENTAL=True \
PYTHONHASHSEED=0 \
vllm serve $MODEL \
    --port 8100 \
    --disable-async-output-proc \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --load-format dummy \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnector","kv_role":"kv_both"}'
