#! /bin/bash

MODEL="meta-llama/Llama-3.1-8B-Instruct"
lmcache_config_file=lmcache_config.yaml

LMCACHE_LOG_LEVEL=DEBUG \
VLLM_USE_V1=0 \
VLLM_SKIP_WARMUP=true \
PT_HPU_GPU_MIGRATION=1 \
VLLM_DELAYED_SAMPLING=0 \
LMCACHE_CONFIG_FILE=$lmcache_config_file \
LMCACHE_USE_EXPERIMENTAL=True \
vllm serve $MODEL \
    --port 8100 \
    --disable-async-output-proc \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnector","kv_role":"kv_both"}'
