#! /bin/bash

model=meta-llama/Llama-3.1-8B

echo "=============With APC RUN===================="
PROFILE=0 \
VLLM_SKIP_3D_WARMUP=1 \
VLLM_DISABLE_COMPILE_FSDPA=1 \
VLLM_PROMPT_SEQ_BUCKET_MAX=512 \
PT_HPU_LAZY_MODE=1 \
python benchmark_prefix_caching.py \
        --model $model \
	--max-model-len 4096 \
	--max-num-seqs 8 \
        --num-prompts 1 \
        --repeat-count 24 \
        --input-length-range 250:256 \
        --output-len 1
