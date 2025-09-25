#! /bin/bash

model=meta-llama/Llama-3.1-8B

echo "=============Without APC RUN===================="
VLLM_PROMPT_SEQ_BUCKET_MAX=512 \
python benchmark_prefix_caching.py \
	--model $model \
        --enable-prefix-caching \
	--max-model-len 4096 \
	--max-num-seqs 16 \
        --num-prompts 1 \
        --repeat-count 100 \
        --input-length-range 250:256
        --output-len 4

echo "=============With APC RUN===================="
VLLM_SKIP_3D_WARMUP=1 \
VLLM_DISABLE_COMPILE_FSDPA=1 \
VLLM_PROMPT_SEQ_BUCKET_MAX=512 \
python benchmark_prefix_caching.py \
        --model $model \
	--max-model-len 4096 \
	--max-num-seqs 16 \
        --enable-prefix-caching \
        --num-prompts 1 \
        --repeat-count 100 \
        --input-length-range 250:256
        --output-len 4
