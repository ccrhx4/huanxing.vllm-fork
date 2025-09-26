#! /bin/bash

model=meta-llama/Llama-3.1-8B

echo "=============Without APC RUN===================="
VLLM_PROMPT_SEQ_BUCKET_MAX=512 \
python benchmark_prefix_caching.py \
	--model $model \
	--max-model-len 4096 \
	--max-num-seqs 8 \
        --num-prompts 1 \
        --repeat-count 2000 \
        --input-length-range 256:260 \
        --output-len 1

echo "=============With APC RUN===================="
VLLM_SKIP_3D_WARMUP=1 \
VLLM_DISABLE_COMPILE_FSDPA=1 \
VLLM_PROMPT_SEQ_BUCKET_MAX=512 \
python benchmark_prefix_caching.py \
        --model $model \
	--max-model-len 4096 \
	--max-num-seqs 8 \
        --enable-prefix-caching \
        --num-prompts 1 \
        --repeat-count 2000 \
        --input-length-range 256:260 \
        --output-len 1
