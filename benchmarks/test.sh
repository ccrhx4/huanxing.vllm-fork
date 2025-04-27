#!/bin/bash

model=meta-llama/Meta-Llama-3-8B
tp=1

export VLLM_PROMPT_BS_BUCKET_MIN=8
export VLLM_PROMPT_BS_BUCKET_MAX=8
export VLLM_PROMPT_BS_BUCKET_STEP=8

export VLLM_PROMPT_SEQ_BUCKET_MIN=384
export VLLM_PROMPT_SEQ_BUCKET_MAX=1792
export VLLM_PROMPT_SEQ_BUCKET_STEP=128

export VLLM_DECODE_BS_BUCKET_MIN=8
export VLLM_DECODE_BS_BUCKET_MAX=8
export VLLM_DECODE_BS_BUCKET_STEP=4

export VLLM_DECODE_BLOCK_BUCKET_MIN=24
export VLLM_DECODE_BLOCK_BUCKET_MAX=48
export VLLM_DECODE_BLOCK_BUCKET_STEP=2

echo "running with repetition penalty 1.06"
python3 benchmark_latency.py --model meta-llama/Meta-Llama-3-8B  --batch-size 8 --input-len 512 --use-v2-block-manager --output-len 18 --max-model-len 4096 --num_iters 5 --repetition-penalty 1.06

echo "running with no repetition penalty"
python3 benchmark_latency.py --model meta-llama/Meta-Llama-3-8B  --batch-size 8 --input-len 512 --use-v2-block-manager --output-len 18 --max-model-len 4096 --num_iters 5

