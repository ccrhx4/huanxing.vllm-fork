#!/bin/bash

model=meta-llama/Meta-Llama-3-8B
tp=1

export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_MAX=1
export VLLM_PROMPT_BS_BUCKET_STEP=1

export VLLM_PROMPT_SEQ_BUCKET_MIN=384
export VLLM_PROMPT_SEQ_BUCKET_MAX=1792
export VLLM_PROMPT_SEQ_BUCKET_STEP=128

export VLLM_DECODE_BS_BUCKET_MIN=1
export VLLM_DECODE_BS_BUCKET_MAX=1
export VLLM_DECODE_BS_BUCKET_STEP=4

export VLLM_DECODE_BLOCK_BUCKET_MIN=8
export VLLM_DECODE_BLOCK_BUCKET_MAX=16
export VLLM_DECODE_BLOCK_BUCKET_STEP=2

echo "running with repetition penalty 1.06"
python3 benchmark_latency.py --model meta-llama/Meta-Llama-3-8B  --batch-size 1 --input-len 512 --use-v2-block-manager --output-len 1024 --max-model-len 4096 --num_iters 5 --repetition-penalty 1.06

echo "running without repetition penalty"
python3 benchmark_latency.py --model meta-llama/Meta-Llama-3-8B  --batch-size 1 --input-len 512 --use-v2-block-manager --output-len 1024 --max-model-len 4096 --num_iters 5
