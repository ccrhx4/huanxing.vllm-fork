#!/bin/bash

export VLLM_PROMPT_BS_BUCKET_MIN=1
export VLLM_PROMPT_BS_BUCKET_MAX=1
export VLLM_PROMPT_BS_BUCKET_STEP=1

export VLLM_PROMPT_SEQ_BUCKET_MIN=512
export VLLM_PROMPT_SEQ_BUCKET_MAX=512
export VLLM_PROMPT_SEQ_BUCKET_STEP=1

export VLLM_DECODE_BS_BUCKET_MIN=1
export VLLM_DECODE_BS_BUCKET_MAX=1
export VLLM_DECODE_BS_BUCKET_STEP=1

export VLLM_DECODE_BLOCK_BUCKET_MIN=4
export VLLM_DECODE_BLOCK_BUCKET_MAX=12
export VLLM_DECODE_BLOCK_BUCKET_STEP=1

echo "running good warmup"
python3 benchmark_latency.py --model meta-llama/Meta-Llama-3-8B  --batch-size 1 --input-len 512 --output-len 1024 --max-model-len 4096 --num_iters 10 --num_iters_warmup 5

export VLLM_DECODE_BLOCK_BUCKET_MIN=512
export VLLM_DECODE_BLOCK_BUCKET_MAX=1536
export VLLM_DECODE_BLOCK_BUCKET_STEP=128

echo "running bad warmup"
python3 benchmark_latency.py --model meta-llama/Meta-Llama-3-8B  --batch-size 1 --input-len 512 --output-len 1024 --max-model-len 4096 --num_iters 10 --num_iters_warmup 5
