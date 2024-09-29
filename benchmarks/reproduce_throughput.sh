#!/bin/bash

model=meta-llama/Meta-Llama-3-8B

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
export VLLM_DECODE_BLOCK_BUCKET_MAX=8
export VLLM_DECODE_BLOCK_BUCKET_STEP=1

export VLLM_PREFILL_USE_FUSEDSDPA=0

repetition_penalty=1.06

export HABANA_PROFILE=0
export GRAPH_VISUALIZATION=0
export VLLM_PROFILER_ENABLED=false

echo "running with repetition penalty 1.06"
python3 benchmark_throughput.py --model $model  --num-prompts 1 --input-len 512 --output-len 4 --max-model-len 4096 --repetition-penalty $repetition_penalty

echo "running without repetition penalty"
python3 benchmark_throughput.py --model $model  --num-prompts 1 --input-len 512 --output-len 4 --max-model-len 4096
