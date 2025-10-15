#! /bin/bash

model=meta-llama/Llama-3.1-8B

echo "=============Without APC RUN===================="
VLLM_PROMPT_USE_FUSEDSDPA=1 \
VLLM_DISABLE_COMPILE_PREFILL=1 \
VLLM_SKIP_3D_WARMUP=0 \
VLLM_DISABLE_COMPILE_FSDPA=1 \
VLLM_PROMPT_SEQ_BUCKET_MAX=512 \
VLLM_DELAYED_SAMPLING=false \
python benchmark_prefix_caching.py \
	--model $model \
	--max-model-len 16384 \
	--max-num-seqs 8 \
        --num-prompts 100 \
        --repeat-count 5 \
        --input-length-range 15360:15360 \
	--disable_async_output_proc \
        --output-len 1

echo "=============Without APC RUN===================="
VLLM_PROMPT_USE_FUSEDSDPA=1 \
VLLM_DISABLE_COMPILE_PREFILL=0 \
VLLM_SKIP_3D_WARMUP=0 \
VLLM_DISABLE_COMPILE_FSDPA=0 \
VLLM_PROMPT_SEQ_BUCKET_MAX=512 \
VLLM_DELAYED_SAMPLING=false \
python benchmark_prefix_caching.py \
        --model $model \
        --max-model-len  16384\
        --max-num-seqs 8 \
        --num-prompts 100 \
        --repeat-count 5 \
        --input-length-range 15360:15360 \
	--disable_async_output_proc \
        --output-len 1


echo "=============With APC RUN===================="
#VLLM_SKIP_3D_WARMUP=1 \
#VLLM_DISABLE_COMPILE_PREFILL=1 \
#VLLM_DISABLE_COMPILE_FSDPA=1 \
#VLLM_PROMPT_SEQ_BUCKET_MAX=512 \
#python benchmark_prefix_caching.py \
#        --model $model \
#	--max-model-len 4096 \
#	--max-num-seqs 8 \
#        --enable-prefix-caching \
#        --num-prompts 1 \
#        --repeat-count 2000 \
#        --input-length-range 1020:1028 \
#        --output-len 1

echo "=============With APC RUN===================="
#VLLM_SKIP_3D_WARMUP=1 \
#VLLM_PROMPT_USE_FUSEDSDPA=1 \
#VLLM_DISABLE_COMPILE_PREFILL=1 \
#VLLM_DISABLE_COMPILE_FSDPA=1 \
#VLLM_PROMPT_SEQ_BUCKET_MAX=512 \
#python benchmark_prefix_caching.py \
#        --model $model \
#	--max-model-len 8196 \
#	--max-num-seqs 8 \
#        --enable-prefix-caching \
#        --num-prompts 100 \
#        --repeat-count 5 \
#        --input-length-range 512:4096 \
#        --output-len 1
