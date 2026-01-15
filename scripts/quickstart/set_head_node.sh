#! /bin/bash

# set -x
# parameters to be changed
# set IP address of header node
export VLLM_HOST_IP=127.0.0.1
# set NIC interface name of worker IP address
export GLOO_SOCKET_IFNAME=enp154s0d8

# warmup cache folder
export PT_HPU_RECIPE_CACHE_CONFIG=/data/cache/cache_32k,false,32768

# vllm parameters
export max_num_batched_tokens=32768
export max_num_seqs=512
input_min=1
input_max=$max_num_batched_tokens
output_max=$max_num_batched_tokens

# Change to fp8_inc if want to use fp8 kv cache
export KV_CACHE_DTYPE=auto

BASH_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$BASH_DIR"/utils.sh

# INC FP8 quantization
export INC_MEASUREMENT_DUMP_PATH_PREFIX=$(realpath "$BASH_DIR/../..")
export QUANT_CONFIG=$(realpath "$BASH_DIR/../quant_configs/inc_quant_fp8kv_pts_scalar_fp8_mla.json")
if [ -n "$QUANT_CONFIG" ]; then
    export VLLM_REQUANT_FP8_INC=1
    export VLLM_ENABLE_RUNTIME_DEQUANT=1
    export VLLM_HPU_MARK_SCALES_AS_CONST=false
    export VLLM_MOE_N_SLICE=1
    export INC_FORCE_NAIVE_SCALING=1

    # Set RUNTIME_SCALE_PATCHING when scale_format equals "scalar" in quant config
    export RUNTIME_SCALE_PATCHING=1

    # Enable QKV slicing for long prompt
    export INC_APPLY_OOT_PATCH="true"
    export PT_HPU_SDPA_QKV_SLICE_MODE_FWD=1
    export VLLM_HPU_FSDPA_SLICE_SEQ_LEN_THLD=16384
    export PT_HPU_SDPA_BR_FACTOR=4096       # slice size on the query
    export PT_HPU_SDPA_BC_FACTOR=4096       # siice size on the kv
    export VLLM_HPU_FSDPA_SLICE_CHUNK_SIZE=4096 # qkv slice size in fp8 FSDPA
    export VLLM_HPU_FSDPA_SLICE_IMPL="slice_qkv"    # select the fp8 fsdpa impl
    export VLLM_HPU_FSDPA_SLICE_CAUSAL="true"

    # Enable MoE slice to reduce memory footprint
    export VLLM_SUPPORT_MOE_SLICE=True
    export VLLM_MOE_SLICE_LENGTH=8192
    clean_inc_scale
else
    export VLLM_MOE_N_SLICE=8
fi

export HCCL_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1
export VLLM_DELAYED_SAMPLING="true"
export VLLM_MLA_PERFORM_MATRIX_ABSORPTION=0
export VLLM_MLA_DISABLE_REQUANTIZATION=0

ray stop --force


# DO NOT change unless you fully undersand its purpose
export HABANA_VISIBLE_DEVICES="ALL"
export PT_HPU_ENABLE_LAZY_COLLECTIVES="true"
export VLLM_RAY_DISABLE_LOG_TO_DRIVER="1"
export RAY_IGNORE_UNHANDLED_ERRORS="1"
export PT_HPU_WEIGHT_SHARING=0
export HABANA_VISIBLE_MODULES="0,1,2,3,4,5,6,7"
export PT_HPUGRAPH_DISABLE_TENSOR_CACHE=1
export PT_HPU_LAZY_MODE=1

export VLLM_EP_SIZE=16

export block_size=128
# DO NOT change ends...

# memory footprint tunning params
export VLLM_GPU_MEMORY_UTILIZATION=0.9
export VLLM_GRAPH_RESERVED_MEM=0.2
export VLLM_GRAPH_PROMPT_RATIO=0

#export VLLM_SKIP_WARMUP=true



unset VLLM_PROMPT_BS_BUCKET_MIN VLLM_PROMPT_BS_BUCKET_STEP VLLM_PROMPT_BS_BUCKET_MAX
unset VLLM_PROMPT_SEQ_BUCKET_MIN VLLM_PROMPT_SEQ_BUCKET_STEP VLLM_PROMPT_SEQ_BUCKET_MAX
unset VLLM_DECODE_BS_BUCKET_MIN VLLM_DECODE_BS_BUCKET_STEP VLLM_DECODE_BS_BUCKET_MAX
unset VLLM_DECODE_BLOCK_BUCKET_MIN VLLM_DECODE_BLOCK_BUCKET_STEP VLLM_DECODE_BLOCK_BUCKET_MAX

set_bucketing

echo " environments are reseted "

env | grep VLLM
