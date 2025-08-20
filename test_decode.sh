#!/bin/bash

# Define model paths and server details
# MODEL_PATH="/mnt/disk2/hf_models/DeepSeek-R1-BF16-w8afp8-static-no-ste-G2/"
# An alternative model path, currently commented out
MODEL_PATH="/mnt/disk2/hf_models/DeepSeek-R1-G2/"

SERVER_IP="localhost"
SERVER_PORT=8868

PREFILL_ENDPOINT='/v1/prefill/completions'
DECODE_ENDPOINT='/v1/decode/completions'
PROMPTS=256

# Get the output length from the first command-line argument
OUTPUT_LEN=$1

run_benchmark() {
    local output_length=$1
    local endpoint=$2

    python3 benchmarks/benchmark_serving.py \
        --backend vllm \
        --model "$MODEL_PATH" \
        --dataset-name sonnet \
        --request-rate inf \
        --host "$SERVER_IP" \
        --port "$SERVER_PORT" \
        --endpoint "$endpoint" \
        --sonnet-input-len 2000 \
        --sonnet-output-len "$output_length" \
        --sonnet-prefix-len 100 \
        --trust-remote-code \
        --max-concurrency "$PROMPTS" \
        --num-prompts "$PROMPTS" \
        --ignore-eos \
        --burstiness 1000 \
        --dataset-path benchmarks/sonnet.txt \
        --save-result
}

# Call the function with the provided output length
echo "Start Prefill Run."
run_benchmark 1 $PREFILL_ENDPOINT
echo "Start Decode Run, output_len: $OUTPUT_LEN"
run_benchmark "$OUTPUT_LEN" $DECODE_ENDPOINT
