#!/bin/bash
 
# Configuration
PORT=8115
export no_proxy="0.0.0.0,127.0.0.1,localhost"

# Automatically get model name from endpoint
echo "Fetching model name from http://127.0.0.1:$PORT/v1/models..."
if command -v jq >/dev/null 2>&1; then
    MODEL=$(curl -s --connect-timeout 10 "http://127.0.0.1:$PORT/v1/models" | jq -r '.data[0].id' 2>/dev/null)
else
    # Fallback using grep/sed if jq is not installed
    MODEL=$(curl -s --connect-timeout 10 "http://127.0.0.1:$PORT/v1/models" | grep -o '"id":[[:space:]]*"[^"]*"' | head -n 1 | cut -d'"' -f4)
fi

if [ -z "$MODEL" ] || [ "$MODEL" = "null" ]; then
    echo "ERROR: Could not automatically retrieve model name from http://127.0.0.1:$PORT/v1/models"
    echo "Please make sure the vLLM server is running on port $PORT."
    exit 1
fi

echo "Successfully retrieved model name: $MODEL"

INPUT_LEN="${INPUT_LEN:-3500}"
OUTPUT_LEN="${OUTPUT_LEN:-1500}"
NUM_PROMPTS=256 # Total prompts to send per test; increase if results are too fast
LOG_DIR="${LOG_DIR:-./bench_results_moe_interleaved_fusion}"
CONCURRENCY_LIST="${CONCURRENCY_LIST:-1 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30}"
 
mkdir -p $LOG_DIR
 
echo "Starting vLLM Benchmark: Concurrency levels: $CONCURRENCY_LIST"
echo "Log dir: $LOG_DIR"
echo "---------------------------------------------------------"
 
# Loop through concurrency levels (override via CONCURRENCY_LIST env var)
for CONCURRENCY in $CONCURRENCY_LIST;
do
    echo "Running benchmark with Concurrency: $CONCURRENCY..."
    NUM_PROMPTS=$((CONCURRENCY * 10))
    echo "$NUM_PROMPTS"
    # vllm bench serve is the modern recommended command
    vllm bench serve \
        --model $MODEL \
        --port $PORT \
        --host 0.0.0.0 \
        --dataset-name random \
        --random-input-len $INPUT_LEN \
        --random-output-len $OUTPUT_LEN \
        --random-range-ratio '{"input": "0", "output": "0.05"}' \
        --num-prompts $NUM_PROMPTS \
        --max-concurrency $CONCURRENCY \
        --save-result \
        --save-detailed \
        --result-dir $LOG_DIR \
        --backend openai \
        --temperature 0 \
        --top-p 1.0 \
	--endpoint /v1/completions
 
    echo "Finished Concurrency $CONCURRENCY. Results saved to $LOG_DIR"
    echo "---------------------------------------------------------"
    # Optional: Short sleep to let the KV cache clear or server settle
    sleep 5
done
