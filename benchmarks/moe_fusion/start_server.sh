#!/usr/bin/env bash
set -euo pipefail

model="${1:-/hf_models}"
port="${PORT:-8115}"
host="${HOST:-0.0.0.0}"
tp_size="${TP_SIZE:-1}"
log_file="${VLLM_LOG:-./vllm_serve.log}"
pid_file="${VLLM_PID_FILE:-./vllm_serve.pid}"
reasoning_args=()
attention_args=()
quantization_args=()
expert_parallel_args=()
eager_args=()
fewer_layers_args=()
unitrace_args=()
profiler_args=()

# Set USE_EAGER=0 to disable --enforce-eager (default: enabled).
if [[ "${USE_EAGER:-0}" == "1" ]]; then
  eager_args=(--enforce-eager)
fi

# Set USE_TRITON_ATTN=1 to enable --attention_backend triton_attn.
if [[ "${USE_TRITON_ATTN:-0}" == "1" ]]; then
  attention_args=(--attention_backend triton_attn)
fi

# Set USE_FP8_QUANT=1 to enable --quantization fp8.
if [[ "${USE_FP8_QUANT:-0}" == "1" ]]; then
  quantization_args=(--quantization fp8)
fi

# Set USE_EXPERT_PARALLEL=1 to enable --enable-expert-parallel.
if [[ "${USE_EXPERT_PARALLEL:-0}" == "1" ]]; then
  expert_parallel_args=(--enable-expert-parallel)
fi

# Set NUM_LAYERS to load fewer hidden layers with dummy weights.
if [[ -n "${NUM_LAYERS:-}" ]]; then
  fewer_layers_args=(--hf-overrides '{"num_hidden_layers": '"$NUM_LAYERS"'}' --load-format dummy)
fi

# Set unitrace_on=1 to enable unitrace profiling.
if [[ "${unitrace_on:-0}" == "1" || "${UNITRACE_ON:-0}" == "1" ]]; then
  export NEOReadDebugKeys=1
  export EnableImplicitConvertionToCounterBasedEvents=0
  unitrace_args=(
    unitrace
    --chrome-itt-logging
    --chrome-sycl-logging
    --chrome-call-logging
    --chrome-kernel-logging
    --start-paused
    --result-dir "./profiler_8192"
  )
  profiler_args=(--profiler-config.profiler xpu --shutdown-timeout 120)
fi

# Set TORCH_PROFILER_ON=1 to enable the torch profiler.
if [[ "${TORCH_PROFILER_ON:-0}" == "1" ]]; then
  profile_dir_1="${TORCH_PROFILER_DIR:-./profiler_8192_xpu_graph}"
  profiler_args=(-cc.cudagraph_mode="FULL_DECODE_ONLY" --profiler-config.profiler=torch --profiler-config.torch_profiler_dir="$profile_dir_1" --profiler-config.torch_profiler_record_shapes=True)
fi

# Enable Qwen3 reasoning parser only for Qwen3-family models.
if [[ "${model,,}" == *"qwen3"* ]]; then
  reasoning_args=(--reasoning-parser qwen3)
fi

# Set VLLM_XPU_ENABLE_XPU_GRAPH=0 to disable XPU graph execution (default: enabled).
export VLLM_XPU_ENABLE_XPU_GRAPH="${VLLM_XPU_ENABLE_XPU_GRAPH:-1}"
export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_XPU_ENABLE_XPU_GRAPH:-1}"
export VLLM_USE_V2_MODEL_RUNNER=0

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm command not found in PATH." >&2
  exit 1
fi

if [[ -f "$pid_file" ]]; then
  existing_pid="$(cat "$pid_file")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" >/dev/null 2>&1; then
    echo "vLLM serve is already running with PID $existing_pid." >&2
    echo "Stop it first or remove $pid_file." >&2
    exit 1
  fi
fi

nohup env VLLM_WORKER_MULTIPROC_METHOD=spawn "${unitrace_args[@]}" vllm serve "$model" \
  --host "$host" \
  --port "$port" \
  --block-size 64 \
  --max-model-len 32768 \
  --tensor-parallel-size "$tp_size" \
  "${eager_args[@]}" \
  --gpu-memory-util 0.8 \
  --no-enable-prefix-caching \
  --language-model-only \
  --enable-tokenizer-info-endpoint \
  "${quantization_args[@]}" \
  "${expert_parallel_args[@]}" \
  "${attention_args[@]}" \
  "${reasoning_args[@]}" \
  "${fewer_layers_args[@]}" \
  "${profiler_args[@]}" \
  >"$log_file" 2>&1 &
new_pid=$!
echo "$new_pid" > "$pid_file"

echo "Started vLLM serve on host."
echo "PID: $new_pid"
echo "Model: $model"
echo "Endpoint: http://$host:$port"
echo "Log: $log_file"
echo "PID file: $pid_file"
