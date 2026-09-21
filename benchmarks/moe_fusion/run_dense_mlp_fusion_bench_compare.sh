#!/usr/bin/env bash
# Orchestrates a baseline-vs-fusion vLLM serving benchmark for the
# interleaved SwiGLU *dense-FFN* fusion kernel (the non-grouped analog of
# run_moe_fusion_bench_compare.sh -- see /work/fusemlp/design.md). Default
# model is Qwen3.6-27B (dense text FFN, hidden=5120, intermediate=17408),
# TP=4; override MODEL / RESULT_ROOT / TP_SIZE to target a different dense
# model, e.g.:
#   MODEL=/hf_models/some-other-dense-model \
#   RESULT_ROOT=/work/bench_results_dense_mlp_interleaved_fusion_other \
#   benchmarks/moe_fusion/run_dense_mlp_fusion_bench_compare.sh both
#
# For each configuration (baseline: VLLM_XPU_FUSED_DENSE_MLP_INTERLEAVED unset,
# fusion: VLLM_XPU_FUSED_DENSE_MLP_INTERLEAVED=1) this script:
#   1. Starts a vLLM OpenAI-compatible server (via start_server.sh).
#   2. Waits for it to become ready.
#   3. Runs bench.sh (vllm bench serve sweep across CONCURRENCY_LIST)
#      against it, saving results under a per-config LOG_DIR.
#   4. Stops the server.
#
# Usage: benchmarks/moe_fusion/run_dense_mlp_fusion_bench_compare.sh [baseline|fusion|both]
# Default: both.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-/hf_models/Qwen3.6-27B}"
TP_SIZE="${TP_SIZE:-4}"
PORT="${PORT:-8115}"
RESULT_ROOT="${RESULT_ROOT:-/work/bench_results_dense_mlp_interleaved_fusion}"
CONCURRENCY_LIST="${CONCURRENCY_LIST:-1 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30}"
INPUT_LEN="${INPUT_LEN:-3500}"
OUTPUT_LEN="${OUTPUT_LEN:-1500}"
WHICH="${1:-both}"

mkdir -p "$RESULT_ROOT"

wait_for_server() {
  local timeout="${1:-600}"
  local waited=0
  echo "Waiting for vLLM server on port $PORT to become ready (timeout ${timeout}s)..."
  while ! curl -s --connect-timeout 2 "http://127.0.0.1:$PORT/v1/models" \
      | grep -q '"id"'; do
    sleep 5
    waited=$((waited + 5))
    if [ "$waited" -ge "$timeout" ]; then
      echo "ERROR: server did not become ready within ${timeout}s" >&2
      return 1
    fi
  done
  echo "Server is ready after ${waited}s."
}

stop_server() {
  local pid_file="$1"
  if [ -f "$pid_file" ]; then
    local pid
    pid="$(cat "$pid_file")"
    if [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1; then
      echo "Stopping vLLM server (PID $pid)..."
      kill "$pid" 2>/dev/null
      for _ in $(seq 1 30); do
        kill -0 "$pid" >/dev/null 2>&1 || break
        sleep 2
      done
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
    rm -f "$pid_file"
  fi
  # Give the driver/XPU a moment to release device memory before the next run.
  sleep 10
}

run_one() {
  local tag="$1"       # baseline | fusion
  local fuse_flag="$2" # 0 | 1
  local log_dir="$RESULT_ROOT/$tag"
  local pid_file="/tmp/vllm_serve_${tag}.pid"
  local serve_log="/tmp/vllm_serve_${tag}.log"

  echo "=============================================="
  echo "Running config: $tag (VLLM_XPU_FUSED_DENSE_MLP_INTERLEAVED=$fuse_flag)"
  echo "=============================================="

  source /opt/intel/oneapi/setvars.sh --force > /tmp/setvars_bench.log 2>&1

  if [ "$fuse_flag" = "1" ]; then
    export VLLM_XPU_FUSED_DENSE_MLP_INTERLEAVED=1
  else
    unset VLLM_XPU_FUSED_DENSE_MLP_INTERLEAVED
  fi

  # Run with XPU graph capture enabled (default: USE_EAGER=0,
  # VLLM_XPU_ENABLE_XPU_GRAPH=1 per start_server.sh). The graph-capture
  # crashes previously seen at TP=4 have been fixed (structured-outputs
  # bitmask copy stream, top-k/top-p table warmup) and the remaining
  # oneCCL/torch-xpu graph-capture issue has been resolved by the torch
  # rebuild, per user verification with test_prefill_only.sh.
  PORT="$PORT" TP_SIZE="$TP_SIZE" VLLM_LOG="$serve_log" VLLM_PID_FILE="$pid_file" \
    ZE_AFFINITY_MASK="0,1,2,3" \
    "$SCRIPT_DIR/start_server.sh" "$MODEL"

  if ! wait_for_server 900; then
    echo "ERROR: $tag server failed to start, see $serve_log" >&2
    tail -100 "$serve_log" >&2
    stop_server "$pid_file"
    return 1
  fi

  LOG_DIR="$log_dir" CONCURRENCY_LIST="$CONCURRENCY_LIST" PORT="$PORT" \
    INPUT_LEN="$INPUT_LEN" OUTPUT_LEN="$OUTPUT_LEN" \
    "$SCRIPT_DIR/bench.sh"

  stop_server "$pid_file"
}

case "$WHICH" in
  baseline)
    run_one baseline 0
    ;;
  fusion)
    run_one fusion 1
    ;;
  both)
    run_one baseline 0
    run_one fusion 1
    ;;
  *)
    echo "Usage: $0 [baseline|fusion|both]" >&2
    exit 1
    ;;
esac

echo "All requested benchmark runs complete. Results in $RESULT_ROOT/{baseline,fusion}"
